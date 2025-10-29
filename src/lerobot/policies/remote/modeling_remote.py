import logging
import pickle  # nosec B403 - trusted channel between client/server
import threading
import time
from collections import deque
from typing import Any

import grpc
import torch
from torch import Tensor

from lerobot.async_inference.helpers import RemotePolicyConfig, TimedAction, TimedObservation
from lerobot.configs.types import FeatureType
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.utils.constants import OBS_STR

from .configuration_remote import RemoteConfig

logger = logging.getLogger(__name__)


class RemotePolicy(PreTrainedPolicy):
    """
    A policy that proxies inference to the async inference gRPC policy server.
    """

    config_class = RemoteConfig
    name = "remote"

    def __init__(self, config: RemoteConfig):
        super().__init__(config)
        config.validate_features()
        self._vector_name_map: dict[str, list[str]] = {}
        self._image_key_map: dict[str, str] = {}
        self._lerobot_features = self._build_lerobot_features()
        self._thread_state = threading.local()
        self.reset()

    def get_optim_params(self) -> dict:
        return {}

    def reset(self):
        # Reinitialize thread-local state so each worker gets its own queue/session
        self._thread_state = threading.local()

    def _state(self):
        state = self._thread_state
        if not hasattr(state, "action_queue"):
            state.action_queue = deque(maxlen=self.config.n_action_steps)
        if not hasattr(state, "stub") or state.stub is None:
            self._initialize_connection(state)
        return state

    def _initialize_connection(self, state) -> None:
        state.channel = grpc.insecure_channel(
            self.config.server_address,
            options=grpc_channel_options(),
        )
        state.stub = services_pb2_grpc.AsyncInferenceStub(state.channel)
        state.next_timestep = 0

        policy_cfg = RemotePolicyConfig(
            policy_type=self.config.remote_policy_type,
            pretrained_name_or_path=self.config.remote_pretrained_name_or_path,
            lerobot_features=self._lerobot_features,
            actions_per_chunk=self.config.effective_actions_per_chunk,
            device=self.config.remote_policy_device,
            rename_map=self.config.rename_map,
        )

        payload = pickle.dumps(policy_cfg)  # nosec B301 - config originates from local process
        request = services_pb2.PolicySetup(data=payload)

        for attempt in range(1, self.config.retries + 1):
            try:
                state.stub.Ready(services_pb2.Empty(), timeout=self.config.request_timeout)
                state.stub.SendPolicyInstructions(request, timeout=self.config.request_timeout)
                logger.debug("Remote policy handshake completed on attempt %d", attempt)
                return
            except grpc.RpcError as err:
                logger.warning("Remote policy handshake failed on attempt %d: %s", attempt, err)
                self._close_channel(state)
                if attempt == self.config.retries:
                    raise
                time.sleep(0.1)
                state.channel = grpc.insecure_channel(
                    self.config.server_address,
                    options=grpc_channel_options(),
                )
                state.stub = services_pb2_grpc.AsyncInferenceStub(state.channel)

    def _close_channel(self, state) -> None:
        if getattr(state, "channel", None) is not None:
            state.channel.close()
        state.stub = None

    def _build_lerobot_features(self) -> dict[str, dict[str, Any]]:
        """
        Build a hw-style feature dictionary expected by the async inference server.
        Vector features (state/env) are split into individual scalar names, while image features
        are mapped to (H, W, C) tensors keyed by their camera name.
        """
        features: dict[str, dict[str, Any]] = {}
        vector_name_map: dict[str, list[str]] = {}
        image_key_map: dict[str, str] = {}

        for key, feature in self.config.input_features.items():
            if feature.type in (FeatureType.STATE, FeatureType.ENV):
                if not feature.shape or len(feature.shape) != 1:
                    raise ValueError(
                        f"RemotePolicy only supports 1D state features, got shape {feature.shape} for '{key}'."
                    )
                dim = feature.shape[0]
                names = [f"{key.replace('.', '_')}_d{idx}" for idx in range(dim)]
                features[key] = {
                    "dtype": "float32",
                    "shape": (dim,),
                    "names": names,
                }
                vector_name_map[key] = names
            elif feature.type is FeatureType.VISUAL:
                if not feature.shape or len(feature.shape) != 3:
                    raise ValueError(
                        f"RemotePolicy only supports 3D visual features, got shape {feature.shape} for '{key}'."
                    )
                channels, height, width = feature.shape
                camera_base = key.removeprefix(f"{OBS_STR}.images.")
                # Ensure uniqueness if multiple features share the same suffix
                raw_key = camera_base
                counter = 1
                while raw_key in image_key_map.values():
                    raw_key = f"{camera_base}_{counter}"
                    counter += 1

                features[key] = {
                    "dtype": "video",
                    "shape": (height, width, channels),
                    "names": ["height", "width", "channels"],
                }
                image_key_map[key] = raw_key
            else:
                logger.debug("Skipping unsupported feature '%s' of type '%s'", key, feature.type)

        self._vector_name_map = vector_name_map
        self._image_key_map = image_key_map
        return features

    def _prepare_payload(self, batch: dict[str, Tensor]) -> dict[str, Any]:
        if not batch:
            raise ValueError("RemotePolicy received an empty batch.")

        payload: dict[str, Any] = {}
        cpu_batch: dict[str, Any] = {
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

        # Serialize vector features (state/env) into individual scalar entries
        for key, names in self._vector_name_map.items():
            tensor = cpu_batch.get(key)
            if tensor is None:
                continue

            if isinstance(tensor, torch.Tensor):
                if tensor.ndim == 2:
                    tensor = tensor.squeeze(0)
                tensor = tensor.flatten()
                if tensor.numel() != len(names):
                    raise ValueError(
                        f"Feature '{key}' expected {len(names)} values, got shape {tuple(tensor.shape)}."
                    )
                for idx, name in enumerate(names):
                    payload[name] = float(tensor[idx].item())
            else:
                raise TypeError(f"Expected tensor for feature '{key}', got {type(tensor)}")

        # Serialize image features (convert to HWC uint8 tensors)
        for key, raw_key in self._image_key_map.items():
            tensor = cpu_batch.get(key)
            if tensor is None:
                continue

            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Expected tensor for image feature '{key}', got {type(tensor)}")

            if tensor.ndim == 4:
                tensor = tensor.squeeze(0)
            if tensor.ndim != 3:
                raise ValueError(
                    f"Image feature '{key}' must have 3 dimensions after squeeze, got {tensor.ndim}"
                )

            if tensor.dtype != torch.uint8:
                tensor = (tensor.clamp(0.0, 1.0) * 255.0).to(torch.uint8)

            payload[raw_key] = tensor.permute(1, 2, 0).contiguous()

        # Optional task/instruction keys
        for extra_key in ["task", "instruction"]:
            if extra_key in cpu_batch:
                payload[extra_key] = cpu_batch[extra_key]

        for key, value in (self.config.additional_args or {}).items():
            payload[key] = value

        return payload

    def _timed_actions_to_tensor(self, timed_actions: list[TimedAction]) -> Tensor:
        if not timed_actions:
            raise RuntimeError("Remote policy server returned an empty action chunk.")

        actions = []
        for timed_action in timed_actions:
            action = timed_action.get_action()
            if isinstance(action, torch.Tensor):
                actions.append(action.detach().cpu())
            else:
                actions.append(torch.as_tensor(action, dtype=torch.float32))

        stacked = torch.stack(actions, dim=0).unsqueeze(0)  # (B=1, T, A)
        return stacked.to(device=self.config.device, dtype=torch.float32)

    def _request_action_chunk(self, state, batch: dict[str, Tensor]) -> Tensor:
        payload = self._prepare_payload(batch)
        timestamp = time.time()
        timestep = state.next_timestep
        state.next_timestep += 1

        observation = TimedObservation(
            timestamp=timestamp,
            timestep=timestep,
            observation=payload,
            must_go=True,
        )
        packed = pickle.dumps(observation)  # nosec B301 - observation built locally

        iterator = send_bytes_in_chunks(
            packed,
            services_pb2.Observation,
            log_prefix="[RemotePolicy]",
            silent=True,
        )
        state.stub.SendObservations(iterator, timeout=self.config.request_timeout)
        actions_msg = state.stub.GetActions(services_pb2.Empty(), timeout=self.config.request_timeout)
        if not actions_msg.data:
            raise RuntimeError("Remote policy server returned an empty response payload.")

        timed_actions = pickle.loads(actions_msg.data)  # nosec B301 - server is trusted peer
        return self._timed_actions_to_tensor(timed_actions)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict] | tuple[Tensor, None]:
        raise NotImplementedError("RemotePolicy is inference-only")

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        last_error: Exception | None = None

        for attempt in range(1, self.config.retries + 1):
            state = self._state()
            try:
                return self._request_action_chunk(state, batch)
            except grpc.RpcError as err:
                logger.warning("Remote policy RPC failed on attempt %d: %s", attempt, err)
                last_error = err
                self._close_channel(state)
                time.sleep(0.1)
            except Exception as err:
                logger.error("Unexpected error when requesting remote action chunk: %s", err)
                last_error = err
                break

        assert last_error is not None
        raise last_error

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()

        queue = self._state().action_queue

        if len(queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            queue.extend(actions.transpose(0, 1))  # [(B, A)] x T

        return queue.popleft()
