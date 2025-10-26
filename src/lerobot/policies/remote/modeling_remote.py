import numpy as np
import requests
import torch
from collections import deque
from torch import Tensor

from lerobot.utils.messaging import pack_msg, unpack_msg
from lerobot.policies.pretrained import PreTrainedPolicy
from .configuration_remote import RemoteConfig


class RemotePolicy(PreTrainedPolicy):
    """
    A policy that proxies inference to a remote HTTP server.
    """

    config_class = RemoteConfig
    name = "remote"

    def __init__(self, config: RemoteConfig):
        super().__init__(config)
        self.server_url = config.server_url.rstrip("/")
        self.session = requests.Session()
        self.timeout = config.timeout
        self.reset()

    def get_optim_params(self) -> dict:
        return {}

    def reset(self):
        # Queue emits one action per env step; refilled when empty
        self._action_queue = deque(maxlen=self.config.n_action_steps)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict] | tuple[Tensor, None]:
        raise NotImplementedError("RemotePolicy is inference-only")

    def custom_prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch.pop('action')
        batch.pop('next.reward')
        batch.pop('next.done')
        batch.pop('next.truncated')
        batch.pop('info')

        task = batch.pop('task')
        batch['observation.task_instr'] = task

        if not hasattr(self, 'previous_state'):
            self.previous_state = batch["observation.state"].clone()
        
        batch["observation.state"] = torch.stack([self.previous_state, batch["observation.state"]], dim=1)
        self.previous_state = batch["observation.state"][:, -1].clone()

        return batch

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        # Build payload with raw tensors/arrays; pack_msg handles encoding

        batch = self.custom_prepare_batch(batch)
        add_args = self.config.additional_args or {}
        payload = batch | add_args

        packed = pack_msg(payload)

        last_exception = None
        for _ in range(self.config.attempts):
            try:
                resp = self.session.post(
                    f"{self.server_url}/predict",
                    data=packed,
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                break
            except requests.RequestException as e:
                last_exception = e

        if last_exception:
            raise last_exception

        unpacked = unpack_msg(resp.content)
        actions_np = np.asarray(unpacked)

        device = torch.device(self.config.device)
        any_tensor = next((v for v in batch.values() if isinstance(v, torch.Tensor)), None)
        dtype = any_tensor.dtype if isinstance(any_tensor, torch.Tensor) else torch.float32

        actions = torch.from_numpy(actions_np).to(device=device, dtype=dtype)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))  # [(B, A)] x T

        return self._action_queue.popleft()