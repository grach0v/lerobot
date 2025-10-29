from dataclasses import dataclass, field
from typing import Any

from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("remote")
@dataclass
class RemoteConfig(PreTrainedConfig):
    # Identity and device placement
    type: str = field(default="remote", metadata={"help": "Policy type name"})
    device: str = field(default="cpu", metadata={"help": "Device used for returned tensors"})

    # Action execution
    # How many environment steps to execute per policy call. Used by the runtime action queue.
    n_action_steps: int = field(default=1, metadata={"help": "Number of env steps to execute per call"})

    # Remote-specific (gRPC policy server)
    server_address: str = field(
        default="localhost:8080", metadata={"help": "Async inference policy server address (host:port)"}
    )
    request_timeout: float = field(default=30.0, metadata={"help": "gRPC request timeout in seconds"})
    retries: int = field(default=3, metadata={"help": "Number of retry attempts for failed RPC calls"})

    remote_policy_type: str = field(
        default="",
        metadata={"help": "Policy type for the async inference server to load (e.g. act, diffusion)"},
    )
    remote_pretrained_name_or_path: str = field(
        default="",
        metadata={
            "help": (
                "Pretrained model repo ID or path for the async inference server. "
                "Should match a directory containing policy weights or a Hugging Face repo ID."
            )
        },
    )
    remote_policy_device: str = field(
        default="cpu", metadata={"help": "Device on which the async inference server loads the policy"}
    )

    actions_per_chunk: int | None = field(
        default=None,
        metadata={
            "help": (
                "Number of actions returned per chunk by the remote server. "
                "Defaults to `n_action_steps` when not provided."
            )
        },
    )
    rename_map: dict[str, str] = field(
        default_factory=dict,
        metadata={
            "help": (
                "Observation rename map forwarded to the async inference server so it can match "
                "environment keys to the policy's expected features."
            )
        },
    )

    # Additional arguments to inject directly into the observation dict (e.g. {"inference_config": {...}})
    additional_args: dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Extra observation keys to inject directly into observation"},
    )

    # --- Abstract API implementations required by PreTrainedConfig ---
    def get_optimizer_preset(self) -> AdamWConfig:
        """Remote policy is inference-only; return an inert preset for API compatibility."""
        return AdamWConfig(lr=1e-5, weight_decay=0.0, grad_clip_norm=1.0)

    def get_scheduler_preset(self):
        # No scheduler needed for inference-only policy
        return None

    def validate_features(self) -> None:
        if not self.remote_pretrained_name_or_path:
            raise ValueError(
                "RemoteConfig expects `remote_pretrained_name_or_path` to be provided so the server can load the policy."
            )

        remote_cfg: PreTrainedConfig | None = None
        if not self.remote_policy_type or not self.input_features or not self.output_features:
            remote_cfg = PreTrainedConfig.from_pretrained(self.remote_pretrained_name_or_path)

        if not self.remote_policy_type:
            self.remote_policy_type = remote_cfg.type if remote_cfg is not None else ""

        if remote_cfg is not None and remote_cfg.type != self.remote_policy_type:
            raise ValueError(
                f"Loaded remote policy config type '{remote_cfg.type}' does not match "
                f"requested remote_policy_type '{self.remote_policy_type}'."
            )

        if not self.input_features and remote_cfg is not None:
            self.input_features = remote_cfg.input_features

        if not self.output_features and remote_cfg is not None:
            self.output_features = remote_cfg.output_features

        if not self.input_features:
            raise ValueError("RemoteConfig requires `input_features` to be defined.")
        if not self.remote_policy_type:
            raise ValueError("RemoteConfig expects `remote_policy_type` to be set for async inference.")
        if self.effective_actions_per_chunk <= 0:
            raise ValueError("RemoteConfig requires `actions_per_chunk` or `n_action_steps` to be positive.")
        if self.retries < 1:
            raise ValueError("RemoteConfig expects `retries` to be at least 1.")

    @property
    def effective_actions_per_chunk(self) -> int:
        return self.actions_per_chunk or self.n_action_steps

    @property
    def observation_delta_indices(self):
        # No temporal deltas required for observations by default
        return None

    @property
    def action_delta_indices(self):
        # Minimal behavior: align deltas to n_action_steps
        return list(range(self.n_action_steps))

    @property
    def reward_delta_indices(self):
        return None
