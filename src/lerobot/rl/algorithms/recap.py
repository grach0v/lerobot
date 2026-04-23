#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RECAP algorithm wrapper.

Wraps an advantage-conditioned policy (e.g. PiStar06) and a distributional
value policy together under the `RLAlgorithm` interface. Each update step
optionally runs a value-network gradient step, a policy gradient step, or both
— stage-gated via `policy_update_freq` and `value_update_freq` so the same
entrypoint can be used for value-only training (`policy_update_freq=0`),
policy-only fine-tuning (`value_update_freq=0`), or joint updates.

The algorithm expects both models to follow the LeRobot `PreTrainedPolicy`
contract: `forward(batch) -> (loss, dict)`. Advantage labels and value targets
are consumed straight from the dataset batch — the algorithm never materializes
them itself — so the training loop remains stateless with respect to labels.
"""

from __future__ import annotations

import copy
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler

from lerobot.optim.optimizers import OptimizerConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.algorithms.base import RLAlgorithm, RLAlgorithmConfig, TrainingStats
from lerobot.rl.algorithms.recap_targets import (
    EPISODE_SUCCESS_FEATURE,
    RECAP_ADVANTAGE_FEATURE,
    RECAP_VALUE_TARGET_FEATURE,
    compute_recap_advantages,
    threshold_recap_advantages,
)

RECAP_PROMPT_APPLIED_FEATURE = "recap_prompt_applied"

VALUE_WANDB_KEYS = (
    "value_mae",
    "value_rmse",
    "target_bin0_frac",
)
POLICY_WANDB_KEYS = (
    "prompted_frac",
    "loss_unconditional",
    "loss_prompted",
    "loss_prompted_success_episode",
    "loss_prompted_failure_episode",
)


def _clip_grad_norm(parameters, max_norm: float, accelerator=None) -> float:
    clip_max_norm = float("inf") if max_norm <= 0 else max_norm
    if accelerator is not None:
        return float(accelerator.clip_grad_norm_(parameters, clip_max_norm).item())
    if max_norm <= 0:
        return float(
            torch.nn.utils.clip_grad_norm_(parameters, float("inf"), error_if_nonfinite=False).item()
        )
    return float(
        torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=False).item()
    )


def _as_bool_mask(
    batch: dict[str, Tensor],
    key: str,
    *,
    default: bool | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor | None:
    value = batch.get(key)
    if value is None:
        if default is None:
            return None
        reference = batch.get(RECAP_ADVANTAGE_FEATURE)
        if reference is None:
            return None
        if not torch.is_tensor(reference):
            reference = torch.as_tensor(reference, device=device)
        elif device is not None:
            reference = reference.to(device=device)
        return torch.full_like(
            reference.reshape(-1),
            fill_value=default,
            dtype=torch.bool,
            device=reference.device,
        )

    if not torch.is_tensor(value):
        value = torch.as_tensor(value, device=device)
    elif device is not None:
        value = value.to(device=device)
    if value.ndim > 1:
        value = value.squeeze(-1)
    if value.dtype == torch.bool:
        return value.reshape(-1)
    return (value > 0.5).reshape(-1)


def _masked_mean(values: Tensor, mask: torch.Tensor) -> float | None:
    if values.ndim > 1:
        values = values.reshape(values.shape[0], -1).mean(dim=1)
    values = values.reshape(-1)
    mask = mask.reshape(-1)
    if values.shape[0] != mask.shape[0] or not torch.any(mask):
        return None
    return float(values[mask].mean().item())


def _aggregate_numeric_metrics(
    output_dicts: list[dict[str, Any]],
    *,
    keys: tuple[str, ...],
    prefix: str,
    reduce_metric,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key in keys:
        values = [
            float(output_dict[key])
            for output_dict in output_dicts
            if isinstance(output_dict.get(key), (int, float))
        ]
        if values:
            metrics[f"{prefix}/{key}"] = reduce_metric(sum(values) / len(values))
    return metrics


def _policy_loss_breakdown(batch: dict[str, Tensor], per_sample_loss: Tensor) -> dict[str, float]:
    metrics: dict[str, float] = {}
    device = per_sample_loss.device
    prompt_applied = _as_bool_mask(batch, RECAP_PROMPT_APPLIED_FEATURE, default=True, device=device)
    advantage_positive = _as_bool_mask(batch, RECAP_ADVANTAGE_FEATURE, device=device)
    episode_success = _as_bool_mask(batch, EPISODE_SUCCESS_FEATURE, device=device)

    if prompt_applied is None or advantage_positive is None:
        return metrics

    prompt_free = ~prompt_applied

    metrics["prompted_frac"] = float(prompt_applied.float().mean().item())

    split_masks = {
        "loss_unconditional": prompt_free,
        "loss_prompted": prompt_applied,
    }
    if episode_success is not None:
        prompted_success = prompt_applied & episode_success
        prompted_failure = prompt_applied & ~episode_success
        split_masks["loss_prompted_success_episode"] = prompted_success
        split_masks["loss_prompted_failure_episode"] = prompted_failure

    for key, mask in split_masks.items():
        value = _masked_mean(per_sample_loss, mask)
        if value is not None:
            metrics[key] = value

    return metrics


@RLAlgorithmConfig.register_subclass("recap")
@dataclass
class RECAPAlgorithmConfig(RLAlgorithmConfig):
    policy_optimizer: OptimizerConfig | None = None
    value_optimizer: OptimizerConfig | None = None
    policy_utd_ratio: int = 1
    value_utd_ratio: int = 1
    policy_update_freq: int = 1
    value_update_freq: int = 1
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    num_training_steps: int | None = None

    def __post_init__(self) -> None:
        if self.policy_utd_ratio < 1:
            raise ValueError("policy_utd_ratio must be >= 1.")
        if self.value_utd_ratio < 1:
            raise ValueError("value_utd_ratio must be >= 1.")
        if self.policy_update_freq < 0:
            raise ValueError("policy_update_freq must be >= 0.")
        if self.value_update_freq < 0:
            raise ValueError("value_update_freq must be >= 0.")
        if self.policy_update_freq == 0 and self.value_update_freq == 0:
            raise ValueError("At least one of policy_update_freq or value_update_freq must be > 0.")

    @classmethod
    def from_policy_config(cls, policy_cfg, **kwargs):
        value_policy_cfg = kwargs.get("value_policy_cfg")
        policy_optimizer = copy.deepcopy(policy_cfg.get_optimizer_preset())
        if value_policy_cfg is None:
            value_optimizer = copy.deepcopy(policy_cfg.get_optimizer_preset())
        else:
            value_optimizer = copy.deepcopy(value_policy_cfg.get_optimizer_preset())
        return cls(
            policy_optimizer=policy_optimizer,
            value_optimizer=value_optimizer,
        )

    def build_algorithm(self, policy, **kwargs):
        value_policy = kwargs.get("value_policy")
        if value_policy is None:
            raise ValueError("RECAPAlgorithm requires a value_policy.")
        return RECAPAlgorithm(policy=policy, value_policy=value_policy, config=self)


class RECAPAlgorithm(RLAlgorithm):
    def __init__(
        self,
        *,
        policy: PreTrainedPolicy,
        value_policy: PreTrainedPolicy,
        config: RECAPAlgorithmConfig,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.value_policy = value_policy
        self.config = config
        self.policy_optimizer_config = copy.deepcopy(
            config.policy_optimizer or self.policy.config.get_optimizer_preset()
        )
        self.value_optimizer_config = copy.deepcopy(
            config.value_optimizer or self.value_policy.config.get_optimizer_preset()
        )
        self.schedulers: dict[str, LRScheduler] = {}
        self.accelerator = None

    def _unwrap_model(self, model: PreTrainedPolicy) -> PreTrainedPolicy:
        if self.accelerator is not None:
            return self.accelerator.unwrap_model(model)
        return getattr(model, "module", model)

    def _policy_model(self) -> PreTrainedPolicy:
        return self._unwrap_model(self.policy)

    def _value_policy_model(self) -> PreTrainedPolicy:
        return self._unwrap_model(self.value_policy)

    def _reduce_metric(self, value: float) -> float:
        metric = float(value)
        if self.accelerator is None or self.accelerator.num_processes == 1:
            return metric
        reduced = self.accelerator.reduce(
            torch.tensor(metric, device=self.accelerator.device),
            reduction="mean",
        )
        return float(reduced.item())

    def _should_update(self, update_freq: int) -> bool:
        return update_freq > 0 and self.optimization_step % update_freq == 0

    def select_action(self, observation: dict[str, Tensor]) -> Tensor:
        return self._policy_model().select_action(observation)

    def set_accelerator(self, accelerator) -> None:
        self.accelerator = accelerator

    def make_optimizers(self) -> dict[str, torch.optim.Optimizer]:
        policy_model = self._policy_model()
        value_policy_model = self._value_policy_model()
        self.optimizers = {}
        if self.config.policy_update_freq > 0:
            self.optimizers["policy"] = self.policy_optimizer_config.build(
                policy_model.get_optim_params()
            )
        if self.config.value_update_freq > 0:
            self.optimizers["value"] = self.value_optimizer_config.build(
                value_policy_model.get_optim_params()
            )
        self.schedulers = {}
        if self.config.num_training_steps is not None and self.config.num_training_steps > 0:
            policy_scheduler_cfg = policy_model.config.get_scheduler_preset()
            if "policy" in self.optimizers and policy_scheduler_cfg is not None:
                self.schedulers["policy"] = policy_scheduler_cfg.build(
                    self.optimizers["policy"], self.config.num_training_steps
                )
            value_scheduler_cfg = value_policy_model.config.get_scheduler_preset()
            if "value" in self.optimizers and value_scheduler_cfg is not None:
                self.schedulers["value"] = value_scheduler_cfg.build(
                    self.optimizers["value"], self.config.num_training_steps
                )
        return self.optimizers

    def get_weights(self) -> dict[str, Tensor]:
        policy_model = self._policy_model()
        value_policy_model = self._value_policy_model()
        weights: dict[str, Tensor] = {}
        for prefix, state_dict in (
            ("policy", policy_model.state_dict()),
            ("value_policy", value_policy_model.state_dict()),
        ):
            for key, value in state_dict.items():
                weights[f"{prefix}.{key}"] = value.detach().cpu()
        return weights

    def load_weights(self, weights: dict[str, Tensor], device: str = "cpu") -> None:
        policy_weights = {
            key.removeprefix("policy."): value.to(device)
            for key, value in weights.items()
            if key.startswith("policy.")
        }
        value_weights = {
            key.removeprefix("value_policy."): value.to(device)
            for key, value in weights.items()
            if key.startswith("value_policy.")
        }
        policy_model = self._policy_model()
        value_policy_model = self._value_policy_model()
        if policy_weights:
            policy_model.load_state_dict(policy_weights, strict=False)
        if value_weights:
            value_policy_model.load_state_dict(value_weights, strict=False)
        policy_model.to(device)
        value_policy_model.to(device)

    @torch.no_grad()
    def predict_values(self, batch: dict[str, Tensor]) -> Tensor:
        return self._value_policy_model().predict_values(batch)

    def relabel_advantages(
        self,
        *,
        normalized_returns: np.ndarray,
        predicted_values: np.ndarray,
        episode_ranges: list[tuple[int, int, int]],
        n_step: int = 0,
        positive_percentile: float = 0.65,
        intervention_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        advantages = compute_recap_advantages(
            normalized_returns=normalized_returns,
            predicted_values=predicted_values,
            episode_ranges=episode_ranges,
            n_step=n_step,
        )
        indicator, threshold = threshold_recap_advantages(
            advantages=advantages,
            positive_percentile=positive_percentile,
            intervention_mask=intervention_mask,
        )
        return advantages, indicator, threshold

    def _update_value(self, batch: dict[str, Tensor]) -> tuple[float, float, dict[str, Any]]:
        optimizer = self.optimizers["value"]
        self.value_policy.train()
        optimizer.zero_grad()
        autocast_context = self.accelerator.autocast() if self.accelerator is not None else nullcontext()
        with autocast_context:
            loss, output_dict = self.value_policy.forward(batch)
        loss = loss * self.config.value_loss_weight
        if self.accelerator is not None:
            self.accelerator.backward(loss)
        else:
            loss.backward()
        grad_norm = _clip_grad_norm(
            self.value_policy.parameters(),
            getattr(self.value_optimizer_config, "grad_clip_norm", 0.0),
            accelerator=self.accelerator,
        )
        optimizer.step()
        if "value" in self.schedulers:
            self.schedulers["value"].step()
        return float(loss.item()), grad_norm, output_dict

    def _update_policy(self, batch: dict[str, Tensor]) -> tuple[float, float, dict[str, Any]]:
        optimizer = self.optimizers["policy"]
        self.policy.train()
        optimizer.zero_grad()
        autocast_context = self.accelerator.autocast() if self.accelerator is not None else nullcontext()
        with autocast_context:
            per_sample_loss, output_dict = self.policy.forward(batch, reduction="none")
        output_dict = {
            key: value
            for key, value in output_dict.items()
            if key != "loss" and isinstance(value, (int, float))
        }
        output_dict.update(_policy_loss_breakdown(batch, per_sample_loss))
        loss = per_sample_loss.mean() * self.config.policy_loss_weight
        if self.accelerator is not None:
            self.accelerator.backward(loss)
        else:
            loss.backward()
        grad_norm = _clip_grad_norm(
            self.policy.parameters(),
            getattr(self.policy_optimizer_config, "grad_clip_norm", 0.0),
            accelerator=self.accelerator,
        )
        optimizer.step()
        if "policy" in self.schedulers:
            self.schedulers["policy"].step()
        return float(loss.item()), grad_norm, output_dict

    def update(self, batch_iterator) -> TrainingStats:
        if not self.optimizers:
            self.make_optimizers()

        stats = TrainingStats()
        extra: dict[str, float] = {}

        if self._should_update(self.config.value_update_freq):
            value_losses = []
            value_grad_norms = []
            value_output_dicts = []
            for _ in range(max(1, self.config.value_utd_ratio)):
                batch = next(batch_iterator)
                if RECAP_VALUE_TARGET_FEATURE not in batch:
                    raise ValueError(
                        f"RECAP value update requires batches with '{RECAP_VALUE_TARGET_FEATURE}'. "
                        "Run the dataset builder so the column is materialised first."
                    )
                loss, grad_norm, output_dict = self._update_value(batch)
                value_losses.append(loss)
                value_grad_norms.append(grad_norm)
                value_output_dicts.append(output_dict)

            stats.losses["value"] = self._reduce_metric(sum(value_losses) / len(value_losses))
            stats.grad_norms["value"] = self._reduce_metric(
                sum(value_grad_norms) / len(value_grad_norms)
            )
            if value_output_dicts:
                extra.update(
                    _aggregate_numeric_metrics(
                        value_output_dicts,
                        keys=VALUE_WANDB_KEYS,
                        prefix="value",
                        reduce_metric=self._reduce_metric,
                    )
                )
            extra["lr/value"] = self._reduce_metric(self.optimizers["value"].param_groups[0]["lr"])

        if self._should_update(self.config.policy_update_freq):
            policy_losses = []
            policy_grad_norms = []
            policy_output_dicts = []
            for _ in range(max(1, self.config.policy_utd_ratio)):
                batch = next(batch_iterator)
                if RECAP_ADVANTAGE_FEATURE not in batch:
                    raise ValueError(
                        f"RECAP policy update requires batches with '{RECAP_ADVANTAGE_FEATURE}'. "
                        "Run build_recap_policy_dataset before training the policy."
                    )
                loss, grad_norm, output_dict = self._update_policy(batch)
                policy_losses.append(loss)
                policy_grad_norms.append(grad_norm)
                policy_output_dicts.append(output_dict)

            stats.losses["policy"] = self._reduce_metric(sum(policy_losses) / len(policy_losses))
            stats.grad_norms["policy"] = self._reduce_metric(
                sum(policy_grad_norms) / len(policy_grad_norms)
            )
            if policy_output_dicts:
                extra.update(
                    _aggregate_numeric_metrics(
                        policy_output_dicts,
                        keys=POLICY_WANDB_KEYS,
                        prefix="policy",
                        reduce_metric=self._reduce_metric,
                    )
                )
            extra["lr/policy"] = self._reduce_metric(self.optimizers["policy"].param_groups[0]["lr"])

        self.optimization_step += 1
        stats.extra.update(extra)
        return stats
