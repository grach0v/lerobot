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

"""Unit tests for the shared RECAPAlgorithm + RLTrainer wiring.

Uses toy policies (tiny nn.Linear-wrapped modules) to avoid needing any
pretrained checkpoint. These tests exercise stage-gating (policy-only and
value-only phases), dataset-column enforcement, and gradient flow.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from lerobot.optim.optimizers import AdamConfig
from lerobot.rl.algorithms.recap import RECAPAlgorithm, RECAPAlgorithmConfig
from lerobot.rl.algorithms.recap_targets import (
    RECAP_ADVANTAGE_FEATURE,
    RECAP_VALUE_TARGET_FEATURE,
)


class _DummyConfig:
    """Minimal stand-in for PreTrainedConfig so the algorithm can build optimizers."""

    def __init__(self, lr: float = 1e-2):
        self._lr = lr

    def get_optimizer_preset(self):
        return AdamConfig(lr=self._lr, grad_clip_norm=1.0)

    def get_scheduler_preset(self):
        return None


class _PolicyDouble(nn.Module):
    """Tiny policy that returns a per-sample MSE loss against a target in the batch."""

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(4, 4)
        self.config = _DummyConfig()

    def get_optim_params(self):
        return self.parameters()

    def forward(self, batch, reduction: str = "mean"):
        inputs = batch["obs"]
        targets = batch["target"]
        prediction = self.head(inputs)
        per_sample = ((prediction - targets) ** 2).mean(dim=-1)
        if reduction == "none":
            return per_sample, {}
        return per_sample.mean(), {}


class _ValueDouble(nn.Module):
    """Tiny value model: cross-entropy over a discrete target bin."""

    def __init__(self, num_bins: int = 4):
        super().__init__()
        self.head = nn.Linear(4, num_bins)
        self.config = _DummyConfig()

    def get_optim_params(self):
        return self.parameters()

    def forward(self, batch):
        logits = self.head(batch["obs"])
        loss = nn.functional.cross_entropy(logits, batch[RECAP_VALUE_TARGET_FEATURE].reshape(-1))
        return loss, {"value_mae": float(loss.detach().item())}


def _make_batch(batch_size: int = 3, include_advantage: bool = True) -> dict:
    obs = torch.randn(batch_size, 4)
    return {
        "obs": obs,
        "target": torch.randn(batch_size, 4),
        RECAP_VALUE_TARGET_FEATURE: torch.randint(0, 4, (batch_size,), dtype=torch.long),
        RECAP_ADVANTAGE_FEATURE: (
            torch.randint(0, 2, (batch_size,), dtype=torch.float32) if include_advantage else None
        ),
    }


def _batch_iter(n: int, **batch_kwargs):
    for _ in range(n):
        yield _make_batch(**batch_kwargs)


def test_requires_at_least_one_update_path():
    with pytest.raises(ValueError, match="update_freq"):
        RECAPAlgorithmConfig(policy_update_freq=0, value_update_freq=0)


def test_joint_update_runs_both_paths():
    algo = RECAPAlgorithm(
        policy=_PolicyDouble(),
        value_policy=_ValueDouble(),
        config=RECAPAlgorithmConfig(),
    )
    stats = algo.update(iter([_make_batch(), _make_batch()]))
    assert "policy" in stats.losses
    assert "value" in stats.losses
    assert stats.losses["policy"] >= 0.0
    assert stats.losses["value"] >= 0.0


def test_value_only_update_ignores_policy():
    algo = RECAPAlgorithm(
        policy=_PolicyDouble(),
        value_policy=_ValueDouble(),
        config=RECAPAlgorithmConfig(policy_update_freq=0),
    )
    stats = algo.update(iter([_make_batch()]))
    assert "value" in stats.losses
    assert "policy" not in stats.losses


def test_policy_only_update_ignores_value():
    algo = RECAPAlgorithm(
        policy=_PolicyDouble(),
        value_policy=_ValueDouble(),
        config=RECAPAlgorithmConfig(value_update_freq=0),
    )
    stats = algo.update(iter([_make_batch()]))
    assert "policy" in stats.losses
    assert "value" not in stats.losses


def test_policy_update_errors_without_advantage_column():
    algo = RECAPAlgorithm(
        policy=_PolicyDouble(),
        value_policy=_ValueDouble(),
        config=RECAPAlgorithmConfig(value_update_freq=0),
    )
    batch = _make_batch(include_advantage=False)
    batch.pop(RECAP_ADVANTAGE_FEATURE)
    with pytest.raises(ValueError, match=RECAP_ADVANTAGE_FEATURE):
        algo.update(iter([batch]))


def test_value_update_errors_without_target_column():
    algo = RECAPAlgorithm(
        policy=_PolicyDouble(),
        value_policy=_ValueDouble(),
        config=RECAPAlgorithmConfig(policy_update_freq=0),
    )
    batch = _make_batch()
    batch.pop(RECAP_VALUE_TARGET_FEATURE)
    with pytest.raises(ValueError, match=RECAP_VALUE_TARGET_FEATURE):
        algo.update(iter([batch]))


def test_policy_gradient_actually_flows():
    policy = _PolicyDouble()
    algo = RECAPAlgorithm(
        policy=policy,
        value_policy=_ValueDouble(),
        config=RECAPAlgorithmConfig(value_update_freq=0),
    )
    start = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    algo.update(iter([_make_batch()]))
    assert any(
        not torch.equal(start[k], policy.state_dict()[k]) for k in start
    ), "policy weights did not change after one update step"


def test_relabel_advantages_matches_plain_helpers():
    rng = np.random.default_rng(0)
    returns = rng.uniform(-1.0, 0.0, size=20).astype(np.float32)
    values = rng.uniform(-1.0, 0.0, size=20).astype(np.float32)
    algo = RECAPAlgorithm(
        policy=_PolicyDouble(),
        value_policy=_ValueDouble(),
        config=RECAPAlgorithmConfig(),
    )
    advs, indicator, threshold = algo.relabel_advantages(
        normalized_returns=returns,
        predicted_values=values,
        episode_ranges=[(0, 0, 20)],
        n_step=0,
        positive_percentile=0.5,
    )
    assert advs.shape == (20,)
    assert indicator.shape == (20,)
    assert 0.0 < float(indicator.mean()) < 1.0
    assert pytest.approx(threshold, abs=1e-6) == np.quantile(advs, 0.5)
