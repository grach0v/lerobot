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

import json

import numpy as np
import pytest

from lerobot.rl.algorithms.recap_targets import (
    RECAP_NORMALIZED_RETURN_FEATURE,
    RECAP_REWARD_FEATURE,
    RECAP_RETURN_FEATURE,
    RECAP_VALUE_TARGET_FEATURE,
    build_recap_targets_from_success,
    compute_recap_advantages,
    discretize_returns,
    normalize_returns,
    parse_episode_labels,
    parse_episode_labels_csv,
    parse_episode_labels_json,
    returns_from_rewards,
    terminal_rewards_from_success,
    threshold_recap_advantages,
    write_episode_labels_csv,
    write_episode_labels_json,
)


def test_terminal_rewards_success():
    rewards = terminal_rewards_from_success(success=True, episode_len=4, c_fail=10)
    assert rewards.tolist() == [-1.0, -1.0, -1.0, 0.0]


def test_terminal_rewards_failure():
    rewards = terminal_rewards_from_success(success=False, episode_len=4, c_fail=10)
    assert rewards.tolist() == [-1.0, -1.0, -1.0, -10.0]


def test_returns_from_rewards_reverse_cumsum():
    rewards = np.array([-1.0, -1.0, -1.0, 0.0], dtype=np.float32)
    assert returns_from_rewards(rewards).tolist() == [-3.0, -2.0, -1.0, 0.0]


def test_normalize_returns_clamps_positive_to_zero():
    returns = np.array([-3.0, -2.0, -1.0, 0.0, 1.0], dtype=np.float32)
    normalized = normalize_returns(returns, max_episode_steps=4)
    assert normalized.tolist() == [-0.75, -0.5, -0.25, 0.0, 0.0]


def test_discretize_returns_bins_are_sorted():
    normalized = np.array([-1.0, -0.5, -0.1, 0.0], dtype=np.float32)
    bins = discretize_returns(normalized, num_bins=11)
    assert bins.shape == (4,)
    assert np.all(np.diff(bins) >= 0)
    assert bins[0] == 0
    assert bins[-1] == 10


def test_normalize_requires_positive_steps():
    with pytest.raises(ValueError):
        normalize_returns(np.array([-1.0]), max_episode_steps=0)


def test_discretize_requires_multiple_bins():
    with pytest.raises(ValueError):
        discretize_returns(np.array([-0.5]), num_bins=1)


def test_build_recap_targets_shapes_and_keys():
    episode_success = np.array([1, 1, 1, 0, 0], dtype=np.float32)
    episode_ranges = [(0, 0, 3), (1, 3, 5)]
    targets = build_recap_targets_from_success(
        episode_success=episode_success,
        episode_ranges=episode_ranges,
        max_episode_steps=5,
        c_fail=5.0,
        num_bins=6,
    )
    assert set(targets) == {
        RECAP_REWARD_FEATURE,
        RECAP_RETURN_FEATURE,
        RECAP_NORMALIZED_RETURN_FEATURE,
        RECAP_VALUE_TARGET_FEATURE,
    }
    for key in targets:
        assert targets[key].shape == (5,)
    assert targets[RECAP_REWARD_FEATURE][2] == 0.0
    assert targets[RECAP_REWARD_FEATURE][4] == -5.0


def test_compute_advantages_full_return():
    normalized_returns = np.array([-0.6, -0.4, -0.2, 0.0], dtype=np.float32)
    predicted_values = np.array([-0.5, -0.35, -0.15, -0.05], dtype=np.float32)
    advantages = compute_recap_advantages(
        normalized_returns=normalized_returns,
        predicted_values=predicted_values,
        episode_ranges=[(0, 0, 4)],
        n_step=0,
    )
    np.testing.assert_allclose(
        advantages, np.array([-0.1, -0.05, -0.05, 0.05], dtype=np.float32), atol=1e-6
    )


def test_compute_advantages_n_step_bootstrap():
    normalized_returns = np.array([-0.8, -0.6, -0.3, 0.0], dtype=np.float32)
    predicted_values = np.array([-0.7, -0.5, -0.25, -0.05], dtype=np.float32)
    advantages = compute_recap_advantages(
        normalized_returns=normalized_returns,
        predicted_values=predicted_values,
        episode_ranges=[(0, 0, 4)],
        n_step=2,
    )
    np.testing.assert_allclose(
        advantages, np.array([-0.05, -0.15, -0.05, 0.05], dtype=np.float32), atol=1e-6
    )


def test_compute_advantages_shape_mismatch():
    with pytest.raises(ValueError):
        compute_recap_advantages(
            normalized_returns=np.zeros(4, dtype=np.float32),
            predicted_values=np.zeros(3, dtype=np.float32),
            episode_ranges=[(0, 0, 4)],
        )


def test_threshold_respects_percentile():
    advantages = np.array([-0.4, -0.2, 0.1, 0.2], dtype=np.float32)
    indicator, threshold = threshold_recap_advantages(
        advantages=advantages,
        positive_percentile=0.5,
    )
    assert pytest.approx(threshold, abs=1e-6) == np.quantile(advantages, 0.5)
    assert indicator.tolist() == [0.0, 0.0, 1.0, 1.0]


def test_threshold_forces_interventions_positive():
    advantages = np.array([-0.4, -0.2, 0.1, 0.2], dtype=np.float32)
    intervention = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    indicator, threshold = threshold_recap_advantages(
        advantages=advantages,
        positive_percentile=0.75,
        intervention_mask=intervention,
    )
    assert threshold >= 0.1
    assert indicator.tolist() == [0.0, 1.0, 0.0, 1.0]


def test_parse_labels_json_roundtrip(tmp_path):
    labels = {0: True, 1: False, 2: True}
    path = tmp_path / "episode_labels.json"
    write_episode_labels_json(path, labels)
    assert parse_episode_labels_json(path) == labels
    assert parse_episode_labels(path) == labels


def test_parse_labels_json_accepts_compact_form(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"0": "success", "1": "failure"}))
    assert parse_episode_labels_json(path) == {0: True, 1: False}


def test_parse_labels_csv_roundtrip(tmp_path):
    labels = {0: True, 1: False, 2: True}
    path = tmp_path / "episode_labels.csv"
    write_episode_labels_csv(path, labels)
    assert parse_episode_labels_csv(path) == labels
    assert parse_episode_labels(path) == labels


def test_parse_labels_csv_rejects_bad_schema(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("foo,bar\n0,1\n")
    with pytest.raises(ValueError):
        parse_episode_labels_csv(path)
