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

"""Math helpers that turn episode success/fail labels into RECAP targets.

The functions here intentionally operate on plain numpy arrays so they can be
reused from the offline dataset builders, the value-network trainer, and the
rollout recording script without pulling in any torch or dataset dependency.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

EPISODE_SUCCESS_FEATURE = "episode_success"
EPISODE_FAILURE_FEATURE = "episode_failure"
RECAP_INTERVENTION_FEATURE = "intervention"
RECAP_ADVANTAGE_FEATURE = "advantage_condition"
RECAP_ADVANTAGE_SCORE_FEATURE = "recap_advantage"
RECAP_REWARD_FEATURE = "recap_reward"
RECAP_RETURN_FEATURE = "recap_return"
RECAP_NORMALIZED_RETURN_FEATURE = "recap_normalized_return"
RECAP_VALUE_PREDICTION_FEATURE = "recap_value_prediction"
RECAP_VALUE_TARGET_FEATURE = "recap_value_target_bin"

CSV_EPISODE_INDEX_COLUMN = "episode_index"
CSV_SUCCESS_COLUMN = "success"


def _column_to_list(column) -> list:
    if hasattr(column, "tolist"):
        return column.tolist()
    return list(column)


def episode_ranges_from_dataset(dataset) -> list[tuple[int, int, int]]:
    """Return (episode_index, start, end) tuples for every episode in a LeRobotDataset."""
    ranges = []
    from_indices = _column_to_list(dataset.meta.episodes["dataset_from_index"])
    to_indices = _column_to_list(dataset.meta.episodes["dataset_to_index"])
    for episode_index, (start, end) in enumerate(zip(from_indices, to_indices, strict=True)):
        ranges.append((episode_index, int(start), int(end)))
    return ranges


def _coerce_label_value(value: Any, episode_index: int) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"success", "true", "1"}:
            return True
        if normalized in {"failure", "false", "0"}:
            return False
        raise ValueError(f"Unsupported label value '{value}' for episode {episode_index}.")
    return bool(value)


def parse_episode_labels_json(labels_path: str | Path) -> dict[int, bool]:
    """Parse a JSON label file (trossen_recap schema)."""
    with open(labels_path) as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "episode_labels" in raw:
        raw = raw["episode_labels"]

    if not isinstance(raw, dict):
        raise ValueError("Labels JSON must be a mapping of episode index to success/failure label.")

    return {int(key): _coerce_label_value(value, int(key)) for key, value in raw.items()}


def parse_episode_labels_csv(labels_path: str | Path) -> dict[int, bool]:
    """Parse a CSV label file (PR #3245 schema: episode_index,success)."""
    parsed: dict[int, bool] = {}
    with open(labels_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if CSV_EPISODE_INDEX_COLUMN not in row or CSV_SUCCESS_COLUMN not in row:
                raise ValueError(
                    f"CSV label file must have columns '{CSV_EPISODE_INDEX_COLUMN}' and "
                    f"'{CSV_SUCCESS_COLUMN}'. Got {list(row.keys())}."
                )
            episode_index = int(row[CSV_EPISODE_INDEX_COLUMN])
            parsed[episode_index] = _coerce_label_value(row[CSV_SUCCESS_COLUMN], episode_index)
    return parsed


def parse_episode_labels(labels_path: str | Path) -> dict[int, bool]:
    """Load episode labels from either JSON (.json) or CSV (.csv)."""
    path = Path(labels_path)
    if path.suffix.lower() == ".csv":
        return parse_episode_labels_csv(path)
    return parse_episode_labels_json(path)


def write_episode_labels_csv(
    labels_path: str | Path,
    episode_labels: dict[int, bool],
) -> None:
    path = Path(labels_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[CSV_EPISODE_INDEX_COLUMN, CSV_SUCCESS_COLUMN])
        writer.writeheader()
        for episode_index in sorted(episode_labels):
            writer.writerow(
                {
                    CSV_EPISODE_INDEX_COLUMN: episode_index,
                    CSV_SUCCESS_COLUMN: 1 if episode_labels[episode_index] else 0,
                }
            )


def write_episode_labels_json(
    labels_path: str | Path,
    episode_labels: dict[int, bool],
) -> None:
    path = Path(labels_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "episode_labels": {
            str(episode_index): "success" if episode_labels[episode_index] else "failure"
            for episode_index in sorted(episode_labels)
        }
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def build_episode_success_array(
    dataset, episode_labels: dict[int, bool]
) -> tuple[np.ndarray, np.ndarray]:
    total_frames = dataset.meta.total_frames
    episode_success = np.zeros(total_frames, dtype=np.float32)
    episode_failure = np.zeros(total_frames, dtype=np.float32)

    for episode_index, start, end in episode_ranges_from_dataset(dataset):
        if episode_index not in episode_labels:
            raise ValueError(f"Missing success/failure label for episode {episode_index}.")
        success = bool(episode_labels[episode_index])
        episode_success[start:end] = 1.0 if success else 0.0
        episode_failure[start:end] = 0.0 if success else 1.0

    return episode_success, episode_failure


def terminal_rewards_from_success(success: bool, episode_len: int, c_fail: float) -> np.ndarray:
    """-1 per step, terminal bonus of 0 on success or -c_fail on failure."""
    rewards = np.full(episode_len, -1.0, dtype=np.float32)
    rewards[-1] = 0.0 if success else -float(c_fail)
    return rewards


def returns_from_rewards(rewards: np.ndarray) -> np.ndarray:
    """Compute undiscounted return-to-go at every timestep."""
    return np.flip(np.cumsum(np.flip(rewards, axis=0), axis=0)).astype(np.float32)


def normalize_returns(returns: np.ndarray, max_episode_steps: int) -> np.ndarray:
    if max_episode_steps <= 0:
        raise ValueError("max_episode_steps must be > 0.")
    normalized = returns / float(max_episode_steps)
    return np.clip(normalized, -1.0, 0.0).astype(np.float32)


def discretize_returns(normalized_returns: np.ndarray, num_bins: int) -> np.ndarray:
    if num_bins <= 1:
        raise ValueError("num_bins must be > 1.")
    bin_edges = np.linspace(-1.0, 0.0, num_bins + 1)
    return np.clip(np.digitize(normalized_returns, bin_edges) - 1, 0, num_bins - 1).astype(np.int64)


def compute_recap_advantages(
    normalized_returns: np.ndarray,
    predicted_values: np.ndarray,
    episode_ranges: list[tuple[int, int, int]],
    n_step: int = 0,
) -> np.ndarray:
    """Scalar advantage A(o_t, a_t). With n_step=0 collapses to R_t - V(o_t)."""
    normalized_returns = np.asarray(normalized_returns, dtype=np.float32)
    predicted_values = np.asarray(predicted_values, dtype=np.float32)

    if normalized_returns.shape != predicted_values.shape:
        raise ValueError(
            "normalized_returns and predicted_values must have the same shape. "
            f"Got {normalized_returns.shape} vs {predicted_values.shape}."
        )

    advantages = np.zeros_like(normalized_returns, dtype=np.float32)
    n_step = max(int(n_step), 0)

    for _, start, end in episode_ranges:
        episode_returns = normalized_returns[start:end]
        episode_values = predicted_values[start:end]
        episode_len = end - start

        if n_step == 0:
            advantages[start:end] = episode_returns - episode_values
            continue

        episode_advantages = np.empty(episode_len, dtype=np.float32)
        for index in range(episode_len):
            bootstrap_index = index + n_step
            if bootstrap_index >= episode_len:
                episode_advantages[index] = episode_returns[index] - episode_values[index]
            else:
                episode_advantages[index] = (
                    episode_returns[index]
                    - episode_returns[bootstrap_index]
                    + episode_values[bootstrap_index]
                    - episode_values[index]
                )

        advantages[start:end] = episode_advantages

    return advantages


def threshold_recap_advantages(
    advantages: np.ndarray,
    positive_percentile: float,
    intervention_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Binary-threshold advantages at a quantile; force interventions to 1.0 if provided."""
    if not 0.0 <= positive_percentile <= 1.0:
        raise ValueError("positive_percentile must be between 0.0 and 1.0.")

    advantages = np.asarray(advantages, dtype=np.float32)
    threshold = float(np.quantile(advantages, positive_percentile))
    advantage_indicator = (advantages >= threshold).astype(np.float32)

    if intervention_mask is not None:
        intervention_mask = np.asarray(intervention_mask, dtype=np.float32)
        if intervention_mask.shape != advantages.shape:
            raise ValueError(
                "intervention_mask and advantages must have the same shape. "
                f"Got {intervention_mask.shape} vs {advantages.shape}."
            )
        advantage_indicator[intervention_mask > 0] = 1.0

    return advantage_indicator, threshold


def build_recap_targets_from_success(
    episode_success: np.ndarray,
    episode_ranges: list[tuple[int, int, int]],
    *,
    max_episode_steps: int,
    c_fail: float,
    num_bins: int,
) -> dict[str, np.ndarray]:
    """Build per-frame rewards/returns/normalized returns/value bins from episode-level success."""
    total_frames = len(episode_success)
    rewards = np.zeros(total_frames, dtype=np.float32)
    returns = np.zeros(total_frames, dtype=np.float32)
    normalized_returns = np.zeros(total_frames, dtype=np.float32)
    value_target_bins = np.zeros(total_frames, dtype=np.int64)

    for _, start, end in episode_ranges:
        success = bool(episode_success[start])
        episode_len = end - start
        episode_rewards = terminal_rewards_from_success(success=success, episode_len=episode_len, c_fail=c_fail)
        episode_returns = returns_from_rewards(episode_rewards)
        episode_normalized_returns = normalize_returns(episode_returns, max_episode_steps=max_episode_steps)
        episode_bins = discretize_returns(episode_normalized_returns, num_bins=num_bins)

        rewards[start:end] = episode_rewards
        returns[start:end] = episode_returns
        normalized_returns[start:end] = episode_normalized_returns
        value_target_bins[start:end] = episode_bins

    return {
        RECAP_REWARD_FEATURE: rewards,
        RECAP_RETURN_FEATURE: returns,
        RECAP_NORMALIZED_RETURN_FEATURE: normalized_returns,
        RECAP_VALUE_TARGET_FEATURE: value_target_bins,
    }
