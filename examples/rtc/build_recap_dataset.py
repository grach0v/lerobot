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

"""Derive RECAP reward, return, normalized return, and value-bin columns.

Takes a dataset that already has `episode_success` (see label_recap_episodes.py)
and writes a derived dataset with the per-frame targets that
`recap_train_value_network` consumes directly — no separate CSV or JSON cache.
"""

from __future__ import annotations

import argparse

import numpy as np

from lerobot.datasets.dataset_tools import add_features, modify_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.rl.algorithms.recap_io import prepare_derived_dataset_root
from lerobot.rl.algorithms.recap_targets import (
    EPISODE_SUCCESS_FEATURE,
    RECAP_ADVANTAGE_FEATURE,
    RECAP_INTERVENTION_FEATURE,
    RECAP_NORMALIZED_RETURN_FEATURE,
    RECAP_REWARD_FEATURE,
    RECAP_RETURN_FEATURE,
    RECAP_VALUE_TARGET_FEATURE,
    build_recap_targets_from_success,
    episode_ranges_from_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-id", required=True, help="Source dataset repo id.")
    parser.add_argument("--root", default=None, help="Source dataset root.")
    parser.add_argument("--output-repo-id", required=True, help="Derived dataset repo id.")
    parser.add_argument("--output-root", default=None, help="Derived dataset root.")
    parser.add_argument("--num-bins", type=int, default=50, help="Number of value bins.")
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=0,
        help="Maximum task horizon in environment steps. Defaults to the longest observed episode.",
    )
    parser.add_argument(
        "--c-fail",
        type=float,
        default=0.0,
        help="Terminal failure penalty. Defaults to max_episode_steps when omitted.",
    )
    parser.add_argument(
        "--default-advantage",
        choices=["", "positive", "negative"],
        default="",
        help="Optional default advantage_condition to stamp onto every frame, useful for demo-only SFT datasets.",
    )
    parser.add_argument(
        "--default-intervention",
        type=float,
        default=0.0,
        help="Default intervention flag to stamp onto every frame.",
    )
    args = parser.parse_args()

    dataset = LeRobotDataset(args.repo_id, root=args.root)
    if EPISODE_SUCCESS_FEATURE not in dataset.meta.features:
        raise ValueError(
            f"Dataset does not contain '{EPISODE_SUCCESS_FEATURE}'. "
            "Run label_recap_episodes.py first, or record with the RECAP recording script."
        )

    episode_ranges = episode_ranges_from_dataset(dataset)
    max_episode_steps = args.max_episode_steps or max(end - start for _, start, end in episode_ranges)
    c_fail = args.c_fail if args.c_fail > 0 else float(max_episode_steps)
    episode_success = np.asarray(dataset.hf_dataset[EPISODE_SUCCESS_FEATURE], dtype=np.float32)

    recap_targets = build_recap_targets_from_success(
        episode_success=episode_success,
        episode_ranges=episode_ranges,
        max_episode_steps=max_episode_steps,
        c_fail=c_fail,
        num_bins=args.num_bins,
    )

    features: dict[str, tuple[np.ndarray, dict]] = {
        RECAP_REWARD_FEATURE: (
            recap_targets[RECAP_REWARD_FEATURE],
            {"dtype": "float32", "shape": (1,), "names": None},
        ),
        RECAP_RETURN_FEATURE: (
            recap_targets[RECAP_RETURN_FEATURE],
            {"dtype": "float32", "shape": (1,), "names": None},
        ),
        RECAP_NORMALIZED_RETURN_FEATURE: (
            recap_targets[RECAP_NORMALIZED_RETURN_FEATURE],
            {"dtype": "float32", "shape": (1,), "names": None},
        ),
        RECAP_VALUE_TARGET_FEATURE: (
            recap_targets[RECAP_VALUE_TARGET_FEATURE],
            {"dtype": "int64", "shape": (1,), "names": None},
        ),
    }

    if args.default_advantage:
        features[RECAP_ADVANTAGE_FEATURE] = (
            np.full(
                dataset.meta.total_frames,
                1.0 if args.default_advantage == "positive" else 0.0,
                dtype=np.float32,
            ),
            {"dtype": "float32", "shape": (1,), "names": None},
        )

    if RECAP_INTERVENTION_FEATURE not in dataset.meta.features:
        features[RECAP_INTERVENTION_FEATURE] = (
            np.full(dataset.meta.total_frames, args.default_intervention, dtype=np.float32),
            {"dtype": "float32", "shape": (1,), "names": None},
        )

    existing_feature_overrides = [name for name in features if name in dataset.meta.features]
    if existing_feature_overrides:
        stripped_root = None if args.output_root is None else f"{args.output_root}_stripped"
        stripped_repo_id = f"{args.output_repo_id}_stripped"
        prepare_derived_dataset_root(stripped_root, dataset.root)
        dataset = modify_features(
            dataset,
            remove_features=existing_feature_overrides,
            output_dir=stripped_root,
            repo_id=stripped_repo_id,
        )

    prepare_derived_dataset_root(args.output_root, dataset.root)
    output_dataset = add_features(
        dataset,
        features=features,
        output_dir=args.output_root,
        repo_id=args.output_repo_id,
    )
    print(
        f"Wrote RECAP dataset to repo_id={output_dataset.repo_id}, root={output_dataset.root}, "
        f"features={list(output_dataset.meta.features.keys())}, "
        f"max_episode_steps={max_episode_steps}, c_fail={c_fail:.1f}"
    )


if __name__ == "__main__":
    main()
