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

"""Add manual episode success/failure labels to a LeRobot dataset.

Reads a JSON or CSV mapping of episode_index -> success/failure, and writes a
derived LeRobot dataset that materializes `episode_success` and `episode_failure`
as per-frame columns. The source dataset is left untouched.
"""

from __future__ import annotations

import argparse

from lerobot.datasets.dataset_tools import add_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.rl.algorithms.recap_io import prepare_derived_dataset_root
from lerobot.rl.algorithms.recap_targets import (
    EPISODE_FAILURE_FEATURE,
    EPISODE_SUCCESS_FEATURE,
    build_episode_success_array,
    parse_episode_labels,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-id", required=True, help="Source dataset repo id.")
    parser.add_argument("--root", default=None, help="Source dataset root.")
    parser.add_argument("--output-repo-id", required=True, help="Derived dataset repo id.")
    parser.add_argument("--output-root", default=None, help="Derived dataset root.")
    parser.add_argument(
        "--labels-path",
        required=True,
        help="Path to a .json or .csv file mapping episode index to success/failure.",
    )
    args = parser.parse_args()

    dataset = LeRobotDataset(args.repo_id, root=args.root)
    episode_labels = parse_episode_labels(args.labels_path)
    episode_success, episode_failure = build_episode_success_array(dataset, episode_labels)

    prepare_derived_dataset_root(args.output_root, dataset.root)
    output_dataset = add_features(
        dataset,
        features={
            EPISODE_SUCCESS_FEATURE: (
                episode_success,
                {"dtype": "float32", "shape": (1,), "names": None},
            ),
            EPISODE_FAILURE_FEATURE: (
                episode_failure,
                {"dtype": "float32", "shape": (1,), "names": None},
            ),
        },
        output_dir=args.output_root,
        repo_id=args.output_repo_id,
    )
    print(
        f"Wrote labeled dataset to repo_id={output_dataset.repo_id}, root={output_dataset.root}, "
        f"features={list(output_dataset.meta.features.keys())}"
    )


if __name__ == "__main__":
    main()
