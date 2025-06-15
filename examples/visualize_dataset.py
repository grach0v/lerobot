#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""
This script demonstrates how to visualize a LeRobot dataset using the Rerun SDK.

It loads a dataset from the Hugging Face Hub, and logs the images and actions to Rerun
for interactive visualization.
"""

import argparse
from typing import Optional

import rerun as rr
import torch
from jaxtyping import Float

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def visualize_dataset(
    repo_id: str,
    episode_index: int = 0,
    token: Optional[str] = None,
):
    """
    Load a LeRobotDataset and visualize it with Rerun.

    Args:
        repo_id: The Hugging Face Hub repository ID of the dataset.
        episode_index: The index of the episode to visualize.
        token: Optional Hugging Face token for private repositories.
    """
    # Initialize Rerun
    rr.init("LeRobot Dataset Visualization", spawn=True)

    # Load the dataset
    dataset = LeRobotDataset(repo_id, token=token)
    print(f"Loaded dataset '{repo_id}' with {dataset.num_episodes} episodes.")

    if episode_index >= dataset.num_episodes:
        print(f"Error: Episode index {episode_index} is out of bounds.")
        print(f"Please choose an index between 0 and {dataset.num_episodes - 1}.")
        return

    # Get the data for the selected episode
    from_idx = dataset.episode_data_index["from"][episode_index].item()
    to_idx = dataset.episode_data_index["to"][episode_index].item()

    # Log data for each frame in the episode
    for i in range(from_idx, to_idx):
        rr.set_time(sequence="frame", value=i - from_idx)

        frame_data = dataset[i]

        # Log images
        for cam_key in dataset.meta.camera_keys:
            if cam_key in frame_data:
                # Rerun expects HWC format, but dataset provides CHW
                image: Float[torch.Tensor, "c h w"] = frame_data[cam_key]
                rr.log(f"images/{cam_key}", rr.Image(image.permute(1, 2, 0).numpy()))

        # Log actions
        if "action" in frame_data:
            action = frame_data["action"]
            rr.log("action", rr.SeriesLine())
            rr.log("action/components", rr.Scalars(action.numpy()))

        # Log states
        if "observation.state" in frame_data:
            state = frame_data["observation.state"]
            rr.log("state", rr.SeriesLine())
            rr.log("state/components", rr.Scalars(state.numpy()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a LeRobot dataset with Rerun.")
    parser.add_argument(
        "repo_id",
        type=str,
        help="The Hugging Face Hub repository ID of the dataset (e.g., 'lerobot/aloha_mobile_cabinet').",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="The index of the episode to visualize.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        dest="token",
        help="Your Hugging Face token for private repositories.",
    )
    args = parser.parse_args()

    visualize_dataset(args.repo_id, args.episode, args.token) 