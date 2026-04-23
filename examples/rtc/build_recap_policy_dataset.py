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

"""Score a RECAP dataset with a trained value network and materialise the
advantage columns consumed by ``recap_train_pi_star``.

Reads `recap_normalized_return` from the source dataset, runs the value
network over every frame to produce `recap_value_prediction`, then computes
`recap_advantage` (scalar) and `advantage_condition` (binary, thresholded at
the requested quantile). Optional `--force-interventions-positive` flips the
intervention-tagged frames to positive advantage regardless of the score —
useful when the dataset was recorded with human teleop interventions.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.datasets.dataset_tools import add_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.rl.algorithms.recap_io import prepare_derived_dataset_root
from lerobot.rl.algorithms.recap_targets import (
    RECAP_ADVANTAGE_FEATURE,
    RECAP_ADVANTAGE_SCORE_FEATURE,
    RECAP_INTERVENTION_FEATURE,
    RECAP_NORMALIZED_RETURN_FEATURE,
    RECAP_VALUE_PREDICTION_FEATURE,
    compute_recap_advantages,
    episode_ranges_from_dataset,
    threshold_recap_advantages,
)
from lerobot.rl.algorithms.recap_utils import collect_images
from lerobot.rl.algorithms.recap_value_network import RECAPValueNetwork, RECAPValueNetworkConfig


def _load_value_network(checkpoint_path: str, device: torch.device) -> RECAPValueNetwork:
    """Load Jack's RECAPValueNetwork from a .pt checkpoint."""
    print(f"Loading value network from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    vn_config = RECAPValueNetworkConfig(**checkpoint["model_config"])
    vn = RECAPValueNetwork(vn_config)
    vn.load_state_dict(checkpoint["model_state_dict"])
    del checkpoint
    gc.collect()
    vn.eval()
    vn.to(device)
    for param in vn.parameters():
        param.requires_grad = False
    print(f"Value network loaded ({sum(p.numel() for p in vn.parameters()):,} params)")
    return vn


def _build_vn_preprocessor(dataset: LeRobotDataset, tokenizer_name: str, max_length: int):
    """Mirror the preprocessor pistar training uses for advantage precomputation.

    Omits DeviceProcessorStep (value network moves inputs itself) and
    AddBatchDimensionProcessorStep (the dataloader already batches).
    """
    from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
    from lerobot.processor import (
        NormalizerProcessorStep,
        PolicyProcessorPipeline,
        TokenizerProcessorStep,
    )

    max_state_dim = 32
    steps = [
        NormalizerProcessorStep(
            features={},
            norm_map={},
            stats=dataset.meta.stats,
        ),
        Pi05PrepareStateTokenizerProcessorStep(max_state_dim=max_state_dim),
        TokenizerProcessorStep(
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            padding_side="right",
            padding="max_length",
        ),
    ]
    return PolicyProcessorPipeline(steps=steps, name="vn_preprocessor")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-id", required=True, help="Source dataset repo id.")
    parser.add_argument("--root", default=None, help="Source dataset root.")
    parser.add_argument("--output-repo-id", required=True, help="Derived dataset repo id.")
    parser.add_argument("--output-root", default=None, help="Derived dataset root.")
    parser.add_argument(
        "--value-network-checkpoint",
        required=True,
        help="Path to a .pt checkpoint produced by recap_train_value_network.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--n-step",
        type=int,
        default=50,
        help=(
            "Bootstrap horizon for the advantage estimate. Use 50 for post-training "
            "relabeling and 0 for the full-return pre-training variant."
        ),
    )
    parser.add_argument(
        "--positive-percentile",
        type=float,
        default=0.60,
        help=(
            "Frames at or above this advantage quantile get `advantage_condition=1`. "
            "The paper uses ~0.70 for pre-training and ~0.60 for fine-tuning."
        ),
    )
    parser.add_argument(
        "--tokenizer-name",
        default="google/paligemma-3b-pt-224",
    )
    parser.add_argument(
        "--tokenizer-max-length",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--force-interventions-positive",
        action="store_true",
        help="Force `advantage_condition=1` on frames where `intervention=1`.",
    )
    args = parser.parse_args()

    dataset = LeRobotDataset(args.repo_id, root=args.root)
    if RECAP_NORMALIZED_RETURN_FEATURE not in dataset.meta.features:
        raise ValueError(
            f"Source dataset is missing '{RECAP_NORMALIZED_RETURN_FEATURE}'. "
            "Run examples/rtc/build_recap_dataset.py first."
        )
    episode_ranges = episode_ranges_from_dataset(dataset)

    device = torch.device(args.device)
    vn = _load_value_network(args.value_network_checkpoint, device)
    preprocessor = _build_vn_preprocessor(dataset, args.tokenizer_name, args.tokenizer_max_length)

    predicted_values = np.zeros(dataset.num_frames, dtype=np.float32)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
    )

    processed = 0
    with torch.no_grad():
        for batch in loader:
            abs_indices = batch["index"]
            B = int(abs_indices.shape[0])
            batch = preprocessor(batch)
            images = collect_images(batch, vn.config.image_size)
            outputs = vn(batch, images)
            values = outputs["expected_value"].squeeze(-1).detach().cpu().numpy().astype(np.float32)
            for i in range(B):
                predicted_values[int(abs_indices[i].item())] = values[i]
            processed += B
            if processed == dataset.num_frames or processed // 500 != (processed - B) // 500:
                print(f"  scored {processed}/{dataset.num_frames} frames")

    normalized_returns = np.asarray(
        dataset.hf_dataset[RECAP_NORMALIZED_RETURN_FEATURE], dtype=np.float32
    ).reshape(-1)

    advantages = compute_recap_advantages(
        normalized_returns=normalized_returns,
        predicted_values=predicted_values,
        episode_ranges=episode_ranges,
        n_step=args.n_step,
    )

    intervention_mask: np.ndarray | None = None
    if args.force_interventions_positive:
        if RECAP_INTERVENTION_FEATURE not in dataset.meta.features:
            raise ValueError(
                f"Dataset has no '{RECAP_INTERVENTION_FEATURE}' column; "
                "cannot apply --force-interventions-positive."
            )
        intervention_mask = np.asarray(
            dataset.hf_dataset[RECAP_INTERVENTION_FEATURE], dtype=np.float32
        ).reshape(-1)

    advantage_indicator, threshold = threshold_recap_advantages(
        advantages=advantages,
        positive_percentile=args.positive_percentile,
        intervention_mask=intervention_mask,
    )

    prepare_derived_dataset_root(args.output_root, dataset.root)
    output_dataset = add_features(
        dataset,
        features={
            RECAP_VALUE_PREDICTION_FEATURE: (
                predicted_values,
                {"dtype": "float32", "shape": (1,), "names": None},
            ),
            RECAP_ADVANTAGE_SCORE_FEATURE: (
                advantages,
                {"dtype": "float32", "shape": (1,), "names": None},
            ),
            RECAP_ADVANTAGE_FEATURE: (
                advantage_indicator,
                {"dtype": "float32", "shape": (1,), "names": None},
            ),
        },
        output_dir=args.output_root,
        repo_id=args.output_repo_id,
    )

    positive_ratio = float(np.mean(advantage_indicator > 0))
    print(
        f"Wrote policy dataset to repo_id={output_dataset.repo_id}, root={output_dataset.root}. "
        f"Threshold={threshold:.5f}, positive_ratio={positive_ratio:.3f}"
    )

    summary_path = Path(output_dataset.root) / "meta" / "recap_advantage_build_info.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(
            {
                "value_network_checkpoint": args.value_network_checkpoint,
                "n_step": args.n_step,
                "positive_percentile": args.positive_percentile,
                "threshold": threshold,
                "positive_ratio": positive_ratio,
                "force_interventions_positive": args.force_interventions_positive,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
