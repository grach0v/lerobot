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

from lerobot.rl.algorithms.base import RLAlgorithm, RLAlgorithmConfig, TrainingStats
from lerobot.rl.algorithms.recap import RECAPAlgorithm, RECAPAlgorithmConfig


def make_algorithm(
    policy,
    *,
    algorithm_name: str | None = None,
    algorithm_cfg: RLAlgorithmConfig | None = None,
    policy_cfg=None,
    **kwargs,
) -> RLAlgorithm:
    """Instantiate an RLAlgorithm either from a ready-built config or a registry name."""
    if algorithm_cfg is None:
        if algorithm_name is None:
            raise ValueError("Either algorithm_name or algorithm_cfg must be provided.")
        config_cls = RLAlgorithmConfig.get_choice_class(algorithm_name)
        if config_cls is None:
            raise ValueError(f"No RLAlgorithmConfig registered for '{algorithm_name}'.")
        if policy_cfg is not None:
            algorithm_cfg = config_cls.from_policy_config(policy_cfg, **kwargs)
        else:
            algorithm_cfg = config_cls()

    return algorithm_cfg.build_algorithm(policy=policy, **kwargs)


__all__ = [
    "RECAPAlgorithm",
    "RECAPAlgorithmConfig",
    "RLAlgorithm",
    "RLAlgorithmConfig",
    "TrainingStats",
    "make_algorithm",
]
