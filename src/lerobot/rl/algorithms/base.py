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

"""Abstract base for RL-style training algorithms that combine one or more policies."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import draccus
import torch
from torch import Tensor


@dataclass
class TrainingStats:
    losses: dict[str, float] = field(default_factory=dict)
    grad_norms: dict[str, float] = field(default_factory=dict)
    extra: dict[str, float] = field(default_factory=dict)

    def to_log_dict(self) -> dict[str, float]:
        log_dict: dict[str, float] = {}
        for key, value in self.losses.items():
            log_dict[f"loss/{key}"] = value
        for key, value in self.grad_norms.items():
            log_dict[f"grad_norm/{key}"] = value
        log_dict.update(self.extra)
        return log_dict


@dataclass
class RLAlgorithmConfig(draccus.ChoiceRegistry, abc.ABC):
    @property
    def type(self) -> str:
        choice_name = self.get_choice_name(self.__class__)
        if not isinstance(choice_name, str):
            raise TypeError(f"Expected string from get_choice_name, got {type(choice_name)}")
        return choice_name

    @classmethod
    def default_choice_name(cls) -> str | None:
        return None

    @classmethod
    def from_policy_config(cls, policy_cfg, **kwargs):
        _ = (policy_cfg, kwargs)
        return cls()

    @abc.abstractmethod
    def build_algorithm(self, policy, **kwargs):
        raise NotImplementedError


class RLAlgorithm(abc.ABC):
    def __init__(self) -> None:
        self._optimization_step = 0
        self.optimizers: dict[str, torch.optim.Optimizer] = {}

    @property
    def optimization_step(self) -> int:
        return self._optimization_step

    @optimization_step.setter
    def optimization_step(self, value: int) -> None:
        self._optimization_step = int(value)

    def configure_data_iterator(
        self,
        data_mixer,
        batch_size: int,
        *,
        async_prefetch: bool = True,
        queue_size: int = 2,
    ):
        return data_mixer.get_iterator(
            batch_size=batch_size,
            async_prefetch=async_prefetch,
            queue_size=queue_size,
        )

    def get_optimizers(self) -> dict[str, torch.optim.Optimizer]:
        return self.optimizers

    @abc.abstractmethod
    def select_action(self, observation: dict[str, Tensor]) -> Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def make_optimizers(self) -> dict[str, torch.optim.Optimizer]:
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, batch_iterator) -> TrainingStats:
        raise NotImplementedError

    @abc.abstractmethod
    def get_weights(self) -> dict[str, Tensor]:
        raise NotImplementedError

    @abc.abstractmethod
    def load_weights(self, weights: dict[str, Tensor], device: str = "cpu") -> None:
        raise NotImplementedError
