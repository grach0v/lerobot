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

"""Minimal driver that pulls batches from a DataMixer and calls into an RLAlgorithm.

The trainer deliberately stays thin: it only owns the mixer, the optional batch
preprocessor, and the data iterator. Checkpointing, learning-rate scheduling,
and gradient clipping live on the algorithm so that any new RL algorithm can
drop in without touching this file.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from lerobot.rl.algorithms.base import RLAlgorithm, TrainingStats


class RLTrainer:
    def __init__(
        self,
        *,
        algorithm: RLAlgorithm,
        data_mixer,
        batch_size: int,
        preprocess_batch: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        async_prefetch: bool = True,
        queue_size: int = 2,
    ) -> None:
        self.algorithm = algorithm
        self.data_mixer = data_mixer
        self.batch_size = batch_size
        self.preprocess_batch = preprocess_batch
        self.async_prefetch = async_prefetch
        self.queue_size = queue_size
        self._data_iterator: Iterator[dict[str, Any]] | None = None

    def _build_data_iterator(self) -> Iterator[dict[str, Any]]:
        raw_iterator = self.algorithm.configure_data_iterator(
            self.data_mixer,
            self.batch_size,
            async_prefetch=self.async_prefetch,
            queue_size=self.queue_size,
        )
        if self.preprocess_batch is None:
            return raw_iterator
        return self._preprocess_iterator(raw_iterator)

    def _preprocess_iterator(self, raw_iterator) -> Iterator[dict[str, Any]]:
        while True:
            batch = next(raw_iterator)
            yield self.preprocess_batch(batch)

    def reset_data_iterator(self) -> None:
        self._data_iterator = None

    def set_data_mixer(self, data_mixer, *, reset: bool = True) -> None:
        self.data_mixer = data_mixer
        if reset:
            self.reset_data_iterator()

    def training_step(self) -> TrainingStats:
        if self._data_iterator is None:
            self._data_iterator = self._build_data_iterator()
        return self.algorithm.update(self._data_iterator)
