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

"""Iterator-producing data sources used by the RL trainer.

A `DataMixer` exposes a single interface — `get_iterator(batch_size, ...)` — that
yields batches forever. Implementations differ only in how they combine one or
more datasets (single vs. weighted mixture) and whether they use a custom sampler
(for example the per-bin balanced sampler used during value-network training).
"""

from __future__ import annotations

import abc
import random
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch

from lerobot.datasets.utils import cycle


class DataMixer(abc.ABC):
    @abc.abstractmethod
    def get_iterator(
        self,
        batch_size: int,
        *,
        async_prefetch: bool = True,
        queue_size: int = 2,
    ) -> Iterator[dict[str, Any]]:
        raise NotImplementedError


@dataclass
class DatasetSource:
    name: str
    dataset: Any
    weight: float = 1.0


class SingleDatasetMixer(DataMixer):
    def __init__(
        self,
        dataset,
        *,
        num_workers: int = 0,
        shuffle: bool = True,
        pin_memory: bool = False,
        drop_last: bool = False,
        sampler=None,
    ) -> None:
        self.dataset = dataset
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.sampler = sampler

    def get_iterator(
        self,
        batch_size: int,
        *,
        async_prefetch: bool = True,
        queue_size: int = 2,
    ):
        _ = (async_prefetch, queue_size)
        if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
            epoch = 0
            while True:
                self.sampler.set_epoch(epoch)
                dataloader = torch.utils.data.DataLoader(
                    self.dataset,
                    num_workers=self.num_workers,
                    batch_size=batch_size,
                    shuffle=False,
                    sampler=self.sampler,
                    pin_memory=self.pin_memory,
                    drop_last=self.drop_last,
                    prefetch_factor=2 if self.num_workers > 0 else None,
                )
                yield from dataloader
                epoch += 1
            return

        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            num_workers=self.num_workers,
            batch_size=batch_size,
            shuffle=self.shuffle and self.sampler is None,
            sampler=self.sampler,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )
        yield from cycle(dataloader)


class WeightedDatasetMixer(DataMixer):
    """Mix several datasets at a fixed per-batch weight.

    Each source keeps its own shuffled, cycled dataloader. On every draw we pick
    a source by its weight, then yield one batch from that source. The source
    name is attached to the batch under `_data_source` so callers can log or
    filter by provenance.
    """

    def __init__(
        self,
        sources: list[DatasetSource],
        *,
        num_workers: int = 0,
        shuffle: bool = True,
        pin_memory: bool = False,
        drop_last: bool = False,
    ) -> None:
        if len(sources) == 0:
            raise ValueError("At least one dataset source is required.")
        self.sources = sources
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    def get_iterator(
        self,
        batch_size: int,
        *,
        async_prefetch: bool = True,
        queue_size: int = 2,
    ):
        _ = (async_prefetch, queue_size)
        iterators: list[tuple[str, Iterator[Any]]] = []
        weights: list[float] = []
        for source in self.sources:
            dataloader = torch.utils.data.DataLoader(
                source.dataset,
                num_workers=self.num_workers,
                batch_size=batch_size,
                shuffle=self.shuffle,
                pin_memory=self.pin_memory,
                drop_last=self.drop_last,
                prefetch_factor=2 if self.num_workers > 0 else None,
            )
            iterators.append((source.name, cycle(dataloader)))
            weights.append(source.weight)

        while True:
            source_index = random.choices(range(len(iterators)), weights=weights, k=1)[0]
            source_name, iterator = iterators[source_index]
            batch = next(iterator)
            if isinstance(batch, dict):
                batch = dict(batch)
                batch["_data_source"] = source_name
            yield batch
