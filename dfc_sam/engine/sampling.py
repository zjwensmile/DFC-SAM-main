"""Deterministic epoch sampling with an explicit resumable cursor."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sized

import torch
from torch.utils.data import Sampler


class EpochShuffleSampler(Sampler[int]):
    """Generate a seed/epoch-stable permutation and optionally skip a prefix."""

    def __init__(self, data_source: Sized, *, seed: int, epoch: int = 0, start_index: int = 0) -> None:
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.start_index = int(start_index)
        self._validate()

    def _validate(self) -> None:
        if self.epoch < 0:
            raise ValueError("Sampler epoch must be non-negative")
        if not 0 <= self.start_index <= len(self.data_source):
            raise ValueError("Sampler start_index is outside the dataset")

    def set_epoch(self, epoch: int, *, start_index: int = 0) -> None:
        self.epoch = int(epoch)
        self.start_index = int(start_index)
        self._validate()

    def state_dict(self) -> dict[str, int]:
        return {"seed": self.seed, "epoch": self.epoch, "start_index": self.start_index}

    def load_state_dict(self, state: dict[str, int]) -> None:
        if int(state["seed"]) != self.seed:
            raise ValueError("Sampler seed differs from checkpoint")
        self.set_epoch(int(state["epoch"]), start_index=int(state["start_index"]))

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.data_source), generator=generator).tolist()
        yield from order[self.start_index :]

    def __len__(self) -> int:
        return len(self.data_source) - self.start_index


class DistributedEpochShuffleSampler(EpochShuffleSampler):
    """Shard one stable padded epoch permutation across equal-length ranks."""

    def __init__(
        self,
        data_source: Sized,
        *,
        seed: int,
        epoch: int = 0,
        start_index: int = 0,
        rank: int,
        world_size: int,
    ) -> None:
        self.rank = int(rank)
        self.world_size = int(world_size)
        if self.world_size < 2 or not 0 <= self.rank < self.world_size:
            raise ValueError("Distributed sampler requires 0 <= rank < world_size and world_size >= 2")
        super().__init__(data_source, seed=seed, epoch=epoch, start_index=start_index)

    @property
    def samples_per_rank(self) -> int:
        return (len(self.data_source) + self.world_size - 1) // self.world_size

    def _validate(self) -> None:
        if self.epoch < 0:
            raise ValueError("Sampler epoch must be non-negative")
        if not 0 <= self.start_index <= self.samples_per_rank:
            raise ValueError("Distributed sampler start_index is outside the rank shard")

    def state_dict(self) -> dict[str, int]:
        return {
            "seed": self.seed,
            "epoch": self.epoch,
            "start_index": self.start_index,
            "world_size": self.world_size,
        }

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.data_source), generator=generator).tolist()
        total_size = self.samples_per_rank * self.world_size
        if total_size > len(order):
            padding_size = total_size - len(order)
            order.extend((order * math.ceil(padding_size / len(order)))[:padding_size])
        rank_order = order[self.rank : total_size : self.world_size]
        yield from rank_order[self.start_index :]

    def __len__(self) -> int:
        return self.samples_per_rank - self.start_index


class DistributedSequentialSampler(Sampler[int]):
    """Shard validation without padding or duplicating any image."""

    def __init__(self, data_source: Sized, *, rank: int, world_size: int) -> None:
        self.data_source = data_source
        self.rank = int(rank)
        self.world_size = int(world_size)
        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise ValueError("Sequential shard requires 0 <= rank < world_size")

    def __iter__(self) -> Iterator[int]:
        yield from range(self.rank, len(self.data_source), self.world_size)

    def __len__(self) -> int:
        remaining = len(self.data_source) - self.rank
        return 0 if remaining <= 0 else (remaining + self.world_size - 1) // self.world_size
