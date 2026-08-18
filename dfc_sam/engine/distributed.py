"""Minimal deterministic multi-process runtime for one split."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.enabled:
            if self.device.type == "cuda":
                dist.barrier(device_ids=[self.local_rank])
            else:
                dist.barrier()

    def close(self) -> None:
        if self.enabled and dist.is_initialized():
            self.barrier()
            dist.destroy_process_group()

    def any_flag(self, value: bool) -> bool:
        if not self.enabled:
            return bool(value)
        flag = torch.tensor(int(value), dtype=torch.int32, device=self.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())


def initialize_distributed(device_name: str) -> DistributedContext:
    """Initialize NCCL when launched by torchrun and resolve the rank-local GPU."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size < 1 or not 0 <= rank < world_size:
        raise RuntimeError(f"Invalid distributed environment: rank={rank}, world_size={world_size}")
    if world_size == 1:
        device = torch.device(device_name)
        return DistributedContext(rank=0, local_rank=0, world_size=1, device=device)
    if not torch.cuda.is_available():
        raise RuntimeError("Multi-process formal training requires CUDA/NCCL")
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(f"LOCAL_RANK={local_rank} exceeds visible CUDA device count={torch.cuda.device_count()}")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=torch.device("cuda", local_rank),
    )


def local_gradient_accumulation(global_accumulation: int, world_size: int) -> int:
    """Preserve the configured effective batch when one split uses multiple ranks."""
    if global_accumulation < 1 or world_size < 1:
        raise ValueError("Gradient accumulation and world size must be positive")
    if global_accumulation % world_size:
        raise ValueError("train.gradient_accumulation must be divisible by the per-split distributed world size")
    return global_accumulation // world_size


def synchronize_module_buffers(
    module: torch.nn.Module,
    *,
    source_rank: int = 0,
    bucket_bytes: int = 25 << 20,
) -> None:
    """Broadcast model buffers in bounded contiguous dtype/device buckets."""
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return
    if bucket_bytes < 1:
        raise ValueError("Distributed buffer bucket size must be positive")

    grouped: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = {}
    for buffer in module.buffers():
        if buffer.numel():
            grouped.setdefault((buffer.device, buffer.dtype), []).append(buffer)

    with torch.no_grad():
        for buffers in grouped.values():
            bucket: list[torch.Tensor] = []
            size = 0

            def flush() -> None:
                nonlocal bucket, size
                if not bucket:
                    return
                communication_buffer = torch.cat([buffer.reshape(-1) for buffer in bucket])
                dist.broadcast(communication_buffer, src=source_rank)
                offset = 0
                for buffer in bucket:
                    count = buffer.numel()
                    buffer.copy_(communication_buffer[offset : offset + count].view(buffer.shape))
                    offset += count
                bucket = []
                size = 0

            for buffer in buffers:
                buffer_size = buffer.numel() * buffer.element_size()
                if bucket and size + buffer_size > bucket_bytes:
                    flush()
                bucket.append(buffer)
                size += buffer_size
            flush()
