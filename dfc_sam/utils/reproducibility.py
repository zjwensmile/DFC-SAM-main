"""Randomness control and RNG checkpoint state."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool) -> None:
    """Seed Python, NumPy, and PyTorch with explicit deterministic behavior."""
    if deterministic:
        workspace_config = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if workspace_config not in {":4096:8", ":16:8"}:
            raise ValueError(
                "Deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8"
            )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True


def capture_rng_state() -> dict[str, Any]:
    """Capture RNG state for exact checkpoint continuation."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(
    state: dict[str, Any],
    *,
    allow_cuda_device_count_reduction: bool = False,
) -> None:
    """Restore a state returned by :func:`capture_rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        cuda_states = list(state["torch_cuda"])
        current_devices = torch.cuda.device_count()
        if len(cuda_states) != current_devices:
            if not allow_cuda_device_count_reduction or len(cuda_states) < current_devices:
                raise RuntimeError(
                    "CUDA RNG device count differs from checkpoint: "
                    f"{current_devices} != {len(cuda_states)}"
                )
            cuda_states = cuda_states[:current_devices]
        torch.cuda.set_rng_state_all(cuda_states)
