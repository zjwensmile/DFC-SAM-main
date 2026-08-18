"""Atomic, reproducible training checkpoints."""

from __future__ import annotations

import copy
import os
import shutil
import tempfile
from collections.abc import Collection
from pathlib import Path
from typing import Any

import torch

from dfc_sam.utils.hashing import sha256_json
from dfc_sam.utils.reproducibility import capture_rng_state, restore_rng_state


class CheckpointCompatibilityError(RuntimeError):
    """Raised when resume inputs differ from the original run."""


class CheckpointSpaceError(RuntimeError):
    """Raised before serialization when the filesystem cannot hold an atomic checkpoint."""


CHECKPOINT_SPACE_RESERVE_BYTES = 1 << 30


def _checkpoint_storage_bytes(value: Any) -> int:
    storages: set[tuple[str, int, int]] = set()
    visited: set[int] = set()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            storage = item.untyped_storage()
            key = (str(item.device), int(storage.data_ptr()), int(storage.nbytes()))
            storages.add(key)
            return
        if isinstance(item, dict):
            identity = id(item)
            if identity in visited:
                return
            visited.add(identity)
            for key, child in item.items():
                visit(key)
                visit(child)
            return
        if isinstance(item, (list, tuple, set)):
            identity = id(item)
            if identity in visited:
                return
            visited.add(identity)
            for child in item:
                visit(child)

    visit(value)
    return sum(size for _, _, size in storages)


def checkpoint_required_free_bytes(payload: dict[str, Any]) -> int:
    """Conservative free-space requirement for one atomic checkpoint write."""
    return checkpoint_required_free_bytes_from_size(_checkpoint_storage_bytes(payload))


def checkpoint_required_free_bytes_from_size(storage_bytes: int) -> int:
    """Estimate atomic-save headroom from an existing checkpoint's on-disk size."""
    serialization_overhead = max(256 << 20, storage_bytes // 20)
    return storage_bytes + serialization_overhead + CHECKPOINT_SPACE_RESERVE_BYTES


def assert_checkpoint_capacity(payload: dict[str, Any], destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    free = int(shutil.disk_usage(path.parent).free)
    required = checkpoint_required_free_bytes(payload)
    if free < required:
        raise CheckpointSpaceError(
            "Insufficient free space for atomic checkpoint: "
            f"free={free / (1 << 30):.2f} GiB, required={required / (1 << 30):.2f} GiB, "
            f"destination={path}"
        )


class BestCheckpointSelector:
    """Validation mPQ primary, bPQ secondary, earlier epoch tertiary."""

    def __init__(self) -> None:
        self.best: tuple[float, float, int] | None = None

    def is_better(self, *, mpq: float, bpq: float, epoch: int) -> bool:
        candidate = (float(mpq), float(bpq), -int(epoch))
        if self.best is None or candidate > self.best:
            self.best = candidate
            return True
        return False

    def state_dict(self) -> dict[str, float | int] | None:
        if self.best is None:
            return None
        return {"mpq": self.best[0], "bpq": self.best[1], "epoch": -self.best[2]}

    def load_state_dict(self, state: dict[str, float | int] | None) -> None:
        if state is None:
            self.best = None
        else:
            self.best = (float(state["mpq"]), float(state["bpq"]), -int(state["epoch"]))


def checkpoint_payload(
    *,
    epoch: int,
    global_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    config: dict,
    git_commit: str,
    dataset_fingerprint: str,
    split_manifest_sha256: str,
    pseudo_bank_sha256: str | None,
    sampler_state: dict,
    best_selector_state: dict[str, float | int] | None = None,
    micro_step_in_epoch: int = 0,
) -> dict[str, Any]:
    """Build the minimum checkpoint payload mandated by the task specification."""
    return {
        "schema_version": 2,
        "epoch": epoch,
        "micro_step_in_epoch": micro_step_in_epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng_states": capture_rng_state(),
        "sampler_state": sampler_state,
        "config": config,
        "git_commit": git_commit,
        "dataset_fingerprint": dataset_fingerprint,
        "split_manifest_sha256": split_manifest_sha256,
        "pseudo_bank_sha256": pseudo_bank_sha256,
        "best_selector": best_selector_state,
    }


def atomic_torch_save(payload: dict[str, Any], destination: str | Path) -> None:
    """Write a checkpoint completely before replacing the destination."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert_checkpoint_capacity(payload, path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_training_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a framework checkpoint on CPU and validate its mandatory fields."""
    payload = torch.load(Path(path).expanduser().resolve(), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise CheckpointCompatibilityError("Checkpoint payload must be a dictionary")
    mandatory = {
        "epoch",
        "global_step",
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "rng_states",
        "sampler_state",
        "config",
        "git_commit",
        "dataset_fingerprint",
        "split_manifest_sha256",
        "pseudo_bank_sha256",
    }
    missing = sorted(mandatory - set(payload))
    if missing:
        raise CheckpointCompatibilityError(f"Checkpoint is missing fields: {missing}")
    return payload


def restore_training_checkpoint(
    payload: dict[str, Any],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    dataset_fingerprint: str,
    split_manifest_sha256: str,
    pseudo_bank_sha256: str | None,
    strict_model: bool = True,
    current_config: dict[str, Any] | None = None,
    git_commit: str | None = None,
    allowed_checkpoint_git_commits: Collection[str] | None = None,
    allowed_config_paths: Collection[tuple[str, ...]] | None = None,
    allow_cuda_rng_device_count_reduction: bool = False,
) -> dict[str, int]:
    """Strictly restore state after verifying every data-dependent identity."""
    expected = {
        "dataset_fingerprint": dataset_fingerprint,
        "split_manifest_sha256": split_manifest_sha256,
        "pseudo_bank_sha256": pseudo_bank_sha256,
    }
    mismatches = {
        key: {"checkpoint": payload.get(key), "current": value}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise CheckpointCompatibilityError(f"Resume identity mismatch: {mismatches}")
    if current_config is not None and sha256_json(payload["config"]) != sha256_json(current_config):
        checkpoint_config = copy.deepcopy(payload["config"])
        normalized_current = copy.deepcopy(current_config)
        for path in allowed_config_paths or ():
            old_node = checkpoint_config
            new_node = normalized_current
            for key in path[:-1]:
                old_node = old_node[key]
                new_node = new_node[key]
            new_node[path[-1]] = old_node[path[-1]]
        if sha256_json(checkpoint_config) != sha256_json(normalized_current):
            raise CheckpointCompatibilityError("Resume resolved config differs from checkpoint")
    if git_commit is not None and payload["git_commit"] != git_commit:
        allowed = set(allowed_checkpoint_git_commits or ())
        if payload["git_commit"] not in allowed:
            raise CheckpointCompatibilityError("Resume Git commit differs from checkpoint")
    model.load_state_dict(payload["model"], strict=strict_model)
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        if payload["scheduler"] is None:
            raise CheckpointCompatibilityError("Checkpoint has no scheduler state")
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None:
        if payload["scaler"] is None:
            raise CheckpointCompatibilityError("Checkpoint has no scaler state")
        scaler.load_state_dict(payload["scaler"])
    restore_rng_state(
        payload["rng_states"],
        allow_cuda_device_count_reduction=allow_cuda_rng_device_count_reduction,
    )
    return {
        "epoch": int(payload["epoch"]),
        "micro_step_in_epoch": int(payload.get("micro_step_in_epoch", 0)),
        "global_step": int(payload["global_step"]),
    }
