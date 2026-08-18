"""Run-directory provenance and sealed test-fold access."""

from __future__ import annotations

import json
import os
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from dfc_sam.utils.hashing import atomic_write_json, sha256_file, sha256_json

RunRole = Literal["train", "validation", "test"]


class RunSafetyError(RuntimeError):
    """Raised before a run can violate a frozen experimental decision."""


@dataclass(frozen=True)
class GitState:
    commit: str
    dirty: bool


def read_git_state(project_root: str | Path) -> GitState:
    root = Path(project_root).resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return GitState(commit=commit, dirty=dirty)


def assert_role_access(
    role: RunRole,
    *,
    checkpoint: str | Path | None = None,
    frozen_decision: str | Path | None = None,
    allow_test: bool = False,
    config: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Allow train/validation freely, but require a hash-bound freeze for test."""
    if role not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown data role: {role}")
    if role != "test":
        if allow_test or frozen_decision is not None:
            raise RunSafetyError("Test authorization flags are invalid for train/validation")
        return None
    if not allow_test:
        raise RunSafetyError("Test fold access requires the explicit --allow-test flag")
    if checkpoint is None or frozen_decision is None:
        raise RunSafetyError("Test fold access requires a checkpoint and frozen decision artifact")
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    decision_path = Path(frozen_decision).expanduser().resolve()
    if not checkpoint_path.is_file() or not decision_path.is_file():
        raise RunSafetyError("Checkpoint or frozen decision artifact does not exist")
    with decision_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("status") != "frozen_for_test":
        raise RunSafetyError("Decision artifact is not frozen_for_test")
    actual_sha = sha256_file(checkpoint_path)
    if payload.get("checkpoint_sha256") != actual_sha:
        raise RunSafetyError("Frozen decision is bound to a different checkpoint")
    thresholds = payload.get("inference")
    if not isinstance(thresholds, dict) or not thresholds:
        raise RunSafetyError("Frozen decision is missing inference thresholds")
    if config is not None and payload.get("config_sha256") != sha256_json(config):
        raise RunSafetyError("Frozen decision is bound to a different resolved config")
    return payload


def write_frozen_test_decision(
    destination: str | Path,
    *,
    checkpoint: str | Path,
    validation_metrics: dict[str, Any],
    inference: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Seal the validation-selected checkpoint and inference settings."""
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if "mpq" not in validation_metrics or "bpq" not in validation_metrics:
        raise ValueError("Validation metrics must contain mpq and bpq")
    payload = {
        "schema_version": 1,
        "status": "frozen_for_test",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "validation_metrics": validation_metrics,
        "inference": inference,
        "config_sha256": sha256_json(config),
    }
    atomic_write_json(destination, payload)
    return payload


def write_run_provenance(
    destination: str | Path,
    *,
    config: dict[str, Any],
    config_path: str | Path,
    git_state: GitState,
    input_hashes: dict[str, str],
) -> dict[str, Any]:
    """Atomically create the immutable run-start record."""
    destination_path = Path(destination)
    payload = {
        "schema_version": 1,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "config_path": str(Path(config_path).expanduser().resolve()),
        "resolved_config_sha256": sha256_json(config),
        "git_commit": git_state.commit,
        "git_dirty": git_state.dirty,
        "input_hashes": dict(sorted(input_hashes.items())),
    }
    atomic_write_json(destination_path, payload)
    return payload
