#!/usr/bin/env python
"""Run RF-DETR Bridge Warmup then UGCA-v3/ranking as one sealed sequence."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT

from dfc_sam.config import load_config, validate_experiment_config
from dfc_sam.engine.checkpoint import (
    CheckpointSpaceError,
    checkpoint_required_free_bytes_from_size,
)
from dfc_sam.utils.hashing import atomic_write_json, sha256_file

PRIMARY_CONFIG = PROJECT_ROOT / "configs/experiments/rfdetr_2xlarge_dfc_sam_split1.yaml"
BATCH1_CONFIG = PROJECT_ROOT / "configs/experiments/rfdetr_2xlarge_dfc_sam_split1_batch1.yaml"
CHUNK4_CONFIG = PROJECT_ROOT / "configs/experiments/rfdetr_2xlarge_dfc_sam_split1_chunk4.yaml"
ALL_SAMPLES = PROJECT_ROOT / "data/manifests/all_samples.jsonl"
SPLIT_MANIFEST = PROJECT_ROOT / "data/manifests/pannuke_standard_3fold/split_1.json"
METRICS_REPO = PROJECT_ROOT / "third_party/PanNuke-metrics"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/rfdetr_dfc_sam/sam_h_split1"
RF_PYTHON = PROJECT_ROOT / "third_party/rf-detr/.venv/bin/python"
SAM_H_BYTES = 2_564_550_879
SAM_H_SHA256 = "a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e"


def _configure_split(split_id: int) -> None:
    global PRIMARY_CONFIG, BATCH1_CONFIG, CHUNK4_CONFIG, SPLIT_MANIFEST, OUTPUT_ROOT
    config_root = PROJECT_ROOT / "configs/experiments"
    PRIMARY_CONFIG = config_root / f"rfdetr_2xlarge_dfc_sam_split{split_id}.yaml"
    BATCH1_CONFIG = config_root / f"rfdetr_2xlarge_dfc_sam_split{split_id}_batch1.yaml"
    CHUNK4_CONFIG = config_root / f"rfdetr_2xlarge_dfc_sam_split{split_id}_chunk4.yaml"
    SPLIT_MANIFEST = PROJECT_ROOT / f"data/manifests/pannuke_standard_3fold/split_{split_id}.json"
    OUTPUT_ROOT = PROJECT_ROOT / f"outputs/rfdetr_dfc_sam/sam_h_split{split_id}"


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _config_summary(path: Path) -> dict[str, Any]:
    config = load_config(path)
    validate_experiment_config(config)
    train = config["train"]
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "status": config["experiment"]["status"],
        "batch_size_per_gpu": int(train["batch_size_per_gpu"]),
        "world_size": int(train["warmup_world_size_per_split"]),
        "global_gradient_accumulation": int(train["gradient_accumulation"]),
        "local_gradient_accumulation": int(train["gradient_accumulation"])
        // int(train["warmup_world_size_per_split"]),
        "effective_global_batch": int(train["batch_size_per_gpu"])
        * int(train["gradient_accumulation"]),
        "sam_variant": str(config["sam"]["variant"]),
        "sam_weight": str(config["weights"]["sam_vit_h"]),
        "sam_weight_expected_bytes": SAM_H_BYTES,
        "sam_weight_expected_sha256": SAM_H_SHA256,
        "instance_chunk_size": int(config["runtime"]["instance_chunk_size"]),
        "inference_instance_chunk_size": int(config["runtime"]["inference_instance_chunk_size"]),
        "warmup_epochs": int(train["warmup_epochs"]),
        "joint_epochs": int(train["joint_epochs"]),
        "early_stopping": config["validation"]["early_stopping"],
    }


def _write_state(path: Path, *, phase: str, status: str, **extra: Any) -> None:
    previous: dict[str, Any] = {}
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
    atomic_write_json(
        path,
        {
            **previous,
            "schema_version": 1,
            "status": status,
            "phase": phase,
            "updated_at": _now(),
            **extra,
        },
    )


def _assert_sam_h_checkpoint() -> dict[str, Any]:
    config = load_config(PRIMARY_CONFIG)
    path = Path(config["weights"]["sam_vit_h"]).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing official SAM-H checkpoint: {path}")
    actual_bytes = path.stat().st_size
    if actual_bytes != SAM_H_BYTES:
        raise RuntimeError(f"SAM-H size mismatch: {actual_bytes} != {SAM_H_BYTES}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != SAM_H_SHA256:
        raise RuntimeError(f"SAM-H SHA256 mismatch: {actual_sha256} != {SAM_H_SHA256}")
    return {"path": str(path), "bytes": actual_bytes, "sha256": actual_sha256}


def _assert_clean_repository() -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError(f"Formal RF-DETR DFC sequence requires a clean worktree:\n{dirty}")
    return commit


def _assert_resume_storage_headroom(checkpoint: Path) -> dict[str, int]:
    """Reject a resume before GPU startup if one atomic checkpoint cannot fit."""
    checkpoint_bytes = int(checkpoint.stat().st_size)
    free_bytes = int(shutil.disk_usage(checkpoint.parent).free)
    required_bytes = checkpoint_required_free_bytes_from_size(checkpoint_bytes)
    if free_bytes < required_bytes:
        raise CheckpointSpaceError(
            "Insufficient free space to resume before GPU startup: "
            f"free={free_bytes / (1 << 30):.2f} GiB, "
            f"required={required_bytes / (1 << 30):.2f} GiB, "
            f"checkpoint={checkpoint}"
        )
    return {
        "checkpoint_bytes": checkpoint_bytes,
        "free_bytes": free_bytes,
        "required_bytes": required_bytes,
    }


def _base_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": f"{PROJECT_ROOT / 'third_party/rf-detr/src'}:{PROJECT_ROOT}",
            "OMP_NUM_THREADS": "1",
        }
    )
    return environment


def _smoke_command(config: Path, output: Path, batch_size: int) -> list[str]:
    return [
        str(RF_PYTHON),
        str(PROJECT_ROOT / "tools/smoke_train_step.py"),
        "--config",
        str(config),
        "--all-samples",
        str(ALL_SAMPLES),
        "--split-manifest",
        str(SPLIT_MANIFEST),
        "--out",
        str(output),
        "--device",
        "cuda:0",
        "--stage",
        "warmup",
        "--batch-size",
        str(batch_size),
        "--steps",
        "1",
    ]


def _select_memory_profile(run_root: Path, state_path: Path) -> tuple[Path, dict[str, Any]]:
    profiles = (
        (PRIMARY_CONFIG, 2, "sam_h_batch2_chunk8"),
        (BATCH1_CONFIG, 1, "sam_h_batch1_chunk8"),
        (CHUNK4_CONFIG, 1, "sam_h_batch1_chunk4"),
    )
    for index, (config_path, batch_size, name) in enumerate(profiles):
        _write_state(
            state_path,
            phase="preflight",
            status="running",
            attempted_profile=name,
        )
        output = run_root / f"preflight_{name}.json"
        completed = subprocess.run(
            _smoke_command(config_path, output, batch_size),
            cwd=PROJECT_ROOT,
            env=_base_environment(),
            capture_output=True,
            text=True,
        )
        log_text = completed.stdout + completed.stderr
        (run_root / f"preflight_{name}.log").write_text(log_text, encoding="utf-8")
        print(log_text, flush=True)
        if completed.returncode == 0:
            result = json.loads(output.read_text(encoding="utf-8"))
            return config_path, result
        oom = "out of memory" in log_text.lower()
        if index < len(profiles) - 1 and oom:
            print("SAM-H memory profile OOM; retrying the next frozen profile.", flush=True)
            continue
        raise RuntimeError(
            f"{name} preflight failed with exit={completed.returncode}; "
            f"see {run_root / f'preflight_{name}.log'}"
        )
    raise RuntimeError("No RF-DETR DFC memory profile passed preflight")


def _stage_command(
    stage: str,
    config: Path,
    output: Path,
    port: int,
    *,
    resume: Path | None = None,
    resume_source_git_commit: str | None = None,
) -> list[str]:
    tool = "train_bridge.py" if stage == "warmup" else "train_joint.py"
    command = [
        str(RF_PYTHON),
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=4",
        "--master_addr=127.0.0.1",
        f"--master_port={port}",
        str(PROJECT_ROOT / "tools" / tool),
        "--config",
        str(config),
        "--all-samples",
        str(ALL_SAMPLES),
        "--split-manifest",
        str(SPLIT_MANIFEST),
        "--output-root",
        str(output),
        "--device",
        "cuda:0",
        "--metrics-repo",
        str(METRICS_REPO),
        "--execute",
    ]
    if resume is not None:
        command.extend(["--resume", str(resume)])
        if resume_source_git_commit:
            command.extend(["--resume-source-git-commit", resume_source_git_commit])
    return command


def _run_stage(
    stage: str,
    config: Path,
    output: Path,
    port: int,
    *,
    resume: Path | None = None,
    resume_source_git_commit: str | None = None,
) -> None:
    subprocess.run(
        _stage_command(
            stage,
            config,
            output,
            port,
            resume=resume,
            resume_source_git_commit=resume_source_git_commit,
        ),
        cwd=PROJECT_ROOT,
        env=_base_environment(),
        check=True,
    )
    if not (output / "result.json").is_file() or not (output / "best.pt").exists():
        raise RuntimeError(f"{stage} terminated without result.json and best.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--split-id", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--warmup-only", action="store_true")
    parser.add_argument("--resume-incomplete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _configure_split(args.split_id)
    run_root = Path(args.run_root).expanduser().resolve()
    summaries = [
        _config_summary(PRIMARY_CONFIG),
        _config_summary(BATCH1_CONFIG),
        _config_summary(CHUNK4_CONFIG),
    ]
    if args.dry_run:
        print(json.dumps({"profiles": summaries, "test_access": False}, ensure_ascii=False, indent=2))
        return

    run_root.mkdir(parents=True, exist_ok=True)
    state_path = run_root / "sequence_state.json"
    previous_state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if args.resume_incomplete and state_path.is_file()
        else {}
    )
    commit = _assert_clean_repository()
    warmup_output = OUTPUT_ROOT / "warmup"
    resume_checkpoint = warmup_output / "last.pt" if args.resume_incomplete else None
    if args.resume_incomplete:
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(f"Incomplete warmup has no resumable checkpoint: {resume_checkpoint}")
        if (warmup_output / "result.json").is_file():
            raise RuntimeError("Warmup already has result.json; incomplete resume is not applicable")
        if not previous_state.get("selected_profile"):
            raise RuntimeError("Incomplete warmup state has no frozen selected_profile")
        selected_config = Path(previous_state["selected_profile"]["path"]).resolve()
        selected_summary = _config_summary(selected_config)
        if selected_summary["sha256"] != previous_state["selected_profile"]["sha256"]:
            raise RuntimeError("Incomplete warmup selected config changed before resume")
        source_commit = str(previous_state.get("git_commit", "")) or None
    else:
        for stage_output in (warmup_output, OUTPUT_ROOT / "joint"):
            if stage_output.exists() and any(stage_output.iterdir()):
                raise FileExistsError(f"Refusing to overwrite non-empty stage output: {stage_output}")
        selected_config = None
        selected_summary = None
        source_commit = None
    _write_state(
        state_path,
        phase="preflight",
        status="running",
        started_at=_now(),
        split_id=args.split_id,
        warmup_only=args.warmup_only,
        git_commit=commit,
        resume_incomplete=args.resume_incomplete,
        resume_checkpoint=str(resume_checkpoint) if resume_checkpoint is not None else None,
        resume_source_git_commit=source_commit,
        error=None,
        error_type=None,
        failed_at=None,
        profiles=summaries,
        outputs={
            "warmup": str(OUTPUT_ROOT / "warmup"),
            "joint": str(OUTPUT_ROOT / "joint"),
        },
        test_access=False,
    )
    phase = "preflight"
    try:
        if resume_checkpoint is not None:
            storage_preflight = _assert_resume_storage_headroom(resume_checkpoint)
            _write_state(
                state_path,
                phase="preflight",
                status="running",
                storage_preflight=storage_preflight,
            )
        sam_checkpoint = _assert_sam_h_checkpoint()
        _write_state(
            state_path,
            phase="preflight",
            status="running",
            sam_checkpoint=sam_checkpoint,
        )
        if selected_config is None:
            selected_config, preflight = _select_memory_profile(run_root, state_path)
            selection = _config_summary(selected_config)
            selection["preflight_peak_reserved_fraction"] = preflight["peak_reserved_fraction"]
        else:
            selection = dict(previous_state["selected_profile"])
        phase = "warmup"
        _write_state(
            state_path,
            phase=phase,
            status="running",
            selected_profile=selection,
            phase_started_at=_now(),
        )
        _run_stage(
            "warmup",
            selected_config,
            warmup_output,
            29831,
            resume=resume_checkpoint,
            resume_source_git_commit=source_commit,
        )

        if args.warmup_only:
            _write_state(
                state_path,
                phase="complete",
                status="completed",
                warmup_best=str((OUTPUT_ROOT / "warmup/best.pt").resolve()),
                joint_skipped=True,
                completed_at=_now(),
            )
            return

        phase = "joint"
        _write_state(
            state_path,
            phase=phase,
            status="running",
            warmup_best=str((OUTPUT_ROOT / "warmup/best.pt").resolve()),
            phase_started_at=_now(),
        )
        _run_stage("joint", selected_config, OUTPUT_ROOT / "joint", 29832)
        _write_state(
            state_path,
            phase="complete",
            status="completed",
            joint_best=str((OUTPUT_ROOT / "joint/best.pt").resolve()),
            completed_at=_now(),
        )
    except BaseException as error:
        _write_state(
            state_path,
            phase=phase,
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            failed_at=_now(),
        )
        raise


if __name__ == "__main__":
    main()
