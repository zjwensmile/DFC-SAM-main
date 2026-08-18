#!/usr/bin/env python
"""Validation-calibrate and recover RF-DETR/SAM-H UGCA in one sealed sequence."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT

from dfc_sam.config import load_config, validate_experiment_config
from dfc_sam.utils.hashing import atomic_write_json, sha256_file, sha256_json

PYTHON = PROJECT_ROOT / "third_party/rf-detr/.venv/bin/python"
ALL_SAMPLES = PROJECT_ROOT / "data/manifests/all_samples.jsonl"
SPLIT_MANIFEST = PROJECT_ROOT / "data/manifests/pannuke_standard_3fold/split_1.json"
METRICS_REPO = PROJECT_ROOT / "third_party/PanNuke-metrics"
BASE_CONFIG = PROJECT_ROOT / "configs/experiments/rfdetr_2xlarge_dfc_sam_split1.yaml"
C1_CONFIG = PROJECT_ROOT / "configs/experiments/rfdetr_2xlarge_sam_h_ugca_v3_split1.yaml"
C2_CONFIG = PROJECT_ROOT / "configs/experiments/rfdetr_2xlarge_sam_h_ugca_ranking_split1.yaml"
BASE_CHECKPOINT = PROJECT_ROOT / "outputs/rfdetr_dfc_sam/sam_h_split1/joint/best.pt"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/rfdetr_ugca_recovery/split1"
C1_OUTPUT = OUTPUT_ROOT / "c1_ugca_v3"
C2_OUTPUT = OUTPUT_ROOT / "c2_ranking"
SPLIT_ID = 1


def _configure_split(split_id: int) -> None:
    global SPLIT_MANIFEST, BASE_CONFIG, C1_CONFIG, C2_CONFIG
    global BASE_CHECKPOINT, OUTPUT_ROOT, C1_OUTPUT, C2_OUTPUT, SPLIT_ID
    SPLIT_ID = split_id
    config_root = PROJECT_ROOT / "configs/experiments"
    SPLIT_MANIFEST = PROJECT_ROOT / f"data/manifests/pannuke_standard_3fold/split_{split_id}.json"
    BASE_CONFIG = config_root / f"rfdetr_2xlarge_dfc_sam_split{split_id}.yaml"
    C1_CONFIG = config_root / f"rfdetr_2xlarge_sam_h_ugca_v3_split{split_id}.yaml"
    C2_CONFIG = config_root / f"rfdetr_2xlarge_sam_h_ugca_ranking_split{split_id}.yaml"
    parent_stage = "joint" if split_id == 1 else "warmup"
    BASE_CHECKPOINT = PROJECT_ROOT / f"outputs/rfdetr_dfc_sam/sam_h_split{split_id}/{parent_stage}/best.pt"
    OUTPUT_ROOT = PROJECT_ROOT / f"outputs/rfdetr_ugca_recovery/split{split_id}"
    C1_OUTPUT = OUTPUT_ROOT / "c1_ugca_v3"
    C2_OUTPUT = OUTPUT_ROOT / "c2_ranking"

SCORE_PRE_SHARDS = (
    (0.01, 0.02),
    (0.03, 0.05),
    (0.075, 0.10),
    (0.15, 0.20, 0.30, 0.40),
)
SCORE_FINALS = (0.02, 0.05, 0.075, 0.10, 0.15, 0.20, 0.225, 0.25, 0.275, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70)
MASK_SHARDS = ((0.30, 0.35), (0.40, 0.45), (0.50, 0.55), (0.60, 0.65, 0.70))

# Promotion is preregistered before C3 is observed. Changes smaller than 0.10pp
# mPQ are treated as validation noise, and no companion metric may regress >0.10pp.
MIN_MPQ_GAIN = 0.001
MAX_COMPANION_REGRESSION = 0.001
COMPANION_METRICS = ("bpq", "f1det", "macro_f1")


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _env(gpu: int | None = None) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "OMP_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": f"{PROJECT_ROOT / 'third_party/rf-detr/src'}:{PROJECT_ROOT}",
        }
    )
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu) if gpu is not None else "0,1,2,3"
    return environment


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _state(path: Path, *, phase: str, status: str, **extra: Any) -> None:
    previous = _read(path) if path.is_file() else {}
    atomic_write_json(
        path,
        {
            **previous,
            "schema_version": 1,
            "phase": phase,
            "status": status,
            "updated_at": _now(),
            **extra,
        },
    )


def _assert_clean_repository() -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    if dirty:
        raise RuntimeError(f"Formal UGCA recovery requires a clean committed worktree:\n{dirty}")
    return commit


def _validate_inputs(*, include_ranking: bool = True) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    configs = (BASE_CONFIG, C1_CONFIG, C2_CONFIG) if include_ranking else (BASE_CONFIG, C1_CONFIG)
    for path in configs:
        config = load_config(path)
        validate_experiment_config(config)
        summaries.append(
            {
                "path": str(path),
                "resolved_sha256": sha256_json(config),
                "mode": config["train"]["joint_training_mode"],
                "epochs": int(config["train"]["joint_epochs"]),
                "batch_per_gpu": int(config["train"]["batch_size_per_gpu"]),
                "world_size": int(config["train"]["joint_world_size_per_split"]),
                "effective_batch": int(config["train"]["batch_size_per_gpu"])
                * int(config["train"]["gradient_accumulation"]),
                "early_stopping": config["validation"]["early_stopping"]["joint"],
                "loss": {
                    key: float(config["loss"][key])
                    for key in ("lambda_det", "lambda_seg", "lambda_ugca")
                },
            }
        )
    for path in (PYTHON, ALL_SAMPLES, SPLIT_MANIFEST, METRICS_REPO, BASE_CHECKPOINT):
        if not path.exists():
            raise FileNotFoundError(path)
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a non-empty recovery output: {OUTPUT_ROOT}")
    return summaries


def _calibration_command(
    *, config: Path, checkpoint: Path, output: Path, pre: tuple[float, ...],
    finals: tuple[float, ...], masks: tuple[float, ...], parent: Path | None = None,
) -> list[str]:
    command = [
        str(PYTHON), str(PROJECT_ROOT / "tools/calibrate_inference_thresholds.py"),
        "--config", str(config), "--checkpoint", str(checkpoint),
        "--all-samples", str(ALL_SAMPLES), "--split-manifest", str(SPLIT_MANIFEST),
        "--stage", "joint", "--pre-thresholds", *(str(value) for value in pre),
        "--final-score-thresholds", *(str(value) for value in finals),
        "--mask-thresholds", *(str(value) for value in masks),
        "--output", str(output), "--device", "cuda:0", "--metric-workers", "4",
        "--progress-every", "25", "--execute",
    ]
    if parent is not None:
        command.extend(("--parent-calibration", str(parent)))
    return command


def _parallel_calibration_phase(
    *, label: str, phase: str, config: Path, checkpoint: Path, root: Path,
    pre_shards: tuple[tuple[float, ...], ...], finals: tuple[float, ...],
    mask_shards: tuple[tuple[float, ...], ...], parent: Path | None = None,
) -> Path:
    shard_root = root / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[subprocess.Popen[bytes], Any, Path]] = []
    shard_paths: list[Path] = []
    for index in range(4):
        output = shard_root / f"{phase}_part{index + 1}.json"
        log = shard_root / f"{phase}_part{index + 1}.log"
        shard_paths.append(output)
        handle = log.open("wb")
        command = _calibration_command(
            config=config,
            checkpoint=checkpoint,
            output=output,
            pre=pre_shards[index],
            finals=finals,
            masks=mask_shards[index],
            parent=parent,
        )
        process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=_env(index), stdout=handle, stderr=subprocess.STDOUT)
        processes.append((process, handle, log))
        print(f"{label} {phase}: GPU {index}, PID {process.pid}, log={log}", flush=True)
    failures: list[str] = []
    for process, handle, log in processes:
        return_code = process.wait()
        handle.close()
        if return_code:
            failures.append(f"{log} (exit={return_code})")
    if failures:
        raise RuntimeError(f"{label} {phase} calibration shard failure: {failures}")
    merged = root / f"{phase}_grid.json"
    subprocess.run(
        [str(PYTHON), str(PROJECT_ROOT / "tools/merge_threshold_calibration_shards.py"),
         "--shards", *(str(path) for path in shard_paths), "--output", str(merged)],
        cwd=PROJECT_ROOT, env=_env(), check=True,
    )
    return merged


def _calibrate(label: str, config: Path, checkpoint: Path, root: Path) -> Path:
    score = _parallel_calibration_phase(
        label=label, phase="score", config=config, checkpoint=checkpoint, root=root,
        pre_shards=SCORE_PRE_SHARDS, finals=SCORE_FINALS,
        mask_shards=((0.50,),) * 4,
    )
    thresholds = _read(score)["selected"]["thresholds"]
    fixed_pre = ((float(thresholds["pre_threshold"]),),) * 4
    fixed_final = (float(thresholds["final_score_threshold"]),)
    return _parallel_calibration_phase(
        label=label, phase="mask", config=config, checkpoint=checkpoint, root=root,
        pre_shards=fixed_pre, finals=fixed_final, mask_shards=MASK_SHARDS, parent=score,
    )


def _stage_command(config: Path, output: Path, port: int) -> list[str]:
    return [
        str(PYTHON), "-m", "torch.distributed.run", "--nproc_per_node=4",
        "--master_addr=127.0.0.1", f"--master_port={port}",
        str(PROJECT_ROOT / "tools/train_joint.py"),
        "--config", str(config), "--all-samples", str(ALL_SAMPLES),
        "--split-manifest", str(SPLIT_MANIFEST), "--output-root", str(output),
        "--device", "cuda:0", "--metrics-repo", str(METRICS_REPO), "--execute",
    ]


def _run_stage(config: Path, output: Path, port: int) -> None:
    subprocess.run(_stage_command(config, output, port), cwd=PROJECT_ROOT, env=_env(), check=True)
    result = output / "result.json"
    best = output / "best.pt"
    if not result.is_file() or _read(result).get("status") != "completed" or not best.is_file():
        raise RuntimeError(f"Training stage did not produce a completed result and best checkpoint: {output}")


def _core(calibration: Path) -> dict[str, float]:
    metrics = _read(calibration)["selected"]["metrics"]
    return {key: float(metrics[key]) for key in ("bpq", "mpq", "f1det", "macro_f1")}


def should_promote(c0_metrics: dict[str, float], c3_metrics: dict[str, float]) -> bool:
    """Apply the preregistered validation-only recovery gate."""
    deltas = {key: float(c3_metrics[key]) - float(c0_metrics[key]) for key in c0_metrics}
    epsilon = 1.0e-12
    return deltas["mpq"] + epsilon >= MIN_MPQ_GAIN and all(
        deltas[key] + epsilon >= -MAX_COMPANION_REGRESSION for key in COMPANION_METRICS
    )


def _freeze_winner(c0: Path, c3: Path) -> dict[str, Any]:
    c0_metrics = _core(c0)
    c3_metrics = _core(c3)
    deltas = {key: c3_metrics[key] - c0_metrics[key] for key in c0_metrics}
    promoted = should_promote(c0_metrics, c3_metrics)
    winner = "c3_recovered" if promoted else "c0_parent"
    config = C2_CONFIG if promoted else BASE_CONFIG
    checkpoint = C2_OUTPUT / "best.pt" if promoted else BASE_CHECKPOINT
    calibration = c3 if promoted else c0
    decision_root = OUTPUT_ROOT / "decision"
    decision_root.mkdir(parents=True, exist_ok=False)
    comparison = {
        "schema_version": 1,
        "status": "completed",
        "role": "validation",
        "sealed_test_access": False,
        "selection_rule": {
            "primary": f"C3 mPQ - C0 mPQ >= {MIN_MPQ_GAIN}",
            "safety": f"C3-C0 for {list(COMPANION_METRICS)} must each be >= {-MAX_COMPANION_REGRESSION}",
        },
        "c0_parent": c0_metrics,
        "c3_recovered": c3_metrics,
        "c3_minus_c0": deltas,
        "winner": winner,
        "promoted": promoted,
        "selected_checkpoint": str(checkpoint.resolve()),
        "selected_checkpoint_sha256": sha256_file(checkpoint),
        "selected_calibration": str(calibration.resolve()),
        "selected_calibration_sha256": sha256_file(calibration),
        "reference_target": (
            {
                "name": "prior calibrated YOLO26-X UGCA-v3 Split1",
                "bpq": 0.67511,
                "mpq": 0.484803,
                "f1det": 0.799046,
                "macro_f1": 0.661290,
            }
            if SPLIT_ID == 1
            else None
        ),
        "created_at": _now(),
    }
    comparison_path = decision_root / "comparison.json"
    atomic_write_json(comparison_path, comparison)
    subprocess.run(
        [str(PYTHON), str(PROJECT_ROOT / "tools/freeze_threshold_calibration.py"),
         "--base-config", str(config), "--checkpoint", str(checkpoint),
         "--calibration", str(calibration),
         "--output-config", str(decision_root / "calibrated_config.yaml"),
         "--output-decision", str(decision_root / "frozen_test_decision.json")],
        cwd=PROJECT_ROOT, env=_env(), check=True,
    )
    frozen = _read(decision_root / "frozen_test_decision.json")
    frozen["recovery_comparison"] = {
        "artifact": str(comparison_path.resolve()),
        "artifact_sha256": sha256_file(comparison_path),
        "winner": winner,
        "sealed_test_access": False,
    }
    atomic_write_json(decision_root / "frozen_test_decision.json", frozen)
    return comparison


def _freeze_final_c1(c0: Path, c1_calibration: Path) -> dict[str, Any]:
    """Freeze the established C1-only path without repeating rejected ranking."""
    c0_metrics = _core(c0)
    c1_metrics = _core(c1_calibration)
    deltas = {key: c1_metrics[key] - c0_metrics[key] for key in c0_metrics}
    checkpoint = C1_OUTPUT / "best.pt"
    decision_root = OUTPUT_ROOT / "decision"
    decision_root.mkdir(parents=True, exist_ok=False)
    comparison = {
        "schema_version": 1,
        "status": "completed",
        "role": "validation",
        "sealed_test_access": False,
        "protocol": "established_final_path",
        "ranking_repeated": False,
        "c0_parent": c0_metrics,
        "c1_ugca_v3": c1_metrics,
        "c1_minus_c0": deltas,
        "winner": "c1_ugca_v3",
        "selected_checkpoint": str(checkpoint.resolve()),
        "selected_checkpoint_sha256": sha256_file(checkpoint),
        "selected_calibration": str(c1_calibration.resolve()),
        "selected_calibration_sha256": sha256_file(c1_calibration),
        "created_at": _now(),
    }
    comparison_path = decision_root / "comparison.json"
    atomic_write_json(comparison_path, comparison)
    subprocess.run(
        [str(PYTHON), str(PROJECT_ROOT / "tools/freeze_threshold_calibration.py"),
         "--base-config", str(C1_CONFIG), "--checkpoint", str(checkpoint),
         "--calibration", str(c1_calibration),
         "--output-config", str(decision_root / "calibrated_config.yaml"),
         "--output-decision", str(decision_root / "frozen_test_decision.json")],
        cwd=PROJECT_ROOT, env=_env(), check=True,
    )
    frozen = _read(decision_root / "frozen_test_decision.json")
    frozen["recovery_comparison"] = {
        "artifact": str(comparison_path.resolve()),
        "artifact_sha256": sha256_file(comparison_path),
        "winner": "c1_ugca_v3",
        "ranking_repeated": False,
        "sealed_test_access": False,
    }
    atomic_write_json(decision_root / "frozen_test_decision.json", frozen)
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--split-id", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--final-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _configure_split(args.split_id)
    summaries = _validate_inputs(include_ranking=not args.final_only)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "sealed_test_access": False,
                    "sequence": ([
                        "C1 UGCA-v3 only",
                        "C0 exact-parent validation calibration",
                        "C1 validation calibration",
                        "freeze established C1 path",
                    ] if args.final_only else [
                        "C0 parent calibration", "C1 UGCA-v3 only", "C2 ranking only",
                        "C3 calibration", "freeze validation winner",
                    ]),
                    "configs": summaries,
                    "promotion": {
                        "min_mpq_gain": MIN_MPQ_GAIN,
                        "max_companion_regression": MAX_COMPANION_REGRESSION,
                    },
                    "output_root": str(OUTPUT_ROOT),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    run_root = Path(args.run_root).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    state_path = run_root / "sequence_state.json"
    if state_path.exists():
        raise FileExistsError(f"Refusing to reuse an existing sequence state: {state_path}")
    commit = _assert_clean_repository()
    initial_phase = "c1_ugca_v3" if args.final_only else "c0_calibration"
    _state(
        state_path, phase=initial_phase, status="running", started_at=_now(),
        git_commit=commit, split_id=args.split_id, final_only=args.final_only,
        sealed_test_access=False, configs=summaries,
        outputs={
            "c0_calibration": str(OUTPUT_ROOT / "c0_parent_calibration"),
            "c1": str(C1_OUTPUT), "c2": str(C2_OUTPUT),
            "c1_calibration": str(OUTPUT_ROOT / "c1_final_calibration"),
            "c3_calibration": str(OUTPUT_ROOT / "c3_final_calibration"),
            "decision": str(OUTPUT_ROOT / "decision"),
        },
    )
    phase = initial_phase
    try:
        if args.final_only:
            _run_stage(C1_CONFIG, C1_OUTPUT, 29941)
            parent_checkpoint = C1_OUTPUT / "initial_parent.pt"
            if not parent_checkpoint.is_file():
                raise FileNotFoundError(f"C1 did not preserve its exact initial parent: {parent_checkpoint}")
            phase = "c0_calibration"
            _state(
                state_path,
                phase=phase,
                status="running",
                phase_started_at=_now(),
                c1_best=str((C1_OUTPUT / "best.pt").resolve()),
                exact_parent=str(parent_checkpoint.resolve()),
            )
            c0 = _calibrate(
                "C0", BASE_CONFIG, parent_checkpoint, OUTPUT_ROOT / "c0_parent_calibration"
            )
            phase = "c1_calibration"
            _state(
                state_path,
                phase=phase,
                status="running",
                phase_started_at=_now(),
                c1_best=str((C1_OUTPUT / "best.pt").resolve()),
            )
            c1_calibration = _calibrate(
                "C1", C1_CONFIG, C1_OUTPUT / "best.pt", OUTPUT_ROOT / "c1_final_calibration"
            )
            phase = "decision"
            _state(
                state_path,
                phase=phase,
                status="running",
                phase_started_at=_now(),
                c1_metrics=_core(c1_calibration),
            )
            comparison = _freeze_final_c1(c0, c1_calibration)
            _state(
                state_path, phase="complete", status="completed",
                completed_at=_now(), comparison=comparison,
            )
            return
        c0 = _calibrate("C0", BASE_CONFIG, BASE_CHECKPOINT, OUTPUT_ROOT / "c0_parent_calibration")
        phase = "c1_ugca_v3"
        _state(state_path, phase=phase, status="running", phase_started_at=_now(), c0_metrics=_core(c0))
        _run_stage(C1_CONFIG, C1_OUTPUT, 29941)
        phase = "c2_ranking"
        _state(
            state_path,
            phase=phase,
            status="running",
            phase_started_at=_now(),
            c1_best=str((C1_OUTPUT / "best.pt").resolve()),
        )
        _run_stage(C2_CONFIG, C2_OUTPUT, 29942)
        phase = "c3_calibration"
        _state(
            state_path,
            phase=phase,
            status="running",
            phase_started_at=_now(),
            c2_best=str((C2_OUTPUT / "best.pt").resolve()),
        )
        c3 = _calibrate("C3", C2_CONFIG, C2_OUTPUT / "best.pt", OUTPUT_ROOT / "c3_final_calibration")
        phase = "decision"
        _state(state_path, phase=phase, status="running", phase_started_at=_now(), c3_metrics=_core(c3))
        comparison = _freeze_winner(c0, c3)
        _state(state_path, phase="complete", status="completed", completed_at=_now(), comparison=comparison)
    except BaseException as error:
        _state(
            state_path, phase=phase, status="failed", error_type=type(error).__name__,
            error=str(error), failed_at=_now(),
        )
        raise


if __name__ == "__main__":
    main()
