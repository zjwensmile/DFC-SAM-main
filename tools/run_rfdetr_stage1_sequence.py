#!/usr/bin/env python
"""Run selected RF-DETR variants on one PanNuke split and evaluate validation."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT
from train_rfdetr_split1 import VARIANTS

from dfc_sam.engine.runtime import read_git_state
from dfc_sam.utils.hashing import atomic_write_json

SEQUENCE = ("2xlarge", "large")
TRAIN_ATTEMPTS: dict[str, tuple[dict[str, Any], ...]] = {
    "2xlarge": (
        {
            "batch_size_per_gpu": 4,
            "grad_accum_steps": 1,
            "gradient_checkpointing": True,
            "multi_scale": True,
            "expanded_scales": True,
            "profile": "native-880/batch4 throughput-first",
        },
        {
            "batch_size_per_gpu": 2,
            "grad_accum_steps": 2,
            "gradient_checkpointing": True,
            "multi_scale": True,
            "expanded_scales": True,
            "profile": "native-880/batch2 OOM fallback",
        },
        {
            "batch_size_per_gpu": 1,
            "grad_accum_steps": 4,
            "gradient_checkpointing": True,
            "multi_scale": True,
            "expanded_scales": True,
            "profile": "native-880/batch1 OOM fallback",
        },
        {
            "batch_size_per_gpu": 1,
            "grad_accum_steps": 4,
            "gradient_checkpointing": True,
            "multi_scale": False,
            "expanded_scales": False,
            "profile": "native-880/static-scale OOM fallback",
        },
    ),
    "large": (
        {
            "batch_size_per_gpu": 4,
            "grad_accum_steps": 1,
            "gradient_checkpointing": False,
            "multi_scale": True,
            "expanded_scales": True,
            "profile": "native-704/throughput-first",
        },
        {
            "batch_size_per_gpu": 2,
            "grad_accum_steps": 2,
            "gradient_checkpointing": False,
            "multi_scale": True,
            "expanded_scales": True,
            "profile": "native-704/batch2 OOM fallback",
        },
        {
            "batch_size_per_gpu": 1,
            "grad_accum_steps": 4,
            "gradient_checkpointing": True,
            "multi_scale": True,
            "expanded_scales": True,
            "profile": "native-704/checkpoint OOM fallback",
        },
    ),
}
YOLO_BASELINE = (
    PROJECT_ROOT
    / "outputs/yolo_stage1_recovery/20260808_175201/recovery_b/discovery_result.json"
)
def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall <= 0.0 else 2.0 * precision * recall / (precision + recall)


def _absolute_without_resolving(path: Path) -> Path:
    """Make a path absolute while preserving a virtual-environment Python symlink."""
    return Path(os.path.abspath(path.expanduser()))


def _tail_contains_oom(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 2 * 1024 * 1024))
        text = handle.read().decode("utf-8", errors="replace").lower()
    markers = (
        "cuda out of memory",
        "torch.outofmemoryerror",
        "cuda error: out of memory",
        "cudnn_status_alloc_failed",
    )
    return any(marker in text for marker in markers)


def _process_children() -> dict[int, list[int]]:
    """Read the Linux process tree without adding a psutil dependency."""
    children: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
            # comm is parenthesized and may contain spaces, so split only after
            # its final closing parenthesis: state, ppid, pgrp, session, ...
            fields = stat[stat.rfind(")") + 2 :].split()
            ppid = int(fields[1])
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError):
            continue
        children.setdefault(ppid, []).append(int(entry.name))
    return children


def _descendant_pids(root_pid: int) -> list[int]:
    children = _process_children()
    descendants: list[int] = []
    stack = list(children.get(root_pid, ()))
    while stack:
        pid = stack.pop()
        descendants.append(pid)
        stack.extend(children.get(pid, ()))
    return descendants


def _pid_alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        return stat[stat.rfind(")") + 2 :].split()[0] != "Z"
    except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError):
        return False


def _signal_process_tree(root_pid: int, signum: int, *, known: list[int] | None = None) -> list[int]:
    """Signal every known descendant, including Lightning's per-rank sessions."""
    descendants = _descendant_pids(root_pid) if known is None else known
    for pid in reversed(descendants):
        try:
            os.kill(pid, signum)
        except ProcessLookupError:
            pass
    try:
        os.kill(root_pid, signum)
    except ProcessLookupError:
        pass
    return descendants


class Controller:
    def __init__(
        self,
        artifact_root: Path,
        output_root: Path,
        python: Path,
        epochs: int,
        split_id: int = 1,
        models: tuple[str, ...] = SEQUENCE,
    ) -> None:
        self.artifact_root = artifact_root
        self.output_root = output_root
        self.python = python
        self.epochs = epochs
        self.split_id = split_id
        self.sequence = models
        self.dataset = PROJECT_ROOT / f"artifacts/rfdetr_split{split_id}/dataset"
        self.state_path = artifact_root / "state.json"
        self.child: subprocess.Popen[Any] | None = None
        self.stop_requested = False
        git_state = read_git_state(PROJECT_ROOT)
        self.state: dict[str, Any] = {
            "schema_version": 1,
            "state": "STARTING",
            "phase": "startup",
            "controller_pid": os.getpid(),
            "started_at": _now(),
            "updated_at": _now(),
            "git_commit": git_state.commit,
            "git_dirty": git_state.dirty,
            "gpu": "0,1,2,3",
            "python": str(python),
            "split_id": split_id,
            "sequence": list(self.sequence),
            "dashboard_title": "RF-DETR " + " -> ".join(self.sequence),
            "artifact_root": str(artifact_root),
            "output_root": str(output_root),
            "dataset": str(self.dataset),
            "evaluation_protocol": f"Split{split_id} validation; Ultralytics YOLO metric implementation; sealed test untouched",
            "current_model": None,
            "models": {
                name: {
                    "label": "RF-DETR-2XL" if name == "2xlarge" else "RF-DETR-L",
                    "state": "PENDING",
                    "resolution": VARIANTS[name]["resolution"],
                    "license": VARIANTS[name]["license"],
                    "weight": str(VARIANTS[name]["weight"]),
                    "epochs": epochs,
                    "effective_batch": 16,
                    "attempts": [],
                    "evaluation_attempts": [],
                }
                for name in self.sequence
            },
        }

    def write(self) -> None:
        self.state["updated_at"] = _now()
        atomic_write_json(self.state_path, self.state)

    def request_stop(self, _signum: int, _frame: object) -> None:
        self.stop_requested = True
        self.state["state"] = "STOPPING"
        self.write()

    def _run_child(self, command: list[str], log_path: Path, environment: dict[str, str]) -> int:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab", buffering=0) as log:
            self.child = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            while True:
                try:
                    return_code = self.child.wait(timeout=1.0)
                    break
                except subprocess.TimeoutExpired:
                    if not self.stop_requested:
                        continue
                    root_pid = self.child.pid
                    tracked = _signal_process_tree(root_pid, signal.SIGTERM)
                    deadline = time.monotonic() + 15.0
                    while any(_pid_alive(pid) for pid in [root_pid, *tracked]) and time.monotonic() < deadline:
                        time.sleep(0.25)
                    # Lightning puts each DDP rank in a separate session.  A
                    # torchrun process-group signal therefore misses them; use
                    # the exact PID snapshot if graceful shutdown times out.
                    if any(_pid_alive(pid) for pid in [root_pid, *tracked]):
                        _signal_process_tree(root_pid, signal.SIGKILL, known=tracked)
                    try:
                        return_code = self.child.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        _signal_process_tree(root_pid, signal.SIGKILL)
                        return_code = self.child.wait()
                    break
        self.child = None
        return return_code

    def _isolated_environment(self, visible_devices: str) -> dict[str, str]:
        """Return an environment that torchrun children cannot escape via a resolved symlink."""
        environment = os.environ.copy()
        virtual_env = self.python.parent.parent
        python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        site_packages = virtual_env / "lib" / python_version / "site-packages"
        python_paths = [str(PROJECT_ROOT / "third_party/rf-detr/src"), str(site_packages)]
        if existing_python_path := environment.get("PYTHONPATH"):
            python_paths.append(existing_python_path)
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": visible_devices,
                "PATH": f"{self.python.parent}{os.pathsep}{environment.get('PATH', '')}",
                "PYTHONPATH": os.pathsep.join(python_paths),
                "PYTHONUNBUFFERED": "1",
                "PYTHON_EXEC": str(self.python),
                "TOKENIZERS_PARALLELISM": "false",
                "VIRTUAL_ENV": str(virtual_env),
            }
        )
        return environment

    def train_model(self, name: str) -> int:
        entry = self.state["models"][name]
        self.state["current_model"] = name
        self.state["phase"] = "training"
        self.state["state"] = "RUNNING"
        entry["state"] = "RUNNING"
        entry["started_at"] = _now()
        self.write()
        for attempt_index, profile in enumerate(TRAIN_ATTEMPTS[name], start=1):
            attempt_output = self.output_root / f"rfdetr_{name}" / f"attempt_{attempt_index}"
            log_path = self.artifact_root / "logs" / f"train_{name}_attempt{attempt_index}.log"
            attempt = {
                **profile,
                "number": attempt_index,
                "world_size": 4,
                "effective_batch": profile["batch_size_per_gpu"] * profile["grad_accum_steps"] * 4,
                "output": str(attempt_output),
                "log": str(log_path),
                "state": "RUNNING",
                "started_at": _now(),
            }
            entry["attempts"].append(attempt)
            entry["active_output"] = str(attempt_output)
            entry["active_log"] = str(log_path)
            entry["active_attempt"] = attempt_index
            self.write()
            command = [
                str(self.python),
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=4",
                str(PROJECT_ROOT / "tools/train_rfdetr_split1.py"),
                "--split-id",
                str(self.split_id),
                "--variant",
                name,
                "--dataset",
                str(self.dataset),
                "--output",
                str(attempt_output),
                "--resolution",
                str(VARIANTS[name]["resolution"]),
                "--epochs",
                str(self.epochs),
                "--batch-size",
                str(profile["batch_size_per_gpu"]),
                "--grad-accum-steps",
                str(profile["grad_accum_steps"]),
                "--workers",
                "4",
                "--devices",
                "auto",
                "--execute",
            ]
            if profile["gradient_checkpointing"]:
                command.append("--gradient-checkpointing")
            command.append("--multi-scale" if profile["multi_scale"] else "--no-multi-scale")
            command.append("--expanded-scales" if profile["expanded_scales"] else "--no-expanded-scales")
            environment = self._isolated_environment("0,1,2,3")
            environment["RFDETR_SEQUENCE_AUTHORIZED"] = "1"
            return_code = self._run_child(command, log_path, environment)
            attempt["exit_code"] = return_code
            attempt["finished_at"] = _now()
            if return_code == 0:
                checkpoint = attempt_output / "checkpoint_best_total.pth"
                if not checkpoint.is_file():
                    attempt["state"] = "FAILED_NO_CHECKPOINT"
                    entry["state"] = "FAILED"
                    self.write()
                    return 1
                attempt["state"] = "COMPLETED"
                entry["state"] = "TRAINED"
                entry["finished_at"] = _now()
                entry["successful_output"] = str(attempt_output)
                entry["best_checkpoint"] = str(checkpoint)
                self.write()
                return 0
            if self.stop_requested:
                attempt["state"] = "STOPPED"
                entry["state"] = "STOPPED"
                self.write()
                return 130
            if _tail_contains_oom(log_path) and attempt_index < len(TRAIN_ATTEMPTS[name]):
                attempt["state"] = "OOM_FALLBACK"
                entry["state"] = "RETRYING_AFTER_OOM"
                self.write()
                continue
            attempt["state"] = "FAILED"
            entry["state"] = "FAILED"
            entry["finished_at"] = _now()
            self.write()
            return return_code
        return 1

    def evaluate_model(self, name: str) -> int:
        entry = self.state["models"][name]
        self.state["current_model"] = name
        self.state["phase"] = "validation_evaluation"
        self.state["state"] = "RUNNING"
        entry["state"] = "EVALUATING"
        self.write()
        initial_batch = 1 if name == "2xlarge" else 4
        evaluation_batches = (initial_batch,) if initial_batch == 1 else (initial_batch, 2, 1)
        attempt_offset = len(entry["evaluation_attempts"])
        for local_attempt, batch_size in enumerate(evaluation_batches, start=1):
            attempt_index = attempt_offset + local_attempt
            evaluation_output = Path(entry["successful_output"]) / f"evaluation_attempt_{attempt_index}"
            log_path = self.artifact_root / "logs" / f"evaluate_{name}_attempt{attempt_index}.log"
            attempt = {
                "number": attempt_index,
                "batch_size": batch_size,
                "output": str(evaluation_output),
                "log": str(log_path),
                "state": "RUNNING",
                "started_at": _now(),
            }
            entry["evaluation_attempts"].append(attempt)
            entry["active_eval_log"] = str(log_path)
            self.write()
            command = [
                str(self.python),
                str(PROJECT_ROOT / "tools/evaluate_rfdetr_yolo_metrics.py"),
                "--checkpoint",
                entry["best_checkpoint"],
                "--dataset",
                str(self.dataset),
                "--output",
                str(evaluation_output),
                "--batch-size",
                str(batch_size),
                "--confidence-floor",
                "0.001",
                "--max-det",
                "400",
            ]
            environment = self._isolated_environment("0")
            return_code = self._run_child(command, log_path, environment)
            attempt["exit_code"] = return_code
            attempt["finished_at"] = _now()
            if return_code == 0:
                result_path = evaluation_output / "yolo_metrics.json"
                result = json.loads(result_path.read_text(encoding="utf-8"))
                attempt["state"] = "COMPLETED"
                entry["state"] = "COMPLETED"
                entry["evaluation_result"] = str(result_path)
                entry["metrics"] = result["metrics"]
                self.write()
                return 0
            if self.stop_requested:
                attempt["state"] = "STOPPED"
                entry["state"] = "STOPPED"
                self.write()
                return 130
            if _tail_contains_oom(log_path) and attempt_index < len(evaluation_batches):
                attempt["state"] = "OOM_FALLBACK"
                self.write()
                continue
            attempt["state"] = "FAILED"
            entry["state"] = "FAILED"
            self.write()
            return return_code
        return 1

    def resume_evaluation(self) -> int:
        """Resume only common-metric evaluation from completed training checkpoints."""
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)
        if self.state.get("state") not in {"FAILED", "STOPPED"}:
            raise RuntimeError("Evaluation resume requires a terminal FAILED or STOPPED sequence")
        for name in self.sequence:
            entry = self.state["models"][name]
            checkpoint = Path(entry.get("best_checkpoint", ""))
            successful_output = Path(entry.get("successful_output", ""))
            if not checkpoint.is_file() or not successful_output.is_dir():
                raise RuntimeError(f"Cannot resume {name}: completed training artifacts are missing")

        git_state = read_git_state(PROJECT_ROOT)
        self.state.setdefault("resume_history", []).append(
            {
                "resumed_at": _now(),
                "previous_state": self.state.get("state"),
                "previous_phase": self.state.get("phase"),
                "previous_finished_at": self.state.get("finished_at"),
                "evaluation_git_commit": git_state.commit,
                "evaluation_git_dirty": git_state.dirty,
            }
        )
        self.state["controller_pid"] = os.getpid()
        self.state["state"] = "RUNNING"
        self.state["phase"] = "validation_evaluation"
        self.state["current_model"] = None
        self.state.pop("finished_at", None)
        self.write()
        for name in self.sequence:
            entry = self.state["models"][name]
            if entry.get("evaluation_result") and Path(entry["evaluation_result"]).is_file():
                entry["state"] = "COMPLETED"
                continue
            entry["state"] = "TRAINED"
            return_code = self.evaluate_model(name)
            if return_code:
                self.state["state"] = "STOPPED" if self.stop_requested else "FAILED"
                self.state["finished_at"] = _now()
                self.write()
                return return_code
        self.write_comparison()
        self.state["state"] = "COMPLETED"
        self.state["phase"] = "complete"
        self.state["current_model"] = None
        self.state["finished_at"] = _now()
        self.write()
        return 0

    def write_comparison(self) -> None:
        candidates: list[dict[str, Any]] = []
        if self.split_id == 1 and YOLO_BASELINE.is_file():
            baseline = json.loads(YOLO_BASELINE.read_text(encoding="utf-8"))["best_validation"]
            candidates.append(
                {
                    "model": "YOLO26-X Recovery-B",
                    "source": str(YOLO_BASELINE),
                    "metrics": {
                        "precision": float(baseline["precision"]),
                        "recall": float(baseline["recall"]),
                        "f1det": _f1(float(baseline["precision"]), float(baseline["recall"])),
                        "map50": float(baseline["map50"]),
                        "map50_95": float(baseline["map50_95"]),
                    },
                }
            )
        for name in self.sequence:
            entry = self.state["models"][name]
            candidates.append(
                {
                    "model": entry["label"],
                    "source": entry["evaluation_result"],
                    "checkpoint": entry["best_checkpoint"],
                    "metrics": entry["metrics"],
                }
            )
        ranking = sorted(
            candidates,
            key=lambda item: (item["metrics"]["map50_95"], item["metrics"]["f1det"]),
            reverse=True,
        )
        comparison = {
            "schema_version": 1,
            "status": "validation_comparison_complete",
            "selection_split": f"Split{self.split_id} validation",
            "sealed_test_access": False,
            "selection_rule": "mAP50-95 descending, then F1det descending",
            "metric_formula": "Ultralytics YOLO ap_per_class; F1det=2PR/(P+R)",
            "candidates": candidates,
            "ranking": [item["model"] for item in ranking],
            "winner": ranking[0]["model"],
            "note": "Formal PanNuke test remains sealed until one checkpoint is frozen.",
        }
        comparison_path = self.output_root / "comparison.json"
        atomic_write_json(comparison_path, comparison)
        for name in self.sequence:
            checkpoint = Path(self.state["models"][name]["best_checkpoint"])
            stable = self.output_root / f"rfdetr_{name}_best.pth"
            if stable.exists() or stable.is_symlink():
                if stable.resolve() != checkpoint.resolve():
                    raise FileExistsError(f"Refusing to replace mismatched selected checkpoint: {stable}")
            else:
                stable.symlink_to(checkpoint.relative_to(self.output_root))
            self.state["models"][name]["stable_checkpoint"] = str(stable)
        self.state["comparison"] = str(comparison_path)
        self.state["winner"] = comparison["winner"]
        self.write()

    def run(self) -> int:
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        (self.artifact_root / "logs").mkdir(exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=False)
        self.write()
        for name in self.sequence:
            return_code = self.train_model(name)
            if return_code:
                self.state["state"] = "STOPPED" if self.stop_requested else "FAILED"
                self.state["finished_at"] = _now()
                self.write()
                return return_code
        for name in self.sequence:
            return_code = self.evaluate_model(name)
            if return_code:
                self.state["state"] = "STOPPED" if self.stop_requested else "FAILED"
                self.state["finished_at"] = _now()
                self.write()
                return return_code
        self.write_comparison()
        self.state["state"] = "COMPLETED"
        self.state["phase"] = "complete"
        self.state["current_model"] = None
        self.state["finished_at"] = _now()
        self.write()
        return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--split-id", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--models", nargs="+", choices=tuple(VARIANTS), default=list(SEQUENCE))
    parser.add_argument("--resume-evaluation", action="store_true")
    args = parser.parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    controller = Controller(
        args.artifact_root.expanduser().resolve(),
        args.output_root.expanduser().resolve(),
        _absolute_without_resolving(args.python),
        args.epochs,
        args.split_id,
        tuple(args.models),
    )
    if args.resume_evaluation:
        controller.state = json.loads(controller.state_path.read_text(encoding="utf-8"))
        if Path(controller.state["artifact_root"]).resolve() != controller.artifact_root:
            raise RuntimeError("Resume artifact root differs from recorded state")
        if Path(controller.state["output_root"]).resolve() != controller.output_root:
            raise RuntimeError("Resume output root differs from recorded state")
        raise SystemExit(controller.resume_evaluation())
    raise SystemExit(controller.run())


if __name__ == "__main__":
    main()
