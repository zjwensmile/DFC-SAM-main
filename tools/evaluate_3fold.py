#!/usr/bin/env python
"""Run the three public held-out folds, with resumable multi-GPU sharding."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from dfc_sam.models.release_checkpoint import read_release_checkpoint

EXPECTED = {1: (3, 2722), 2: (1, 2656), 3: (2, 2523)}


def _completed(path: Path, split_id: int, shard_index: int, shard_count: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("status") == "completed"
        and int(payload.get("split_id", -1)) == split_id
        and int(payload.get("shard_index", -1)) == shard_index
        and int(payload.get("shard_count", -1)) == shard_count
    )


def _preflight(weights_root: Path, manifest_root: Path) -> dict[int, tuple[Path, Path]]:
    """Validate every checkpoint/manifest mapping before launching a long GPU run."""
    assets = {}
    for split_id, (test_fold, expected_count) in EXPECTED.items():
        weight = weights_root / f"split{split_id}/rfdetr_dfc_sam_2xl_split{split_id}.pt"
        manifest = manifest_root / f"split_{split_id}.json"
        split = json.loads(manifest.read_text(encoding="utf-8"))
        if split.get("test_folds") != [test_fold] or int(split["sample_counts"]["test"]) != expected_count:
            raise ValueError(f"Split{split_id} manifest fold/count mismatch")
        checkpoint = read_release_checkpoint(weight)
        if int(checkpoint.get("split_id", -1)) != split_id or int(checkpoint.get("test_fold", -1)) != test_fold:
            raise ValueError(f"Split{split_id} checkpoint split/test-fold metadata mismatch")
        rotations = checkpoint["config"]["fold_rotations"]
        rotation = rotations.get(split_id, rotations.get(str(split_id)))
        if rotation is None or (
            list(rotation["train"]) != list(split["train_folds"])
            or list(rotation["validation"]) != list(split["val_folds"])
            or list(rotation["test"]) != list(split["test_folds"])
        ):
            raise ValueError(f"Split{split_id} checkpoint and manifest fold rotations differ")
        print(
            f"PREFLIGHT OK Split{split_id}: train=Fold{split['train_folds'][0]} "
            f"validation=Fold{split['val_folds'][0]} test=Fold{test_fold} ({expected_count} images)",
            flush=True,
        )
        assets[split_id] = (weight, manifest)
        del checkpoint
    return assets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-root", type=Path, default=PROJECT_ROOT / "weights")
    parser.add_argument("--all-samples", type=Path, default=PROJECT_ROOT / "data/manifests/base/all_samples.jsonl")
    parser.add_argument("--manifest-root", type=Path, default=PROJECT_ROOT / "data/manifests/pannuke_standard_3fold")
    parser.add_argument("--metrics-repo", type=Path, default=PROJECT_ROOT / "third_party/PanNuke-metrics")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results/threefold_test")
    parser.add_argument("--devices", default="0", help="Comma-separated physical GPU IDs")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    assets = _preflight(args.weights_root, args.manifest_root)
    if args.preflight_only:
        print("All three official PanNuke split mappings are valid.", flush=True)
        return
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    if not devices:
        raise SystemExit("--devices must contain at least one GPU")
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for split_id in EXPECTED:
        weight, manifest = assets[split_id]
        processes = []
        for shard_index, physical_device in enumerate(devices):
            shard_root = output_root / f"split{split_id}" / "shards"
            shard = shard_root / f"shard_{shard_index}.json"
            progress = shard_root / f"shard_{shard_index}_progress.json"
            log = shard_root / f"shard_{shard_index}.log"
            shard_root.mkdir(parents=True, exist_ok=True)
            if args.resume and _completed(shard, split_id, shard_index, len(devices)):
                print(f"SKIP completed Split{split_id} shard{shard_index}", flush=True)
                continue
            command = [
                sys.executable,
                str(PROJECT_ROOT / "tools/evaluate_split.py"),
                "--checkpoint",
                str(weight),
                "--all-samples",
                str(args.all_samples),
                "--split-manifest",
                str(manifest),
                "--metrics-repo",
                str(args.metrics_repo),
                "--split-id",
                str(split_id),
                "--shard-index",
                str(shard_index),
                "--shard-count",
                str(len(devices)),
                "--output",
                str(shard),
                "--progress",
                str(progress),
                "--device",
                "cuda:0",
                "--batch-size",
                str(args.batch_size),
            ]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = physical_device
            environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            handle = log.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command, cwd=PROJECT_ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT
            )
            processes.append((process, handle, log))
            print(f"START Split{split_id} shard{shard_index} on GPU {physical_device}", flush=True)
        failures = []
        for process, handle, log in processes:
            returncode = process.wait()
            handle.close()
            if returncode:
                failures.append(f"{log} (exit={returncode})")
        if failures:
            raise RuntimeError(f"Split{split_id} failed: {failures}")
        print(f"DONE Split{split_id}", flush=True)
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools/summarize_results.py"),
            "--input-root",
            str(output_root),
            "--output-root",
            str(output_root / "summary"),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
