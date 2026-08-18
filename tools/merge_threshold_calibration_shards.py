#!/usr/bin/env python
"""Merge disjoint validation-threshold grids without changing selection rules."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT  # noqa: F401

from dfc_sam.evaluation.threshold_calibration import select_best_candidate
from dfc_sam.utils.hashing import atomic_write_json, sha256_file


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("candidates"), list):
        raise TypeError(f"Invalid calibration shard: {path}")
    return payload


def merge(shard_paths: list[Path], output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite calibration output: {output}")
    payloads = [_read(path) for path in shard_paths]
    first = payloads[0]
    identity_keys = (
        "schema_version",
        "role",
        "stage",
        "split_id",
        "config_sha256",
        "checkpoint_sha256",
        "split_manifest",
        "selection_rule",
        "sample_count",
    )
    for payload in payloads[1:]:
        mismatches = [key for key in identity_keys if payload.get(key) != first.get(key)]
        if mismatches:
            raise RuntimeError(f"Calibration shard identities differ: {mismatches}")

    candidates = [candidate for payload in payloads for candidate in payload["candidates"]]
    candidate_ids = [str(candidate["candidate_id"]) for candidate in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise RuntimeError("Calibration shards contain duplicate threshold candidates")
    candidates.sort(key=lambda item: str(item["candidate_id"]))
    grid = {
        key: sorted(
            {
                float(value)
                for payload in payloads
                for value in payload["candidate_grid"][key]
            }
        )
        for key in ("pre_thresholds", "final_score_thresholds", "mask_thresholds")
    }
    merged = {
        **first,
        "candidate_count": len(candidates),
        "candidate_grid": grid,
        "metric_workers": sum(int(payload.get("metric_workers", 1)) for payload in payloads),
        "elapsed_seconds": max(float(payload["elapsed_seconds"]) for payload in payloads),
        "parallel_merge_timestamp": time.time(),
        "parallel_shards": [
            {
                "artifact": str(path.resolve()),
                "sha256": sha256_file(path),
                "candidate_count": int(payload["candidate_count"]),
                "elapsed_seconds": float(payload["elapsed_seconds"]),
            }
            for path, payload in zip(shard_paths, payloads, strict=True)
        ],
        "selected": select_best_candidate(candidates),
        "candidates": candidates,
        "output": str(output.resolve()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, merged)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    payload = merge([Path(path).expanduser().resolve() for path in args.shards], output)
    print(
        json.dumps(
            {
                "status": "completed",
                "output": str(output),
                "candidate_count": payload["candidate_count"],
                "selected": payload["selected"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
