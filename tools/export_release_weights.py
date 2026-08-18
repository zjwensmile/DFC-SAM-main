#!/usr/bin/env python
"""Strip a formal training checkpoint into a path-neutral inference release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from _bootstrap import PROJECT_ROOT  # noqa: F401

from dfc_sam.models.release_checkpoint import RELEASE_FORMAT


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_config(source: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "dataset",
        "num_classes",
        "class_names",
        "background_channel",
        "image_size",
        "seed",
        "fold_rotations",
        "runtime",
        "detector",
        "sam",
        "bridge",
        "ugca",
        "experiment",
        "supervision",
        "inference",
        "evaluation",
        "augmentation",
    )
    config = {key: source[key] for key in keep if key in source}
    config["evaluation"] = dict(config["evaluation"])
    config["evaluation"]["official_metrics_repo"] = "third_party/PanNuke-metrics"
    config["runtime"] = {
        key: config["runtime"][key]
        for key in (
            "num_workers",
            "pin_memory",
            "persistent_workers",
            "instance_chunk_size",
            "inference_instance_chunk_size",
        )
        if key in config["runtime"]
    }
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--split-id", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    metrics_path = args.metrics.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    training = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
    model = training.get("model") if isinstance(training, dict) else None
    if not isinstance(model, dict):
        raise TypeError("Training checkpoint has no model state dictionary")
    split_id = int(args.split_id)
    if int(config["experiment"]["split_id"]) != split_id:
        raise ValueError("Config split does not match --split-id")
    if int(metrics_payload["split_id"]) != split_id or metrics_payload.get("status") != "completed":
        raise ValueError("Formal metrics do not match a completed split")
    release = {
        "format": RELEASE_FORMAT,
        "format_version": 1,
        "architecture": "RF-DETR-2XL + SAM-H + DFB + UGCA-v3",
        "split_id": split_id,
        "test_fold": int(metrics_payload["test_fold"]),
        "class_names": ["neoplastic", "inflammatory", "connective", "dead", "epithelial"],
        "model": model,
        "config": _safe_config(config),
        "expected_test_metrics": metrics_payload["metrics"],
        "provenance": {
            "source_checkpoint_sha256": _sha256(source),
            "source_project_commit": metrics_payload.get("repositories", {}).get("project", {}).get("commit"),
            "rfdetr_commit": metrics_payload.get("repositories", {}).get("rf_detr", {}).get("commit"),
            "segment_anything_commit": metrics_payload.get("repositories", {})
            .get("segment_anything", {})
            .get("commit"),
            "official_metrics_commit": metrics_payload.get("repositories", {})
            .get("official_metrics", {})
            .get("commit"),
        },
        "licenses": {
            "rfdetr_plus": "PML-1.0; user must accept and comply with the Roboflow Platform Agreement",
            "pannuke": "CC BY-NC-SA 4.0",
            "segment_anything": "Apache-2.0",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(release, output)
    output.chmod(0o644)
    print(json.dumps({"output": str(output), "bytes": output.stat().st_size, "sha256": _sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
