#!/usr/bin/env python
"""Resolve public training templates against user-downloaded initialization weights."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml
from _bootstrap import PROJECT_ROOT  # noqa: F401

from dfc_sam.config import load_config


def _class_weights(split_manifest: Path) -> list[float]:
    payload = json.loads(split_manifest.read_text(encoding="utf-8"))
    counts = [int(value) for value in payload["class_instance_counts"]["train"]]
    if len(counts) != 5 or any(value <= 0 for value in counts):
        raise ValueError(f"Invalid training class counts in {split_manifest}: {counts}")
    inverse_roots = [1.0 / math.sqrt(value) for value in counts]
    mean_weight = sum(inverse_roots) / len(inverse_roots)
    return [value / mean_weight for value in inverse_roots]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--sam-h", type=Path, required=True)
    parser.add_argument("--warmup", type=Path)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.template)
    config.pop("_meta", None)
    config["weights"]["detector_stage1"] = str(args.detector.expanduser().resolve())
    config["weights"]["sam_vit_h"] = str(args.sam_h.expanduser().resolve())
    config["loss"]["ugca_class_weights"] = _class_weights(args.split_manifest.expanduser().resolve())
    # Public test configs are self-contained architecture bases. Remove their
    # validation-frozen class-aware/TTA policy when starting a new training run.
    config["inference"] = {
        "pre_threshold": 0.05,
        "max_instances": 400,
        "mask_threshold": 0.50,
        "final_score_threshold": 0.25,
        "use_box_nms": False,
        "use_mask_nms": False,
    }
    if args.warmup is not None:
        config["weights"]["warmup_stage2"] = str(args.warmup.expanduser().resolve())
    config["all_samples_manifest"] = str((Path("data/manifests/base/all_samples.jsonl")).resolve())
    config["evaluation"]["official_metrics_repo"] = str(Path("third_party/PanNuke-metrics").resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
