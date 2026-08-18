#!/usr/bin/env python
"""Prepare official PanNuke arrays and deterministic three-fold manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from _bootstrap import PROJECT_ROOT  # noqa: F401
from PIL import Image

from dfc_sam.data.manifests import build_manifests, build_standard_3fold_manifests
from dfc_sam.data.pannuke_discovery import write_discovery
from dfc_sam.data.pannuke_instances import extract_instances

EXPECTED_FOLD_COUNTS = {1: 2656, 2: 2523, 3: 2722}


def _label_lines(mask: np.ndarray) -> str:
    lines = []
    height, width = mask.shape[:2]
    for instance in extract_instances(mask):
        x1, y1, x2, y2 = (float(value) for value in instance.box_xyxy)
        center_x = (x1 + x2) / (2.0 * width)
        center_y = (y1 + y2) / (2.0 * height)
        box_width = (x2 - x1) / width
        box_height = (y2 - y1) / height
        lines.append(f"{int(instance.class_id)} {center_x:.8f} {center_y:.8f} {box_width:.8f} {box_height:.8f}")
    return "\n".join(lines) + ("\n" if lines else "")


def _prepare_fold(item: dict, prepared: Path) -> dict[str, int]:
    fold = int(item["fold"])
    images = np.load(item["images"], mmap_mode="r")
    masks = np.load(item["masks"], mmap_mode="r")
    tissues = np.load(item["types"], mmap_mode="r")
    expected = EXPECTED_FOLD_COUNTS[fold]
    if not (len(images) == len(masks) == len(tissues) == expected):
        raise ValueError(f"Fold{fold} expected {expected} aligned samples")
    image_root = prepared / "images" / f"fold{fold}"
    label_root = prepared / "labels" / f"fold{fold}"
    image_root.mkdir(parents=True, exist_ok=True)
    label_root.mkdir(parents=True, exist_ok=True)
    for index in range(expected):
        image_id = f"fold{fold}_{index:06d}"
        image_path = image_root / f"{image_id}.png"
        label_path = label_root / f"{image_id}.txt"
        if not image_path.exists():
            image = np.asarray(images[index])
            if image.shape != (256, 256, 3):
                raise ValueError(f"Unexpected image shape at {image_id}: {image.shape}")
            Image.fromarray(image.astype(np.uint8, copy=False), mode="RGB").save(image_path)
        if not label_path.exists():
            label_path.write_text(_label_lines(np.asarray(masks[index])), encoding="utf-8")
        if index % 250 == 0:
            print(f"Fold{fold}: {index}/{expected}", flush=True)
    return {"fold": fold, "images": expected}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=PROJECT_ROOT / "data/pannuke/raw")
    parser.add_argument("--prepared-root", type=Path, default=PROJECT_ROOT / "data/pannuke/prepared")
    parser.add_argument("--manifest-root", type=Path, default=PROJECT_ROOT / "data/manifests")
    args = parser.parse_args()
    raw = args.raw_root.expanduser().resolve()
    prepared = args.prepared_root.expanduser().resolve()
    manifests = args.manifest_root.expanduser().resolve()
    manifests.mkdir(parents=True, exist_ok=True)
    discovery_path = manifests / "discovered_raw.json"
    discovery = write_discovery(raw, discovery_path)
    prepared.mkdir(parents=True, exist_ok=True)
    fold_results = [_prepare_fold(item, prepared) for item in discovery["folds"]]
    base_root = manifests / "base"
    base = build_manifests(discovery_path, base_root, prepared_root=prepared)
    standard_root = manifests / "pannuke_standard_3fold"
    standard = build_standard_3fold_manifests(
        base["all_samples"],
        base_root / "dataset_fingerprint.json",
        standard_root,
        prepared_root=prepared,
        seed=42,
    )
    summary = {
        "status": "completed",
        "folds": fold_results,
        "sample_count": sum(EXPECTED_FOLD_COUNTS.values()),
        "all_samples": base["all_samples"],
        "standard_protocol": standard,
    }
    (manifests / "preparation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
