"""Immutable nested mask-budget manifests for PanNuke mixed supervision."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from dfc_sam.config import MIXED_MASK_RATIOS
from dfc_sam.constants import NUM_CLASSES, PANNUKE_CLASSES, PANNUKE_TISSUES
from dfc_sam.utils.hashing import atomic_write_json, sha256_file, sha256_json

from .pannuke_dataset import load_sample_records


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _label_counts(path: str | Path) -> np.ndarray:
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 5:
                raise ValueError(f"Invalid YOLO label at {path}:{line_number}")
            class_id = int(fields[0])
            if class_id not in range(NUM_CLASSES):
                raise ValueError(f"Invalid class id at {path}:{line_number}: {class_id}")
            counts[class_id] += 1
    return counts


def _profiles(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    class_counts = np.stack([_label_counts(record["yolo_label"]) for record in records])
    totals = class_counts.sum(axis=1)
    quantile_edges = np.unique(np.quantile(totals, [0.2, 0.4, 0.6, 0.8]))
    density_bin = np.searchsorted(quantile_edges, totals, side="right")
    density_bins = np.zeros((len(records), len(quantile_edges) + 1), dtype=np.float64)
    density_bins[np.arange(len(records)), density_bin] = 1.0

    tissue_index = {name: index for index, name in enumerate(PANNUKE_TISSUES)}
    tissues = np.zeros((len(records), len(PANNUKE_TISSUES)), dtype=np.float64)
    for row, record in enumerate(records):
        tissues[row, tissue_index[str(record["tissue"])]] = 1.0

    presence = (class_counts > 0).astype(np.float64)
    counts = class_counts.astype(np.float64)
    features = np.concatenate((tissues, presence, counts, density_bins), axis=1)

    weights = np.concatenate(
        (
            np.full(tissues.shape[1], 1.0 / tissues.shape[1]),
            np.full(NUM_CLASSES, 1.0 / NUM_CLASSES),
            np.full(NUM_CLASSES, 1.0 / NUM_CLASSES),
            np.full(density_bins.shape[1], 1.0 / density_bins.shape[1]),
        )
    )
    dead_count_offset = tissues.shape[1] + NUM_CLASSES
    weights[dead_count_offset + PANNUKE_CLASSES.index("dead")] *= 2.0
    return class_counts, features * weights


def _extend_nested_selection(
    selected: list[int],
    available: set[int],
    features: np.ndarray,
    target_size: int,
    target_fraction: float,
    rng: np.random.Generator,
) -> None:
    """Greedily reduce multivariate deficits while preserving prior selections."""
    total_features = features.sum(axis=0)
    target_features = total_features * target_fraction
    current = features[selected].sum(axis=0) if selected else np.zeros(features.shape[1])
    scale = np.maximum(target_features, 1.0e-12)

    while len(selected) < target_size:
        candidates = np.fromiter(sorted(available), dtype=np.int64)
        deficit = np.maximum(target_features - current, 0.0)
        gain = np.minimum(features[candidates], deficit) / scale
        overshoot = np.maximum(features[candidates] - deficit, 0.0) / np.maximum(total_features, 1.0)
        scores = gain.sum(axis=1) - 0.02 * overshoot.sum(axis=1)
        scores += rng.random(len(candidates)) * 1.0e-12
        chosen = int(candidates[int(scores.argmax())])
        selected.append(chosen)
        available.remove(chosen)
        current += features[chosen]


def _select_stratified_subset(
    candidate_indices: set[int],
    features: np.ndarray,
    target_size: int,
    rng: np.random.Generator,
) -> set[int]:
    """Select a deterministic multivariate subset from an already sealed set."""
    if target_size < 0 or target_size > len(candidate_indices):
        raise ValueError("Subset target size is outside the candidate population")
    ordered = np.asarray(sorted(candidate_indices), dtype=np.int64)
    local_features = features[ordered]
    selected: list[int] = []
    available = set(range(len(ordered)))
    fraction = target_size / len(ordered) if len(ordered) else 0.0
    _extend_nested_selection(selected, available, local_features, target_size, fraction, rng)
    return {int(ordered[index]) for index in selected}


def _statistics(records: list[dict[str, Any]], class_counts: np.ndarray, indices: set[int]) -> dict[str, Any]:
    ordered = sorted(indices)
    counts = class_counts[ordered].sum(axis=0) if ordered else np.zeros(NUM_CLASSES, dtype=np.int64)
    presence = (class_counts[ordered] > 0).sum(axis=0) if ordered else np.zeros(NUM_CLASSES, dtype=np.int64)
    tissues = Counter(str(records[index]["tissue"]) for index in ordered)
    return {
        "image_count": len(ordered),
        "total_instances": int(counts.sum()),
        "class_instance_counts": {name: int(counts[class_id]) for class_id, name in enumerate(PANNUKE_CLASSES)},
        "class_presence_image_counts": {name: int(presence[class_id]) for class_id, name in enumerate(PANNUKE_CLASSES)},
        "tissue_image_counts": dict(sorted(tissues.items())),
    }


def build_nested_supervision_manifests(
    all_samples_manifest: str | Path,
    split_manifest: str | Path,
    destination: str | Path,
    *,
    selection_seed: int,
    ratios: tuple[float, ...] = MIXED_MASK_RATIOS,
    calibration_fraction: float = 0.2,
) -> list[Path]:
    """Create nested image-level strong/weak assignments for one official split."""
    split = _read_json(split_manifest)
    split_id = int(split["split_id"])
    train_folds = {int(value) for value in split["train_folds"]}
    records = [record for record in load_sample_records(all_samples_manifest) if int(record["fold"]) in train_folds]
    if len(records) != int(split["sample_counts"]["train"]):
        raise ValueError("Training record count does not match split manifest")
    if tuple(sorted(ratios)) != tuple(ratios) or any(
        not any(abs(value - allowed) < 1.0e-9 for allowed in MIXED_MASK_RATIOS) for value in ratios
    ):
        raise ValueError(f"Ratios must be an ordered subset of {MIXED_MASK_RATIOS}")
    if any("yolo_label" not in record for record in records):
        raise ValueError("Every training record requires a sealed YOLO detection label")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0,1)")

    class_counts, features = _profiles(records)
    rng = np.random.default_rng(selection_seed)
    selected: list[int] = []
    available = set(range(len(records)))
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    parent_sha = None
    all_indices = set(range(len(records)))
    dataset_fingerprint = str(split["dataset_fingerprint"])

    for ratio in ratios:
        target_size = round(len(records) * ratio)
        _extend_nested_selection(selected, available, features, target_size, ratio, rng)
        strong_indices = set(selected)
        weak_indices = all_indices - strong_indices
        calibration_size = max(1, round(len(strong_indices) * calibration_fraction))
        calibration_rng = np.random.default_rng(selection_seed + split_id * 10_000 + round(ratio * 100))
        calibration_indices = _select_stratified_subset(
            strong_indices,
            features,
            calibration_size,
            calibration_rng,
        )
        adaptation_indices = strong_indices - calibration_indices
        if not adaptation_indices:
            raise ValueError("Teacher adaptation partition must not be empty")
        strong_ids = sorted(str(records[index]["image_id"]) for index in strong_indices)
        weak_ids = sorted(str(records[index]["image_id"]) for index in weak_indices)
        adaptation_ids = sorted(str(records[index]["image_id"]) for index in adaptation_indices)
        calibration_ids = sorted(str(records[index]["image_id"]) for index in calibration_indices)
        samples = [
            {
                "image_id": str(record["image_id"]),
                "supervision": "strong" if index in strong_indices else "box_only",
            }
            for index, record in enumerate(records)
        ]
        payload = {
            "schema_version": 2,
            "split_id": split_id,
            "train_folds": sorted(train_folds),
            "mask_ratio": ratio,
            "selection_unit": "image",
            "selection_seed": selection_seed,
            "dataset_fingerprint": dataset_fingerprint,
            "all_samples_manifest_sha256": sha256_file(all_samples_manifest),
            "split_manifest_sha256": sha256_file(split_manifest),
            "nesting_parent_sha256": parent_sha,
            "stratification": [
                "tissue",
                "five_class_presence",
                "per_class_instance_count",
                "dead_instance_count",
                "total_instance_count_quantile",
            ],
            "strong_image_ids": strong_ids,
            "weak_image_ids": weak_ids,
            "teacher_partition": {
                "calibration_fraction": calibration_fraction,
                "adaptation_image_ids": adaptation_ids,
                "quality_calibration_image_ids": calibration_ids,
            },
            "samples": samples,
            "statistics": {
                "all": _statistics(records, class_counts, all_indices),
                "strong": _statistics(records, class_counts, strong_indices),
                "weak": _statistics(records, class_counts, weak_indices),
                "teacher_adaptation": _statistics(records, class_counts, adaptation_indices),
                "quality_calibration": _statistics(records, class_counts, calibration_indices),
            },
        }
        payload["sha256"] = sha256_json(payload)
        path = output / f"mask{round(ratio * 100):02d}_seed{selection_seed}.json"
        atomic_write_json(path, payload)
        paths.append(path)
        parent_sha = payload["sha256"]
    verify_nested_supervision_manifests(paths)
    return paths


def verify_nested_supervision_manifests(paths: list[str | Path]) -> dict[str, Any]:
    """Validate hashes, complete partitions, and strict subset nesting."""
    payloads = [_read_json(path) for path in paths]
    payloads.sort(key=lambda item: float(item["mask_ratio"]))
    previous: set[str] = set()
    split_ids = {int(payload["split_id"]) for payload in payloads}
    if len(split_ids) != 1:
        raise ValueError("A nested manifest family must belong to one split")
    for payload in payloads:
        expected = sha256_json({key: value for key, value in payload.items() if key != "sha256"})
        if payload["sha256"] != expected:
            raise ValueError("Supervision manifest checksum mismatch")
        strong = set(payload["strong_image_ids"])
        weak = set(payload["weak_image_ids"])
        if strong & weak or len(strong | weak) != len(payload["samples"]):
            raise ValueError("Strong and weak images must form a complete disjoint partition")
        teacher_partition = payload.get("teacher_partition", {})
        adaptation = set(teacher_partition.get("adaptation_image_ids", []))
        calibration = set(teacher_partition.get("quality_calibration_image_ids", []))
        if adaptation & calibration or adaptation | calibration != strong:
            raise ValueError("Teacher adaptation and quality calibration must partition strong images")
        if not adaptation or not calibration:
            raise ValueError("Teacher adaptation and quality calibration partitions must both be non-empty")
        if not previous.issubset(strong):
            raise ValueError("Strong-image subsets are not nested")
        sample_mapping = {item["image_id"]: item["supervision"] for item in payload["samples"]}
        if len(sample_mapping) != len(payload["samples"]):
            raise ValueError("Supervision manifest repeats image IDs")
        if any(sample_mapping[image_id] != "strong" for image_id in strong):
            raise ValueError("Strong ID list disagrees with samples")
        if any(sample_mapping[image_id] != "box_only" for image_id in weak):
            raise ValueError("Weak ID list disagrees with samples")
        previous = strong
    return {
        "split_id": split_ids.pop(),
        "ratios": [float(payload["mask_ratio"]) for payload in payloads],
        "strong_counts": [len(payload["strong_image_ids"]) for payload in payloads],
        "status": "passed",
    }
