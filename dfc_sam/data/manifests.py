"""Build immutable all-sample and three-fold manifests."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from dfc_sam.constants import EXPECTED_TOTAL_COUNTS, PANNUKE_FOLD_ROTATIONS, normalize_tissue
from dfc_sam.utils.hashing import atomic_write_json, sha256_file, sha256_json


def load_discovery(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if {item["fold"] for item in payload["folds"]} != {1, 2, 3}:
        raise ValueError("Discovery manifest must contain folds 1, 2, and 3")
    return payload


def build_manifests(
    discovery_path: str | Path,
    output_root: str | Path,
    *,
    prepared_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build sample and split manifests without loading image/mask arrays into RAM."""
    discovery = load_discovery(discovery_path)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    prepared = Path(prepared_root).expanduser().resolve() if prepared_root is not None else None

    records = []
    fold_counts: Counter[int] = Counter()
    fold_tissues: dict[int, Counter[str]] = {fold: Counter() for fold in (1, 2, 3)}
    for item in sorted(discovery["folds"], key=lambda row: row["fold"]):
        fold = int(item["fold"])
        images = np.load(item["images"], mmap_mode="r")
        masks = np.load(item["masks"], mmap_mode="r")
        tissues = np.load(item["types"], mmap_mode="r")
        if not (len(images) == len(masks) == len(tissues)):
            raise ValueError(f"Fold {fold} images/masks/types lengths differ")
        for index, raw_tissue in enumerate(tissues):
            tissue = normalize_tissue(str(raw_tissue))
            record = {
                "image_id": f"fold{fold}_{index:06d}",
                "fold": fold,
                "raw_index": index,
                "tissue": tissue,
                "images_npy": item["images"],
                "masks_npy": item["masks"],
                "types_npy": item["types"],
            }
            if prepared is not None:
                label_path = prepared / "labels" / f"fold{fold}" / f"{record['image_id']}.txt"
                if not label_path.is_file():
                    raise FileNotFoundError(f"Prepared YOLO label is missing: {label_path}")
                record["yolo_label"] = str(label_path)
            records.append(record)
            fold_counts[fold] += 1
            fold_tissues[fold][tissue] += 1

    if len(records) not in EXPECTED_TOTAL_COUNTS:
        raise ValueError(f"Unexpected PanNuke total sample count: {len(records)}")
    all_samples_path = output / "all_samples.jsonl"
    with all_samples_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    dataset_payload = {
        "schema_version": 1,
        "sample_count": len(records),
        "fold_counts": dict(sorted(fold_counts.items())),
        "discovery": discovery,
    }
    dataset_payload["dataset_fingerprint"] = sha256_json(dataset_payload)
    atomic_write_json(output / "dataset_fingerprint.json", dataset_payload)

    split_paths = {}
    for split_id, rotation in PANNUKE_FOLD_ROTATIONS.items():
        split_payload = {
            "schema_version": 1,
            "split_id": split_id,
            "train_folds": list(rotation["train"]),
            "val_folds": list(rotation["validation"]),
            "test_folds": list(rotation["test"]),
            "dataset_fingerprint": dataset_payload["dataset_fingerprint"],
            "sample_counts": {
                role: sum(fold_counts[fold] for fold in folds)
                for role, folds in (
                    ("train", rotation["train"]),
                    ("validation", rotation["validation"]),
                    ("test", rotation["test"]),
                )
            },
            "tissue_image_counts": {f"fold{fold}": dict(sorted(fold_tissues[fold].items())) for fold in (1, 2, 3)},
            "class_instance_counts": "PENDING_FULL_MASK_AUDIT",
        }
        split_path = output / f"split_{split_id}.json"
        atomic_write_json(split_path, split_payload)
        split_paths[split_id] = str(split_path.resolve())

    return {
        "all_samples": str(all_samples_path.resolve()),
        "dataset_fingerprint": dataset_payload["dataset_fingerprint"],
        "splits": split_paths,
    }


def _label_counts(path: str | Path) -> tuple[int, ...]:
    counts = [0] * 5
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 5:
                raise ValueError(f"Invalid YOLO label at {path}:{line_number}")
            class_id = int(fields[0])
            if class_id not in range(5):
                raise ValueError(f"Invalid YOLO class at {path}:{line_number}: {class_id}")
            counts[class_id] += 1
    return tuple(counts)


def _class_counts(records: list[dict[str, Any]], label_counts: dict[str, tuple[int, ...]]) -> list[int]:
    return [sum(label_counts[str(record["image_id"])][class_id] for record in records) for class_id in range(5)]


def _tissue_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(record["tissue"]) for record in records).items()))


def _write_detector_data(
    output: Path,
    split_id: int,
    roles: dict[str, list[dict[str, Any]]],
    prepared_root: Path,
) -> dict[str, str]:
    detector_root = output / "detector"
    detector_root.mkdir(parents=True, exist_ok=True)
    list_paths: dict[str, Path] = {}
    for role in ("train", "validation", "test"):
        list_path = detector_root / f"split{split_id}_{role}.txt"
        with list_path.open("w", encoding="utf-8") as handle:
            for record in roles[role]:
                image_path = prepared_root / "images" / f"fold{record['fold']}" / f"{record['image_id']}.png"
                if not image_path.is_file():
                    raise FileNotFoundError(f"Prepared YOLO image is missing: {image_path}")
                handle.write(str(image_path.resolve()) + "\n")
        list_paths[role] = list_path.resolve()

    data_path = detector_root / f"split{split_id}.yaml"
    data_payload = {
        "path": str(prepared_root.resolve()),
        "train": str(list_paths["train"]),
        "val": str(list_paths["validation"]),
        "test": str(list_paths["test"]),
        "names": ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"],
    }
    # JSON is valid YAML and keeps this generated artifact deterministic.
    atomic_write_json(data_path, data_payload)
    return {
        "data_yaml": str(data_path.resolve()),
        "data_yaml_sha256": sha256_file(data_path),
        **{f"{role}_list": str(path) for role, path in list_paths.items()},
        **{f"{role}_list_sha256": sha256_file(path) for role, path in list_paths.items()},
    }


def build_standard_3fold_manifests(
    all_samples_path: str | Path,
    dataset_fingerprint_path: str | Path,
    output_root: str | Path,
    *,
    prepared_root: str | Path,
    seed: int = 42,
) -> dict[str, Any]:
    """Build the official one-train/one-validation/one-test fold rotations."""
    all_samples_path = Path(all_samples_path).expanduser().resolve()
    dataset_fingerprint_path = Path(dataset_fingerprint_path).expanduser().resolve()
    prepared = Path(prepared_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = load_sample_records_for_standard_protocol(all_samples_path)
    with dataset_fingerprint_path.open(encoding="utf-8") as handle:
        dataset_payload = json.load(handle)
    dataset_fingerprint = str(dataset_payload["dataset_fingerprint"])

    label_counts: dict[str, tuple[int, ...]] = {}
    for record in records:
        label_path = record.get("yolo_label")
        if not label_path:
            raise ValueError(f"Sample has no YOLO label: {record['image_id']}")
        label_counts[str(record["image_id"])] = _label_counts(label_path)

    split_paths: dict[int, str] = {}
    detector_configs: dict[int, dict[str, str]] = {}
    for split_id, rotation in PANNUKE_FOLD_ROTATIONS.items():
        roles = {
            role: [record for record in records if int(record["fold"]) in folds] for role, folds in rotation.items()
        }
        role_ids = {role: [str(record["image_id"]) for record in role_records] for role, role_records in roles.items()}
        if set(role_ids["train"]) & set(role_ids["validation"]):
            raise RuntimeError("Standard train/validation partitions overlap")
        if (set(role_ids["train"]) | set(role_ids["validation"])) & set(role_ids["test"]):
            raise RuntimeError("Standard development/test partitions overlap")
        detector_configs[split_id] = _write_detector_data(output, split_id, roles, prepared)
        split_payload: dict[str, Any] = {
            "schema_version": 2,
            "protocol": "pannuke_standard_3fold",
            "partition_source": "PanNuke official pre-defined folds",
            "partition_method": "one_train_fold_one_validation_fold_one_test_fold",
            "split_id": split_id,
            "seed": int(seed),
            "train_folds": list(rotation["train"]),
            "val_folds": list(rotation["validation"]),
            "test_folds": list(rotation["test"]),
            "train_image_ids": role_ids["train"],
            "validation_image_ids": role_ids["validation"],
            "test_image_ids": role_ids["test"],
            "dataset_fingerprint": dataset_fingerprint,
            "sample_counts": {role: len(role_records) for role, role_records in roles.items()},
            "class_instance_counts": {
                role: _class_counts(role_records, label_counts) for role, role_records in roles.items()
            },
            "tissue_image_counts": {role: _tissue_counts(role_records) for role, role_records in roles.items()},
            "detector_data": detector_configs[split_id],
        }
        split_payload["split_fingerprint"] = sha256_json(split_payload)
        split_path = output / f"split_{split_id}.json"
        atomic_write_json(split_path, split_payload)
        split_paths[split_id] = str(split_path.resolve())

    summary = {
        "protocol": "pannuke_standard_3fold",
        "partition_source": "PanNuke official pre-defined folds",
        "partition_method": "one_train_fold_one_validation_fold_one_test_fold",
        "seed": int(seed),
        "all_samples": str(all_samples_path),
        "dataset_fingerprint": dataset_fingerprint,
        "splits": split_paths,
        "detector_configs": detector_configs,
    }
    summary["protocol_fingerprint"] = sha256_json(summary)
    atomic_write_json(output / "protocol.json", summary)
    return summary


def load_sample_records_for_standard_protocol(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    identities: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            image_id = str(record["image_id"])
            if image_id in identities:
                raise ValueError(f"Duplicate image_id: {image_id}")
            identities.add(image_id)
            records.append(record)
    if {int(record["fold"]) for record in records} != {1, 2, 3}:
        raise ValueError("Standard protocol requires PanNuke folds 1, 2, and 3")
    return records
