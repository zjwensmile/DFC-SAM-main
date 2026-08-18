"""Read-only audit of the original PanNuke NPY release."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from dfc_sam.constants import (
    EXPECTED_TOTAL_COUNTS,
    NUM_CLASSES,
    PANNUKE_BACKGROUND_CHANNEL,
    PANNUKE_TISSUES,
    normalize_tissue,
)
from dfc_sam.utils.hashing import atomic_write_json, sha256_file


def _sample_indices(length: int, limit: int | None) -> range | np.ndarray:
    if limit is None or limit >= length:
        return range(length)
    if limit <= 0:
        raise ValueError("sample_limit must be positive or None")
    return np.linspace(0, length - 1, num=limit, dtype=np.int64)


def _image_digest(image: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(image)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def audit_pannuke(
    discovery_path: str | Path,
    *,
    sample_limit: int | None = None,
    hash_source_files: bool = True,
    check_cross_fold_duplicates: bool = True,
) -> dict[str, Any]:
    """Audit shapes, values, channels, tissues, overlaps, hashes, and duplicates.

    `sample_limit=None` is the formal full audit. A positive limit is a framework
    smoke mode and is recorded in the output; it must not be used as Stage 0 evidence.
    """
    with Path(discovery_path).open(encoding="utf-8") as handle:
        discovery = json.load(handle)

    report: dict[str, Any] = {
        "schema_version": 1,
        "discovery": str(Path(discovery_path).resolve()),
        "audit_scope": "full" if sample_limit is None else f"sampled:{sample_limit}_per_fold",
        "folds": {},
        "cross_fold_duplicate_images": [],
        "status": "passed",
        "warnings": [],
    }
    total = 0
    seen_images: dict[str, str] = {}

    for item in sorted(discovery["folds"], key=lambda row: row["fold"]):
        fold = int(item["fold"])
        images = np.load(item["images"], mmap_mode="r")
        masks = np.load(item["masks"], mmap_mode="r")
        tissues = np.load(item["types"], mmap_mode="r")
        expected_n = len(images)
        total += expected_n
        if images.shape != (expected_n, 256, 256, 3):
            raise ValueError(f"Fold {fold} images shape is invalid: {images.shape}")
        if masks.shape != (expected_n, 256, 256, 6):
            raise ValueError(f"Fold {fold} masks shape is invalid: {masks.shape}")
        if tissues.shape != (expected_n,):
            raise ValueError(f"Fold {fold} types shape is invalid: {tissues.shape}")
        if not (len(images) == len(masks) == len(tissues)):
            raise ValueError(f"Fold {fold} images/masks/types lengths differ")

        raw_tissues = sorted({str(value) for value in tissues})
        normalized_tissues = sorted({normalize_tissue(value) for value in raw_tissues})
        if set(normalized_tissues) != set(PANNUKE_TISSUES):
            raise ValueError(f"Fold {fold} does not contain the expected 19 tissues")

        image_min = float("inf")
        image_max = float("-inf")
        image_nonfinite = 0
        image_nonintegral = 0
        mask_nonfinite = 0
        mask_negative = 0
        mask_nonintegral = 0
        overlap_pixels = 0
        background_mismatch_pixels = 0
        max_instance_ids = [0] * NUM_CLASSES

        for index_value in _sample_indices(expected_n, sample_limit):
            index = int(index_value)
            image = np.asarray(images[index])
            mask = np.asarray(masks[index])
            finite_image = np.isfinite(image)
            image_nonfinite += int(np.size(image) - np.count_nonzero(finite_image))
            if finite_image.any():
                image_min = min(image_min, float(image[finite_image].min()))
                image_max = max(image_max, float(image[finite_image].max()))
            image_nonintegral += int(np.count_nonzero(finite_image & (image != np.floor(image))))

            finite_mask = np.isfinite(mask)
            mask_nonfinite += int(np.size(mask) - np.count_nonzero(finite_mask))
            mask_negative += int(np.count_nonzero(finite_mask & (mask < 0)))
            mask_nonintegral += int(np.count_nonzero(finite_mask & (mask != np.floor(mask))))
            positive = mask[..., :NUM_CLASSES] > 0
            overlap_pixels += int(np.count_nonzero(positive.sum(axis=-1) > 1))
            expected_background = ~positive.any(axis=-1)
            actual_background = mask[..., PANNUKE_BACKGROUND_CHANNEL] > 0
            background_mismatch_pixels += int(np.count_nonzero(expected_background != actual_background))
            for class_id in range(NUM_CLASSES):
                max_instance_ids[class_id] = max(max_instance_ids[class_id], int(mask[..., class_id].max()))

            if check_cross_fold_duplicates:
                digest = _image_digest(image)
                image_id = f"fold{fold}_{index:06d}"
                previous = seen_images.get(digest)
                if previous is not None:
                    report["cross_fold_duplicate_images"].append([previous, image_id])
                else:
                    seen_images[digest] = image_id

        files = {}
        for key in ("images", "masks", "types"):
            path = Path(item[key])
            files[key] = {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path) if hash_source_files else "SKIPPED_IN_SMOKE_MODE",
            }
        fold_report = {
            "sample_count": expected_n,
            "images_shape": list(images.shape),
            "images_dtype": str(images.dtype),
            "masks_shape": list(masks.shape),
            "masks_dtype": str(masks.dtype),
            "types_shape": list(tissues.shape),
            "types_dtype": str(tissues.dtype),
            "raw_tissues": raw_tissues,
            "normalized_tissues": normalized_tissues,
            "checked_samples": len(_sample_indices(expected_n, sample_limit)),
            "image_range": [image_min, image_max],
            "image_nonfinite_values": image_nonfinite,
            "image_nonintegral_values": image_nonintegral,
            "mask_nonfinite_values": mask_nonfinite,
            "mask_negative_values": mask_negative,
            "mask_nonintegral_values": mask_nonintegral,
            "multi_class_overlap_pixels": overlap_pixels,
            "background_mismatch_pixels": background_mismatch_pixels,
            "max_instance_id_by_class": max_instance_ids,
            "files": files,
        }
        report["folds"][f"fold{fold}"] = fold_report
        if any((image_nonfinite, mask_nonfinite, mask_negative, mask_nonintegral)):
            raise ValueError(f"Fold {fold} contains invalid image or mask values")
        if image_min < 0 or image_max > 255:
            raise ValueError(f"Fold {fold} images cannot be safely represented as uint8")
        if overlap_pixels:
            report["warnings"].append(
                f"fold{fold}: preserved {overlap_pixels} cross-class overlap pixels; no automatic repair"
            )
        if background_mismatch_pixels:
            report["warnings"].append(
                f"fold{fold}: preserved {background_mismatch_pixels} background-channel mismatches; "
                "background remains excluded from model targets"
            )

    if total not in EXPECTED_TOTAL_COUNTS:
        raise ValueError(f"Unexpected PanNuke total sample count: {total}")
    if report["cross_fold_duplicate_images"]:
        report["status"] = "failed"
        raise ValueError(f"Cross-fold duplicate images detected: {report['cross_fold_duplicate_images'][:10]}")
    return report


def write_audit(
    discovery_path: str | Path,
    destination: str | Path,
    *,
    sample_limit: int | None = None,
    hash_source_files: bool = True,
    check_cross_fold_duplicates: bool = True,
) -> dict[str, Any]:
    """Run an audit and persist its complete report."""
    report = audit_pannuke(
        discovery_path,
        sample_limit=sample_limit,
        hash_source_files=hash_source_files,
        check_cross_fold_duplicates=check_cross_fold_duplicates,
    )
    atomic_write_json(destination, report)
    return report
