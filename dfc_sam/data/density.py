"""Instance-density ranking without image preprocessing or full-array loading."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from dfc_sam.constants import NUM_CLASSES


def count_instances_in_raw_mask(raw_mask: np.ndarray) -> tuple[int, tuple[int, ...]]:
    """Count positive instance IDs in each of the five PanNuke class channels."""
    if raw_mask.shape != (256, 256, NUM_CLASSES + 1):
        raise ValueError(f"Expected PanNuke mask [256,256,{NUM_CLASSES + 1}], got {raw_mask.shape}")
    counts = tuple(
        int(np.count_nonzero(np.unique(raw_mask[..., class_id])))
        for class_id in range(NUM_CLASSES)
    )
    return sum(counts), counts


def rank_manifest_records_by_density(
    records: Sequence[dict[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    """Return the densest records with indices compatible with PanNukeDataset."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    arrays: dict[str, np.ndarray] = {}
    ranking = []
    for dataset_index, record in enumerate(records):
        path = str(record["masks_npy"])
        if path not in arrays:
            arrays[path] = np.load(path, mmap_mode="r")
        raw_mask = np.asarray(arrays[path][int(record["raw_index"])])
        total, per_class = count_instances_in_raw_mask(raw_mask)
        ranking.append(
            {
                "dataset_index": dataset_index,
                "image_id": str(record["image_id"]),
                "fold": int(record["fold"]),
                "tissue": str(record["tissue"]),
                "instance_count": total,
                "per_class_instance_count": list(per_class),
            }
        )
    ranking.sort(key=lambda item: (-item["instance_count"], item["image_id"]))
    return ranking[:top_k]
