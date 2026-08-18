"""Pixel competition for official mutually assigned PanNuke evaluation maps."""

from __future__ import annotations

import numpy as np

from dfc_sam.constants import NUM_CLASSES


def resolve_pannuke_instances(
    mask_probabilities: np.ndarray,
    class_probabilities: np.ndarray,
    *,
    mask_threshold: float,
) -> np.ndarray:
    """Resolve only overlapping pixels; never suppress complete instances."""
    if mask_probabilities.ndim != 3:
        raise ValueError("mask_probabilities must have shape [K,H,W]")
    instance_count, height, width = mask_probabilities.shape
    if class_probabilities.shape != (instance_count, NUM_CLASSES):
        raise ValueError(f"class_probabilities must have shape {(instance_count, NUM_CLASSES)}")
    if not 0 <= mask_threshold <= 1:
        raise ValueError("mask_threshold must be in [0,1]")

    result = np.zeros((height, width, NUM_CLASSES), dtype=np.uint32)
    if instance_count == 0:
        return result
    class_ids = class_probabilities.argmax(axis=1)
    class_scores = class_probabilities.max(axis=1)
    response = mask_probabilities * class_scores[:, None, None]
    eligible = mask_probabilities >= mask_threshold
    winning_scores = np.where(eligible, response, -np.inf)
    winner = winning_scores.argmax(axis=0)
    has_winner = eligible.any(axis=0)

    next_id = np.ones(NUM_CLASSES, dtype=np.uint32)
    for instance_index in range(instance_count):
        assigned = has_winner & (winner == instance_index)
        if not assigned.any():
            continue
        class_id = int(class_ids[instance_index])
        result[..., class_id][assigned] = next_id[class_id]
        next_id[class_id] += 1
    return result
