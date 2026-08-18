"""Training-fold-only QWPM calibration and validation-only inference helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor

from dfc_sam.constants import PANNUKE_CLASSES


def quality_calibration_curve(
    quality: Tensor,
    dice: Tensor,
    class_ids: Tensor,
    tissues: Sequence[str],
    thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    """Report threshold/coverage/Dice without selecting a threshold from val/test masks."""
    count = int(quality.numel())
    if dice.shape != quality.shape or class_ids.shape != quality.shape or len(tissues) != count:
        raise ValueError("Calibration quality, Dice, class, and tissue identities must align")
    if not torch.isfinite(quality).all() or not torch.isfinite(dice).all():
        raise ValueError("Calibration quality and Dice must be finite")
    if ((quality < 0) | (quality > 1) | (dice < 0) | (dice > 1)).any():
        raise ValueError("Calibration quality and Dice must be in [0,1]")
    if ((class_ids < 0) | (class_ids >= len(PANNUKE_CLASSES))).any():
        raise ValueError("Calibration class IDs are outside PanNuke's five classes")

    rows = []
    tissue_names = sorted(set(tissues))
    for threshold in sorted({float(value) for value in thresholds}):
        if not 0 <= threshold < 1:
            raise ValueError("Quality thresholds must be in [0,1)")
        retained = quality >= threshold
        retained_count = int(retained.sum())
        row: dict[str, Any] = {
            "threshold": threshold,
            "retained_instances": retained_count,
            "coverage": retained_count / count if count else 0.0,
            "mean_dice": float(dice[retained].mean()) if retained_count else 0.0,
            "class_coverage": {},
            "tissue_coverage": {},
        }
        for class_id, class_name in enumerate(PANNUKE_CLASSES):
            member = class_ids == class_id
            denominator = int(member.sum())
            row["class_coverage"][class_name] = (
                int((retained & member).sum()) / denominator if denominator else None
            )
        for tissue in tissue_names:
            member = torch.tensor([value == tissue for value in tissues], device=retained.device)
            denominator = int(member.sum())
            row["tissue_coverage"][tissue] = int((retained & member).sum()) / denominator
        rows.append(row)
    return rows


def select_validation_threshold(
    candidates: Sequence[float],
    evaluate: Callable[[float], float],
) -> tuple[float, dict[float, float]]:
    """Select the highest-scoring threshold with a deterministic lower-value tie break."""
    if not candidates:
        raise ValueError("At least one threshold candidate is required")
    ordered = sorted({float(value) for value in candidates})
    if any(value < 0.0 or value > 1.0 for value in ordered):
        raise ValueError("Threshold candidates must be in [0,1]")
    scores = {value: float(evaluate(value)) for value in ordered}
    if any(score != score for score in scores.values()):
        raise ValueError("Calibration scores must not be NaN")
    selected = max(ordered, key=lambda value: (scores[value], -value))
    return selected, scores
