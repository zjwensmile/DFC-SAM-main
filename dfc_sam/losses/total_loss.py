"""Explicit DFC-SAM loss composition."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class LossWeights:
    detection: float
    segmentation: float
    ugca: float


def total_loss(detection: Tensor, segmentation: Tensor, ugca: Tensor, weights: LossWeights) -> Tensor:
    """Combine losses without hidden coefficients."""
    if any(loss.numel() != 1 for loss in (detection, segmentation, ugca)):
        raise ValueError("Detection, segmentation, and UGCA objectives must each be scalar")
    return weights.detection * detection + weights.segmentation * segmentation + weights.ugca * ugca
