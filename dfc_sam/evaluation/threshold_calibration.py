"""Validation-only inference-threshold calibration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .overlap_resolution import resolve_pannuke_instances


@dataclass(frozen=True)
class InferenceThresholds:
    """One NMS-free inference operating point."""

    pre_threshold: float
    final_score_threshold: float
    mask_threshold: float

    def __post_init__(self) -> None:
        for name, value in (
            ("pre_threshold", self.pre_threshold),
            ("final_score_threshold", self.final_score_threshold),
            ("mask_threshold", self.mask_threshold),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0,1], got {value}")

    @property
    def candidate_id(self) -> str:
        def component(value: float) -> str:
            return f"{value:.4f}".replace(".", "p")

        return (
            f"pre_{component(self.pre_threshold)}"
            f"__final_{component(self.final_score_threshold)}"
            f"__mask_{component(self.mask_threshold)}"
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "pre_threshold": float(self.pre_threshold),
            "final_score_threshold": float(self.final_score_threshold),
            "mask_threshold": float(self.mask_threshold),
        }


def resolve_threshold_candidate(
    mask_probabilities: np.ndarray,
    class_probabilities: np.ndarray,
    base_scores: np.ndarray,
    thresholds: InferenceThresholds,
) -> tuple[np.ndarray, int]:
    """Filter a low-pre-threshold forward pass at one exact operating point."""
    instance_count = int(mask_probabilities.shape[0])
    if mask_probabilities.ndim != 3:
        raise ValueError("mask_probabilities must have shape [K,H,W]")
    if class_probabilities.shape[0] != instance_count or base_scores.shape != (instance_count,):
        raise ValueError("Candidate probability arrays do not align")
    refined_scores = class_probabilities.max(axis=-1)
    retained = (base_scores >= thresholds.pre_threshold) & (
        refined_scores >= thresholds.final_score_threshold
    )
    prediction = resolve_pannuke_instances(
        mask_probabilities[retained],
        class_probabilities[retained],
        mask_threshold=thresholds.mask_threshold,
    )
    return prediction, int(retained.sum())


def selection_key(result: dict[str, Any]) -> tuple[float, ...]:
    """Rank candidates by the preregistered validation-only metric hierarchy."""
    metrics = result["metrics"]
    thresholds = result["thresholds"]
    mask_threshold = float(thresholds["mask_threshold"])
    return (
        float(metrics["mpq"]),
        float(metrics["bpq"]),
        float(metrics["f1det"]),
        float(metrics["macro_f1"]),
        float(thresholds["pre_threshold"]),
        float(thresholds["final_score_threshold"]),
        -abs(mask_threshold - 0.5),
        mask_threshold,
    )


def select_best_candidate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Select one candidate without consulting test-fold outcomes."""
    if not results:
        raise ValueError("Threshold calibration requires at least one candidate")
    return max(results, key=selection_key)
