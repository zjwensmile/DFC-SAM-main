"""Deterministic geometric TTA utilities for instance-mask candidate fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from torch import Tensor

from dfc_sam.constants import NUM_CLASSES

TTA_VIEWS = ("identity", "hflip", "vflip", "rot180")


def transform_spatial(tensor: Tensor, view: str) -> Tensor:
    """Apply a self-inverse square-image transform to trailing spatial dimensions."""
    if view == "identity":
        return tensor
    if view == "hflip":
        return tensor.flip(-1)
    if view == "vflip":
        return tensor.flip(-2)
    if view == "rot180":
        return tensor.flip((-2, -1))
    raise ValueError(f"Unknown TTA view: {view}")


@dataclass(frozen=True)
class TTAFusionSettings:
    match_iou: float
    min_votes: int
    mask_fusion: str
    class_fusion: str
    class_matching: str
    pre_threshold: float
    final_threshold_shift: float
    mask_threshold: float
    view_count: int = 4
    match_mask_threshold: float = 0.5
    min_votes_by_class: list[int] | tuple[int, ...] | None = None
    final_threshold_shifts_by_class: list[float] | tuple[float, ...] | None = None

    def validate(self) -> None:
        if self.view_count not in {1, 4}:
            raise ValueError("TTA view_count must be 1 or 4")
        if not 1 <= self.min_votes <= self.view_count:
            raise ValueError("TTA min_votes must be in [1,view_count]")
        if not 0.0 <= self.match_iou <= 1.0:
            raise ValueError("TTA match_iou must be in [0,1]")
        if not 0.0 <= self.match_mask_threshold <= 1.0:
            raise ValueError("TTA match_mask_threshold must be in [0,1]")
        if not 0.0 <= self.pre_threshold <= 1.0 or not 0.0 <= self.mask_threshold <= 1.0:
            raise ValueError("TTA thresholds must be in [0,1]")
        if not -0.5 <= self.final_threshold_shift <= 0.5:
            raise ValueError("TTA final_threshold_shift must be in [-0.5,0.5]")
        if self.mask_fusion not in {"mean", "max"}:
            raise ValueError("TTA mask_fusion must be mean or max")
        if self.class_fusion not in {"mean", "score_weighted"}:
            raise ValueError("TTA class_fusion must be mean or score_weighted")
        if self.class_matching not in {"agnostic", "same"}:
            raise ValueError("TTA class_matching must be agnostic or same")
        if self.min_votes_by_class is not None:
            if len(self.min_votes_by_class) != NUM_CLASSES or any(
                not 1 <= int(value) <= self.view_count for value in self.min_votes_by_class
            ):
                raise ValueError("TTA min_votes_by_class must contain valid votes for every class")
        if self.final_threshold_shifts_by_class is not None:
            if len(self.final_threshold_shifts_by_class) != NUM_CLASSES or any(
                not -0.5 <= float(value) <= 0.5 for value in self.final_threshold_shifts_by_class
            ):
                raise ValueError("TTA final_threshold_shifts_by_class must contain five values in [-0.5,0.5]")


def _class_thresholds(
    settings: TTAFusionSettings,
    base_final_thresholds: list[float] | tuple[float, ...],
) -> np.ndarray:
    shifts = np.full(NUM_CLASSES, settings.final_threshold_shift, dtype=np.float32)
    if settings.final_threshold_shifts_by_class is not None:
        shifts += np.asarray(settings.final_threshold_shifts_by_class, dtype=np.float32)
    return np.clip(np.asarray(base_final_thresholds, dtype=np.float32) + shifts, 0.0, 1.0)


def filter_view_candidates(
    view: dict[str, np.ndarray],
    *,
    pre_threshold: float,
) -> list[dict[str, Any]]:
    """Apply only the detector prefilter before cross-view score fusion."""
    masks = np.asarray(view["mask_probabilities"], dtype=np.float32)
    probabilities = np.asarray(view["class_probabilities"], dtype=np.float32)
    base_scores = np.asarray(view["base_scores"], dtype=np.float32)
    if masks.ndim != 3 or probabilities.shape != (masks.shape[0], NUM_CLASSES):
        raise ValueError("TTA view candidate arrays are misaligned")
    if base_scores.shape != (masks.shape[0],):
        raise ValueError("TTA base scores are misaligned")
    classes = probabilities.argmax(axis=1)
    scores = probabilities.max(axis=1)
    retained = base_scores >= float(pre_threshold)
    return [
        {
            "mask": masks[index],
            "probabilities": probabilities[index],
            "class_id": int(classes[index]),
            "score": float(scores[index]),
        }
        for index in np.flatnonzero(retained)
    ]


def _cluster_mask(cluster: list[dict[str, Any]]) -> np.ndarray:
    return np.mean(np.stack([member["mask"] for member in cluster]), axis=0)


def _append_view(
    clusters: list[list[dict[str, Any]]],
    candidates: list[dict[str, Any]],
    settings: TTAFusionSettings,
) -> None:
    if not candidates:
        return
    if not clusters:
        clusters.extend([[candidate] for candidate in candidates])
        return
    prototypes = [_cluster_mask(cluster) for cluster in clusters]
    cluster_classes = [
        int(np.mean(np.stack([member["probabilities"] for member in cluster]), axis=0).argmax())
        for cluster in clusters
    ]
    # A 4x strided binary representation is exact enough for correspondence
    # while avoiding thousands of full 256x256 Python-level IoU operations.
    candidate_binary = np.stack(
        [candidate["mask"][::4, ::4] >= settings.match_mask_threshold for candidate in candidates]
    )
    prototype_binary = np.stack(
        [prototype[::4, ::4] >= settings.match_mask_threshold for prototype in prototypes]
    )
    intersection = np.logical_and(candidate_binary[:, None], prototype_binary[None, :]).sum(axis=(-2, -1))
    union = np.logical_or(candidate_binary[:, None], prototype_binary[None, :]).sum(axis=(-2, -1))
    ious = np.divide(intersection, union, out=np.ones_like(intersection, dtype=np.float64), where=union > 0)
    if settings.class_matching == "same":
        candidate_classes = np.asarray([candidate["class_id"] for candidate in candidates])
        compatible = candidate_classes[:, None] == np.asarray(cluster_classes)[None, :]
        ious = np.where(compatible, ious, -1.0)
    eligible_pairs = np.argwhere(ious >= settings.match_iou)
    pairs = sorted(
        ((-float(ious[candidate_index, cluster_index]), int(candidate_index), int(cluster_index))
         for candidate_index, cluster_index in eligible_pairs),
    )
    used_candidates: set[int] = set()
    used_clusters: set[int] = set()
    for _, candidate_index, cluster_index in pairs:
        if candidate_index in used_candidates or cluster_index in used_clusters:
            continue
        clusters[cluster_index].append(candidates[candidate_index])
        used_candidates.add(candidate_index)
        used_clusters.add(cluster_index)
    clusters.extend(
        [candidate] for index, candidate in enumerate(candidates) if index not in used_candidates
    )


def fuse_tta_views(
    views: list[dict[str, np.ndarray]],
    settings: TTAFusionSettings,
    *,
    base_final_thresholds: list[float] | tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Match candidates one-to-one across views and fuse retained consensus clusters."""
    settings.validate()
    if len(views) < settings.view_count:
        raise ValueError("Insufficient TTA views")
    filtered = [
        filter_view_candidates(
            view,
            pre_threshold=settings.pre_threshold,
        )
        for view in views[: settings.view_count]
    ]
    if settings.view_count == 1:
        candidates = filtered[0]
        thresholds = _class_thresholds(settings, base_final_thresholds)
        candidates = [
            candidate
            for candidate in candidates
            if candidate["score"] >= thresholds[candidate["class_id"]]
        ]
        masks = np.stack([candidate["mask"] for candidate in candidates]) if candidates else np.zeros(
            (0, 256, 256), dtype=np.float32
        )
        probabilities = (
            np.stack([candidate["probabilities"] for candidate in candidates])
            if candidates
            else np.zeros((0, NUM_CLASSES), dtype=np.float32)
        )
        return masks, probabilities, {
            "view_candidates": len(candidates),
            "clusters": len(candidates),
            "retained": len(candidates),
        }

    clusters: list[list[dict[str, Any]]] = []
    for candidates in filtered:
        _append_view(clusters, candidates, settings)
    retained_clusters = [cluster for cluster in clusters if len(cluster) >= 1]
    fused_masks = []
    fused_probabilities = []
    cluster_votes = []
    for cluster in retained_clusters:
        masks = np.stack([member["mask"] for member in cluster])
        probabilities = np.stack([member["probabilities"] for member in cluster])
        fused_masks.append(masks.mean(axis=0) if settings.mask_fusion == "mean" else masks.max(axis=0))
        if settings.class_fusion == "mean":
            fused_probabilities.append(probabilities.mean(axis=0))
        else:
            weights = np.asarray([member["score"] for member in cluster], dtype=np.float32)
            fused_probabilities.append(
                (probabilities * weights[:, None]).sum(axis=0) / max(float(weights.sum()), 1.0e-12)
            )
        cluster_votes.append(len(cluster))
    mask_shape = views[0]["mask_probabilities"].shape[-2:]
    masks_array = (
        np.stack(fused_masks).astype(np.float32, copy=False)
        if fused_masks
        else np.zeros((0, *mask_shape), dtype=np.float32)
    )
    probabilities_array = (
        np.stack(fused_probabilities).astype(np.float32, copy=False)
        if fused_probabilities
        else np.zeros((0, NUM_CLASSES), dtype=np.float32)
    )
    if probabilities_array.shape[0]:
        thresholds = _class_thresholds(settings, base_final_thresholds)
        classes = probabilities_array.argmax(axis=1)
        scores = probabilities_array.max(axis=1)
        required_votes = np.full(NUM_CLASSES, settings.min_votes, dtype=np.int64)
        if settings.min_votes_by_class is not None:
            required_votes = np.asarray(settings.min_votes_by_class, dtype=np.int64)
        retained = (scores >= thresholds[classes]) & (
            np.asarray(cluster_votes, dtype=np.int64) >= required_votes[classes]
        )
        masks_array = masks_array[retained]
        probabilities_array = probabilities_array[retained]
    return masks_array, probabilities_array, {
        "view_candidates": sum(len(candidates) for candidates in filtered),
        "clusters": len(clusters),
        "retained": int(probabilities_array.shape[0]),
    }
