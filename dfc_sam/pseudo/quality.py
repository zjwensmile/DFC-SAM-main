"""QWPM candidate quality and fixed continuous supervision weights."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class CandidateQuality:
    predicted_iou: Tensor
    stability: Tensor
    box_consistency: Tensor
    total: Tensor


def binary_iou(first: Tensor, second: Tensor, epsilon: float = 1.0e-6) -> Tensor:
    """Compute IoU along the final two mask dimensions."""
    first = first.bool()
    second = second.bool()
    intersection = (first & second).sum(dim=(-2, -1)).float()
    union = (first | second).sum(dim=(-2, -1)).float()
    return torch.where(union > 0, intersection / (union + epsilon), torch.ones_like(union))


def masks_to_boxes(masks: Tensor) -> tuple[Tensor, Tensor]:
    """Return half-open xyxy boxes and a non-empty flag for [N,H,W] masks."""
    if masks.ndim != 3:
        raise ValueError(f"Expected masks [N,H,W], got {masks.shape}")
    count, height, width = masks.shape
    binary = masks.bool()
    boxes = torch.zeros(count, 4, dtype=torch.float32, device=masks.device)
    nonempty = binary.flatten(1).any(dim=1)
    for index in nonempty.nonzero(as_tuple=False).flatten():
        ys, xs = binary[index].nonzero(as_tuple=True)
        boxes[index] = torch.stack((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)).float()
    boxes[:, 0::2] /= width
    boxes[:, 1::2] /= height
    return boxes, nonempty


def box_iou_aligned(first: Tensor, second: Tensor, epsilon: float = 1.0e-6) -> Tensor:
    """Compute aligned IoU for two [N,4] half-open xyxy box tensors."""
    top_left = torch.maximum(first[:, :2], second[:, :2])
    bottom_right = torch.minimum(first[:, 2:], second[:, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=-1)
    first_area = (first[:, 2:] - first[:, :2]).clamp_min(0).prod(dim=-1)
    second_area = (second[:, 2:] - second[:, :2]).clamp_min(0).prod(dim=-1)
    union = first_area + second_area - intersection
    return intersection / (union + epsilon)


def candidate_quality(
    logits: Tensor,
    predicted_iou: Tensor,
    gt_boxes_normalized: Tensor,
    *,
    mask_threshold: float,
    stability_delta: float,
) -> CandidateQuality:
    """Score N candidates using predicted IoU, threshold stability, and GT-box consistency."""
    if logits.ndim != 3:
        raise ValueError(f"Expected logits [N,H,W], got {logits.shape}")
    probability = logits.sigmoid()
    high = probability > mask_threshold + stability_delta
    low = probability > mask_threshold - stability_delta
    stability = binary_iou(high, low)
    binary = probability > mask_threshold
    predicted_boxes, nonempty = masks_to_boxes(binary)
    consistency = box_iou_aligned(predicted_boxes, gt_boxes_normalized)
    consistency = torch.where(nonempty, consistency, torch.zeros_like(consistency))
    predicted = predicted_iou.clamp(0, 1)
    total = (predicted * stability * consistency).clamp_min(0).pow(1.0 / 3.0)
    return CandidateQuality(predicted, stability, consistency, total)


def select_best_candidate(quality: CandidateQuality, candidate_count: int = 3) -> Tensor:
    """Return the best candidate index for each instance from flattened [N*K] scores."""
    if quality.total.numel() % candidate_count:
        raise ValueError("Candidate quality length must be divisible by candidate_count")
    return quality.total.view(-1, candidate_count).argmax(dim=1)


def pseudo_mask_weight(quality: Tensor, threshold: float) -> Tensor:
    """Apply the paper-specified thresholded continuous QWPM weight."""
    if not 0 <= threshold < 1:
        raise ValueError("quality threshold must be in [0,1)")
    quality = quality.clamp(0, 1)
    return torch.where(quality >= threshold, (quality - threshold) / (1 - threshold), torch.zeros_like(quality))
