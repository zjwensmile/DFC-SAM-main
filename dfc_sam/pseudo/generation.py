"""Deterministic selection of one fixed pseudo mask from SAM Teacher candidates."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from dfc_sam.data.geometry import GeometryRecord, normalize_boxes_xyxy
from dfc_sam.models.teacher_adapter import TeacherOutput

from .quality import candidate_quality, pseudo_mask_weight, select_best_candidate


@dataclass
class SelectedPseudoMasks:
    masks: Tensor
    quality: Tensor
    weights: Tensor
    candidate_indices: Tensor


def select_teacher_pseudo_masks(
    output: TeacherOutput,
    boxes_original_xyxy: Tensor,
    geometries: list[GeometryRecord],
    *,
    mask_threshold: float,
    stability_delta: float,
    quality_threshold: float,
) -> SelectedPseudoMasks:
    """Choose candidates and QWPM weights without consulting any hidden mask."""
    logits = output.mask_logits
    predicted_iou = output.iou_prediction
    if logits.ndim != 4 or predicted_iou.shape != logits.shape[:2]:
        raise ValueError("Teacher candidates must have shapes [N,K,H,W] and [N,K]")
    if boxes_original_xyxy.ndim != 2 or boxes_original_xyxy.shape[1] != 4:
        raise ValueError("Teacher target boxes must have shape [T,4]")
    if output.target_flat_index.shape != (logits.shape[0],):
        raise ValueError("Teacher flat target identities do not align with candidates")
    candidate_count = logits.shape[1]
    if candidate_count < 1:
        raise ValueError("Teacher must emit at least one mask candidate")

    normalized_boxes = []
    ordered_boxes = boxes_original_xyxy.index_select(0, output.target_flat_index.to(boxes_original_xyxy.device))
    for box, batch_index in zip(ordered_boxes, output.target_batch_index, strict=True):
        normalized_boxes.append(
            normalize_boxes_xyxy(box.unsqueeze(0), geometries[int(batch_index)].original_hw).squeeze(0)
        )
    boxes = torch.stack(normalized_boxes).to(device=logits.device, dtype=logits.dtype)
    flat_logits = logits.flatten(0, 1)
    flat_iou = predicted_iou.flatten()
    repeated_boxes = boxes.repeat_interleave(candidate_count, dim=0)
    quality = candidate_quality(
        flat_logits,
        flat_iou,
        repeated_boxes,
        mask_threshold=mask_threshold,
        stability_delta=stability_delta,
    )
    selected = select_best_candidate(quality, candidate_count=candidate_count)
    row = torch.arange(logits.shape[0], device=logits.device)
    selected_logits = logits[row, selected]
    selected_quality = quality.total.view(-1, candidate_count)[row, selected]
    masks = selected_logits.sigmoid() > mask_threshold
    weights = pseudo_mask_weight(selected_quality, quality_threshold)
    return SelectedPseudoMasks(masks, selected_quality, weights, selected)
