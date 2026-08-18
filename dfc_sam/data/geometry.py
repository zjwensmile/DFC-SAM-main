"""Shared original/YOLO/SAM geometry records and differentiable coordinate transforms."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class GeometryRecord:
    original_hw: tuple[int, int]
    augmented_hw: tuple[int, int]
    yolo_input_hw: tuple[int, int]
    yolo_scale_xy: tuple[float, float]
    yolo_pad_xy: tuple[float, float]
    sam_input_hw: tuple[int, int]
    sam_resized_hw: tuple[int, int]
    sam_pad_xy: tuple[int, int]


def transform_boxes_xyxy(boxes: Tensor, scale_xy: tuple[float, float], pad_xy: tuple[float, float]) -> Tensor:
    """Apply an affine resize/pad transform to continuous half-open xyxy boxes."""
    scale = boxes.new_tensor([scale_xy[0], scale_xy[1], scale_xy[0], scale_xy[1]])
    pad = boxes.new_tensor([pad_xy[0], pad_xy[1], pad_xy[0], pad_xy[1]])
    return boxes * scale + pad


def inverse_transform_boxes_xyxy(boxes: Tensor, scale_xy: tuple[float, float], pad_xy: tuple[float, float]) -> Tensor:
    """Invert :func:`transform_boxes_xyxy` without integer rounding."""
    scale = boxes.new_tensor([scale_xy[0], scale_xy[1], scale_xy[0], scale_xy[1]])
    pad = boxes.new_tensor([pad_xy[0], pad_xy[1], pad_xy[0], pad_xy[1]])
    return (boxes - pad) / scale


def normalize_boxes_xyxy(boxes: Tensor, image_hw: tuple[int, int]) -> Tensor:
    """Map pixel-space xyxy boxes to [0, 1] coordinates."""
    height, width = image_hw
    divisor = boxes.new_tensor([width, height, width, height])
    return boxes / divisor


def denormalize_boxes_xyxy(boxes: Tensor, image_hw: tuple[int, int]) -> Tensor:
    """Map normalized xyxy boxes back to continuous pixel coordinates."""
    height, width = image_hw
    multiplier = boxes.new_tensor([width, height, width, height])
    return boxes * multiplier


def boxes_original_to_sam_xyxy(boxes: Tensor, geometry: GeometryRecord) -> Tensor:
    """Map original-image boxes into SAM's resized-and-padded prompt coordinates."""
    original_h, original_w = geometry.original_hw
    resized_h, resized_w = geometry.sam_resized_hw
    return transform_boxes_xyxy(
        boxes,
        (resized_w / original_w, resized_h / original_h),
        (float(geometry.sam_pad_xy[0]), float(geometry.sam_pad_xy[1])),
    )


def build_box_grid(boxes_normalized: Tensor, output_hw: tuple[int, int]) -> Tensor:
    """Build an align_corners=False grid for bilinear sampling inside each box."""
    if boxes_normalized.ndim != 2 or boxes_normalized.shape[1] != 4:
        raise ValueError(f"Expected boxes [N,4], got {tuple(boxes_normalized.shape)}")
    output_h, output_w = output_hw
    boxes = boxes_normalized.clamp(0.0, 1.0)
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    x2 = torch.maximum(x2, x1)
    y2 = torch.maximum(y2, y1)
    x_fraction = (torch.arange(output_w, device=boxes.device, dtype=boxes.dtype) + 0.5) / output_w
    y_fraction = (torch.arange(output_h, device=boxes.device, dtype=boxes.dtype) + 0.5) / output_h
    sample_x = x1[:, None] + (x2 - x1)[:, None] * x_fraction[None]
    sample_y = y1[:, None] + (y2 - y1)[:, None] * y_fraction[None]
    grid_x = sample_x[:, None, :].expand(-1, output_h, -1)
    grid_y = sample_y[:, :, None].expand(-1, -1, output_w)
    return torch.stack((grid_x * 2 - 1, grid_y * 2 - 1), dim=-1)
