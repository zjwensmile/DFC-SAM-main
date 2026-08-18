"""Shared deterministic YOLO/SAM image geometry for PanNuke."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from .geometry import GeometryRecord, transform_boxes_xyxy


@dataclass(frozen=True)
class PreparedImage:
    yolo_image: Tensor
    sam_resized_image: Tensor
    geometry: GeometryRecord


def _resize(image: Tensor, output_hw: tuple[int, int], *, antialias: bool = True) -> Tensor:
    return F.interpolate(
        image.unsqueeze(0).float(),
        size=output_hw,
        mode="bilinear",
        align_corners=False,
        antialias=antialias,
    ).squeeze(0)


def prepare_yolo_sam_inputs(
    image_chw: Tensor,
    *,
    yolo_hw: tuple[int, int] = (640, 640),
    sam_long_side: int = 1024,
    yolo_pad_value: float = 114.0,
    yolo_resize_antialias: bool = True,
) -> PreparedImage:
    """Create the two model inputs and one authoritative geometry record."""
    if image_chw.ndim != 3 or image_chw.shape[0] != 3:
        raise ValueError(f"Expected RGB image [3,H,W], got {tuple(image_chw.shape)}")
    original_h, original_w = (int(value) for value in image_chw.shape[-2:])

    yolo_h, yolo_w = yolo_hw
    yolo_scale = min(yolo_h / original_h, yolo_w / original_w)
    resized_h = min(yolo_h, max(1, round(original_h * yolo_scale)))
    resized_w = min(yolo_w, max(1, round(original_w * yolo_scale)))
    yolo_resized = _resize(
        image_chw,
        (resized_h, resized_w),
        antialias=yolo_resize_antialias,
    )
    pad_y = (yolo_h - resized_h) // 2
    pad_x = (yolo_w - resized_w) // 2
    pad_bottom = yolo_h - resized_h - pad_y
    pad_right = yolo_w - resized_w - pad_x
    yolo_image = F.pad(
        yolo_resized,
        (pad_x, pad_right, pad_y, pad_bottom),
        value=float(yolo_pad_value),
    )
    yolo_image = yolo_image / 255.0

    sam_scale = sam_long_side / max(original_h, original_w)
    sam_h = min(sam_long_side, max(1, round(original_h * sam_scale)))
    sam_w = min(sam_long_side, max(1, round(original_w * sam_scale)))
    sam_resized = _resize(image_chw, (sam_h, sam_w))

    geometry = GeometryRecord(
        original_hw=(original_h, original_w),
        augmented_hw=(original_h, original_w),
        yolo_input_hw=yolo_hw,
        yolo_scale_xy=(resized_w / original_w, resized_h / original_h),
        yolo_pad_xy=(float(pad_x), float(pad_y)),
        sam_input_hw=(sam_long_side, sam_long_side),
        sam_resized_hw=(sam_h, sam_w),
        sam_pad_xy=(0, 0),
    )
    return PreparedImage(yolo_image, sam_resized, geometry)


def build_model_input_transform(config: Mapping[str, Any]):
    """Build detector/SAM geometry from the resolved model configuration."""
    detector = config["detector"]
    sam = config["sam"]
    image_size = int(detector.get("imgsz", 640))
    return partial(
        prepare_yolo_sam_inputs,
        yolo_hw=(image_size, image_size),
        sam_long_side=int(sam.get("input_size", 1024)),
        yolo_resize_antialias=bool(detector.get("resize_antialias", True)),
    )


def boxes_xyxy_to_normalized_xywh(boxes_xyxy: Tensor, image_hw: tuple[int, int]) -> Tensor:
    """Convert half-open xyxy boxes into Ultralytics normalized center-xywh targets."""
    if boxes_xyxy.ndim != 2 or boxes_xyxy.shape[-1] != 4:
        raise ValueError(f"Expected boxes [N,4], got {tuple(boxes_xyxy.shape)}")
    height, width = image_hw
    top_left = boxes_xyxy[:, :2]
    bottom_right = boxes_xyxy[:, 2:]
    center = (top_left + bottom_right) / 2
    size = bottom_right - top_left
    result = torch.cat((center, size), dim=-1)
    divisor = result.new_tensor([width, height, width, height])
    return result / divisor


def boxes_to_yolo_targets(boxes_original_xyxy: Tensor, geometry: GeometryRecord) -> Tensor:
    """Map original-image boxes to normalized YOLO xywh targets."""
    transformed = transform_boxes_xyxy(
        boxes_original_xyxy,
        geometry.yolo_scale_xy,
        geometry.yolo_pad_xy,
    )
    return boxes_xyxy_to_normalized_xywh(transformed, geometry.yolo_input_hw)
