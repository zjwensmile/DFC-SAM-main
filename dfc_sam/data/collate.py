"""Batch PanNuke samples without exposing box-only ground-truth masks."""

from __future__ import annotations

from typing import Any

import torch


def collate_pannuke(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Build Ultralytics targets plus a separately indexed supervised-mask tensor."""
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    yolo_images = torch.stack([sample["yolo_image"] for sample in samples])
    sam_shapes = {tuple(sample["sam_resized_image"].shape) for sample in samples}
    if len(sam_shapes) != 1:
        raise ValueError("SAM-resized images must share a shape before batching")
    sam_images = torch.stack([sample["sam_resized_image"] for sample in samples])

    batch_indices = []
    classes = []
    boxes = []
    boxes_original = []
    mask_supervised = []
    supervised_target_indices = []
    supervised_masks = []
    target_image_ids = []
    target_instance_indices = []
    flat_offset = 0
    for batch_index, sample in enumerate(samples):
        target = sample["target"]
        if sample["supervision"] == "box_only" and sample.get("train_mask") is not None:
            raise ValueError(f"Box-only sample exposed a train mask: {sample['image_id']}")
        count = int(target["cls"].shape[0])
        batch_indices.append(torch.full((count,), batch_index, dtype=torch.long))
        classes.append(target["cls"])
        boxes.append(target["bboxes"])
        boxes_original.append(target["boxes_original_xyxy"])
        mask_supervised.append(target["mask_supervised"])
        target_image_ids.extend([sample["image_id"]] * count)
        target_instance_indices.append(torch.arange(count, dtype=torch.long))
        if "masks" in target:
            supervised_target_indices.append(torch.arange(flat_offset, flat_offset + count, dtype=torch.long))
            supervised_masks.append(target["masks"])
        elif bool(target["mask_supervised"].any()):
            raise ValueError("A mask-supervised sample is missing its masks")
        flat_offset += count

    def concatenate(parts: list[torch.Tensor], empty_shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        return torch.cat(parts) if parts else torch.empty(empty_shape, dtype=dtype)

    return {
        "image_ids": [sample["image_id"] for sample in samples],
        "tissues": [sample["tissue"] for sample in samples],
        "geometries": [sample["geometry"] for sample in samples],
        "yolo_images": yolo_images,
        "sam_resized_images": sam_images,
        "target_batch": {
            "batch_idx": concatenate(batch_indices, (0,), torch.long),
            "cls": concatenate(classes, (0,), torch.long),
            "bboxes": concatenate(boxes, (0, 4), torch.float32),
        },
        "target_boxes_original_xyxy": concatenate(boxes_original, (0, 4), torch.float32),
        "target_image_ids": target_image_ids,
        "target_instance_indices": concatenate(target_instance_indices, (0,), torch.long),
        "mask_supervised": concatenate(mask_supervised, (0,), torch.bool),
        "supervised_target_indices": concatenate(supervised_target_indices, (0,), torch.long),
        "supervised_masks": concatenate(supervised_masks, (0, 256, 256), torch.bool),
    }
