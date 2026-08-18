"""Streaming validation metrics for checkpoint selection."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from dfc_sam.constants import NUM_CLASSES
from dfc_sam.engine.trainer import move_batch_to_device

from .inference import batch_predictions
from .official_metrics import OfficialMetricAccumulator


def _ground_truth_batch(batch: dict[str, Any]) -> list[np.ndarray]:
    """Reconstruct mutually exclusive official maps from exposed validation masks."""
    flat_classes = batch["target_batch"]["cls"].detach().cpu().long()
    flat_batches = batch["target_batch"]["batch_idx"].detach().cpu().long()
    supervised_flat = batch["supervised_target_indices"].detach().cpu().long()
    if supervised_flat.numel() != flat_classes.numel():
        raise RuntimeError("Validation requires every target mask")
    lookup = torch.full((flat_classes.numel(),), -1, dtype=torch.long)
    lookup[supervised_flat] = torch.arange(supervised_flat.numel())
    masks = batch["supervised_masks"].detach().cpu().bool()
    results = []
    for batch_index in range(len(batch["image_ids"])):
        official = np.zeros((256, 256, NUM_CLASSES), dtype=np.uint32)
        flat_indices = (flat_batches == batch_index).nonzero(as_tuple=False).flatten()
        next_ids = [1] * NUM_CLASSES
        for flat_index_tensor in flat_indices:
            flat_index = int(flat_index_tensor)
            mask_index = int(lookup[flat_index])
            if mask_index < 0:
                raise RuntimeError("Validation target unexpectedly lacks a mask")
            class_id = int(flat_classes[flat_index])
            mask = masks[mask_index].numpy()
            official[..., class_id][mask] = next_ids[class_id]
            next_ids[class_id] += 1
        results.append(official)
    return results


def validate_student(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    stage: str,
    config: dict[str, Any],
    device: torch.device,
    accumulator: OfficialMetricAccumulator,
) -> dict[str, Any]:
    """Evaluate one manifest-ordered validation fold without writing giant arrays."""
    model.eval()
    with torch.inference_mode():
        for raw_batch in loader:
            ground_truth = _ground_truth_batch(raw_batch)
            batch = move_batch_to_device(raw_batch, device)
            output = model(
                batch["yolo_images"],
                batch["sam_resized_images"],
                stage=stage,
                pre_threshold=float(config["inference"]["pre_threshold"]),
                max_instances=int(config["inference"]["max_instances"]),
            )
            predictions = batch_predictions(
                output,
                batch_size=len(batch["image_ids"]),
                mask_threshold=float(config["inference"]["mask_threshold"]),
                final_score_threshold=float(config["inference"]["final_score_threshold"]),
                logit_blend=float(config["inference"].get("logit_blend", 1.0)),
                quality_power=float(config["inference"].get("quality_power", 1.0)),
                quality_powers_by_class=config["inference"].get("quality_powers_by_class"),
                final_score_thresholds_by_class=config["inference"].get(
                    "final_score_thresholds_by_class"
                ),
                sam_iou_power=float(config["inference"].get("sam_iou_power", 0.0)),
                mask_stability_power=float(config["inference"].get("mask_stability_power", 0.0)),
                mask_stability_delta=float(config["inference"].get("mask_stability_delta", 0.05)),
            )
            for image_id, tissue, truth, prediction in zip(
                batch["image_ids"],
                batch["tissues"],
                ground_truth,
                predictions,
                strict=True,
            ):
                accumulator.update(
                    truth,
                    prediction["official_map"],
                    tissue=str(tissue),
                    image_id=str(image_id),
                )
    return accumulator.compute()
