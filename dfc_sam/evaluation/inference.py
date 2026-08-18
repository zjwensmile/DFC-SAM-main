"""Manifest-ordered prediction and PanNuke array construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from dfc_sam.constants import NUM_CLASSES
from dfc_sam.engine.runtime import assert_role_access
from dfc_sam.engine.trainer import move_batch_to_device
from dfc_sam.utils.hashing import atomic_write_json

from .overlap_resolution import resolve_pannuke_instances
from .pannuke_export import export_official_arrays


def resolve_final_score_thresholds(inference: dict[str, Any]) -> list[float]:
    """Expand an optional class threshold vector from its scalar fallback."""
    values = inference.get("final_score_thresholds_by_class")
    if values is None:
        values = [float(inference["final_score_threshold"])] * NUM_CLASSES
    if not isinstance(values, list | tuple) or len(values) != NUM_CLASSES:
        raise ValueError(f"final_score_thresholds_by_class must contain {NUM_CLASSES} values")
    thresholds = [float(value) for value in values]
    if any(not 0.0 <= value <= 1.0 for value in thresholds):
        raise ValueError("final score thresholds must be in [0,1]")
    return thresholds


def quality_scaled_class_probabilities(
    output: Any,
    *,
    logit_blend: float = 1.0,
    quality_power: float = 1.0,
    quality_powers_by_class: list[float] | tuple[float, ...] | None = None,
    sam_iou_power: float = 0.0,
    mask_stability_power: float = 0.0,
    mask_stability_threshold: float = 0.5,
    mask_stability_delta: float = 0.05,
) -> torch.Tensor:
    """Return class distributions scaled by learned candidate quality when available."""
    unit_interval = (logit_blend, quality_power, sam_iou_power, mask_stability_power)
    if any(not 0.0 <= float(value) <= 1.0 for value in unit_interval):
        raise ValueError("score-fusion powers must be in [0,1]")
    if not 0.0 <= float(mask_stability_threshold) <= 1.0:
        raise ValueError("mask_stability_threshold must be in [0,1]")
    if not 0.0 <= float(mask_stability_delta) <= 0.5:
        raise ValueError("mask_stability_delta must be in [0,0.5]")
    refined_logits = output.refined_logits.float()
    raw_base_logits = getattr(output, "base_logits", None)
    if raw_base_logits is None and float(logit_blend) != 1.0:
        raise ValueError("logit_blend < 1 requires output.base_logits")
    base_logits = refined_logits if raw_base_logits is None else raw_base_logits.float()
    logits = base_logits + float(logit_blend) * (refined_logits - base_logits)
    mode = str(getattr(output, "class_probability_mode", "softmax"))
    if mode == "sigmoid":
        probabilities = logits.sigmoid()
    elif mode == "softmax":
        probabilities = logits.softmax(dim=-1)
    else:
        raise ValueError(f"Unknown class_probability_mode: {mode}")
    if output.ugca is not None and output.ugca.quality_score is not None:
        if quality_powers_by_class is not None:
            if len(quality_powers_by_class) != probabilities.shape[-1]:
                raise ValueError("quality_powers_by_class does not match the class dimension")
            powers = probabilities.new_tensor(quality_powers_by_class)
            if bool(((powers < 0.0) | (powers > 1.0)).any()):
                raise ValueError("quality_powers_by_class values must be in [0,1]")
        else:
            powers = probabilities.new_full((probabilities.shape[-1],), float(quality_power))
        quality = output.ugca.quality_score.float().clamp(1.0e-6, 1.0)
        if quality.shape != probabilities.shape[:1]:
            raise ValueError("UGCA quality scores do not align with refined logits")
        probabilities = probabilities * quality[:, None].pow(powers[None, :])

    scalar_quality = probabilities.new_ones(probabilities.shape[0])
    if sam_iou_power > 0.0:
        iou_prediction = getattr(output, "iou_prediction", None)
        if iou_prediction is None or iou_prediction.shape != (probabilities.shape[0], 1):
            raise ValueError("sam_iou_power requires aligned output.iou_prediction [N,1]")
        scalar_quality = scalar_quality * iou_prediction[:, 0].float().clamp(1.0e-6, 1.0).pow(
            float(sam_iou_power)
        )
    if mask_stability_power > 0.0:
        mask_logits = getattr(output, "mask_logits", None)
        if mask_logits is None or mask_logits.shape[:2] != (probabilities.shape[0], 1):
            raise ValueError("mask_stability_power requires aligned output.mask_logits [N,1,H,W]")
        mask_probability = mask_logits[:, 0].float().sigmoid()
        high = mask_probability > min(1.0, mask_stability_threshold + mask_stability_delta)
        low = mask_probability > max(0.0, mask_stability_threshold - mask_stability_delta)
        intersection = (high & low).sum(dim=(-2, -1)).float()
        union = (high | low).sum(dim=(-2, -1)).float()
        stability = torch.where(union > 0, intersection / union.clamp_min(1.0), torch.ones_like(union))
        scalar_quality = scalar_quality * stability.clamp(1.0e-6, 1.0).pow(float(mask_stability_power))
    return probabilities * scalar_quality[:, None]


def load_ground_truth_arrays(records: list[dict[str, Any]]) -> np.ndarray:
    """Load only the five positive-class channels in manifest order."""
    cache: dict[str, np.ndarray] = {}
    arrays = []
    for record in records:
        path = str(record["masks_npy"])
        if path not in cache:
            cache[path] = np.load(path, mmap_mode="r")
        raw = np.asarray(cache[path][int(record["raw_index"])])
        if raw.shape != (256, 256, NUM_CLASSES + 1):
            raise ValueError(f"Invalid PanNuke GT shape for {record['image_id']}: {raw.shape}")
        arrays.append(np.asarray(raw[..., :NUM_CLASSES], dtype=np.uint16))
    return np.stack(arrays) if arrays else np.zeros((0, 256, 256, NUM_CLASSES), dtype=np.uint16)


def batch_predictions(
    output: Any,
    *,
    batch_size: int,
    mask_threshold: float,
    final_score_threshold: float,
    logit_blend: float = 1.0,
    quality_power: float = 1.0,
    quality_powers_by_class: list[float] | tuple[float, ...] | None = None,
    final_score_thresholds_by_class: list[float] | tuple[float, ...] | None = None,
    sam_iou_power: float = 0.0,
    mask_stability_power: float = 0.0,
    mask_stability_delta: float = 0.05,
) -> list[dict[str, np.ndarray]]:
    class_probabilities = quality_scaled_class_probabilities(
        output,
        logit_blend=logit_blend,
        quality_power=quality_power,
        quality_powers_by_class=quality_powers_by_class,
        sam_iou_power=sam_iou_power,
        mask_stability_power=mask_stability_power,
        mask_stability_threshold=mask_threshold,
        mask_stability_delta=mask_stability_delta,
    )
    mask_probabilities = output.mask_logits[:, 0].float().sigmoid()
    base_logits = output.base_logits.float()
    refined_logits = output.refined_logits.float()
    if output.ugca is None:
        if str(getattr(output, "class_probability_mode", "softmax")) == "sigmoid":
            evidence = base_logits.sigmoid()
            probability = evidence / evidence.sum(dim=-1, keepdim=True).clamp_min(
                torch.finfo(evidence.dtype).eps
            )
        else:
            probability = base_logits.softmax(dim=-1)
        entropy = -(probability * probability.clamp_min(torch.finfo(probability.dtype).eps).log()).sum(
            dim=-1
        )
        entropy = entropy / np.log(probability.shape[-1])
        ugca_gate = entropy.new_zeros(entropy.shape)
    else:
        entropy = output.ugca.entropy.float()
        ugca_gate = output.ugca.gate.float()
    results = []
    for batch_index in range(batch_size):
        indices = (output.selected.batch_index == batch_index).nonzero(as_tuple=False).flatten()
        probabilities = class_probabilities.index_select(0, indices)
        masks = mask_probabilities.index_select(0, indices)
        boxes = output.boxes_xyxy_normalized.index_select(0, indices)
        queries = output.selected.query_index.index_select(0, indices)
        selected_base_logits = base_logits.index_select(0, indices)
        selected_refined_logits = refined_logits.index_select(0, indices)
        selected_entropy = entropy.index_select(0, indices)
        selected_gate = ugca_gate.index_select(0, indices)
        selected_scale_weights = output.bridge.scale_weights.index_select(0, indices)
        selected_iou = output.iou_prediction.index_select(0, indices)
        if output.ugca is None or output.ugca.quality_score is None:
            selected_quality = probabilities.new_ones(probabilities.shape[0])
        else:
            selected_quality = output.ugca.quality_score.float().index_select(0, indices)
        scores, predicted_classes = probabilities.max(dim=-1)
        if final_score_thresholds_by_class is None:
            retained = scores >= final_score_threshold
        else:
            if len(final_score_thresholds_by_class) != probabilities.shape[-1]:
                raise ValueError("final_score_thresholds_by_class does not match the class dimension")
            thresholds = probabilities.new_tensor(final_score_thresholds_by_class)
            retained = scores >= thresholds.index_select(0, predicted_classes)
        probabilities = probabilities[retained]
        masks = masks[retained]
        boxes = boxes[retained]
        queries = queries[retained]
        selected_base_logits = selected_base_logits[retained]
        selected_refined_logits = selected_refined_logits[retained]
        selected_entropy = selected_entropy[retained]
        selected_gate = selected_gate[retained]
        selected_scale_weights = selected_scale_weights[retained]
        selected_iou = selected_iou[retained]
        selected_quality = selected_quality[retained]
        probabilities_np = probabilities.detach().cpu().numpy()
        masks_np = masks.detach().cpu().numpy()
        results.append(
            {
                "official_map": resolve_pannuke_instances(
                    masks_np,
                    probabilities_np,
                    mask_threshold=mask_threshold,
                ),
                "mask_probabilities": masks_np,
                "class_probabilities": probabilities_np,
                "class_ids": probabilities.argmax(dim=-1).detach().cpu().numpy(),
                "class_scores": probabilities.amax(dim=-1).detach().cpu().numpy(),
                "base_logits": selected_base_logits.detach().cpu().numpy(),
                "refined_logits": selected_refined_logits.detach().cpu().numpy(),
                "entropy": selected_entropy.detach().cpu().numpy(),
                "ugca_gate": selected_gate.detach().cpu().numpy(),
                "scale_weights": selected_scale_weights.detach().cpu().numpy(),
                "sam_iou_prediction": selected_iou.detach().cpu().numpy(),
                "ugca_quality_score": selected_quality.detach().cpu().numpy(),
                "boxes_xyxy_original": (boxes * 256.0).detach().cpu().numpy(),
                "boxes_xyxy_normalized": boxes.detach().cpu().numpy(),
                "query_indices": queries.detach().cpu().numpy(),
            }
        )
    return results


def predict_pannuke(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    role: str,
    stage: str,
    config: dict[str, Any],
    device: torch.device,
    destination: str | Path,
    checkpoint: str | Path,
    frozen_decision: str | Path | None = None,
    allow_test: bool = False,
    save_raw_instances: bool = True,
) -> dict[str, Any]:
    """Predict validation or explicitly authorized test data in manifest order."""
    assert_role_access(
        role,  # type: ignore[arg-type]
        checkpoint=checkpoint,
        frozen_decision=frozen_decision,
        allow_test=allow_test,
        config=config,
    )
    if role not in {"validation", "test"}:
        raise ValueError("Prediction role must be validation or test")
    inference = config["inference"]
    output_root = Path(destination)
    raw_root = output_root / "raw_instances"
    if save_raw_instances:
        raw_root.mkdir(parents=True, exist_ok=True)
    else:
        output_root.mkdir(parents=True, exist_ok=True)
    model.eval()
    official_maps = []
    image_ids: list[str] = []
    tissues: list[str] = []

    with torch.inference_mode():
        for raw_batch in loader:
            batch = move_batch_to_device(raw_batch, device)
            output = model(
                batch["yolo_images"],
                batch["sam_resized_images"],
                stage=stage,
                pre_threshold=float(inference["pre_threshold"]),
                max_instances=int(inference["max_instances"]),
            )
            predictions = batch_predictions(
                output,
                batch_size=len(batch["image_ids"]),
                mask_threshold=float(inference["mask_threshold"]),
                final_score_threshold=float(inference["final_score_threshold"]),
                logit_blend=float(inference.get("logit_blend", 1.0)),
                quality_power=float(inference.get("quality_power", 1.0)),
                quality_powers_by_class=inference.get("quality_powers_by_class"),
                final_score_thresholds_by_class=inference.get("final_score_thresholds_by_class"),
                sam_iou_power=float(inference.get("sam_iou_power", 0.0)),
                mask_stability_power=float(inference.get("mask_stability_power", 0.0)),
                mask_stability_delta=float(inference.get("mask_stability_delta", 0.05)),
            )
            for image_id, tissue, prediction in zip(
                batch["image_ids"],
                batch["tissues"],
                predictions,
                strict=True,
            ):
                official_maps.append(prediction.pop("official_map"))
                image_ids.append(str(image_id))
                tissues.append(str(tissue))
                if save_raw_instances:
                    np.savez_compressed(raw_root / f"{image_id}.npz", **prediction)
    stacked = np.stack(official_maps) if official_maps else None
    if stacked is not None and stacked.max(initial=0) > np.iinfo(np.uint16).max:
        raise RuntimeError("Official instance IDs exceed uint16 export capacity")
    masks = (
        stacked.astype(np.uint16, copy=False)
        if official_maps
        else np.zeros((0, 256, 256, NUM_CLASSES), dtype=np.uint16)
    )
    official_root = output_root / "pannuke_official"
    export_official_arrays(
        masks,
        tissues,
        image_ids,
        official_root,
        metadata={
            "role": role,
            "stage": stage,
            "pre_threshold": float(inference["pre_threshold"]),
            "mask_threshold": float(inference["mask_threshold"]),
            "final_score_threshold": float(inference["final_score_threshold"]),
            "logit_blend": float(inference.get("logit_blend", 1.0)),
            "quality_power": float(inference.get("quality_power", 1.0)),
            "quality_powers_by_class": inference.get("quality_powers_by_class"),
            "final_score_thresholds_by_class": inference.get("final_score_thresholds_by_class"),
            "sam_iou_power": float(inference.get("sam_iou_power", 0.0)),
            "mask_stability_power": float(inference.get("mask_stability_power", 0.0)),
            "mask_stability_delta": float(inference.get("mask_stability_delta", 0.05)),
            "box_nms": False,
            "mask_nms": False,
        },
    )
    summary = {
        "role": role,
        "sample_count": len(image_ids),
        "image_ids": image_ids,
        "official_root": str(official_root.resolve()),
        "raw_instances_saved": bool(save_raw_instances),
        "raw_instance_root": str(raw_root.resolve()) if save_raw_instances else None,
    }
    atomic_write_json(output_root / "prediction_summary.json", summary)
    return summary
