#!/usr/bin/env python
"""Evaluate one public PanNuke test shard with a self-contained release weight."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from _bootstrap import PROJECT_ROOT  # noqa: F401
from torch.utils.data import DataLoader, Subset

from dfc_sam.constants import PANNUKE_CLASSES
from dfc_sam.data.collate import collate_pannuke
from dfc_sam.data.pannuke_dataset import PanNukeDataset
from dfc_sam.data.transforms import build_model_input_transform
from dfc_sam.engine.trainer import move_batch_to_device
from dfc_sam.evaluation.inference import quality_scaled_class_probabilities, resolve_final_score_thresholds
from dfc_sam.evaluation.official_metrics import OfficialMetricAccumulator, load_official_primitives
from dfc_sam.evaluation.overlap_resolution import resolve_pannuke_instances
from dfc_sam.evaluation.tta import TTA_VIEWS, TTAFusionSettings, fuse_tta_views, transform_spatial
from dfc_sam.evaluation.validation import _ground_truth_batch
from dfc_sam.models.release_checkpoint import load_release_model
from dfc_sam.utils.hashing import atomic_write_json, sha256_file
from dfc_sam.utils.reproducibility import seed_everything


def _tta_settings(config: dict[str, Any]) -> TTAFusionSettings:
    payload = dict(config["inference"]["tta"])
    if not payload.pop("enabled", False) or tuple(payload.pop("views", ())) != TTA_VIEWS:
        raise ValueError("Release evaluation requires the frozen four-view TTA")
    settings = TTAFusionSettings(**payload)
    settings.validate()
    return settings


def _extract_views(
    output: Any, *, batch_size: int, view: str, inference: dict[str, Any]
) -> list[dict[str, np.ndarray]]:
    probabilities = quality_scaled_class_probabilities(
        output,
        logit_blend=float(inference.get("logit_blend", 1.0)),
        quality_power=float(inference.get("quality_power", 1.0)),
        quality_powers_by_class=inference.get("quality_powers_by_class"),
        sam_iou_power=float(inference.get("sam_iou_power", 0.0)),
        mask_stability_power=float(inference.get("mask_stability_power", 0.0)),
        mask_stability_threshold=float(inference["mask_threshold"]),
        mask_stability_delta=float(inference.get("mask_stability_delta", 0.05)),
    )
    masks = transform_spatial(output.mask_logits[:, 0].float().sigmoid(), view)
    scores = output.base_logits.float().sigmoid().amax(dim=-1)
    result = []
    for batch_index in range(batch_size):
        indices = (output.selected.batch_index == batch_index).nonzero(as_tuple=False).flatten()
        result.append(
            {
                "mask_probabilities": masks.index_select(0, indices).cpu().numpy(),
                "class_probabilities": probabilities.index_select(0, indices).cpu().numpy(),
                "base_scores": scores.index_select(0, indices).cpu().numpy(),
            }
        )
    return result


def _aggregation_state(metrics: dict[str, Any]) -> dict[str, Any]:
    per_image = metrics.pop("per_image")
    tissue = {}
    for name in sorted({str(item["tissue"]) for item in per_image}):
        items = [item for item in per_image if item["tissue"] == name]
        tissue[name] = {"images": len(items)}
        for metric in ("bpq", "mpq"):
            values = np.asarray([item[metric] for item in items], dtype=np.float64)
            finite = values[np.isfinite(values)]
            tissue[name][metric] = {"sum": float(finite.sum()), "count": int(finite.size)}
    values = np.asarray([item["class_pq"] for item in per_image], dtype=np.float64)
    return {
        "tissue": tissue,
        "class_pq": [
            {
                "sum": float(values[np.isfinite(values[:, index]), index].sum()),
                "count": int(np.isfinite(values[:, index]).sum()),
            }
            for index in range(len(PANNUKE_CLASSES))
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--all-samples", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--metrics-repo", type=Path, required=True)
    parser.add_argument("--split-id", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count or args.batch_size < 1:
        raise SystemExit("Invalid shard or batch size")
    output = args.output.expanduser().resolve()
    progress = args.progress.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    progress.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    if int(split["split_id"]) != args.split_id:
        raise ValueError("Manifest split does not match --split-id")
    model, checkpoint = load_release_model(args.checkpoint, device=device)
    if int(checkpoint["split_id"]) != args.split_id:
        raise ValueError("Checkpoint split does not match --split-id")
    if list(split["test_folds"]) != [int(checkpoint["test_fold"])]:
        raise ValueError("Checkpoint test fold does not match the manifest test fold")
    config = checkpoint["config"]
    settings = _tta_settings(config)
    dataset = PanNukeDataset(
        args.all_samples,
        args.split_manifest,
        role="test",
        transform=build_model_input_transform(config),
    )
    expected_size = int(split["sample_counts"]["test"])
    if len(dataset) != expected_size:
        raise ValueError(f"Test dataset size mismatch: {len(dataset)} != {expected_size}")
    indices = list(range(args.shard_index, len(dataset), args.shard_count))
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(config["runtime"].get("num_workers", 4)),
        pin_memory=bool(config["runtime"].get("pin_memory", True)),
        persistent_workers=(
            bool(config["runtime"].get("persistent_workers", True)) and int(config["runtime"].get("num_workers", 4)) > 0
        ),
        collate_fn=collate_pannuke,
    )
    seed_everything(int(config.get("seed", 42)), deterministic=True)
    primitives = load_official_primitives(
        args.metrics_repo,
        expected_commit=str(config["evaluation"]["official_metrics_commit"]),
    )
    accumulator = OfficialMetricAccumulator(
        get_fast_pq=primitives[0],
        binarize=primitives[1],
        remap_label=primitives[2],
        match_iou=float(config["evaluation"]["pq_match_iou"]),
    )
    inference = config["inference"]
    thresholds = resolve_final_score_thresholds(inference)
    started = time.monotonic()
    processed = 0
    with torch.inference_mode():
        for raw_batch in loader:
            truths = _ground_truth_batch(raw_batch)
            batch = move_batch_to_device(raw_batch, device)
            views = []
            for view in TTA_VIEWS:
                model_output = model(
                    transform_spatial(batch["yolo_images"], view),
                    transform_spatial(batch["sam_resized_images"], view),
                    stage="joint",
                    pre_threshold=settings.pre_threshold,
                    max_instances=int(inference["max_instances"]),
                )
                views.append(
                    _extract_views(model_output, batch_size=len(batch["image_ids"]), view=view, inference=inference)
                )
            for batch_index, (image_id, tissue, truth) in enumerate(
                zip(batch["image_ids"], batch["tissues"], truths, strict=True)
            ):
                masks, probabilities, _ = fuse_tta_views(
                    [view[batch_index] for view in views], settings, base_final_thresholds=thresholds
                )
                prediction = resolve_pannuke_instances(masks, probabilities, mask_threshold=settings.mask_threshold)
                accumulator.update(truth, prediction, tissue=str(tissue), image_id=str(image_id))
                processed += 1
            elapsed = time.monotonic() - started
            fraction = processed / max(len(indices), 1)
            atomic_write_json(
                progress,
                {
                    "status": "running",
                    "split_id": args.split_id,
                    "shard_index": args.shard_index,
                    "processed": processed,
                    "total": len(indices),
                    "fraction": fraction,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": elapsed * (1.0 - fraction) / fraction if fraction else None,
                },
            )
            print(f"Split{args.split_id} shard{args.shard_index}: {processed}/{len(indices)}", flush=True)
    metrics = accumulator.compute()
    aggregation = _aggregation_state(metrics)
    payload = {
        "status": "completed",
        "role": "test",
        "split_id": args.split_id,
        "test_fold": int(checkpoint["test_fold"]),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "dataset_size": len(dataset),
        "dataset_indices": indices,
        "sample_count": processed,
        "elapsed_seconds": time.monotonic() - started,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "metrics": metrics,
        "aggregation_state": aggregation,
    }
    atomic_write_json(output, payload)
    atomic_write_json(
        progress,
        {
            "status": "completed",
            "split_id": args.split_id,
            "shard_index": args.shard_index,
            "processed": processed,
            "total": len(indices),
            "fraction": 1.0,
            "elapsed_seconds": payload["elapsed_seconds"],
            "eta_seconds": 0.0,
        },
    )


if __name__ == "__main__":
    main()
