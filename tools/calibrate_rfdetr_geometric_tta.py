#!/usr/bin/env python
"""Optional legacy validation-only four-view TTA experiment.

This utility is retained for research comparisons and is not used by the
single-view public release evaluator.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from _bootstrap import PROJECT_ROOT  # noqa: E402,F401
from torch.utils.data import DataLoader  # noqa: E402

from dfc_sam.config import load_config, validate_experiment_config  # noqa: E402
from dfc_sam.data.collate import collate_pannuke  # noqa: E402
from dfc_sam.data.pannuke_dataset import PanNukeDataset  # noqa: E402
from dfc_sam.data.transforms import build_model_input_transform  # noqa: E402
from dfc_sam.engine.checkpoint import load_training_checkpoint  # noqa: E402
from dfc_sam.engine.trainer import move_batch_to_device  # noqa: E402
from dfc_sam.evaluation.inference import (  # noqa: E402
    quality_scaled_class_probabilities,
    resolve_final_score_thresholds,
)
from dfc_sam.evaluation.official_metrics import OfficialMetricAccumulator, load_official_primitives  # noqa: E402
from dfc_sam.evaluation.overlap_resolution import resolve_pannuke_instances  # noqa: E402
from dfc_sam.evaluation.tta import TTA_VIEWS, TTAFusionSettings, fuse_tta_views, transform_spatial  # noqa: E402
from dfc_sam.evaluation.validation import _ground_truth_batch  # noqa: E402
from dfc_sam.models.factory import build_dfc_sam  # noqa: E402
from dfc_sam.utils.hashing import atomic_write_json, sha256_file, sha256_json  # noqa: E402
from dfc_sam.utils.reproducibility import seed_everything  # noqa: E402


def _read_candidates(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("TTA candidate plan must be a non-empty list")
    identifiers = [str(candidate["candidate_id"]) for candidate in payload]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("TTA candidate plan repeats candidate_id")
    for candidate in payload:
        TTAFusionSettings(**{key: value for key, value in candidate.items() if key != "candidate_id"}).validate()
    return payload


def _selection_key(candidate: dict[str, Any]) -> tuple[float, ...]:
    metrics = candidate["metrics"]
    settings = candidate["settings"]
    return (
        float(metrics["mpq"]) + float(metrics["f1det"]),
        float(metrics["mpq"]),
        float(metrics["f1det"]),
        float(metrics["bpq"]),
        float(metrics["macro_f1"]),
        -float(settings["view_count"]),
        -float(settings["min_votes"]),
    )


def _extract_views(output: Any, batch_size: int, view: str, inference: dict[str, Any]) -> list[dict[str, np.ndarray]]:
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
    base_scores = output.base_logits.float().sigmoid().amax(dim=-1)
    results = []
    for batch_index in range(batch_size):
        indices = (output.selected.batch_index == batch_index).nonzero(as_tuple=False).flatten()
        results.append(
            {
                "mask_probabilities": masks.index_select(0, indices).cpu().numpy(),
                "class_probabilities": probabilities.index_select(0, indices).cpu().numpy(),
                "base_scores": base_scores.index_select(0, indices).cpu().numpy(),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--all-samples", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--metric-workers", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.dry_run == args.execute:
        raise SystemExit("Choose exactly one of --dry-run or --execute")
    config = load_config(args.config)
    validate_experiment_config(config)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    candidate_path = Path(args.candidates).expanduser().resolve()
    candidates = _read_candidates(candidate_path)
    output_path = Path(args.output).expanduser().resolve()
    plan = {
        "schema_version": 1,
        "role": "validation",
        "method": "rfdetr_sam_h_geometric_tta",
        "split_id": int(config["experiment"]["split_id"]),
        "config": str(Path(args.config).expanduser().resolve()),
        "config_sha256": sha256_json(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "candidate_plan": str(candidate_path),
        "candidate_plan_sha256": sha256_file(candidate_path),
        "candidate_count": len(candidates),
        "views": list(TTA_VIEWS),
        "sample_count": None,
        "sealed_test_access": False,
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if output_path.exists():
        raise FileExistsError(output_path)

    device = torch.device(args.device)
    seed_everything(int(config["experiment"]["seed"]), deterministic=bool(config["train"]["deterministic"]))
    dataset = PanNukeDataset(
        args.all_samples,
        args.split_manifest,
        role="validation",
        transform=build_model_input_transform(config),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["train"]["batch_size_per_gpu"]),
        shuffle=False,
        num_workers=int(config["runtime"]["num_workers"]),
        pin_memory=bool(config["runtime"]["pin_memory"]),
        persistent_workers=bool(config["runtime"]["persistent_workers"]),
        collate_fn=collate_pannuke,
    )
    model = build_dfc_sam(config).to(device)
    model.load_state_dict(load_training_checkpoint(checkpoint)["model"], strict=True)
    model.eval()
    get_fast_pq, binarize, remap_label = load_official_primitives(
        config["evaluation"]["official_metrics_repo"],
        expected_commit=str(config["evaluation"]["official_metrics_commit"]),
    )
    accumulators = [
        OfficialMetricAccumulator(
            get_fast_pq=get_fast_pq,
            binarize=binarize,
            remap_label=remap_label,
            match_iou=float(config["evaluation"]["pq_match_iou"]),
        )
        for _ in candidates
    ]
    retained_counts = np.zeros(len(candidates), dtype=np.int64)
    emitted_counts = np.zeros(len(candidates), dtype=np.int64)
    inference = config["inference"]
    base_thresholds = resolve_final_score_thresholds(inference)
    min_pre = min(float(candidate["pre_threshold"]) for candidate in candidates)
    processed = 0
    started = time.monotonic()

    def evaluate(
        index: int,
        *,
        views: list[dict[str, np.ndarray]],
        truth: np.ndarray,
        tissue: str,
        image_id: str,
    ) -> tuple[int, int]:
        payload = candidates[index]
        settings = TTAFusionSettings(**{key: value for key, value in payload.items() if key != "candidate_id"})
        masks, probabilities, counts = fuse_tta_views(
            views,
            settings,
            base_final_thresholds=base_thresholds,
        )
        prediction = resolve_pannuke_instances(
            masks,
            probabilities,
            mask_threshold=settings.mask_threshold,
        )
        accumulators[index].update(truth, prediction, tissue=tissue, image_id=image_id)
        emitted = sum(max(0, int(np.unique(prediction[..., channel]).size) - 1) for channel in range(5))
        return int(counts["retained"]), emitted

    with ThreadPoolExecutor(max_workers=args.metric_workers) as executor, torch.inference_mode():
        for raw_batch in loader:
            truth_batch = _ground_truth_batch(raw_batch)
            batch = move_batch_to_device(raw_batch, device)
            batch_views: list[list[dict[str, np.ndarray]]] = []
            for view in TTA_VIEWS:
                output = model(
                    transform_spatial(batch["yolo_images"], view),
                    transform_spatial(batch["sam_resized_images"], view),
                    stage="joint",
                    pre_threshold=min_pre,
                    max_instances=int(inference["max_instances"]),
                )
                batch_views.append(_extract_views(output, len(batch["image_ids"]), view, inference))
            for batch_index, (image_id, tissue, truth) in enumerate(
                zip(batch["image_ids"], batch["tissues"], truth_batch, strict=True)
            ):
                views = [values[batch_index] for values in batch_views]
                futures = [
                    executor.submit(
                        evaluate,
                        index,
                        views=views,
                        truth=truth,
                        tissue=str(tissue),
                        image_id=str(image_id),
                    )
                    for index in range(len(candidates))
                ]
                for index, future in enumerate(futures):
                    retained, emitted = future.result()
                    retained_counts[index] += retained
                    emitted_counts[index] += emitted
                processed += 1
                if processed % args.progress_every == 0 or processed == len(dataset):
                    print(
                        f"validation progress: {processed}/{len(dataset)} images "
                        f"({(time.monotonic() - started) / 60.0:.1f} min)",
                        flush=True,
                    )
            del output, batch_views

    results = []
    for index, candidate in enumerate(candidates):
        metrics = accumulators[index].compute()
        metrics.pop("per_image", None)
        results.append(
            {
                "candidate_id": candidate["candidate_id"],
                "settings": {key: value for key, value in candidate.items() if key != "candidate_id"},
                "retained_clusters": int(retained_counts[index]),
                "emitted_instances": int(emitted_counts[index]),
                "metrics": metrics,
            }
        )
    payload = {
        **plan,
        "sample_count": processed,
        "elapsed_seconds": time.monotonic() - started,
        "selected": max(results, key=_selection_key),
        "candidates": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_path, payload)
    print(json.dumps({"status": "completed", "selected": payload["selected"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
