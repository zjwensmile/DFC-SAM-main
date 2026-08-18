#!/usr/bin/env python
"""Calibrate NMS-free inference thresholds on one validation fold."""

from __future__ import annotations

import argparse
import itertools
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
from dfc_sam.evaluation.inference import quality_scaled_class_probabilities  # noqa: E402
from dfc_sam.evaluation.official_metrics import (  # noqa: E402
    OfficialMetricAccumulator,
    load_official_primitives,
)
from dfc_sam.evaluation.threshold_calibration import (  # noqa: E402
    InferenceThresholds,
    resolve_threshold_candidate,
    select_best_candidate,
)
from dfc_sam.evaluation.validation import _ground_truth_batch  # noqa: E402
from dfc_sam.models.factory import build_dfc_sam  # noqa: E402
from dfc_sam.utils.hashing import atomic_write_json, sha256_file, sha256_json  # noqa: E402
from dfc_sam.utils.reproducibility import seed_everything  # noqa: E402

SELECTION_RULE = (
    "maximize validation mPQ; break exact ties by bPQ, F1det, Macro-F1, higher "
    "pre-threshold, higher final-score threshold, then mask threshold nearest 0.5"
)


def _unique(values: list[float]) -> list[float]:
    return sorted({float(value) for value in values})


def _candidate_count(prediction: np.ndarray) -> int:
    return sum(max(0, int(np.unique(prediction[..., index]).size) - 1) for index in range(5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--all-samples", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--stage", choices=("warmup", "joint"), default="joint")
    parser.add_argument("--pre-thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--final-score-thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--mask-thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--parent-calibration", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--metric-workers", type=int, default=6)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.dry_run == args.execute:
        raise SystemExit("Choose exactly one of --dry-run or --execute")
    if args.metric_workers < 1 or args.progress_every < 1:
        raise SystemExit("--metric-workers and --progress-every must be positive")

    config = load_config(args.config)
    validate_experiment_config(config)
    pre_thresholds = _unique(args.pre_thresholds)
    final_thresholds = _unique(args.final_score_thresholds)
    mask_thresholds = _unique(args.mask_thresholds)
    candidates = [
        InferenceThresholds(pre, final, mask)
        for pre, final, mask in itertools.product(
            pre_thresholds,
            final_thresholds,
            mask_thresholds,
        )
    ]
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    plan = {
        "schema_version": 1,
        "role": "validation",
        "stage": args.stage,
        "split_id": int(config["experiment"]["split_id"]),
        "config": str(Path(args.config).expanduser().resolve()),
        "config_sha256": sha256_json(config),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "split_manifest": str(Path(args.split_manifest).expanduser().resolve()),
        "candidate_count": len(candidates),
        "candidate_grid": {
            "pre_thresholds": pre_thresholds,
            "final_score_thresholds": final_thresholds,
            "mask_thresholds": mask_thresholds,
        },
        "selection_rule": SELECTION_RULE,
        "parent_calibrations": [
            {
                "artifact": str(Path(path).expanduser().resolve()),
                "artifact_sha256": sha256_file(Path(path).expanduser().resolve()),
            }
            for path in args.parent_calibration
        ],
        "model_forward_pre_threshold": min(pre_thresholds),
        "max_instances": int(config["inference"]["max_instances"]),
        "metric_workers": args.metric_workers,
        "output": str(output_path),
        "execute": args.execute,
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if output_path.exists():
        raise SystemExit(f"Refusing to overwrite calibration output: {output_path}")

    device = torch.device(args.device)
    seed_everything(
        int(config["experiment"]["seed"]),
        deterministic=bool(config["train"]["deterministic"]),
    )
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
        persistent_workers=bool(
            config["runtime"]["persistent_workers"] and int(config["runtime"]["num_workers"]) > 0
        ),
        collate_fn=collate_pannuke,
    )
    model = build_dfc_sam(config).to(device)
    checkpoint = load_training_checkpoint(checkpoint_path)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    repository = config["evaluation"]["official_metrics_repo"]
    commit = str(config["evaluation"]["official_metrics_commit"])
    match_iou = float(config["evaluation"]["pq_match_iou"])
    get_fast_pq, binarize, remap_label = load_official_primitives(
        repository,
        expected_commit=commit,
    )
    official_accumulators = [
        OfficialMetricAccumulator(
            get_fast_pq=get_fast_pq,
            binarize=binarize,
            remap_label=remap_label,
            match_iou=match_iou,
        )
        for _ in candidates
    ]
    retained_counts = np.zeros(len(candidates), dtype=np.int64)
    emitted_counts = np.zeros(len(candidates), dtype=np.int64)
    started = time.monotonic()
    processed = 0

    def evaluate_candidate(
        candidate_index: int,
        *,
        masks: np.ndarray,
        class_probabilities: np.ndarray,
        base_scores: np.ndarray,
        truth: np.ndarray,
        tissue: str,
        image_id: str,
    ) -> tuple[int, int]:
        prediction, retained = resolve_threshold_candidate(
            masks,
            class_probabilities,
            base_scores,
            candidates[candidate_index],
        )
        official_accumulators[candidate_index].update(
            truth,
            prediction,
            tissue=tissue,
            image_id=image_id,
        )
        return retained, _candidate_count(prediction)

    with ThreadPoolExecutor(max_workers=args.metric_workers) as executor, torch.inference_mode():
        for raw_batch in loader:
            ground_truth = _ground_truth_batch(raw_batch)
            batch = move_batch_to_device(raw_batch, device)
            output = model(
                batch["yolo_images"],
                batch["sam_resized_images"],
                stage=args.stage,
                pre_threshold=min(pre_thresholds),
                max_instances=int(config["inference"]["max_instances"]),
            )
            base_scores_all = output.base_logits.float().sigmoid().amax(dim=-1)
            class_probabilities_all = quality_scaled_class_probabilities(output)
            masks_all = output.mask_logits[:, 0].float().sigmoid()
            for batch_index, (image_id, tissue, truth) in enumerate(
                zip(batch["image_ids"], batch["tissues"], ground_truth, strict=True)
            ):
                indices = (output.selected.batch_index == batch_index).nonzero(
                    as_tuple=False
                ).flatten()
                masks = masks_all.index_select(0, indices).cpu().numpy()
                class_probabilities = class_probabilities_all.index_select(0, indices).cpu().numpy()
                base_scores = base_scores_all.index_select(0, indices).cpu().numpy()
                futures = [
                    executor.submit(
                        evaluate_candidate,
                        candidate_index,
                        masks=masks,
                        class_probabilities=class_probabilities,
                        base_scores=base_scores,
                        truth=truth,
                        tissue=str(tissue),
                        image_id=str(image_id),
                    )
                    for candidate_index in range(len(candidates))
                ]
                for candidate_index, future in enumerate(futures):
                    retained, emitted = future.result()
                    retained_counts[candidate_index] += retained
                    emitted_counts[candidate_index] += emitted
                processed += 1
                if processed % args.progress_every == 0 or processed == len(dataset):
                    elapsed = time.monotonic() - started
                    print(
                        f"validation progress: {processed}/{len(dataset)} images "
                        f"({elapsed / 60.0:.1f} min)",
                        flush=True,
                    )

    results: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        official = official_accumulators[index].compute()
        official.pop("per_image", None)
        results.append(
            {
                "candidate_id": candidate.candidate_id,
                "thresholds": candidate.as_dict(),
                "retained_candidates": int(retained_counts[index]),
                "emitted_instances": int(emitted_counts[index]),
                "metrics": official,
            }
        )
    selected = select_best_candidate(results)
    payload: dict[str, Any] = {
        **plan,
        "execute": True,
        "sample_count": processed,
        "elapsed_seconds": time.monotonic() - started,
        "selected": selected,
        "candidates": results,
    }
    atomic_write_json(output_path, payload)
    print(
        json.dumps(
            {
                "status": "completed",
                "output": str(output_path),
                "sample_count": processed,
                "selected": selected,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
