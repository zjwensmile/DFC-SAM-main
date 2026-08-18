#!/usr/bin/env python
"""Validation-only Dead/Inflammatory quality and threshold calibration."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from _bootstrap import PROJECT_ROOT  # noqa: E402,F401
from torch.utils.data import DataLoader  # noqa: E402

from dfc_sam.config import load_config, validate_experiment_config  # noqa: E402
from dfc_sam.constants import PANNUKE_CLASSES  # noqa: E402
from dfc_sam.data.collate import collate_pannuke  # noqa: E402
from dfc_sam.data.pannuke_dataset import PanNukeDataset  # noqa: E402
from dfc_sam.data.transforms import build_model_input_transform  # noqa: E402
from dfc_sam.engine.checkpoint import load_training_checkpoint  # noqa: E402
from dfc_sam.engine.trainer import move_batch_to_device  # noqa: E402
from dfc_sam.evaluation.inference import quality_scaled_class_probabilities  # noqa: E402
from dfc_sam.evaluation.official_metrics import OfficialMetricAccumulator, load_official_primitives  # noqa: E402
from dfc_sam.evaluation.overlap_resolution import resolve_pannuke_instances  # noqa: E402
from dfc_sam.evaluation.validation import _ground_truth_batch  # noqa: E402
from dfc_sam.models.factory import build_dfc_sam  # noqa: E402
from dfc_sam.utils.hashing import atomic_write_json, sha256_file, sha256_json  # noqa: E402
from dfc_sam.utils.reproducibility import seed_everything  # noqa: E402

DEAD_INDEX = PANNUKE_CLASSES.index("dead")
INFLAMMATORY_INDEX = PANNUKE_CLASSES.index("inflammatory")
BASE_POWER = 0.25
BASE_THRESHOLD = 0.35
PRE_THRESHOLD = 0.40
LOGIT_BLEND = 1.0


def _unique(values: list[float]) -> list[float]:
    return sorted({float(value) for value in values})


def _component(value: float) -> str:
    return f"{value:.3f}".replace(".", "p")


@dataclass(frozen=True)
class Candidate:
    dead_power: float
    inflammatory_power: float
    dead_threshold: float
    inflammatory_threshold: float
    mask_threshold: float
    base_power: float = BASE_POWER
    base_threshold: float = BASE_THRESHOLD
    pre_threshold: float = PRE_THRESHOLD
    logit_blend: float = LOGIT_BLEND

    def quality_powers(self) -> list[float]:
        values = [self.base_power] * len(PANNUKE_CLASSES)
        values[DEAD_INDEX] = self.dead_power
        values[INFLAMMATORY_INDEX] = self.inflammatory_power
        return values

    def final_thresholds(self) -> list[float]:
        values = [self.base_threshold] * len(PANNUKE_CLASSES)
        values[DEAD_INDEX] = self.dead_threshold
        values[INFLAMMATORY_INDEX] = self.inflammatory_threshold
        return values

    @property
    def candidate_id(self) -> str:
        return "__".join(
            (
                f"deadq_{_component(self.dead_power)}",
                f"inflq_{_component(self.inflammatory_power)}",
                f"deadt_{_component(self.dead_threshold)}",
                f"inflt_{_component(self.inflammatory_threshold)}",
                f"mask_{_component(self.mask_threshold)}",
            )
        )

    def settings(self) -> dict[str, Any]:
        return {
            "logit_blend": self.logit_blend,
            "quality_power": self.base_power,
            "quality_powers_by_class": self.quality_powers(),
            "pre_threshold": self.pre_threshold,
            "final_score_threshold": self.base_threshold,
            "final_score_thresholds_by_class": self.final_thresholds(),
            "mask_threshold": self.mask_threshold,
        }


def selection_key(candidate: dict[str, Any]) -> tuple[float, ...]:
    metrics = candidate["metrics"]
    settings = candidate["settings"]
    powers = settings["quality_powers_by_class"]
    thresholds = settings["final_score_thresholds_by_class"]
    return (
        float(metrics["mpq"]),
        float(metrics["bpq"]),
        float(metrics["f1det"]),
        float(metrics["macro_f1"]),
        -abs(float(powers[DEAD_INDEX]) - float(settings["quality_power"])),
        -abs(float(powers[INFLAMMATORY_INDEX]) - float(settings["quality_power"])),
        -abs(float(thresholds[DEAD_INDEX]) - float(settings["final_score_threshold"])),
        -abs(float(thresholds[INFLAMMATORY_INDEX]) - float(settings["final_score_threshold"])),
        -abs(float(settings["mask_threshold"]) - 0.55),
    )


def _candidate_count(prediction: np.ndarray) -> int:
    return sum(max(0, int(np.unique(prediction[..., index]).size) - 1) for index in range(5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--all-samples", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--dead-quality-powers", type=float, nargs="+", required=True)
    parser.add_argument("--inflammatory-quality-powers", type=float, nargs="+", required=True)
    parser.add_argument("--dead-thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--inflammatory-thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--mask-thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--metric-workers", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.dry_run == args.execute:
        raise SystemExit("Choose exactly one of --dry-run or --execute")

    config = load_config(args.config)
    validate_experiment_config(config)
    inference = config["inference"]
    base_power = float(inference["quality_power"])
    base_threshold = float(inference["final_score_threshold"])
    pre_threshold = float(inference["pre_threshold"])
    logit_blend = float(inference["logit_blend"])
    grids = {
        "dead_quality_powers": _unique(args.dead_quality_powers),
        "inflammatory_quality_powers": _unique(args.inflammatory_quality_powers),
        "dead_thresholds": _unique(args.dead_thresholds),
        "inflammatory_thresholds": _unique(args.inflammatory_thresholds),
        "mask_thresholds": _unique(args.mask_thresholds),
    }
    candidates = [
        Candidate(
            *values,
            base_power=base_power,
            base_threshold=base_threshold,
            pre_threshold=pre_threshold,
            logit_blend=logit_blend,
        )
        for values in itertools.product(*grids.values())
    ]
    for candidate in candidates:
        if any(not 0.0 <= value <= 1.0 for value in (
            candidate.dead_power,
            candidate.inflammatory_power,
            candidate.dead_threshold,
            candidate.inflammatory_threshold,
            candidate.mask_threshold,
        )):
            raise SystemExit("All class-aware calibration values must be in [0,1]")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    plan = {
        "schema_version": 1,
        "role": "validation",
        "method": "rfdetr_class_aware_scores",
        "stage": "joint",
        "split_id": int(config["experiment"]["split_id"]),
        "config": str(Path(args.config).expanduser().resolve()),
        "config_sha256": sha256_json(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "split_manifest": str(Path(args.split_manifest).expanduser().resolve()),
        "candidate_count": len(candidates),
        "candidate_grid": grids,
        "selection_rule": "maximize validation mPQ, then bPQ, F1det, Macro-F1, then minimal class deviation",
        "sample_count": None,
        "execute": bool(args.execute),
        "sealed_test_access": False,
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if output_path.exists():
        raise SystemExit(f"Refusing to overwrite class-aware calibration: {output_path}")

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
    started = time.monotonic()
    processed = 0

    def evaluate(
        index: int,
        *,
        masks: np.ndarray,
        probabilities: np.ndarray,
        truth: np.ndarray,
        tissue: str,
        image_id: str,
    ) -> tuple[int, int]:
        candidate = candidates[index]
        class_ids = probabilities.argmax(axis=-1)
        scores = probabilities.max(axis=-1)
        thresholds = np.asarray(candidate.final_thresholds(), dtype=np.float32)
        retained = scores >= thresholds[class_ids]
        prediction = resolve_pannuke_instances(
            masks[retained],
            probabilities[retained],
            mask_threshold=candidate.mask_threshold,
        )
        accumulators[index].update(truth, prediction, tissue=tissue, image_id=image_id)
        return int(retained.sum()), _candidate_count(prediction)

    power_vectors = {tuple(candidate.quality_powers()) for candidate in candidates}
    with ThreadPoolExecutor(max_workers=args.metric_workers) as executor, torch.inference_mode():
        for raw_batch in loader:
            ground_truth = _ground_truth_batch(raw_batch)
            batch = move_batch_to_device(raw_batch, device)
            output = model(
                batch["yolo_images"],
                batch["sam_resized_images"],
                stage="joint",
                pre_threshold=pre_threshold,
                max_instances=int(config["inference"]["max_instances"]),
            )
            masks_all = output.mask_logits[:, 0].float().sigmoid()
            probabilities = {
                powers: quality_scaled_class_probabilities(
                    output,
                    logit_blend=logit_blend,
                    quality_power=base_power,
                    quality_powers_by_class=powers,
                )
                for powers in power_vectors
            }
            for batch_index, (image_id, tissue, truth) in enumerate(
                zip(batch["image_ids"], batch["tissues"], ground_truth, strict=True)
            ):
                indices = (output.selected.batch_index == batch_index).nonzero(as_tuple=False).flatten()
                masks = masks_all.index_select(0, indices).cpu().numpy()
                image_probabilities = {
                    powers: values.index_select(0, indices).cpu().numpy()
                    for powers, values in probabilities.items()
                }
                futures = [
                    executor.submit(
                        evaluate,
                        index,
                        masks=masks,
                        probabilities=image_probabilities[tuple(candidate.quality_powers())],
                        truth=truth,
                        tissue=str(tissue),
                        image_id=str(image_id),
                    )
                    for index, candidate in enumerate(candidates)
                ]
                for index, future in enumerate(futures):
                    retained, emitted = future.result()
                    retained_counts[index] += retained
                    emitted_counts[index] += emitted
                processed += 1
                if processed % args.progress_every == 0 or processed == len(dataset):
                    elapsed = time.monotonic() - started
                    print(
                        f"validation progress: {processed}/{len(dataset)} images ({elapsed / 60.0:.1f} min)",
                        flush=True,
                    )

    results = []
    for index, candidate in enumerate(candidates):
        metrics = accumulators[index].compute()
        metrics.pop("per_image", None)
        results.append(
            {
                "candidate_id": candidate.candidate_id,
                "settings": candidate.settings(),
                "retained_candidates": int(retained_counts[index]),
                "emitted_instances": int(emitted_counts[index]),
                "metrics": metrics,
            }
        )
    payload = {
        **plan,
        "sample_count": processed,
        "elapsed_seconds": time.monotonic() - started,
        "selected": max(results, key=selection_key),
        "candidates": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_path, payload)
    print(json.dumps({"status": "completed", "selected": payload["selected"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
