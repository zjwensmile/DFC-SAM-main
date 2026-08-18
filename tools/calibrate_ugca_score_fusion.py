#!/usr/bin/env python
"""Validation-only calibration of base/refined logits and learned quality strength."""

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
from dfc_sam.data.collate import collate_pannuke  # noqa: E402
from dfc_sam.data.pannuke_dataset import PanNukeDataset  # noqa: E402
from dfc_sam.data.transforms import build_model_input_transform  # noqa: E402
from dfc_sam.engine.checkpoint import load_training_checkpoint  # noqa: E402
from dfc_sam.engine.trainer import move_batch_to_device  # noqa: E402
from dfc_sam.evaluation.official_metrics import OfficialMetricAccumulator, load_official_primitives  # noqa: E402
from dfc_sam.evaluation.threshold_calibration import (  # noqa: E402
    InferenceThresholds,
    resolve_threshold_candidate,
)
from dfc_sam.evaluation.validation import _ground_truth_batch  # noqa: E402
from dfc_sam.models.factory import build_dfc_sam  # noqa: E402
from dfc_sam.utils.hashing import atomic_write_json, sha256_file, sha256_json  # noqa: E402
from dfc_sam.utils.reproducibility import seed_everything  # noqa: E402

SELECTION_RULE = (
    "maximize validation mPQ; break ties by bPQ, F1det, Macro-F1, then prefer "
    "weaker UGCA blend/quality scaling and conservative higher thresholds"
)


def _unique(values: list[float]) -> list[float]:
    return sorted({float(value) for value in values})


def _component(value: float) -> str:
    return f"{value:.4f}".replace(".", "p")


@dataclass(frozen=True)
class FusionCandidate:
    logit_blend: float
    quality_power: float
    thresholds: InferenceThresholds

    def __post_init__(self) -> None:
        if not 0.0 <= self.logit_blend <= 1.0:
            raise ValueError("logit_blend must be in [0,1]")
        if not 0.0 <= self.quality_power <= 1.0:
            raise ValueError("quality_power must be in [0,1]")

    @property
    def candidate_id(self) -> str:
        return (
            f"blend_{_component(self.logit_blend)}__quality_{_component(self.quality_power)}__"
            f"{self.thresholds.candidate_id}"
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "logit_blend": self.logit_blend,
            "quality_power": self.quality_power,
            **self.thresholds.as_dict(),
        }


def selection_key(result: dict[str, Any]) -> tuple[float, ...]:
    metrics = result["metrics"]
    settings = result["settings"]
    return (
        float(metrics["mpq"]),
        float(metrics["bpq"]),
        float(metrics["f1det"]),
        float(metrics["macro_f1"]),
        -float(settings["logit_blend"]),
        -float(settings["quality_power"]),
        float(settings["pre_threshold"]),
        float(settings["final_score_threshold"]),
        -abs(float(settings["mask_threshold"]) - 0.5),
    )


def select_best(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("Fusion calibration has no candidates")
    return max(results, key=selection_key)


def _candidate_count(prediction: np.ndarray) -> int:
    return sum(max(0, int(np.unique(prediction[..., index]).size) - 1) for index in range(5))


def _probabilities(output: Any, blend: float, quality_power: float) -> torch.Tensor:
    base = output.base_logits.float()
    refined = output.refined_logits.float()
    logits = base + float(blend) * (refined - base)
    mode = str(getattr(output, "class_probability_mode", "softmax"))
    if mode == "sigmoid":
        probabilities = logits.sigmoid()
    elif mode == "softmax":
        probabilities = logits.softmax(dim=-1)
    else:
        raise ValueError(f"Unknown class_probability_mode: {mode}")
    quality = None if output.ugca is None else output.ugca.quality_score
    if quality is not None and quality_power > 0.0:
        scale = quality.float().clamp(1.0e-6, 1.0).pow(float(quality_power))
        probabilities = probabilities * scale[:, None]
    return probabilities


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--all-samples", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--logit-blends", type=float, nargs="+", required=True)
    parser.add_argument("--quality-powers", type=float, nargs="+", required=True)
    parser.add_argument("--pre-thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--final-score-thresholds", type=float, nargs="+", required=True)
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
    blends = _unique(args.logit_blends)
    powers = _unique(args.quality_powers)
    pre_thresholds = _unique(args.pre_thresholds)
    final_thresholds = _unique(args.final_score_thresholds)
    mask_thresholds = _unique(args.mask_thresholds)
    candidates = [
        FusionCandidate(blend, power, InferenceThresholds(pre, final, mask))
        for blend, power, pre, final, mask in itertools.product(
            blends, powers, pre_thresholds, final_thresholds, mask_thresholds
        )
    ]
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    plan = {
        "schema_version": 1,
        "role": "validation",
        "method": "ugca_score_fusion",
        "stage": "joint",
        "split_id": int(config["experiment"]["split_id"]),
        "config": str(Path(args.config).expanduser().resolve()),
        "config_sha256": sha256_json(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "split_manifest": str(Path(args.split_manifest).expanduser().resolve()),
        "candidate_count": len(candidates),
        "candidate_grid": {
            "logit_blends": blends,
            "quality_powers": powers,
            "pre_thresholds": pre_thresholds,
            "final_score_thresholds": final_thresholds,
            "mask_thresholds": mask_thresholds,
        },
        "selection_rule": SELECTION_RULE,
        "model_forward_pre_threshold": min(pre_thresholds),
        "max_instances": int(config["inference"]["max_instances"]),
        "metric_workers": int(args.metric_workers),
        "output": str(output_path),
        "execute": bool(args.execute),
        "sealed_test_access": False,
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if output_path.exists():
        raise SystemExit(f"Refusing to overwrite fusion calibration: {output_path}")

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
    payload = load_training_checkpoint(checkpoint)
    model.load_state_dict(payload["model"], strict=True)
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
        base_scores: np.ndarray,
        truth: np.ndarray,
        tissue: str,
        image_id: str,
    ) -> tuple[int, int]:
        prediction, retained = resolve_threshold_candidate(
            masks,
            probabilities,
            base_scores,
            candidates[index].thresholds,
        )
        accumulators[index].update(truth, prediction, tissue=tissue, image_id=image_id)
        return retained, _candidate_count(prediction)

    pairs = sorted({(candidate.logit_blend, candidate.quality_power) for candidate in candidates})
    with ThreadPoolExecutor(max_workers=args.metric_workers) as executor, torch.inference_mode():
        for raw_batch in loader:
            ground_truth = _ground_truth_batch(raw_batch)
            batch = move_batch_to_device(raw_batch, device)
            output = model(
                batch["yolo_images"],
                batch["sam_resized_images"],
                stage="joint",
                pre_threshold=min(pre_thresholds),
                max_instances=int(config["inference"]["max_instances"]),
            )
            base_scores_all = output.base_logits.float().sigmoid().amax(dim=-1)
            masks_all = output.mask_logits[:, 0].float().sigmoid()
            probability_cache = {pair: _probabilities(output, *pair) for pair in pairs}
            for batch_index, (image_id, tissue, truth) in enumerate(
                zip(batch["image_ids"], batch["tissues"], ground_truth, strict=True)
            ):
                indices = (output.selected.batch_index == batch_index).nonzero(as_tuple=False).flatten()
                masks = masks_all.index_select(0, indices).cpu().numpy()
                base_scores = base_scores_all.index_select(0, indices).cpu().numpy()
                image_probabilities = {
                    pair: values.index_select(0, indices).cpu().numpy()
                    for pair, values in probability_cache.items()
                }
                futures = [
                    executor.submit(
                        evaluate,
                        index,
                        masks=masks,
                        probabilities=image_probabilities[(candidate.logit_blend, candidate.quality_power)],
                        base_scores=base_scores,
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
                "settings": candidate.as_dict(),
                "retained_candidates": int(retained_counts[index]),
                "emitted_instances": int(emitted_counts[index]),
                "metrics": metrics,
            }
        )
    completed = {
        **plan,
        "execute": True,
        "sample_count": processed,
        "elapsed_seconds": time.monotonic() - started,
        "selected": select_best(results),
        "candidates": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_path, completed)
    print(json.dumps({"status": "completed", "selected": completed["selected"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
