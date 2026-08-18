#!/usr/bin/env python
"""Validate or train one RF-DETR detector on an official PanNuke split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT

RFDETR_ROOT = PROJECT_ROOT / "third_party/rf-detr"
REPOSITORY_COMMIT = "c107a748bbf369ad455cd41d85b9c91a5a0ecbad"
PANNUKE_NUM_CLASSES = 5
VARIANTS: dict[str, dict[str, Any]] = {
    "2xlarge": {
        "class_name": "RFDETR2XLarge",
        "weight": RFDETR_ROOT / "weights/rf-detr-xxlarge.pth",
        "weight_md5": "e3204689c1f0280427e4c33e6a2ac6cd",
        "resolution": 880,
        "patch_size": 20,
        "license": "PML-1.0",
        "extension": "rfdetr-plus==1.0.2",
        "parameters_millions": 126.9,
    },
    "large": {
        "class_name": "RFDETRLarge",
        "weight": RFDETR_ROOT / "weights/rf-detr-large-2026.pth",
        "weight_md5": "5cb72153541cbcb9aa6efa26222acc75",
        "resolution": 704,
        "patch_size": 16,
        "license": "Apache-2.0",
        "extension": None,
        "parameters_millions": 33.9,
    },
}


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_images(dataset: Path, split: str) -> int:
    directory = dataset / split / "images"
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return sum(path.is_symlink() or path.is_file() for path in directory.iterdir())


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Validate immutable inputs and return the complete training plan."""
    variant = VARIANTS[args.variant]
    dataset = (args.dataset or PROJECT_ROOT / f"artifacts/rfdetr_split{args.split_id}/dataset").expanduser().resolve()
    weight = (args.weight or variant["weight"]).expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not (dataset / "data.yaml").is_file() or not (dataset / "provenance.json").is_file():
        raise FileNotFoundError(f"Prepared, provenance-recorded dataset is missing under {dataset}")
    provenance = json.loads((dataset / "provenance.json").read_text(encoding="utf-8"))
    if provenance.get("protocol") != "pannuke_standard_3fold" or provenance.get("split_id") != args.split_id:
        raise RuntimeError(f"RF-DETR training requires official PanNuke Split {args.split_id}")
    if provenance.get("sealed_test_access") is not False:
        raise RuntimeError("Dataset provenance does not prove sealed-test construction")
    if not weight.is_file():
        raise FileNotFoundError(weight)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    weight_md5 = _digest(weight, "md5") if local_rank == 0 else str(variant["weight_md5"])
    if weight_md5 != variant["weight_md5"]:
        raise ValueError(f"{variant['class_name']} weight MD5 mismatch: {weight_md5}")
    block_size = int(variant["patch_size"]) * 2
    if args.resolution <= 0 or args.resolution % block_size:
        raise ValueError(f"resolution must be positive and divisible by {block_size}")
    if min(args.epochs, args.batch_size, args.grad_accum_steps, args.workers) <= 0:
        raise ValueError("Epoch, batch, accumulation, and worker values must be positive")

    world_size = int(os.environ.get("WORLD_SIZE", "4"))
    if args.execute and world_size != 4:
        raise RuntimeError(f"Formal training requires exactly four DDP processes, got WORLD_SIZE={world_size}")
    return {
        "schema_version": 2,
        "stage": "rfdetr_detector_challenger",
        "model": variant["class_name"],
        "variant": args.variant,
        "official_repository": "https://github.com/roboflow/rf-detr",
        "repository_commit": REPOSITORY_COMMIT,
        "license": variant["license"],
        "extension": variant["extension"],
        "parameters_millions": variant["parameters_millions"],
        "num_classes": PANNUKE_NUM_CLASSES,
        "class_head_outputs_including_no_object": PANNUKE_NUM_CLASSES + 1,
        "weight": str(weight),
        "weight_md5": weight_md5,
        "weight_sha256": _digest(weight, "sha256") if local_rank == 0 else "verified_on_rank0",
        "dataset": str(dataset),
        "dataset_provenance": str(dataset / "provenance.json"),
        "output": str(output),
        "split_id": args.split_id,
        "evaluation_split": "validation",
        "sealed_test_access": False,
        "train_images": _count_images(dataset, "train"),
        "validation_images": _count_images(dataset, "valid"),
        "settings": {
            "resolution": args.resolution,
            "num_classes": PANNUKE_NUM_CLASSES,
            "epochs": args.epochs,
            "batch_size_per_gpu": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "world_size": world_size,
            "effective_batch": args.batch_size * args.grad_accum_steps * world_size,
            "lr": args.lr,
            "lr_encoder": args.lr_encoder,
            "early_stopping": True,
            "early_stopping_patience": args.patience,
            "early_stopping_min_delta": args.min_delta,
            "skip_best_epochs": 5,
            "eval_max_dets": 400,
            "amp_dtype": "fp16",
            "seed": 42,
            "gradient_checkpointing": args.gradient_checkpointing,
            "multi_scale": args.multi_scale,
            "expanded_scales": args.expanded_scales,
            "use_ema": True,
        },
    }


def _model_class(variant: str) -> type[Any]:
    if variant == "2xlarge":
        from rfdetr import RFDETR2XLarge

        return RFDETR2XLarge
    from rfdetr import RFDETRLarge

    return RFDETRLarge


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-id", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--weight", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, required=True, help="Per-GPU batch size")
    parser.add_argument("--grad-accum-steps", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-encoder", type=float, default=1.5e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--multi-scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expanded-scales", action=argparse.BooleanOptionalAction, default=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--model-smoke", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    plan = build_plan(args)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank == 0:
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    model_class = _model_class(args.variant)
    if args.dry_run:
        return
    model = model_class(
        pretrain_weights=plan["weight"],
        gradient_checkpointing=args.gradient_checkpointing,
        # Pin this explicitly.  Both official starter checkpoints have a
        # 90-class COCO head, and the current RF-DETR variant configs otherwise
        # treat that default as a user override instead of adapting to PanNuke.
        num_classes=PANNUKE_NUM_CLASSES,
    )
    if model.model_config.num_classes != PANNUKE_NUM_CLASSES:
        raise RuntimeError(
            f"RF-DETR class-head mismatch: expected {PANNUKE_NUM_CLASSES}, "
            f"got {model.model_config.num_classes}"
        )
    class_embed = model.model.model.class_embed
    if class_embed.weight.shape[0] != PANNUKE_NUM_CLASSES + 1:
        raise RuntimeError(
            "RF-DETR classifier output mismatch: expected "
            f"{PANNUKE_NUM_CLASSES + 1} including no-object, got {class_embed.weight.shape[0]}"
        )
    if args.model_smoke:
        print(
            f"MODEL_SMOKE_OK variant={args.variant} classes={PANNUKE_NUM_CLASSES} "
            f"head_outputs={class_embed.weight.shape[0]}",
            flush=True,
        )
        return
    if os.environ.get("RFDETR_SEQUENCE_AUTHORIZED") != "1":
        raise RuntimeError("Formal execution is only allowed through scripts/rfdetr_stage1_sequence.sh")

    output = Path(plan["output"])
    model.train(
        dataset_dir=plan["dataset"],
        dataset_file="yolo",
        output_dir=plan["output"],
        resolution=args.resolution,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        num_workers=args.workers,
        devices=args.devices,
        seed=42,
        use_ema=True,
        early_stopping=True,
        early_stopping_patience=args.patience,
        early_stopping_min_delta=args.min_delta,
        skip_best_epochs=5,
        checkpoint_interval=5,
        eval_max_dets=400,
        eval_interval=1,
        log_per_class_metrics=True,
        progress_bar="tqdm",
        amp_dtype="fp16",
        tensorboard=False,
        wandb=False,
        run_test=False,
        multi_scale=args.multi_scale,
        expanded_scales=args.expanded_scales,
        notes=plan,
    )
    if local_rank == 0:
        best = output / "checkpoint_best_total.pth"
        if not best.is_file():
            raise FileNotFoundError(f"Training ended without best checkpoint: {best}")
        completion = {
            "schema_version": 1,
            "status": "completed",
            "plan": plan,
            "best_checkpoint": str(best),
            "best_checkpoint_sha256": _digest(best, "sha256"),
        }
        (output / "training_result.json").write_text(
            json.dumps(completion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
