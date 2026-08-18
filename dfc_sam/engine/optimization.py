"""Stage-specific optimizer and scheduler construction."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .stages import Stage


@dataclass(frozen=True)
class OptimizationBundle:
    """Optimizer plus its fully recorded parameter-group summary."""

    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    parameter_groups: tuple[dict[str, Any], ...]


def _trainable(module: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def _group(name: str, parameters: Iterable[nn.Parameter], learning_rate: float) -> dict[str, Any]:
    unique: list[nn.Parameter] = []
    seen: set[int] = set()
    for parameter in parameters:
        identity = id(parameter)
        if identity not in seen:
            seen.add(identity)
            unique.append(parameter)
    if not unique:
        raise ValueError(f"Optimizer group {name!r} has no trainable parameters")
    return {"name": name, "params": unique, "lr": float(learning_rate)}


def build_stage_optimization(
    model: nn.Module,
    stage: Stage,
    train_config: Mapping[str, Any],
    *,
    steps_per_epoch: int,
    teacher_sam: nn.Module | None = None,
) -> OptimizationBundle:
    """Build the exact trainable groups for one DFC-SAM stage.

    The scheduler is stepped once per optimizer update, not once per
    micro-batch. Cosine decay is an explicitly recorded engineering default.
    """
    if str(train_config.get("optimizer", "")).lower() != "adamw":
        raise ValueError("Phase-0 formal runner currently supports optimizer=adamw only")
    if str(train_config.get("scheduler", "cosine")).lower() != "cosine":
        raise ValueError("Phase-0 formal runner currently supports scheduler=cosine only")
    accumulation = int(train_config["gradient_accumulation"])
    if accumulation < 1 or steps_per_epoch < 1:
        raise ValueError("gradient_accumulation and steps_per_epoch must be positive")
    rates = train_config["learning_rates"]

    groups: list[dict[str, Any]]
    if stage is Stage.TEACHER:
        if teacher_sam is None:
            raise ValueError("Teacher optimization requires teacher_sam")
        groups = [_group("teacher_decoder", _trainable(teacher_sam.mask_decoder), rates["teacher_decoder"])]
        epochs = int(train_config["teacher_epochs"])
    elif stage is Stage.WARMUP:
        groups = [
            _group("bridge", _trainable(model.feature_bridge), rates.get("warmup_bridge", rates["bridge"])),
            _group(
                "student_decoder",
                _trainable(model.student_sam.mask_decoder),
                rates.get("warmup_student_decoder", rates["student_decoder"]),
            ),
        ]
        epochs = int(train_config["warmup_epochs"])
    elif stage is Stage.JOINT:
        training_mode = str(train_config.get("joint_training_mode", "full"))
        if training_mode == "ugca_only":
            groups = [
                _group("ugca", _trainable(model.ugca), rates.get("joint_ugca", rates["ugca"]))
            ]
        elif training_mode in {"ugca_ranking", "rf_frozen_sequential"}:
            quality_network = getattr(model.ugca, "quality_network", None)
            if quality_network is None:
                raise ValueError("ugca_ranking Joint requires a quality-aware UGCA")
            quality_parameters = _trainable(quality_network)
            quality_ids = {id(parameter) for parameter in quality_parameters}
            shared_parameters = [
                parameter
                for parameter in _trainable(model.ugca)
                if id(parameter) not in quality_ids
            ]
            groups = [
                _group(
                    "ugca_shared",
                    shared_parameters,
                    rates.get("joint_ugca_shared", rates.get("joint_ugca", rates["ugca"])),
                ),
                _group(
                    "ugca_quality",
                    quality_parameters,
                    rates.get("joint_ugca_quality", rates.get("joint_ugca", rates["ugca"])),
                ),
            ]
        elif training_mode == "controlled":
            groups = [
                _group("bridge", _trainable(model.feature_bridge), rates.get("joint_bridge", rates["bridge"])),
            ]
            decoder_parameters = _trainable(model.student_sam.mask_decoder)
            if decoder_parameters:
                groups.append(
                    _group(
                        "student_decoder",
                        decoder_parameters,
                        rates.get("joint_student_decoder", rates["student_decoder"]),
                    )
                )
        elif training_mode == "controlled_adaptive":
            groups = [
                _group("bridge", _trainable(model.feature_bridge), rates.get("joint_bridge", rates["bridge"])),
            ]
            decoder_parameters = _trainable(model.student_sam.mask_decoder)
            if decoder_parameters:
                groups.append(
                    _group(
                        "student_decoder",
                        decoder_parameters,
                        rates.get("joint_student_decoder", rates["student_decoder"]),
                    )
                )
            quality_network = getattr(model.ugca, "quality_network", None)
            if quality_network is None:
                raise ValueError("controlled_adaptive Joint requires a quality-aware UGCA")
            quality_parameters = _trainable(quality_network)
            quality_ids = {id(parameter) for parameter in quality_parameters}
            shared_parameters = [
                parameter
                for parameter in _trainable(model.ugca)
                if id(parameter) not in quality_ids
            ]
            groups.extend(
                (
                    _group(
                        "ugca_shared",
                        shared_parameters,
                        rates.get("joint_ugca_shared", rates.get("joint_ugca", rates["ugca"])),
                    ),
                    _group(
                        "ugca_quality",
                        quality_parameters,
                        rates.get("joint_ugca_quality", rates.get("joint_ugca", rates["ugca"])),
                    ),
                )
            )
        elif training_mode == "detector_head_adaptive":
            quality_network = getattr(model.ugca, "quality_network", None)
            if quality_network is None:
                raise ValueError("detector_head_adaptive Joint requires a quality-aware UGCA")
            quality_parameters = _trainable(quality_network)
            quality_ids = {id(parameter) for parameter in quality_parameters}
            shared_parameters = [
                parameter
                for parameter in _trainable(model.ugca)
                if id(parameter) not in quality_ids
            ]
            groups = [
                _group(
                    "detector_head",
                    _trainable(model.detector.head),
                    rates.get("joint_detector_head", rates.get("detector_head", rates["detector"])),
                ),
                _group(
                    "ugca_shared",
                    shared_parameters,
                    rates.get("joint_ugca_shared", rates.get("joint_ugca", rates["ugca"])),
                ),
                _group(
                    "ugca_quality",
                    quality_parameters,
                    rates.get("joint_ugca_quality", rates.get("joint_ugca", rates["ugca"])),
                ),
            ]
        elif training_mode == "full":
            detector_scope = str(train_config.get("joint_detector_scope", "all"))
            if detector_scope == "all":
                detector_name = "detector"
                detector_module = model.detector
                detector_rate = rates.get("joint_detector", rates["detector"])
            elif detector_scope == "head":
                detector_name = "detector_head"
                detector_module = model.detector.head
                detector_rate = rates.get("detector_head", rates.get("joint_detector", rates["detector"]))
            else:
                raise ValueError(f"Unknown joint_detector_scope: {detector_scope}")
            groups = [
                _group(detector_name, _trainable(detector_module), detector_rate),
                _group("bridge", _trainable(model.feature_bridge), rates.get("joint_bridge", rates["bridge"])),
            ]
            decoder_parameters = _trainable(model.student_sam.mask_decoder)
            if decoder_parameters:
                groups.append(
                    _group(
                        "student_decoder",
                        decoder_parameters,
                        rates.get("joint_student_decoder", rates["student_decoder"]),
                    )
                )
            groups.append(_group("ugca", _trainable(model.ugca), rates.get("joint_ugca", rates["ugca"])))
        else:
            raise ValueError(f"Unknown joint_training_mode: {training_mode}")
        epochs = int(train_config["joint_epochs"])
    else:
        raise ValueError("Detector Stage I remains in the pinned Ultralytics workflow")
    if epochs < 1:
        raise ValueError(f"{stage.value} epochs must be positive")

    parameter_ids = [id(parameter) for group in groups for parameter in group["params"]]
    if len(parameter_ids) != len(set(parameter_ids)):
        raise RuntimeError("A trainable parameter appears in multiple optimizer groups")
    optimizer = torch.optim.AdamW(groups, weight_decay=float(train_config["weight_decay"]))
    updates_per_epoch = (steps_per_epoch + accumulation - 1) // accumulation
    total_updates = max(1, updates_per_epoch * epochs)
    minimum_ratio = float(train_config.get("minimum_lr_ratio", 0.01))
    if not 0.0 <= minimum_ratio <= 1.0:
        raise ValueError("train.minimum_lr_ratio must be in [0,1]")
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda update: minimum_ratio
        + (1.0 - minimum_ratio)
        * 0.5
        * (1.0 + math.cos(math.pi * min(update, total_updates) / total_updates)),
    )
    summary = tuple(
        {
            "name": str(group["name"]),
            "learning_rate": float(group["lr"]),
            "parameter_tensors": len(group["params"]),
            "parameter_elements": sum(parameter.numel() for parameter in group["params"]),
        }
        for group in groups
    )
    return OptimizationBundle(optimizer, scheduler, summary)
