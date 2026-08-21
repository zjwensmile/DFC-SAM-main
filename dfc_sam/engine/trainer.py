"""Single-GPU deterministic epoch loops for Teacher and DFC-SAM students."""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from dfc_sam.data.collate import collate_pannuke
from dfc_sam.losses.mask_losses import per_instance_mask_loss
from dfc_sam.losses.supervision import ResolvedMaskSupervision, resolve_mask_supervision
from dfc_sam.losses.teacher_objective import compute_teacher_objective
from dfc_sam.losses.total_loss import LossWeights
from dfc_sam.losses.training_objective import compute_training_objective
from dfc_sam.models.teacher_adapter import BoxPromptedTeacher
from dfc_sam.pseudo.store import PseudoMaskStore
from dfc_sam.utils.hashing import atomic_write_json

from .sampling import DistributedEpochShuffleSampler, EpochShuffleSampler
from .stages import Stage, apply_stage_modes


@dataclass
class EpochSummary:
    epoch: int
    micro_batches: int
    optimizer_updates: int
    skipped_nonfinite_updates: int
    skipped_no_gradient_batches: int
    images: int
    instances: int
    total_loss: float
    detection_loss: float
    segmentation_loss: float
    ugca_loss: float
    learning_rates: dict[str, float]
    completed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _ChunkedBatchResult:
    total: torch.Tensor
    detection: torch.Tensor
    segmentation: torch.Tensor
    ugca: torch.Tensor
    mask_instances: int
    gradient_produced: bool


def seed_worker(worker_id: int) -> None:
    """Derive NumPy/Python worker seeds from PyTorch's deterministic worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _set_dataset_epoch(dataset: Dataset, epoch: int) -> None:
    current: Any = dataset
    visited: set[int] = set()
    while id(current) not in visited:
        visited.add(id(current))
        setter = getattr(current, "set_epoch", None)
        if setter is not None:
            setter(epoch)
            return
        nested = getattr(current, "dataset", None)
        if nested is None:
            return
        current = nested


def build_epoch_loader(
    dataset: Dataset,
    *,
    seed: int,
    epoch: int,
    start_index: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, EpochShuffleSampler]:
    """Create one explicit epoch permutation; no hidden DataLoader shuffle state."""
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    _set_dataset_epoch(dataset, epoch)
    sampler = (
        DistributedEpochShuffleSampler(
            dataset,
            seed=seed,
            epoch=epoch,
            start_index=start_index,
            rank=rank,
            world_size=world_size,
        )
        if world_size > 1
        else EpochShuffleSampler(dataset, seed=seed, epoch=epoch, start_index=start_index)
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(epoch) + 1_000_003 * int(rank))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=bool(persistent_workers and num_workers > 0),
        collate_fn=collate_pannuke,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )
    return loader, sampler


def move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
    *,
    include_mask_supervision: bool = True,
) -> dict[str, Any]:
    """Move tensor fields while preserving identities, tissues, and geometry objects."""
    result = dict(batch)
    keys = [
        "yolo_images",
        "sam_resized_images",
        "target_boxes_original_xyxy",
    ]
    if include_mask_supervision:
        keys.extend(
            (
                "target_instance_indices",
                "mask_supervised",
                "supervised_target_indices",
                "supervised_masks",
            )
        )
    for key in keys:
        result[key] = batch[key].to(device, non_blocking=True)
    result["target_batch"] = {key: value.to(device, non_blocking=True) for key, value in batch["target_batch"].items()}
    return result


def build_grad_scaler(train_config: dict[str, Any], device: torch.device) -> torch.amp.GradScaler:
    enabled = bool(train_config["amp"]) and device.type == "cuda"
    if enabled and str(train_config.get("amp_dtype", "fp16")).lower() != "fp16":
        raise ValueError("Tesla V100 formal runs support amp_dtype=fp16 only")
    initial_scale = float(train_config.get("amp_initial_scale", 1024.0))
    if initial_scale <= 0:
        raise ValueError("amp_initial_scale must be positive")
    return torch.amp.GradScaler(
        device.type,
        enabled=enabled,
        init_scale=initial_scale,
        growth_interval=2000,
    )


def _all_finite(parameters: list[nn.Parameter]) -> bool:
    return all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in parameters)


def _multiply_gradients(parameters: list[nn.Parameter], factor: float) -> None:
    if factor == 1.0:
        return
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(factor)


def _any_rank_has_gradient(local_has_gradient: bool, device: torch.device) -> bool:
    """Make the accumulation-boundary decision collectively on every rank."""
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return bool(local_has_gradient)
    flag = torch.tensor(int(local_has_gradient), dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def _materialize_synchronized_gradients(parameters: list[nn.Parameter]) -> list[nn.Parameter]:
    """Return globally active parameters, filling missing rank-local gradients with zeros."""
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return [parameter for parameter in parameters if parameter.grad is not None]
    if not parameters:
        return []
    device = parameters[0].device
    local_presence = torch.tensor(
        [parameter.grad is not None for parameter in parameters],
        dtype=torch.uint8,
        device=device,
    )
    dist.all_reduce(local_presence, op=dist.ReduceOp.MAX)
    active: list[nn.Parameter] = []
    for parameter, globally_present in zip(parameters, local_presence.tolist(), strict=True):
        if not globally_present:
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        active.append(parameter)
    return active


def _all_reduce_materialized_gradients(
    parameters: list[nn.Parameter],
    *,
    bucket_bytes: int = 25 << 20,
) -> None:
    """Average already-materialized gradients in bounded coalesced buckets."""
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return
    if bucket_bytes < 1:
        raise ValueError("Distributed gradient bucket size must be positive")

    world_size = dist.get_world_size()
    bucket: list[nn.Parameter] = []
    size = 0

    def flush() -> None:
        nonlocal bucket, size
        if not bucket:
            return
        flattened = torch.cat([parameter.grad.reshape(-1) for parameter in bucket])
        dist.all_reduce(flattened, op=dist.ReduceOp.SUM)
        flattened.div_(world_size)
        offset = 0
        for parameter in bucket:
            count = parameter.numel()
            parameter.grad.copy_(flattened[offset : offset + count].view_as(parameter))
            offset += count
        bucket = []
        size = 0

    for parameter in parameters:
        parameter_bytes = parameter.numel() * parameter.element_size()
        if bucket and size + parameter_bytes > bucket_bytes:
            flush()
        bucket.append(parameter)
        size += parameter_bytes
    flush()


def _synchronize_gradients(parameters: list[nn.Parameter], *, bucket_bytes: int = 25 << 20) -> None:
    """Materialize and average gradients across ranks in bounded coalesced buckets."""
    if bucket_bytes < 1:
        raise ValueError("Distributed gradient bucket size must be positive")
    active = _materialize_synchronized_gradients(parameters)
    _all_reduce_materialized_gradients(active, bucket_bytes=bucket_bytes)


def _optimizer_boundary(
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    parameters: list[nn.Parameter],
    accumulated: int,
    target_accumulation: int,
) -> bool:
    """Apply one update and return False when GradScaler safely skipped it."""
    active = _materialize_synchronized_gradients(parameters)
    if not active:
        raise RuntimeError("Optimizer boundary was entered without a gradient on any rank")
    if scaler.is_enabled():
        # GradScaler initializes lazily on the first scale() call. A rank can
        # legitimately reach its first collective update with only zero-filled
        # remote gradients, so initialize its scale before calling unscale_().
        scaler.scale(torch.zeros((), device=active[0].device))
    scaler.unscale_(optimizer)
    _multiply_gradients(active, target_accumulation / accumulated)
    _all_reduce_materialized_gradients(active)
    finite = _all_finite(active)
    if not finite and not scaler.is_enabled():
        raise FloatingPointError("Non-finite gradient with AMP disabled")
    if finite:
        scaler.step(optimizer)
        scaler.update()
    else:
        # `unscale_` records found-inf rank-locally. A non-finite gradient may
        # originate on a different rank, so every rank must skip and back off.
        scaler.update(new_scale=float(scaler.get_scale()) * 0.5)
    optimizer.zero_grad(set_to_none=True)
    if finite:
        scheduler.step()
    return finite


def reduce_epoch_summary(summary: EpochSummary, device: torch.device) -> EpochSummary:
    """Combine rank-local counters and weighted losses into one split summary."""
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return summary
    weighted = torch.tensor(
        [
            summary.micro_batches,
            summary.images,
            summary.instances,
            summary.skipped_no_gradient_batches,
            summary.total_loss * summary.micro_batches,
            summary.detection_loss * summary.micro_batches,
            summary.segmentation_loss * summary.micro_batches,
            summary.ugca_loss * summary.micro_batches,
        ],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(weighted, op=dist.ReduceOp.SUM)
    local_control = torch.tensor(
        [summary.optimizer_updates, summary.skipped_nonfinite_updates, int(summary.completed)],
        dtype=torch.int64,
        device=device,
    )
    control_min = local_control.clone()
    control_max = local_control.clone()
    dist.all_reduce(control_min, op=dist.ReduceOp.MIN)
    dist.all_reduce(control_max, op=dist.ReduceOp.MAX)
    if not torch.equal(control_min[:2], control_max[:2]):
        raise RuntimeError("Distributed ranks disagreed on optimizer update counts")
    batches = int(weighted[0].item())
    if batches < 1:
        raise RuntimeError("Distributed epoch summary has no batches")
    return EpochSummary(
        epoch=summary.epoch,
        micro_batches=batches,
        optimizer_updates=int(control_min[0].item()),
        skipped_nonfinite_updates=int(control_min[1].item()),
        skipped_no_gradient_batches=int(weighted[3].item()),
        images=int(weighted[1].item()),
        instances=int(weighted[2].item()),
        total_loss=float(weighted[4].item() / batches),
        detection_loss=float(weighted[5].item() / batches),
        segmentation_loss=float(weighted[6].item() / batches),
        ugca_loss=float(weighted[7].item() / batches),
        learning_rates=summary.learning_rates,
        completed=bool(control_min[2].item()),
    )


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        enabled=enabled,
    )


def _expected_no_gradient_student_batch(objective: Any, stage: Stage) -> bool:
    """Identify a valid Warmup batch for which native assignment exposed no mask query."""
    if objective.total.requires_grad:
        return False
    if stage is Stage.WARMUP and objective.mask_instances == 0:
        return True
    raise RuntimeError(
        "Student objective unexpectedly has no trainable graph: "
        f"stage={stage.value}, mask_instances={objective.mask_instances}"
    )


def _require_finite_scalar(value: torch.Tensor, name: str) -> None:
    if value.numel() != 1 or not bool(torch.isfinite(value)):
        raise FloatingPointError(f"Non-finite student {name}")


def _quality_ranking_loss(
    quality_logits: torch.Tensor,
    quality_targets: torch.Tensor,
    batch_indices: torch.Tensor,
    *,
    min_target_gap: float,
    margin: float,
) -> torch.Tensor:
    """Rank higher-IoU candidates above lower-quality candidates per image.

    The existing BCE objective calibrates each candidate independently.  This
    term only compares candidates belonging to the same image and only when
    their detached IoU/negative targets differ by a meaningful gap.  Weighting
    by that gap prioritizes true-vs-hard-negative and high-vs-low-IoU pairs
    without treating near-identical mask qualities as ordered ground truth.
    """
    if quality_logits.ndim != 1 or quality_targets.shape != quality_logits.shape:
        raise ValueError("quality logits and targets must be same-length vectors")
    if batch_indices.shape != quality_logits.shape:
        raise ValueError("quality ranking batch_indices must identify every logit")
    if min_target_gap <= 0.0 or margin < 0.0:
        raise ValueError("quality ranking requires positive target gap and non-negative margin")
    zero = quality_logits.sum() * 0.0
    per_image: list[torch.Tensor] = []
    for batch_index in batch_indices.unique(sorted=True):
        members = batch_indices == batch_index
        scores = quality_logits[members].float()
        targets = quality_targets[members].detach().float()
        if scores.numel() < 2:
            continue
        target_gap = targets[:, None] - targets[None, :]
        valid = target_gap >= min_target_gap
        if not bool(valid.any()):
            continue
        score_gap = scores[:, None] - scores[None, :]
        pair_losses = F.softplus(float(margin) - score_gap)
        weights = target_gap[valid]
        per_image.append((pair_losses[valid] * weights).sum() / weights.sum().clamp_min(1.0e-6))
    return torch.stack(per_image).mean() if per_image else zero


def _is_primary_rank() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def _write_live_student_progress(
    path: Path | None,
    *,
    stage: Stage,
    epoch: int,
    images: int,
    expected_images: int,
    micro_batches: int,
    expected_micro_batches: int,
    totals: dict[str, float],
    started_at: float,
) -> None:
    """Publish rank-zero local progress without synchronizing distributed work."""
    if path is None or not _is_primary_rank():
        return
    fraction = min(1.0, images / max(expected_images, 1))
    elapsed = max(0.0, time.monotonic() - started_at)
    eta = elapsed * (1.0 - fraction) / fraction if fraction > 0.0 else None
    divisor = max(micro_batches, 1)
    atomic_write_json(
        path,
        {
            "phase": "train",
            "stage": stage.value,
            "epoch_zero_based": int(epoch),
            "epoch_display": int(epoch) + 1,
            "completed_images_rank0": int(images),
            "expected_images_rank0": int(expected_images),
            "completed_micro_batches_rank0": int(micro_batches),
            "expected_micro_batches_rank0": int(expected_micro_batches),
            "fraction": float(fraction),
            "elapsed_seconds": float(elapsed),
            "eta_seconds": None if eta is None else float(eta),
            "mean_loss_so_far": {key: float(value / divisor) for key, value in totals.items()},
            "updated_at_unix": time.time(),
        },
    )


def _backward_student_batch_by_instance_chunks(
    model: nn.Module,
    batch: dict[str, Any],
    *,
    stage: Stage,
    strategy: str,
    pseudo_store: PseudoMaskStore | None,
    scaler: torch.amp.GradScaler,
    accumulation_target: int,
    weights: LossWeights,
    dice_epsilon: float,
    mask_boundary_loss_weight: float = 0.0,
    mask_boundary_kernel_size: int = 3,
    device: torch.device,
    amp_enabled: bool,
    ugca_class_weights: torch.Tensor | None = None,
    ugca_label_smoothing: float = 0.0,
    ugca_loss_normalization: str = "legacy_batch",
    ugca_gate_loss_weight: float = 0.0,
    ugca_preservation_loss_weight: float = 0.0,
    ugca_quality_loss_weight: float = 0.0,
    ugca_quality_mask_threshold: float = 0.5,
    ugca_quality_ranking_loss_weight: float = 0.0,
    ugca_quality_ranking_min_target_gap: float = 0.1,
    ugca_quality_ranking_margin: float = 0.0,
) -> _ChunkedBatchResult:
    """Backpropagate an exact full-image objective without retaining every SAM graph."""
    with _autocast(device, amp_enabled):
        context = model.prepare_training_context(
            batch["yolo_images"],
            batch["sam_resized_images"],
            target_batch=batch["target_batch"],
            stage=stage,
        )
        reference = context.detector.base_logits
        zero = reference.sum() * 0.0
        if stage is Stage.JOINT and weights.detection != 0.0:
            detection, _ = model.assignment.detection_loss(
                context.detector.raw_one2many,
                context.detector.raw_one2one,
                batch["target_batch"],
            )
        else:
            detection = zero
    _require_finite_scalar(detection, "detection objective")

    if weights.segmentation != 0.0:
        resolved = resolve_mask_supervision(
            context.selected,
            batch,
            strategy=strategy,
            pseudo_store=pseudo_store,
            device=device,
        )
    else:
        resolved = ResolvedMaskSupervision(
            query_indices=torch.empty(0, dtype=torch.long, device=device),
            targets=torch.empty((0, 256, 256), dtype=torch.bool, device=device),
            weights=torch.empty(0, dtype=torch.float32, device=device),
            sources=[],
        )
    effective_mass = resolved.weights.sum()
    segmentation = torch.zeros((), dtype=torch.float32, device=device)
    ugca_loss = torch.zeros((), dtype=torch.float32, device=device)
    selected_count = int(context.selected.query_index.numel())
    selected_positive = context.selected.target_index >= 0
    positive_target_indices = context.selected.target_index[selected_positive]
    positive_batches = context.selected.batch_index[selected_positive].long()
    selected_classes = batch["target_batch"]["cls"].long().index_select(0, positive_target_indices)
    positive_count = int(selected_classes.numel())
    if stage is Stage.JOINT and positive_count:
        ugca_denominator = (
            ugca_class_weights.index_select(0, selected_classes).sum()
            if ugca_class_weights is not None
            else selected_classes.new_tensor(float(positive_count), dtype=torch.float32)
        ).clamp_min(torch.finfo(torch.float32).eps)
    else:
        ugca_denominator = torch.tensor(1.0, device=device)
    image_count = int(batch["yolo_images"].shape[0])
    selected_batches = context.selected.batch_index.long()
    per_image_class_mass = torch.zeros(image_count, dtype=torch.float32, device=device)
    per_image_instance_count = torch.zeros(image_count, dtype=torch.float32, device=device)
    per_image_quality_count = torch.zeros(image_count, dtype=torch.float32, device=device)
    per_image_correct_count = torch.zeros(image_count, dtype=torch.float32, device=device)
    selected_base_correct = torch.empty(0, dtype=torch.bool, device=device)
    if positive_count and ugca_preservation_loss_weight:
        selected_base_logits = context.detector.base_logits[
            context.selected.batch_index[selected_positive],
            context.selected.query_index[selected_positive],
        ]
        selected_base_correct = selected_base_logits.argmax(dim=-1) == selected_classes
    if positive_count and ugca_loss_normalization == "per_image":
        class_mass = (
            ugca_class_weights.index_select(0, selected_classes)
            if ugca_class_weights is not None
            else torch.ones(positive_count, dtype=torch.float32, device=device)
        )
        per_image_class_mass.scatter_add_(0, positive_batches, class_mass)
        per_image_instance_count.scatter_add_(
            0,
            positive_batches,
            torch.ones(positive_count, dtype=torch.float32, device=device),
        )
        if ugca_preservation_loss_weight:
            per_image_correct_count.scatter_add_(
                0,
                positive_batches,
                selected_base_correct.float(),
            )
    if selected_count and ugca_loss_normalization == "per_image":
        per_image_quality_count.scatter_add_(
            0,
            selected_batches,
            torch.ones(selected_count, dtype=torch.float32, device=device),
        )
    target_to_supervised_mask = torch.empty(0, dtype=torch.long, device=device)
    if ugca_quality_loss_weight:
        target_count = int(batch["target_batch"]["cls"].numel())
        target_to_supervised_mask = torch.full((target_count,), -1, dtype=torch.long, device=device)
        supervised_target_indices = batch["supervised_target_indices"].long()
        target_to_supervised_mask[supervised_target_indices] = torch.arange(
            supervised_target_indices.numel(), device=device, dtype=torch.long
        )
        if positive_count and bool((target_to_supervised_mask[positive_target_indices] < 0).any()):
            raise RuntimeError("Quality-aware UGCA requires masks for every matched positive")
    if ugca_loss_normalization not in {"legacy_batch", "per_image"}:
        raise ValueError(f"Unknown UGCA loss normalization: {ugca_loss_normalization}")
    gradient_produced = False

    for start in range(0, selected_count, model.instance_chunk_size):
        stop = min(start + model.instance_chunk_size, selected_count)
        with _autocast(device, amp_enabled):
            output = model.decode_training_chunk(context, start, stop)
            in_chunk = (resolved.query_indices >= start) & (resolved.query_indices < stop)
            if weights.segmentation != 0.0 and bool(in_chunk.any()):
                local_query_indices = resolved.query_indices[in_chunk] - start
                per_instance = per_instance_mask_loss(
                    output.mask_logits.index_select(0, local_query_indices),
                    resolved.targets[in_chunk],
                    epsilon=dice_epsilon,
                    boundary_weight=mask_boundary_loss_weight,
                    boundary_kernel_size=mask_boundary_kernel_size,
                )
                segmentation_piece = (per_instance * resolved.weights[in_chunk]).sum() / (effective_mass + dice_epsilon)
            else:
                segmentation_piece = output.mask_logits.sum() * 0.0

            if stage is Stage.JOINT:
                if output.ugca is None:
                    raise RuntimeError("Joint chunk did not produce UGCA output")
                classes = batch["target_batch"]["cls"].long()
                chunk_positive = output.selected.target_index >= 0
                chunk_target_indices = output.selected.target_index[chunk_positive]
                chunk_batches = output.selected.batch_index[chunk_positive].long()
                matched_classes = classes.index_select(0, chunk_target_indices)
                classification_piece = output.refined_logits.sum() * 0.0
                if bool(chunk_positive.any()):
                    per_instance_ce = F.cross_entropy(
                        output.refined_logits[chunk_positive].float(),
                        matched_classes,
                        weight=ugca_class_weights,
                        label_smoothing=float(ugca_label_smoothing),
                        reduction="none",
                    )
                    if ugca_loss_normalization == "per_image":
                        ce_denominators = per_image_class_mass.index_select(0, chunk_batches).clamp_min(
                            torch.finfo(torch.float32).eps
                        )
                        classification_piece = (per_instance_ce / ce_denominators).sum() / image_count
                    else:
                        classification_piece = per_instance_ce.sum() / ugca_denominator

                gate_piece = output.refined_logits.sum() * 0.0
                if ugca_gate_loss_weight and bool(chunk_positive.any()):
                    if output.ugca.gate_logits is None:
                        raise RuntimeError("UGCA gate supervision requires gate_logits")
                    base_correct = output.base_logits[chunk_positive].argmax(dim=-1) == matched_classes
                    gate_targets = (~base_correct).float()
                    per_instance_gate = F.binary_cross_entropy_with_logits(
                        output.ugca.gate_logits[chunk_positive].float(),
                        gate_targets,
                        reduction="none",
                    )
                    if ugca_loss_normalization == "per_image":
                        gate_denominators = per_image_instance_count.index_select(0, chunk_batches).clamp_min(1.0)
                        gate_piece = (per_instance_gate / gate_denominators).sum() / image_count
                    else:
                        gate_piece = per_instance_gate.sum() / max(selected_count, 1)

                preservation_piece = output.refined_logits.sum() * 0.0
                if ugca_preservation_loss_weight and bool(chunk_positive.any()):
                    detector = getattr(model, "detector", None)
                    if getattr(detector, "class_probability_mode", "softmax") == "sigmoid":
                        base_evidence = output.base_logits.float().sigmoid()
                        base_probability = base_evidence / base_evidence.sum(
                            dim=-1,
                            keepdim=True,
                        ).clamp_min(torch.finfo(base_evidence.dtype).eps)
                    else:
                        base_probability = output.base_logits.float().softmax(dim=-1)
                    base_probability = base_probability.detach()
                    per_instance_kl = F.kl_div(
                        output.refined_logits.float().log_softmax(dim=-1),
                        base_probability,
                        reduction="none",
                    ).sum(dim=-1)
                    base_correct = output.base_logits[chunk_positive].argmax(dim=-1) == matched_classes
                    if bool(base_correct.any()):
                        if ugca_loss_normalization == "per_image":
                            preserve_denominators = per_image_correct_count.index_select(0, chunk_batches).clamp_min(
                                1.0
                            )
                            preservation_piece = (
                                per_instance_kl[chunk_positive] * base_correct.float() / preserve_denominators
                            ).sum() / image_count
                        else:
                            preservation_piece = (
                                per_instance_kl[chunk_positive] * base_correct.float()
                            ).sum() / selected_base_correct.float().sum().clamp_min(1.0)
                quality_piece = output.refined_logits.sum() * 0.0
                ranking_piece = output.refined_logits.sum() * 0.0
                if ugca_quality_loss_weight or ugca_quality_ranking_loss_weight:
                    if output.ugca.quality_logits is None:
                        raise RuntimeError("Quality-aware supervision requires quality_logits")
                    quality_targets = torch.zeros(
                        output.selected.query_index.numel(),
                        dtype=torch.float32,
                        device=device,
                    )
                    if bool(chunk_positive.any()):
                        mask_indices = target_to_supervised_mask.index_select(0, chunk_target_indices)
                        target_masks = batch["supervised_masks"].index_select(0, mask_indices).bool()
                        predicted_masks = (
                            output.mask_logits[chunk_positive, 0].float().detach().sigmoid()
                            >= ugca_quality_mask_threshold
                        )
                        intersection = (predicted_masks & target_masks).sum(dim=(-2, -1)).float()
                        union = (predicted_masks | target_masks).sum(dim=(-2, -1)).float()
                        quality_targets[chunk_positive] = intersection / union.clamp_min(1.0)
                    per_instance_quality = F.binary_cross_entropy_with_logits(
                        output.ugca.quality_logits.float(),
                        quality_targets,
                        reduction="none",
                    )
                    if ugca_loss_normalization == "per_image":
                        quality_batches = output.selected.batch_index.long()
                        quality_denominators = per_image_quality_count.index_select(
                            0, quality_batches
                        ).clamp_min(1.0)
                        quality_piece = (per_instance_quality / quality_denominators).sum() / image_count
                    else:
                        quality_piece = per_instance_quality.sum() / max(selected_count, 1)
                    if ugca_quality_ranking_loss_weight:
                        ranking_piece = _quality_ranking_loss(
                            output.ugca.quality_logits,
                            quality_targets,
                            output.selected.batch_index.long(),
                            min_target_gap=ugca_quality_ranking_min_target_gap,
                            margin=ugca_quality_ranking_margin,
                        )

                ugca_piece = (
                    classification_piece
                    + ugca_gate_loss_weight * gate_piece
                    + ugca_preservation_loss_weight * preservation_piece
                    + ugca_quality_loss_weight * quality_piece
                    + ugca_quality_ranking_loss_weight * ranking_piece
                )
            else:
                ugca_piece = output.mask_logits.sum() * 0.0

            chunk_total = weights.segmentation * segmentation_piece + (
                weights.ugca * ugca_piece if stage is Stage.JOINT else 0.0
            )
        _require_finite_scalar(segmentation_piece, "segmentation chunk objective")
        _require_finite_scalar(ugca_piece, "UGCA chunk objective")
        _require_finite_scalar(chunk_total, "chunk objective")
        segmentation = segmentation + segmentation_piece.detach().float()
        ugca_loss = ugca_loss + ugca_piece.detach().float()

        has_chunk_signal = bool(in_chunk.any()) or stage is Stage.JOINT
        if has_chunk_signal:
            scaler.scale(chunk_total / accumulation_target).backward(
                retain_graph=stage is Stage.JOINT and weights.detection != 0.0,
            )
            gradient_produced = True
        del output, segmentation_piece, ugca_piece, chunk_total

    if stage is Stage.JOINT and weights.detection != 0.0:
        detection_total = weights.detection * detection
        _require_finite_scalar(detection_total, "weighted detection objective")
        scaler.scale(detection_total / accumulation_target).backward()
        gradient_produced = True

    detached_detection = detection.detach().float()
    detached_total = (
        weights.detection * detached_detection + weights.segmentation * segmentation + weights.ugca * ugca_loss
    )
    _require_finite_scalar(detached_total, "forward objective")
    return _ChunkedBatchResult(
        total=detached_total,
        detection=detached_detection,
        segmentation=segmentation,
        ugca=ugca_loss,
        mask_instances=int(resolved.query_indices.numel()),
        gradient_produced=gradient_produced,
    )


def train_student_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    epoch: int,
    stage: Stage,
    strategy: str,
    pseudo_store: PseudoMaskStore | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    config: dict[str, Any],
    device: torch.device,
    stop_requested: Callable[[], bool] | None = None,
    live_progress_path: Path | None = None,
) -> EpochSummary:
    """Run one complete warm-up or joint epoch with fixed pseudo supervision."""
    if stage not in {Stage.WARMUP, Stage.JOINT}:
        raise ValueError("Student epoch stage must be warmup or joint")
    model.train()
    apply_stage_modes(
        stage,
        detector=model.detector,
        bridge=model.feature_bridge,
        student_sam=model.student_sam,
        ugca=model.ugca,
        joint_detector_scope=str(config["train"].get("joint_detector_scope", "all")),
        joint_training_mode=str(config["train"].get("joint_training_mode", "full")),
    )
    accumulation_target = int(config["train"]["gradient_accumulation"])
    max_nonfinite = int(config["train"].get("amp_backoff_retries", 5))
    weights = LossWeights(
        detection=float(config["loss"]["lambda_det"]),
        segmentation=float(config["loss"]["lambda_seg"]),
        ugca=float(config["loss"]["lambda_ugca"]),
    )
    configured_class_weights = config["loss"].get("ugca_class_weights")
    ugca_class_weights = None
    if configured_class_weights is not None:
        ugca_class_weights = torch.tensor(configured_class_weights, dtype=torch.float32, device=device)
    ugca_label_smoothing = float(config["loss"].get("ugca_label_smoothing", 0.0))
    ugca_loss_normalization = str(config["loss"].get("ugca_loss_normalization", "legacy_batch"))
    ugca_gate_loss_weight = float(config["loss"].get("ugca_gate_loss_weight", 0.0))
    ugca_preservation_loss_weight = float(config["loss"].get("ugca_preservation_loss_weight", 0.0))
    ugca_quality_loss_weight = float(config["loss"].get("ugca_quality_loss_weight", 0.0))
    ugca_quality_mask_threshold = float(config["loss"].get("ugca_quality_mask_threshold", 0.5))
    ugca_quality_ranking_loss_weight = float(config["loss"].get("ugca_quality_ranking_loss_weight", 0.0))
    ugca_quality_ranking_min_target_gap = float(
        config["loss"].get("ugca_quality_ranking_min_target_gap", 0.1)
    )
    ugca_quality_ranking_margin = float(config["loss"].get("ugca_quality_ranking_margin", 0.0))
    parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"] if parameter.requires_grad
    ]
    optimizer.zero_grad(set_to_none=True)
    totals = dict(total=0.0, detection=0.0, segmentation=0.0, ugca=0.0)
    images = instances = updates = skipped = accumulated = micro_batches = 0
    gradient_batches = skipped_no_gradient = 0
    amp_enabled = scaler.is_enabled()
    interrupted = False
    expected_images = len(loader.sampler) if loader.sampler is not None else len(loader.dataset)
    try:
        expected_micro_batches = len(loader)
    except TypeError:
        # Minimal test/rehearsal iterables need not implement __len__. Formal
        # DataLoaders do, while the image progress remains valid either way.
        expected_micro_batches = 0
    started_at = time.monotonic()

    for raw_batch in loader:
        batch = move_batch_to_device(
            raw_batch,
            device,
            include_mask_supervision=weights.segmentation != 0.0 or ugca_quality_loss_weight != 0.0,
        )
        supports_chunked_backward = all(
            hasattr(model, name)
            for name in ("prepare_training_context", "decode_training_chunk", "instance_chunk_size")
        )
        if supports_chunked_backward:
            objective = _backward_student_batch_by_instance_chunks(
                model,
                batch,
                stage=stage,
                strategy=strategy,
                pseudo_store=pseudo_store,
                scaler=scaler,
                accumulation_target=accumulation_target,
                weights=weights,
                dice_epsilon=float(config["loss"]["dice_epsilon"]),
                mask_boundary_loss_weight=float(config["loss"].get("mask_boundary_loss_weight", 0.0)),
                mask_boundary_kernel_size=int(config["loss"].get("mask_boundary_kernel_size", 3)),
                ugca_class_weights=ugca_class_weights,
                ugca_label_smoothing=ugca_label_smoothing,
                ugca_loss_normalization=ugca_loss_normalization,
                ugca_gate_loss_weight=ugca_gate_loss_weight,
                ugca_preservation_loss_weight=ugca_preservation_loss_weight,
                ugca_quality_loss_weight=ugca_quality_loss_weight,
                ugca_quality_mask_threshold=ugca_quality_mask_threshold,
                ugca_quality_ranking_loss_weight=ugca_quality_ranking_loss_weight,
                ugca_quality_ranking_min_target_gap=ugca_quality_ranking_min_target_gap,
                ugca_quality_ranking_margin=ugca_quality_ranking_margin,
                device=device,
                amp_enabled=amp_enabled,
            )
            gradient_was_produced = objective.gradient_produced
        else:
            with _autocast(device, amp_enabled):
                output = model(
                    batch["yolo_images"],
                    batch["sam_resized_images"],
                    target_batch=batch["target_batch"],
                    stage=stage,
                )
                objective = compute_training_objective(
                    output,
                    batch,
                    model.assignment,
                    stage=stage,
                    strategy=strategy,
                    pseudo_store=pseudo_store,
                    weights=weights,
                    dice_epsilon=float(config["loss"]["dice_epsilon"]),
                    mask_boundary_loss_weight=float(config["loss"].get("mask_boundary_loss_weight", 0.0)),
                    mask_boundary_kernel_size=int(config["loss"].get("mask_boundary_kernel_size", 3)),
                    ugca_class_weights=ugca_class_weights,
                    ugca_label_smoothing=ugca_label_smoothing,
                )
            if not bool(torch.isfinite(objective.total)):
                raise FloatingPointError("Non-finite student forward objective")
            if _expected_no_gradient_student_batch(objective, stage):
                gradient_was_produced = False
            else:
                scaler.scale(objective.total / accumulation_target).backward()
                gradient_was_produced = True
        micro_batches += 1
        accumulated += 1
        images += len(batch["image_ids"])
        instances += int(batch["target_batch"]["cls"].numel())
        totals["total"] += float(objective.total.detach().float().cpu())
        totals["detection"] += float(objective.detection.detach().float().cpu())
        totals["segmentation"] += float(objective.segmentation.detach().float().cpu())
        totals["ugca"] += float(objective.ugca.detach().float().cpu())
        _write_live_student_progress(
            live_progress_path,
            stage=stage,
            epoch=epoch,
            images=images,
            expected_images=expected_images,
            micro_batches=micro_batches,
            expected_micro_batches=expected_micro_batches,
            totals=totals,
            started_at=started_at,
        )
        if not gradient_was_produced:
            skipped_no_gradient += 1
        else:
            gradient_batches += 1
        if accumulated == accumulation_target:
            if _any_rank_has_gradient(bool(gradient_batches), parameters[0].device):
                finite = _optimizer_boundary(
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    parameters=parameters,
                    accumulated=accumulated,
                    target_accumulation=accumulation_target,
                )
                updates += int(finite)
                skipped += int(not finite)
            else:
                optimizer.zero_grad(set_to_none=True)
            accumulated = 0
            gradient_batches = 0
            if skipped > max_nonfinite:
                raise FloatingPointError("Too many non-finite AMP updates in one epoch")
            if stop_requested is not None and stop_requested() and images < expected_images:
                interrupted = True
                break
    if accumulated:
        if _any_rank_has_gradient(bool(gradient_batches), parameters[0].device):
            finite = _optimizer_boundary(
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                parameters=parameters,
                accumulated=accumulated,
                target_accumulation=accumulation_target,
            )
            updates += int(finite)
            skipped += int(not finite)
        else:
            optimizer.zero_grad(set_to_none=True)
    if micro_batches == 0:
        raise RuntimeError("Training epoch received no batches")
    divisor = float(micro_batches)
    return EpochSummary(
        epoch=epoch,
        micro_batches=micro_batches,
        optimizer_updates=updates,
        skipped_nonfinite_updates=skipped,
        skipped_no_gradient_batches=skipped_no_gradient,
        images=images,
        instances=instances,
        total_loss=totals["total"] / divisor,
        detection_loss=totals["detection"] / divisor,
        segmentation_loss=totals["segmentation"] / divisor,
        ugca_loss=totals["ugca"] / divisor,
        learning_rates={
            str(group.get("name", index)): float(group["lr"]) for index, group in enumerate(optimizer.param_groups)
        },
        completed=not interrupted,
    )


def train_teacher_epoch(
    teacher: BoxPromptedTeacher,
    loader: DataLoader,
    *,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    config: dict[str, Any],
    device: torch.device,
    stop_requested: Callable[[], bool] | None = None,
) -> EpochSummary:
    """Run one Teacher decoder epoch over the sealed adaptation partition."""
    apply_stage_modes(
        Stage.TEACHER,
        detector=nn.Identity(),
        bridge=nn.Identity(),
        student_sam=teacher.sam,
        ugca=nn.Identity(),
        teacher_sam=teacher.sam,
    )
    accumulation_target = int(config["train"]["gradient_accumulation"])
    max_nonfinite = int(config["train"].get("amp_backoff_retries", 5))
    parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"] if parameter.requires_grad
    ]
    optimizer.zero_grad(set_to_none=True)
    total_loss_value = 0.0
    images = instances = updates = skipped = accumulated = micro_batches = 0
    amp_enabled = scaler.is_enabled()
    interrupted = False
    expected_images = len(loader.sampler) if loader.sampler is not None else len(loader.dataset)

    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device)
        with _autocast(device, amp_enabled):
            output = teacher(
                batch["sam_resized_images"],
                batch["target_boxes_original_xyxy"],
                batch["target_batch"]["batch_idx"].long(),
                batch["geometries"],
                multimask_output=False,
            )
            objective = compute_teacher_objective(
                output,
                batch,
                dice_epsilon=float(config["loss"]["dice_epsilon"]),
            )
        if not bool(torch.isfinite(objective.total)):
            raise FloatingPointError("Non-finite Teacher forward objective")
        scaler.scale(objective.total / accumulation_target).backward()
        accumulated += 1
        micro_batches += 1
        images += len(batch["image_ids"])
        instances += objective.instances
        total_loss_value += float(objective.total.detach().float().cpu())
        if accumulated == accumulation_target:
            finite = _optimizer_boundary(
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                parameters=parameters,
                accumulated=accumulated,
                target_accumulation=accumulation_target,
            )
            updates += int(finite)
            skipped += int(not finite)
            accumulated = 0
            if skipped > max_nonfinite:
                raise FloatingPointError("Too many non-finite Teacher AMP updates in one epoch")
            if stop_requested is not None and stop_requested() and images < expected_images:
                interrupted = True
                break
    if accumulated:
        finite = _optimizer_boundary(
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            parameters=parameters,
            accumulated=accumulated,
            target_accumulation=accumulation_target,
        )
        updates += int(finite)
        skipped += int(not finite)
    if micro_batches == 0:
        raise RuntimeError("Teacher epoch received no batches")
    return EpochSummary(
        epoch=epoch,
        micro_batches=micro_batches,
        optimizer_updates=updates,
        skipped_nonfinite_updates=skipped,
        skipped_no_gradient_batches=0,
        images=images,
        instances=instances,
        total_loss=total_loss_value / micro_batches,
        detection_loss=math.nan,
        segmentation_loss=total_loss_value / micro_batches,
        ugca_loss=math.nan,
        learning_rates={
            str(group.get("name", index)): float(group["lr"]) for index, group in enumerate(optimizer.param_groups)
        },
        completed=not interrupted,
    )
