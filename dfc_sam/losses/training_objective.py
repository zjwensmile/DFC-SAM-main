"""Stage-aware full and mixed DFC-SAM training objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor
from torch.nn import functional as F

from dfc_sam.models.assignment_adapter import AssignmentAdapter
from dfc_sam.models.dfc_sam import DFCStudentOutput
from dfc_sam.pseudo.store import PseudoMaskStore

from .mask_losses import weighted_mask_loss
from .supervision import resolve_mask_supervision
from .total_loss import LossWeights, total_loss


@dataclass
class TrainingObjective:
    total: Tensor
    detection: Tensor
    segmentation: Tensor
    ugca: Tensor
    detection_components: dict[str, Tensor]
    mask_instances: int
    effective_mask_weight: float
    ground_truth_masks: int
    pseudo_masks: int


def compute_training_objective(
    output: DFCStudentOutput,
    batch: dict[str, Any],
    assignment: AssignmentAdapter,
    *,
    stage: str,
    strategy: str,
    pseudo_store: PseudoMaskStore | None,
    weights: LossWeights,
    dice_epsilon: float,
    mask_boundary_loss_weight: float = 0.0,
    mask_boundary_kernel_size: int = 3,
    ugca_class_weights: Tensor | None = None,
    ugca_label_smoothing: float = 0.0,
) -> TrainingObjective:
    """Compute exactly the losses allowed by warm-up or joint training."""
    stage_value = str(getattr(stage, "value", stage))
    if stage_value not in {"warmup", "joint"}:
        raise ValueError(f"Unsupported training stage: {stage_value}")
    reference = output.mask_logits
    zero = reference.sum() * 0.0
    detection_components: dict[str, Tensor] = {}
    if stage_value == "joint":
        detection, detection_components = assignment.detection_loss(
            output.detector.raw_one2many,
            output.detector.raw_one2one,
            batch["target_batch"],
        )
    else:
        detection = zero

    resolved = resolve_mask_supervision(
        output.selected,
        batch,
        strategy=strategy,
        pseudo_store=pseudo_store,
        device=reference.device,
    )
    if resolved.query_indices.numel():
        segmentation = weighted_mask_loss(
            output.mask_logits.index_select(0, resolved.query_indices),
            resolved.targets,
            resolved.weights,
            epsilon=dice_epsilon,
            boundary_weight=mask_boundary_loss_weight,
            boundary_kernel_size=mask_boundary_kernel_size,
        )
    else:
        segmentation = zero

    if stage_value == "joint":
        if output.ugca is None:
            raise RuntimeError("Joint training requires an enabled UGCA output")
        if output.selected.target_index.numel():
            classes = batch["target_batch"]["cls"].to(output.refined_logits.device).long()
            matched_classes = classes.index_select(0, output.selected.target_index)
            ugca_loss = F.cross_entropy(
                output.refined_logits.float(),
                matched_classes,
                weight=None if ugca_class_weights is None else ugca_class_weights.to(output.refined_logits.device),
                label_smoothing=float(ugca_label_smoothing),
            )
        else:
            ugca_loss = zero
    else:
        if output.ugca is not None:
            raise RuntimeError("Warm-up must not execute UGCA")
        ugca_loss = zero

    effective_weights = LossWeights(
        detection=weights.detection if stage_value == "joint" else 0.0,
        segmentation=weights.segmentation,
        ugca=weights.ugca if stage_value == "joint" else 0.0,
    )
    objective = total_loss(detection, segmentation, ugca_loss, effective_weights)
    return TrainingObjective(
        total=objective,
        detection=detection,
        segmentation=segmentation,
        ugca=ugca_loss,
        detection_components=detection_components,
        mask_instances=int(resolved.query_indices.numel()),
        effective_mask_weight=float(resolved.weights.detach().sum().cpu()),
        ground_truth_masks=resolved.sources.count("ground_truth"),
        pseudo_masks=resolved.sources.count("pseudo"),
    )
