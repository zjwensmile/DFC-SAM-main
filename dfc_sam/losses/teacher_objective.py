"""SAM Teacher adaptation objective on the sealed D_M adaptation subset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from dfc_sam.models.teacher_adapter import TeacherOutput

from .mask_losses import per_instance_mask_loss


@dataclass
class TeacherObjective:
    total: Tensor
    instances: int


def compute_teacher_objective(
    output: TeacherOutput,
    batch: dict[str, Any],
    *,
    dice_epsilon: float,
) -> TeacherObjective:
    """Compute BCE-with-logits plus soft Dice; no detector or pseudo loss exists here."""
    if output.mask_logits.ndim != 4 or output.mask_logits.shape[1] != 1:
        raise ValueError("Teacher adaptation requires multimask_output=False")
    supervised_flat = batch["supervised_target_indices"].long()
    lookup = torch.full(
        (int(batch["target_batch"]["cls"].numel()),),
        -1,
        dtype=torch.long,
        device=supervised_flat.device,
    )
    lookup[supervised_flat] = torch.arange(supervised_flat.numel(), device=lookup.device)
    supervised_index = lookup.index_select(0, output.target_flat_index.to(lookup.device))
    if (supervised_index < 0).any():
        raise RuntimeError("Teacher adaptation received a target without an exposed ground-truth mask")
    source_masks = batch["supervised_masks"]
    targets = source_masks.index_select(0, supervised_index.to(source_masks.device)).to(output.mask_logits.device)
    losses = per_instance_mask_loss(output.mask_logits[:, 0], targets, epsilon=dice_epsilon)
    if losses.numel() == 0:
        raise RuntimeError("Teacher adaptation batch contains no mask-supervised instances")
    return TeacherObjective(losses.mean(), int(losses.numel()))
