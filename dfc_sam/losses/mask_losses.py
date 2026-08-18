"""BCE-with-logits plus soft Dice mask objectives."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def _soft_boundary(values: Tensor, kernel_size: int) -> Tensor:
    """Return a differentiable morphological boundary map for ``[N,H,W]`` values."""
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("boundary kernel_size must be an odd integer >= 3")
    padding = kernel_size // 2
    expanded = values[:, None]
    maximum = F.max_pool2d(expanded, kernel_size, stride=1, padding=padding)
    minimum = -F.max_pool2d(-expanded, kernel_size, stride=1, padding=padding)
    return (maximum - minimum)[:, 0]


def per_instance_mask_loss(
    logits: Tensor,
    targets: Tensor,
    epsilon: float = 1.0e-6,
    *,
    boundary_weight: float = 0.0,
    boundary_kernel_size: int = 3,
) -> Tensor:
    """Return one BCE+Dice objective, optionally augmented by boundary Dice."""
    if logits.ndim == 4 and logits.shape[1] == 1:
        logits = logits[:, 0]
    if targets.ndim == 4 and targets.shape[1] == 1:
        targets = targets[:, 0]
    if logits.shape != targets.shape or logits.ndim != 3:
        raise ValueError(f"Expected matching [N,H,W] logits and targets, got {logits.shape} and {targets.shape}")
    targets = targets.to(dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction="none").mean(dim=(-2, -1))
    probability = logits.float().sigmoid()
    intersection = (probability * targets.float()).sum(dim=(-2, -1))
    denominator = probability.sum(dim=(-2, -1)) + targets.float().sum(dim=(-2, -1))
    dice = 1.0 - (2.0 * intersection + epsilon) / (denominator + epsilon)
    if boundary_weight < 0.0:
        raise ValueError("boundary_weight must be non-negative")
    if boundary_weight == 0.0:
        return bce + dice
    predicted_boundary = _soft_boundary(probability, boundary_kernel_size)
    target_boundary = _soft_boundary(targets.float(), boundary_kernel_size)
    boundary_intersection = (predicted_boundary * target_boundary).sum(dim=(-2, -1))
    boundary_denominator = predicted_boundary.sum(dim=(-2, -1)) + target_boundary.sum(dim=(-2, -1))
    boundary_dice = 1.0 - (2.0 * boundary_intersection + epsilon) / (boundary_denominator + epsilon)
    return bce + dice + float(boundary_weight) * boundary_dice


def weighted_mask_loss(
    logits: Tensor,
    targets: Tensor,
    weights: Tensor,
    epsilon: float = 1.0e-6,
    *,
    boundary_weight: float = 0.0,
    boundary_kernel_size: int = 3,
) -> Tensor:
    """Normalize the weighted instance loss by effective supervision mass."""
    per_instance = per_instance_mask_loss(
        logits,
        targets,
        epsilon,
        boundary_weight=boundary_weight,
        boundary_kernel_size=boundary_kernel_size,
    )
    if weights.shape != per_instance.shape:
        raise ValueError(f"weights must have shape {per_instance.shape}, got {weights.shape}")
    if not torch.isfinite(weights).all() or (weights < 0).any() or (weights > 1).any():
        raise ValueError("Mask supervision weights must be finite and in [0,1]")
    return (per_instance * weights).sum() / (weights.sum() + epsilon)
