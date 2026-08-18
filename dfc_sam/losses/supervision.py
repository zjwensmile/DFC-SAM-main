"""Resolve matched-query masks identically across full, No-pseudo, Naive, and QWS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from dfc_sam.models.assignment_adapter import MatchedQueries
from dfc_sam.pseudo.store import PseudoMaskStore


@dataclass
class ResolvedMaskSupervision:
    query_indices: Tensor
    targets: Tensor
    weights: Tensor
    sources: list[str]


def resolve_mask_supervision(
    matched: MatchedQueries,
    batch: dict[str, Any],
    *,
    strategy: str,
    pseudo_store: PseudoMaskStore | None,
    device: torch.device | str,
) -> ResolvedMaskSupervision:
    """Build mask targets for matched positives without changing native assignment."""
    if strategy not in {"full", "no_pseudo", "naive_mixed", "qws_mixed"}:
        raise ValueError(f"Unknown supervision strategy: {strategy}")
    requires_pseudo = strategy in {"naive_mixed", "qws_mixed"}
    if requires_pseudo != (pseudo_store is not None):
        raise ValueError(f"{strategy} pseudo-store requirement mismatch")

    flat_count = int(batch["target_batch"]["cls"].numel())
    gt_lookup = torch.full((flat_count,), -1, dtype=torch.long)
    supervised_indices = batch["supervised_target_indices"].cpu().long()
    gt_lookup[supervised_indices] = torch.arange(supervised_indices.numel())

    query_indices = []
    masks = []
    weights = []
    sources = []
    for query_position, flat_target_tensor in enumerate(matched.target_index.detach().cpu()):
        flat_target = int(flat_target_tensor)
        # Quality-aware UGCA appends unmatched hard negatives with target -1.
        # They are valid quality targets (quality=0), but never mask targets.
        # Indexing gt_lookup[-1] would silently alias the final ground-truth
        # instance and corrupt Bridge/decoder supervision.
        if flat_target < 0:
            continue
        if flat_target >= flat_count:
            raise RuntimeError(f"Matched target index is out of range: {flat_target} >= {flat_count}")
        gt_index = int(gt_lookup[flat_target])
        if gt_index >= 0:
            query_indices.append(query_position)
            masks.append(batch["supervised_masks"][gt_index].to(device=device, dtype=torch.bool))
            weights.append(1.0)
            sources.append("ground_truth")
            continue
        if strategy in {"full"}:
            raise RuntimeError("Full supervision matched a target without a ground-truth mask")
        if strategy == "no_pseudo":
            continue
        assert pseudo_store is not None
        image_id = str(batch["target_image_ids"][flat_target])
        instance_index = int(batch["target_instance_indices"][flat_target])
        pseudo_mask, qws_weight = pseudo_store.get(image_id, instance_index)
        query_indices.append(query_position)
        masks.append(pseudo_mask.to(device=device, dtype=torch.bool))
        weights.append(1.0 if strategy == "naive_mixed" else qws_weight)
        sources.append("pseudo")

    target_tensor = torch.stack(masks) if masks else torch.empty((0, 256, 256), dtype=torch.bool, device=device)
    return ResolvedMaskSupervision(
        query_indices=torch.tensor(query_indices, dtype=torch.long, device=device),
        targets=target_tensor,
        weights=torch.tensor(weights, dtype=torch.float32, device=device),
        sources=sources,
    )
