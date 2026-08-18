"""Select matched positive one-to-one queries using the pinned native criterion."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class MatchedQueries:
    batch_index: Tensor
    query_index: Tensor
    target_index: Tensor
    target_gt_index_within_image: Tensor


class AssignmentAdapter:
    """Reuse Ultralytics assignment and expose only its matched-positive identity."""

    def __init__(self, detection_model: nn.Module) -> None:
        self.model = detection_model

    def _full_criterion(self):
        if getattr(self.model, "criterion", None) is None:
            if isinstance(getattr(self.model, "args", None), dict):
                # A serialized inference checkpoint retains only a small args dict,
                # while the native loss expects the complete attribute-style cfg.
                from ultralytics.cfg import get_cfg

                self.model.args = get_cfg(overrides=self.model.args)
            self.model.criterion = self.model.init_criterion()
        criterion = self.model.criterion
        if not hasattr(criterion, "one2one"):
            raise TypeError("Pinned YOLO criterion does not expose a one-to-one loss")
        return criterion

    def select_matched_positives(self, raw_one2one: dict[str, Tensor], batch: dict[str, Tensor]) -> MatchedQueries:
        """Return query-to-flattened-target mappings from the native one-to-one assigner."""
        criterion = self._full_criterion().one2one
        assignment, _, _ = criterion.get_assigned_targets_and_loss(raw_one2one, batch)
        foreground, target_gt_index, _, _, _ = assignment
        batch_index, query_index = foreground.nonzero(as_tuple=True)
        within = target_gt_index[batch_index, query_index].long()

        flat_batch = batch["batch_idx"].view(-1).long()
        counts = torch.bincount(flat_batch, minlength=raw_one2one["boxes"].shape[0])
        offsets = torch.zeros(counts.numel() + 1, dtype=torch.long, device=counts.device)
        offsets[1:] = counts.cumsum(0)
        target_index = offsets[batch_index] + within
        if target_index.numel() and int(target_index.max()) >= flat_batch.numel():
            raise RuntimeError("Native assignment produced an out-of-range target index")
        return MatchedQueries(batch_index, query_index, target_index, within)

    def detection_loss(
        self,
        raw_one2many: dict[str, Tensor],
        raw_one2one: dict[str, Tensor],
        batch: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute the pinned native end-to-end detector objective."""
        criterion = self._full_criterion()
        loss_components, detached_components = criterion(
            {"one2many": raw_one2many, "one2one": raw_one2one},
            batch,
        )
        if loss_components.ndim != 1 or loss_components.numel() != len(detached_components):
            raise RuntimeError(
                "Pinned detector criterion must return one loss value per named component"
            )
        return loss_components.sum(), detached_components
