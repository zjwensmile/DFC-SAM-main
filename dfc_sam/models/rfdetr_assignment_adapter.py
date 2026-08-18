"""Hungarian positive-query identity for the frozen RF-DETR detector."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from .assignment_adapter import MatchedQueries


class RFDETRAssignmentAdapter:
    """Use RF-DETR's native matcher while keeping detector loss disabled."""

    def __init__(self, matcher: Any | None = None, *, training_args: Any | None = None) -> None:
        if matcher is None:
            try:
                from rfdetr.models.matcher import HungarianMatcher, build_matcher
            except ImportError as exc:  # pragma: no cover - isolated RF environment
                raise ImportError("RF-DETR HungarianMatcher is unavailable") from exc
            matcher = (
                build_matcher(training_args)
                if training_args is not None
                else HungarianMatcher(
                    cost_class=2.0,
                    cost_bbox=5.0,
                    cost_giou=2.0,
                    focal_alpha=0.25,
                )
            )
        self.matcher = matcher

    @staticmethod
    def _targets(batch: dict[str, Tensor], batch_size: int) -> list[dict[str, Tensor]]:
        flat_batch = batch["batch_idx"].view(-1).long()
        classes = batch["cls"].view(-1).long()
        boxes = batch["bboxes"].view(-1, 4)
        return [
            {"labels": classes[flat_batch == index], "boxes": boxes[flat_batch == index]}
            for index in range(batch_size)
        ]

    def select_matched_positives(
        self,
        raw_one2one: dict[str, Tensor],
        batch: dict[str, Tensor],
    ) -> MatchedQueries:
        batch_size = int(raw_one2one["pred_logits"].shape[0])
        targets = self._targets(batch, batch_size)
        matches = self.matcher(raw_one2one, targets, group_detr=1)
        counts = torch.tensor(
            [target["labels"].numel() for target in targets],
            device=raw_one2one["pred_logits"].device,
            dtype=torch.long,
        )
        offsets = torch.zeros(batch_size + 1, device=counts.device, dtype=torch.long)
        offsets[1:] = counts.cumsum(0)
        batch_parts: list[Tensor] = []
        query_parts: list[Tensor] = []
        within_parts: list[Tensor] = []
        for batch_index, (query_index, target_within) in enumerate(matches):
            query_index = query_index.to(device=counts.device, dtype=torch.long)
            target_within = target_within.to(device=counts.device, dtype=torch.long)
            batch_parts.append(torch.full_like(query_index, batch_index))
            query_parts.append(query_index)
            within_parts.append(target_within)
        empty = counts.new_empty(0)
        batch_indices = torch.cat(batch_parts) if batch_parts else empty
        query_indices = torch.cat(query_parts) if query_parts else empty
        within = torch.cat(within_parts) if within_parts else empty
        target_indices = offsets.index_select(0, batch_indices) + within
        if target_indices.numel() and int(target_indices.max()) >= int(counts.sum()):
            raise RuntimeError("RF-DETR matcher produced an out-of-range target index")
        return MatchedQueries(batch_indices, query_indices, target_indices, within)

    @staticmethod
    def detection_loss(*_args: Any, **_kwargs: Any) -> tuple[Tensor, dict[str, Tensor]]:
        raise RuntimeError(
            "RF-DETR detector loss is intentionally disabled in DFC-SAM. "
            "Freeze the selected Stage-I detector and train Bridge, then UGCA."
        )
