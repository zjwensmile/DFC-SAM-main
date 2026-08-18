"""Evidence-based adapter for the pinned Ultralytics YOLO26 detection model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from dfc_sam.constants import PANNUKE_CLASSES


@dataclass
class DetectorOutput:
    p3: Tensor
    p4: Tensor
    p5: Tensor
    raw_one2many: dict[str, Tensor]
    raw_one2one: dict[str, Tensor]
    decoded_boxes_xyxy: Tensor
    base_logits: Tensor
    semantic_features: Tensor
    query_indices: Tensor


class YOLO26Adapter(nn.Module):
    """Expose YOLO26 pyramid, one-to-one logits, boxes, and pre-classifier semantics.

    The adapter does not patch upstream source. Native mode preserves the pinned
    Detect.forward() detach boundary; coupled mode sends undetached neck features
    through the same one-to-one head parameters.
    """

    class_probability_mode = "softmax"
    feature_source_names = ("p3", "p4", "p5")

    def __init__(self, detection_model: nn.Module) -> None:
        super().__init__()
        self.model = detection_model
        self._validate_architecture()

    @property
    def head(self) -> nn.Module:
        return self.model.model[-1]

    @property
    def semantic_dim(self) -> int:
        return int(self.head.one2one_cv3[0][-1].in_channels)

    @property
    def pyramid_dims(self) -> tuple[int, int, int]:
        return tuple(int(branch[0].conv.in_channels) for branch in self.head.one2one_cv2)

    def _validate_architecture(self) -> None:
        head = self.head
        required = ("one2one_cv2", "one2one_cv3", "forward_head", "_get_decode_boxes", "stride")
        missing = [name for name in required if not hasattr(head, name)]
        if missing:
            raise TypeError(f"Pinned YOLO26 head is missing required attributes: {missing}")
        if len(head.one2one_cv2) != 3 or len(head.one2one_cv3) != 3:
            raise ValueError("DFC-SAM requires exactly three YOLO pyramid levels")
        if int(head.nc) != len(PANNUKE_CLASSES):
            raise ValueError(f"YOLO checkpoint has nc={head.nc}, expected {len(PANNUKE_CLASSES)}")
        names = tuple(str(self.model.names[index]).lower() for index in range(len(PANNUKE_CLASSES)))
        if names != PANNUKE_CLASSES:
            raise ValueError(f"YOLO checkpoint class order mismatch: {names!r}")

    def _forward_backbone_neck(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        value: Any = images
        saved: list[Any] = []
        for module in self.model.model[:-1]:
            if module.f != -1:
                value = (
                    saved[module.f]
                    if isinstance(module.f, int)
                    else [value if index == -1 else saved[index] for index in module.f]
                )
            value = module(value)
            saved.append(value if module.i in self.model.save else None)

        final = self.head
        if final.f == -1:
            pyramid = value
        else:
            pyramid = (
                saved[final.f]
                if isinstance(final.f, int)
                else [value if index == -1 else saved[index] for index in final.f]
            )
        if not isinstance(pyramid, list) or len(pyramid) != 3:
            raise RuntimeError("Could not resolve YOLO26 P3/P4/P5 inputs from the pinned graph")
        return tuple(pyramid)

    @staticmethod
    def _classification_with_semantics(branch: nn.Sequential, feature: Tensor) -> tuple[Tensor, Tensor]:
        semantic = feature
        for layer in branch[:-1]:
            semantic = layer(semantic)
        return branch[-1](semantic), semantic

    def forward(
        self,
        images: Tensor,
        *,
        coupled_grad: bool,
        decode_boxes_grad: bool = True,
    ) -> DetectorOutput:
        pyramid = self._forward_backbone_neck(images)
        head = self.head
        one2many = head.forward_head(list(pyramid), **head.one2many)
        one2one_features = list(pyramid) if coupled_grad else [feature.detach() for feature in pyramid]
        batch_size = images.shape[0]

        boxes_per_level = []
        scores_per_level = []
        semantic_per_level = []
        for level, feature in enumerate(one2one_features):
            boxes_per_level.append(head.one2one_cv2[level](feature).view(batch_size, 4 * head.reg_max, -1))
            score_map, semantic_map = self._classification_with_semantics(head.one2one_cv3[level], feature)
            scores_per_level.append(score_map.view(batch_size, head.nc, -1))
            semantic_per_level.append(semantic_map.flatten(2).transpose(1, 2))

        one2one = {
            "boxes": torch.cat(boxes_per_level, dim=-1),
            "scores": torch.cat(scores_per_level, dim=-1),
            "feats": one2one_features,
        }
        if decode_boxes_grad:
            decoded = head._get_decode_boxes(one2one).transpose(1, 2)
        else:
            # Box coordinates are only evidence for frozen downstream paths.
            # Decoding them under no_grad avoids retaining an unnecessary DFL
            # graph and makes cached anchors created by validation-safe
            # inference_mode legal to reuse in the subsequent training epoch.
            with torch.no_grad():
                decoded = head._get_decode_boxes(
                    {
                        "boxes": one2one["boxes"].detach(),
                        # Ultralytics uses the first feature only to refresh
                        # its anchor grid shape; detached metadata is enough.
                        "feats": [feature.detach() for feature in one2one["feats"]],
                    }
                ).transpose(1, 2)
        logits = one2one["scores"].transpose(1, 2)
        semantics = torch.cat(semantic_per_level, dim=1)
        query_indices = torch.arange(logits.shape[1], device=images.device, dtype=torch.long)
        return DetectorOutput(
            p3=pyramid[0],
            p4=pyramid[1],
            p5=pyramid[2],
            raw_one2many=one2many,
            raw_one2one=one2one,
            decoded_boxes_xyxy=decoded,
            base_logits=logits,
            semantic_features=semantics,
            query_indices=query_indices,
        )


def load_yolo26_adapter(checkpoint: str | Path) -> YOLO26Adapter:
    """Load a pinned Ultralytics checkpoint and return its DFC adapter."""
    from ultralytics import YOLO

    wrapper = YOLO(str(Path(checkpoint).expanduser().resolve()), task="detect")
    return YOLO26Adapter(wrapper.model)
