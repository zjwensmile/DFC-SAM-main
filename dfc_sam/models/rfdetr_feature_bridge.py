"""RF-DETR-specific Bridge using shallow/deep/fused detector evidence."""

from __future__ import annotations

from torch import Tensor

from .feature_bridge import BridgeOutput, DifferentiableFeatureBridge


class RFDETRFeatureBridge(DifferentiableFeatureBridge):
    """Project RF-DETR's three multi-depth sources into SAM dense prompts.

    RF-DETR-2XL exposes only one spatial P4 projector output.  The first two
    inputs are therefore real shallow/deep DINO features at the same 44x44
    grid, not synthetically resized P3/P5 maps.  The inherited SAA learns an
    instance-conditioned mixture of semantic depth and the fused P4 feature.
    """

    feature_source_names = ("dino_shallow", "dino_deep", "projected_p4")

    def forward(
        self,
        pyramid: tuple[Tensor, Tensor, Tensor],
        boxes_xyxy_normalized: Tensor,
        semantic_features: Tensor,
        batch_index: Tensor,
    ) -> BridgeOutput:
        if tuple(int(feature.shape[-1]) for feature in pyramid) != (44, 44, 44):
            raise ValueError(
                "RF-DETR-2XL Bridge expects three native 44x44 feature sources at 880px"
            )
        return super().forward(
            pyramid,
            boxes_xyxy_normalized,
            semantic_features,
            batch_index,
        )
