"""Differentiable multi-scale feature bridge, soft box gate, and instance-level SAA."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class BridgeOutput:
    dense_prompt_embeddings: Tensor
    scale_weights: Tensor
    spatial_gate: Tensor


class Projection(nn.Sequential):
    """Paper-specified 1x1 convolution, GroupNorm, and SiLU projection."""

    def __init__(self, input_dim: int, output_dim: int, groups: int) -> None:
        if output_dim % groups:
            raise ValueError(f"output_dim={output_dim} must be divisible by groups={groups}")
        super().__init__(
            nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False),
            nn.GroupNorm(groups, output_dim),
            nn.SiLU(inplace=True),
        )


class SoftBoxGate(nn.Module):
    """Continuous rectangular gate with transition width measured in latent cells."""

    def __init__(self, output_hw: tuple[int, int] = (64, 64), transition_cells: float = 0.5) -> None:
        super().__init__()
        if transition_cells <= 0:
            raise ValueError("transition_cells must be positive")
        self.output_hw = output_hw
        self.transition_cells = float(transition_cells)

    def forward(self, boxes_xyxy_normalized: Tensor) -> Tensor:
        if boxes_xyxy_normalized.ndim != 2 or boxes_xyxy_normalized.shape[-1] != 4:
            raise ValueError(f"Expected boxes [N,4], got {tuple(boxes_xyxy_normalized.shape)}")
        height, width = self.output_hw
        boxes = boxes_xyxy_normalized.clamp(0.0, 1.0)
        x1, y1, x2, y2 = boxes.unbind(dim=-1)
        x2 = torch.maximum(x2, x1)
        y2 = torch.maximum(y2, y1)

        x = (torch.arange(width, dtype=boxes.dtype, device=boxes.device) + 0.5) / width
        y = (torch.arange(height, dtype=boxes.dtype, device=boxes.device) + 0.5) / height
        tau_x = self.transition_cells / width
        tau_y = self.transition_cells / height
        gate_x = torch.sigmoid((x[None] - x1[:, None]) / tau_x) * torch.sigmoid((x2[:, None] - x[None]) / tau_x)
        gate_y = torch.sigmoid((y[None] - y1[:, None]) / tau_y) * torch.sigmoid((y2[:, None] - y[None]) / tau_y)
        return gate_y[:, None, :, None] * gate_x[:, None, None, :]


class DifferentiableFeatureBridge(nn.Module):
    """Map P3/P4/P5 and instance semantics to SAM dense prompt embeddings."""

    def __init__(
        self,
        pyramid_dims: tuple[int, int, int],
        semantic_dim: int,
        *,
        embed_dim: int = 256,
        embed_hw: tuple[int, int] = (64, 64),
        group_norm_groups: int = 32,
        saa_hidden_dim: int = 256,
        gate_transition_cells: float = 0.5,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.embed_hw = embed_hw
        self.projections = nn.ModuleList(
            Projection(input_dim, embed_dim, group_norm_groups) for input_dim in pyramid_dims
        )
        self.saa_score = nn.Sequential(
            nn.Linear(embed_dim + semantic_dim, saa_hidden_dim),
            nn.SiLU(),
            nn.Linear(saa_hidden_dim, 1),
        )
        self.gate = SoftBoxGate(embed_hw, gate_transition_cells)

    def forward(
        self,
        pyramid: tuple[Tensor, Tensor, Tensor],
        boxes_xyxy_normalized: Tensor,
        semantic_features: Tensor,
        batch_index: Tensor,
    ) -> BridgeOutput:
        if len(pyramid) != 3:
            raise ValueError("The bridge requires exactly P3, P4, and P5")
        instance_count = boxes_xyxy_normalized.shape[0]
        if semantic_features.shape[0] != instance_count or batch_index.shape != (instance_count,):
            raise ValueError("boxes, semantic_features, and batch_index must identify the same instances")
        if batch_index.dtype != torch.long:
            raise TypeError("batch_index must use torch.long")

        aligned = [
            F.interpolate(projection(feature), size=self.embed_hw, mode="bilinear", align_corners=False)
            for projection, feature in zip(self.projections, pyramid, strict=False)
        ]
        gate = self.gate(boxes_xyxy_normalized)
        denominator = gate.sum(dim=(-2, -1)).clamp_min(torch.finfo(gate.dtype).eps)

        instance_maps = torch.stack(
            [feature.index_select(0, batch_index) for feature in aligned],
            dim=1,
        )
        pooled = (instance_maps * gate[:, None]).sum(dim=(-2, -1)) / denominator[:, None]
        semantics = semantic_features[:, None].expand(-1, 3, -1)
        scores = self.saa_score(torch.cat((pooled, semantics), dim=-1)).squeeze(-1)
        scale_weights = scores.softmax(dim=-1)
        fused = (instance_maps * scale_weights[:, :, None, None, None]).sum(dim=1)
        dense_prompt = gate * fused
        return BridgeOutput(dense_prompt, scale_weights, gate)
