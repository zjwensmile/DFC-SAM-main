"""Uncertainty-gated cross-attention classification refinement."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from dfc_sam.data.geometry import build_box_grid


def _class_evidence(base_logits: Tensor, mode: str) -> tuple[Tensor, Tensor]:
    """Return detector-native evidence and a normalized entropy distribution."""
    if mode == "softmax":
        probability = base_logits.float().softmax(dim=-1)
        return probability, probability
    if mode == "sigmoid":
        evidence = base_logits.float().sigmoid()
        probability = evidence / evidence.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(evidence.dtype).eps
        )
        return evidence, probability
    raise ValueError(f"Unknown class_probability_mode: {mode}")


@dataclass
class UGCAOutput:
    refined_logits: Tensor
    entropy: Tensor
    gate: Tensor
    morphology_vector: Tensor
    attention_weights: Tensor
    gate_logits: Tensor | None = None
    quality_logits: Tensor | None = None
    quality_score: Tensor | None = None


class UGCA(nn.Module):
    """Use detached local SAM morphology to refine uncertain detector logits."""

    def __init__(
        self,
        semantic_dim: int,
        morphology_dim: int = 32,
        num_classes: int = 5,
        *,
        attention_dim: int = 256,
        num_heads: int = 8,
        grid_hw: tuple[int, int] = (14, 14),
        dropout: float = 0.0,
        class_probability_mode: str = "softmax",
    ) -> None:
        super().__init__()
        self.variant = "v1"
        if attention_dim % num_heads:
            raise ValueError("attention_dim must be divisible by num_heads")
        self.num_classes = num_classes
        self.grid_hw = grid_hw
        if class_probability_mode not in {"softmax", "sigmoid"}:
            raise ValueError("class_probability_mode must be 'softmax' or 'sigmoid'")
        self.class_probability_mode = class_probability_mode
        self.query_projection = nn.Linear(semantic_dim, attention_dim)
        self.morphology_projection = nn.Linear(morphology_dim, attention_dim)
        self.attention = nn.MultiheadAttention(
            attention_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_projection = nn.Linear(attention_dim, attention_dim)
        self.residual_classifier = nn.Linear(attention_dim, num_classes)
        # Joint training must begin at the validation-selected Warmup operating
        # point. A zero residual makes the first refined logits exactly equal
        # to the detector logits while still allowing the classifier to learn.
        nn.init.zeros_(self.residual_classifier.weight)
        nn.init.zeros_(self.residual_classifier.bias)
        self.gate_slope_raw = nn.Parameter(torch.tensor(0.0))
        self.gate_threshold_raw = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        semantic_features: Tensor,
        base_logits: Tensor,
        morphology: Tensor,
        boxes_xyxy_normalized: Tensor,
        *,
        mask_logits: Tensor | None = None,
        iou_prediction: Tensor | None = None,
    ) -> UGCAOutput:
        instance_count = semantic_features.shape[0]
        if base_logits.shape != (instance_count, self.num_classes):
            raise ValueError(
                f"Expected base_logits {(instance_count, self.num_classes)}, got {tuple(base_logits.shape)}"
            )
        if morphology.shape[0] != instance_count or boxes_xyxy_normalized.shape != (instance_count, 4):
            raise ValueError("UGCA inputs must identify the same N instances")

        morphology_source = morphology.detach()
        box_source = boxes_xyxy_normalized.detach()
        grid = build_box_grid(box_source, self.grid_hw)
        local = F.grid_sample(morphology_source, grid, mode="bilinear", align_corners=False)
        tokens = local.flatten(2).transpose(1, 2)

        query = self.query_projection(semantic_features).unsqueeze(1)
        key_value = self.morphology_projection(tokens)
        attended, weights = self.attention(query, key_value, key_value, need_weights=True)
        morphology_vector = self.output_projection(attended.squeeze(1))

        _, probability = _class_evidence(base_logits, self.class_probability_mode)
        probability = probability.to(base_logits.dtype)
        entropy = -(probability * probability.clamp_min(torch.finfo(probability.dtype).eps).log()).sum(dim=-1)
        entropy = entropy / math.log(self.num_classes)
        detached_entropy = entropy.detach()
        slope = F.softplus(self.gate_slope_raw)
        threshold = torch.sigmoid(self.gate_threshold_raw)
        gate_logits = slope * (detached_entropy - threshold)
        gate = torch.sigmoid(gate_logits)
        residual = self.residual_classifier(morphology_vector)
        refined = base_logits + gate[:, None] * residual
        return UGCAOutput(refined, entropy, gate, morphology_vector, weights, gate_logits)


class ConservativeUGCAV2(nn.Module):
    """Conservative classification refinement with a learned correction gate."""

    def __init__(
        self,
        semantic_dim: int,
        morphology_dim: int = 32,
        num_classes: int = 5,
        *,
        attention_dim: int = 256,
        num_heads: int = 8,
        grid_hw: tuple[int, int] = (14, 14),
        dropout: float = 0.0,
        gate_hidden_dim: int = 64,
        residual_logit_cap: float = 1.0,
        class_probability_mode: str = "softmax",
    ) -> None:
        super().__init__()
        if attention_dim % num_heads:
            raise ValueError("attention_dim must be divisible by num_heads")
        if gate_hidden_dim < 1 or residual_logit_cap <= 0.0:
            raise ValueError("gate_hidden_dim and residual_logit_cap must be positive")
        self.variant = "conservative_v2"
        self.num_classes = num_classes
        self.grid_hw = grid_hw
        self.residual_logit_cap = float(residual_logit_cap)
        if class_probability_mode not in {"softmax", "sigmoid"}:
            raise ValueError("class_probability_mode must be 'softmax' or 'sigmoid'")
        self.class_probability_mode = class_probability_mode
        # probabilities + entropy/confidence/margin + width/height/area/aspect
        self.evidence_dim = num_classes + 7
        self.query_projection = nn.Linear(semantic_dim, attention_dim)
        self.morphology_projection = nn.Linear(morphology_dim, attention_dim)
        self.attention = nn.MultiheadAttention(
            attention_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_projection = nn.Linear(attention_dim, attention_dim)
        self.evidence_projection = nn.Linear(self.evidence_dim, attention_dim)
        self.fusion_norm = nn.LayerNorm(attention_dim)
        self.gate_network = nn.Sequential(
            nn.Linear(attention_dim + self.evidence_dim, gate_hidden_dim),
            nn.SiLU(),
            nn.Linear(gate_hidden_dim, 1),
        )
        self.residual_classifier = nn.Sequential(
            nn.Linear(attention_dim, attention_dim // 2),
            nn.SiLU(),
            nn.Linear(attention_dim // 2, num_classes),
        )
        nn.init.zeros_(self.residual_classifier[-1].weight)
        nn.init.zeros_(self.residual_classifier[-1].bias)
        nn.init.zeros_(self.gate_network[-1].weight)
        nn.init.constant_(self.gate_network[-1].bias, -2.0)

    def _evidence(self, base_logits: Tensor, boxes: Tensor) -> tuple[Tensor, Tensor]:
        detector_evidence, probability = _class_evidence(
            base_logits,
            self.class_probability_mode,
        )
        entropy = -(probability * probability.clamp_min(torch.finfo(probability.dtype).eps).log()).sum(dim=-1)
        entropy = entropy / math.log(self.num_classes)
        top_two = detector_evidence.topk(k=2, dim=-1).values
        confidence = top_two[:, 0]
        margin = top_two[:, 0] - top_two[:, 1]
        width_height = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0.0)
        width, height = width_height.unbind(dim=-1)
        area = width * height
        aspect = torch.log((width + 1.0e-4) / (height + 1.0e-4)).clamp(-4.0, 4.0)
        evidence = torch.cat(
            (
                detector_evidence,
                entropy[:, None],
                confidence[:, None],
                margin[:, None],
                width[:, None],
                height[:, None],
                area[:, None],
                aspect[:, None],
            ),
            dim=-1,
        )
        return evidence.detach(), entropy

    def forward(
        self,
        semantic_features: Tensor,
        base_logits: Tensor,
        morphology: Tensor,
        boxes_xyxy_normalized: Tensor,
        *,
        mask_logits: Tensor | None = None,
        iou_prediction: Tensor | None = None,
    ) -> UGCAOutput:
        instance_count = semantic_features.shape[0]
        if base_logits.shape != (instance_count, self.num_classes):
            raise ValueError(
                f"Expected base_logits {(instance_count, self.num_classes)}, got {tuple(base_logits.shape)}"
            )
        if morphology.shape[0] != instance_count or boxes_xyxy_normalized.shape != (instance_count, 4):
            raise ValueError("UGCA inputs must identify the same N instances")

        boxes = boxes_xyxy_normalized.detach()
        grid = build_box_grid(boxes, self.grid_hw)
        local = F.grid_sample(morphology.detach(), grid, mode="bilinear", align_corners=False)
        tokens = local.flatten(2).transpose(1, 2)
        evidence, entropy = self._evidence(base_logits, boxes)
        query_vector = self.query_projection(semantic_features)
        key_value = self.morphology_projection(tokens)
        attended, weights = self.attention(
            query_vector.unsqueeze(1),
            key_value,
            key_value,
            need_weights=True,
        )
        morphology_vector = self.fusion_norm(
            query_vector
            + self.output_projection(attended.squeeze(1))
            + self.evidence_projection(evidence.to(query_vector.dtype))
        )
        gate_logits = self.gate_network(
            torch.cat((morphology_vector, evidence.to(morphology_vector.dtype)), dim=-1)
        ).squeeze(-1)
        gate = gate_logits.sigmoid()
        raw_residual = self.residual_classifier(morphology_vector)
        residual = self.residual_logit_cap * raw_residual.tanh()
        refined = base_logits + gate[:, None] * residual
        return UGCAOutput(refined, entropy, gate, morphology_vector, weights, gate_logits)


class QualityAwareUGCAV3(ConservativeUGCAV2):
    """Conservative class refinement plus an independently supervised quality score."""

    def __init__(
        self,
        semantic_dim: int,
        morphology_dim: int = 32,
        num_classes: int = 5,
        *,
        attention_dim: int = 256,
        num_heads: int = 8,
        grid_hw: tuple[int, int] = (14, 14),
        dropout: float = 0.0,
        gate_hidden_dim: int = 64,
        residual_logit_cap: float = 1.0,
        quality_hidden_dim: int = 64,
        class_probability_mode: str = "softmax",
    ) -> None:
        super().__init__(
            semantic_dim,
            morphology_dim,
            num_classes,
            attention_dim=attention_dim,
            num_heads=num_heads,
            grid_hw=grid_hw,
            dropout=dropout,
            gate_hidden_dim=gate_hidden_dim,
            residual_logit_cap=residual_logit_cap,
            class_probability_mode=class_probability_mode,
        )
        if quality_hidden_dim < 1:
            raise ValueError("quality_hidden_dim must be positive")
        self.variant = "quality_aware_v3"
        # Detector absolute score, SAM IoU estimate, soft mask area, and
        # probability-weighted foreground confidence complement v2 evidence.
        self.quality_evidence_dim = self.evidence_dim + 4
        self.quality_network = nn.Sequential(
            nn.Linear(attention_dim + self.quality_evidence_dim, quality_hidden_dim),
            nn.SiLU(),
            nn.Linear(quality_hidden_dim, 1),
        )
        nn.init.zeros_(self.quality_network[-1].weight)
        # Preserve the Warmup inference work point at initialization. Hard
        # negatives then have to learn candidate-specific suppression.
        nn.init.constant_(self.quality_network[-1].bias, 4.0)

    def forward(
        self,
        semantic_features: Tensor,
        base_logits: Tensor,
        morphology: Tensor,
        boxes_xyxy_normalized: Tensor,
        *,
        mask_logits: Tensor | None = None,
        iou_prediction: Tensor | None = None,
    ) -> UGCAOutput:
        if mask_logits is None or iou_prediction is None:
            raise ValueError("quality_aware_v3 requires mask_logits and iou_prediction")
        instance_count = semantic_features.shape[0]
        if mask_logits.shape[:2] != (instance_count, 1):
            raise ValueError("mask_logits must have shape [N,1,H,W]")
        if iou_prediction.shape != (instance_count, 1):
            raise ValueError("iou_prediction must have shape [N,1]")

        base = super().forward(
            semantic_features,
            base_logits,
            morphology,
            boxes_xyxy_normalized,
        )
        evidence, _ = self._evidence(base_logits, boxes_xyxy_normalized.detach())
        detector_score = base_logits.float().sigmoid().amax(dim=-1, keepdim=True).detach()
        sam_iou = iou_prediction.float().clamp(0.0, 1.0).detach()
        mask_probability = mask_logits.float().sigmoid().detach()
        soft_area = mask_probability.mean(dim=(-2, -1))
        foreground_confidence = (
            mask_probability.square().sum(dim=(-2, -1))
            / mask_probability.sum(dim=(-2, -1)).clamp_min(torch.finfo(torch.float32).eps)
        )
        quality_evidence = torch.cat(
            (evidence.float(), detector_score, sam_iou, soft_area, foreground_confidence),
            dim=-1,
        )
        quality_logits = self.quality_network(
            torch.cat(
                (base.morphology_vector, quality_evidence.to(base.morphology_vector.dtype)),
                dim=-1,
            )
        ).squeeze(-1)
        quality_score = quality_logits.float().sigmoid()
        return UGCAOutput(
            refined_logits=base.refined_logits,
            entropy=base.entropy,
            gate=base.gate,
            morphology_vector=base.morphology_vector,
            attention_weights=base.attention_weights,
            gate_logits=base.gate_logits,
            quality_logits=quality_logits,
            quality_score=quality_score,
        )
