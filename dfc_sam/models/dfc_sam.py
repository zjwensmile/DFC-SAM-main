"""End-to-end student forward wiring with explicit instance identity."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .assignment_adapter import AssignmentAdapter, MatchedQueries
from .feature_bridge import BridgeOutput, DifferentiableFeatureBridge
from .sam_instance_decoder import SAMInstanceDecoder
from .ugca import UGCA, UGCAOutput
from .yolo26_adapter import DetectorOutput


@dataclass
class DFCStudentOutput:
    detector: DetectorOutput
    selected: MatchedQueries
    boxes_xyxy_normalized: Tensor
    base_logits: Tensor
    refined_logits: Tensor
    semantic_features: Tensor
    mask_logits: Tensor
    iou_prediction: Tensor
    morphology: Tensor
    bridge: BridgeOutput
    ugca: UGCAOutput | None
    class_probability_mode: str = "softmax"


@dataclass
class DFCTrainingContext:
    """Shared per-image tensors retained while instances are decoded in chunks."""

    detector: DetectorOutput
    selected: MatchedQueries
    image_embedding: Tensor | None
    yolo_hw: tuple[int, int]
    stage: str


class DFCSAM(nn.Module):
    """Compose the pinned detector, feature bridge, student SAM decoder, and UGCA."""

    def __init__(
        self,
        detector: nn.Module,
        student_sam: nn.Module,
        feature_bridge: DifferentiableFeatureBridge,
        student_decoder: SAMInstanceDecoder,
        ugca: UGCA,
        *,
        assignment: Any | None = None,
        instance_chunk_size: int = 32,
        inference_instance_chunk_size: int = 32,
        training_hard_negative_ratio: float = 0.0,
        training_hard_negative_max_per_image: int = 0,
        training_hard_negative_score_threshold: float = 0.05,
    ) -> None:
        super().__init__()
        if instance_chunk_size <= 0 or inference_instance_chunk_size <= 0:
            raise ValueError("Training and inference instance chunk sizes must be positive")
        if training_hard_negative_ratio < 0.0 or training_hard_negative_max_per_image < 0:
            raise ValueError("Hard-negative ratio and cap must be non-negative")
        if not 0.0 <= training_hard_negative_score_threshold <= 1.0:
            raise ValueError("Hard-negative score threshold must be in [0,1]")
        self.detector = detector
        self.student_sam = student_sam
        self.feature_bridge = feature_bridge
        self.student_decoder = student_decoder
        self.ugca = ugca
        self.assignment = assignment if assignment is not None else AssignmentAdapter(detector.model)
        self.instance_chunk_size = instance_chunk_size
        self.inference_instance_chunk_size = inference_instance_chunk_size
        self.training_hard_negative_ratio = float(training_hard_negative_ratio)
        self.training_hard_negative_max_per_image = int(training_hard_negative_max_per_image)
        self.training_hard_negative_score_threshold = float(training_hard_negative_score_threshold)
        # SAM's fixed dense positional encoding is precomputed on CPU. Besides
        # avoiding a redundant prompt-encoder call, this keeps strict CUDA
        # determinism from hitting the unsupported cumsum kernel.
        self.register_buffer(
            "sam_dense_pe",
            student_sam.prompt_encoder.get_dense_pe().detach().clone(),
            persistent=False,
        )

    @staticmethod
    def _gather(tensor: Tensor, selected: MatchedQueries) -> Tensor:
        return tensor[selected.batch_index, selected.query_index]

    def _mask_path_trainable(self) -> bool:
        """Whether a loss is allowed to backpropagate through Bridge/SAM."""
        return any(parameter.requires_grad for parameter in self.feature_bridge.parameters()) or any(
            parameter.requires_grad for parameter in self.student_sam.mask_decoder.parameters()
        )

    @staticmethod
    def _prefilter(logits: Tensor, threshold: float, topk: int) -> MatchedQueries:
        """High-recall per-image selection without comparing candidate overlap."""
        batch_indices = []
        query_indices = []
        scores = logits.sigmoid().amax(dim=-1)
        for batch_index in range(logits.shape[0]):
            eligible = (scores[batch_index] >= threshold).nonzero(as_tuple=False).flatten()
            if eligible.numel() > topk:
                order = scores[batch_index, eligible].topk(topk).indices
                eligible = eligible[order]
            batch_indices.append(torch.full_like(eligible, batch_index))
            query_indices.append(eligible)
        batch = torch.cat(batch_indices) if batch_indices else logits.new_empty(0, dtype=torch.long)
        query = torch.cat(query_indices) if query_indices else logits.new_empty(0, dtype=torch.long)
        empty = torch.full_like(query, -1)
        return MatchedQueries(batch, query, empty, empty)

    def _augment_with_hard_negatives(
        self,
        selected: MatchedQueries,
        base_logits: Tensor,
    ) -> MatchedQueries:
        """Append deterministic top-scoring non-assigned queries for quality supervision."""
        ratio = self.training_hard_negative_ratio
        cap = self.training_hard_negative_max_per_image
        if ratio <= 0.0 or cap <= 0:
            return selected
        scores = base_logits.detach().sigmoid().amax(dim=-1)
        negative_batches = []
        negative_queries = []
        for batch_index in range(base_logits.shape[0]):
            positive_queries = selected.query_index[selected.batch_index == batch_index]
            positive_count = int(positive_queries.numel())
            if positive_count == 0:
                continue
            candidate_mask = scores[batch_index] >= self.training_hard_negative_score_threshold
            candidate_mask = candidate_mask.clone()
            candidate_mask[positive_queries] = False
            eligible = candidate_mask.nonzero(as_tuple=False).flatten()
            requested = min(cap, int(math.ceil(positive_count * ratio)))
            if requested <= 0 or eligible.numel() == 0:
                continue
            if eligible.numel() > requested:
                order = scores[batch_index, eligible].topk(requested).indices
                eligible = eligible.index_select(0, order)
            negative_batches.append(torch.full_like(eligible, batch_index))
            negative_queries.append(eligible)
        if not negative_queries:
            return selected
        negative_count = sum(int(values.numel()) for values in negative_queries)
        negative_targets = torch.full(
            (negative_count,),
            -1,
            dtype=selected.target_index.dtype,
            device=selected.target_index.device,
        )
        return MatchedQueries(
            batch_index=torch.cat((selected.batch_index, *negative_batches)),
            query_index=torch.cat((selected.query_index, *negative_queries)),
            target_index=torch.cat((selected.target_index, negative_targets)),
            target_gt_index_within_image=torch.cat(
                (selected.target_gt_index_within_image, negative_targets.clone())
            ),
        )

    def _decode_chunks(self, image_embedding: Tensor, dense_pe: Tensor, prompts: Tensor, batch_index: Tensor):
        masks = []
        iou_predictions = []
        morphology = []
        batch_size = image_embedding.shape[0]
        pe_batch = dense_pe.expand(batch_size, -1, -1, -1)
        for start in range(0, prompts.shape[0], self.instance_chunk_size):
            stop = min(start + self.instance_chunk_size, prompts.shape[0])
            indices = batch_index[start:stop]
            chunk_masks, chunk_iou, chunk_morphology = self.student_decoder.forward_instances(
                image_embedding.index_select(0, indices),
                pe_batch.index_select(0, indices),
                prompts[start:stop],
                multimask_output=False,
            )
            masks.append(chunk_masks)
            iou_predictions.append(chunk_iou)
            morphology.append(chunk_morphology)
        return torch.cat(masks), torch.cat(iou_predictions), torch.cat(morphology)

    @staticmethod
    def _slice_selected(selected: MatchedQueries, start: int, stop: int) -> MatchedQueries:
        return MatchedQueries(
            batch_index=selected.batch_index[start:stop],
            query_index=selected.query_index[start:stop],
            target_index=selected.target_index[start:stop],
            target_gt_index_within_image=selected.target_gt_index_within_image[start:stop],
        )

    def prepare_training_context(
        self,
        yolo_images: Tensor,
        sam_resized_images: Tensor,
        *,
        target_batch: dict[str, Tensor],
        stage: str,
    ) -> DFCTrainingContext:
        """Run shared image work once before memory-bounded instance decoding."""
        stage_value = str(getattr(stage, "value", stage))
        if stage_value not in {"warmup", "joint"}:
            raise ValueError(f"Training context does not support stage: {stage_value}")
        coupled = stage_value == "joint"
        detector_output = self.detector(
            yolo_images,
            coupled_grad=coupled,
            decode_boxes_grad=self._mask_path_trainable(),
        )
        selected = self.assignment.select_matched_positives(
            detector_output.raw_one2one,
            target_batch,
        )
        if coupled:
            selected = self._augment_with_hard_negatives(selected, detector_output.base_logits)
        image_embedding = None
        if selected.query_index.numel():
            with torch.no_grad():
                image_embedding = self.student_sam.image_encoder(self.student_sam.preprocess(sam_resized_images))
        return DFCTrainingContext(
            detector=detector_output,
            selected=selected,
            image_embedding=image_embedding,
            yolo_hw=(int(yolo_images.shape[-2]), int(yolo_images.shape[-1])),
            stage=stage_value,
        )

    def decode_training_chunk(
        self,
        context: DFCTrainingContext,
        start: int,
        stop: int,
    ) -> DFCStudentOutput:
        """Decode one matched-instance slice without retaining other SAM graphs."""
        instance_count = int(context.selected.query_index.numel())
        if not 0 <= start < stop <= instance_count:
            raise ValueError(f"Invalid training chunk [{start}:{stop}] for {instance_count} instances")
        if stop - start > self.instance_chunk_size:
            raise ValueError("Training chunk exceeds configured instance_chunk_size")
        if context.image_embedding is None:
            raise RuntimeError("Non-empty training chunk has no SAM image embedding")

        selected = self._slice_selected(context.selected, start, stop)
        detector_output = context.detector
        boxes = self._gather(detector_output.decoded_boxes_xyxy, selected)
        yolo_height, yolo_width = context.yolo_hw
        divisor = boxes.new_tensor([yolo_width, yolo_height, yolo_width, yolo_height])
        boxes_normalized = (boxes / divisor).clamp(0, 1)
        semantics = self._gather(detector_output.semantic_features, selected)
        base_logits = self._gather(detector_output.base_logits, selected)
        batch_size = context.image_embedding.shape[0]
        dense_pe = self.sam_dense_pe.expand(batch_size, -1, -1, -1)
        indices = selected.batch_index
        mask_path_trainable = self._mask_path_trainable()
        if mask_path_trainable:
            bridge = self.feature_bridge(
                (detector_output.p3, detector_output.p4, detector_output.p5),
                boxes_normalized,
                semantics,
                selected.batch_index,
            )
            mask_logits, iou_prediction, morphology = self.student_decoder.forward_instances(
                context.image_embedding.index_select(0, indices),
                dense_pe.index_select(0, indices),
                bridge.dense_prompt_embeddings,
                multimask_output=False,
            )
        else:
            # UGCA intentionally consumes morphology and quality evidence as
            # detached information.  When the complete mask path is frozen,
            # retaining its graph only wastes memory (and can OOM a detector-head
            # probe) without creating a valid gradient route to any trainable
            # parameter.  Keep the detector's semantic/base-logit branch live.
            with torch.no_grad():
                bridge = self.feature_bridge(
                    (
                        detector_output.p3.detach(),
                        detector_output.p4.detach(),
                        detector_output.p5.detach(),
                    ),
                    boxes_normalized.detach(),
                    semantics.detach(),
                    selected.batch_index,
                )
                mask_logits, iou_prediction, morphology = self.student_decoder.forward_instances(
                    context.image_embedding.index_select(0, indices),
                    dense_pe.index_select(0, indices),
                    bridge.dense_prompt_embeddings,
                    multimask_output=False,
                )
        coupled = context.stage == "joint"
        ugca = (
            self.ugca(
                semantics,
                base_logits,
                morphology,
                boxes_normalized,
                mask_logits=mask_logits,
                iou_prediction=iou_prediction,
            )
            if coupled
            else None
        )
        refined_logits = ugca.refined_logits if ugca is not None else base_logits
        return DFCStudentOutput(
            detector=detector_output,
            selected=selected,
            boxes_xyxy_normalized=boxes_normalized,
            base_logits=base_logits,
            refined_logits=refined_logits,
            semantic_features=semantics,
            mask_logits=mask_logits,
            iou_prediction=iou_prediction,
            morphology=morphology,
            bridge=bridge,
            ugca=ugca,
            class_probability_mode=str(getattr(self.detector, "class_probability_mode", "softmax")),
        )

    def _forward_inference_chunks(
        self,
        detector_output: DetectorOutput,
        selected: MatchedQueries,
        image_embedding: Tensor,
        *,
        yolo_hw: tuple[int, int],
        coupled: bool,
    ) -> DFCStudentOutput:
        """Run Bridge, SAM, and UGCA in bounded chunks and retain prediction tensors only."""
        masks = []
        iou_predictions = []
        refined_logits = []
        scale_weights = []
        ugca_parts: list[UGCAOutput] = []
        instance_count = int(selected.query_index.numel())
        for start in range(0, instance_count, self.inference_instance_chunk_size):
            stop = min(start + self.inference_instance_chunk_size, instance_count)
            selected_chunk = self._slice_selected(selected, start, stop)
            boxes = self._gather(detector_output.decoded_boxes_xyxy, selected_chunk)
            yolo_height, yolo_width = yolo_hw
            divisor = boxes.new_tensor([yolo_width, yolo_height, yolo_width, yolo_height])
            boxes_normalized = (boxes / divisor).clamp(0, 1)
            semantics = self._gather(detector_output.semantic_features, selected_chunk)
            base_logits = self._gather(detector_output.base_logits, selected_chunk)
            bridge = self.feature_bridge(
                (detector_output.p3, detector_output.p4, detector_output.p5),
                boxes_normalized,
                semantics,
                selected_chunk.batch_index,
            )
            batch_size = image_embedding.shape[0]
            dense_pe = self.sam_dense_pe.expand(batch_size, -1, -1, -1)
            indices = selected_chunk.batch_index
            chunk_masks, chunk_iou, morphology = self.student_decoder.forward_instances(
                image_embedding.index_select(0, indices),
                dense_pe.index_select(0, indices),
                bridge.dense_prompt_embeddings,
                multimask_output=False,
            )
            chunk_ugca = (
                self.ugca(
                    semantics,
                    base_logits,
                    morphology,
                    boxes_normalized,
                    mask_logits=chunk_masks,
                    iou_prediction=chunk_iou,
                )
                if coupled
                else None
            )
            masks.append(chunk_masks)
            iou_predictions.append(chunk_iou)
            scale_weights.append(bridge.scale_weights)
            refined_logits.append(chunk_ugca.refined_logits if chunk_ugca is not None else base_logits)
            if chunk_ugca is not None:
                ugca_parts.append(chunk_ugca)
            # Prediction outputs above do not retain dense prompts or morphology.
            # Drop those chunk-local tensors before constructing the next chunk.
            del bridge, morphology, chunk_ugca

        boxes = self._gather(detector_output.decoded_boxes_xyxy, selected)
        yolo_height, yolo_width = yolo_hw
        divisor = boxes.new_tensor([yolo_width, yolo_height, yolo_width, yolo_height])
        boxes_normalized = (boxes / divisor).clamp(0, 1)
        semantics = self._gather(detector_output.semantic_features, selected)
        base_logits = self._gather(detector_output.base_logits, selected)
        embed_height, embed_width = self.feature_bridge.embed_hw
        compact_bridge = BridgeOutput(
            dense_prompt_embeddings=detector_output.p3.new_empty(
                (0, self.feature_bridge.embed_dim, embed_height, embed_width)
            ),
            scale_weights=torch.cat(scale_weights),
            spatial_gate=detector_output.p3.new_empty((0, 1, embed_height, embed_width)),
        )
        compact_morphology = detector_output.p3.new_empty((0, self.ugca.morphology_projection.in_features, 256, 256))
        combined_ugca = None
        if coupled:
            gate_logit_parts = [part.gate_logits for part in ugca_parts if part.gate_logits is not None]
            quality_logit_parts = [part.quality_logits for part in ugca_parts if part.quality_logits is not None]
            quality_score_parts = [part.quality_score for part in ugca_parts if part.quality_score is not None]
            combined_ugca = UGCAOutput(
                refined_logits=torch.cat([part.refined_logits for part in ugca_parts]),
                entropy=torch.cat([part.entropy for part in ugca_parts]),
                gate=torch.cat([part.gate for part in ugca_parts]),
                morphology_vector=torch.cat([part.morphology_vector for part in ugca_parts]),
                attention_weights=torch.cat([part.attention_weights for part in ugca_parts]),
                gate_logits=(torch.cat(gate_logit_parts) if len(gate_logit_parts) == len(ugca_parts) else None),
                quality_logits=(
                    torch.cat(quality_logit_parts) if len(quality_logit_parts) == len(ugca_parts) else None
                ),
                quality_score=(
                    torch.cat(quality_score_parts) if len(quality_score_parts) == len(ugca_parts) else None
                ),
            )
        return DFCStudentOutput(
            detector=detector_output,
            selected=selected,
            boxes_xyxy_normalized=boxes_normalized,
            base_logits=base_logits,
            refined_logits=torch.cat(refined_logits),
            semantic_features=semantics,
            mask_logits=torch.cat(masks),
            iou_prediction=torch.cat(iou_predictions),
            morphology=compact_morphology,
            bridge=compact_bridge,
            ugca=combined_ugca,
            class_probability_mode=str(getattr(self.detector, "class_probability_mode", "softmax")),
        )

    def forward(
        self,
        yolo_images: Tensor,
        sam_resized_images: Tensor,
        *,
        target_batch: dict[str, Tensor] | None = None,
        stage: str = "joint",
        pre_threshold: float = 0.05,
        max_instances: int = 400,
    ) -> DFCStudentOutput:
        """Run a train or inference forward while preserving the prescribed gradient boundaries."""
        stage_value = str(getattr(stage, "value", stage))
        if stage_value not in {"warmup", "joint", "II-B", "II-C", "III"}:
            raise ValueError(f"Unknown DFC-SAM stage: {stage_value}")
        coupled = stage_value in {"joint", "III"}
        detector_output = self.detector(yolo_images, coupled_grad=coupled)
        if self.training:
            if target_batch is None:
                raise ValueError("Training requires target_batch for native one-to-one assignment")
            selected = self.assignment.select_matched_positives(detector_output.raw_one2one, target_batch)
            if coupled:
                selected = self._augment_with_hard_negatives(selected, detector_output.base_logits)
        else:
            selected = self._prefilter(detector_output.base_logits, pre_threshold, max_instances)
        if selected.query_index.numel() == 0:
            empty_boxes = detector_output.decoded_boxes_xyxy.new_empty((0, 4))
            empty_logits = detector_output.base_logits.new_empty((0, detector_output.base_logits.shape[-1]))
            empty_semantics = detector_output.semantic_features.new_empty(
                (0, detector_output.semantic_features.shape[-1])
            )
            embed_height, embed_width = self.feature_bridge.embed_hw
            empty_bridge = BridgeOutput(
                dense_prompt_embeddings=detector_output.p3.new_empty(
                    (0, self.feature_bridge.embed_dim, embed_height, embed_width)
                ),
                scale_weights=detector_output.p3.new_empty((0, 3)),
                spatial_gate=detector_output.p3.new_empty((0, 1, embed_height, embed_width)),
            )
            empty_morphology = detector_output.p3.new_empty((0, self.ugca.morphology_projection.in_features, 256, 256))
            empty_ugca = None
            if coupled:
                token_count = self.ugca.grid_hw[0] * self.ugca.grid_hw[1]
                attention_dim = self.ugca.output_projection.out_features
                empty_ugca = UGCAOutput(
                    refined_logits=empty_logits,
                    entropy=empty_logits.new_empty((0,)),
                    gate=empty_logits.new_empty((0,)),
                    morphology_vector=empty_logits.new_empty((0, attention_dim)),
                    attention_weights=empty_logits.new_empty((0, 1, token_count)),
                    gate_logits=empty_logits.new_empty((0,)),
                    quality_logits=(
                        empty_logits.new_empty((0,)) if hasattr(self.ugca, "quality_network") else None
                    ),
                    quality_score=(
                        empty_logits.new_empty((0,)) if hasattr(self.ugca, "quality_network") else None
                    ),
                )
            return DFCStudentOutput(
                detector=detector_output,
                selected=selected,
                boxes_xyxy_normalized=empty_boxes,
                base_logits=empty_logits,
                refined_logits=empty_logits,
                semantic_features=empty_semantics,
                mask_logits=detector_output.p3.new_empty((0, 1, 256, 256)),
                iou_prediction=detector_output.p3.new_empty((0, 1)),
                morphology=empty_morphology,
                bridge=empty_bridge,
                ugca=empty_ugca,
                class_probability_mode=str(getattr(self.detector, "class_probability_mode", "softmax")),
            )

        if not self.training:
            with torch.no_grad():
                image_embedding = self.student_sam.image_encoder(self.student_sam.preprocess(sam_resized_images))
            return self._forward_inference_chunks(
                detector_output,
                selected,
                image_embedding,
                yolo_hw=(
                    int(yolo_images.shape[-2]),
                    int(yolo_images.shape[-1]),
                ),
                coupled=coupled,
            )

        boxes = self._gather(detector_output.decoded_boxes_xyxy, selected)
        yolo_height, yolo_width = yolo_images.shape[-2:]
        divisor = boxes.new_tensor([yolo_width, yolo_height, yolo_width, yolo_height])
        boxes_normalized = (boxes / divisor).clamp(0, 1)
        semantics = self._gather(detector_output.semantic_features, selected)
        base_logits = self._gather(detector_output.base_logits, selected)

        bridge = self.feature_bridge(
            (detector_output.p3, detector_output.p4, detector_output.p5),
            boxes_normalized,
            semantics,
            selected.batch_index,
        )
        with torch.no_grad():
            image_embedding = self.student_sam.image_encoder(self.student_sam.preprocess(sam_resized_images))
        mask_logits, iou_prediction, morphology = self._decode_chunks(
            image_embedding,
            self.sam_dense_pe,
            bridge.dense_prompt_embeddings,
            selected.batch_index,
        )
        ugca = (
            self.ugca(
                semantics,
                base_logits,
                morphology,
                boxes_normalized,
                mask_logits=mask_logits,
                iou_prediction=iou_prediction,
            )
            if coupled
            else None
        )
        refined_logits = ugca.refined_logits if ugca is not None else base_logits
        return DFCStudentOutput(
            detector=detector_output,
            selected=selected,
            boxes_xyxy_normalized=boxes_normalized,
            base_logits=base_logits,
            refined_logits=refined_logits,
            semantic_features=semantics,
            mask_logits=mask_logits,
            iou_prediction=iou_prediction,
            morphology=morphology,
            bridge=bridge,
            ugca=ugca,
            class_probability_mode=str(getattr(self.detector, "class_probability_mode", "softmax")),
        )
