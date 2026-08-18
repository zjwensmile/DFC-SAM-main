"""Box-prompted SAM Teacher forward pass with frozen encoders."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from dfc_sam.data.geometry import GeometryRecord, boxes_original_to_sam_xyxy


@dataclass
class TeacherOutput:
    """Low-resolution SAM candidates in flattened target order."""

    mask_logits: Tensor
    iou_prediction: Tensor
    target_batch_index: Tensor
    target_instance_index: Tensor
    target_flat_index: Tensor


class BoxPromptedTeacher(nn.Module):
    """Adapt only SAM's mask decoder using sealed ground-truth box prompts."""

    def __init__(self, sam: nn.Module, *, low_res_hw: tuple[int, int] = (256, 256)) -> None:
        super().__init__()
        self.sam = sam
        self.low_res_hw = low_res_hw
        self.register_buffer(
            "dense_pe",
            sam.prompt_encoder.get_dense_pe().detach().clone(),
            persistent=False,
        )

    def forward(
        self,
        sam_resized_images: Tensor,
        boxes_original_xyxy: Tensor,
        target_batch_index: Tensor,
        geometries: list[GeometryRecord],
        *,
        multimask_output: bool,
    ) -> TeacherOutput:
        if sam_resized_images.ndim != 4 or sam_resized_images.shape[1] != 3:
            raise ValueError("SAM images must have shape [B,3,H,W]")
        if boxes_original_xyxy.ndim != 2 or boxes_original_xyxy.shape[1] != 4:
            raise ValueError("Teacher boxes must have shape [N,4]")
        if target_batch_index.shape != (boxes_original_xyxy.shape[0],):
            raise ValueError("Teacher target_batch_index must align with boxes")
        if len(geometries) != sam_resized_images.shape[0]:
            raise ValueError("One geometry record is required per image")

        with torch.no_grad():
            image_embedding = self.sam.image_encoder(self.sam.preprocess(sam_resized_images))

        all_masks = []
        all_iou = []
        all_batch_indices = []
        all_instance_indices = []
        for batch_index, geometry in enumerate(geometries):
            flat_indices = (target_batch_index == batch_index).nonzero(as_tuple=False).flatten()
            if flat_indices.numel() == 0:
                continue
            prompt_boxes = boxes_original_to_sam_xyxy(
                boxes_original_xyxy.index_select(0, flat_indices),
                geometry,
            )
            with torch.no_grad():
                sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                    points=None,
                    boxes=prompt_boxes,
                    masks=None,
                )
            masks, predicted_iou = self.sam.mask_decoder(
                image_embeddings=image_embedding[batch_index : batch_index + 1],
                image_pe=self.dense_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            all_masks.append(masks)
            all_iou.append(predicted_iou)
            all_batch_indices.append(torch.full_like(flat_indices, batch_index))
            all_instance_indices.append(torch.arange(flat_indices.numel(), device=flat_indices.device))

        candidate_count = 3 if multimask_output else 1
        if not all_masks:
            empty_masks = sam_resized_images.new_empty((0, candidate_count, *self.low_res_hw))
            empty_iou = sam_resized_images.new_empty((0, candidate_count))
            empty_index = target_batch_index.new_empty((0,))
            return TeacherOutput(empty_masks, empty_iou, empty_index, empty_index, empty_index)
        return TeacherOutput(
            mask_logits=torch.cat(all_masks),
            iou_prediction=torch.cat(all_iou),
            target_batch_index=torch.cat(all_batch_indices),
            target_instance_index=torch.cat(all_instance_indices),
            target_flat_index=torch.cat(
                [
                    (target_batch_index == batch_index).nonzero(as_tuple=False).flatten()
                    for batch_index in range(len(geometries))
                    if bool((target_batch_index == batch_index).any())
                ]
            ),
        )
