"""Instance-batched SAM mask decoder without the original N² repeat_interleave behavior."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


class SAMInstanceDecoder(nn.Module):
    """Expose the SAM decoder's instance-aligned path and upscaled morphology."""

    def __init__(self, mask_decoder: nn.Module) -> None:
        super().__init__()
        self.mask_decoder = mask_decoder

    def forward_instances(
        self,
        image_embeddings: Tensor,
        image_pe: Tensor,
        dense_prompt_embeddings: Tensor,
        multimask_output: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Decode N aligned instances and return logits, IoU prediction, and dense morphology."""
        if image_embeddings.ndim != 4:
            raise ValueError("image_embeddings must have shape [N,C,H,W]")
        if image_embeddings.shape != image_pe.shape or image_embeddings.shape != dense_prompt_embeddings.shape:
            raise ValueError("image embeddings, PE, and dense prompts must have identical [N,C,H,W] shapes")
        instance_count = image_embeddings.shape[0]
        decoder: Any = self.mask_decoder

        output_tokens = torch.cat((decoder.iou_token.weight, decoder.mask_tokens.weight), dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(instance_count, -1, -1)
        empty_sparse = torch.empty(
            instance_count,
            0,
            output_tokens.shape[-1],
            device=image_embeddings.device,
            dtype=image_embeddings.dtype,
        )
        tokens = torch.cat((output_tokens.to(dtype=image_embeddings.dtype), empty_sparse), dim=1)
        source = image_embeddings + dense_prompt_embeddings
        batch, channels, height, width = source.shape
        hidden, source = decoder.transformer(source, image_pe, tokens)
        iou_token_out = hidden[:, 0]
        mask_tokens_out = hidden[:, 1 : 1 + decoder.num_mask_tokens]

        source = source.transpose(1, 2).view(batch, channels, height, width)
        upscaled = decoder.output_upscaling(source)
        hyper = torch.stack(
            [
                decoder.output_hypernetworks_mlps[index](mask_tokens_out[:, index])
                for index in range(decoder.num_mask_tokens)
            ],
            dim=1,
        )
        batch, channels, height, width = upscaled.shape
        masks = (hyper @ upscaled.view(batch, channels, height * width)).view(batch, -1, height, width)
        iou_prediction = decoder.iou_prediction_head(iou_token_out)

        selected = slice(1, None) if multimask_output else slice(0, 1)
        return masks[:, selected], iou_prediction[:, selected], upscaled
