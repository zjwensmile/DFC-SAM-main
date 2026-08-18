"""Pinned generic SAM construction and freeze policy."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def load_sam(checkpoint: str | Path, *, variant: str) -> nn.Module:
    """Strictly load a supported official SAM checkpoint by configured variant."""
    if variant not in {"vit_b", "vit_h"}:
        raise ValueError(f"Unsupported SAM variant: {variant}")
    from segment_anything import sam_model_registry

    model = sam_model_registry[variant](checkpoint=str(Path(checkpoint).expanduser().resolve()))
    return model


def load_sam_vit_b(checkpoint: str | Path) -> nn.Module:
    """Backward-compatible loader for the established SAM-ViT-B pipelines."""
    return load_sam(checkpoint, variant="vit_b")


def configure_teacher(sam: nn.Module) -> nn.Module:
    """Freeze teacher encoders and leave only the mask decoder trainable."""
    sam.requires_grad_(False)
    sam.mask_decoder.requires_grad_(True)
    return sam


def configure_student(sam: nn.Module) -> nn.Module:
    """Freeze the student image encoder; prompt encoder is unused and frozen."""
    sam.image_encoder.requires_grad_(False)
    sam.prompt_encoder.requires_grad_(False)
    sam.mask_decoder.requires_grad_(True)
    return sam


def load_teacher_decoder_into_student(student_sam: nn.Module, checkpoint: str | Path) -> str:
    """Strictly copy a ratio-specific Teacher decoder into a mixed-run student."""
    path = Path(checkpoint).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("Teacher checkpoint must contain a dictionary payload")

    state = None
    for key in ("teacher_mask_decoder", "mask_decoder"):
        candidate = payload.get(key)
        if isinstance(candidate, dict):
            state = candidate
            break
    if state is None and isinstance(payload.get("model"), dict):
        model_state = payload["model"]
        prefixes = (
            "mask_decoder.",
            "teacher_sam.mask_decoder.",
            "module.mask_decoder.",
            "module.teacher_sam.mask_decoder.",
        )
        for prefix in prefixes:
            extracted = {key[len(prefix) :]: value for key, value in model_state.items() if key.startswith(prefix)}
            if extracted:
                state = extracted
                break
    if state is None:
        raise KeyError("Teacher checkpoint does not expose a mask-decoder state")
    student_sam.mask_decoder.load_state_dict(state, strict=True)
    return str(path)
