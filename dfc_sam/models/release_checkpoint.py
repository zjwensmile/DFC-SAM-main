"""Self-contained RF-DETR-2XL + SAM-H release checkpoint support."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from dfc_sam.constants import PANNUKE_CLASSES

from .dfc_sam import DFCSAM
from .rfdetr_adapter import RFDETR2XLAdapter
from .rfdetr_assignment_adapter import RFDETRAssignmentAdapter
from .rfdetr_feature_bridge import RFDETRFeatureBridge
from .sam_instance_decoder import SAMInstanceDecoder
from .teacher_sam import configure_student
from .ugca import QualityAwareUGCAV3

RELEASE_FORMAT = "dfc-sam-rfdetr-2xl-v1"


def _build_empty_detector() -> RFDETR2XLAdapter:
    """Construct the licensed 2XL architecture without downloading base weights."""
    try:
        from rfdetr import RFDETR2XLarge
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError(
            "RF-DETR-2XL requires rfdetr-plus==1.0.2. Install it only after "
            "reviewing and accepting the Roboflow PML-1.0 license."
        ) from exc
    wrapper = RFDETR2XLarge(
        pretrain_weights=None,
        force_no_pretrain=True,
        num_classes=len(PANNUKE_CLASSES),
        accept_platform_model_license=True,
    )
    wrapper.model.class_names = [name.title() for name in PANNUKE_CLASSES]
    wrapper.model.model.eval()
    return RFDETR2XLAdapter(wrapper)


def _build_empty_sam(variant: str) -> nn.Module:
    if variant != "vit_h":
        raise ValueError(f"This release supports SAM-H only, got {variant!r}")
    from segment_anything import sam_model_registry

    return configure_student(sam_model_registry[variant](checkpoint=None))


def build_release_model(config: Mapping[str, Any]) -> DFCSAM:
    """Build an uninitialized architecture ready for a complete release state dict."""
    if str(config["detector"]["architecture"]) != "rfdetr_2xlarge":
        raise ValueError("Release checkpoint is not RF-DETR-2XL")
    detector = _build_empty_detector()
    student_sam = _build_empty_sam(str(config["sam"]["variant"]))
    sam_config = config["sam"]
    bridge_config = config["bridge"]
    ugca_config = config["ugca"]
    if str(ugca_config.get("variant")) != "quality_aware_v3":
        raise ValueError("Release checkpoint must use UGCA quality_aware_v3")
    bridge = RFDETRFeatureBridge(
        detector.pyramid_dims,
        detector.semantic_dim,
        embed_dim=int(sam_config["embed_dim"]),
        embed_hw=tuple(int(value) for value in sam_config["embed_hw"]),
        group_norm_groups=int(bridge_config["group_norm_groups"]),
        saa_hidden_dim=int(bridge_config["semantic_hidden_dim"]),
        gate_transition_cells=float(bridge_config["gate_transition_cells"]),
    )
    decoder = SAMInstanceDecoder(student_sam.mask_decoder)
    ugca = QualityAwareUGCAV3(
        semantic_dim=detector.semantic_dim,
        morphology_dim=int(sam_config["upscaled_dim"]),
        num_classes=int(config["detector"]["num_classes"]),
        attention_dim=int(ugca_config["attention_dim"]),
        num_heads=int(ugca_config["num_heads"]),
        grid_hw=tuple(int(value) for value in ugca_config["grid_hw"]),
        dropout=float(ugca_config["dropout"]),
        class_probability_mode=str(detector.class_probability_mode),
        gate_hidden_dim=int(ugca_config.get("gate_hidden_dim", 64)),
        residual_logit_cap=float(ugca_config.get("residual_logit_cap", 1.0)),
        quality_hidden_dim=int(ugca_config.get("quality_hidden_dim", 64)),
    )
    return DFCSAM(
        detector,
        student_sam,
        bridge,
        decoder,
        ugca,
        assignment=RFDETRAssignmentAdapter(training_args=detector.wrapper.model.args),
        instance_chunk_size=int(config["runtime"]["instance_chunk_size"]),
        inference_instance_chunk_size=int(config["runtime"]["inference_instance_chunk_size"]),
        training_hard_negative_ratio=float(ugca_config.get("hard_negative_ratio", 0.0)),
        training_hard_negative_max_per_image=int(ugca_config.get("hard_negative_max_per_image", 0)),
        training_hard_negative_score_threshold=float(ugca_config.get("hard_negative_score_threshold", 0.05)),
    )


def read_release_checkpoint(path: str | Path) -> dict[str, Any]:
    """Safely read a tensor/basic-types-only inference checkpoint on CPU."""
    resolved = Path(path).expanduser().resolve()
    payload = torch.load(resolved, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, dict) or payload.get("format") != RELEASE_FORMAT:
        raise ValueError(f"Not a {RELEASE_FORMAT} checkpoint: {resolved}")
    if not isinstance(payload.get("model"), dict) or not isinstance(payload.get("config"), dict):
        raise TypeError("Release checkpoint is missing model/config dictionaries")
    return payload


def load_release_model(path: str | Path, *, device: str | torch.device = "cpu") -> tuple[DFCSAM, dict[str, Any]]:
    """Construct and strictly restore the complete detector, SAM-H, DFB, and UGCA."""
    payload = read_release_checkpoint(path)
    model = build_release_model(payload["config"])
    incompatible = model.load_state_dict(payload["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Release state mismatch: {incompatible}")
    # Do not retain a second mapped copy of the ~2.9 GB state after restoration.
    del payload["model"]
    model.to(torch.device(device)).eval()
    return model, payload
