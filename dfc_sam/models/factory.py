"""Construct the complete DFC-SAM student from a resolved experiment config."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from torch import nn

from .dfc_sam import DFCSAM
from .feature_bridge import DifferentiableFeatureBridge
from .rfdetr_assignment_adapter import RFDETRAssignmentAdapter
from .rfdetr_feature_bridge import RFDETRFeatureBridge
from .sam_instance_decoder import SAMInstanceDecoder
from .teacher_sam import configure_student, load_sam, load_teacher_decoder_into_student
from .ugca import UGCA, ConservativeUGCAV2, QualityAwareUGCAV3
from .yolo26_adapter import load_yolo26_adapter


def _required_path(config: Mapping[str, Any], section: str, key: str) -> Path:
    raw = config.get(section, {}).get(key)
    if raw in (None, "", "REQUIRED"):
        raise ValueError(f"Missing required config path: {section}.{key}")
    path = Path(str(raw)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def build_dfc_sam(config: Mapping[str, Any]) -> DFCSAM:
    """Strictly load the detector and configured SAM, then wire the DFC modules."""
    architecture = str(config["detector"]["architecture"])
    detector_checkpoint = _required_path(config, "weights", "detector_stage1")
    if architecture == "yolo26x":
        detector = load_yolo26_adapter(detector_checkpoint)
        assignment = None
        bridge_type = DifferentiableFeatureBridge
    elif architecture == "rfdetr_2xlarge":
        from .rfdetr_adapter import load_rfdetr_2xlarge_adapter

        detector = load_rfdetr_2xlarge_adapter(detector_checkpoint)
        assignment = RFDETRAssignmentAdapter(training_args=detector.wrapper.model.args)
        bridge_type = RFDETRFeatureBridge
    else:
        raise ValueError(f"Unsupported detector architecture: {architecture}")
    sam_variant = str(config["sam"]["variant"])
    sam_weight_key = {"vit_b": "sam_vit_b", "vit_h": "sam_vit_h"}.get(sam_variant)
    if sam_weight_key is None:
        raise ValueError(f"Unsupported SAM variant: {sam_variant}")
    student_sam: nn.Module = configure_student(
        load_sam(
            _required_path(config, "weights", sam_weight_key),
            variant=sam_variant,
        )
    )
    if config["supervision"]["mode"] == "mixed":
        load_teacher_decoder_into_student(
            student_sam,
            _required_path(config, "weights", "teacher_stage2a"),
        )

    sam_config = config["sam"]
    bridge_config = config["bridge"]
    ugca_config = config["ugca"]
    bridge = bridge_type(
        detector.pyramid_dims,
        detector.semantic_dim,
        embed_dim=int(sam_config["embed_dim"]),
        embed_hw=tuple(int(value) for value in sam_config["embed_hw"]),
        group_norm_groups=int(bridge_config["group_norm_groups"]),
        saa_hidden_dim=int(bridge_config["semantic_hidden_dim"]),
        gate_transition_cells=float(bridge_config["gate_transition_cells"]),
    )
    decoder = SAMInstanceDecoder(student_sam.mask_decoder)
    ugca_arguments = dict(
        semantic_dim=detector.semantic_dim,
        morphology_dim=int(sam_config["upscaled_dim"]),
        num_classes=int(config["detector"]["num_classes"]),
        attention_dim=int(ugca_config["attention_dim"]),
        num_heads=int(ugca_config["num_heads"]),
        grid_hw=tuple(int(value) for value in ugca_config["grid_hw"]),
        dropout=float(ugca_config["dropout"]),
        class_probability_mode=str(detector.class_probability_mode),
    )
    variant = str(ugca_config.get("variant", "v1"))
    if variant == "v1":
        ugca = UGCA(**ugca_arguments)
    elif variant == "conservative_v2":
        ugca = ConservativeUGCAV2(
            **ugca_arguments,
            gate_hidden_dim=int(ugca_config.get("gate_hidden_dim", 64)),
            residual_logit_cap=float(ugca_config.get("residual_logit_cap", 1.0)),
        )
    elif variant == "quality_aware_v3":
        ugca = QualityAwareUGCAV3(
            **ugca_arguments,
            gate_hidden_dim=int(ugca_config.get("gate_hidden_dim", 64)),
            residual_logit_cap=float(ugca_config.get("residual_logit_cap", 1.0)),
            quality_hidden_dim=int(ugca_config.get("quality_hidden_dim", 64)),
        )
    else:
        raise ValueError(f"Unknown UGCA variant: {variant}")
    return DFCSAM(
        detector,
        student_sam,
        bridge,
        decoder,
        ugca,
        assignment=assignment,
        instance_chunk_size=int(config["runtime"]["instance_chunk_size"]),
        inference_instance_chunk_size=int(config["runtime"]["inference_instance_chunk_size"]),
        training_hard_negative_ratio=float(ugca_config.get("hard_negative_ratio", 0.0)),
        training_hard_negative_max_per_image=int(ugca_config.get("hard_negative_max_per_image", 0)),
        training_hard_negative_score_threshold=float(ugca_config.get("hard_negative_score_threshold", 0.05)),
    )
