"""Paper-defined stage freeze/train policies."""

from __future__ import annotations

from enum import Enum

from torch import nn


class Stage(str, Enum):
    DETECTOR = "detector"
    TEACHER = "teacher"
    WARMUP = "warmup"
    JOINT = "joint"


def _set(module: nn.Module | None, trainable: bool) -> None:
    if module is not None:
        module.requires_grad_(trainable)


def apply_stage_policy(
    stage: Stage,
    *,
    detector: nn.Module,
    bridge: nn.Module,
    student_sam: nn.Module,
    ugca: nn.Module,
    teacher_sam: nn.Module | None = None,
    train_student_decoder_in_joint: bool = False,
    joint_detector_scope: str = "all",
    joint_training_mode: str = "full",
) -> dict[str, bool]:
    """Apply the fixed stage policy and return a concise trainability report."""
    for module in (detector, bridge, student_sam, ugca, teacher_sam):
        _set(module, False)

    if stage is Stage.DETECTOR:
        _set(detector, True)
    elif stage is Stage.TEACHER:
        if teacher_sam is None:
            raise ValueError("Teacher stage requires a Teacher SAM module")
        _set(teacher_sam.mask_decoder, True)
    elif stage is Stage.WARMUP:
        _set(bridge, True)
        _set(student_sam.mask_decoder, True)
    elif stage is Stage.JOINT:
        if joint_training_mode in {"ugca_only", "ugca_ranking", "rf_frozen_sequential"}:
            _set(ugca, True)
        elif joint_training_mode == "controlled":
            _set(bridge, True)
            _set(student_sam.mask_decoder, train_student_decoder_in_joint)
        elif joint_training_mode == "controlled_adaptive":
            # Detector remains immutable. Bridge/decoder adapt the mask path,
            # while UGCA follows the resulting morphology/quality distribution
            # under separately constrained optimizer groups.
            _set(bridge, True)
            _set(student_sam.mask_decoder, train_student_decoder_in_joint)
            _set(ugca, True)
        elif joint_training_mode == "detector_head_adaptive":
            # A deliberately narrow upstream probe: only the detector head and
            # the quality-aware classifier may adapt.  The bridge and every SAM
            # component remain fixed, so any validation change is attributable
            # to candidate / base-logit adaptation rather than mask-path drift.
            head = getattr(detector, "head", None)
            if head is None:
                raise ValueError("detector_head_adaptive requires detector.head")
            _set(head, True)
            _set(ugca, True)
        elif joint_training_mode == "full":
            if joint_detector_scope == "all":
                _set(detector, True)
            elif joint_detector_scope == "head":
                head = getattr(detector, "head", None)
                if head is None:
                    raise ValueError("joint_detector_scope=head requires detector.head")
                _set(head, True)
            else:
                raise ValueError(f"Unknown joint_detector_scope: {joint_detector_scope}")
            _set(bridge, True)
            _set(student_sam.mask_decoder, train_student_decoder_in_joint)
            _set(ugca, True)
        else:
            raise ValueError(f"Unknown joint_training_mode: {joint_training_mode}")
    else:
        raise ValueError(f"Unknown stage: {stage}")

    # Encoders and the complete teacher outside its decoder adaptation remain frozen.
    _set(student_sam.image_encoder, False)
    _set(student_sam.prompt_encoder, False)
    if teacher_sam is not None:
        _set(teacher_sam.image_encoder, False)
        _set(teacher_sam.prompt_encoder, False)
        if stage is not Stage.TEACHER:
            _set(teacher_sam.mask_decoder, False)

    return {
        "detector": any(parameter.requires_grad for parameter in detector.parameters()),
        "bridge": any(parameter.requires_grad for parameter in bridge.parameters()),
        "student_image_encoder": any(parameter.requires_grad for parameter in student_sam.image_encoder.parameters()),
        "student_mask_decoder": any(parameter.requires_grad for parameter in student_sam.mask_decoder.parameters()),
        "ugca": any(parameter.requires_grad for parameter in ugca.parameters()),
        "teacher_image_encoder": bool(
            teacher_sam is not None
            and any(parameter.requires_grad for parameter in teacher_sam.image_encoder.parameters())
        ),
        "teacher_prompt_encoder": bool(
            teacher_sam is not None
            and any(parameter.requires_grad for parameter in teacher_sam.prompt_encoder.parameters())
        ),
        "teacher_mask_decoder": bool(
            teacher_sam is not None
            and any(parameter.requires_grad for parameter in teacher_sam.mask_decoder.parameters())
        ),
    }


def apply_stage_modes(
    stage: Stage,
    *,
    detector: nn.Module,
    bridge: nn.Module,
    student_sam: nn.Module,
    ugca: nn.Module,
    teacher_sam: nn.Module | None = None,
    joint_detector_scope: str = "all",
    joint_training_mode: str = "full",
) -> None:
    """Set training/evaluation modes without updating frozen normalization state."""
    for module in (detector, bridge, student_sam, ugca, teacher_sam):
        if module is not None:
            module.eval()
    if stage is Stage.TEACHER:
        if teacher_sam is None:
            raise ValueError("Teacher stage requires a Teacher SAM module")
        teacher_sam.mask_decoder.train()
    elif stage is Stage.WARMUP:
        bridge.train()
        student_sam.mask_decoder.train()
    elif stage is Stage.JOINT:
        if joint_training_mode in {"ugca_only", "ugca_ranking", "rf_frozen_sequential"}:
            ugca.train()
        elif joint_training_mode == "controlled":
            bridge.train()
            if any(parameter.requires_grad for parameter in student_sam.mask_decoder.parameters()):
                student_sam.mask_decoder.train()
        elif joint_training_mode == "controlled_adaptive":
            bridge.train()
            if any(parameter.requires_grad for parameter in student_sam.mask_decoder.parameters()):
                student_sam.mask_decoder.train()
            ugca.train()
        elif joint_training_mode == "detector_head_adaptive":
            head = getattr(detector, "head", None)
            if head is None:
                raise ValueError("detector_head_adaptive requires detector.head")
            # detector was put in eval() above, so frozen backbone/neck buffers
            # stay immutable while the trainable prediction head uses train mode.
            head.train()
            ugca.train()
        elif joint_training_mode == "full":
            if joint_detector_scope == "all":
                detector.train()
            elif joint_detector_scope == "head":
                head = getattr(detector, "head", None)
                if head is None:
                    raise ValueError("joint_detector_scope=head requires detector.head")
                # Keep frozen backbone/neck normalization buffers fixed.
                head.train()
            else:
                raise ValueError(f"Unknown joint_detector_scope: {joint_detector_scope}")
            bridge.train()
            if any(parameter.requires_grad for parameter in student_sam.mask_decoder.parameters()):
                student_sam.mask_decoder.train()
            ugca.train()
        else:
            raise ValueError(f"Unknown joint_training_mode: {joint_training_mode}")
    elif stage is Stage.DETECTOR:
        detector.train()
    else:
        raise ValueError(f"Unknown stage: {stage}")
    # These modules are frozen by protocol regardless of their parent's mode.
    student_sam.image_encoder.eval()
    student_sam.prompt_encoder.eval()
    if teacher_sam is not None:
        teacher_sam.image_encoder.eval()
        teacher_sam.prompt_encoder.eval()
