"""YAML configuration loading, inheritance, and protocol validation."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .constants import NUM_CLASSES, PANNUKE_CLASSES, PANNUKE_FOLD_ROTATIONS

MIXED_MASK_RATIOS = (0.10, 0.20, 0.30, 0.50)
MIXED_STRATEGIES = ("no_pseudo", "naive_mixed", "qws_mixed")


class ConfigError(ValueError):
    """Raised when a resolved experiment violates the reproduction protocol."""


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries while replacing non-dictionary values."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ConfigError(f"Top-level YAML must be a mapping: {path}")
    return payload


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML file with optional `base` inheritance and merge-style `includes`."""
    config_path = Path(path).expanduser().resolve()
    payload = _read_yaml(config_path)
    resolved: dict[str, Any] = {}

    base = payload.pop("base", None)
    if base is not None:
        base_path = (config_path.parent / str(base)).resolve()
        resolved = deep_merge(resolved, load_config(base_path))

    includes = payload.pop("includes", None)
    if includes is not None:
        if isinstance(includes, dict):
            include_values = includes.values()
        elif isinstance(includes, list | tuple):
            include_values = includes
        else:
            raise ConfigError(f"`includes` must be a mapping or sequence: {config_path}")
        for include in include_values:
            include_path = (config_path.parent / str(include)).resolve()
            resolved = deep_merge(resolved, load_config(include_path))

    resolved = deep_merge(resolved, payload)
    resolved["_meta"] = {
        **resolved.get("_meta", {}),
        "resolved_from": str(config_path),
    }
    return resolved


def validate_experiment_config(config: dict[str, Any]) -> None:
    """Reject protocol-breaking settings before a run is created."""
    experiment = config.get("experiment", {})
    if experiment.get("protocol") not in {"pannuke_from_generic", "pannuke_standard_3fold"}:
        raise ConfigError("Main experiments must use protocol=pannuke_from_generic or pannuke_standard_3fold")
    split_id = experiment.get("split_id")
    if split_id not in PANNUKE_FOLD_ROTATIONS:
        raise ConfigError(f"split_id must be one of {tuple(PANNUKE_FOLD_ROTATIONS)}, got {split_id!r}")

    if int(config.get("num_classes", config.get("detector", {}).get("num_classes", -1))) != NUM_CLASSES:
        raise ConfigError(f"PanNuke must have exactly {NUM_CLASSES} positive classes")
    names = tuple(str(name).lower() for name in config.get("class_names", PANNUKE_CLASSES))
    if names != PANNUKE_CLASSES:
        raise ConfigError(f"PanNuke class order mismatch: {names!r}")

    detector = config.get("detector", {})
    sam_variant = str(config.get("sam", {}).get("variant", ""))
    if sam_variant not in {"vit_b", "vit_h"}:
        raise ConfigError(f"sam.variant must be vit_b or vit_h, got {sam_variant!r}")
    architecture = str(detector.get("architecture", ""))
    joint_training_mode = str(config.get("train", {}).get("joint_training_mode", "full"))
    if architecture not in {"yolo26x", "rfdetr_2xlarge"}:
        raise ConfigError(f"Unsupported detector architecture: {architecture!r}")
    if architecture == "rfdetr_2xlarge":
        if int(detector.get("imgsz", 0)) != 880:
            raise ConfigError("RF-DETR-2XL DFC-SAM integration is frozen at detector.imgsz=880")
        if bool(detector.get("resize_antialias", True)):
            raise ConfigError("RF-DETR-2XL requires detector.resize_antialias=false")
        detector_frozen = bool(detector.get("frozen_in_dfc_sam", False))
        detector_coupled = bool(detector.get("dfc_coupled_mode", False))
        if joint_training_mode == "full":
            if detector_frozen or not detector_coupled:
                raise ConfigError(
                    "Full RF-DETR Stage III requires frozen_in_dfc_sam=false "
                    "and dfc_coupled_mode=true"
                )
        elif not detector_frozen or detector_coupled:
            raise ConfigError(
                "Non-joint RF-DETR configurations require frozen_in_dfc_sam=true "
                "and dfc_coupled_mode=false"
            )
    inference = config.get("inference", {})
    for key in ("logit_blend", "quality_power", "sam_iou_power", "mask_stability_power"):
        value = float(inference.get(key, 1.0))
        if not 0.0 <= value <= 1.0:
            raise ConfigError(f"inference.{key} must be in [0,1]")
    stability_delta = float(inference.get("mask_stability_delta", 0.05))
    if not 0.0 <= stability_delta <= 0.5:
        raise ConfigError("inference.mask_stability_delta must be in [0,0.5]")
    tta = inference.get("tta", {})
    if tta and bool(tta.get("enabled", False)):
        if tuple(tta.get("views", ())) != ("identity", "hflip", "vflip", "rot180"):
            raise ConfigError("inference.tta.views must be the frozen four-view geometric set")
        if int(tta.get("view_count", 0)) != 4:
            raise ConfigError("inference.tta.view_count must be 4 when enabled")
        votes = int(tta.get("min_votes", 0))
        if not 1 <= votes <= 4:
            raise ConfigError("inference.tta.min_votes must be in [1,4]")
        for key in ("match_iou", "match_mask_threshold", "pre_threshold", "mask_threshold"):
            if not 0.0 <= float(tta.get(key, -1.0)) <= 1.0:
                raise ConfigError(f"inference.tta.{key} must be in [0,1]")
        if not -0.5 <= float(tta.get("final_threshold_shift", -1.0)) <= 0.5:
            raise ConfigError("inference.tta.final_threshold_shift must be in [-0.5,0.5]")
        if str(tta.get("mask_fusion")) not in {"mean", "max"}:
            raise ConfigError("inference.tta.mask_fusion must be mean or max")
        if str(tta.get("class_fusion")) not in {"mean", "score_weighted"}:
            raise ConfigError("inference.tta.class_fusion must be mean or score_weighted")
        if str(tta.get("class_matching")) not in {"agnostic", "same"}:
            raise ConfigError("inference.tta.class_matching must be agnostic or same")
        class_votes = tta.get("min_votes_by_class")
        if class_votes is not None and (
            not isinstance(class_votes, list | tuple)
            or len(class_votes) != NUM_CLASSES
            or any(not 1 <= int(value) <= 4 for value in class_votes)
        ):
            raise ConfigError("inference.tta.min_votes_by_class must contain five values in [1,4]")
        class_shifts = tta.get("final_threshold_shifts_by_class")
        if class_shifts is not None and (
            not isinstance(class_shifts, list | tuple)
            or len(class_shifts) != NUM_CLASSES
            or any(not -0.5 <= float(value) <= 0.5 for value in class_shifts)
        ):
            raise ConfigError("inference.tta.final_threshold_shifts_by_class must contain five values in [-0.5,0.5]")
    for key in ("quality_powers_by_class", "final_score_thresholds_by_class"):
        values = inference.get(key)
        if values is not None:
            if not isinstance(values, list | tuple) or len(values) != NUM_CLASSES:
                raise ConfigError(f"inference.{key} must contain {NUM_CLASSES} values")
            if any(not 0.0 <= float(value) <= 1.0 for value in values):
                raise ConfigError(f"inference.{key} values must be in [0,1]")
    if detector.get("use_box_nms", False) or inference.get("use_box_nms", False):
        raise ConfigError("Box NMS is forbidden for the main DFC-SAM protocol")
    if inference.get("use_mask_nms", False):
        raise ConfigError("Mask NMS is forbidden for the main DFC-SAM protocol")
    if config.get("sam", {}).get("student_uses_prompt_encoder", True):
        raise ConfigError("The student must not call the SAM prompt encoder")
    if config.get("ugca", {}).get("use_mask_attention_bias", True):
        raise ConfigError("Mask-aware attention bias is not part of the specified UGCA")

    supervision = config.get("supervision", {})
    mode = supervision.get("mode")
    strategy = supervision.get("strategy")
    ratio = supervision.get("mask_ratio")
    use_teacher = bool(supervision.get("use_teacher", False))
    use_pseudo_bank = bool(supervision.get("use_pseudo_bank", False))
    use_qwpm = bool(supervision.get("use_qwpm", False))
    initialization = config.get("initialization", {})
    student_sam_initialization = initialization.get("student_sam")
    student_initialization = initialization.get("student_decoder")

    if mode == "full":
        if strategy != "full" or float(ratio) != 1.0:
            raise ConfigError("Full supervision requires strategy=full and mask_ratio=1.0")
        if use_teacher or use_pseudo_bank or use_qwpm:
            raise ConfigError("Full supervision forbids Teacher, pseudo-bank, and QWPM")
        expected_initialization = f"generic_sam_{sam_variant}"
        if (
            student_sam_initialization != expected_initialization
            or student_initialization != expected_initialization
        ):
            raise ConfigError(
                f"Full supervision SAM and decoder must start from {expected_initialization}"
            )
        for key in ("teacher_stage2a", "quality_calibration", "pseudo_bank"):
            if config.get("weights", {}).get(key) not in (None, ""):
                raise ConfigError(f"Full supervision forbids weights.{key}")
    elif mode == "mixed":
        if sam_variant != "vit_b":
            raise ConfigError("Existing mixed-supervision Teacher checkpoints are restricted to SAM-ViT-B")
        if student_sam_initialization != "generic_sam_vit_b":
            raise ConfigError("Mixed supervision student SAM must start from generic_sam_vit_b")
        try:
            normalized_ratio = float(ratio)
        except (TypeError, ValueError) as error:
            raise ConfigError(f"Mixed supervision has invalid mask_ratio={ratio!r}") from error
        if not any(abs(normalized_ratio - expected) < 1.0e-9 for expected in MIXED_MASK_RATIOS):
            raise ConfigError(f"Mixed mask_ratio must be one of {MIXED_MASK_RATIOS}, got {ratio!r}")
        if strategy not in MIXED_STRATEGIES:
            raise ConfigError(f"Mixed strategy must be one of {MIXED_STRATEGIES}, got {strategy!r}")
        if not use_teacher:
            raise ConfigError("Every mixed strategy must share the ratio-specific Teacher initialization")
        expected_pseudo = strategy in {"naive_mixed", "qws_mixed"}
        if use_pseudo_bank != expected_pseudo:
            raise ConfigError(f"{strategy} requires use_pseudo_bank={expected_pseudo}")
        if use_qwpm != (strategy == "qws_mixed"):
            raise ConfigError("Only qws_mixed may enable QWPM")
        if student_initialization != "teacher_stage2a":
            raise ConfigError("Mixed strategies must share the ratio-specific Teacher decoder initialization")
        manifest = supervision.get("manifest")
        if manifest in (None, "", "REQUIRED"):
            raise ConfigError("Mixed supervision requires an immutable supervision manifest")
        if config.get("weights", {}).get("quality_calibration") in (None, "", "REQUIRED"):
            raise ConfigError("Mixed supervision requires a ratio-specific quality calibration artifact path")
    else:
        raise ConfigError(f"supervision.mode must be full or mixed, got {mode!r}")

    flow = config.get("train", {}).get("flow")
    if flow not in {"warmup_joint", "direct_joint"}:
        raise ConfigError("train.flow must be warmup_joint or direct_joint")
    if mode == "mixed" and flow != "warmup_joint":
        raise ConfigError("Mixed supervision must use warmup_joint for paired comparison")

    train = config.get("train", {})
    joint_training_mode = str(train.get("joint_training_mode", "full"))
    if joint_training_mode not in {
        "full",
        "ugca_only",
        "ugca_ranking",
        "rf_frozen_sequential",
        "controlled",
        "controlled_adaptive",
        "detector_head_adaptive",
    }:
        raise ConfigError(
            "train.joint_training_mode must be full, ugca_only, ugca_ranking, rf_frozen_sequential, "
            "controlled, controlled_adaptive, or detector_head_adaptive"
        )
    if str(train.get("joint_detector_scope", "all")) not in {"all", "head"}:
        raise ConfigError("train.joint_detector_scope must be all or head")
    if joint_training_mode == "ugca_only":
        loss = config.get("loss", {})
        if float(loss.get("lambda_det", 0.0)) != 0.0 or float(loss.get("lambda_seg", 0.0)) != 0.0:
            raise ConfigError("ugca_only requires loss.lambda_det=0 and loss.lambda_seg=0")
    elif joint_training_mode == "ugca_ranking":
        loss = config.get("loss", {})
        if flow != "warmup_joint":
            raise ConfigError("ugca_ranking Joint requires train.flow=warmup_joint")
        if float(loss.get("lambda_det", 0.0)) != 0.0 or float(loss.get("lambda_seg", 0.0)) != 0.0:
            raise ConfigError("ugca_ranking requires loss.lambda_det=0 and loss.lambda_seg=0")
        if float(loss.get("lambda_ugca", 0.0)) <= 0.0:
            raise ConfigError("ugca_ranking requires loss.lambda_ugca>0")
        if bool(train.get("train_student_decoder_in_joint", False)):
            raise ConfigError("ugca_ranking requires train_student_decoder_in_joint=false")
        joint_initialization = config.get("weights", {}).get("joint_initialization")
        if joint_initialization in (None, "", "REQUIRED"):
            raise ConfigError("ugca_ranking requires weights.joint_initialization")
        if str(config.get("ugca", {}).get("variant", "v1")) != "quality_aware_v3":
            raise ConfigError("ugca_ranking requires ugca.variant=quality_aware_v3")
        if float(loss.get("ugca_quality_ranking_loss_weight", 0.0)) <= 0.0:
            raise ConfigError("ugca_ranking requires loss.ugca_quality_ranking_loss_weight>0")
        rates = train.get("learning_rates", {})
        for key in ("joint_ugca_shared", "joint_ugca_quality"):
            if float(rates.get(key, 0.0)) <= 0.0:
                raise ConfigError(f"ugca_ranking requires positive train.learning_rates.{key}")
        if not bool(config.get("validation", {}).get("initialize_joint_best_from_parent", False)):
            raise ConfigError("ugca_ranking requires validation.initialize_joint_best_from_parent=true")
    elif joint_training_mode == "rf_frozen_sequential":
        loss = config.get("loss", {})
        if architecture != "rfdetr_2xlarge":
            raise ConfigError("rf_frozen_sequential is reserved for RF-DETR-2XL")
        if flow != "warmup_joint":
            raise ConfigError("rf_frozen_sequential requires train.flow=warmup_joint")
        if float(loss.get("lambda_det", 0.0)) != 0.0:
            raise ConfigError("rf_frozen_sequential keeps RF-DETR immutable and requires lambda_det=0")
        if float(loss.get("lambda_seg", 0.0)) <= 0.0 or float(loss.get("lambda_ugca", 0.0)) <= 0.0:
            raise ConfigError("rf_frozen_sequential requires positive segmentation and UGCA weights")
        if bool(train.get("train_student_decoder_in_joint", False)):
            raise ConfigError("rf_frozen_sequential freezes Bridge/SAM during the Joint UGCA stage")
        if str(config.get("ugca", {}).get("variant", "v1")) != "quality_aware_v3":
            raise ConfigError("rf_frozen_sequential requires ugca.variant=quality_aware_v3")
        if float(loss.get("ugca_quality_ranking_loss_weight", 0.0)) <= 0.0:
            raise ConfigError("rf_frozen_sequential requires positive quality ranking loss")
        rates = train.get("learning_rates", {})
        for key in ("joint_ugca_shared", "joint_ugca_quality"):
            if float(rates.get(key, 0.0)) <= 0.0:
                raise ConfigError(f"rf_frozen_sequential requires positive train.learning_rates.{key}")
        if not bool(config.get("validation", {}).get("initialize_joint_best_from_parent", False)):
            raise ConfigError("rf_frozen_sequential requires parent-initialized Joint model selection")
    elif joint_training_mode == "controlled":
        loss = config.get("loss", {})
        if flow != "warmup_joint":
            raise ConfigError("controlled Joint requires train.flow=warmup_joint")
        if float(loss.get("lambda_det", 0.0)) != 0.0:
            raise ConfigError("controlled Joint requires loss.lambda_det=0")
        if float(loss.get("lambda_seg", 0.0)) <= 0.0:
            raise ConfigError("controlled Joint requires loss.lambda_seg>0")
        if float(loss.get("lambda_ugca", 0.0)) != 0.0:
            raise ConfigError("controlled Joint keeps UGCA fixed and requires loss.lambda_ugca=0")
        if not bool(train.get("train_student_decoder_in_joint", False)):
            raise ConfigError("controlled Joint requires train_student_decoder_in_joint=true")
        joint_initialization = config.get("weights", {}).get("joint_initialization")
        if joint_initialization in (None, "", "REQUIRED"):
            raise ConfigError("controlled Joint requires weights.joint_initialization")
    elif joint_training_mode == "controlled_adaptive":
        loss = config.get("loss", {})
        if flow != "warmup_joint":
            raise ConfigError("controlled_adaptive Joint requires train.flow=warmup_joint")
        if float(loss.get("lambda_det", 0.0)) != 0.0:
            raise ConfigError("controlled_adaptive Joint requires loss.lambda_det=0")
        if float(loss.get("lambda_seg", 0.0)) <= 0.0:
            raise ConfigError("controlled_adaptive Joint requires loss.lambda_seg>0")
        if float(loss.get("lambda_ugca", 0.0)) <= 0.0:
            raise ConfigError("controlled_adaptive Joint requires loss.lambda_ugca>0")
        if not bool(train.get("train_student_decoder_in_joint", False)):
            raise ConfigError("controlled_adaptive Joint requires train_student_decoder_in_joint=true")
        joint_initialization = config.get("weights", {}).get("joint_initialization")
        if joint_initialization in (None, "", "REQUIRED"):
            raise ConfigError("controlled_adaptive Joint requires weights.joint_initialization")
        if str(config.get("ugca", {}).get("variant", "v1")) != "quality_aware_v3":
            raise ConfigError("controlled_adaptive Joint requires ugca.variant=quality_aware_v3")
        rates = train.get("learning_rates", {})
        for key in ("joint_bridge", "joint_student_decoder", "joint_ugca_shared", "joint_ugca_quality"):
            if float(rates.get(key, 0.0)) <= 0.0:
                raise ConfigError(f"controlled_adaptive Joint requires positive train.learning_rates.{key}")
        if not bool(config.get("validation", {}).get("initialize_joint_best_from_parent", False)):
            raise ConfigError("controlled_adaptive Joint requires validation.initialize_joint_best_from_parent=true")
    elif joint_training_mode == "detector_head_adaptive":
        loss = config.get("loss", {})
        if flow != "warmup_joint":
            raise ConfigError("detector_head_adaptive Joint requires train.flow=warmup_joint")
        if float(loss.get("lambda_det", 0.0)) <= 0.0:
            raise ConfigError("detector_head_adaptive Joint requires loss.lambda_det>0")
        if float(loss.get("lambda_seg", 0.0)) != 0.0:
            raise ConfigError("detector_head_adaptive Joint requires loss.lambda_seg=0")
        if float(loss.get("lambda_ugca", 0.0)) <= 0.0:
            raise ConfigError("detector_head_adaptive Joint requires loss.lambda_ugca>0")
        if bool(train.get("train_student_decoder_in_joint", False)):
            raise ConfigError("detector_head_adaptive Joint requires train_student_decoder_in_joint=false")
        if str(train.get("joint_detector_scope", "all")) != "head":
            raise ConfigError("detector_head_adaptive Joint requires train.joint_detector_scope=head")
        joint_initialization = config.get("weights", {}).get("joint_initialization")
        if joint_initialization in (None, "", "REQUIRED"):
            raise ConfigError("detector_head_adaptive Joint requires weights.joint_initialization")
        if str(config.get("ugca", {}).get("variant", "v1")) != "quality_aware_v3":
            raise ConfigError("detector_head_adaptive Joint requires ugca.variant=quality_aware_v3")
        rates = train.get("learning_rates", {})
        for key in ("joint_detector_head", "joint_ugca_shared", "joint_ugca_quality"):
            if float(rates.get(key, 0.0)) <= 0.0:
                raise ConfigError(f"detector_head_adaptive Joint requires positive train.learning_rates.{key}")
        if not bool(config.get("validation", {}).get("initialize_joint_best_from_parent", False)):
            raise ConfigError(
                "detector_head_adaptive Joint requires validation.initialize_joint_best_from_parent=true"
            )
    elif joint_training_mode == "full":
        loss = config.get("loss", {})
        if flow != "warmup_joint":
            raise ConfigError("Full Joint training requires train.flow=warmup_joint")
        if any(float(loss.get(key, 0.0)) <= 0.0 for key in ("lambda_det", "lambda_seg", "lambda_ugca")):
            raise ConfigError("Full Joint training requires positive detection, segmentation, and UGCA losses")
        if not bool(train.get("train_student_decoder_in_joint", False)):
            raise ConfigError("Full Joint training requires train_student_decoder_in_joint=true")
        rates = train.get("learning_rates", {})
        for key in ("joint_detector", "joint_bridge", "joint_student_decoder", "joint_ugca"):
            if float(rates.get(key, 0.0)) <= 0.0:
                raise ConfigError(f"Full Joint training requires positive train.learning_rates.{key}")
        if architecture == "rfdetr_2xlarge":
            expected_weights = {"lambda_det": 0.25, "lambda_seg": 1.0, "lambda_ugca": 0.25}
            for key, expected in expected_weights.items():
                if abs(float(loss.get(key, 0.0)) - expected) > 1.0e-12:
                    raise ConfigError(f"Paper RF protocol requires loss.{key}={expected}")
    for stage_name in ("warmup", "joint"):
        world_size_key = f"{stage_name}_world_size_per_split"
        world_size = int(train.get(world_size_key, 1))
        if world_size < 1:
            raise ConfigError(f"train.{world_size_key} must be positive")
        if int(train.get("gradient_accumulation", 0)) % world_size:
            raise ConfigError(f"train.gradient_accumulation must be divisible by {world_size_key}")
    if str(train.get("optimizer", "")).lower() != "adamw":
        raise ConfigError("Formal DFC-SAM stages require train.optimizer=adamw")
    if str(train.get("scheduler", "cosine")).lower() != "cosine":
        raise ConfigError("Formal DFC-SAM stages require train.scheduler=cosine")
    if int(train.get("gradient_accumulation", 0)) < 1:
        raise ConfigError("train.gradient_accumulation must be positive")
    if any(int(train.get(key, 0)) < 1 for key in ("teacher_epochs", "warmup_epochs", "joint_epochs")):
        raise ConfigError("Teacher, warm-up, and joint epoch counts must be positive")
    validation = config.get("validation", {})
    if validation.get("checkpoint_metric") != "mpq" or validation.get("tie_break_metric") != "bpq":
        raise ConfigError("Checkpoint selection must use validation mPQ then bPQ")
    class_weights = config.get("loss", {}).get("ugca_class_weights")
    if class_weights is not None:
        if not isinstance(class_weights, list | tuple) or len(class_weights) != NUM_CLASSES:
            raise ConfigError(f"loss.ugca_class_weights must contain {NUM_CLASSES} values")
        if any(float(value) <= 0.0 for value in class_weights):
            raise ConfigError("loss.ugca_class_weights must be positive")
    label_smoothing = float(config.get("loss", {}).get("ugca_label_smoothing", 0.0))
    if not 0.0 <= label_smoothing < 1.0:
        raise ConfigError("loss.ugca_label_smoothing must be in [0,1)")
    boundary_weight = float(config.get("loss", {}).get("mask_boundary_loss_weight", 0.0))
    boundary_kernel = int(config.get("loss", {}).get("mask_boundary_kernel_size", 3))
    if not 0.0 <= boundary_weight <= 1.0:
        raise ConfigError("loss.mask_boundary_loss_weight must be in [0,1]")
    if boundary_kernel < 3 or boundary_kernel % 2 == 0:
        raise ConfigError("loss.mask_boundary_kernel_size must be an odd integer >= 3")
    ugca = config.get("ugca", {})
    variant = str(ugca.get("variant", "v1"))
    if variant not in {"v1", "conservative_v2", "quality_aware_v3"}:
        raise ConfigError("ugca.variant must be v1, conservative_v2, or quality_aware_v3")
    if variant in {"conservative_v2", "quality_aware_v3"}:
        if int(ugca.get("gate_hidden_dim", 0)) < 1:
            raise ConfigError("conservative_v2 requires a positive ugca.gate_hidden_dim")
        if float(ugca.get("residual_logit_cap", 0.0)) <= 0.0:
            raise ConfigError("conservative_v2 requires a positive ugca.residual_logit_cap")
        for key in ("ugca_gate_loss_weight", "ugca_preservation_loss_weight"):
            if float(config.get("loss", {}).get(key, -1.0)) < 0.0:
                raise ConfigError(f"conservative_v2 requires non-negative loss.{key}")
        if str(config.get("loss", {}).get("ugca_loss_normalization", "")) != "per_image":
            raise ConfigError(f"{variant} requires loss.ugca_loss_normalization=per_image")
    if variant == "quality_aware_v3":
        if int(ugca.get("quality_hidden_dim", 0)) < 1:
            raise ConfigError("quality_aware_v3 requires a positive ugca.quality_hidden_dim")
        if float(ugca.get("hard_negative_ratio", 0.0)) <= 0.0:
            raise ConfigError("quality_aware_v3 requires ugca.hard_negative_ratio > 0")
        if int(ugca.get("hard_negative_max_per_image", 0)) < 1:
            raise ConfigError("quality_aware_v3 requires a positive ugca.hard_negative_max_per_image")
        hard_negative_threshold = float(ugca.get("hard_negative_score_threshold", -1.0))
        if not 0.0 <= hard_negative_threshold <= 1.0:
            raise ConfigError("ugca.hard_negative_score_threshold must be in [0,1]")
        if float(config.get("loss", {}).get("ugca_quality_loss_weight", 0.0)) <= 0.0:
            raise ConfigError("quality_aware_v3 requires loss.ugca_quality_loss_weight > 0")
        quality_mask_threshold = float(config.get("loss", {}).get("ugca_quality_mask_threshold", -1.0))
        if not 0.0 <= quality_mask_threshold <= 1.0:
            raise ConfigError("loss.ugca_quality_mask_threshold must be in [0,1]")
    ranking_weight = float(config.get("loss", {}).get("ugca_quality_ranking_loss_weight", 0.0))
    if ranking_weight < 0.0:
        raise ConfigError("loss.ugca_quality_ranking_loss_weight must be non-negative")
    if ranking_weight > 0.0:
        if variant != "quality_aware_v3":
            raise ConfigError("quality ranking loss requires ugca.variant=quality_aware_v3")
        target_gap = float(config.get("loss", {}).get("ugca_quality_ranking_min_target_gap", -1.0))
        if not 0.0 < target_gap <= 1.0:
            raise ConfigError("loss.ugca_quality_ranking_min_target_gap must be in (0,1]")
        margin = float(config.get("loss", {}).get("ugca_quality_ranking_margin", -1.0))
        if margin < 0.0:
            raise ConfigError("loss.ugca_quality_ranking_margin must be non-negative")
    color_augmentation = config.get("augmentation", {}).get("train_color", {})
    if color_augmentation and bool(color_augmentation.get("enabled", False)):
        probability = float(color_augmentation.get("probability", -1.0))
        if not 0.0 <= probability <= 1.0:
            raise ConfigError("augmentation.train_color.probability must be in [0,1]")
        for key in ("stain_strength", "brightness_strength"):
            if not 0.0 <= float(color_augmentation.get(key, -1.0)) <= 0.5:
                raise ConfigError(f"augmentation.train_color.{key} must be in [0,0.5]")
    early_stopping = validation.get("early_stopping", {})
    for stage_name in ("warmup", "joint"):
        settings = early_stopping.get(stage_name, {})
        if settings and bool(settings.get("enabled", False)):
            if int(settings.get("min_epochs", 0)) < 1 or int(settings.get("patience", 0)) < 1:
                raise ConfigError(f"validation.early_stopping.{stage_name} needs positive min_epochs/patience")
            if float(settings.get("min_delta", 0.0)) < 0.0:
                raise ConfigError(f"validation.early_stopping.{stage_name}.min_delta must be non-negative")
    if experiment.get("status") == "formal_frozen":
        commit = config.get("evaluation", {}).get("official_metrics_commit")
        if not commit or commit in {"REQUIRED", "TBD"}:
            raise ConfigError("formal_frozen requires a pinned official_metrics_commit")
        repository = config.get("evaluation", {}).get("official_metrics_repo")
        if not repository or not Path(str(repository)).expanduser().is_absolute():
            raise ConfigError("formal_frozen requires an absolute official_metrics_repo")
        all_samples = config.get("all_samples_manifest")
        if not all_samples or not Path(str(all_samples)).expanduser().is_absolute():
            raise ConfigError("formal_frozen requires an absolute all_samples_manifest")


def dump_yaml(config: dict[str, Any], path: str | Path) -> None:
    """Write a resolved config in a stable, human-readable form."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
