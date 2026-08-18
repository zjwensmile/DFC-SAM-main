"""Validate stage dependencies and produce dry-run records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dfc_sam.config import load_config, validate_experiment_config

from .stages import Stage


def _sam_requirement(config: dict[str, Any]) -> tuple[str, str]:
    variant = str(config.get("sam", {}).get("variant", "vit_b"))
    key = {"vit_b": "sam_vit_b", "vit_h": "sam_vit_h"}.get(variant)
    if key is None:
        raise ValueError(f"Unsupported SAM variant: {variant}")
    return key, f"weights.{key}"


def _requirements(config: dict[str, Any], stage: Stage) -> tuple[bool, tuple[tuple[str, str], ...]]:
    """Return protocol permission plus dotted config paths required by a stage."""
    supervision = config["supervision"]
    mode = supervision["mode"]
    strategy = supervision["strategy"]
    if stage is Stage.DETECTOR:
        return True, (("yolo_generic", "weights.yolo_generic"),)
    if stage is Stage.TEACHER:
        return mode == "mixed", (
            _sam_requirement(config),
            ("supervision_manifest", "supervision.manifest"),
        )
    if stage is Stage.WARMUP:
        requirements = [
            ("detector_stage1", "weights.detector_stage1"),
            _sam_requirement(config),
        ]
        if mode == "mixed":
            requirements.extend(
                (
                    ("teacher_stage2a", "weights.teacher_stage2a"),
                    ("supervision_manifest", "supervision.manifest"),
                )
            )
            if strategy in {"naive_mixed", "qws_mixed"}:
                requirements.append(("pseudo_bank", "weights.pseudo_bank"))
        return config["train"]["flow"] == "warmup_joint", tuple(requirements)
    if stage is Stage.JOINT:
        requirements = [
            ("detector_stage1", "weights.detector_stage1"),
            _sam_requirement(config),
        ]
        if str(config["train"].get("joint_training_mode", "full")) in {
            "ugca_ranking",
            "controlled",
            "controlled_adaptive",
            "detector_head_adaptive",
        }:
            requirements.append(("joint_initialization", "weights.joint_initialization"))
        elif config["train"]["flow"] == "warmup_joint":
            requirements.append(("warmup_stage2", "weights.warmup_stage2"))
        if mode == "mixed":
            requirements.extend(
                (
                    ("teacher_stage2a", "weights.teacher_stage2a"),
                    ("supervision_manifest", "supervision.manifest"),
                )
            )
            if strategy in {"naive_mixed", "qws_mixed"}:
                requirements.append(("pseudo_bank", "weights.pseudo_bank"))
        return True, tuple(requirements)
    raise ValueError(f"Unknown stage: {stage}")


def _get(config: dict[str, Any], dotted: str) -> Any:
    value: Any = config
    for key in dotted.split("."):
        value = value.get(key) if isinstance(value, dict) else None
    return value


def inspect_stage(config_path: str | Path, stage: Stage) -> dict[str, Any]:
    """Resolve config and report missing dependencies without mutating run state."""
    config = load_config(config_path)
    validate_experiment_config(config)
    protocol_allowed, requirements = _requirements(config, stage)
    dependencies = {}
    missing = []
    for name, dotted_key in requirements:
        raw_path = _get(config, dotted_key)
        path = Path(str(raw_path)).expanduser() if raw_path not in (None, "", "REQUIRED") else None
        exists = bool(path) and path.exists()
        dependencies[name] = {"config_key": dotted_key, "path": raw_path, "exists": exists}
        if not exists:
            missing.append(name)
    formal_frozen = config["experiment"].get("status") == "formal_frozen"
    if stage is Stage.WARMUP:
        world_size = int(config["train"].get("warmup_world_size_per_split", 1))
    elif stage is Stage.JOINT:
        world_size = int(config["train"].get("joint_world_size_per_split", 1))
    else:
        world_size = 1
    global_accumulation = int(config["train"]["gradient_accumulation"])
    return {
        "experiment": config["experiment"],
        "supervision": {
            key: config["supervision"][key]
            for key in ("mode", "strategy", "mask_ratio", "use_teacher", "use_pseudo_bank", "use_qwpm")
        },
        "stage": stage.value,
        "execution": {
            "world_size_per_split": world_size,
            "batch_size_per_gpu": int(config["train"]["batch_size_per_gpu"]),
            "global_gradient_accumulation": global_accumulation,
            "local_gradient_accumulation": global_accumulation // world_size,
            "effective_global_batch_size": int(config["train"]["batch_size_per_gpu"]) * global_accumulation,
        },
        "dependencies": dependencies,
        "protocol_allowed": protocol_allowed,
        "ready": protocol_allowed and not missing,
        "missing": missing,
        "protocol_blocker": None if protocol_allowed else f"{stage.value} is disabled for this protocol/flow",
        "formal_training_authorized": formal_frozen,
        "note": (
            "Formal runner is implemented; execution remains locked until experiment.status=formal_frozen."
            if not formal_frozen
            else "Configuration is marked formal_frozen; --execute still performs dependency and clean-Git checks."
        ),
    }
