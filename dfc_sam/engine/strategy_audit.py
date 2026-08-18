"""Static acceptance audit for the adjusted PanNuke experiment strategy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dfc_sam.config import load_config, validate_experiment_config
from dfc_sam.data.supervision_manifests import verify_nested_supervision_manifests

from .experiment_matrix import validate_matrix_index


def audit_adjusted_strategy(project_root: str | Path) -> dict[str, Any]:
    """Fail closed if full and mixed protocols are crossed or matrix artifacts drift."""
    root = Path(project_root).resolve()
    full_configs = [root / "configs" / "experiments" / f"full_split{split_id}.yaml" for split_id in (1, 2, 3)]
    for path in full_configs:
        config = load_config(path)
        validate_experiment_config(config)
        if config["supervision"]["use_teacher"] or config["supervision"]["use_pseudo_bank"]:
            raise ValueError(f"Full experiment crossed into the mixed protocol: {path}")

    direct = load_config(root / "configs" / "experiments" / "full_split1_direct_joint.yaml")
    validate_experiment_config(direct)
    if direct["experiment"]["split_id"] != 1 or direct["train"]["flow"] != "direct_joint":
        raise ValueError("Direct-joint ablation must be limited to Split 1")

    manifest_reports = []
    calibration_counts: dict[str, dict[str, int]] = {}
    for split_id in (1, 2, 3):
        directory = root / "data" / "supervision" / f"split{split_id}"
        paths = sorted(directory.glob("mask*_seed2026.json"))
        report = verify_nested_supervision_manifests(paths)
        manifest_reports.append(report)
        calibration_counts[f"split{split_id}"] = {}
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            tag = f"mask{round(float(payload['mask_ratio']) * 100):02d}"
            calibration_counts[f"split{split_id}"][tag] = len(
                payload["teacher_partition"]["quality_calibration_image_ids"]
            )

    matrix = validate_matrix_index(root / "configs" / "experiments" / "mixed" / "matrix_index.json")
    return {
        "status": "passed",
        "full_main_runs": 3,
        "direct_joint_ablations": 1,
        "mixed_paired_runs": matrix["run_count"],
        "matrix_sha256": matrix["sha256"],
        "supervision_manifests": manifest_reports,
        "quality_calibration_image_counts": calibration_counts,
    }
