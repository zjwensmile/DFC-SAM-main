"""Generate and audit the paired full/mixed PanNuke experiment matrix."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from dfc_sam.config import MIXED_MASK_RATIOS, MIXED_STRATEGIES, load_config, validate_experiment_config
from dfc_sam.utils.hashing import atomic_write_json, sha256_file, sha256_json


def _ratio_tag(ratio: float) -> str:
    return f"mask{round(ratio * 100):02d}"


def _detector_checkpoint(detector_root: Path, split_id: int) -> Path:
    return detector_root / f"cycle{split_id}" / "weights" / "best.pt"


def _resolved_config_sha256(path: str | Path) -> str:
    resolved = load_config(path)
    resolved.pop("_meta", None)
    return sha256_json(resolved)


def build_mixed_experiment_matrix(
    project_root: str | Path,
    detector_root: str | Path,
    destination: str | Path,
    *,
    selection_seed: int = 2026,
    model_seed: int = 42,
) -> Path:
    """Write 3 splits x 4 ratios x 3 paired strategy override configs."""
    project = Path(project_root).resolve()
    detectors = Path(detector_root).resolve()
    output = Path(destination).resolve()
    output.mkdir(parents=True, exist_ok=True)
    groups = []
    for split_id in (1, 2, 3):
        detector = _detector_checkpoint(detectors, split_id)
        if not detector.is_file():
            raise FileNotFoundError(detector)
        for ratio in MIXED_MASK_RATIOS:
            tag = _ratio_tag(ratio)
            group_id = f"split{split_id}_{tag}_seed{selection_seed}"
            manifest = project / "data" / "supervision" / f"split{split_id}" / f"{tag}_seed{selection_seed}.json"
            if not manifest.is_file():
                raise FileNotFoundError(manifest)
            shared_root = project / "outputs" / "mixed" / f"split{split_id}" / tag / f"seed{selection_seed}"
            teacher = shared_root / "teacher" / "best.pt"
            calibration = shared_root / "teacher" / "quality_calibration.json"
            pseudo_bank = shared_root / "pseudo_bank"
            threshold = shared_root / "shared_inference_threshold.json"
            config_paths = {}
            for strategy in MIXED_STRATEGIES:
                use_pseudo = strategy in {"naive_mixed", "qws_mixed"}
                use_qwpm = strategy == "qws_mixed"
                payload: dict[str, Any] = {
                    "base": "../mixed_template.yaml",
                    "experiment": {
                        "name": f"pannuke_{group_id}_{strategy}",
                        "protocol": "pannuke_from_generic",
                        "split_id": split_id,
                        "seed": model_seed,
                        "status": "strategy_locked_not_formal",
                    },
                    "weights": {
                        "detector_stage1": str(detector),
                        "teacher_stage2a": str(teacher),
                        "quality_calibration": str(calibration),
                        "warmup_stage2": str(shared_root / strategy / "warmup" / "best.pt"),
                        # Kept identical even for No-pseudo as immutable group lineage.
                        "pseudo_bank": str(pseudo_bank),
                    },
                    "supervision": {
                        "strategy": strategy,
                        "mask_ratio": ratio,
                        "manifest": str(manifest),
                        "use_teacher": True,
                        "use_pseudo_bank": use_pseudo,
                        "use_qwpm": use_qwpm,
                        "pseudo_weight_mode": {
                            "no_pseudo": "none",
                            "naive_mixed": "uniform",
                            "qws_mixed": "quality",
                        }[strategy],
                    },
                    "pairing": {
                        "group_id": group_id,
                        "selection_seed": selection_seed,
                        "model_seed": model_seed,
                        "batch_order_seed": model_seed,
                        "shared_teacher": str(teacher),
                        "shared_quality_calibration": str(calibration),
                        "shared_pseudo_bank": str(pseudo_bank),
                        "shared_inference_threshold": str(threshold),
                    },
                }
                path = output / f"{group_id}_{strategy}.yaml"
                path.write_text(
                    yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
                    encoding="utf-8",
                )
                config_paths[strategy] = str(path)
            report = validate_paired_group(list(config_paths.values()))
            groups.append(
                {
                    "group_id": group_id,
                    "split_id": split_id,
                    "mask_ratio": ratio,
                    "selection_seed": selection_seed,
                    "manifest": str(manifest),
                    "manifest_sha256": sha256_file(manifest),
                    "detector": str(detector),
                    "detector_sha256": sha256_file(detector),
                    "teacher_output": str(teacher),
                    "quality_calibration_output": str(calibration),
                    "pseudo_bank_output": str(pseudo_bank),
                    "configs": config_paths,
                    "config_file_sha256": {
                        strategy: sha256_file(path) for strategy, path in config_paths.items()
                    },
                    "resolved_config_sha256": {
                        strategy: _resolved_config_sha256(path) for strategy, path in config_paths.items()
                    },
                    "paired_audit": report,
                }
            )
    payload = {
        "schema_version": 1,
        "model_seed": model_seed,
        "selection_seed": selection_seed,
        "group_count": len(groups),
        "run_count": len(groups) * len(MIXED_STRATEGIES),
        "groups": groups,
    }
    payload["sha256"] = sha256_json(payload)
    index = output / "matrix_index.json"
    atomic_write_json(index, payload)
    return index


def validate_paired_group(config_paths: list[str | Path]) -> dict[str, Any]:
    """Ensure No-pseudo, Naive, and QWS differ only in registered strategy fields."""
    configs = [load_config(path) for path in config_paths]
    for config in configs:
        validate_experiment_config(config)
    by_strategy = {config["supervision"]["strategy"]: config for config in configs}
    if set(by_strategy) != set(MIXED_STRATEGIES):
        raise ValueError(f"Paired group must contain exactly {MIXED_STRATEGIES}")
    reference = by_strategy["no_pseudo"]
    shared_values = {
        "split_id": reference["experiment"]["split_id"],
        "model_seed": reference["experiment"]["seed"],
        "mask_ratio": float(reference["supervision"]["mask_ratio"]),
        "manifest": reference["supervision"]["manifest"],
        "detector_stage1": reference["weights"]["detector_stage1"],
        "teacher_stage2a": reference["weights"]["teacher_stage2a"],
        "quality_calibration": reference["weights"]["quality_calibration"],
        "pseudo_bank": reference["weights"]["pseudo_bank"],
        "train": reference["train"],
        "inference": reference["inference"],
        "pairing": reference["pairing"],
    }
    for strategy, config in by_strategy.items():
        candidate = {
            "split_id": config["experiment"]["split_id"],
            "model_seed": config["experiment"]["seed"],
            "mask_ratio": float(config["supervision"]["mask_ratio"]),
            "manifest": config["supervision"]["manifest"],
            "detector_stage1": config["weights"]["detector_stage1"],
            "teacher_stage2a": config["weights"]["teacher_stage2a"],
            "quality_calibration": config["weights"]["quality_calibration"],
            "pseudo_bank": config["weights"]["pseudo_bank"],
            "train": config["train"],
            "inference": config["inference"],
            "pairing": config["pairing"],
        }
        if candidate != shared_values:
            raise ValueError(f"{strategy} violates paired shared settings")

    manifest = json.loads(Path(shared_values["manifest"]).read_text(encoding="utf-8"))
    if int(manifest["split_id"]) != int(shared_values["split_id"]):
        raise ValueError("Manifest split_id disagrees with paired configs")
    if abs(float(manifest["mask_ratio"]) - float(shared_values["mask_ratio"])) > 1.0e-9:
        raise ValueError("Manifest mask_ratio disagrees with paired configs")
    expected_manifest_hash = sha256_json({key: value for key, value in manifest.items() if key != "sha256"})
    if manifest["sha256"] != expected_manifest_hash:
        raise ValueError("Manifest embedded checksum is invalid")
    return {
        "status": "passed",
        "group_id": reference["pairing"]["group_id"],
        "strategies": list(MIXED_STRATEGIES),
        "manifest_embedded_sha256": manifest["sha256"],
        "shared_detector": shared_values["detector_stage1"],
        "shared_teacher": shared_values["teacher_stage2a"],
        "shared_quality_calibration": shared_values["quality_calibration"],
        "shared_pseudo_bank": shared_values["pseudo_bank"],
    }


def validate_matrix_index(path: str | Path) -> dict[str, Any]:
    """Revalidate every group and the top-level immutable matrix checksum."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = sha256_json({key: value for key, value in payload.items() if key != "sha256"})
    if payload["sha256"] != expected:
        raise ValueError("Experiment matrix checksum mismatch")
    for group in payload["groups"]:
        for strategy, config_path in group["configs"].items():
            if sha256_file(config_path) != group["config_file_sha256"][strategy]:
                raise ValueError(f"Mixed config file checksum mismatch: {config_path}")
            if _resolved_config_sha256(config_path) != group["resolved_config_sha256"][strategy]:
                raise ValueError(f"Resolved mixed config checksum mismatch: {config_path}")
        validate_paired_group(list(group["configs"].values()))
    if int(payload["group_count"]) != 12 or int(payload["run_count"]) != 36:
        raise ValueError("The formal mixed matrix must contain 12 paired groups and 36 runs")
    return {
        "status": "passed",
        "group_count": int(payload["group_count"]),
        "run_count": int(payload["run_count"]),
        "sha256": payload["sha256"],
    }
