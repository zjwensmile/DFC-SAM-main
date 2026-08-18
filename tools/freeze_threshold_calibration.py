#!/usr/bin/env python
"""Bind a validation-selected threshold calibration to a new test decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT  # noqa: F401

from dfc_sam.config import dump_yaml, load_config, validate_experiment_config
from dfc_sam.engine.runtime import write_frozen_test_decision
from dfc_sam.utils.hashing import atomic_write_json, sha256_file, sha256_json


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--output-decision", required=True)
    args = parser.parse_args()

    base_config = load_config(args.base_config)
    validate_experiment_config(base_config)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    calibration_path = Path(args.calibration).expanduser().resolve()
    calibration = _read_json(calibration_path)
    if calibration.get("role") != "validation" or not calibration.get("execute"):
        raise SystemExit("Calibration artifact must be an executed validation-only sweep")
    if int(calibration.get("split_id", -1)) != int(base_config["experiment"]["split_id"]):
        raise SystemExit("Calibration split does not match the base config")
    if calibration.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise SystemExit("Calibration is bound to a different checkpoint")
    if calibration.get("config_sha256") != sha256_json(base_config):
        raise SystemExit("Calibration is bound to a different base config")

    selected = calibration["selected"]
    thresholds = selected["thresholds"]
    calibrated_config = dict(base_config)
    calibrated_config["inference"] = {
        **base_config["inference"],
        "pre_threshold": float(thresholds["pre_threshold"]),
        "final_score_threshold": float(thresholds["final_score_threshold"]),
        "mask_threshold": float(thresholds["mask_threshold"]),
    }
    output_config = Path(args.output_config).expanduser().resolve()
    output_decision = Path(args.output_decision).expanduser().resolve()
    if output_config.exists() or output_decision.exists():
        raise SystemExit("Refusing to overwrite a calibrated config or frozen decision")
    dump_yaml(calibrated_config, output_config)
    reloaded_config = load_config(output_config)
    validate_experiment_config(reloaded_config)
    decision = write_frozen_test_decision(
        output_decision,
        checkpoint=checkpoint,
        validation_metrics=selected["metrics"],
        inference=reloaded_config["inference"],
        config=reloaded_config,
    )
    decision["threshold_calibration"] = {
        "artifact": str(calibration_path),
        "artifact_sha256": sha256_file(calibration_path),
        "candidate_id": selected["candidate_id"],
        "selection_rule": calibration["selection_rule"],
        "role": "validation",
    }
    atomic_write_json(output_decision, decision)
    print(
        json.dumps(
            {
                "status": "frozen_for_test",
                "config": str(output_config),
                "decision": str(output_decision),
                "thresholds": thresholds,
                "validation_metrics": {
                    key: selected["metrics"][key]
                    for key in ("bpq", "mpq", "f1det", "macro_f1")
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
