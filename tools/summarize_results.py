#!/usr/bin/env python
"""Merge public test shards and write the requested three-fold paper tables."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from _bootstrap import PROJECT_ROOT  # noqa: F401

from dfc_sam.constants import PANNUKE_CLASSES, PANNUKE_TISSUES
from dfc_sam.utils.hashing import atomic_write_json

METRICS = ("bpq", "mpq", "f1det", "macro_f1")
CLASS_ORDER = ("neoplastic", "epithelial", "inflammatory", "connective", "dead")
EXPECTED = {1: (3, 2722), 2: (1, 2656), 3: (2, 2523)}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _merge_split(paths: list[Path], split_id: int) -> dict[str, Any]:
    shards = [_read(path) for path in paths]
    shard_count = len(shards)
    if sorted(int(item["shard_index"]) for item in shards) != list(range(shard_count)):
        raise ValueError(f"Split{split_id} shard coverage is incomplete")
    test_fold, sample_count = EXPECTED[split_id]
    indices = sorted(int(index) for item in shards for index in item["dataset_indices"])
    if indices != list(range(sample_count)):
        raise ValueError(f"Split{split_id} test indices are incomplete")
    hashes = {str(item["checkpoint_sha256"]) for item in shards}
    if len(hashes) != 1 or any(int(item["test_fold"]) != test_fold for item in shards):
        raise ValueError(f"Split{split_id} provenance mismatch")
    tissue = {}
    for name in PANNUKE_TISSUES:
        sources = [item["aggregation_state"]["tissue"].get(name, {}) for item in shards]
        row = {"images": sum(int(source.get("images", 0)) for source in sources)}
        for metric in ("bpq", "mpq"):
            total = sum(float(source.get(metric, {}).get("sum", 0.0)) for source in sources)
            count = sum(int(source.get(metric, {}).get("count", 0)) for source in sources)
            row[metric] = total / count if count else float("nan")
        tissue[name] = row
    detection = {key: sum(int(item["metrics"]["detection"][key]) for item in shards) for key in ("tp", "fp", "fn")}
    detection.update(
        {
            "precision": _ratio(detection["tp"], detection["tp"] + detection["fp"]),
            "recall": _ratio(detection["tp"], detection["tp"] + detection["fn"]),
            "f1": _ratio(2 * detection["tp"], 2 * detection["tp"] + detection["fp"] + detection["fn"]),
        }
    )
    per_class = {}
    for name in PANNUKE_CLASSES:
        sources = [item["metrics"]["classification"]["per_class"][name] for item in shards]
        counts = {key: sum(int(source[key]) for source in sources) for key in ("tp", "fp", "fn")}
        counts["f1"] = _ratio(2 * counts["tp"], 2 * counts["tp"] + counts["fp"] + counts["fn"])
        per_class[name] = counts
    class_pq = {}
    for index, name in enumerate(PANNUKE_CLASSES):
        sources = [item["aggregation_state"]["class_pq"][index] for item in shards]
        count = sum(int(source["count"]) for source in sources)
        class_pq[name] = sum(float(source["sum"]) for source in sources) / count if count else float("nan")
    return {
        "split_id": split_id,
        "test_fold": test_fold,
        "sample_count": sample_count,
        "checkpoint_sha256": hashes.pop(),
        "bpq": float(np.nanmean([row["bpq"] for row in tissue.values()])),
        "mpq": float(np.nanmean([row["mpq"] for row in tissue.values()])),
        "f1det": detection["f1"],
        "macro_f1": float(np.mean([per_class[name]["f1"] for name in PANNUKE_CLASSES])),
        "class_pq": class_pq,
        "tissue": tissue,
        "detection": detection,
        "classification": {"per_class": per_class},
    }


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.input_root.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    splits = []
    for split_id in (1, 2, 3):
        paths = sorted(
            path
            for path in (source / f"split{split_id}" / "shards").glob("shard_*.json")
            if path.stem.removeprefix("shard_").isdigit()
        )
        if not paths:
            raise FileNotFoundError(f"No Split{split_id} shards under {source}")
        splits.append(_merge_split(paths, split_id))
    three_fold = {
        key: {
            "mean": statistics.mean(float(item[key]) for item in splits),
            "std_population": statistics.pstdev(float(item[key]) for item in splits),
        }
        for key in METRICS
    }
    tissue_rows = []
    for name in PANNUKE_TISSUES:
        bpq = [100.0 * item["tissue"][name]["bpq"] for item in splits]
        mpq = [100.0 * item["tissue"][name]["mpq"] for item in splits]
        tissue_rows.append(
            {
                "tissue": name,
                **{f"split{index}_bpq": bpq[index - 1] for index in (1, 2, 3)},
                **{f"split{index}_mpq": mpq[index - 1] for index in (1, 2, 3)},
                "mean_bpq": statistics.mean(bpq),
                "std_bpq": statistics.pstdev(bpq),
                "mean_mpq": statistics.mean(mpq),
                "std_mpq": statistics.pstdev(mpq),
            }
        )
    class_rows = []
    for name in CLASS_ORDER:
        values = [100.0 * item["class_pq"][name] for item in splits]
        class_rows.append(
            {
                "class": name,
                "split1_pq": values[0],
                "split2_pq": values[1],
                "split3_pq": values[2],
                "mean_pq": statistics.mean(values),
                "std_pq": statistics.pstdev(values),
            }
        )
    fold_mpq_cls = [statistics.mean(100.0 * item["class_pq"][name] for name in CLASS_ORDER) for item in splits]
    payload = {
        "status": "completed",
        "metric_scale": "fractions unless a key ends with _percent",
        "splits": splits,
        "three_fold": three_fold,
        "paper_row_percent": {key: 100.0 * three_fold[key]["mean"] for key in METRICS},
        "paper_std_population_percent": {key: 100.0 * three_fold[key]["std_population"] for key in METRICS},
        "tissue_percent": tissue_rows,
        "class_pq_percent": class_rows,
        "mpq_cls_percent": {
            "split1": fold_mpq_cls[0],
            "split2": fold_mpq_cls[1],
            "split3": fold_mpq_cls[2],
            "mean": statistics.mean(fold_mpq_cls),
            "std_population": statistics.pstdev(fold_mpq_cls),
        },
    }
    atomic_write_json(output / "summary.json", payload)
    _write_csv(output / "per_tissue.csv", list(tissue_rows[0]), tissue_rows)
    _write_csv(output / "per_class_pq.csv", list(class_rows[0]), class_rows)
    main_rows = []
    for item in splits:
        main_rows.append({"result": f"Split{item['split_id']}", **{key: 100.0 * item[key] for key in METRICS}})
    main_rows.append({"result": "Mean", **payload["paper_row_percent"]})
    main_rows.append({"result": "Std(population)", **payload["paper_std_population_percent"]})
    _write_csv(output / "threefold_metrics.csv", ["result", *METRICS], main_rows)
    lines = [
        "# DFC-SAM(RF)-H PanNuke three-fold test results",
        "",
        "| Result | bPQ (%) | mPQ (%) | F1det (%) | Macro-F1 (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in main_rows:
        lines.append(
            f"| {row['result']} | {row['bpq']:.3f} | {row['mpq']:.3f} | {row['f1det']:.3f} | {row['macro_f1']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Five-class PQ (%)",
            "",
            "| Class | Split1 | Split2 | Split3 | Mean ± population SD |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in class_rows:
        lines.append(
            f"| {row['class']} | {row['split1_pq']:.3f} | {row['split2_pq']:.3f} | "
            f"{row['split3_pq']:.3f} | {row['mean_pq']:.3f} ± {row['std_pq']:.3f} |"
        )
    mpq_cls = payload["mpq_cls_percent"]
    lines.append(
        f"| **mPQ_cls** | {mpq_cls['split1']:.3f} | {mpq_cls['split2']:.3f} | "
        f"{mpq_cls['split3']:.3f} | **{mpq_cls['mean']:.3f} ± {mpq_cls['std_population']:.3f}** |"
    )
    lines.extend(
        [
            "",
            "## PanNuke 19-tissue bPQ/mPQ (%)",
            "",
            "| Tissue | Split1 bPQ/mPQ | Split2 bPQ/mPQ | Split3 bPQ/mPQ | Mean bPQ/mPQ |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in tissue_rows:
        lines.append(
            f"| {row['tissue']} | {row['split1_bpq']:.3f}/{row['split1_mpq']:.3f} | "
            f"{row['split2_bpq']:.3f}/{row['split2_mpq']:.3f} | "
            f"{row['split3_bpq']:.3f}/{row['split3_mpq']:.3f} | "
            f"**{row['mean_bpq']:.3f}/{row['mean_mpq']:.3f}** |"
        )
    lines.append("")
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": str(output / "summary.json"),
                "split_results_percent": {
                    f"split{item['split_id']}": {
                        key: 100.0 * float(item[key]) for key in METRICS
                    }
                    for item in splits
                },
                "paper_row_percent": payload["paper_row_percent"],
                "mpq_cls_percent": payload["mpq_cls_percent"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
