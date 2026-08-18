"""Schema and integrity checks for immutable QWPM pseudo-mask banks."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from dfc_sam.utils.hashing import atomic_write_json, sha256_file, sha256_json

REQUIRED_RECORD_KEYS = {
    "image_id",
    "instance_index",
    "class_id",
    "box_xyxy",
    "mask_path",
    "quality",
    "weight",
}
REQUIRED_LINEAGE_KEYS = {
    "split_id",
    "mask_ratio",
    "selection_seed",
    "teacher_checkpoint_sha256",
    "supervision_manifest_sha256",
    "quality_calibration_sha256",
    "mask_threshold",
    "stability_delta",
    "quality_threshold",
}


def write_pseudo_bank_index(
    records: Iterable[dict[str, Any]],
    destination: str | Path,
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Write a stable bank index after validating every referenced mask artifact."""
    missing_lineage = REQUIRED_LINEAGE_KEYS - metadata.keys()
    if missing_lineage:
        raise ValueError(f"Pseudo-bank metadata is missing lineage keys: {sorted(missing_lineage)}")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    normalized = []
    identities = set()
    for raw in records:
        missing = REQUIRED_RECORD_KEYS - raw.keys()
        if missing:
            raise ValueError(f"Pseudo-bank record is missing keys: {sorted(missing)}")
        record = dict(raw)
        identity = (str(record["image_id"]), int(record["instance_index"]))
        if identity in identities:
            raise ValueError(f"Duplicate pseudo-mask identity: {identity}")
        identities.add(identity)
        class_id = int(record["class_id"])
        quality = float(record["quality"])
        weight = float(record["weight"])
        if class_id not in range(5):
            raise ValueError(f"Invalid PanNuke class id: {class_id}")
        if not (0.0 <= quality <= 1.0 and 0.0 <= weight <= 1.0):
            raise ValueError("Pseudo-mask quality and weight must be in [0,1]")
        mask_path = Path(str(record["mask_path"])).expanduser().resolve()
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        record.update(
            {
                "image_id": identity[0],
                "instance_index": identity[1],
                "class_id": class_id,
                "quality": quality,
                "weight": weight,
                "mask_path": str(mask_path),
                "mask_sha256": sha256_file(mask_path),
            }
        )
        normalized.append(record)

    normalized.sort(key=lambda item: (item["image_id"], item["instance_index"]))
    index_path = destination / "index.jsonl"
    with index_path.open("w", encoding="utf-8") as handle:
        for record in normalized:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    payload = {
        "schema_version": 1,
        "record_count": len(normalized),
        "index_path": str(index_path),
        "index_sha256": sha256_file(index_path),
        "metadata": metadata,
    }
    payload["bank_fingerprint"] = sha256_json(payload)
    atomic_write_json(destination / "bank_meta.json", payload)
    return payload


def validate_pseudo_bank(root: str | Path) -> dict[str, Any]:
    """Verify bank metadata, index checksum, uniqueness, and all mask checksums."""
    root = Path(root).resolve()
    with (root / "bank_meta.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    index_path = Path(metadata["index_path"])
    if sha256_file(index_path) != metadata["index_sha256"]:
        raise ValueError("Pseudo-bank index checksum mismatch")
    count = 0
    identities = set()
    with index_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            identity = (record["image_id"], int(record["instance_index"]))
            if identity in identities:
                raise ValueError(f"Duplicate pseudo-mask identity: {identity}")
            identities.add(identity)
            if sha256_file(record["mask_path"]) != record["mask_sha256"]:
                raise ValueError(f"Pseudo-mask checksum mismatch: {record['mask_path']}")
            count += 1
    if count != int(metadata["record_count"]):
        raise ValueError("Pseudo-bank record count mismatch")
    expected_fingerprint = sha256_json({key: value for key, value in metadata.items() if key != "bank_fingerprint"})
    if metadata["bank_fingerprint"] != expected_fingerprint:
        raise ValueError("Pseudo-bank metadata fingerprint mismatch")
    return metadata
