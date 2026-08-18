#!/usr/bin/env python
"""Verify downloaded release weights and prepared PanNuke fold manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT

EXPECTED = {1: (3, 2722), 2: (1, 2656), 3: (2, 2523)}
PRETRAINED = {
    "pretrained/rf-detr-xxlarge.pth": "bf418652d5e07ad441599acfadc65bb1767029079e7a343a6d61675ec5d553ae",
    "pretrained/sam_vit_h_4b8939.pth": "a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-root", type=Path, default=PROJECT_ROOT / "weights")
    parser.add_argument("--manifest-root", type=Path, default=PROJECT_ROOT / "data/manifests/pannuke_standard_3fold")
    parser.add_argument(
        "--require-pretrained",
        action="store_true",
        help="Also require the RF-DETR-2XL and SAM-H initialization weights used for new training",
    )
    args = parser.parse_args()
    checksums = {}
    checksum_path = args.weights_root / "SHA256SUMS"
    if checksum_path.is_file():
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                digest, relative = line.split(maxsplit=1)
                checksums[relative.lstrip("*")] = digest
    result = []
    for split_id, (test_fold, test_count) in EXPECTED.items():
        relative = f"split{split_id}/rfdetr_dfc_sam_2xl_split{split_id}.pt"
        weight = args.weights_root / relative
        manifest = args.manifest_root / f"split_{split_id}.json"
        if not weight.is_file() or not manifest.is_file():
            raise FileNotFoundError(f"Missing Split{split_id} weight or manifest")
        split = json.loads(manifest.read_text(encoding="utf-8"))
        if split.get("test_folds") != [test_fold] or int(split["sample_counts"]["test"]) != test_count:
            raise ValueError(f"Split{split_id} fold/count mismatch")
        digest = _sha256(weight)
        if relative in checksums and checksums[relative] != digest:
            raise ValueError(f"Split{split_id} SHA256 mismatch")
        result.append({"split_id": split_id, "test_fold": test_fold, "test_images": test_count, "sha256": digest})
    pretrained = []
    for relative, expected_digest in PRETRAINED.items():
        weight = args.weights_root / relative
        if not weight.is_file():
            if args.require_pretrained:
                raise FileNotFoundError(f"Missing training initialization checkpoint: {weight}")
            continue
        digest = _sha256(weight)
        if digest != expected_digest or (relative in checksums and checksums[relative] != digest):
            raise ValueError(f"Pretrained checkpoint SHA256 mismatch: {relative}")
        pretrained.append({"path": relative, "sha256": digest})
    print(
        json.dumps(
            {
                "status": "ok",
                "splits": result,
                "pretrained": pretrained,
                "total_test_images": 7901,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
