#!/usr/bin/env python
"""Prepare a test-sealed RF-DETR view of an official PanNuke split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT


CLASS_NAMES = ("Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial")
MANIFEST_ROOT = PROJECT_ROOT / "data/manifests/pannuke_standard_3fold/detector"


def _default_manifests(split_id: int) -> dict[str, Path]:
    return {
        "train": MANIFEST_ROOT / f"split{split_id}_train.txt",
        "valid": MANIFEST_ROOT / f"split{split_id}_validation.txt",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(path: Path) -> list[Path]:
    if not path.is_file():
        raise FileNotFoundError(path)
    images = [Path(line.strip()).expanduser().resolve() for line in path.read_text().splitlines() if line.strip()]
    if not images:
        raise ValueError(f"Manifest is empty: {path}")
    if len(images) != len(set(images)):
        raise ValueError(f"Manifest contains duplicate images: {path}")
    return images


def _label_path(image: Path) -> Path:
    parts = list(image.parts)
    try:
        image_index = parts.index("images")
    except ValueError as exc:
        raise ValueError(f"Image path has no 'images' component: {image}") from exc
    parts[image_index] = "labels"
    return Path(*parts).with_suffix(".txt")


def _validate_label(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(path)
    count = 0
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"Expected five YOLO fields at {path}:{line_number}")
        class_id = int(fields[0])
        coords = tuple(float(value) for value in fields[1:])
        if class_id not in range(len(CLASS_NAMES)):
            raise ValueError(f"Invalid class id {class_id} at {path}:{line_number}")
        if any(value < 0.0 or value > 1.0 for value in coords):
            raise ValueError(f"Non-normalized box at {path}:{line_number}")
        count += 1
    return count


def _safe_symlink(source: Path, destination: Path) -> None:
    if destination.is_symlink():
        if destination.resolve() != source:
            raise FileExistsError(f"Refusing to replace mismatched symlink: {destination}")
        return
    if destination.exists():
        raise FileExistsError(f"Refusing to replace existing path: {destination}")
    destination.symlink_to(source)


def _data_yaml(output: Path) -> str:
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    return (
        f"path: {output}\n"
        "train: train/images\n"
        "val: valid/images\n"
        f"nc: {len(CLASS_NAMES)}\n"
        "names:\n"
        f"{names}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-id", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    defaults = _default_manifests(args.split_id)
    manifests = {
        "train": (args.train_manifest or defaults["train"]).expanduser().resolve(),
        "valid": (args.validation_manifest or defaults["valid"]).expanduser().resolve(),
    }
    output = (args.output or PROJECT_ROOT / f"artifacts/rfdetr_split{args.split_id}/dataset").expanduser().resolve()
    split_images = {name: _read_manifest(path) for name, path in manifests.items()}
    overlap = set(split_images["train"]) & set(split_images["valid"])
    if overlap:
        raise ValueError(f"Train/validation overlap detected: {len(overlap)} images")

    summary: dict[str, object] = {
        "schema_version": 1,
        "protocol": "pannuke_standard_3fold",
        "split_id": args.split_id,
        "sealed_test_access": False,
        "format": "yolo",
        "output": str(output),
        "class_names": list(CLASS_NAMES),
        "manifests": {},
        "splits": {},
    }
    for split, images in split_images.items():
        object_count = 0
        for image in images:
            if not image.is_file():
                raise FileNotFoundError(image)
            object_count += _validate_label(_label_path(image))
        summary["manifests"][split] = {  # type: ignore[index]
            "path": str(manifests[split]),
            "sha256": _sha256(manifests[split]),
        }
        summary["splits"][split] = {"images": len(images), "objects": object_count}  # type: ignore[index]

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    for split, images in split_images.items():
        image_dir = output / split / "images"
        label_dir = output / split / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for image in images:
            label = _label_path(image)
            _safe_symlink(image, image_dir / image.name)
            _safe_symlink(label, label_dir / label.name)

    data_yaml = output / "data.yaml"
    expected_yaml = _data_yaml(output)
    if data_yaml.exists() and data_yaml.read_text() != expected_yaml:
        raise FileExistsError(f"Refusing to replace mismatched dataset config: {data_yaml}")
    data_yaml.write_text(expected_yaml)
    (output / "provenance.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"Prepared RF-DETR Split-{args.split_id} dataset view: {output}")


if __name__ == "__main__":
    main()
