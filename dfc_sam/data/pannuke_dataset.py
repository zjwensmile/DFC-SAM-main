"""Manifest-only PanNuke dataset with a sealed box-only supervision path."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from dfc_sam.utils.hashing import sha256_json

from .pannuke_instances import extract_instances
from .transforms import PreparedImage, boxes_to_yolo_targets, prepare_yolo_sam_inputs

Role = Literal["train", "validation", "test"]
Supervision = Literal["full", "strong", "box_only"]


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def load_sample_records(path: str | Path) -> list[dict[str, Any]]:
    """Load a stable JSONL manifest and reject duplicate image identities."""
    records = []
    identities = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            image_id = str(record["image_id"])
            if image_id in identities:
                raise ValueError(f"Duplicate image_id in manifest: {image_id}")
            identities.add(image_id)
            records.append(record)
    return records


def _load_yolo_detection_label(path: str | Path, image_hw: tuple[int, int]) -> tuple[Tensor, Tensor]:
    """Read class/box annotations without touching instance-mask storage."""
    labels = []
    normalized_xywh = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 5:
                raise ValueError(f"Invalid YOLO label at {path}:{line_number}")
            class_id = int(fields[0])
            box = [float(value) for value in fields[1:]]
            if class_id not in range(5) or any(value < 0.0 or value > 1.0 for value in box):
                raise ValueError(f"Out-of-range YOLO label at {path}:{line_number}")
            labels.append(class_id)
            normalized_xywh.append(box)
    classes = torch.tensor(labels, dtype=torch.long)
    xywh = torch.tensor(normalized_xywh, dtype=torch.float32).reshape(-1, 4)
    height, width = image_hw
    scale = xywh.new_tensor([width, height, width, height])
    xywh = xywh * scale
    centers, sizes = xywh[:, :2], xywh[:, 2:]
    boxes = torch.cat((centers - sizes / 2, centers + sizes / 2), dim=-1)
    return boxes, classes


class PanNukeDataset(Dataset):
    """Read only the folds declared for one split role.

    A ``box_only`` sample returns boxes/classes and a false supervision vector,
    but never returns its ground-truth instance masks.
    """

    def __init__(
        self,
        all_samples_manifest: str | Path,
        split_manifest: str | Path,
        *,
        role: Role,
        supervision_by_image: Mapping[str, Supervision] | None = None,
        transform: Callable[[Tensor], PreparedImage] = prepare_yolo_sam_inputs,
        train_color_augmentation: Mapping[str, Any] | None = None,
        augmentation_seed: int = 0,
    ) -> None:
        if role not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split role: {role}")
        split = _read_json(split_manifest)
        fold_key = {"train": "train_folds", "validation": "val_folds", "test": "test_folds"}[role]
        allowed_folds = {int(value) for value in split[fold_key]}
        role_ids_key = {
            "train": "train_image_ids",
            "validation": "validation_image_ids",
            "test": "test_image_ids",
        }[role]
        allowed_ids = None
        if role_ids_key in split:
            role_ids = [str(value) for value in split[role_ids_key]]
            allowed_ids = set(role_ids)
            if len(allowed_ids) != len(role_ids):
                raise ValueError(f"{role_ids_key} contains duplicate image IDs")
        self.records = []
        for record in load_sample_records(all_samples_manifest):
            if int(record["fold"]) not in allowed_folds:
                continue
            if allowed_ids is not None and str(record["image_id"]) not in allowed_ids:
                continue
            self.records.append(record)
        expected_count = int(split["sample_counts"][role])
        if len(self.records) != expected_count:
            raise ValueError(f"{role} manifest count mismatch: {len(self.records)} != {expected_count}")
        self.role = role
        self.supervision_by_image = dict(supervision_by_image) if supervision_by_image is not None else None
        if self.supervision_by_image is not None:
            if role != "train":
                raise ValueError("A mixed-supervision manifest may only be attached to the training role")
            expected_ids = {str(record["image_id"]) for record in self.records}
            provided_ids = set(self.supervision_by_image)
            if provided_ids != expected_ids:
                missing = sorted(expected_ids - provided_ids)
                extra = sorted(provided_ids - expected_ids)
                raise ValueError(
                    f"Mixed supervision must cover the complete train fold; missing={missing[:3]}, extra={extra[:3]}"
                )
        self.transform = transform
        self.train_color_augmentation = dict(train_color_augmentation or {})
        self.augmentation_seed = int(augmentation_seed)
        self.augmentation_epoch = 0
        if self.train_color_augmentation and role != "train":
            raise ValueError("Color augmentation may only be enabled for the training role")
        self._array_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _array(self, path: str) -> np.ndarray:
        if path not in self._array_cache:
            self._array_cache[path] = np.load(path, mmap_mode="r")
        return self._array_cache[path]

    def _supervision(self, image_id: str) -> Supervision:
        default: Supervision = "full"
        supervision = self.supervision_by_image[image_id] if self.supervision_by_image is not None else default
        if supervision not in {"full", "strong", "box_only"}:
            raise ValueError(f"Invalid supervision for {image_id}: {supervision}")
        if self.role != "train" and supervision == "box_only":
            raise ValueError(f"{self.role} masks may not be hidden by a training supervision manifest")
        return supervision

    def set_epoch(self, epoch: int) -> None:
        self.augmentation_epoch = int(epoch)

    def _augment_color(self, image: Tensor, image_id: str) -> Tensor:
        settings = self.train_color_augmentation
        if not bool(settings.get("enabled", False)):
            return image
        generator = torch.Generator()
        digest = sha256_json([self.augmentation_seed, self.augmentation_epoch, image_id])
        generator.manual_seed(int(digest[:16], 16) % (2**63 - 1))
        probability = float(settings.get("probability", 0.8))
        if float(torch.rand((), generator=generator)) >= probability:
            return image
        stain_strength = float(settings.get("stain_strength", 0.15))
        brightness_strength = float(settings.get("brightness_strength", 0.10))
        stain_scale = 1.0 + (
            torch.rand((3, 1, 1), generator=generator) * 2.0 - 1.0
        ) * stain_strength
        optical_density = -torch.log((image.clamp(0, 255) + 1.0) / 256.0)
        augmented = 256.0 * torch.exp(-optical_density * stain_scale) - 1.0
        brightness = 1.0 + (
            float(torch.rand((), generator=generator)) * 2.0 - 1.0
        ) * brightness_strength
        return (augmented * brightness).clamp(0, 255)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        raw_index = int(record["raw_index"])
        image = np.asarray(self._array(record["images_npy"])[raw_index])
        if image.shape != (256, 256, 3) or not np.isfinite(image).all():
            raise ValueError(f"Invalid PanNuke image: {record['image_id']}")
        image_tensor = torch.from_numpy(np.array(image, copy=True, order="C")).permute(2, 0, 1).float()
        image_tensor = self._augment_color(image_tensor, str(record["image_id"]))
        prepared = self.transform(image_tensor)

        supervision = self._supervision(str(record["image_id"]))
        if supervision == "box_only":
            label_path = record.get("yolo_label")
            if not label_path:
                raise ValueError(f"Box-only sample has no sealed detection label: {record['image_id']}")
            boxes, labels = _load_yolo_detection_label(label_path, tuple(image.shape[:2]))
            instances = None
        else:
            raw_mask = np.asarray(self._array(record["masks_npy"])[raw_index])
            instances = extract_instances(raw_mask)
            boxes = torch.tensor([instance.box_xyxy for instance in instances], dtype=torch.float32).reshape(-1, 4)
            labels = torch.tensor([instance.class_id for instance in instances], dtype=torch.long)
        yolo_boxes = boxes_to_yolo_targets(boxes, prepared.geometry)
        is_mask_supervised = supervision != "box_only"
        target: dict[str, Any] = {
            "boxes_original_xyxy": boxes,
            "bboxes": yolo_boxes,
            "cls": labels,
            "mask_supervised": torch.full((len(labels),), is_mask_supervised, dtype=torch.bool),
        }
        if is_mask_supervised:
            assert instances is not None
            target["masks"] = torch.from_numpy(
                np.stack([instance.mask for instance in instances], axis=0)
                if instances
                else np.zeros((0, 256, 256), dtype=bool)
            )
        return {
            "image_id": str(record["image_id"]),
            "fold": int(record["fold"]),
            "raw_index": raw_index,
            "tissue": str(record["tissue"]),
            "supervision": supervision,
            "train_mask": target.get("masks"),
            "yolo_image": prepared.yolo_image,
            "sam_resized_image": prepared.sam_resized_image,
            "geometry": prepared.geometry,
            "target": target,
        }


def load_supervision_manifest(
    path: str | Path,
    *,
    expected_split_id: int | None = None,
    expected_mask_ratio: float | None = None,
) -> dict[str, Supervision]:
    """Load ``image_id -> strong|box_only`` assignments from an immutable JSON file."""
    payload = _read_json(path)
    expected_hash = sha256_json({key: value for key, value in payload.items() if key != "sha256"})
    if payload.get("sha256") != expected_hash:
        raise ValueError("Supervision manifest embedded checksum mismatch")
    if expected_split_id is not None and int(payload["split_id"]) != expected_split_id:
        raise ValueError("Supervision manifest split_id mismatch")
    if expected_mask_ratio is not None and abs(float(payload["mask_ratio"]) - expected_mask_ratio) > 1.0e-9:
        raise ValueError("Supervision manifest mask_ratio mismatch")
    samples: Sequence[dict[str, Any]] = payload["samples"]
    mapping: dict[str, Supervision] = {}
    for sample in samples:
        image_id = str(sample["image_id"])
        supervision = str(sample["supervision"])
        if supervision not in {"strong", "box_only"}:
            raise ValueError(f"Mixed manifest has invalid supervision: {supervision}")
        if image_id in mapping:
            raise ValueError(f"Mixed manifest repeats image_id: {image_id}")
        mapping[image_id] = supervision  # type: ignore[assignment]
    if set(payload["strong_image_ids"]) | set(payload["weak_image_ids"]) != set(mapping):
        raise ValueError("Supervision manifest ID lists do not cover samples")
    if set(payload["strong_image_ids"]) & set(payload["weak_image_ids"]):
        raise ValueError("Strong and weak image lists overlap")
    teacher_partition = payload.get("teacher_partition")
    if not isinstance(teacher_partition, dict):
        raise ValueError("Mixed manifest is missing the sealed Teacher/calibration partition")
    adaptation = set(teacher_partition.get("adaptation_image_ids", []))
    calibration = set(teacher_partition.get("quality_calibration_image_ids", []))
    strong = set(payload["strong_image_ids"])
    if adaptation & calibration or adaptation | calibration != strong:
        raise ValueError("Teacher adaptation and quality calibration must partition strong images")
    return mapping


def load_teacher_partition(path: str | Path) -> dict[str, set[str]]:
    """Load the sealed, disjoint Teacher-adaptation and calibration image IDs."""
    load_supervision_manifest(path)
    payload = _read_json(path)
    partition = payload["teacher_partition"]
    return {
        "adaptation": set(partition["adaptation_image_ids"]),
        "quality_calibration": set(partition["quality_calibration_image_ids"]),
    }
