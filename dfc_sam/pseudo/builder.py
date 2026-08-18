"""Materialize one strategy-agnostic pseudo bank shared by Naive and QWS."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from torch import Tensor

from dfc_sam.models.teacher_adapter import TeacherOutput

from .bank import write_pseudo_bank_index
from .generation import SelectedPseudoMasks


class PseudoBankBuilder:
    """Accumulate selected Teacher masks while preserving target identity."""

    def __init__(self, destination: str | Path) -> None:
        self.destination = Path(destination).resolve()
        self.mask_root = self.destination / "masks"
        self.mask_root.mkdir(parents=True, exist_ok=True)
        self.records: list[dict[str, Any]] = []
        self.identities: set[tuple[str, int]] = set()

    def add_batch(
        self,
        selected: SelectedPseudoMasks,
        output: TeacherOutput,
        *,
        target_image_ids: list[str],
        target_instance_indices: Tensor,
        target_classes: Tensor,
        target_boxes_original_xyxy: Tensor,
    ) -> None:
        """Save a generated batch; inputs contain boxes/classes only, never hidden masks."""
        count = selected.masks.shape[0]
        if any(
            tensor.shape[0] != count
            for tensor in (
                selected.quality,
                selected.weights,
                selected.candidate_indices,
                output.target_flat_index,
            )
        ):
            raise ValueError("Selected pseudo-mask fields do not align")
        for output_index, flat_tensor in enumerate(output.target_flat_index.detach().cpu()):
            flat_index = int(flat_tensor)
            image_id = str(target_image_ids[flat_index])
            instance_index = int(target_instance_indices[flat_index])
            identity = (image_id, instance_index)
            if identity in self.identities:
                raise ValueError(f"Duplicate pseudo-mask identity: {identity}")
            self.identities.add(identity)
            filename = hashlib.sha256(f"{image_id}:{instance_index}".encode()).hexdigest() + ".npy"
            mask_path = self.mask_root / filename
            np.save(mask_path, selected.masks[output_index].detach().cpu().numpy().astype(np.uint8))
            self.records.append(
                {
                    "image_id": image_id,
                    "instance_index": instance_index,
                    "class_id": int(target_classes[flat_index]),
                    "box_xyxy": [
                        float(value) for value in target_boxes_original_xyxy[flat_index].detach().cpu().tolist()
                    ],
                    "mask_path": str(mask_path),
                    "quality": float(selected.quality[output_index].detach().cpu()),
                    "weight": float(selected.weights[output_index].detach().cpu()),
                    "candidate_index": int(selected.candidate_indices[output_index].detach().cpu()),
                }
            )

    def finalize(self, *, metadata: dict[str, Any]) -> dict[str, Any]:
        return write_pseudo_bank_index(self.records, self.destination, metadata=metadata)
