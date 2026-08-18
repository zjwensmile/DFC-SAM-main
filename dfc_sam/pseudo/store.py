"""Read-only pseudo-mask access keyed by immutable target identity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .bank import validate_pseudo_bank


class PseudoMaskStore:
    """Load a validated fixed bank without exposing any ground-truth mask path."""

    def __init__(self, root: str | Path) -> None:
        metadata = validate_pseudo_bank(root)
        self.metadata = metadata
        self.records: dict[tuple[str, int], dict[str, Any]] = {}
        with Path(metadata["index_path"]).open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                identity = (str(record["image_id"]), int(record["instance_index"]))
                self.records[identity] = record

    def get(self, image_id: str, instance_index: int) -> tuple[Tensor, float]:
        identity = (str(image_id), int(instance_index))
        if identity not in self.records:
            raise KeyError(f"Pseudo-bank has no target identity: {identity}")
        record = self.records[identity]
        mask = np.load(record["mask_path"], allow_pickle=False)
        if mask.shape != (256, 256):
            raise ValueError(f"Pseudo-mask must be 256x256: {record['mask_path']}")
        return torch.from_numpy(np.asarray(mask, dtype=bool)), float(record["weight"])
