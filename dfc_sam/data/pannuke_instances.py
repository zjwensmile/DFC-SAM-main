"""Convert one PanNuke six-channel mask into explicit class-local instances."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dfc_sam.constants import NUM_CLASSES


@dataclass(frozen=True)
class PanNukeInstance:
    class_id: int
    raw_instance_id: int
    mask: np.ndarray
    box_xyxy: tuple[int, int, int, int]
    global_instance_id: int


def extract_instances(mask: np.ndarray) -> list[PanNukeInstance]:
    """Extract all positive instances, retaining tiny and border-truncated objects."""
    if mask.shape != (256, 256, 6):
        raise ValueError(f"Expected one mask with shape (256, 256, 6), got {mask.shape}")
    if not np.isfinite(mask).all() or (mask < 0).any() or not np.equal(mask, np.floor(mask)).all():
        raise ValueError("PanNuke mask must contain finite non-negative integer-valued IDs")

    instances = []
    global_id = 1
    for class_id in range(NUM_CLASSES):
        channel = mask[..., class_id]
        for raw_id_value in np.unique(channel):
            raw_id = int(raw_id_value)
            if raw_id == 0:
                continue
            binary = channel == raw_id
            ys, xs = np.nonzero(binary)
            if ys.size == 0:
                raise AssertionError("A unique positive instance ID produced an empty mask")
            box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            instances.append(PanNukeInstance(class_id, raw_id, binary, box, global_id))
            global_id += 1
    return instances
