"""Write official PanNuke arrays with strict shape/order assertions."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from dfc_sam.constants import NUM_CLASSES, official_tissue_name
from dfc_sam.utils.hashing import atomic_write_json


def export_official_arrays(
    masks: np.ndarray,
    tissues: Sequence[str],
    image_ids: Sequence[str],
    destination: str | Path,
    *,
    metadata: dict | None = None,
) -> None:
    """Export masks.npy, types.npy, image_ids.npy, and export_meta.json."""
    if masks.ndim != 4 or masks.shape[1:] != (256, 256, NUM_CLASSES):
        raise ValueError(f"Expected masks [N,256,256,{NUM_CLASSES}], got {masks.shape}")
    if masks.shape[0] != len(tissues) or masks.shape[0] != len(image_ids):
        raise ValueError("masks, tissues, and image_ids must have the same length")
    if not np.issubdtype(masks.dtype, np.integer) or masks.min(initial=0) < 0:
        raise ValueError("Official masks must use a non-negative integer dtype")
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("image_ids must be unique and in manifest order")

    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "masks.npy", masks)
    np.save(root / "types.npy", np.asarray([official_tissue_name(value) for value in tissues]))
    np.save(root / "image_ids.npy", np.asarray(image_ids))
    atomic_write_json(
        root / "export_meta.json",
        {
            "schema_version": 1,
            "sample_count": masks.shape[0],
            "mask_shape": list(masks.shape),
            "channel_order": ["neoplastic", "inflammatory", "connective", "dead", "epithelial"],
            **(metadata or {}),
        },
    )
