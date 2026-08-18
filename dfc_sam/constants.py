"""Protocol constants shared by data, model, export, and evaluation code."""

from typing import Final

PANNUKE_CLASSES: Final = (
    "neoplastic",
    "inflammatory",
    "connective",
    "dead",
    "epithelial",
)
PANNUKE_BACKGROUND_CHANNEL: Final = 5
NUM_CLASSES: Final = len(PANNUKE_CLASSES)

PANNUKE_TISSUES: Final = (
    "Adrenal Gland",
    "Bile Duct",
    "Bladder",
    "Breast",
    "Cervix",
    "Colon",
    "Esophagus",
    "Head & Neck",
    "Kidney",
    "Liver",
    "Lung",
    "Ovarian",
    "Pancreatic",
    "Prostate",
    "Skin",
    "Stomach",
    "Testis",
    "Thyroid",
    "Uterus",
)

OFFICIAL_PANNUKE_TISSUES: Final = (
    "Adrenal_gland",
    "Bile-duct",
    "Bladder",
    "Breast",
    "Cervix",
    "Colon",
    "Esophagus",
    "HeadNeck",
    "Kidney",
    "Liver",
    "Lung",
    "Ovarian",
    "Pancreatic",
    "Prostate",
    "Skin",
    "Stomach",
    "Testis",
    "Thyroid",
    "Uterus",
)

TISSUE_ALIASES: Final = {
    "Adrenal_gland": "Adrenal Gland",
    "Bile-duct": "Bile Duct",
    "HeadNeck": "Head & Neck",
}
OFFICIAL_TISSUE_NAMES: Final = {
    normalized: official
    for official, normalized in zip(OFFICIAL_PANNUKE_TISSUES, PANNUKE_TISSUES, strict=True)
}

PANNUKE_FOLD_ROTATIONS: Final = {
    1: {"train": (1,), "validation": (2,), "test": (3,)},
    2: {"train": (2,), "validation": (3,), "test": (1,)},
    3: {"train": (3,), "validation": (1,), "test": (2,)},
}

EXPECTED_FOLD_COUNTS: Final = {1: 2656, 2: 2523, 3: 2722}
EXPECTED_TOTAL_COUNTS: Final = {7901, 7904}


def normalize_tissue(value: str) -> str:
    """Normalize an official PanNuke tissue string without inventing new labels."""
    normalized = TISSUE_ALIASES.get(value, value)
    if normalized not in PANNUKE_TISSUES:
        raise ValueError(f"Unexpected PanNuke tissue type: {value!r}")
    return normalized


def official_tissue_name(value: str) -> str:
    """Return the exact spelling expected by the pinned official metrics script."""
    return OFFICIAL_TISSUE_NAMES[normalize_tissue(value)]
