"""Discover exactly one PanNuke images/masks/types triplet per official fold."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from dfc_sam.utils.hashing import atomic_write_json

FOLD_PATTERN = re.compile(r"(?:^|[^a-z0-9])fold[ _-]?([123])(?:[^0-9]|$)", re.IGNORECASE)


@dataclass(frozen=True)
class FoldFiles:
    fold: int
    images: str
    masks: str
    types: str


def infer_fold(path: Path) -> int:
    """Infer a fold only from explicit fold markers in the path."""
    matches = {int(match) for match in FOLD_PATTERN.findall(str(path))}
    if len(matches) != 1:
        raise ValueError(f"Cannot infer one unique fold from {path}: {sorted(matches)}")
    return matches.pop()


def discover_pannuke(raw_root: str | Path) -> list[FoldFiles]:
    """Recursively discover and validate the three official array triplets."""
    root = Path(raw_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"PanNuke raw root does not exist: {root}")

    candidates: dict[int, dict[str, list[Path]]] = {
        fold: {"images": [], "masks": [], "types": []} for fold in (1, 2, 3)
    }
    names = {"images.npy": "images", "masks.npy": "masks", "types.npy": "types"}
    for file_name, kind in names.items():
        for path in root.rglob(file_name):
            candidates[infer_fold(path)][kind].append(path.resolve())

    errors = []
    folds = []
    for fold in (1, 2, 3):
        fold_errors = []
        for kind, paths in candidates[fold].items():
            if len(paths) != 1:
                fold_errors.append(f"fold{fold} {kind}: expected 1 candidate, found {len(paths)}: {paths}")
        errors.extend(fold_errors)
        if not fold_errors:
            folds.append(
                FoldFiles(
                    fold=fold,
                    images=str(candidates[fold]["images"][0]),
                    masks=str(candidates[fold]["masks"][0]),
                    types=str(candidates[fold]["types"][0]),
                )
            )
    if errors:
        raise RuntimeError("PanNuke discovery is ambiguous or incomplete:\n" + "\n".join(errors))
    return folds


def write_discovery(raw_root: str | Path, destination: str | Path) -> dict:
    """Discover the dataset and persist absolute paths."""
    folds = discover_pannuke(raw_root)
    payload = {
        "schema_version": 1,
        "raw_root": str(Path(raw_root).expanduser().resolve()),
        "folds": [asdict(item) for item in folds],
    }
    atomic_write_json(destination, payload)
    return payload
