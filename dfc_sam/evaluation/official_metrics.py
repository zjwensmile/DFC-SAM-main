"""Pinned adapter around the official PanNuke metric primitives."""

from __future__ import annotations

import importlib.util
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from dfc_sam.constants import PANNUKE_CLASSES, PANNUKE_TISSUES, normalize_tissue
from dfc_sam.evaluation.instance_metrics import InstanceMetricAccumulator


class OfficialMetricsError(RuntimeError):
    """Raised when the official repository is absent, modified, or incompatible."""


class OfficialMetricAccumulator:
    """Stream manifest-ordered image metrics without retaining full arrays."""

    def __init__(
        self,
        *,
        get_fast_pq: Callable[..., Any],
        binarize: Callable[[np.ndarray], np.ndarray],
        remap_label: Callable[[np.ndarray], np.ndarray],
        match_iou: float,
    ) -> None:
        self.get_fast_pq = get_fast_pq
        self.binarize = binarize
        self.remap_label = remap_label
        self.match_iou = float(match_iou)
        self.per_image: list[dict[str, Any]] = []
        self.instance_metrics = InstanceMetricAccumulator(
            get_fast_pq=get_fast_pq,
            match_iou=match_iou,
        )

    def update(
        self,
        ground_truth: np.ndarray,
        prediction: np.ndarray,
        *,
        tissue: str,
        image_id: str,
    ) -> None:
        expected_shape = (*ground_truth.shape[:2], len(PANNUKE_CLASSES))
        if ground_truth.shape != expected_shape or prediction.shape != expected_shape:
            raise ValueError("One image must use matching [H,W,5] official arrays")
        class_pq: list[float] = []
        for class_index in range(len(PANNUKE_CLASSES)):
            true_class = self.remap_label(
                ground_truth[..., class_index].astype(np.int32, copy=False)
            )
            pred_class = self.remap_label(
                prediction[..., class_index].astype(np.int32, copy=False)
            )
            if len(np.unique(true_class)) == 1:
                class_pq.append(float("nan"))
            else:
                statistics, _ = self.get_fast_pq(
                    true_class,
                    pred_class,
                    match_iou=self.match_iou,
                )
                class_pq.append(float(statistics[2]))
        true_binary = self.binarize(ground_truth)
        pred_binary = self.binarize(prediction)
        if len(np.unique(true_binary)) == 1:
            binary_pq = float("nan")
            predicted_ids = np.unique(pred_binary)
            binary_pairing = [
                [],
                [],
                [],
                predicted_ids[predicted_ids != 0].tolist(),
            ]
        else:
            binary_statistics, binary_pairing = self.get_fast_pq(
                true_binary,
                pred_binary,
                match_iou=self.match_iou,
            )
            binary_pq = float(binary_statistics[2])
        self.instance_metrics.update_from_pairing(
            ground_truth,
            prediction,
            pairing=binary_pairing,
        )
        self.per_image.append(
            {
                "image_id": str(image_id),
                "tissue": normalize_tissue(str(tissue)),
                "bpq": binary_pq,
                "mpq": _nanmean(class_pq),
                "class_pq": class_pq,
            }
        )

    def compute(self) -> dict[str, Any]:
        if not self.per_image:
            raise RuntimeError("Official metric accumulator has no images")
        class_values = np.asarray(
            [item["class_pq"] for item in self.per_image],
            dtype=np.float64,
        )
        tissue_metrics = {}
        for tissue in PANNUKE_TISSUES:
            items = [item for item in self.per_image if item["tissue"] == tissue]
            tissue_metrics[tissue] = {
                "images": len(items),
                "bpq": _nanmean([item["bpq"] for item in items]),
                "mpq": _nanmean([item["mpq"] for item in items]),
            }
        return {
            "sample_count": len(self.per_image),
            "bpq": _nanmean([value["bpq"] for value in tissue_metrics.values()]),
            "mpq": _nanmean([value["mpq"] for value in tissue_metrics.values()]),
            "class_pq": {
                name: _nanmean(class_values[:, index].tolist())
                for index, name in enumerate(PANNUKE_CLASSES)
            },
            "tissue": tissue_metrics,
            "per_image": self.per_image,
            "aggregation": "official macro-average over 19 tissues; empty-GT classes skipped",
            **self.instance_metrics.compute(),
        }


def _nanmean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(array)) if array.size and not np.isnan(array).all() else float("nan")


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("_dfc_sam_pinned_pannuke_metrics", path)
    if spec is None or spec.loader is None:
        raise OfficialMetricsError(f"Cannot import official metrics module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_official_primitives(
    repository: str | Path,
    *,
    expected_commit: str,
) -> tuple[
    Callable[..., Any],
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray], np.ndarray],
]:
    """Verify the clean repository commit, then expose official primitives."""
    if not expected_commit or expected_commit in {"REQUIRED", "TBD"}:
        raise OfficialMetricsError("A concrete official PanNuke metrics commit is required")
    root = Path(repository).expanduser().resolve()
    if not (root / ".git").is_dir():
        raise OfficialMetricsError(f"Official metrics Git repository is missing: {root}")
    actual = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != expected_commit:
        raise OfficialMetricsError(f"Official metrics commit mismatch: {actual} != {expected_commit}")
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise OfficialMetricsError("Official metrics repository has uncommitted changes")
    utils_path = root / "utils.py"
    if not utils_path.is_file():
        raise OfficialMetricsError(f"Official metrics utils.py is missing: {utils_path}")
    module = _load_module(utils_path)
    get_fast_pq = getattr(module, "get_fast_pq", None)
    binarize = getattr(module, "binarize", None)
    remap_label = getattr(module, "remap_label", None)
    if not callable(get_fast_pq) or not callable(binarize) or not callable(remap_label):
        raise OfficialMetricsError(
            "Official utils.py must expose get_fast_pq, binarize, and remap_label"
        )
    return get_fast_pq, binarize, remap_label


def evaluate_with_primitives(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    tissues: Sequence[str],
    *,
    get_fast_pq: Callable[..., Any],
    binarize: Callable[[np.ndarray], np.ndarray],
    remap_label: Callable[[np.ndarray], np.ndarray],
    match_iou: float = 0.5,
) -> dict[str, Any]:
    """Aggregate official primitives by image, class, and tissue."""
    if ground_truth.shape != prediction.shape or ground_truth.ndim != 4:
        raise ValueError("Ground truth and prediction must share [N,H,W,C] shape")
    if ground_truth.shape[-1] != len(PANNUKE_CLASSES) or len(tissues) != ground_truth.shape[0]:
        raise ValueError("PanNuke arrays or tissue list have invalid dimensions")
    accumulator = OfficialMetricAccumulator(
        get_fast_pq=get_fast_pq,
        binarize=binarize,
        remap_label=remap_label,
        match_iou=match_iou,
    )
    for image_index, tissue in enumerate(tissues):
        accumulator.update(
            ground_truth[image_index],
            prediction[image_index],
            tissue=str(tissue),
            image_id=str(image_index),
        )
    return accumulator.compute()


def evaluate_official_arrays(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    tissues: Sequence[str],
    *,
    repository: str | Path,
    expected_commit: str,
    match_iou: float = 0.5,
) -> dict[str, Any]:
    """Evaluate official PQ plus instance F1 after verifying the pinned commit."""
    get_fast_pq, binarize, remap_label = load_official_primitives(
        repository,
        expected_commit=expected_commit,
    )
    return evaluate_with_primitives(
        ground_truth,
        prediction,
        tissues,
        get_fast_pq=get_fast_pq,
        binarize=binarize,
        remap_label=remap_label,
        match_iou=match_iou,
    )
