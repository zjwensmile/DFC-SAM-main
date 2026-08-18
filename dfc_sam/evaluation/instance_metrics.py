"""Detection and phenotype F1 metrics on PanNuke instance maps."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from dfc_sam.constants import PANNUKE_CLASSES


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _typed_binary_map(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match official ``binarize`` semantics and retain each instance class.

    PanNuke channels are expected to be mutually exclusive.  Iterating classes
    in channel order deliberately mirrors the pinned official implementation;
    the final remap also makes the IDs safe for ``get_fast_pq``.
    """
    if array.ndim != 3 or array.shape[-1] != len(PANNUKE_CLASSES):
        raise ValueError("One PanNuke image must have shape [H,W,5]")
    if not np.issubdtype(array.dtype, np.integer) or array.min(initial=0) < 0:
        raise ValueError("PanNuke instance arrays must contain non-negative integers")

    binary = np.zeros(array.shape[:2], dtype=np.int32)
    class_by_old_id: list[int] = []
    next_id = 1
    for class_index in range(len(PANNUKE_CLASSES)):
        channel = array[..., class_index]
        for instance_id in np.unique(channel):
            if instance_id == 0:
                continue
            binary[channel == instance_id] = next_id
            class_by_old_id.append(class_index)
            next_id += 1

    # A malformed overlapping array can overwrite a complete earlier instance.
    # Drop such vanished IDs and return the contiguous ordering required by the
    # official fast matcher.
    present = np.unique(binary)
    present = present[present != 0]
    remapped = np.zeros_like(binary)
    classes = np.empty(len(present), dtype=np.int64)
    for new_id, old_id in enumerate(present.tolist(), start=1):
        remapped[binary == old_id] = new_id
        classes[new_id - 1] = class_by_old_id[old_id - 1]
    return remapped, classes


class InstanceMetricAccumulator:
    """Pool instance-level detection and five-class counts across images."""

    def __init__(self, *, get_fast_pq: Callable[..., Any], match_iou: float) -> None:
        self.get_fast_pq = get_fast_pq
        self.match_iou = float(match_iou)
        self.images = 0
        self.detection_tp = 0
        self.detection_fp = 0
        self.detection_fn = 0
        class_count = len(PANNUKE_CLASSES)
        self.class_tp = np.zeros(class_count, dtype=np.int64)
        self.class_fp = np.zeros(class_count, dtype=np.int64)
        self.class_fn = np.zeros(class_count, dtype=np.int64)
        self.class_ground_truth = np.zeros(class_count, dtype=np.int64)
        self.class_prediction = np.zeros(class_count, dtype=np.int64)

    def update(self, ground_truth: np.ndarray, prediction: np.ndarray) -> None:
        if ground_truth.shape != prediction.shape:
            raise ValueError("Ground-truth and prediction image shapes differ")
        true_map, true_classes = _typed_binary_map(ground_truth)
        pred_map, pred_classes = _typed_binary_map(prediction)
        if not true_classes.size and not pred_classes.size:
            pairing = [[], [], [], []]
        else:
            _, pairing = self.get_fast_pq(
                true_map,
                pred_map,
                match_iou=self.match_iou,
            )
        self.update_from_pairing(
            ground_truth,
            prediction,
            pairing=pairing,
            typed_maps=(true_map, true_classes, pred_map, pred_classes),
        )

    def update_from_pairing(
        self,
        ground_truth: np.ndarray,
        prediction: np.ndarray,
        *,
        pairing: Any,
        typed_maps: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> None:
        """Accumulate counts from a binary match already computed for PQ.

        This lets validation reuse the official binary-PQ pairing instead of
        running the comparatively expensive instance matcher a second time.
        """
        if ground_truth.shape != prediction.shape:
            raise ValueError("Ground-truth and prediction image shapes differ")
        if typed_maps is None:
            true_map, true_classes = _typed_binary_map(ground_truth)
            pred_map, pred_classes = _typed_binary_map(prediction)
        else:
            true_map, true_classes, pred_map, pred_classes = typed_maps
        self.images += 1
        self.class_ground_truth += np.bincount(
            true_classes, minlength=len(PANNUKE_CLASSES)
        )
        self.class_prediction += np.bincount(
            pred_classes, minlength=len(PANNUKE_CLASSES)
        )
        paired_true = np.asarray(pairing[0], dtype=np.int64)
        paired_pred = np.asarray(pairing[1], dtype=np.int64)
        unpaired_true = np.asarray(pairing[2], dtype=np.int64)
        unpaired_pred = np.asarray(pairing[3], dtype=np.int64)

        self.detection_tp += int(paired_true.size)
        self.detection_fn += int(unpaired_true.size)
        self.detection_fp += int(unpaired_pred.size)

        if unpaired_true.size:
            self.class_fn += np.bincount(
                true_classes[unpaired_true - 1], minlength=len(PANNUKE_CLASSES)
            )
        if unpaired_pred.size:
            self.class_fp += np.bincount(
                pred_classes[unpaired_pred - 1], minlength=len(PANNUKE_CLASSES)
            )
        for true_id, pred_id in zip(paired_true.tolist(), paired_pred.tolist(), strict=True):
            true_class = int(true_classes[true_id - 1])
            pred_class = int(pred_classes[pred_id - 1])
            if true_class == pred_class:
                self.class_tp[true_class] += 1
            else:
                self.class_fn[true_class] += 1
                self.class_fp[pred_class] += 1

    def state_dict(self) -> dict[str, Any]:
        """Return additive counts suitable for distributed object gathering."""
        return {
            "images": self.images,
            "detection_tp": self.detection_tp,
            "detection_fp": self.detection_fp,
            "detection_fn": self.detection_fn,
            "class_tp": self.class_tp.tolist(),
            "class_fp": self.class_fp.tolist(),
            "class_fn": self.class_fn.tolist(),
            "class_ground_truth": self.class_ground_truth.tolist(),
            "class_prediction": self.class_prediction.tolist(),
        }

    def merge_state_dict(self, state: dict[str, Any]) -> None:
        """Add one rank's instance counts into this accumulator."""
        self.images += int(state["images"])
        self.detection_tp += int(state["detection_tp"])
        self.detection_fp += int(state["detection_fp"])
        self.detection_fn += int(state["detection_fn"])
        self.class_tp += np.asarray(state["class_tp"], dtype=np.int64)
        self.class_fp += np.asarray(state["class_fp"], dtype=np.int64)
        self.class_fn += np.asarray(state["class_fn"], dtype=np.int64)
        self.class_ground_truth += np.asarray(state["class_ground_truth"], dtype=np.int64)
        self.class_prediction += np.asarray(state["class_prediction"], dtype=np.int64)

    def compute(self) -> dict[str, Any]:
        detection_precision = _safe_ratio(
            self.detection_tp, self.detection_tp + self.detection_fp
        )
        detection_recall = _safe_ratio(
            self.detection_tp, self.detection_tp + self.detection_fn
        )
        detection_f1 = _safe_ratio(
            2 * self.detection_tp,
            2 * self.detection_tp + self.detection_fp + self.detection_fn,
        )
        per_class: dict[str, dict[str, int | float]] = {}
        class_f1 = []
        for index, name in enumerate(PANNUKE_CLASSES):
            tp = int(self.class_tp[index])
            fp = int(self.class_fp[index])
            fn = int(self.class_fn[index])
            precision = _safe_ratio(tp, tp + fp)
            recall = _safe_ratio(tp, tp + fn)
            f1 = _safe_ratio(2 * tp, 2 * tp + fp + fn)
            class_f1.append(f1)
            per_class[name] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "ground_truth_instances": int(self.class_ground_truth[index]),
                "predicted_instances": int(self.class_prediction[index]),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        macro_f1 = float(np.mean(class_f1))
        return {
            "f1det": detection_f1,
            "macro_f1": macro_f1,
            "detection": {
                "tp": self.detection_tp,
                "fp": self.detection_fp,
                "fn": self.detection_fn,
                "precision": detection_precision,
                "recall": detection_recall,
                "f1": detection_f1,
            },
            "classification": {
                "macro_f1": macro_f1,
                "per_class": per_class,
            },
            "instance_metric_protocol": {
                "images": self.images,
                "matching": f"one-to-one binary instance IoU > {self.match_iou:g}",
                "aggregation": "pooled TP/FP/FN over the fold; five-class unweighted macro-F1",
                "wrong_class_pair": "FN for the true class and FP for the predicted class",
                "zero_division": 0.0,
            },
        }


def evaluate_instance_arrays(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    *,
    get_fast_pq: Callable[..., Any],
    match_iou: float = 0.5,
) -> dict[str, Any]:
    """Evaluate pooled F1det and Macro-F1 for manifest-ordered arrays."""
    if ground_truth.shape != prediction.shape or ground_truth.ndim != 4:
        raise ValueError("Ground truth and prediction must share [N,H,W,C] shape")
    if ground_truth.shape[-1] != len(PANNUKE_CLASSES):
        raise ValueError("PanNuke arrays must have five positive-class channels")
    accumulator = InstanceMetricAccumulator(
        get_fast_pq=get_fast_pq,
        match_iou=match_iou,
    )
    for image_index in range(ground_truth.shape[0]):
        accumulator.update(ground_truth[image_index], prediction[image_index])
    return accumulator.compute()
