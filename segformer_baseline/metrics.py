"""Binary metrics with explicit undefined-denominator behavior."""

from __future__ import annotations

import numpy as np


METRIC_POLICY = (
    "Dataset metrics use the sum of pixel confusion matrices (rows=GT, columns=prediction). "
    "A ratio with zero denominator is null, not 0 or 1. mIoU averages defined class IoUs. "
    "Per-image metrics are diagnostic and are not averaged to produce dataset metrics."
)


def confusion_matrix(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    indices = target.reshape(-1).astype(np.int64) * 2 + prediction.reshape(-1).astype(np.int64)
    return np.bincount(indices, minlength=4).reshape(2, 2).astype(np.int64)


def ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def metrics_from_confusion(confusion: np.ndarray) -> dict:
    tn, fp, fn, tp = (int(value) for value in confusion.reshape(-1))
    foreground_iou = ratio(tp, tp + fp + fn)
    background_iou = ratio(tn, tn + fp + fn)
    ious = [value for value in (background_iou, foreground_iou) if value is not None]
    dice = ratio(2 * tp, 2 * tp + fp + fn)
    return {
        "foreground_iou": foreground_iou, "background_iou": background_iou,
        "miou": float(np.mean(ious)) if ious else None,
        "dice": dice, "foreground_f1": dice,
        "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
        "pixel_accuracy": ratio(tp + tn, tp + tn + fp + fn),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }
