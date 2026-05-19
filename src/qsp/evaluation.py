"""Shared evaluation metrics."""

from __future__ import annotations

import numpy as np

from custom_qnn_financial_pipeline import directional_accuracy, regression_metrics, rmse


def binary_direction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return accuracy, precision, recall, and F1 for binary direction labels."""

    truth = np.asarray(y_true, dtype=int).ravel()
    pred = np.asarray(y_pred, dtype=int).ravel()
    if truth.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    tp = int(((truth == 1) & (pred == 1)).sum())
    tn = int(((truth == 0) & (pred == 0)).sum())
    fp = int(((truth == 0) & (pred == 1)).sum())
    fn = int(((truth == 1) & (pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": (tp + tn) / len(truth) if len(truth) else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


__all__ = [
    "binary_direction_metrics",
    "directional_accuracy",
    "regression_metrics",
    "rmse",
]
