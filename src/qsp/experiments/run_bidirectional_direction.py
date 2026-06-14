"""Run the bidirectional up/down direction-prediction track.

This experiment deliberately reuses the interim ANN, QQBN, and QQTN models
instead of introducing a new BiLSTM-first architecture. "Bidirectional" here
means predicting the next-day price direction: up for a positive next return,
down for a non-positive next return.
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from qsp.data import add_technical_indicators, download_ohlcv_with_fallback
from qsp.evaluation import binary_direction_metrics
from qsp.models.quantum_inspired import (
    ANNClassifier,
    QQBNClassifier,
    QQTNClassifier,
    count_trainable_parameters,
    default_device,
    predict_binary_classifier,
    train_binary_classifier,
)


BENCHMARK_SYMBOLS = ("AAPL", "MSFT", "GOOGL", "NVDA")

DIRECTION_FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "RSI",
    "MACD",
    "MACD_Signal",
    "MACD_Hist",
    "SMA5",
    "SMA10",
    "SMA20",
    "ADX",
    "Return_1",
    "Log_Return_1",
    "Return_2",
    "Return_5",
    "Momentum_3",
    "Momentum_5",
    "Volatility_5",
    "Volatility_10",
    "Volatility_20",
    "Close_SMA5_Ratio",
    "Close_SMA10_Ratio",
    "Close_SMA20_Ratio",
    "High_Low_Range",
    "Open_Close_Range",
    "Volume_Change",
    "Volume_ZScore_20",
]


@dataclass(frozen=True)
class DirectionDataset:
    """Time-ordered dataset for next-day up/down prediction."""

    symbol: str
    train_x: np.ndarray
    val_x: np.ndarray
    test_x: np.ndarray
    train_y: np.ndarray
    val_y: np.ndarray
    test_y: np.ndarray
    train_next_returns: np.ndarray
    val_next_returns: np.ndarray
    test_next_returns: np.ndarray
    train_momentum_pred: np.ndarray
    val_momentum_pred: np.ndarray
    test_momentum_pred: np.ndarray
    train_dates: pd.Series
    val_dates: pd.Series
    test_dates: pd.Series
    train_prev_close: np.ndarray
    val_prev_close: np.ndarray
    test_prev_close: np.ndarray
    train_actual_close: np.ndarray
    val_actual_close: np.ndarray
    test_actual_close: np.ndarray
    feature_columns: list[str]
    source: str
    note: str


@dataclass(frozen=True)
class ThresholdCalibration:
    """Validation-selected probability threshold and validation metrics."""

    threshold: float
    metrics: dict[str, float]
    objective: str


def add_bidirectional_direction_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add lagged, present-day-only features and the next-day direction target."""

    data = add_technical_indicators(frame).copy()
    close = data["Close"].astype(float)
    volume = data["Volume"].astype(float)

    data["Target_Date"] = pd.Series(data.index, index=data.index).shift(-1)
    data["Actual_Next_Close"] = close.shift(-1)
    data["Next_Return"] = close.pct_change().shift(-1)
    data["Target"] = (data["Next_Return"] > 0.0).astype(int)
    data["Prev_Close"] = close

    data["Log_Return_1"] = np.log(close / close.shift(1))
    data["Return_2"] = close.pct_change(2)
    data["Return_5"] = close.pct_change(5)
    data["Momentum_3"] = close / close.shift(3) - 1.0
    data["Momentum_5"] = close / close.shift(5) - 1.0
    data["Volatility_5"] = data["Return_1"].rolling(window=5, min_periods=5).std()
    data["Volatility_10"] = data["Return_1"].rolling(window=10, min_periods=10).std()
    data["Volatility_20"] = data["Return_1"].rolling(window=20, min_periods=20).std()
    data["SMA10"] = close.rolling(window=10, min_periods=10).mean()
    data["SMA20"] = close.rolling(window=20, min_periods=20).mean()
    data["Close_SMA5_Ratio"] = close / data["SMA5"] - 1.0
    data["Close_SMA10_Ratio"] = close / data["SMA10"] - 1.0
    data["Close_SMA20_Ratio"] = close / data["SMA20"] - 1.0
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_Hist"] = data["MACD"] - data["MACD_Signal"]
    data["High_Low_Range"] = (data["High"] - data["Low"]) / close.replace(0.0, np.nan)
    data["Open_Close_Range"] = (data["Open"] - close) / close.replace(0.0, np.nan)
    data["Volume_Change"] = volume.pct_change()
    volume_mean = volume.rolling(window=20, min_periods=20).mean()
    volume_std = volume.rolling(window=20, min_periods=20).std()
    data["Volume_ZScore_20"] = (volume - volume_mean) / volume_std.replace(0.0, np.nan)

    data = data.replace([np.inf, -np.inf], np.nan)
    required = DIRECTION_FEATURE_COLUMNS + [
        "Next_Return",
        "Target_Date",
        "Actual_Next_Close",
        "Prev_Close",
        "Target",
    ]
    return data.dropna(subset=required).copy()


def prepare_bidirectional_direction_dataset(
    symbol: str,
    start: str = "2018-01-01",
    max_rows: int = 1200,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    frame: pd.DataFrame | None = None,
    source: str | None = None,
    note: str = "",
) -> DirectionDataset:
    """Prepare a train/validation/test split with train-only scaling."""

    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must leave a non-empty test split.")

    if frame is None:
        downloaded, downloaded_source, downloaded_note = download_ohlcv_with_fallback(
            symbol=symbol,
            start=start,
        )
        frame = downloaded
        source = downloaded_source
        note = downloaded_note
    elif source is None:
        source = "provided frame"

    data = add_bidirectional_direction_features(frame).tail(max_rows).copy()
    train_end = int(len(data) * train_ratio)
    val_end = int(len(data) * (train_ratio + val_ratio))
    if train_end < 20 or val_end - train_end < 5 or len(data) - val_end < 5:
        raise ValueError("Not enough rows for train/validation/test direction splits.")

    raw_features = data[DIRECTION_FEATURE_COLUMNS].to_numpy(dtype=float)
    targets = data["Target"].to_numpy(dtype=int)
    next_returns = data["Next_Return"].to_numpy(dtype=float)
    momentum_pred = (data["Return_1"].to_numpy(dtype=float) > 0.0).astype(int)
    prev_close = data["Prev_Close"].to_numpy(dtype=float)
    actual_close = data["Actual_Next_Close"].to_numpy(dtype=float)
    target_dates = pd.to_datetime(data["Target_Date"]).reset_index(drop=True)

    scaler = MinMaxScaler()
    scaler.fit(raw_features[:train_end])
    scaled = scaler.transform(raw_features)

    return DirectionDataset(
        symbol=symbol,
        train_x=scaled[:train_end].astype(np.float32),
        val_x=scaled[train_end:val_end].astype(np.float32),
        test_x=scaled[val_end:].astype(np.float32),
        train_y=targets[:train_end],
        val_y=targets[train_end:val_end],
        test_y=targets[val_end:],
        train_next_returns=next_returns[:train_end],
        val_next_returns=next_returns[train_end:val_end],
        test_next_returns=next_returns[val_end:],
        train_momentum_pred=momentum_pred[:train_end],
        val_momentum_pred=momentum_pred[train_end:val_end],
        test_momentum_pred=momentum_pred[val_end:],
        train_dates=target_dates.iloc[:train_end].reset_index(drop=True),
        val_dates=target_dates.iloc[train_end:val_end].reset_index(drop=True),
        test_dates=target_dates.iloc[val_end:].reset_index(drop=True),
        train_prev_close=prev_close[:train_end],
        val_prev_close=prev_close[train_end:val_end],
        test_prev_close=prev_close[val_end:],
        train_actual_close=actual_close[:train_end],
        val_actual_close=actual_close[train_end:val_end],
        test_actual_close=actual_close[val_end:],
        feature_columns=list(DIRECTION_FEATURE_COLUMNS),
        source=str(source),
        note=note,
    )


def _direction_confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    truth = np.asarray(y_true, dtype=int).ravel()
    pred = np.asarray(y_pred, dtype=int).ravel()
    if truth.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    return {
        "tp": int(((truth == 1) & (pred == 1)).sum()),
        "tn": int(((truth == 0) & (pred == 0)).sum()),
        "fp": int(((truth == 0) & (pred == 1)).sum()),
        "fn": int(((truth == 1) & (pred == 0)).sum()),
    }


def _direction_metric_pack(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return direction metrics used in saved result tables."""

    truth = np.asarray(y_true, dtype=int).ravel()
    pred = np.asarray(y_pred, dtype=int).ravel()
    metrics = binary_direction_metrics(truth, pred)
    counts = _direction_confusion_counts(truth, pred)
    positive_count = counts["tp"] + counts["fn"]
    negative_count = counts["tn"] + counts["fp"]
    sensitivity = counts["tp"] / positive_count if positive_count else 0.0
    specificity = counts["tn"] / negative_count if negative_count else 0.0
    metrics["sensitivity"] = sensitivity
    metrics["specificity"] = specificity
    metrics["balanced_accuracy"] = 0.5 * (sensitivity + specificity)
    metrics["predicted_up_rate"] = float(pred.mean()) if len(pred) else 0.0
    metrics["true_up_rate"] = float(truth.mean()) if len(truth) else 0.0
    return metrics


def calibrate_probability_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray | None = None,
    objective: str = "f1",
) -> ThresholdCalibration:
    """Choose a validation threshold from validation probabilities."""

    truth = np.asarray(y_true, dtype=int).ravel()
    proba = np.asarray(probabilities, dtype=float).ravel()
    if truth.shape != proba.shape:
        raise ValueError("y_true and probabilities must have the same shape.")
    if thresholds is None:
        thresholds = np.linspace(0.30, 0.70, 81)
    if objective not in {"f1", "accuracy", "balanced_accuracy"}:
        raise ValueError("objective must be 'f1', 'accuracy', or 'balanced_accuracy'.")

    candidates = []
    has_both_truth_classes = len(np.unique(truth)) == 2
    for threshold in thresholds:
        pred = (proba >= float(threshold)).astype(int)
        pred_has_both_classes = len(np.unique(pred)) == 2
        metrics = _direction_metric_pack(truth, pred)
        candidates.append((float(threshold), metrics, pred_has_both_classes))

    eligible = candidates
    if has_both_truth_classes and any(candidate[2] for candidate in candidates):
        eligible = [candidate for candidate in candidates if candidate[2]]

    def sort_key(item: tuple[float, dict[str, float], bool]) -> tuple[float, ...]:
        threshold, metrics, pred_has_both_classes = item
        if objective == "accuracy":
            return (metrics["accuracy"], metrics["f1"], metrics["balanced_accuracy"], -abs(threshold - 0.5))
        if objective == "balanced_accuracy":
            return (
                metrics["balanced_accuracy"],
                metrics["accuracy"],
                metrics["f1"],
                1.0 if pred_has_both_classes else 0.0,
                -abs(threshold - 0.5),
            )
        return (metrics["f1"], metrics["accuracy"], metrics["balanced_accuracy"], -abs(threshold - 0.5))

    best_threshold, best_metrics, _ = max(
        eligible,
        key=sort_key,
    )
    return ThresholdCalibration(threshold=best_threshold, metrics=best_metrics, objective=objective)


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _information_coefficient(probabilities: np.ndarray, next_returns: np.ndarray) -> float:
    if len(probabilities) < 2 or np.std(probabilities) == 0.0 or np.std(next_returns) == 0.0:
        return 0.0
    return float(np.corrcoef(probabilities, next_returns)[0, 1])


def _strategy_sharpe(predictions: np.ndarray, next_returns: np.ndarray) -> float:
    positions = np.where(np.asarray(predictions, dtype=int) == 1, 1.0, -1.0)
    strategy_returns = positions * np.asarray(next_returns, dtype=float)
    std = float(np.std(strategy_returns))
    if std == 0.0:
        return 0.0
    return float(np.sqrt(252.0) * np.mean(strategy_returns) / std)


def _baseline_result(
    dataset: DirectionDataset,
    model_name: str,
    predictions: np.ndarray,
    majority_accuracy: float,
    momentum_accuracy: float,
) -> dict[str, object]:
    metrics = _direction_metric_pack(dataset.test_y, predictions)
    return {
        "Model name": model_name,
        "Dataset / asset": dataset.symbol,
        "Model family": "Baseline",
        "Feature set": "Target distribution" if model_name == "Majority baseline" else "Previous-day return sign",
        "Directional Accuracy": metrics["accuracy"],
        "Accuracy": metrics["accuracy"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "F1": metrics["f1"],
        "Specificity": metrics["specificity"],
        "Balanced Accuracy": metrics["balanced_accuracy"],
        "Predicted up rate": metrics["predicted_up_rate"],
        "True up rate": metrics["true_up_rate"],
        "Majority baseline accuracy": majority_accuracy,
        "Momentum baseline accuracy": momentum_accuracy,
        "Validation threshold": "N/A",
        "Threshold objective": "N/A",
        "Validation F1": "N/A",
        "Validation accuracy": "N/A",
        "Validation balanced accuracy": "N/A",
        "Sharpe ratio": _strategy_sharpe(predictions, dataset.test_next_returns),
        "Information coefficient": "N/A",
        "Training time": 0.0,
        "Inference time": 0.0,
        "Parameter count": 0,
        "Data source": dataset.source,
        "Notes": "Time-ordered test baseline for the next-day up/down prediction track.",
    }


def _save_probability_plot(
    output_dir: Path,
    symbol: str,
    model_name: str,
    dates: pd.Series,
    truth: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> None:
    path = output_dir / f"{symbol}_{_safe_name(model_name)}_probability.png"
    plt.figure(figsize=(9, 4))
    plt.plot(dates, probabilities, label="Probability up", linewidth=1.8)
    plt.scatter(dates, truth, label="Actual direction", s=16, alpha=0.7)
    plt.axhline(threshold, color="black", linestyle="--", linewidth=1.1, label="Validation threshold")
    plt.ylim(-0.08, 1.08)
    plt.xlabel("Target date")
    plt.ylabel("Direction / probability")
    plt.title(f"{symbol} {model_name} direction probabilities")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _save_confusion_matrix_plot(
    output_dir: Path,
    symbol: str,
    model_name: str,
    truth: np.ndarray,
    predictions: np.ndarray,
) -> None:
    path = output_dir / f"{symbol}_{_safe_name(model_name)}_confusion_matrix.png"
    truth = np.asarray(truth, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    matrix = np.asarray(
        [
            [int(((truth == 0) & (predictions == 0)).sum()), int(((truth == 0) & (predictions == 1)).sum())],
            [int(((truth == 1) & (predictions == 0)).sum()), int(((truth == 1) & (predictions == 1)).sum())],
        ],
        dtype=int,
    )
    plt.figure(figsize=(4.6, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.title(f"{symbol} {model_name} confusion matrix")
    plt.xticks([0, 1], ["Pred down", "Pred up"])
    plt.yticks([0, 1], ["Actual down", "Actual up"])
    for row in range(2):
        for col in range(2):
            plt.text(col, row, str(matrix[row, col]), ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _train_and_evaluate_model(
    dataset: DirectionDataset,
    model_name: str,
    factory,
    output_dir: Path,
    epochs: int,
    hidden_dim: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
    majority_accuracy: float,
    momentum_accuracy: float,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    torch.manual_seed(seed)
    model = factory(input_dim=dataset.train_x.shape[1], hidden_dim=hidden_dim)

    train_started = time.perf_counter()
    history = train_binary_classifier(
        model,
        dataset.train_x,
        dataset.train_y,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
        batch_size=64,
        val_ratio=0.15,
        patience=14,
        weight_decay=1e-4,
        gradient_clip=1.0,
        device=device,
        balance_classes=True,
    )
    training_time = time.perf_counter() - train_started

    infer_started = time.perf_counter()
    _, val_probabilities = predict_binary_classifier(model, dataset.val_x, device=device)
    _, test_probabilities = predict_binary_classifier(model, dataset.test_x, device=device)
    inference_time = time.perf_counter() - infer_started

    base_notes = (
        "Bidirectional direction track: next-day up/down target, train-only feature scaling, "
        "time-ordered train/validation/test split, validation-calibrated probability threshold. "
        f"Feature count={len(dataset.feature_columns)}, hidden_dim={hidden_dim}, device={history.device}."
    )
    if model_name == "QQTN":
        base_notes = (
            "Main bidirectional model. Reuses the qutrit-inspired three-state feature map from the interim QQTN track. "
            + base_notes
        )
    elif model_name == "QQBN":
        base_notes = "Qubit-inspired comparison model from the interim QQBN track. " + base_notes
    else:
        base_notes = "Classical ANN comparison from the interim qubit/qutrit paper track. " + base_notes
    if dataset.note:
        base_notes += f" {dataset.note}"

    variants = [(model_name, "f1", base_notes)]
    if model_name == "QQTN":
        variants.append(
            (
                "QQTN balanced threshold",
                "balanced_accuracy",
                "Same trained QQTN probabilities as the main row, but the validation threshold optimizes balanced accuracy "
                "to reduce one-sided up/down predictions. "
                + base_notes,
            )
        )

    result_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    prediction_payload: dict[str, np.ndarray] = {}
    for display_name, objective, notes in variants:
        calibration = calibrate_probability_threshold(
            dataset.val_y,
            val_probabilities,
            objective=objective,
        )
        test_predictions = (test_probabilities >= calibration.threshold).astype(int)
        metrics = _direction_metric_pack(dataset.test_y, test_predictions)
        _save_probability_plot(
            output_dir,
            dataset.symbol,
            display_name,
            dataset.test_dates,
            dataset.test_y,
            test_probabilities,
            calibration.threshold,
        )
        _save_confusion_matrix_plot(output_dir, dataset.symbol, display_name, dataset.test_y, test_predictions)
        result_rows.append(
            {
                "Model name": display_name,
                "Dataset / asset": dataset.symbol,
                "Model family": model.__class__.__name__,
                "Feature set": ", ".join(dataset.feature_columns),
                "Directional Accuracy": metrics["accuracy"],
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1": metrics["f1"],
                "Specificity": metrics["specificity"],
                "Balanced Accuracy": metrics["balanced_accuracy"],
                "Predicted up rate": metrics["predicted_up_rate"],
                "True up rate": metrics["true_up_rate"],
                "Majority baseline accuracy": majority_accuracy,
                "Momentum baseline accuracy": momentum_accuracy,
                "Validation threshold": calibration.threshold,
                "Threshold objective": calibration.objective,
                "Validation F1": calibration.metrics["f1"],
                "Validation accuracy": calibration.metrics["accuracy"],
                "Validation balanced accuracy": calibration.metrics["balanced_accuracy"],
                "Sharpe ratio": _strategy_sharpe(test_predictions, dataset.test_next_returns),
                "Information coefficient": _information_coefficient(test_probabilities, dataset.test_next_returns),
                "Training time": training_time,
                "Inference time": inference_time,
                "Parameter count": count_trainable_parameters(model),
                "Data source": dataset.source,
                "Notes": notes,
            }
        )
        threshold_rows.append(
            {
                "symbol": dataset.symbol,
                "model": display_name,
                "threshold": calibration.threshold,
                "threshold_objective": calibration.objective,
                "validation_accuracy": calibration.metrics["accuracy"],
                "validation_precision": calibration.metrics["precision"],
                "validation_recall": calibration.metrics["recall"],
                "validation_f1": calibration.metrics["f1"],
                "validation_specificity": calibration.metrics["specificity"],
                "validation_balanced_accuracy": calibration.metrics["balanced_accuracy"],
            }
        )
        prediction_payload[f"{display_name} Probability up"] = test_probabilities
        prediction_payload[f"{display_name} Predicted direction"] = test_predictions

    logs = [
        {
            "symbol": dataset.symbol,
            "model": model_name,
            "epoch": index + 1,
            "train_loss": history.train_losses[index] if index < len(history.train_losses) else np.nan,
            "val_loss": loss,
            "device": history.device,
        }
        for index, loss in enumerate(history.losses)
    ]
    probability_payload = {
        "val_probabilities": val_probabilities,
        "test_probabilities": test_probabilities,
    }
    return result_rows, logs, threshold_rows, prediction_payload, probability_payload


def _ensemble_result(
    dataset: DirectionDataset,
    output_dir: Path,
    val_probabilities: np.ndarray,
    test_probabilities: np.ndarray,
    majority_accuracy: float,
    momentum_accuracy: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, np.ndarray]]:
    """Evaluate a probability ensemble over the reused interim model family."""

    variants = [
        (
            "ANN+QQBN+QQTN ensemble",
            "accuracy",
            "Validation-calibrated probability average over the reused interim ANN, QQBN, and QQTN models. "
            "This is an ensemble view, not a new paper architecture.",
        ),
        (
            "ANN+QQBN+QQTN balanced ensemble",
            "balanced_accuracy",
            "Same probability average as the ensemble row, but with the validation threshold optimized for balanced accuracy. "
            "This row is included to inspect the up/down class balance directly.",
        ),
    ]
    result_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    prediction_payload: dict[str, np.ndarray] = {}
    for display_name, objective, notes in variants:
        calibration = calibrate_probability_threshold(
            dataset.val_y,
            val_probabilities,
            objective=objective,
        )
        predictions = (test_probabilities >= calibration.threshold).astype(int)
        metrics = _direction_metric_pack(dataset.test_y, predictions)
        _save_probability_plot(
            output_dir,
            dataset.symbol,
            display_name,
            dataset.test_dates,
            dataset.test_y,
            test_probabilities,
            calibration.threshold,
        )
        _save_confusion_matrix_plot(output_dir, dataset.symbol, display_name, dataset.test_y, predictions)
        result_rows.append(
            {
                "Model name": display_name,
                "Dataset / asset": dataset.symbol,
                "Model family": "ANN/QQBN/QQTN calibrated ensemble",
                "Feature set": ", ".join(dataset.feature_columns),
                "Directional Accuracy": metrics["accuracy"],
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1": metrics["f1"],
                "Specificity": metrics["specificity"],
                "Balanced Accuracy": metrics["balanced_accuracy"],
                "Predicted up rate": metrics["predicted_up_rate"],
                "True up rate": metrics["true_up_rate"],
                "Majority baseline accuracy": majority_accuracy,
                "Momentum baseline accuracy": momentum_accuracy,
                "Validation threshold": calibration.threshold,
                "Threshold objective": calibration.objective,
                "Validation F1": calibration.metrics["f1"],
                "Validation accuracy": calibration.metrics["accuracy"],
                "Validation balanced accuracy": calibration.metrics["balanced_accuracy"],
                "Sharpe ratio": _strategy_sharpe(predictions, dataset.test_next_returns),
                "Information coefficient": _information_coefficient(test_probabilities, dataset.test_next_returns),
                "Training time": 0.0,
                "Inference time": 0.0,
                "Parameter count": 0,
                "Data source": dataset.source,
                "Notes": notes,
            }
        )
        threshold_rows.append(
            {
                "symbol": dataset.symbol,
                "model": display_name,
                "threshold": calibration.threshold,
                "threshold_objective": calibration.objective,
                "validation_accuracy": calibration.metrics["accuracy"],
                "validation_precision": calibration.metrics["precision"],
                "validation_recall": calibration.metrics["recall"],
                "validation_f1": calibration.metrics["f1"],
                "validation_specificity": calibration.metrics["specificity"],
                "validation_balanced_accuracy": calibration.metrics["balanced_accuracy"],
            }
        )
        prediction_payload[f"{display_name} Probability up"] = test_probabilities
        prediction_payload[f"{display_name} Predicted direction"] = predictions
    return result_rows, threshold_rows, prediction_payload


def _aggregate_result_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Add one average row per model for presentation."""

    frame = pd.DataFrame(rows)
    assets = sorted(set(frame["Dataset / asset"].astype(str)))
    asset_label = f"Average ({len(assets)} assets)"
    asset_list = ", ".join(assets)
    aggregate_rows: list[dict[str, object]] = []
    numeric_columns = [
        "Directional Accuracy",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Specificity",
        "Balanced Accuracy",
        "Predicted up rate",
        "True up rate",
        "Majority baseline accuracy",
        "Momentum baseline accuracy",
        "Sharpe ratio",
        "Training time",
        "Inference time",
        "Parameter count",
    ]
    for model_name, group in frame.groupby("Model name", sort=False):
        row = {
            "Model name": model_name,
            "Dataset / asset": asset_label,
            "Model family": str(group["Model family"].iloc[0]),
            "Feature set": str(group["Feature set"].iloc[0]),
            "Validation threshold": "per-symbol",
            "Threshold objective": "per-symbol",
            "Validation F1": "per-symbol",
            "Validation accuracy": "per-symbol",
            "Validation balanced accuracy": "per-symbol",
            "Information coefficient": "per-symbol",
            "Data source": ", ".join(sorted(set(group["Data source"].astype(str)))),
            "Notes": f"Mean across {asset_list} for the bidirectional direction track.",
        }
        for column in numeric_columns:
            row[column] = float(pd.to_numeric(group[column], errors="coerce").mean())
        aggregate_rows.append(row)
    return aggregate_rows


def run_bidirectional_direction_experiment(
    symbols: list[str] | tuple[str, ...] = BENCHMARK_SYMBOLS,
    start: str = "2018-01-01",
    output_dir: Path = Path("output/bidirectional_direction"),
    epochs: int = 120,
    max_rows: int = 1200,
    hidden_dim: int = 48,
    learning_rate: float = 0.003,
    seed: int = 42,
    frames: Mapping[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Train baselines, ANN, QQBN, and QQTN for all requested symbols."""

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = default_device()

    result_rows: list[dict[str, object]] = []
    training_logs: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []

    model_factories = {
        "ANN": ANNClassifier,
        "QQBN": QQBNClassifier,
        "QQTN": QQTNClassifier,
    }

    for symbol in symbols:
        frame = frames.get(symbol) if frames is not None and symbol in frames else None
        dataset = prepare_bidirectional_direction_dataset(
            symbol=symbol,
            start=start,
            max_rows=max_rows,
            frame=frame,
            source="provided frame" if frame is not None else None,
        )

        majority_class = int(np.mean(dataset.train_y) >= 0.5)
        majority_predictions = np.full_like(dataset.test_y, majority_class, dtype=int)
        momentum_predictions = dataset.test_momentum_pred.astype(int)
        majority_accuracy = binary_direction_metrics(dataset.test_y, majority_predictions)["accuracy"]
        momentum_accuracy = binary_direction_metrics(dataset.test_y, momentum_predictions)["accuracy"]

        symbol_rows = [
            _baseline_result(dataset, "Majority baseline", majority_predictions, majority_accuracy, momentum_accuracy),
            _baseline_result(dataset, "Momentum baseline", momentum_predictions, majority_accuracy, momentum_accuracy),
        ]
        prediction_frame = pd.DataFrame(
            {
                "Date": dataset.test_dates,
                "Actual direction": dataset.test_y,
                "Next return": dataset.test_next_returns,
                "Previous close": dataset.test_prev_close,
                "Actual next close": dataset.test_actual_close,
                "Majority baseline Predicted direction": majority_predictions,
                "Momentum baseline Predicted direction": momentum_predictions,
            }
        )
        _save_confusion_matrix_plot(
            output_dir,
            symbol,
            "Majority baseline",
            dataset.test_y,
            majority_predictions,
        )
        _save_confusion_matrix_plot(
            output_dir,
            symbol,
            "Momentum baseline",
            dataset.test_y,
            momentum_predictions,
        )

        probability_cache: dict[str, dict[str, np.ndarray]] = {}
        for model_name, factory in model_factories.items():
            rows, logs, model_threshold_rows, prediction_payload, probability_payload = _train_and_evaluate_model(
                dataset=dataset,
                model_name=model_name,
                factory=factory,
                output_dir=output_dir,
                epochs=epochs,
                hidden_dim=hidden_dim,
                learning_rate=learning_rate,
                seed=seed,
                device=device,
                majority_accuracy=majority_accuracy,
                momentum_accuracy=momentum_accuracy,
            )
            symbol_rows.extend(rows)
            training_logs.extend(logs)
            threshold_rows.extend(model_threshold_rows)
            for column, values in prediction_payload.items():
                prediction_frame[column] = values
            probability_cache[model_name] = probability_payload

        if {"ANN", "QQBN", "QQTN"}.issubset(probability_cache):
            ensemble_val_probabilities = np.mean(
                [probability_cache[name]["val_probabilities"] for name in ("ANN", "QQBN", "QQTN")],
                axis=0,
            )
            ensemble_test_probabilities = np.mean(
                [probability_cache[name]["test_probabilities"] for name in ("ANN", "QQBN", "QQTN")],
                axis=0,
            )
            ensemble_rows, ensemble_threshold_rows, ensemble_payload = _ensemble_result(
                dataset=dataset,
                output_dir=output_dir,
                val_probabilities=ensemble_val_probabilities,
                test_probabilities=ensemble_test_probabilities,
                majority_accuracy=majority_accuracy,
                momentum_accuracy=momentum_accuracy,
            )
            symbol_rows.extend(ensemble_rows)
            threshold_rows.extend(ensemble_threshold_rows)
            for column, values in ensemble_payload.items():
                prediction_frame[column] = values

        result_rows.extend(symbol_rows)
        pd.DataFrame(symbol_rows).to_csv(output_dir / f"{symbol}_result_table.csv", index=False)
        prediction_frame.to_csv(output_dir / f"{symbol}_predictions.csv", index=False)

    aggregate_rows = _aggregate_result_rows(result_rows)
    result = pd.DataFrame(result_rows + aggregate_rows)
    result.to_csv(output_dir / "combined_result_table.csv", index=False)
    pd.DataFrame(training_logs).to_csv(output_dir / "training_log.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(output_dir / "thresholds.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the bidirectional up/down prediction track.")
    parser.add_argument("--symbols", nargs="+", default=list(BENCHMARK_SYMBOLS))
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--output-dir", type=Path, default=Path("output/bidirectional_direction"))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--max-rows", type=int, default=1200)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(
        run_bidirectional_direction_experiment(
            symbols=args.symbols,
            start=args.start,
            output_dir=args.output_dir,
            epochs=args.epochs,
            max_rows=args.max_rows,
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
            seed=args.seed,
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()


__all__ = [
    "BENCHMARK_SYMBOLS",
    "DIRECTION_FEATURE_COLUMNS",
    "DirectionDataset",
    "ThresholdCalibration",
    "add_bidirectional_direction_features",
    "calibrate_probability_threshold",
    "prepare_bidirectional_direction_dataset",
    "run_bidirectional_direction_experiment",
]
