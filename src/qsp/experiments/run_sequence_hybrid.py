"""Run a stronger GPU-friendly sequence hybrid experiment.

This module is deliberately separate from the preserved HQNN-FSP circuit. It
explores whether a richer classical temporal encoder paired with a
qutrit-inspired head can produce a more useful directional forecast on local
hardware.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from qsp.data import add_technical_indicators, download_ohlcv_with_fallback
from qsp.evaluation import binary_direction_metrics
from qsp.models.quantum_inspired import count_trainable_parameters, default_device
from qsp.models.sequence_hybrids import (
    BidirectionalLSTMBaseline,
    TemporalQQTNHybrid,
    predict_sequence_classifier,
    train_sequence_classifier,
)


FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Volume", "RSI", "MACD", "SMA5", "ADX", "Return_1"]


@dataclass(frozen=True)
class SequenceDataset:
    train_x: np.ndarray
    test_x: np.ndarray
    train_y: np.ndarray
    test_y: np.ndarray
    train_next_returns: np.ndarray
    test_next_returns: np.ndarray
    train_prev_close: np.ndarray
    test_prev_close: np.ndarray
    train_actual_close: np.ndarray
    test_actual_close: np.ndarray
    test_dates: pd.Series
    source: str
    note: str


def _prepare_sequence_direction_data(
    symbol: str,
    start: str,
    max_rows: int,
    window_size: int,
) -> SequenceDataset:
    frame, source, note = download_ohlcv_with_fallback(symbol=symbol, start=start)
    data = add_technical_indicators(frame)
    data["Return_1"] = data["Close"].pct_change()
    data["Next_Return"] = data["Close"].pct_change().shift(-1)
    data["Target"] = (data["Next_Return"] > 0.0).astype(int)
    data["Prev_Close"] = data["Close"]
    data["Actual_Next_Close"] = data["Close"].shift(-1)
    data = data.dropna().tail(max_rows).copy()

    split_idx = int(len(data) * 0.8)
    if split_idx <= window_size + 5 or len(data) - split_idx < 10:
        raise ValueError("Not enough rows for the requested sequence hybrid experiment.")

    scaler = MinMaxScaler()
    scaler.fit(data.iloc[:split_idx][FEATURE_COLUMNS].to_numpy(dtype=float))
    features = scaler.transform(data[FEATURE_COLUMNS].to_numpy(dtype=float))
    targets = data["Target"].to_numpy(dtype=int)
    next_returns = data["Next_Return"].to_numpy(dtype=float)
    prev_close = data["Prev_Close"].to_numpy(dtype=float)
    actual_next_close = data["Actual_Next_Close"].to_numpy(dtype=float)
    dates = pd.to_datetime(data.index)

    sequences: list[np.ndarray] = []
    seq_targets: list[int] = []
    seq_next_returns: list[float] = []
    seq_prev_close: list[float] = []
    seq_actual_close: list[float] = []
    seq_dates: list[pd.Timestamp] = []
    sample_end_indices: list[int] = []
    for end_idx in range(window_size - 1, len(data)):
        start_idx = end_idx - window_size + 1
        sequences.append(features[start_idx : end_idx + 1])
        seq_targets.append(int(targets[end_idx]))
        seq_next_returns.append(float(next_returns[end_idx]))
        seq_prev_close.append(float(prev_close[end_idx]))
        seq_actual_close.append(float(actual_next_close[end_idx]))
        seq_dates.append(dates[end_idx + 1] if end_idx + 1 < len(dates) else dates[end_idx])
        sample_end_indices.append(end_idx)

    sample_end_indices = np.asarray(sample_end_indices, dtype=int)
    train_mask = sample_end_indices < split_idx
    test_mask = ~train_mask
    if train_mask.sum() < 20 or test_mask.sum() < 10:
        raise ValueError("The sequence split produced too few train or test samples.")

    return SequenceDataset(
        train_x=np.asarray(sequences, dtype=np.float32)[train_mask],
        test_x=np.asarray(sequences, dtype=np.float32)[test_mask],
        train_y=np.asarray(seq_targets, dtype=int)[train_mask],
        test_y=np.asarray(seq_targets, dtype=int)[test_mask],
        train_next_returns=np.asarray(seq_next_returns, dtype=float)[train_mask],
        test_next_returns=np.asarray(seq_next_returns, dtype=float)[test_mask],
        train_prev_close=np.asarray(seq_prev_close, dtype=float)[train_mask],
        test_prev_close=np.asarray(seq_prev_close, dtype=float)[test_mask],
        train_actual_close=np.asarray(seq_actual_close, dtype=float)[train_mask],
        test_actual_close=np.asarray(seq_actual_close, dtype=float)[test_mask],
        test_dates=pd.Series(np.asarray(seq_dates, dtype="datetime64[ns]")[test_mask]),
        source=source,
        note=note,
    )


def _save_loss_plot(output_dir: Path, model_name: str, train_losses: list[float], val_losses: list[float]) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="train")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("BCE loss")
    plt.title(f"{model_name} loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name.lower()}_loss.png", dpi=160)
    plt.close()


def _save_price_curve(
    output_dir: Path,
    model_name: str,
    dates: pd.Series,
    actual_close: np.ndarray,
    predicted_close: np.ndarray,
    naive_close: np.ndarray,
) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(dates, actual_close, label="Actual next close", linewidth=2.0)
    plt.plot(dates, predicted_close, label=f"{model_name} implied next close", linewidth=1.8)
    plt.plot(dates, naive_close, label="Naive previous close", linewidth=1.4, linestyle="--")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"{model_name} holdout curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name.lower()}_actual_vs_predicted.png", dpi=160)
    plt.close()


def _implied_close_path(
    probabilities: np.ndarray,
    train_targets: np.ndarray,
    train_next_returns: np.ndarray,
    prev_close_test: np.ndarray,
) -> np.ndarray:
    up_returns = train_next_returns[train_targets == 1]
    down_returns = train_next_returns[train_targets == 0]
    up_mean = float(up_returns.mean()) if len(up_returns) else float(max(train_next_returns.mean(), 0.0))
    down_mean = float(down_returns.mean()) if len(down_returns) else float(min(train_next_returns.mean(), 0.0))
    expected_returns = probabilities * up_mean + (1.0 - probabilities) * down_mean
    return prev_close_test * (1.0 + expected_returns)


def run_sequence_hybrid_experiment(
    symbol: str = "AAPL",
    start: str = "2018-01-01",
    output_dir: Path = Path("output/sequence_hybrid_aapl"),
    epochs: int = 50,
    max_rows: int = 900,
    window_size: int = 20,
    hidden_dim: int = 64,
    learning_rate: float = 0.001,
) -> pd.DataFrame:
    """Train a stronger BiLSTM baseline and a BiLSTM-QQTN hybrid."""

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = _prepare_sequence_direction_data(
        symbol=symbol,
        start=start,
        max_rows=max_rows,
        window_size=window_size,
    )
    device = default_device()

    model_factories = {
        "BiLSTM baseline": lambda: BidirectionalLSTMBaseline(
            input_dim=dataset.train_x.shape[-1],
            hidden_dim=hidden_dim,
            dropout=0.25,
        ),
        "BiLSTM-QQTN hybrid": lambda: TemporalQQTNHybrid(
            input_dim=dataset.train_x.shape[-1],
            hidden_dim=hidden_dim,
            qutrit_dim=max(16, hidden_dim // 2),
            dropout=0.25,
        ),
    }

    rows = []
    logs = []
    for model_name, factory in model_factories.items():
        model = factory()
        train_start = time.perf_counter()
        history = train_sequence_classifier(
            model,
            dataset.train_x,
            dataset.train_y,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=64,
            patience=10,
            device=device,
        )
        train_time = time.perf_counter() - train_start

        infer_start = time.perf_counter()
        pred, probabilities = predict_sequence_classifier(model, dataset.test_x, device=device)
        inference_time = time.perf_counter() - infer_start
        metrics = binary_direction_metrics(dataset.test_y, pred)
        implied_close = _implied_close_path(
            probabilities,
            dataset.train_y,
            dataset.train_next_returns,
            dataset.test_prev_close,
        )
        naive_close = dataset.test_prev_close
        rmse = float(np.sqrt(np.mean((dataset.test_actual_close - implied_close) ** 2)))
        mae = float(np.mean(np.abs(dataset.test_actual_close - implied_close)))
        _save_loss_plot(output_dir, model_name.replace(" ", "_"), history.train_losses, history.losses)
        _save_price_curve(
            output_dir,
            model_name.replace(" ", "_"),
            dataset.test_dates,
            dataset.test_actual_close,
            implied_close,
            naive_close,
        )
        logs.extend(
            {
                "model": model_name,
                "epoch": index + 1,
                "train_loss": history.train_losses[index] if index < len(history.train_losses) else np.nan,
                "val_loss": loss,
                "device": history.device,
            }
            for index, loss in enumerate(history.losses)
        )

        notes = (
            "Time-ordered 80/20 split, next-day direction target, implied next-close curve derived from train-set mean up/down returns. "
            f"Window={window_size}, hidden_dim={hidden_dim}, device={history.device}."
        )
        if model_name == "BiLSTM-QQTN hybrid":
            notes = (
                "Bidirectional LSTM encoder with attention pooling followed by a qutrit-inspired feature map and dense classifier. "
                + notes
            )
        else:
            notes = "Bidirectional LSTM encoder with attention pooling and classical dense head. " + notes
        if dataset.note:
            notes += f" {dataset.note}"

        rows.append(
            {
                "Model name": model_name,
                "Dataset / asset": symbol,
                "Number of qubits": "N/A",
                "VQC layers": "N/A",
                "Feature set": ", ".join(FEATURE_COLUMNS),
                "RMSE": rmse,
                "MAE": mae,
                "Directional Accuracy": metrics["accuracy"],
                "Training time": train_time,
                "Inference time": inference_time,
                "Parameter count": count_trainable_parameters(model),
                "Circuit depth": "N/A",
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1": metrics["f1"],
                "Data source": dataset.source,
                "Notes": notes,
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(output_dir / f"{symbol}_result_table.csv", index=False)
    pd.DataFrame(logs).to_csv(output_dir / f"{symbol}_training_log.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the sequence hybrid directional forecasting experiment.")
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--output-dir", type=Path, default=Path("output/sequence_hybrid_aapl"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max-rows", type=int, default=900)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()
    print(
        run_sequence_hybrid_experiment(
            symbol=args.symbol,
            start=args.start,
            output_dir=args.output_dir,
            epochs=args.epochs,
            max_rows=args.max_rows,
            window_size=args.window_size,
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
        )
    )


if __name__ == "__main__":
    main()
