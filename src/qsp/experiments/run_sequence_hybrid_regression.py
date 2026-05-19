"""Run a GPU-friendly sequence regression experiment with a qutrit-inspired head."""

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
from qsp.models.quantum_inspired import count_trainable_parameters, default_device
from qsp.models.sequence_hybrids import (
    BidirectionalLSTMBaseline,
    TemporalQQTNHybrid,
    predict_sequence_regressor,
    train_sequence_regressor,
)
from custom_qnn_financial_pipeline import directional_accuracy


FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Volume", "RSI", "MACD", "SMA5", "ADX", "Return_1"]


@dataclass(frozen=True)
class SequenceRegressionDataset:
    train_x: np.ndarray
    test_x: np.ndarray
    train_y_scaled: np.ndarray
    test_y_scaled: np.ndarray
    train_y_price: np.ndarray
    test_y_price: np.ndarray
    train_prev_close: np.ndarray
    test_prev_close: np.ndarray
    test_dates: pd.Series
    target_scaler: MinMaxScaler
    source: str
    note: str


def _prepare_sequence_regression_data(
    symbol: str,
    start: str,
    max_rows: int,
    window_size: int,
) -> SequenceRegressionDataset:
    frame, source, note = download_ohlcv_with_fallback(symbol=symbol, start=start)
    data = add_technical_indicators(frame)
    data["Return_1"] = data["Close"].pct_change()
    data["Target_Close"] = data["Close"].shift(-1)
    data["Prev_Close"] = data["Close"]
    data = data.dropna().tail(max_rows).copy()

    split_idx = int(len(data) * 0.8)
    if split_idx <= window_size + 5 or len(data) - split_idx < 10:
        raise ValueError("Not enough rows for the requested sequence regression experiment.")

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    feature_scaler.fit(data.iloc[:split_idx][FEATURE_COLUMNS].to_numpy(dtype=float))
    target_scaler.fit(data.iloc[:split_idx][["Target_Close"]].to_numpy(dtype=float))

    features = feature_scaler.transform(data[FEATURE_COLUMNS].to_numpy(dtype=float))
    targets_scaled = target_scaler.transform(data[["Target_Close"]].to_numpy(dtype=float)).ravel()
    targets_price = data["Target_Close"].to_numpy(dtype=float)
    prev_close = data["Prev_Close"].to_numpy(dtype=float)
    dates = pd.to_datetime(data.index)

    sequences: list[np.ndarray] = []
    seq_targets_scaled: list[float] = []
    seq_targets_price: list[float] = []
    seq_prev_close: list[float] = []
    seq_dates: list[pd.Timestamp] = []
    sample_end_indices: list[int] = []
    for end_idx in range(window_size - 1, len(data)):
        start_idx = end_idx - window_size + 1
        sequences.append(features[start_idx : end_idx + 1])
        seq_targets_scaled.append(float(targets_scaled[end_idx]))
        seq_targets_price.append(float(targets_price[end_idx]))
        seq_prev_close.append(float(prev_close[end_idx]))
        seq_dates.append(dates[end_idx + 1] if end_idx + 1 < len(dates) else dates[end_idx])
        sample_end_indices.append(end_idx)

    sample_end_indices = np.asarray(sample_end_indices, dtype=int)
    train_mask = sample_end_indices < split_idx
    test_mask = ~train_mask
    if train_mask.sum() < 20 or test_mask.sum() < 10:
        raise ValueError("The sequence split produced too few train or test samples.")

    return SequenceRegressionDataset(
        train_x=np.asarray(sequences, dtype=np.float32)[train_mask],
        test_x=np.asarray(sequences, dtype=np.float32)[test_mask],
        train_y_scaled=np.asarray(seq_targets_scaled, dtype=np.float32)[train_mask],
        test_y_scaled=np.asarray(seq_targets_scaled, dtype=np.float32)[test_mask],
        train_y_price=np.asarray(seq_targets_price, dtype=np.float64)[train_mask],
        test_y_price=np.asarray(seq_targets_price, dtype=np.float64)[test_mask],
        train_prev_close=np.asarray(seq_prev_close, dtype=np.float64)[train_mask],
        test_prev_close=np.asarray(seq_prev_close, dtype=np.float64)[test_mask],
        test_dates=pd.Series(np.asarray(seq_dates, dtype="datetime64[ns]")[test_mask]),
        target_scaler=target_scaler,
        source=source,
        note=note,
    )


def _save_loss_plot(output_dir: Path, model_name: str, train_losses: list[float], val_losses: list[float]) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="train")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
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
    plt.plot(dates, predicted_close, label=f"{model_name} predicted", linewidth=1.8)
    plt.plot(dates, naive_close, label="Naive previous close", linewidth=1.4, linestyle="--")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"{model_name} holdout curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name.lower()}_actual_vs_predicted.png", dpi=160)
    plt.close()


def run_sequence_hybrid_regression_experiment(
    symbol: str = "AAPL",
    start: str = "2018-01-01",
    output_dir: Path = Path("output/sequence_hybrid_regression"),
    epochs: int = 60,
    max_rows: int = 1000,
    window_size: int = 24,
    hidden_dim: int = 96,
    learning_rate: float = 0.001,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = _prepare_sequence_regression_data(
        symbol=symbol,
        start=start,
        max_rows=max_rows,
        window_size=window_size,
    )
    device = default_device()

    model_factories = {
        "BiLSTM regressor": lambda: BidirectionalLSTMBaseline(
            input_dim=dataset.train_x.shape[-1],
            hidden_dim=hidden_dim,
            dropout=0.15,
        ),
        "BiLSTM-QQTN regressor": lambda: TemporalQQTNHybrid(
            input_dim=dataset.train_x.shape[-1],
            hidden_dim=hidden_dim,
            qutrit_dim=max(16, hidden_dim // 2),
            dropout=0.15,
        ),
    }

    rows = []
    logs = []
    for model_name, factory in model_factories.items():
        model = factory()
        train_start = time.perf_counter()
        history = train_sequence_regressor(
            model,
            dataset.train_x,
            dataset.train_y_scaled,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=64,
            patience=12,
            device=device,
        )
        train_time = time.perf_counter() - train_start

        infer_start = time.perf_counter()
        pred_scaled = predict_sequence_regressor(model, dataset.test_x, device=device)
        inference_time = time.perf_counter() - infer_start
        pred_price = dataset.target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        rmse = float(np.sqrt(np.mean((dataset.test_y_price - pred_price) ** 2)))
        mae = float(np.mean(np.abs(dataset.test_y_price - pred_price)))
        da = float(directional_accuracy(dataset.test_y_price, pred_price, dataset.test_prev_close))
        _save_loss_plot(output_dir, model_name.replace(" ", "_"), history.train_losses, history.losses)
        _save_price_curve(
            output_dir,
            model_name.replace(" ", "_"),
            dataset.test_dates,
            dataset.test_y_price,
            pred_price,
            dataset.test_prev_close,
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
            "Time-ordered 80/20 split, next-day close regression target, MinMax-scaled target, and holdout price-curve comparison. "
            f"Window={window_size}, hidden_dim={hidden_dim}, device={history.device}."
        )
        if model_name == "BiLSTM-QQTN regressor":
            notes = (
                "Bidirectional LSTM encoder with attention pooling followed by a qutrit-inspired feature map and dense regression head. "
                + notes
            )
        else:
            notes = "Bidirectional LSTM encoder with attention pooling and classical regression head. " + notes
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
                "Directional Accuracy": da,
                "Training time": train_time,
                "Inference time": inference_time,
                "Parameter count": count_trainable_parameters(model),
                "Circuit depth": "N/A",
                "Data source": dataset.source,
                "Notes": notes,
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(output_dir / f"{symbol}_result_table.csv", index=False)
    pd.DataFrame(logs).to_csv(output_dir / f"{symbol}_training_log.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the sequence hybrid regression experiment.")
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--output-dir", type=Path, default=Path("output/sequence_hybrid_regression"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--max-rows", type=int, default=1000)
    parser.add_argument("--window-size", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()
    print(
        run_sequence_hybrid_regression_experiment(
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
