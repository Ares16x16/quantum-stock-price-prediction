"""Run ANN, QQBN, and QQTN experiments from the qubit/qutrit paper."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

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
    predict_binary_classifier,
    train_binary_classifier,
)


FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Volume", "RSI", "MACD", "SMA5", "ADX", "Return_1"]


def _prepare_direction_data(symbol: str, start: str, max_samples: int):
    frame, source, note = download_ohlcv_with_fallback(symbol=symbol, start=start)
    data = add_technical_indicators(frame)
    data["Next_Return"] = data["Close"].pct_change().shift(-1)
    data["Target"] = (data["Next_Return"] > 0.0).astype(int)
    data = data.dropna().tail(max_samples)

    features = data[FEATURE_COLUMNS].to_numpy(dtype=float)
    targets = data["Target"].to_numpy(dtype=int)
    next_returns = data["Next_Return"].to_numpy(dtype=float)
    split = int(len(data) * 0.8)
    if split < 10 or len(data) - split < 5:
        raise ValueError("Not enough samples for ANN/QQBN/QQTN experiment.")

    scaler = MinMaxScaler()
    train_x = scaler.fit_transform(features[:split])
    test_x = scaler.transform(features[split:])
    return train_x, test_x, targets[:split], targets[split:], next_returns[split:], source, note


def _information_coefficient(probabilities: np.ndarray, next_returns: np.ndarray) -> float:
    if len(probabilities) < 2 or np.std(probabilities) == 0.0 or np.std(next_returns) == 0.0:
        return 0.0
    return float(np.corrcoef(probabilities, next_returns)[0, 1])


def _strategy_sharpe(predictions: np.ndarray, next_returns: np.ndarray) -> float:
    positions = np.where(predictions == 1, 1.0, -1.0)
    strategy_returns = positions * next_returns
    std = float(np.std(strategy_returns))
    if std == 0.0:
        return 0.0
    return float(np.sqrt(252.0) * np.mean(strategy_returns) / std)


def _save_loss_plot(output_dir: Path, model_name: str, losses: list[float]) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel("Epoch")
    plt.ylabel("BCE loss")
    plt.title(f"{model_name} training loss")
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name.lower()}_loss.png", dpi=160)
    plt.close()


def run_quantum_inspired_experiment(
    symbol: str = "AAPL",
    start: str = "2018-01-01",
    output_dir: Path = Path("output/quantum_inspired"),
    epochs: int = 60,
    max_samples: int = 420,
    hidden_dim: int = 48,
) -> pd.DataFrame:
    """Train ANN, QQBN, and QQTN on next-day direction prediction."""

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    train_x, test_x, train_y, test_y, next_returns, source, source_note = _prepare_direction_data(
        symbol,
        start,
        max_samples,
    )

    model_factories = {
        "ANN": ANNClassifier,
        "QQBN": QQBNClassifier,
        "QQTN": QQTNClassifier,
    }
    rows = []
    logs = []
    for model_name, factory in model_factories.items():
        torch.manual_seed(42)
        model = factory(input_dim=train_x.shape[1], hidden_dim=hidden_dim)
        started = time.perf_counter()
        history = train_binary_classifier(model, train_x, train_y, epochs=epochs, learning_rate=0.01, seed=42)
        train_time = time.perf_counter() - started

        infer_start = time.perf_counter()
        pred, probabilities = predict_binary_classifier(model, test_x)
        inference_time = time.perf_counter() - infer_start
        metrics = binary_direction_metrics(test_y, pred)
        sharpe = _strategy_sharpe(pred, next_returns)
        info_coef = _information_coefficient(probabilities, next_returns)
        _save_loss_plot(output_dir, model_name, history.losses)
        logs.extend({"model": model_name, "epoch": index + 1, "loss": loss} for index, loss in enumerate(history.losses))

        if model_name == "ANN":
            notes = "Classical neural-network baseline from the qubit/qutrit paper."
        elif model_name == "QQBN":
            notes = "Qubit-inspired two-state feature map followed by a trainable neural layer."
        else:
            notes = "Qutrit-inspired three-state feature map followed by a trainable neural layer."
        if source_note:
            notes += f" {source_note}"

        rows.append(
            {
                "Model name": model_name,
                "Dataset / asset": symbol,
                "Number of qubits": "N/A",
                "VQC layers": "N/A",
                "Feature set": ", ".join(FEATURE_COLUMNS),
                "RMSE": "N/A",
                "MAE": "N/A",
                "Directional Accuracy": metrics["accuracy"],
                "Training time": train_time,
                "Inference time": inference_time,
                "Parameter count": count_trainable_parameters(model),
                "Circuit depth": "N/A",
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1": metrics["f1"],
                "Sharpe ratio": sharpe,
                "Information coefficient": info_coef,
                "Data source": source,
                "Notes": f"{notes} Training used Adam, BCEWithLogitsLoss, hidden_dim={hidden_dim}, epochs={epochs}.",
            }
        )

    predictions = pd.DataFrame({"target": test_y, "next_return": next_returns})
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / f"{symbol}_result_table.csv", index=False)
    pd.DataFrame(logs).to_csv(output_dir / f"{symbol}_training_log.csv", index=False)
    predictions.to_csv(output_dir / f"{symbol}_test_targets.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ANN, QQBN, and QQTN reproduction experiment.")
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--output-dir", type=Path, default=Path("output/quantum_inspired"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--max-samples", type=int, default=420)
    parser.add_argument("--hidden-dim", type=int, default=48)
    args = parser.parse_args()
    print(
        run_quantum_inspired_experiment(
            symbol=args.symbol,
            start=args.start,
            output_dir=args.output_dir,
            epochs=args.epochs,
            max_samples=args.max_samples,
            hidden_dim=args.hidden_dim,
        )
    )


if __name__ == "__main__":
    main()
