"""Helpers for the Streamlit demo.

The web demo must stay responsive, so it uses lightweight predictions:

- naive next close: previous close
- ContextualQNN direction: cached binary direction scaffold
- multi-level ContextualQNN regime prediction with d=4

It intentionally does not run expensive Qiskit CustomQNN training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qsp.data import (
    download_ohlcv_with_fallback,
    make_binary_context_dataset,
    make_discrete_context_dataset,
)
from qsp.evaluation import binary_direction_metrics
from qsp.models.contextual_qnn import ContextualQNN, ContextualQNNConfig
from qsp.models.contextual_qnn_multilevel import (
    MultiLevelContextualQNN,
    MultiLevelContextualQNNConfig,
)

SUPPORTED_TICKERS: dict[str, str] = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "NVDA": "NVIDIA",
    "TSLA": "Tesla",
    "0700.HK": "Tencent",
    "BTC-USD": "Bitcoin USD",
}

@dataclass(frozen=True)
class QuickPrediction:
    symbol: str
    company: str
    last_close: float
    previous_close: float
    last_date: str
    naive_next_close: float
    latest_context: str
    contextual_probability_up: float
    contextual_direction: str
    holdout_accuracy: float
    holdout_f1: float
    train_samples: int
    test_samples: int
    data_source: str


@dataclass(frozen=True)
class MultiLevelQuickPrediction:
    symbol: str
    company: str
    last_close: float
    previous_close: float
    last_date: str
    latest_context: str
    predicted_bucket: int
    predicted_bucket_label: str
    holdout_accuracy: float
    train_samples: int
    test_samples: int
    data_source: str


def _download_or_fallback(symbol: str, start: str) -> tuple[pd.DataFrame, str]:
    frame, source, _note = download_ohlcv_with_fallback(symbol=symbol, start=start)
    return frame, source


def _bucket_label(bin_edges: np.ndarray, level: int) -> str:
    low = 100.0 * float(bin_edges[level])
    high = 100.0 * float(bin_edges[level + 1])
    return f"{low:.2f}% to {high:.2f}%"


def run_quick_prediction(
    symbol: str,
    start: str = "2018-01-01",
    context_length: int = 2,
    epochs: int = 20,
    max_samples: int = 128,
) -> tuple[QuickPrediction, pd.DataFrame]:
    """Run a cached lightweight prediction for one selected ticker."""

    if symbol not in SUPPORTED_TICKERS:
        raise ValueError(f"Unsupported symbol: {symbol}")

    frame, data_source = _download_or_fallback(symbol, start)
    close = frame["Close"].astype(float).to_numpy()
    if len(close) < context_length + 5:
        raise ValueError(f"Not enough price history for {symbol}.")

    data = make_binary_context_dataset(close, context_length=context_length, horizon=1)
    contexts = data.contexts[-max_samples:]
    targets = data.targets[-max_samples:]
    split = int(len(contexts) * 0.8)
    train_x, test_x = contexts[:split], contexts[split:]
    train_y, test_y = targets[:split], targets[split:]

    model = ContextualQNN(
        ContextualQNNConfig(
            context_length=context_length,
            horizon=1,
            num_layers=4,
            seed=42,
            learning_rate=0.3,
            spsa_perturbation=0.01,
        )
    )
    model.fit(train_x, train_y, epochs=epochs)
    pred = model.predict(test_x)
    metrics = binary_direction_metrics(test_y, pred)

    latest_context = data.contexts[-1]
    prob_up = model.probability_up(latest_context)
    direction = "Up / positive return" if prob_up >= 0.5 else "Down / non-positive return"
    last_date = str(pd.Timestamp(frame.index[-1]).date())
    prediction = QuickPrediction(
        symbol=symbol,
        company=SUPPORTED_TICKERS[symbol],
        last_close=float(close[-1]),
        previous_close=float(close[-2]),
        last_date=last_date,
        naive_next_close=float(close[-1]),
        latest_context="".join(str(int(bit)) for bit in latest_context),
        contextual_probability_up=float(prob_up),
        contextual_direction=direction,
        holdout_accuracy=float(metrics["accuracy"]),
        holdout_f1=float(metrics["f1"]),
        train_samples=len(train_x),
        test_samples=len(test_x),
        data_source=data_source,
    )
    recent = frame.tail(90).rename_axis("Date").reset_index()
    return prediction, recent


def run_multilevel_quick_prediction(
    symbol: str,
    start: str = "2018-01-01",
    context_length: int = 2,
    num_levels: int = 4,
    epochs: int = 80,
    max_samples: int = 160,
) -> tuple[MultiLevelQuickPrediction, pd.DataFrame, pd.DataFrame]:
    """Run a cached d=4 ContextualQNN regime prediction for one selected ticker."""

    if symbol not in SUPPORTED_TICKERS:
        raise ValueError(f"Unsupported symbol: {symbol}")

    frame, data_source = _download_or_fallback(symbol, start)
    close = frame["Close"].astype(float).to_numpy()
    if len(close) < context_length + 5:
        raise ValueError(f"Not enough price history for {symbol}.")

    data = make_discrete_context_dataset(
        close,
        context_length=context_length,
        horizon=1,
        num_levels=num_levels,
    )
    contexts = data.contexts[-max_samples:]
    targets = data.targets[-max_samples:]
    split = int(len(contexts) * 0.8)
    train_x, test_x = contexts[:split], contexts[split:]
    train_y, test_y = targets[:split], targets[split:]

    model = MultiLevelContextualQNN(
        MultiLevelContextualQNNConfig(
            context_length=context_length,
            horizon=1,
            num_levels=num_levels,
            num_layers=4,
            seed=42,
            learning_rate=0.05,
            spsa_perturbation=0.01,
        )
    )
    model.fit(train_x, train_y, epochs=epochs)
    pred = model.predict(test_x)
    accuracy = float(np.mean(pred == test_y)) if len(test_y) else 0.0

    latest_context = data.contexts[-1]
    proba = model.probability_distribution(latest_context)
    predicted_bucket = int(np.argmax(proba))
    bucket_labels = [_bucket_label(data.bin_edges, level) for level in range(num_levels)]

    prediction = MultiLevelQuickPrediction(
        symbol=symbol,
        company=SUPPORTED_TICKERS[symbol],
        last_close=float(close[-1]),
        previous_close=float(close[-2]),
        last_date=str(pd.Timestamp(frame.index[-1]).date()),
        latest_context="-".join(str(int(level)) for level in latest_context),
        predicted_bucket=predicted_bucket,
        predicted_bucket_label=bucket_labels[predicted_bucket],
        holdout_accuracy=accuracy,
        train_samples=len(train_x),
        test_samples=len(test_x),
        data_source=data_source,
    )
    recent = frame.tail(90).rename_axis("Date").reset_index()
    probability_frame = pd.DataFrame(
        {
            "bucket": list(range(num_levels)),
            "label": bucket_labels,
            "probability": proba,
        }
    )
    return prediction, recent, probability_frame


__all__ = [
    "MultiLevelQuickPrediction",
    "QuickPrediction",
    "SUPPORTED_TICKERS",
    "run_multilevel_quick_prediction",
    "run_quick_prediction",
]
