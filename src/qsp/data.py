"""Data utilities for stock-price QNN experiments."""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from custom_qnn_financial_pipeline import (
    FinancialDataset,
    add_technical_indicators,
    download_ohlcv,
    prepare_financial_dataset,
)


@dataclass(frozen=True)
class BinaryContextDataset:
    """Binary return contexts used by the Contextual QNN reproduction."""

    contexts: np.ndarray
    targets: np.ndarray
    returns: np.ndarray
    context_length: int
    horizon: int


def binary_return_labels(close_prices: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    """Encode one-step returns as 0 for non-positive and 1 for positive."""

    close = np.asarray(list(close_prices), dtype=float)
    if close.ndim != 1 or close.size < 2:
        raise ValueError("close_prices must contain at least two prices.")
    returns = np.diff(close) / close[:-1]
    labels = (returns > 0.0).astype(int)
    return labels, returns


def make_binary_context_dataset(
    close_prices: Iterable[float],
    context_length: int = 2,
    horizon: int = 1,
) -> BinaryContextDataset:
    """Create paper-style context/target sequences for binary quantization."""

    if context_length < 1:
        raise ValueError("context_length must be positive.")
    if horizon < 1:
        raise ValueError("horizon must be positive.")

    labels, returns = binary_return_labels(close_prices)
    last_start = len(labels) - context_length - horizon + 1
    if last_start <= 0:
        raise ValueError("Not enough labels for the requested context_length and horizon.")

    contexts = []
    targets = []
    for start in range(last_start):
        contexts.append(labels[start : start + context_length])
        targets.append(labels[start + context_length + horizon - 1])

    return BinaryContextDataset(
        contexts=np.asarray(contexts, dtype=int),
        targets=np.asarray(targets, dtype=int),
        returns=returns,
        context_length=context_length,
        horizon=horizon,
    )


def context_target_distribution(
    contexts: np.ndarray,
    targets: np.ndarray,
) -> pd.DataFrame:
    """Return empirical P(target | context) for binary Contextual QNN work."""

    contexts = np.asarray(contexts, dtype=int)
    targets = np.asarray(targets, dtype=int).ravel()
    if contexts.ndim != 2:
        raise ValueError("contexts must have shape (n_samples, context_length).")
    if len(contexts) != len(targets):
        raise ValueError("contexts and targets must contain the same number of samples.")

    rows = []
    for context in sorted({tuple(row.tolist()) for row in contexts}):
        mask = np.all(contexts == np.asarray(context), axis=1)
        count = int(mask.sum())
        p_up = float(targets[mask].mean()) if count else 0.0
        rows.append(
            {
                "context": "".join(str(bit) for bit in context),
                "count": count,
                "p_down": 1.0 - p_up,
                "p_up": p_up,
            }
        )
    return pd.DataFrame(rows)


def generate_sample_ohlcv(symbol: str, periods: int = 420, seed: int = 42) -> pd.DataFrame:
    """Create deterministic OHLCV data for offline demos when yfinance is unavailable."""

    symbol_seed = seed + sum(ord(char) for char in symbol)
    rng = np.random.default_rng(symbol_seed)
    base = 100.0 + (symbol_seed % 50)
    drift = 0.00035
    volatility = 0.018
    noise = rng.normal(drift, volatility, size=periods)
    seasonal = 0.004 * np.sin(np.linspace(0.0, 8.0 * np.pi, periods))
    returns = noise + seasonal
    close = base * np.exp(np.cumsum(returns))
    open_ = close * (1.0 + rng.normal(0.0, 0.003, size=periods))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.001, 0.012, size=periods))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.001, 0.012, size=periods))
    volume = rng.integers(5_000_000, 80_000_000, size=periods)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=periods)
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )


def download_ohlcv_with_fallback(
    symbol: str,
    start: str = "2018-01-01",
    cache_dir: str | None = None,
) -> tuple[pd.DataFrame, str, str]:
    """Download yfinance data, or return deterministic sample data with an error note."""

    try:
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            frame = download_ohlcv(symbol=symbol, start=start, cache_dir=Path(cache_dir) if cache_dir else None)
        return frame, "yfinance", ""
    except Exception as exc:
        note = f"yfinance unavailable: {exc.__class__.__name__}: {exc}"
        return generate_sample_ohlcv(symbol), "deterministic sample", note


__all__ = [
    "BinaryContextDataset",
    "FinancialDataset",
    "add_technical_indicators",
    "binary_return_labels",
    "context_target_distribution",
    "download_ohlcv",
    "download_ohlcv_with_fallback",
    "generate_sample_ohlcv",
    "make_binary_context_dataset",
    "prepare_financial_dataset",
]
