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


@dataclass(frozen=True)
class DiscreteContextDataset:
    """Multi-level return contexts for higher-resolution Contextual QNN work."""

    contexts: np.ndarray
    targets: np.ndarray
    returns: np.ndarray
    context_length: int
    horizon: int
    num_levels: int
    bin_edges: np.ndarray


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


def discrete_return_labels(
    close_prices: Iterable[float],
    num_levels: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize returns into `num_levels` density-based buckets."""

    if num_levels < 2:
        raise ValueError("num_levels must be at least 2.")
    close = np.asarray(list(close_prices), dtype=float)
    if close.ndim != 1 or close.size < 2:
        raise ValueError("close_prices must contain at least two prices.")

    returns = np.diff(close) / close[:-1]
    quantiles = np.linspace(0.0, 1.0, num_levels + 1)
    bin_edges = np.quantile(returns, quantiles)
    # If repeated edges collapse the quantile bins, fall back to equal-width bins.
    if len(np.unique(np.round(bin_edges, 12))) < num_levels + 1:
        bin_edges = np.linspace(float(returns.min()), float(returns.max()), num_levels + 1)

    # Ensure the outer edges include the min/max robustly.
    bin_edges = np.asarray(bin_edges, dtype=float)
    bin_edges[0] = np.nextafter(bin_edges[0], -np.inf)
    bin_edges[-1] = np.nextafter(bin_edges[-1], np.inf)
    labels = np.digitize(returns, bin_edges[1:-1], right=False).astype(int)
    return labels, returns, bin_edges


def make_discrete_context_dataset(
    close_prices: Iterable[float],
    context_length: int = 2,
    horizon: int = 1,
    num_levels: int = 4,
) -> DiscreteContextDataset:
    """Create context/target sequences for multi-level return quantization."""

    if context_length < 1:
        raise ValueError("context_length must be positive.")
    if horizon < 1:
        raise ValueError("horizon must be positive.")
    labels, returns, bin_edges = discrete_return_labels(close_prices, num_levels=num_levels)
    last_start = len(labels) - context_length - horizon + 1
    if last_start <= 0:
        raise ValueError("Not enough labels for the requested context_length and horizon.")

    contexts = []
    targets = []
    for start in range(last_start):
        contexts.append(labels[start : start + context_length])
        targets.append(labels[start + context_length + horizon - 1])

    return DiscreteContextDataset(
        contexts=np.asarray(contexts, dtype=int),
        targets=np.asarray(targets, dtype=int),
        returns=returns,
        context_length=context_length,
        horizon=horizon,
        num_levels=num_levels,
        bin_edges=bin_edges,
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


def multilevel_context_target_distribution(
    contexts: np.ndarray,
    targets: np.ndarray,
    num_levels: int,
) -> pd.DataFrame:
    """Return empirical P(target | context) for multi-level contexts."""

    contexts = np.asarray(contexts, dtype=int)
    targets = np.asarray(targets, dtype=int).ravel()
    if contexts.ndim != 2:
        raise ValueError("contexts must have shape (n_samples, context_length).")
    if len(contexts) != len(targets):
        raise ValueError("contexts and targets must contain the same number of samples.")

    rows = []
    for context in sorted({tuple(row.tolist()) for row in contexts}):
        mask = np.all(contexts == np.asarray(context), axis=1)
        row = {
            "context": "-".join(str(int(level)) for level in context),
            "count": int(mask.sum()),
        }
        for level in range(num_levels):
            row[f"p_{level}"] = float(np.mean(targets[mask] == level)) if mask.any() else 0.0
        rows.append(row)
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
    "DiscreteContextDataset",
    "FinancialDataset",
    "add_technical_indicators",
    "binary_return_labels",
    "context_target_distribution",
    "discrete_return_labels",
    "download_ohlcv",
    "download_ohlcv_with_fallback",
    "generate_sample_ohlcv",
    "make_binary_context_dataset",
    "make_discrete_context_dataset",
    "multilevel_context_target_distribution",
    "prepare_financial_dataset",
]
