import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from qsp.experiments.run_bidirectional_direction import (
    BENCHMARK_SYMBOLS,
    DIRECTION_FEATURE_COLUMNS,
    add_bidirectional_direction_features,
    calibrate_probability_threshold,
    prepare_bidirectional_direction_dataset,
    run_bidirectional_direction_experiment,
)


def _sample_ohlcv(periods: int = 130, seed: int = 42, base: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=periods)
    trend = np.linspace(0.0, 0.25, periods)
    cycle = 0.025 * np.sin(np.linspace(0.0, 8.0 * np.pi, periods))
    noise = rng.normal(0.0, 0.008, periods)
    close = base * np.exp(np.cumsum(0.001 + cycle + trend / periods + noise))
    open_ = close * (1.0 + rng.normal(0.0, 0.003, periods))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.002, 0.012, periods))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.002, 0.012, periods))
    volume = rng.integers(1_000_000, 5_000_000, periods)
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


def test_direction_features_use_next_return_label():
    frame = _sample_ohlcv()
    data = add_bidirectional_direction_features(frame)
    row = data.iloc[5]
    current_close = float(frame.loc[row.name, "Close"])
    next_position = frame.index.get_loc(row.name) + 1
    next_close = float(frame.iloc[next_position]["Close"])
    expected_return = next_close / current_close - 1.0
    assert np.isclose(row["Next_Return"], expected_return)
    assert int(row["Target"]) == int(expected_return > 0.0)


def test_feature_rows_do_not_depend_on_later_prices():
    frame = _sample_ohlcv()
    data = add_bidirectional_direction_features(frame)
    checked_index = data.index[10]

    modified = frame.copy()
    later_rows = modified.index > checked_index
    modified.loc[later_rows, "Close"] = modified.loc[later_rows, "Close"] * 3.0
    modified.loc[later_rows, "High"] = modified.loc[later_rows, "High"] * 3.0
    modified.loc[later_rows, "Low"] = modified.loc[later_rows, "Low"] * 3.0
    modified.loc[later_rows, "Open"] = modified.loc[later_rows, "Open"] * 3.0
    modified_data = add_bidirectional_direction_features(modified)

    original_features = data.loc[checked_index, DIRECTION_FEATURE_COLUMNS].astype(float).to_numpy()
    modified_features = modified_data.loc[checked_index, DIRECTION_FEATURE_COLUMNS].astype(float).to_numpy()
    assert np.allclose(original_features, modified_features)


def test_prepare_dataset_uses_ordered_train_validation_test_split():
    dataset = prepare_bidirectional_direction_dataset("AAPL", frame=_sample_ohlcv(), max_rows=100)
    assert len(dataset.train_x) > len(dataset.val_x) > 0
    assert len(dataset.test_x) > 0
    assert dataset.train_dates.max() < dataset.val_dates.min()
    assert dataset.val_dates.max() < dataset.test_dates.min()
    assert dataset.train_x.shape[1] == len(DIRECTION_FEATURE_COLUMNS)
    assert dataset.train_x.shape[1] == dataset.val_x.shape[1] == dataset.test_x.shape[1]


def test_threshold_calibration_returns_valid_threshold_and_metrics():
    y_true = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.15, 0.35, 0.62, 0.82])
    calibration = calibrate_probability_threshold(y_true, probabilities)
    assert 0.30 <= calibration.threshold <= 0.70
    assert 0.0 <= calibration.metrics["accuracy"] <= 1.0
    assert 0.0 <= calibration.metrics["f1"] <= 1.0


def test_bidirectional_direction_smoke_outputs_all_benchmark_symbols(tmp_path):
    frames = {
        symbol: _sample_ohlcv(periods=95, seed=42 + index, base=100.0 + 15.0 * index)
        for index, symbol in enumerate(BENCHMARK_SYMBOLS)
    }
    result = run_bidirectional_direction_experiment(
        symbols=list(BENCHMARK_SYMBOLS),
        output_dir=tmp_path,
        epochs=1,
        max_rows=80,
        hidden_dim=8,
        learning_rate=0.005,
        frames=frames,
    )
    symbol_rows = result[result["Dataset / asset"].isin(BENCHMARK_SYMBOLS)]
    assert set(symbol_rows["Dataset / asset"]) == set(BENCHMARK_SYMBOLS)
    assert set(["Majority baseline", "Momentum baseline", "ANN", "QQBN", "QQTN"]).issubset(
        set(symbol_rows["Model name"])
    )
    assert "QQTN balanced threshold" in set(symbol_rows["Model name"])
    assert "ANN+QQBN+QQTN ensemble" in set(symbol_rows["Model name"])
    assert "ANN+QQBN+QQTN balanced ensemble" in set(symbol_rows["Model name"])
    assert "Balanced Accuracy" in result.columns
    assert "Predicted up rate" in result.columns
    assert (tmp_path / "combined_result_table.csv").exists()
    assert (tmp_path / "AAPL_predictions.csv").exists()
    thresholds = pd.read_csv(tmp_path / "thresholds.csv")
    assert "threshold_objective" in thresholds.columns
    assert "balanced_accuracy" in set(thresholds["threshold_objective"])
    assert "accuracy" in set(thresholds["threshold_objective"])
    predictions = pd.read_csv(tmp_path / "AAPL_predictions.csv")
    probability_columns = [column for column in predictions.columns if column.endswith("Probability up")]
    assert probability_columns
    for column in probability_columns:
        assert predictions[column].between(0.0, 1.0).all()
