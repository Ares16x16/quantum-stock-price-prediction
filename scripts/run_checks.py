"""Lightweight checks that do not require pytest.

Run from the repository root:
    python scripts/run_checks.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def check_ast() -> None:
    files = (
        list((ROOT / "src").rglob("*.py"))
        + list((ROOT / "app").rglob("*.py"))
        + list((ROOT / "tests").rglob("*.py"))
        + [ROOT / "custom_qnn_financial_pipeline.py", ROOT / "reproduce_quantum_circuit.py"]
    )
    for path in files:
        ast.parse(path.read_text(encoding="utf-8"))
    print(f"AST parse OK: {len(files)} files")


def check_contextual_qnn() -> None:
    from qsp.data import (
        context_target_distribution,
        make_binary_context_dataset,
        make_discrete_context_dataset,
        multilevel_context_target_distribution,
    )
    from qsp.evaluation import binary_direction_metrics
    from qsp.models.contextual_qnn import ContextualQNN, ContextualQNNConfig, make_qmtl_asset_ids
    from qsp.models.contextual_qnn_multilevel import (
        MultiLevelContextualQNN,
        MultiLevelContextualQNNConfig,
    )

    close = [10, 11, 10, 12, 13, 12]
    data = make_binary_context_dataset(close, context_length=2, horizon=1)
    assert data.contexts.shape == (3, 2)
    assert data.targets.shape == (3,)

    dist = context_target_distribution(data.contexts, data.targets)
    assert ((dist["p_down"] + dist["p_up"] - 1.0).abs() < 1e-12).all()

    model = ContextualQNN(ContextualQNNConfig(context_length=2, num_assets=2, seed=42))
    asset_ids = make_qmtl_asset_ids([2, 1])
    loss = model.spsa_step(data.contexts, data.targets, asset_ids)
    assert loss == loss

    metrics = binary_direction_metrics(data.targets, model.predict(data.contexts, asset_ids))
    assert 0.0 <= metrics["accuracy"] <= 1.0

    multilevel = make_discrete_context_dataset(close, context_length=2, horizon=1, num_levels=4)
    assert multilevel.contexts.shape == (3, 2)
    assert multilevel.targets.shape == (3,)
    dist4 = multilevel_context_target_distribution(multilevel.contexts, multilevel.targets, num_levels=4)
    prob_cols = [f"p_{level}" for level in range(4)]
    assert ((dist4[prob_cols].sum(axis=1) - 1.0).abs() < 1e-12).all()

    multilevel_model = MultiLevelContextualQNN(MultiLevelContextualQNNConfig(context_length=2, num_levels=4))
    loss4 = multilevel_model.spsa_step(multilevel.contexts, multilevel.targets)
    assert loss4 == loss4
    pred4 = multilevel_model.predict_proba(multilevel.contexts)
    assert pred4.shape == (3, 4)
    assert np.allclose(pred4.sum(axis=1), 1.0)
    print("Contextual QNN checks OK")


def check_web_demo_helpers() -> None:
    from qsp.web_demo import SUPPORTED_TICKERS, run_multilevel_quick_prediction, run_quick_prediction

    assert "AAPL" in SUPPORTED_TICKERS
    assert "BTC-USD" in SUPPORTED_TICKERS
    prediction_binary, recent_binary, comparison_binary = run_quick_prediction("AAPL", epochs=1, max_samples=64)
    assert prediction_binary.symbol == "AAPL"
    assert not recent_binary.empty
    assert not comparison_binary.empty

    prediction, recent, probabilities, comparison = run_multilevel_quick_prediction("AAPL", epochs=1, max_samples=64)
    assert prediction.symbol == "AAPL"
    assert not recent.empty
    assert abs(float(probabilities["probability"].sum()) - 1.0) < 1e-9
    assert not comparison.empty
    print("Web demo helper checks OK")


def check_quantum_inspired_models() -> None:
    import numpy as np
    import torch

    from qsp.models.quantum_inspired import (
        QQBNClassifier,
        QQTNClassifier,
        predict_binary_classifier,
        train_binary_classifier,
    )

    train_x = np.asarray(
        [
            [0.1, 0.2, 0.3],
            [0.2, 0.2, 0.4],
            [0.8, 0.7, 0.6],
            [0.9, 0.8, 0.7],
        ],
        dtype=float,
    )
    train_y = np.asarray([0, 0, 1, 1], dtype=int)
    device = torch.device("cpu")
    for model in [QQBNClassifier(3), QQTNClassifier(3)]:
        history = train_binary_classifier(model, train_x, train_y, epochs=1, device=device)
        pred, proba = predict_binary_classifier(model, train_x, device=device)
        assert len(history.losses) == 1
        assert pred.shape == train_y.shape
        assert ((0.0 <= proba) & (proba <= 1.0)).all()
    print("Quantum-inspired model checks OK")


def check_sequence_hybrids() -> None:
    import torch

    from qsp.models.sequence_hybrids import (
        BidirectionalLSTMBaseline,
        TemporalQQTNHybrid,
        predict_sequence_classifier,
        train_sequence_classifier,
    )

    train_x = np.random.default_rng(42).random((16, 6, 4), dtype=np.float32)
    train_y = np.asarray([0, 1] * 8, dtype=int)
    device = torch.device("cpu")
    for model in [BidirectionalLSTMBaseline(4, hidden_dim=16), TemporalQQTNHybrid(4, hidden_dim=16, qutrit_dim=8)]:
        history = train_sequence_classifier(model, train_x, train_y, epochs=1, device=device)
        pred, proba = predict_sequence_classifier(model, train_x, device=device)
        assert len(history.losses) == 1
        assert pred.shape == train_y.shape
        assert ((0.0 <= proba) & (proba <= 1.0)).all()
    print("Sequence hybrid checks OK")


def check_bidirectional_direction_helpers() -> None:
    import pandas as pd

    from qsp.experiments.run_bidirectional_direction import (
        add_bidirectional_direction_features,
        calibrate_probability_threshold,
        prepare_bidirectional_direction_dataset,
    )

    dates = np.arange(90)
    close = 100.0 + 0.08 * dates + 2.0 * np.sin(dates / 4.0)
    frame = pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": 1_000_000 + 1000 * dates,
        },
        index=pd.bdate_range("2023-01-02", periods=len(dates)),
    )
    engineered = add_bidirectional_direction_features(frame)
    assert {"Target", "Next_Return", "Volatility_20", "MACD_Signal"}.issubset(engineered.columns)
    dataset = prepare_bidirectional_direction_dataset("AAPL", frame=frame, max_rows=70)
    assert len(dataset.train_x) > 0
    assert len(dataset.val_x) > 0
    assert len(dataset.test_x) > 0
    calibration = calibrate_probability_threshold(
        np.asarray([0, 0, 1, 1]),
        np.asarray([0.2, 0.4, 0.7, 0.9]),
    )
    assert 0.3 <= calibration.threshold <= 0.7
    print("Bidirectional direction helper checks OK")


def check_custom_qnn_architecture() -> None:
    from qsp.models.custom_qnn import build_custom_qnn_circuit, build_original_custom_qnn_circuit

    original = build_original_custom_qnn_circuit()
    refactored, input_params, weight_params = build_custom_qnn_circuit()
    assert original.num_qubits == refactored.num_qubits == 5
    assert dict(original.count_ops()) == dict(refactored.count_ops())
    assert original.depth() == refactored.depth() == 33
    assert len(input_params) == 5
    assert len(weight_params) == 49
    print("CustomQNN architecture check OK")


def main() -> None:
    check_ast()
    check_contextual_qnn()
    check_web_demo_helpers()
    check_quantum_inspired_models()
    check_sequence_hybrids()
    check_bidirectional_direction_helpers()
    check_custom_qnn_architecture()
    print("All lightweight checks passed")


if __name__ == "__main__":
    main()
