"""Lightweight checks that do not require pytest.

Run from the repository root:
    python scripts/run_checks.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

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
    from qsp.data import context_target_distribution, make_binary_context_dataset
    from qsp.evaluation import binary_direction_metrics
    from qsp.models.contextual_qnn import ContextualQNN, ContextualQNNConfig, make_qmtl_asset_ids

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
    print("Contextual QNN checks OK")


def check_web_demo_helpers() -> None:
    from qsp.web_demo import SUPPORTED_TICKERS

    assert "AAPL" in SUPPORTED_TICKERS
    assert "BTC-USD" in SUPPORTED_TICKERS
    print("Web demo helper checks OK")


def check_quantum_inspired_models() -> None:
    import numpy as np

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
    for model in [QQBNClassifier(3), QQTNClassifier(3)]:
        history = train_binary_classifier(model, train_x, train_y, epochs=1)
        pred, proba = predict_binary_classifier(model, train_x)
        assert len(history.losses) == 1
        assert pred.shape == train_y.shape
        assert ((0.0 <= proba) & (proba <= 1.0)).all()
    print("Quantum-inspired model checks OK")


def check_custom_qnn_architecture() -> None:
    from qsp.models.custom_qnn import build_custom_qnn_circuit, build_original_custom_qnn_circuit

    original = build_original_custom_qnn_circuit()
    refactored, input_params, weight_params = build_custom_qnn_circuit()
    assert original.num_qubits == refactored.num_qubits == 5
    assert dict(original.count_ops()) == dict(refactored.count_ops())
    assert original.depth() == refactored.depth() == 33
    assert len(input_params) == 5
    assert len(weight_params) == 44
    print("CustomQNN architecture check OK")


def main() -> None:
    check_ast()
    check_contextual_qnn()
    check_web_demo_helpers()
    check_quantum_inspired_models()
    check_custom_qnn_architecture()
    print("All lightweight checks passed")


if __name__ == "__main__":
    main()
