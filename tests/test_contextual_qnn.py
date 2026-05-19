import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from qsp.data import context_target_distribution, make_binary_context_dataset
from qsp.evaluation import binary_direction_metrics
from qsp.models.contextual_qnn import ContextualQNN, ContextualQNNConfig, make_qmtl_asset_ids


def test_binary_context_dataset_shapes():
    close = [10, 11, 10, 12, 13, 12]
    data = make_binary_context_dataset(close, context_length=2, horizon=1)
    assert data.contexts.shape == (3, 2)
    assert data.targets.shape == (3,)
    assert set(data.targets.tolist()).issubset({0, 1})


def test_context_distribution_probabilities_sum_to_one():
    contexts = np.asarray([[0, 1], [0, 1], [1, 0]])
    targets = np.asarray([1, 0, 1])
    dist = context_target_distribution(contexts, targets)
    assert np.allclose(dist["p_down"] + dist["p_up"], 1.0)


def test_contextual_qnn_probability_range_and_training():
    model = ContextualQNN(ContextualQNNConfig(context_length=2, num_layers=1, seed=42))
    contexts = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.asarray([0, 1, 1, 0])
    before = model.fidelity_loss(contexts, targets)
    losses = model.fit(contexts, targets, epochs=2)
    after = model.fidelity_loss(contexts, targets)
    probs = model.predict_proba(contexts)
    assert len(losses) == 2
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert np.isfinite(before)
    assert np.isfinite(after)


def test_qmtl_two_asset_toy_runs():
    model = ContextualQNN(ContextualQNNConfig(context_length=2, num_assets=2, seed=42))
    contexts = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.asarray([0, 1, 1, 0])
    asset_ids = make_qmtl_asset_ids([2, 2])
    loss = model.spsa_step(contexts, targets, asset_ids)
    assert np.isfinite(loss)
    assert model.predict(contexts, asset_ids).shape == targets.shape


def test_binary_direction_metrics():
    metrics = binary_direction_metrics(np.asarray([0, 1, 1]), np.asarray([0, 0, 1]))
    assert metrics["accuracy"] == 2 / 3
    assert 0.0 <= metrics["f1"] <= 1.0
