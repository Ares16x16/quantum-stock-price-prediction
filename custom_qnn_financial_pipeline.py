"""Trainable custom QNN pipeline for the MSc COMP7705 capstone project.

This file keeps the manually reproduced Qiskit circuit architecture intact and
adds reusable training utilities around it:

- architecture verification against the zero-angle original circuit
- EstimatorQNN + TorchConnector construction
- dummy optimizer sanity test
- AAPL/yfinance preprocessing
- naive, LSTM, standalone QNN, and HybridQNN1 training/evaluation helpers

The quantum gate order, entanglement pattern, Ppr blocks, and data-encoding
locations follow the existing ``reproduce_quantum_circuit.py`` implementation.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Literal, Sequence

DEPENDENCY_PACKAGES = [
    "qiskit",
    "qiskit-machine-learning",
    "torch",
    "yfinance",
    "scikit-learn",
    "pandas",
    "numpy",
    "matplotlib",
    "pylatexenc",
]


def _metadata_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in DEPENDENCY_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from qiskit import QuantumCircuit
    from qiskit.circuit import Gate, Parameter, ParameterVector
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_machine_learning.connectors import TorchConnector
    from qiskit_machine_learning.neural_networks import EstimatorQNN
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.preprocessing import MinMaxScaler
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    print("Third-party dependency import failed. Installed versions:")
    for package, version in _metadata_versions().items():
        print(f"  {package}: {version}")
    raise


DEFAULT_NUM_QUBITS = 5
DEFAULT_NUM_LAYERS = 1
DEFAULT_OUTPUT_DIR = Path("output/qnn_pipeline")
PROJECT_FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "RSI",
    "MACD",
    "SMA5",
    "ADX",
    "Return_1",
]
RESULT_COLUMNS = [
    "Model name",
    "Dataset / asset",
    "Number of qubits",
    "VQC layers",
    "Feature set",
    "RMSE",
    "MAE",
    "Directional Accuracy",
    "Training time",
    "Inference time",
    "Parameter count",
    "Circuit depth",
    "Notes",
]


@dataclass(frozen=True)
class CircuitStats:
    num_qubits: int
    gate_counts: dict[str, int]
    depth: int
    num_parameters: int


@dataclass
class FinancialDataset:
    symbol: str
    selected_features: list[str]
    feature_selection_method: str
    feature_scaler: MinMaxScaler
    target_scaler: MinMaxScaler
    train_qnn_x: np.ndarray
    test_qnn_x: np.ndarray
    train_seq_x: np.ndarray
    test_seq_x: np.ndarray
    train_y_scaled: np.ndarray
    test_y_scaled: np.ndarray
    train_y_price: np.ndarray
    test_y_price: np.ndarray
    train_prev_close: np.ndarray
    test_prev_close: np.ndarray
    raw_frame: pd.DataFrame


def limit_dataset_samples(
    dataset: FinancialDataset,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
) -> FinancialDataset:
    """Return a shallow dataset copy limited for fast simulator smoke tests."""

    train_slice = slice(None if not max_train_samples else -max_train_samples, None)
    test_slice = slice(0, None if not max_test_samples else max_test_samples)
    return FinancialDataset(
        symbol=dataset.symbol,
        selected_features=dataset.selected_features,
        feature_selection_method=dataset.feature_selection_method,
        feature_scaler=dataset.feature_scaler,
        target_scaler=dataset.target_scaler,
        train_qnn_x=dataset.train_qnn_x[train_slice],
        test_qnn_x=dataset.test_qnn_x[test_slice],
        train_seq_x=dataset.train_seq_x[train_slice],
        test_seq_x=dataset.test_seq_x[test_slice],
        train_y_scaled=dataset.train_y_scaled[train_slice],
        test_y_scaled=dataset.test_y_scaled[test_slice],
        train_y_price=dataset.train_y_price[train_slice],
        test_y_price=dataset.test_y_price[test_slice],
        train_prev_close=dataset.train_prev_close[train_slice],
        test_prev_close=dataset.test_prev_close[test_slice],
        raw_frame=dataset.raw_frame,
    )


def dependency_versions() -> dict[str, str]:
    """Return key dependency versions for debugging Colab/local issues."""

    return _metadata_versions()


def print_dependency_versions() -> None:
    """Print package versions before Qiskit ML construction/training."""

    print("Dependency versions:")
    for package, version in dependency_versions().items():
        print(f"  {package}: {version}")


def compute_angle_encoding(
    features: Sequence[float],
    f_transform=None,
    g_transform=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Fig. 4-style theta/phi angles from normalized features."""

    x = np.asarray(features, dtype=float)
    f_values = x if f_transform is None else np.asarray(f_transform(x), dtype=float)
    g_values = x if g_transform is None else np.asarray(g_transform(x), dtype=float)
    theta = np.arcsin(np.clip(f_values, -1.0, 1.0))
    phi = np.arccos(np.clip(g_values, -1.0, 1.0))
    return theta, phi


def ppr_gate(width: int) -> Gate:
    """Return the custom Ppr(0) block used in the reproduced circuit.

    The paper/source circuit does not define the Ppr unitary separately. The
    existing project treats it as a labelled identity block so the transcribed
    architecture remains drawable and executable.
    """

    gate = Gate(name="Ppr", num_qubits=width, params=[0])
    gate.definition = QuantumCircuit(width)
    return gate


def build_original_angle_encoding_circuit(
    num_qubits: int = DEFAULT_NUM_QUBITS,
) -> QuantumCircuit:
    """Build the original zero-angle Fig. 4 encoding circuit."""

    circuit = QuantumCircuit(num_qubits, name="original_angle_encoding")
    for qubit in range(num_qubits):
        circuit.ry(0, qubit)
        circuit.rz(0, qubit)
    return circuit


def build_parameterized_angle_encoding_circuit(
    num_qubits: int = DEFAULT_NUM_QUBITS,
    input_mode: Literal["single", "dual"] = "single",
) -> tuple[QuantumCircuit, list[Parameter]]:
    """Build the parameterized Fig. 4 encoding circuit.

    ``single`` mode exposes one input angle per qubit and uses it in both the
    Ry/Rz encoding locations. This keeps the QNN input dimension equal to the
    selected financial feature dimension.

    ``dual`` mode exposes separate theta and phi parameters for Ry/Rz, matching
    the two-angle Fig. 4 formula directly. Use this if preprocessing expands
    each feature into ``arcsin(x)`` and ``arccos(x)`` angles.
    """

    if input_mode not in {"single", "dual"}:
        raise ValueError("input_mode must be either 'single' or 'dual'.")

    size = num_qubits if input_mode == "single" else 2 * num_qubits
    input_vector = ParameterVector("x", size)
    input_params = list(input_vector)

    circuit = QuantumCircuit(num_qubits, name="parameterized_angle_encoding")
    for qubit in range(num_qubits):
        circuit.ry(input_vector[qubit], qubit)
        rz_param = input_vector[qubit] if input_mode == "single" else input_vector[num_qubits + qubit]
        circuit.rz(rz_param, qubit)
    return circuit, input_params


def _append_rotation(
    circuit: QuantumCircuit,
    axis: Literal["rx", "ry", "rz"],
    qubit: int,
    weight_params: list[Parameter] | None,
) -> None:
    """Append a zero or trainable rotation without changing operation order."""

    angle: float | Parameter
    if weight_params is None:
        angle = 0
    else:
        angle = Parameter(f"theta_{len(weight_params):03d}_{axis}_q{qubit}")
        weight_params.append(angle)
    getattr(circuit, axis)(angle, qubit)


def build_original_qnn_regressor_circuit(
    num_qubits: int = DEFAULT_NUM_QUBITS,
) -> QuantumCircuit:
    """Transcribe the original zero-angle Fig. 6 QNN regressor circuit."""

    circuit, _ = _build_qnn_regressor_circuit(num_qubits, parameterized=False)
    circuit.name = "original_qnn_regressor"
    return circuit


def build_parameterized_qnn_regressor_circuit(
    num_qubits: int = DEFAULT_NUM_QUBITS,
) -> tuple[QuantumCircuit, list[Parameter]]:
    """Transcribe Fig. 6 with trainable parameters at existing rotation gates."""

    circuit, weight_params = _build_qnn_regressor_circuit(num_qubits, parameterized=True)
    circuit.name = "parameterized_qnn_regressor"
    return circuit, weight_params


def _build_qnn_regressor_circuit(
    num_qubits: int,
    parameterized: bool,
) -> tuple[QuantumCircuit, list[Parameter]]:
    """Shared Fig. 6 builder used for original and trainable circuits."""

    if num_qubits != DEFAULT_NUM_QUBITS:
        raise ValueError("The manually reproduced Fig. 6 circuit is a five-qubit circuit.")

    weights: list[Parameter] | None = [] if parameterized else None
    circuit = QuantumCircuit(num_qubits)
    rot = lambda axis, qubit: _append_rotation(circuit, axis, qubit, weights)

    # Gate order copied from reproduce_quantum_circuit.py::build_qnn_regressor_circuit.
    circuit.h([0, 1, 2, 3, 4])
    circuit.cx(1, 2)
    circuit.cx(0, 4)
    rot("rz", 2)
    rot("rz", 4)
    circuit.cx(1, 2)
    circuit.cx(0, 4)
    rot("ry", 0)
    rot("ry", 4)
    circuit.cx(1, 3)
    rot("rz", 0)
    rot("rz", 3)
    rot("rz", 4)
    rot("rx", 0)
    circuit.cx(1, 3)
    rot("rz", 0)
    rot("ry", 1)
    circuit.cx(2, 3)
    rot("rz", 1)
    rot("rz", 3)
    circuit.append(ppr_gate(2), [0, 1])
    circuit.cx(2, 3)
    rot("rx", 1)
    rot("ry", 2)
    rot("ry", 3)
    rot("rz", 1)
    rot("rz", 2)
    rot("rz", 3)
    circuit.append(ppr_gate(2), [1, 2])
    rot("rx", 2)
    rot("rz", 2)
    circuit.append(ppr_gate(2), [2, 3])
    rot("ry", 2)
    rot("ry", 2)
    rot("rx", 3)
    rot("rz", 3)
    circuit.append(ppr_gate(2), [3, 4])
    rot("rz", 2)
    rot("rz", 2)
    rot("rx", 4)
    rot("rz", 4)
    circuit.cx(2, 3)
    circuit.append(ppr_gate(5), [0, 1, 2, 3, 4])
    rot("ry", 0)
    rot("ry", 3)
    rot("ry", 4)

    # Continuation line of the folded drawing.
    rot("ry", 0)
    rot("rz", 0)
    rot("rz", 0)
    circuit.cx(0, 1)
    rot("ry", 1)
    rot("ry", 1)
    rot("rz", 1)
    rot("rz", 1)
    rot("ry", 3)
    rot("rz", 3)
    rot("rz", 3)
    rot("ry", 4)
    rot("rz", 4)
    rot("rz", 4)

    return circuit, ([] if weights is None else weights)


def build_original_custom_qnn_circuit(
    num_qubits: int = DEFAULT_NUM_QUBITS,
    num_layers: int = DEFAULT_NUM_LAYERS,
) -> QuantumCircuit:
    """Compose the original zero-angle Fig. 4 + Fig. 6 circuit."""

    if num_layers != 1:
        raise ValueError(
            "The current manually reproduced architecture has one Fig. 6 QNN regressor block. "
            "Repeating it would change the preserved circuit architecture."
        )
    circuit = build_original_angle_encoding_circuit(num_qubits).compose(
        build_original_qnn_regressor_circuit(num_qubits)
    )
    circuit.name = "original_custom_qnn"
    return circuit


def build_custom_qnn_circuit(
    num_qubits: int = DEFAULT_NUM_QUBITS,
    num_layers: int = DEFAULT_NUM_LAYERS,
    input_mode: Literal["single", "dual"] = "single",
) -> tuple[QuantumCircuit, list[Parameter], list[Parameter]]:
    """Return a trainable QNN circuit plus input and weight parameters.

    The function preserves the existing project circuit. ``num_layers`` is kept
    in the signature for project compatibility, but only ``1`` is accepted until
    a deliberate new repeated-layer architecture is approved.
    """

    if num_layers != 1:
        raise ValueError(
            "num_layers must be 1 to preserve the current manually rebuilt circuit. "
            "Using more layers would repeat gates and change the architecture."
        )

    encoding, input_params = build_parameterized_angle_encoding_circuit(
        num_qubits=num_qubits,
        input_mode=input_mode,
    )
    regressor, weight_params = build_parameterized_qnn_regressor_circuit(num_qubits)
    circuit = encoding.compose(regressor)
    circuit.name = "custom_trainable_qnn"
    return circuit, input_params, weight_params


def circuit_stats(circuit: QuantumCircuit) -> CircuitStats:
    """Return qubits, gate counts, depth, and parameter count."""

    return CircuitStats(
        num_qubits=circuit.num_qubits,
        gate_counts=dict(circuit.count_ops()),
        depth=circuit.depth(),
        num_parameters=circuit.num_parameters,
    )


def operation_signature(circuit: QuantumCircuit) -> list[tuple[str, tuple[int, ...], int]]:
    """Represent architecture while ignoring concrete parameter values."""

    signature: list[tuple[str, tuple[int, ...], int]] = []
    for instruction in circuit.data:
        qubits = tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits)
        signature.append((instruction.operation.name, qubits, instruction.operation.num_qubits))
    return signature


def draw_circuit(
    circuit: QuantumCircuit,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    fold: int = 25,
) -> tuple[Path, Path]:
    """Save circuit text and PNG drawings."""

    output_dir.mkdir(parents=True, exist_ok=True)
    text_path = output_dir / f"{circuit.name}.txt"
    png_path = output_dir / f"{circuit.name}.png"
    text_path.write_text(str(circuit.draw(output="text", fold=fold)), encoding="utf-8")
    figure = circuit.draw(output="mpl", fold=fold)
    figure.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return text_path, png_path


def verify_architecture_unchanged(
    num_qubits: int = DEFAULT_NUM_QUBITS,
    num_layers: int = DEFAULT_NUM_LAYERS,
    input_mode: Literal["single", "dual"] = "single",
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    draw: bool = True,
) -> dict[str, object]:
    """Compare original and parameterized circuits and optionally draw both."""

    original = build_original_custom_qnn_circuit(num_qubits, num_layers)
    refactored, input_params, weight_params = build_custom_qnn_circuit(
        num_qubits=num_qubits,
        num_layers=num_layers,
        input_mode=input_mode,
    )

    original_stats = circuit_stats(original)
    refactored_stats = circuit_stats(refactored)
    same_gate_counts = original_stats.gate_counts == refactored_stats.gate_counts
    same_qubits = original_stats.num_qubits == refactored_stats.num_qubits
    same_depth = original_stats.depth == refactored_stats.depth
    same_signature = operation_signature(original) == operation_signature(refactored)
    unchanged = same_qubits and same_gate_counts and same_depth and same_signature

    if draw:
        draw_circuit(original, output_dir)
        draw_circuit(refactored, output_dir)

    report = {
        "original_stats": original_stats,
        "refactored_stats": refactored_stats,
        "same_qubits": same_qubits,
        "same_gate_counts": same_gate_counts,
        "same_depth": same_depth,
        "same_operation_signature": same_signature,
        "architecture_unchanged": unchanged,
        "num_input_params": len(input_params),
        "num_weight_params": len(weight_params),
    }

    print("Architecture verification")
    print(f"  Original stats:   {original_stats}")
    print(f"  Refactored stats: {refactored_stats}")
    print(f"  Input params: {len(input_params)}")
    print(f"  Weight params: {len(weight_params)}")
    print(f"  Architecture unchanged: {unchanged}")
    if not unchanged:
        raise AssertionError("Refactored circuit architecture differs from the original.")
    return report


def z_observable(num_qubits: int, output_qubit: int = 0) -> SparsePauliOp:
    """Create a one-output Pauli-Z observable for regression."""

    if output_qubit < 0 or output_qubit >= num_qubits:
        raise ValueError("output_qubit is out of range.")
    label = ["I"] * num_qubits
    label[num_qubits - 1 - output_qubit] = "Z"
    return SparsePauliOp.from_list([("".join(label), 1.0)])


def create_estimator_qnn(
    circuit: QuantumCircuit,
    input_params: Sequence[Parameter],
    weight_params: Sequence[Parameter],
    output_qubit: int = 0,
    input_gradients: bool = True,
) -> EstimatorQNN:
    """Create the Qiskit Machine Learning EstimatorQNN."""

    print_dependency_versions()
    observable = z_observable(circuit.num_qubits, output_qubit)
    return EstimatorQNN(
        circuit=circuit,
        observables=observable,
        input_params=list(input_params),
        weight_params=list(weight_params),
        input_gradients=input_gradients,
    )


def create_torch_qnn_layer(
    qnn: EstimatorQNN,
    seed: int = 42,
    init_scale: float = 0.1,
) -> TorchConnector:
    """Wrap an EstimatorQNN in a PyTorch-compatible trainable layer."""

    rng = np.random.default_rng(seed)
    initial_weights = init_scale * rng.uniform(-1.0, 1.0, size=qnn.num_weights)
    return TorchConnector(qnn, initial_weights=initial_weights)


def scaled_target_to_expectation(y_scaled: np.ndarray) -> np.ndarray:
    """Map MinMax-scaled target values from [0, 1] to QNN range [-1, 1]."""

    return 2.0 * y_scaled - 1.0


def expectation_to_scaled_target(y_expectation: np.ndarray) -> np.ndarray:
    """Map QNN expectation values from [-1, 1] back to [0, 1]."""

    return (y_expectation + 1.0) / 2.0


def run_dummy_sanity_test(
    num_qubits: int = DEFAULT_NUM_QUBITS,
    num_layers: int = DEFAULT_NUM_LAYERS,
    learning_rate: float = 0.05,
    seed: int = 42,
) -> dict[str, object]:
    """Run one forward/backward/optimizer step on dummy data."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    circuit, input_params, weight_params = build_custom_qnn_circuit(num_qubits, num_layers)
    qnn = create_estimator_qnn(circuit, input_params, weight_params, input_gradients=True)
    qnn_layer = create_torch_qnn_layer(qnn, seed=seed)

    x = torch.rand(1, len(input_params), dtype=torch.float32)
    y = torch.tensor([[0.25]], dtype=torch.float32)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(qnn_layer.parameters(), lr=learning_rate)

    before_weights = qnn_layer.weight.detach().clone()
    pred_before = qnn_layer(x).view_as(y)
    loss_before = loss_fn(pred_before, y)
    optimizer.zero_grad()
    loss_before.backward()
    optimizer.step()
    after_weights = qnn_layer.weight.detach().clone()

    with torch.no_grad():
        pred_after = qnn_layer(x).view_as(y)
        loss_after = loss_fn(pred_after, y)

    weights_changed = not torch.allclose(before_weights, after_weights)
    print("Dummy sanity test")
    print(f"  input shape: {tuple(x.shape)}")
    print(f"  target: {y.item():.6f}")
    print(f"  prediction before: {pred_before.item():.6f}")
    print(f"  prediction after:  {pred_after.item():.6f}")
    print(f"  loss before: {loss_before.item():.6f}")
    print(f"  loss after:  {loss_after.item():.6f}")
    print(f"  trainable weights changed: {weights_changed}")
    if not weights_changed:
        raise AssertionError("QNN optimizer step did not update trainable weights.")

    return {
        "loss_before": float(loss_before.item()),
        "loss_after": float(loss_after.item()),
        "weights_changed": weights_changed,
        "num_weights": qnn.num_weights,
    }


def _flatten_yfinance_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance MultiIndex columns when a single ticker is downloaded."""

    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.copy()
        frame.columns = [col[0] for col in frame.columns]
    return frame


def download_ohlcv(
    symbol: str = "AAPL",
    start: str = "2018-01-01",
    end: str | None = None,
    auto_adjust: bool = True,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Download public OHLCV market data with yfinance."""

    import yfinance as yf

    if cache_dir is None:
        cache_root = Path(os.environ.get("LOCALAPPDATA", Path.home()))
        cache_dir = cache_root / "quantum_stock_price_prediction" / "yfinance_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.cache.set_cache_location(str(cache_dir.resolve()))
    yf.set_tz_cache_location(str(cache_dir.resolve()))
    frame = yf.download(symbol, start=start, end=end, auto_adjust=auto_adjust, progress=False)
    if frame.empty:
        raise ValueError(f"No data downloaded for {symbol}.")
    frame = _flatten_yfinance_columns(frame)
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Downloaded data is missing required columns: {sorted(missing)}")
    return frame


def add_technical_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Add RSI, MACD, SMA5, ADX, and lagged returns without extra TA packages."""

    data = frame.copy()
    close = data["Close"]
    high = data["High"]
    low = data["Low"]

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    data["RSI"] = 100.0 - (100.0 / (1.0 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    data["MACD"] = ema12 - ema26
    data["SMA5"] = close.rolling(window=5, min_periods=5).mean()
    data["Return_1"] = close.pct_change()

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    true_range = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(window=14, min_periods=14).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=data.index).rolling(14).sum() / atr.replace(0.0, np.nan)
    minus_di = 100.0 * pd.Series(minus_dm, index=data.index).rolling(14).sum() / atr.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    data["ADX"] = dx.rolling(window=14, min_periods=14).mean()
    return data


def select_features(
    features_scaled: np.ndarray,
    target_scaled: np.ndarray,
    feature_names: Sequence[str],
    num_features: int,
    method: Literal["selectkbest", "pca"] = "selectkbest",
    fit_rows: slice | np.ndarray | None = None,
) -> tuple[np.ndarray, list[str], object | None, str]:
    """Reduce features to match the number of qubits."""

    if features_scaled.shape[1] <= num_features:
        return features_scaled, list(feature_names), None, "none"

    fit_x = features_scaled if fit_rows is None else features_scaled[fit_rows]
    fit_y = target_scaled if fit_rows is None else target_scaled[fit_rows]

    if method == "selectkbest":
        selector = SelectKBest(score_func=f_regression, k=num_features)
        selector.fit(fit_x, fit_y.ravel())
        selected = selector.transform(features_scaled)
        selected_names = [name for name, keep in zip(feature_names, selector.get_support()) if keep]
        return selected, selected_names, selector, "SelectKBest(f_regression)"

    if method == "pca":
        pca = PCA(n_components=num_features, random_state=42)
        pca.fit(fit_x)
        selected = pca.transform(features_scaled)
        selected_names = [f"PC{i + 1}" for i in range(num_features)]
        return selected, selected_names, pca, "PCA"

    raise ValueError("method must be either 'selectkbest' or 'pca'.")


def create_supervised_sequences(
    selected_features: np.ndarray,
    y_scaled: np.ndarray,
    y_price: np.ndarray,
    prev_close: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create time-window sequences plus single-step QNN inputs."""

    y_scaled_flat = np.asarray(y_scaled, dtype=float).ravel()
    y_price_flat = np.asarray(y_price, dtype=float).ravel()
    prev_close_flat = np.asarray(prev_close, dtype=float).ravel()

    seq_x: list[np.ndarray] = []
    qnn_x: list[np.ndarray] = []
    seq_y_scaled: list[float] = []
    seq_y_price: list[float] = []
    seq_prev_close: list[float] = []

    for end_idx in range(window_size - 1, len(selected_features)):
        start_idx = end_idx - window_size + 1
        seq_x.append(selected_features[start_idx : end_idx + 1])
        qnn_x.append(selected_features[end_idx])
        seq_y_scaled.append(float(y_scaled_flat[end_idx]))
        seq_y_price.append(float(y_price_flat[end_idx]))
        seq_prev_close.append(float(prev_close_flat[end_idx]))

    return (
        np.asarray(seq_x, dtype=np.float32),
        np.asarray(qnn_x, dtype=np.float32),
        np.asarray(seq_y_scaled, dtype=np.float32).reshape(-1, 1),
        np.asarray(seq_y_price, dtype=np.float32).reshape(-1, 1),
        np.asarray(seq_prev_close, dtype=np.float32).reshape(-1, 1),
    )


def prepare_financial_dataset(
    symbol: str = "AAPL",
    start: str = "2018-01-01",
    end: str | None = None,
    num_qubits: int = DEFAULT_NUM_QUBITS,
    window_size: int = 10,
    train_ratio: float = 0.8,
    feature_selection: Literal["selectkbest", "pca"] = "selectkbest",
) -> FinancialDataset:
    """Download AAPL data, engineer features, normalize, select, and split."""

    raw = add_technical_indicators(download_ohlcv(symbol=symbol, start=start, end=end))
    raw["Target_Close"] = raw["Close"].shift(-1)
    raw = raw.dropna(subset=PROJECT_FEATURES + ["Target_Close"]).copy()

    x_raw = raw[PROJECT_FEATURES].astype(float).to_numpy()
    y_price = raw["Target_Close"].astype(float).to_numpy().reshape(-1, 1)
    prev_close = raw["Close"].astype(float).to_numpy().reshape(-1, 1)

    total_sequences = len(raw) - window_size + 1
    split_idx = int(total_sequences * train_ratio)
    if split_idx <= 0 or split_idx >= total_sequences:
        raise ValueError("train_ratio/window_size produced an invalid train/test split.")
    train_last_end_idx = window_size - 1 + split_idx - 1
    train_feature_rows = slice(0, train_last_end_idx + 1)
    train_sample_end_rows = slice(window_size - 1, train_last_end_idx + 1)

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    feature_scaler.fit(x_raw[train_feature_rows])
    target_scaler.fit(y_price[train_sample_end_rows])
    x_scaled_all = feature_scaler.transform(x_raw)
    y_scaled_all = target_scaler.transform(y_price)

    x_selected, selected_names, _, selection_name = select_features(
        x_scaled_all,
        y_scaled_all,
        PROJECT_FEATURES,
        num_qubits,
        method=feature_selection,
        fit_rows=train_sample_end_rows,
    )
    seq_x, qnn_x, seq_y_scaled, seq_y_price, seq_prev_close = create_supervised_sequences(
        x_selected,
        y_scaled_all,
        y_price,
        prev_close,
        window_size=window_size,
    )

    print("Financial dataset prepared")
    print(f"  symbol: {symbol}")
    print(f"  rows after indicators: {len(raw)}")
    print(f"  window_size: {window_size}")
    print(f"  selected features ({selection_name}): {selected_names}")
    print(f"  train samples: {split_idx}")
    print(f"  test samples: {len(seq_x) - split_idx}")

    return FinancialDataset(
        symbol=symbol,
        selected_features=selected_names,
        feature_selection_method=selection_name,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        train_qnn_x=qnn_x[:split_idx],
        test_qnn_x=qnn_x[split_idx:],
        train_seq_x=seq_x[:split_idx],
        test_seq_x=seq_x[split_idx:],
        train_y_scaled=seq_y_scaled[:split_idx],
        test_y_scaled=seq_y_scaled[split_idx:],
        train_y_price=seq_y_price[:split_idx],
        test_y_price=seq_y_price[split_idx:],
        train_prev_close=seq_prev_close[:split_idx],
        test_prev_close=seq_prev_close[split_idx:],
        raw_frame=raw,
    )


def inverse_target(dataset: FinancialDataset, y_scaled: np.ndarray) -> np.ndarray:
    """Invert normalized target close prices back to price units."""

    return dataset.target_scaler.inverse_transform(np.asarray(y_scaled).reshape(-1, 1))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""

    return math.sqrt(mean_squared_error(np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()))


def directional_accuracy(
    y_true_price: np.ndarray,
    y_pred_price: np.ndarray,
    prev_close: np.ndarray,
) -> float:
    """Compare predicted and actual next-step price direction."""

    actual_direction = np.sign(np.asarray(y_true_price).ravel() - np.asarray(prev_close).ravel())
    predicted_direction = np.sign(np.asarray(y_pred_price).ravel() - np.asarray(prev_close).ravel())
    return float(np.mean(actual_direction == predicted_direction))


def regression_metrics(
    y_true_price: np.ndarray,
    y_pred_price: np.ndarray,
    prev_close: np.ndarray,
) -> dict[str, float]:
    """Return RMSE, MAE, and directional accuracy."""

    return {
        "RMSE": rmse(y_true_price, y_pred_price),
        "MAE": float(mean_absolute_error(np.asarray(y_true_price).ravel(), np.asarray(y_pred_price).ravel())),
        "Directional Accuracy": directional_accuracy(y_true_price, y_pred_price, prev_close),
    }


def count_torch_parameters(model: nn.Module) -> int:
    """Count trainable PyTorch parameters."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def evaluate_naive_previous_close(dataset: FinancialDataset) -> dict[str, object]:
    """Naive baseline: next close equals previous close."""

    start_time = time.perf_counter()
    pred = dataset.test_prev_close.copy()
    inference_time = time.perf_counter() - start_time
    metrics = regression_metrics(dataset.test_y_price, pred, dataset.test_prev_close)
    return {
        **metrics,
        "pred_price": pred,
        "Training time": 0.0,
        "Inference time": inference_time,
        "Parameter count": 0,
    }


class LSTMRegressor(nn.Module):
    """Small reproducible PyTorch LSTM baseline for price regression."""

    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :])


def train_lstm_baseline(
    dataset: FinancialDataset,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    hidden_size: int = 32,
    seed: int = 42,
) -> tuple[LSTMRegressor, dict[str, object]]:
    """Train the classical LSTM baseline."""

    torch.manual_seed(seed)
    model = LSTMRegressor(input_size=dataset.train_seq_x.shape[-1], hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(
            torch.tensor(dataset.train_seq_x, dtype=torch.float32),
            torch.tensor(dataset.train_y_scaled, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    losses: list[float] = []
    start_time = time.perf_counter()
    model.train()
    for _ in range(epochs):
        epoch_losses: list[float] = []
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        losses.append(float(np.mean(epoch_losses)))
    training_time = time.perf_counter() - start_time

    model.eval()
    infer_start = time.perf_counter()
    with torch.no_grad():
        pred_scaled = model(torch.tensor(dataset.test_seq_x, dtype=torch.float32)).numpy()
    inference_time = time.perf_counter() - infer_start
    pred_price = inverse_target(dataset, pred_scaled)
    metrics = regression_metrics(dataset.test_y_price, pred_price, dataset.test_prev_close)

    history = {
        **metrics,
        "losses": losses,
        "pred_price": pred_price,
        "Training time": training_time,
        "Inference time": inference_time,
        "Parameter count": count_torch_parameters(model),
    }
    return model, history


def train_standalone_qnn(
    dataset: FinancialDataset,
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 0.02,
    seed: int = 42,
) -> tuple[TorchConnector, dict[str, object]]:
    """Train the custom circuit directly as a one-output QNN regressor."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    circuit, input_params, weight_params = build_custom_qnn_circuit(num_qubits=dataset.train_qnn_x.shape[1])
    qnn = create_estimator_qnn(circuit, input_params, weight_params, input_gradients=True)
    model = create_torch_qnn_layer(qnn, seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    y_train_expectation = scaled_target_to_expectation(dataset.train_y_scaled)
    loader = DataLoader(
        TensorDataset(
            torch.tensor(dataset.train_qnn_x, dtype=torch.float32),
            torch.tensor(y_train_expectation, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    losses: list[float] = []
    start_time = time.perf_counter()
    model.train()
    for _ in range(epochs):
        epoch_losses: list[float] = []
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_x).view_as(batch_y)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        losses.append(float(np.mean(epoch_losses)))
    training_time = time.perf_counter() - start_time

    model.eval()
    infer_start = time.perf_counter()
    with torch.no_grad():
        pred_expectation = model(torch.tensor(dataset.test_qnn_x, dtype=torch.float32)).numpy()
    inference_time = time.perf_counter() - infer_start
    pred_scaled = expectation_to_scaled_target(pred_expectation)
    pred_price = inverse_target(dataset, pred_scaled)
    metrics = regression_metrics(dataset.test_y_price, pred_price, dataset.test_prev_close)

    history = {
        **metrics,
        "losses": losses,
        "pred_price": pred_price,
        "Training time": training_time,
        "Inference time": inference_time,
        "Parameter count": qnn.num_weights,
        "Circuit depth": circuit.depth(),
    }
    return model, history


class HybridQNN1(nn.Module):
    """LSTM feature extractor followed by the preserved custom QNN regressor."""

    def __init__(
        self,
        qnn_layer: TorchConnector,
        qnn_input_dim: int,
        sequence_feature_dim: int,
        hidden_size: int = 32,
    ):
        super().__init__()
        self.lstm = nn.LSTM(sequence_feature_dim, hidden_size, batch_first=True)
        self.to_angles = nn.Sequential(nn.Linear(hidden_size, qnn_input_dim), nn.Tanh())
        self.qnn = qnn_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        angles = math.pi * self.to_angles(output[:, -1, :])
        return self.qnn(angles)


def build_hybrid_qnn1(
    sequence_feature_dim: int,
    qnn_input_dim: int = DEFAULT_NUM_QUBITS,
    hidden_size: int = 32,
    seed: int = 42,
) -> tuple[HybridQNN1, QuantumCircuit]:
    """Create the HybridQNN1 model skeleton."""

    circuit, input_params, weight_params = build_custom_qnn_circuit(num_qubits=qnn_input_dim)
    qnn = create_estimator_qnn(circuit, input_params, weight_params, input_gradients=True)
    qnn_layer = create_torch_qnn_layer(qnn, seed=seed)
    model = HybridQNN1(
        qnn_layer=qnn_layer,
        qnn_input_dim=len(input_params),
        sequence_feature_dim=sequence_feature_dim,
        hidden_size=hidden_size,
    )
    return model, circuit


def train_hybrid_qnn1(
    dataset: FinancialDataset,
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 0.001,
    hidden_size: int = 32,
    seed: int = 42,
) -> tuple[HybridQNN1, dict[str, object]]:
    """Train HybridQNN1: LSTM sequential processing plus QNN regression layer."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    model, circuit = build_hybrid_qnn1(
        sequence_feature_dim=dataset.train_seq_x.shape[-1],
        qnn_input_dim=dataset.train_qnn_x.shape[1],
        hidden_size=hidden_size,
        seed=seed,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    y_train_expectation = scaled_target_to_expectation(dataset.train_y_scaled)
    loader = DataLoader(
        TensorDataset(
            torch.tensor(dataset.train_seq_x, dtype=torch.float32),
            torch.tensor(y_train_expectation, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    losses: list[float] = []
    start_time = time.perf_counter()
    model.train()
    for _ in range(epochs):
        epoch_losses: list[float] = []
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_x).view_as(batch_y)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        losses.append(float(np.mean(epoch_losses)))
    training_time = time.perf_counter() - start_time

    model.eval()
    infer_start = time.perf_counter()
    with torch.no_grad():
        pred_expectation = model(torch.tensor(dataset.test_seq_x, dtype=torch.float32)).numpy()
    inference_time = time.perf_counter() - infer_start
    pred_scaled = expectation_to_scaled_target(pred_expectation)
    pred_price = inverse_target(dataset, pred_scaled)
    metrics = regression_metrics(dataset.test_y_price, pred_price, dataset.test_prev_close)

    history = {
        **metrics,
        "losses": losses,
        "pred_price": pred_price,
        "Training time": training_time,
        "Inference time": inference_time,
        "Parameter count": count_torch_parameters(model),
        "Circuit depth": circuit.depth(),
    }
    return model, history


def make_result_row(
    model_name: str,
    dataset: FinancialDataset,
    metrics: dict[str, object],
    num_qubits: int = DEFAULT_NUM_QUBITS,
    vqc_layers: int = DEFAULT_NUM_LAYERS,
    notes: str = "",
    circuit_depth: int | str = "",
) -> dict[str, object]:
    """Create one row for the final result table."""

    return {
        "Model name": model_name,
        "Dataset / asset": dataset.symbol,
        "Number of qubits": num_qubits,
        "VQC layers": vqc_layers,
        "Feature set": ", ".join(dataset.selected_features),
        "RMSE": metrics.get("RMSE", ""),
        "MAE": metrics.get("MAE", ""),
        "Directional Accuracy": metrics.get("Directional Accuracy", ""),
        "Training time": metrics.get("Training time", ""),
        "Inference time": metrics.get("Inference time", ""),
        "Parameter count": metrics.get("Parameter count", ""),
        "Circuit depth": metrics.get("Circuit depth", circuit_depth),
        "Notes": notes,
    }


def empty_result_table() -> pd.DataFrame:
    """Return a template table with required project columns and rows."""

    rows = [
        {"Model name": "Naive previous-close baseline"},
        {"Model name": "Classical LSTM"},
        {"Model name": "QLSTM teammate reproduced result"},
        {"Model name": "Standalone CustomQNN"},
        {"Model name": "HybridQNN1"},
    ]
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def save_training_log(
    histories: dict[str, dict[str, object]],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Save per-epoch losses to CSV for models that expose a loss curve."""

    output_dir.mkdir(parents=True, exist_ok=True)
    max_len = max((len(history.get("losses", [])) for history in histories.values()), default=0)
    rows: list[dict[str, object]] = []
    for epoch in range(max_len):
        row: dict[str, object] = {"epoch": epoch + 1}
        for name, history in histories.items():
            losses = history.get("losses", [])
            row[name] = losses[epoch] if epoch < len(losses) else np.nan
        rows.append(row)
    path = output_dir / "training_log.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def plot_loss_curve(
    losses: Sequence[float],
    title: str,
    output_path: Path,
) -> Path:
    """Plot and save a training loss curve."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def plot_actual_vs_predicted(
    y_true_price: np.ndarray,
    y_pred_price: np.ndarray,
    title: str,
    output_path: Path,
) -> Path:
    """Plot and save actual vs predicted close prices."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 4))
    plt.plot(np.asarray(y_true_price).ravel(), label="Actual next close")
    plt.plot(np.asarray(y_pred_price).ravel(), label="Predicted next close")
    plt.xlabel("Test sample")
    plt.ylabel("Price")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def run_aapl_smoke_pipeline(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    epochs_lstm: int = 3,
    epochs_qnn: int = 1,
    epochs_hybrid: int = 1,
    max_train_samples: int | None = 32,
    max_test_samples: int | None = 16,
) -> pd.DataFrame:
    """Small end-to-end AAPL run. Keep epochs low for simulator feasibility."""

    verify_architecture_unchanged(output_dir=output_dir, draw=True)
    full_dataset = prepare_financial_dataset(symbol="AAPL", num_qubits=DEFAULT_NUM_QUBITS)
    dataset = limit_dataset_samples(
        full_dataset,
        max_train_samples=max_train_samples,
        max_test_samples=max_test_samples,
    )
    subset_label = "Full dataset" if not max_train_samples and not max_test_samples else "Smoke subset"
    note_prefix = "Full dataset." if subset_label == "Full dataset" else "Smoke subset."
    print(
        f"{subset_label}: "
        f"train={len(dataset.train_y_scaled)} / {len(full_dataset.train_y_scaled)}, "
        f"test={len(dataset.test_y_scaled)} / {len(full_dataset.test_y_scaled)}"
    )
    original_depth = build_original_custom_qnn_circuit().depth()

    naive = evaluate_naive_previous_close(dataset)
    _, lstm_history = train_lstm_baseline(dataset, epochs=epochs_lstm)
    _, qnn_history = train_standalone_qnn(dataset, epochs=epochs_qnn)
    _, hybrid_history = train_hybrid_qnn1(dataset, epochs=epochs_hybrid)

    histories = {
        "Classical LSTM": lstm_history,
        "Standalone CustomQNN": qnn_history,
        "HybridQNN1": hybrid_history,
    }
    save_training_log(histories, output_dir)
    plot_loss_curve(lstm_history["losses"], "Classical LSTM loss", output_dir / "lstm_loss.png")
    plot_loss_curve(qnn_history["losses"], "Standalone CustomQNN loss", output_dir / "custom_qnn_loss.png")
    plot_loss_curve(hybrid_history["losses"], "HybridQNN1 loss", output_dir / "hybrid_qnn1_loss.png")
    plot_actual_vs_predicted(
        dataset.test_y_price,
        lstm_history["pred_price"],
        "AAPL actual vs Classical LSTM predicted",
        output_dir / "lstm_actual_vs_predicted.png",
    )
    plot_actual_vs_predicted(
        dataset.test_y_price,
        qnn_history["pred_price"],
        "AAPL actual vs Standalone CustomQNN predicted",
        output_dir / "custom_qnn_actual_vs_predicted.png",
    )
    plot_actual_vs_predicted(
        dataset.test_y_price,
        hybrid_history["pred_price"],
        "AAPL actual vs HybridQNN1 predicted",
        output_dir / "hybrid_qnn1_actual_vs_predicted.png",
    )

    rows = [
        make_result_row(
            "Naive previous-close baseline",
            dataset,
            naive,
            notes=f"{note_prefix} Predicts next close as previous close.",
            circuit_depth="N/A",
        ),
        make_result_row(
            "Classical LSTM",
            dataset,
            lstm_history,
            notes=f"{note_prefix} Simple PyTorch LSTM baseline.",
            circuit_depth="N/A",
        ),
        {
            **make_result_row(
                "QLSTM teammate reproduced result",
                dataset,
                {},
                notes="Fill from teammate Colab result when available.",
                circuit_depth="TBD",
            ),
            "RMSE": "TBD",
            "MAE": "TBD",
            "Directional Accuracy": "TBD",
        },
        make_result_row(
            "Standalone CustomQNN",
            dataset,
            qnn_history,
            notes=f"{note_prefix} Preserved circuit used directly as QNN regressor.",
            circuit_depth=original_depth,
        ),
        make_result_row(
            "HybridQNN1",
            dataset,
            hybrid_history,
            notes=f"{note_prefix} LSTM feature extractor with preserved QNN final regression layer.",
            circuit_depth=original_depth,
        ),
    ]
    result_table = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    result_table.to_csv(output_dir / "result_table.csv", index=False)
    print(result_table)
    return result_table


def teammate_update_text() -> str:
    """Short explanation suitable for teammates."""

    return (
        "The original circuit architecture was preserved and converted into a trainable QNN "
        "component. The data-encoding rotations are now input parameters, and the existing "
        "variational rotation locations are trainable weights without changing gate order or "
        "entanglement. A dummy forward-pass and optimizer-step test was completed to confirm "
        "that TorchConnector backpropagation updates the QNN weights. The next step is training "
        "on AAPL data and comparing against the naive, LSTM, and QLSTM baselines."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Custom QNN financial prediction pipeline.")
    parser.add_argument("--verify", action="store_true", help="Verify/draw original vs refactored circuits.")
    parser.add_argument("--dummy", action="store_true", help="Run a one-step dummy QNN training sanity test.")
    parser.add_argument("--aapl-smoke", action="store_true", help="Run a small AAPL smoke pipeline.")
    parser.add_argument("--epochs-lstm", type=int, default=3, help="Epochs for the AAPL smoke LSTM.")
    parser.add_argument("--epochs-qnn", type=int, default=1, help="Epochs for the AAPL smoke standalone QNN.")
    parser.add_argument("--epochs-hybrid", type=int, default=1, help="Epochs for the AAPL smoke HybridQNN1.")
    parser.add_argument("--max-train-samples", type=int, default=32, help="Max training samples for smoke runs.")
    parser.add_argument("--max-test-samples", type=int, default=16, help="Max test samples for smoke runs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if args.verify:
        verify_architecture_unchanged(output_dir=args.output_dir, draw=True)
    if args.dummy:
        run_dummy_sanity_test()
    if args.aapl_smoke:
        run_aapl_smoke_pipeline(
            output_dir=args.output_dir,
            epochs_lstm=args.epochs_lstm,
            epochs_qnn=args.epochs_qnn,
            epochs_hybrid=args.epochs_hybrid,
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
        )
    if not (args.verify or args.dummy or args.aapl_smoke):
        parser.print_help()


if __name__ == "__main__":
    main()
