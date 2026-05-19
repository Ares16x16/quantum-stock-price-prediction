"""Run a higher-resolution Contextual QNN experiment with d=4 quantization."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from qsp.data import (
    download_ohlcv_with_fallback,
    make_discrete_context_dataset,
    multilevel_context_target_distribution,
)
from qsp.models.contextual_qnn_multilevel import (
    MultiLevelContextualQNN,
    MultiLevelContextualQNNConfig,
)


def multiclass_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth = np.asarray(y_true, dtype=int).ravel()
    pred = np.asarray(y_pred, dtype=int).ravel()
    return float(np.mean(truth == pred)) if len(truth) else 0.0


def _bucket_label(bin_edges: np.ndarray, level: int) -> str:
    low = 100.0 * float(bin_edges[level])
    high = 100.0 * float(bin_edges[level + 1])
    return f"{low:.2f}% to {high:.2f}%"


def run_multilevel_single_asset(
    symbol: str = "AAPL",
    start: str = "2018-01-01",
    output_dir: Path = Path("output/contextual_qnn_multilevel"),
    context_length: int = 2,
    num_levels: int = 4,
    epochs: int = 240,
    max_samples: int = 256,
    num_layers: int = 4,
    learning_rate: float = 0.05,
    spsa_perturbation: float = 0.01,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, source, note = download_ohlcv_with_fallback(symbol=symbol, start=start)
    data = make_discrete_context_dataset(
        frame["Close"].to_numpy(),
        context_length=context_length,
        horizon=1,
        num_levels=num_levels,
    )
    contexts = data.contexts[-max_samples:]
    targets = data.targets[-max_samples:]
    split = int(len(contexts) * 0.8)
    train_x, test_x = contexts[:split], contexts[split:]
    train_y, test_y = targets[:split], targets[split:]

    config = MultiLevelContextualQNNConfig(
        context_length=context_length,
        horizon=1,
        num_levels=num_levels,
        num_layers=num_layers,
        seed=42,
        learning_rate=learning_rate,
        spsa_perturbation=spsa_perturbation,
    )
    model = MultiLevelContextualQNN(config)
    started = time.perf_counter()
    losses = model.fit(train_x, train_y, epochs=epochs)
    train_time = time.perf_counter() - started

    infer_start = time.perf_counter()
    pred = model.predict(test_x)
    proba = model.predict_proba(test_x)
    inference_time = time.perf_counter() - infer_start
    accuracy = multiclass_accuracy(test_y, pred)

    multilevel_context_target_distribution(contexts, targets, num_levels=num_levels).to_csv(
        output_dir / f"{symbol}_context_distribution_d{num_levels}.csv",
        index=False,
    )
    pd.DataFrame({"epoch": range(1, len(losses) + 1), "fidelity_loss": losses}).to_csv(
        output_dir / f"{symbol}_loss_d{num_levels}.csv",
        index=False,
    )

    bucket_map = {_bucket_label(data.bin_edges, level): level for level in range(num_levels)}
    bucket_labels = {level: label for label, level in bucket_map.items()}
    pred_frame = pd.DataFrame({"target": test_y, "prediction": pred})
    for level in range(num_levels):
        pred_frame[f"p_{level}"] = proba[:, level]
        pred_frame[f"target_label_{level}"] = bucket_labels[level]
    pred_frame.to_csv(output_dir / f"{symbol}_predictions_d{num_levels}.csv", index=False)

    notes = (
        f"{epochs} epochs, {len(contexts)} context-target samples, d={num_levels}, "
        f"T={context_length}, num_layers={num_layers}, learning_rate={learning_rate}, "
        f"spsa_perturbation={spsa_perturbation}. Data source: {source}."
    )
    if note:
        notes += f" {note}"
    result = pd.DataFrame(
        [
            {
                "Model name": f"ContextualQNN-d{num_levels}",
                "Dataset / asset": symbol,
                "Number of qubits": config.num_qubits,
                "VQC layers": num_layers,
                "Feature set": f"Density-based return buckets, d={num_levels}, T={context_length}, tau=1",
                "RMSE": "N/A",
                "MAE": "N/A",
                "Directional Accuracy": accuracy,
                "Training time": train_time,
                "Inference time": inference_time,
                "Parameter count": model.num_parameters,
                "Circuit depth": model.circuit_depth_estimate,
                "Notes": notes,
            }
        ]
    )
    result.to_csv(output_dir / f"{symbol}_result_table_d{num_levels}.csv", index=False)

    pd.DataFrame(
        {
            "bucket": list(range(num_levels)),
            "label": [_bucket_label(data.bin_edges, level) for level in range(num_levels)],
        }
    ).to_csv(output_dir / f"{symbol}_bucket_labels_d{num_levels}.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run higher-resolution Contextual QNN reproduction scaffold.")
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--output-dir", type=Path, default=Path("output/contextual_qnn_multilevel"))
    parser.add_argument("--context-length", type=int, default=2)
    parser.add_argument("--num-levels", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--spsa-perturbation", type=float, default=0.01)
    args = parser.parse_args()

    print(
        run_multilevel_single_asset(
            symbol=args.symbol,
            start=args.start,
            output_dir=args.output_dir,
            context_length=args.context_length,
            num_levels=args.num_levels,
            epochs=args.epochs,
            max_samples=args.max_samples,
            num_layers=args.num_layers,
            learning_rate=args.learning_rate,
            spsa_perturbation=args.spsa_perturbation,
        )
    )


if __name__ == "__main__":
    main()
