"""Run a lightweight Contextual QNN reproduction experiment."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from qsp.data import context_target_distribution, download_ohlcv_with_fallback, make_binary_context_dataset
from qsp.evaluation import binary_direction_metrics
from qsp.models.contextual_qnn import ContextualQNN, ContextualQNNConfig, make_qmtl_asset_ids


def _asset_contexts(symbol: str, start: str, context_length: int, horizon: int):
    frame, source, note = download_ohlcv_with_fallback(symbol=symbol, start=start)
    dataset = make_binary_context_dataset(frame["Close"].to_numpy(), context_length, horizon)
    return dataset, source, note


def _paper_result_row(
    model_name: str,
    asset: str,
    context_length: int,
    epochs: int,
    samples: int,
    metrics: dict[str, float],
    train_time: float,
    inference_time: float | str,
    parameter_count: int,
    depth: int,
    source: str,
    note: str,
    config_note: str,
) -> dict[str, object]:
    source_note = f"Data source: {source}."
    if note:
        source_note += f" {note}"
    return {
        "Model name": model_name,
        "Dataset / asset": asset,
        "Number of qubits": context_length + 1,
        "VQC layers": 1,
        "Feature set": f"Binary return context, d=2, T={context_length}, tau=1",
        "RMSE": "N/A",
        "MAE": "N/A",
        "Directional Accuracy": metrics["accuracy"],
        "Training time": train_time,
        "Inference time": inference_time,
        "Parameter count": parameter_count,
        "Circuit depth": depth,
        "Notes": f"{epochs} epochs, {samples} context-target samples. {config_note}. {source_note}",
    }


def run_single_asset(
    symbol: str = "AAPL",
    start: str = "2018-01-01",
    output_dir: Path = Path("output/contextual_qnn"),
    context_length: int = 2,
    epochs: int = 100,
    max_samples: int = 128,
    num_layers: int = 4,
    learning_rate: float = 0.3,
    spsa_perturbation: float = 0.01,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    data, source, source_note = _asset_contexts(symbol, start, context_length, 1)
    contexts = data.contexts[-max_samples:]
    targets = data.targets[-max_samples:]
    split = int(len(contexts) * 0.8)
    train_x, test_x = contexts[:split], contexts[split:]
    train_y, test_y = targets[:split], targets[split:]

    config = ContextualQNNConfig(
        context_length=context_length,
        horizon=1,
        num_layers=num_layers,
        seed=42,
        learning_rate=learning_rate,
        spsa_perturbation=spsa_perturbation,
    )
    model = ContextualQNN(config)
    started = time.perf_counter()
    losses = model.fit(train_x, train_y, epochs=epochs)
    train_time = time.perf_counter() - started

    infer_start = time.perf_counter()
    pred = model.predict(test_x)
    proba = model.predict_proba(test_x)
    inference_time = time.perf_counter() - infer_start
    metrics = binary_direction_metrics(test_y, pred)

    context_target_distribution(contexts, targets).to_csv(output_dir / f"{symbol}_context_distribution.csv", index=False)
    pd.DataFrame({"epoch": range(1, len(losses) + 1), "fidelity_loss": losses}).to_csv(
        output_dir / f"{symbol}_loss.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "target": test_y,
            "prediction": pred,
            "probability_up": proba,
        }
    ).to_csv(output_dir / f"{symbol}_predictions.csv", index=False)
    config_note = (
        f"num_layers={num_layers}, learning_rate={learning_rate}, "
        f"spsa_perturbation={spsa_perturbation}"
    )
    result = pd.DataFrame(
        [
            _paper_result_row(
                "ContextualQNN",
                symbol,
                context_length,
                epochs,
                len(contexts),
                metrics,
                train_time,
                inference_time,
                model.num_parameters,
                model.circuit_depth_estimate,
                source,
                source_note,
                config_note,
            )
        ]
    )
    result.to_csv(output_dir / f"{symbol}_result_table.csv", index=False)
    return result


def run_two_asset_qmtl(
    symbols: list[str],
    start: str,
    output_dir: Path,
    context_length: int,
    epochs: int,
    max_samples_per_asset: int,
    num_layers: int = 3,
    learning_rate: float = 0.1,
    spsa_perturbation: float = 0.01,
) -> pd.DataFrame:
    if len(symbols) != 2:
        raise ValueError("The first QMTL scaffold expects exactly two assets.")
    output_dir.mkdir(parents=True, exist_ok=True)

    contexts_list = []
    targets_list = []
    sources = []
    notes = []
    for symbol in symbols:
        data, source, note = _asset_contexts(symbol, start, context_length, 1)
        contexts_list.append(data.contexts[-max_samples_per_asset:])
        targets_list.append(data.targets[-max_samples_per_asset:])
        sources.append(source)
        if note:
            notes.append(f"{symbol}: {note}")

    contexts = np.vstack(contexts_list)
    targets = np.concatenate(targets_list)
    asset_ids = make_qmtl_asset_ids([len(x) for x in contexts_list])

    split = int(len(contexts) * 0.8)
    model = ContextualQNN(
        ContextualQNNConfig(
            context_length=context_length,
            num_layers=num_layers,
            num_assets=2,
            seed=42,
            learning_rate=learning_rate,
            spsa_perturbation=spsa_perturbation,
        )
    )
    started = time.perf_counter()
    losses = model.fit(contexts[:split], targets[:split], asset_ids[:split], epochs=epochs)
    train_time = time.perf_counter() - started
    infer_start = time.perf_counter()
    pred = model.predict(contexts[split:], asset_ids[split:])
    inference_time = time.perf_counter() - infer_start
    metrics = binary_direction_metrics(targets[split:], pred)

    pd.DataFrame({"epoch": range(1, len(losses) + 1), "fidelity_loss": losses}).to_csv(
        output_dir / "qmtl_two_asset_loss.csv",
        index=False,
    )
    config_note = (
        f"num_layers={num_layers}, learning_rate={learning_rate}, "
        f"spsa_perturbation={spsa_perturbation}"
    )
    result = pd.DataFrame(
        [
            _paper_result_row(
                "ContextualQNN-QMTL",
                "+".join(symbols),
                context_length,
                epochs,
                len(contexts),
                metrics,
                train_time,
                inference_time,
                model.num_parameters,
                model.circuit_depth_estimate,
                "+".join(sorted(set(sources))),
                " ".join(notes),
                config_note,
            )
        ]
    )
    result.to_csv(output_dir / "qmtl_two_asset_result_table.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Contextual QNN reproduction scaffold.")
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--output-dir", type=Path, default=Path("output/contextual_qnn"))
    parser.add_argument("--context-length", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.3)
    parser.add_argument("--spsa-perturbation", type=float, default=0.01)
    parser.add_argument("--qmtl", action="store_true")
    parser.add_argument("--qmtl-symbols", nargs=2, default=["AAPL", "MSFT"])
    args = parser.parse_args()

    if args.qmtl:
        print(
            run_two_asset_qmtl(
                args.qmtl_symbols,
                args.start,
                args.output_dir,
                args.context_length,
                args.epochs,
                args.max_samples,
                num_layers=args.num_layers,
                learning_rate=args.learning_rate,
                spsa_perturbation=args.spsa_perturbation,
            )
        )
    else:
        print(
            run_single_asset(
                args.symbol,
                args.start,
                args.output_dir,
                args.context_length,
                args.epochs,
                args.max_samples,
                num_layers=args.num_layers,
                learning_rate=args.learning_rate,
                spsa_perturbation=args.spsa_perturbation,
            )
        )


if __name__ == "__main__":
    main()
