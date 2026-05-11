# AAPL Full-Dataset Run Summary

Run date: 2026-05-11

## Run Configuration

- Dataset: AAPL from yfinance.
- Engineered rows after indicators: 2072.
- Time-series split: 1650 train samples, 413 test samples.
- Feature matching: `SelectKBest(f_regression)` to 5 features.
- Selected features: `Open`, `High`, `Low`, `Close`, `SMA5`.
- LSTM epochs: 10.
- Standalone CustomQNN epochs: 1.
- HybridQNN1 epochs: 1.
- Qiskit: 2.4.1.
- qiskit-machine-learning: 0.9.0.
- Torch: 2.11.0+cpu.

## Circuit Verification

- Original circuit: 5 qubits, depth 33, gate counts `rz:29`, `ry:20`, `cx:10`, `h:5`, `rx:5`, `Ppr:5`.
- Refactored circuit: 5 qubits, depth 33, same gate counts.
- Parameter split: 5 input parameters, 44 trainable QNN weight parameters.
- Result: architecture unchanged.

## Results

| Model | RMSE | MAE | Directional Accuracy | Training Time | Inference Time | Parameter Count | Circuit Depth |
|---|---:|---:|---:|---:|---:|---:|---:|
| Naive previous-close baseline | 4.0443 | 2.7724 | 0.0024 | 0.0000s | 0.000004s | 0 | N/A |
| Classical LSTM | 12.0400 | 9.6950 | 0.4915 | 1.8545s | 0.0177s | 5025 | N/A |
| Standalone CustomQNN | 65.7847 | 57.4034 | 0.4625 | 627.3531s | 1.3130s | 44 | 33 |
| HybridQNN1 | 52.5749 | 46.0116 | 0.4649 | 593.7860s | 1.2446s | 5201 | 33 |

## Interpretation

This is a full-dataset execution check, not a final model-quality result. The QNN models ran end-to-end on the full AAPL split, but each quantum model trained for only one epoch because CPU simulator gradients are expensive. The naive previous-close baseline remains the strongest RMSE/MAE baseline in this run, so future model tuning must beat RMSE `4.0443` and MAE `2.7724` before it is worth claiming improvement.

The LSTM loss converged on the normalized training objective, but test RMSE is still worse than naive. This suggests the next work should focus on target definition and baseline tuning before spending many hours on QNN epochs.

## Next Move

1. Use next-day return prediction as the main regression target for QNN/HQC, while keeping next-close as a reported baseline task.
2. Train the classical LSTM baseline more carefully first: learning-rate sweep, hidden size sweep, early stopping, and compare against naive.
3. Run Standalone CustomQNN on smaller controlled subsets with 5, 10, and 20 epochs to see whether the QNN loss can actually improve before launching long full-dataset runs.
4. Run HybridQNN1 only after the standalone QNN has a useful training curve.
5. Add teammate QLSTM metrics into `result_table.csv` once available.
