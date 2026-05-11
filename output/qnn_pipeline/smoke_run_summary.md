# AAPL QNN Smoke Run Summary

Run date: 2026-05-11

## Circuit Verification

- Original circuit: 5 qubits, depth 33, gate counts `rz:29`, `ry:20`, `cx:10`, `h:5`, `rx:5`, `Ppr:5`.
- Refactored trainable circuit: 5 qubits, depth 33, same gate counts.
- Parameter split: 5 input parameters, 44 trainable weight parameters.
- Result: architecture unchanged.

## Dummy QNN Sanity Test

- Dependency versions: Qiskit 2.4.1, qiskit-machine-learning 0.9.0, Torch 2.11.0+cpu.
- Input shape: `(1, 5)`.
- Loss before optimizer step: `0.020231`.
- Loss after optimizer step: `0.000062`.
- Trainable weights changed: `True`.
- Result: QNN forward pass, MSE loss, backpropagation, and optimizer update work.

## AAPL Smoke Dataset

- Full engineered dataset rows after indicators: 2072.
- Time-series split: 1650 train samples, 413 test samples.
- Smoke subset: 32 train samples, 16 test samples.
- Feature selection: `SelectKBest(f_regression)`.
- Selected features: `Open`, `High`, `Low`, `Close`, `SMA5`.

## Smoke Metrics

| Model | RMSE | MAE | Directional Accuracy | Training Time | Inference Time | Parameter Count | Circuit Depth |
|---|---:|---:|---:|---:|---:|---:|---:|
| Naive previous-close baseline | 3.7616 | 2.7550 | 0.0000 | 0.0000s | 0.000004s | 0 | N/A |
| Classical LSTM | 162.2995 | 162.2418 | 0.4375 | 1.7040s | 0.0795s | 5025 | N/A |
| Standalone CustomQNN | 35.2043 | 34.8863 | 0.4375 | 12.4483s | 0.0564s | 44 | 33 |
| HybridQNN1 | 68.7254 | 68.5889 | 0.4375 | 12.6211s | 0.0531s | 5201 | 33 |

## Interpretation

This smoke run validates code execution and trainability, not final model quality. The neural models are intentionally undertrained with very small sample counts and low epochs. The naive baseline should remain the first comparison target for full-scale experiments.

## Next Move

1. Run a controlled LSTM-only baseline on the full AAPL split first.
2. Increase QNN/HybridQNN1 training gradually: 64, 128, then 256 training samples before attempting the full split.
3. Use early stopping and keep QNN epochs low until runtime is measured.
4. Add the teammate QLSTM result row once their Colab metrics are available.
5. After AAPL is stable, repeat on one crypto asset and one additional stock.
