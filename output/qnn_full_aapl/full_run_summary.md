# AAPL Regression Run

This artifact records the full AAPL regression benchmark built around the preserved HQNN-FSP circuit.

## Configuration

- Data source: yfinance AAPL OHLCV data
- Rows after feature engineering: 2072
- Time split: 1650 training samples, 413 test samples
- Feature matching: `SelectKBest(f_regression)` reduced the candidate feature set to 5 inputs
- Selected features: `Open`, `High`, `Low`, `Close`, `SMA5`
- Optimizer: Adam
- Loss: mean squared error on the scaled target
- LSTM epochs: 10
- Standalone CustomQNN epochs: 1
- HybridQNN1 epochs: 1

## Circuit

The original and trainable circuits both use 5 qubits with depth 33. Gate counts remain unchanged: `rz:29`, `ry:20`, `cx:10`, `h:5`, `rx:5`, and `Ppr:5`. The trainable wrapper exposes 5 input parameters and 44 trainable circuit weights.

## Result Table

| Model | RMSE | MAE | Directional Accuracy | Training Time | Inference Time |
|---|---:|---:|---:|---:|---:|
| Naive previous-close baseline | 4.0443 | 2.7724 | 0.0024 | 0.0000s | 0.000004s |
| Classical LSTM | 12.0400 | 9.6950 | 0.4915 | 1.8545s | 0.0177s |
| Standalone CustomQNN | 65.7847 | 57.4034 | 0.4625 | 627.3531s | 1.3130s |
| HybridQNN1 | 52.5749 | 46.0116 | 0.4649 | 593.7860s | 1.2446s |

## Interpretation

This run is useful as a reproducible baseline because it confirms that the preserved circuit can be trained end-to-end inside the project pipeline. The quantum regression models do not yet outperform the naive previous-close baseline on next-close RMSE or MAE. The flat-looking CustomQNN and HybridQNN1 prediction curves are therefore a model-quality issue from limited training, not a plotting failure.
