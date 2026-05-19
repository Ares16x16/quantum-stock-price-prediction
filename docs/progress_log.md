# Project Progress

This note records the current state of the implementation for the COMP7705 capstone project, *Comparative Analysis of Quantum-Enhanced Neural Networks in Financial Price Prediction*.

## Reproduced HQNN-FSP Circuit

The first implementation track follows *HQNN-FSP: A Hybrid Classical-Quantum Neural Network for Regression-Based Financial Stock Market Prediction* (`2503.15403v1.pdf`). The original circuit was rebuilt in Qiskit and then wrapped as a trainable Qiskit Machine Learning component.

The trainable version keeps the original circuit layout intact. The circuit uses 5 qubits, depth 33, and the same operation counts as the reproduced reference circuit: `rz:29`, `ry:20`, `cx:10`, `h:5`, `rx:5`, and `Ppr:5`. The refactor adds parameter management only: 5 input angles for data encoding and 44 trainable angles for optimization.

The Qiskit `EstimatorQNN` and PyTorch `TorchConnector` path has also been tested with a one-step dummy optimization run. The test confirmed that forward evaluation, loss computation, gradient propagation, and parameter updates all execute correctly.

The training path for this regression model is straightforward: normalized financial features are encoded as circuit inputs, the trainable circuit weights are optimized with Adam, and the loss is mean squared error on the scaled target. For the current full AAPL benchmark, the standalone CustomQNN and HybridQNN1 were both limited to one epoch because simulator-based gradients are slow on CPU.

## AAPL Regression Baseline

The AAPL regression run uses engineered OHLCV features and a time-ordered train/test split. The selected feature set is `Open`, `High`, `Low`, `Close`, and `SMA5`.

| Model | RMSE | MAE | Directional Accuracy | Training Time |
|---|---:|---:|---:|---:|
| Naive previous-close | 4.0443 | 2.7724 | 0.0024 | 0.0000s |
| Classical LSTM | 12.0400 | 9.6950 | 0.4915 | 1.8545s |
| Standalone CustomQNN | 65.7847 | 57.4034 | 0.4625 | 627.3531s |
| HybridQNN1 | 52.5749 | 46.0116 | 0.4649 | 593.7860s |

These figures are an execution baseline rather than a final performance claim. On raw next-close prediction, the previous-close baseline is still very strong. This result supports a shift toward return and direction prediction for the next quantum experiments.

## Contextual QNN Track

The second implementation track follows *Contextual Quantum Neural Networks for Stock Price Prediction*. The current code uses binary return quantization with `d=2`, context length `T=2`, and forecast horizon `tau=1`. It includes contextual sequence generation, a fidelity-style objective, SPSA-style updates, and a two-asset QMTL structure with shared and asset-specific parameters.

Current output files are stored in `output/contextual_qnn`.

| Model | Asset | Samples | Directional Accuracy | Training Time |
|---|---|---:|---:|---:|
| ContextualQNN | AAPL | 128 | 0.6923 | 0.9493s |
| ContextualQNN-QMTL | AAPL+MSFT | 320 | 0.5469 | 5.6619s |

Both rows now use live yfinance data.

The short training time is expected for this implementation. It is not running a full Qiskit estimator or a shot-based backend. Instead, it uses a very small 3-qubit statevector model implemented in NumPy, a recent-window dataset, and an SPSA-style update. That makes it useful for fast paper-aligned experiments and the interactive demo, but it should not be compared directly to the much heavier Qiskit CustomQNN training time.

The current AAPL result has also been trained more aggressively than the earlier baseline. Increasing the layers and epochs drives the fidelity loss close to zero, but the holdout directional accuracy remains `0.6923`. This suggests that the present bottleneck is no longer simple undertraining; it is the expressive limit of the current small binary-context setup. The two-asset QMTL path, however, did improve after longer training and a slightly larger sample window.

## Qubit/Qutrit Track

The third implementation track follows *Quantum Inspired Qubit Qutrit Neural Networks for Real Time Financial Forecasting*. The repository now includes three direction-classification models:

| Model | Representation | Directional Accuracy | F1 | Sharpe Ratio | Information Coefficient |
|---|---|---:|---:|---:|---:|
| ANN | Classical feature vector | 0.5238 | 0.6667 | 1.6947 | -0.0465 |
| QQBN | Qubit-inspired two-state feature map | 0.5238 | 0.6154 | -0.0855 | -0.1071 |
| QQTN | Qutrit-inspired three-state feature map | 0.5833 | 0.6789 | 2.9647 | 0.0798 |

These models use normalized technical indicators, an 80/20 time-ordered split, Adam, and binary cross-entropy loss. The QQBN path encodes each feature into a two-state qubit-inspired representation, while QQTN uses a three-state qutrit-inspired representation before the trainable feed-forward layers. The current saved AAPL run uses live yfinance data.

## Demo Status

The Streamlit dashboard loads saved artifacts, renders circuit diagrams, compares model result tables, and runs a lightweight ContextualQNN direction demo after a ticker is selected. It does not train the expensive CustomQNN inside the web interface.
