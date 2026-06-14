# Project Progress

This note records the current state of the implementation for the COMP7705 capstone project, *Comparative Analysis of Quantum-Enhanced Neural Networks in Financial Price Prediction*.

## Reproduced HQNN-FSP Circuit

The first implementation track follows *HQNN-FSP: A Hybrid Classical-Quantum Neural Network for Regression-Based Financial Stock Market Prediction* (`2503.15403v1.pdf`). The original circuit was rebuilt in Qiskit and then wrapped as a trainable Qiskit Machine Learning component.

The trainable version keeps the original circuit layout intact. The circuit uses 5 qubits, depth 33, and the same operation counts as the reproduced reference circuit: `rz:29`, `ry:20`, `cx:10`, `h:5`, `rx:5`, and `Ppr:5`. The refactor adds parameter management only: 5 input angles for data encoding and 44 trainable angles for optimization.

The Qiskit `EstimatorQNN` and PyTorch `TorchConnector` path has also been tested with a one-step dummy optimization run. The test confirmed that forward evaluation, loss computation, gradient propagation, and parameter updates all execute correctly.

The training path for this regression model is straightforward: normalized financial features are encoded as circuit inputs, the trainable circuit weights are optimized with Adam, and the loss is mean squared error on the scaled target. For the current full AAPL benchmark, the standalone CustomQNN and HybridQNN1 were both limited to one epoch because simulator-based gradients are slow on CPU.

At the moment, the local environment has CUDA-enabled PyTorch but does not have `qiskit-aer` installed. That means the preserved `EstimatorQNN` path is still CPU-bound. GPU acceleration is available for the PyTorch-only experiments, but not yet for the preserved Qiskit regression circuit.

## AAPL Regression Baseline

The AAPL regression run uses engineered OHLCV features and a time-ordered train/test split. The selected feature set is `Open`, `High`, `Low`, `Close`, and `SMA5`.

| Model | RMSE | MAE | Directional Accuracy | Training Time |
|---|---:|---:|---:|---:|
| Naive previous-close | 4.0443 | 2.7724 | 0.0024 | 0.0000s |
| Classical LSTM | 12.0400 | 9.6950 | 0.4915 | 1.8545s |
| Classical LSTM (GPU refined) | 7.9431 | 6.2807 | 0.4831 | 10.3119s |
| Standalone CustomQNN | 65.7847 | 57.4034 | 0.4625 | 627.3531s |
| HybridQNN1 | 52.5749 | 46.0116 | 0.4649 | 593.7860s |

These figures are an execution baseline rather than a final performance claim. On raw next-close prediction, the previous-close baseline is still very strong. The refined GPU LSTM narrows the gap substantially in RMSE and MAE, but it still does not beat the naive previous-close baseline on this setup. This result supports a shift toward return and direction prediction for the next quantum experiments.

## Contextual QNN Track

The second implementation track follows *Contextual Quantum Neural Networks for Stock Price Prediction*. The current code uses binary return quantization with `d=2`, context length `T=2`, and forecast horizon `tau=1`. It includes contextual sequence generation, a fidelity-style objective, SPSA-style updates, and a two-asset QMTL structure with shared and asset-specific parameters.

Current output files are stored in `output/contextual_qnn` and `output/contextual_qnn_multilevel`.

| Model | Asset | Samples | Directional Accuracy | Training Time |
|---|---|---:|---:|---:|
| ContextualQNN | AAPL | 128 | 0.6923 | 0.9493s |
| ContextualQNN-QMTL | AAPL+MSFT | 320 | 0.5469 | 5.6619s |
| ContextualQNN-QMTL | AAPL+MSFT+GOOGL+AMZN | 384 | 0.5325 | 6.9307s |
| ContextualQNN-d4 | AAPL | 256 | 0.4231 | 31.8448s |

All three contextual rows now use live yfinance data.

The short training time in the binary configuration is expected for this implementation. It is not running a full Qiskit estimator or a shot-based backend. Instead, it uses a very small statevector model implemented in NumPy, a recent-window dataset, and an SPSA-style update. That makes it useful for fast paper-aligned experiments and the interactive demo, but it should not be compared directly to the much heavier Qiskit CustomQNN training time.

The current AAPL result has also been trained more aggressively than the earlier baseline. Increasing the layers and epochs drives the fidelity loss close to zero, but the holdout directional accuracy remains `0.6923`. This suggests that the present bottleneck is no longer simple undertraining; it is the expressive limit of the current small binary-context setup. The two-asset QMTL path, however, did improve after longer training and a slightly larger sample window.

The next local step has now been implemented as a higher-resolution `d=4` variant. This version keeps `T=2` and `tau=1`, but replaces binary labels with four density-based return buckets. In the current AAPL run it uses 6 qubits, 4 layers, 240 SPSA-style updates, and reaches `0.4231` exact-match multiclass accuracy on live yfinance data. The score is lower than the binary model because the task is harder: the model must choose among four return regimes rather than only up versus down.

## Qubit/Qutrit Track

The third implementation track follows *Quantum Inspired Qubit Qutrit Neural Networks for Real Time Financial Forecasting*. The repository now includes three direction-classification models:

| Model | Representation | Directional Accuracy | F1 | Sharpe Ratio | Information Coefficient |
|---|---|---:|---:|---:|---:|
| ANN | Classical feature vector | 0.5238 | 0.6667 | 1.6947 | -0.0465 |
| QQBN | Qubit-inspired two-state feature map | 0.5238 | 0.6154 | -0.0855 | -0.1071 |
| QQTN | Qutrit-inspired three-state feature map | 0.5833 | 0.6789 | 2.9647 | 0.0798 |

These models use normalized technical indicators, an 80/20 time-ordered split, Adam, and binary cross-entropy loss. The QQBN path encodes each feature into a two-state qubit-inspired representation, while QQTN uses a three-state qutrit-inspired representation before the trainable feed-forward layers. The current saved AAPL run uses live yfinance data. A later GPU-backed training refactor improved speed and made device usage explicit, but it did not consistently beat the earlier best QQTN accuracy on AAPL, so the original saved QQTN row remains the stronger benchmark in this repository.

## Bidirectional Direction Track

The next implementation step is now a dedicated bidirectional up/down prediction track. In this track, "bidirectional" means next-day price direction:

```text
Target = 1 if Close[t + 1] > Close[t], otherwise 0
```

This track intentionally reuses the interim `ANN`, `QQBN`, and `QQTN` models instead of making BiLSTM the first model. The direct reference is the qubit/qutrit paper, with the ContextualQNN paper supporting the binary future-return framing and the HQNN-FSP paper motivating stronger technical-indicator feature engineering.

Implementation file:
`src/qsp/experiments/run_bidirectional_direction.py`

Presentation reference:
`docs/bidirectional_direction_prediction.md`

The new experiment benchmarks `AAPL`, `MSFT`, `GOOGL`, and `NVDA`. Compared with the previous `run_quantum_inspired` AAPL-only run, it adds:

- a four-stock benchmark;
- richer lagged features, rolling volatility, momentum, SMA ratios, MACD signal/histogram, and volume features;
- train-only feature scaling;
- explicit time-ordered train/validation/test splits;
- validation-calibrated probability thresholds;
- calibrated `ANN+QQBN+QQTN` ensemble comparison rows;
- majority-class and momentum baselines;
- row-level prediction CSVs, threshold logs, probability plots, and confusion matrices.

Saved outputs are stored in `output/bidirectional_direction`.

Current saved four-stock average:

| Model | Directional Accuracy | Precision | Recall | F1 | Balanced Accuracy | Sharpe Ratio |
|---|---:|---:|---:|---:|---:|---:|
| Majority baseline | 0.5111 | 0.5111 | 1.0000 | 0.6761 | 0.5000 | 0.6380 |
| Momentum baseline | 0.5028 | 0.5136 | 0.5110 | 0.5123 | 0.5015 | 0.0170 |
| ANN | 0.5097 | 0.5106 | 0.9813 | 0.6713 | 0.4991 | 0.5457 |
| QQBN | 0.5111 | 0.5111 | 1.0000 | 0.6761 | 0.5000 | 0.6380 |
| QQTN | 0.5236 | 0.5182 | 0.9866 | 0.6789 | 0.5134 | 0.9538 |
| QQTN balanced threshold | 0.5236 | 0.5222 | 0.8480 | 0.6275 | 0.5166 | 0.2241 |
| ANN+QQBN+QQTN ensemble | 0.5111 | 0.5112 | 0.9974 | 0.6756 | 0.5002 | 0.6160 |
| ANN+QQBN+QQTN balanced ensemble | 0.4972 | 0.3792 | 0.7500 | 0.5034 | 0.5000 | 0.2643 |

The QQTN row is the main result. It gives a modest average directional-accuracy improvement over both baselines while staying within the already-present qubit/qutrit model family. The `QQTN balanced threshold` row uses the same trained QQTN probabilities but selects the threshold by validation balanced accuracy, improving balanced accuracy and reducing one-sided up predictions.

The ensemble rows were added as a lightweight improvement attempt over the reused interim family. They do not beat QQTN on the current four-stock average, so they are retained as comparison evidence rather than promoted as the main result.

## Sequence Hybrid Experiment

To explore a stronger local model without changing the preserved HQNN-FSP circuit, the repository now includes a separate GPU-friendly sequence hybrid path. This experiment uses a bidirectional LSTM encoder with attention pooling and compares:

- a classical sequence baseline;
- a qutrit-inspired hybrid head built on the same temporal encoder.

The current AAPL run is stored in `output/sequence_hybrid_aapl`. It predicts next-day direction, then converts the direction probabilities into an implied next-close curve for visualization.

| Model | RMSE | MAE | Directional Accuracy | Training Time |
|---|---:|---:|---:|---:|
| BiLSTM baseline | 3.7144 | 2.6781 | 0.5250 | 3.5769s |
| BiLSTM-QQTN hybrid | 3.7154 | 2.6731 | 0.5250 | 1.1582s |

This path does not claim an exact paper reproduction. Its role is different: it is a local experimental extension that combines a stronger temporal encoder with a quantum-inspired head so the dashboard can show a richer non-flat prediction curve under GPU-friendly training.

This sequence path is not the first bidirectional deliverable because the interim presentation did not introduce BiLSTM as the main model. It remains available as an optional later extension.

