# Paper Coverage

The project currently has three primary paper tracks plus one separate fourth-paper extension. Each paper has a separate implementation role so that the repository remains clear and easy to explain during presentation.

| Paper | Implementation in this repository |
|---|---|
| *HQNN-FSP: A Hybrid Classical-Quantum Neural Network for Regression-Based Financial Stock Market Prediction* | Reproduced Qiskit circuit, trainable `EstimatorQNN`, AAPL regression pipeline |
| *Contextual Quantum Neural Networks for Stock Price Prediction* | Binary context model, higher-resolution `d=4` context model, fidelity-style loss, SPSA-style update, multi-asset QMTL run |
| *Quantum Inspired Qubit Qutrit Neural Networks for Real Time Financial Forecasting* | ANN, QQBN, QQTN direction-classification run, four-stock bidirectional direction track |
| *BLS-QLSTM: a novel hybrid quantum neural network for stock index forecasting* | Separate optional BiLSTM/sequence-learning extension, not the first bidirectional model |

## HQNN-FSP

This paper is the source for the custom QNN circuit used in the regression pipeline. The circuit is preserved as the canonical Qiskit implementation in this repository. The training wrapper separates financial feature inputs from trainable circuit weights while keeping the reproduced gate sequence unchanged.

Included from the paper:

- reproduced encoding-plus-ansatz circuit structure in Qiskit;
- trainable `EstimatorQNN` wrapper around the preserved circuit;
- standalone QNN regression path and HybridQNN1 regression path;
- AAPL benchmark outputs with RMSE, MAE, directional accuracy, training time, inference time, parameter count, and circuit depth.

Not included yet:

- a stronger full benchmark that beats the naive previous-close baseline;
- long multi-epoch Qiskit training on the full AAPL split;
- hardware execution beyond simulator-friendly runs.

## Contextual Quantum Neural Networks

The Contextual QNN implementation follows the paper's use of recent return contexts to predict a future return distribution. The local binary version uses `d=2`, `T=2`, and `tau=1`, which keeps the simulation small enough for local execution. A second local variant now uses density-based `d=4` buckets with the same `T=2, tau=1` structure, so the next-step symbol is harder and the qubit count rises from 3 to 6. The QMTL version uses shared parameters plus asset-specific parameters and now runs both a two-asset and a four-asset live-data experiment.

The current implementation is a compact reproduction of the paper's modelling idea rather than a full hardware-scale circuit study. It is suitable for the interim stage because it produces real artifacts: context distributions, loss logs, prediction outputs, and result tables.

Included from the paper:

- binary return context modelling;
- higher-resolution return-bucket modelling with `d=4` in a local simulator setting;
- conditional next-step distribution prediction;
- fidelity-style distribution objective;
- SPSA-style optimization for a lightweight simulator path;
- share-and-specify style two-asset QMTL structure.

Not included yet:

- the paper's full quantum batch gradient update (QBGU) workflow;
- larger portfolio experiments such as the 4-asset and 8-asset studies;
- the paper's noise study and richer quantization settings beyond the current small setup.

## Qubit/Qutrit Neural Networks

The qubit/qutrit paper compares a classical ANN, a qubit-based neural network, and a qutrit-based neural network. The repository implements the same comparison structure with a reproducible PyTorch pipeline:

- ANN uses the normalized financial feature vector directly.
- QQBN maps each feature into a two-state qubit-inspired representation.
- QQTN maps each feature into a three-state qutrit-inspired representation.

Qutrits are not native Qiskit qubits, so the QQTN path is implemented as a qutrit-inspired simulator rather than a physical qutrit circuit. The result table records accuracy, precision, recall, F1, Sharpe ratio, information coefficient, training time, inference time, and parameter count.

The bidirectional direction implementation reuses this same ANN/QQBN/QQTN model family. It extends the earlier AAPL-only run into a four-stock up/down benchmark for `AAPL`, `MSFT`, `GOOGL`, and `NVDA`, adds richer lagged technical features, uses train-only scaling, calibrates probability thresholds on a validation split, and reports majority and momentum baselines. The implementation reference document is `docs/bidirectional_direction_prediction.md`.

Included from the paper:

- ANN versus qubit-style versus qutrit-style comparison;
- financial direction-classification setup;
- training-time and predictive-metric comparison across model families.

Experimental extension in this repository:

- the four-stock bidirectional direction track using the reused QQTN model as the main model;
- calibrated `ANN+QQBN+QQTN` ensemble comparison rows, retained as diagnostic evidence because they do not beat QQTN on the current four-stock average.

Not included yet:

- the paper's original Indian market data source and exact preprocessing stack;
- a physical qutrit device implementation;
- a one-to-one reproduction of every algorithm block from the paper figures.

## BLS-QLSTM / BiLSTM Extension

The fourth reference is kept as a later sequence-learning extension rather than mixed into the first bidirectional direction result. The repository implements a local, GPU-friendly comparison between:

- a bidirectional LSTM baseline with attention pooling;
- a BiLSTM encoder followed by a qutrit-inspired QQTN-style head.

This extension is useful for discussing what can be attempted next when the project moves beyond the interim-aligned ANN/QQBN/QQTN family. It is not used as the main model because BiLSTM was not part of the interim presentation scope.

Included from the reference direction:

- stronger temporal sequence modelling;
- hybrid classical/quantum-inspired sequence-head comparison;
- saved AAPL result tables, training curves, and implied next-close plots.

Not included yet:

- a full reproduction of the BLS-QLSTM architecture;
- multi-index or multi-market experiments;
- use as the headline result.
