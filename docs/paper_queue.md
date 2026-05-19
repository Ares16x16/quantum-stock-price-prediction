# Paper Coverage

The project currently tracks three papers. Each paper has a separate implementation role so that the repository remains clear and easy to explain during the interim presentation.

| Paper | Implementation in this repository |
|---|---|
| *HQNN-FSP: A Hybrid Classical-Quantum Neural Network for Regression-Based Financial Stock Market Prediction* | Reproduced Qiskit circuit, trainable `EstimatorQNN`, AAPL regression pipeline |
| *Contextual Quantum Neural Networks for Stock Price Prediction* | Binary context model, fidelity-style loss, SPSA-style update, two-asset QMTL run |
| *Quantum Inspired Qubit Qutrit Neural Networks for Real Time Financial Forecasting* | ANN, QQBN, and QQTN direction-classification run |

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

The Contextual QNN implementation follows the paper's use of recent return contexts to predict a future return distribution. The current version uses `d=2`, `T=2`, and `tau=1`, which keeps the simulation small enough for local execution. The QMTL version uses shared parameters plus asset-specific parameters for the AAPL/MSFT two-asset case.

The current implementation is a compact reproduction of the paper's modelling idea rather than a full hardware-scale circuit study. It is suitable for the interim stage because it produces real artifacts: context distributions, loss logs, prediction outputs, and result tables.

Included from the paper:

- binary return context modelling;
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

Included from the paper:

- ANN versus qubit-style versus qutrit-style comparison;
- financial direction-classification setup;
- training-time and predictive-metric comparison across model families.

Not included yet:

- the paper's original Indian market data source and exact preprocessing stack;
- a physical qutrit device implementation;
- a one-to-one reproduction of every algorithm block from the paper figures.
