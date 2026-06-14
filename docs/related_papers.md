# Related Papers

This project currently centers on three papers already collected in the repository:

1. *HQNN-FSP: A Hybrid Classical-Quantum Neural Network for Regression-Based Financial Stock Market Prediction*
2. *Contextual Quantum Neural Networks for Stock Price Prediction*
3. *Quantum inspired qubit qutrit neural networks for real time financial forecasting*

To broaden the literature review without drifting away from the proposal, the following paper is worth keeping in view:

## BLS-QLSTM: a novel hybrid quantum neural network for stock index forecasting

This paper proposes a hybrid model that combines a broad learning system with a quantum LSTM for stock-index forecasting. It is relevant because it sits between the two themes already present in this repository: classical temporal modelling and quantum-enhanced sequence learning. It does not replace the preserved HQNN-FSP circuit, but it is useful as a reference when discussing why stronger classical front-ends can improve the quality of the final prediction curve.

Reference:
- https://www.nature.com/articles/s41599-025-05348-z

## Why it matters for this repository

The current codebase now separates the work into three primary paper tracks plus one later sequence-learning extension:

- preserved Qiskit circuit reproduction for HQNN-FSP;
- contextual-distribution modelling for the Contextual QNN paper;
- quantum-inspired qubit/qutrit models, including the four-stock direction benchmark;
- a separate GPU-friendly sequence hybrid for later experimentation.

This separation keeps the contributions academically legible. The preserved circuit work remains faithful to the original architecture, while the stronger sequence hybrid serves as an experimental extension rather than a claim of exact reproduction.
