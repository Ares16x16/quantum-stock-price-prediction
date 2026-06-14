# Quantum Stock Price Prediction

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-2.4+-6929C4?logo=qiskit&logoColor=white)](https://qiskit.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Project-Interim%20Implementation-0A7E07)](#current-status)

Implementation workspace for the MSc COMP7705 capstone project:
**Comparative Analysis of Quantum-Enhanced Neural Networks in Financial Price Prediction**

This repository reproduces and compares several quantum and quantum-inspired approaches for financial forecasting, with a focus on reproducible local execution and mentor-facing artifacts.

## Overview

The current repository covers three paper tracks:

1. **HQNN-FSP: A Hybrid Classical-Quantum Neural Network for Regression-Based Financial Stock Market Prediction**
2. **Contextual Quantum Neural Networks for Stock Price Prediction**
3. **Quantum Inspired Qubit Qutrit Neural Networks for Real Time Financial Forecasting**

It also includes:

- a preserved Qiskit circuit reproduction;
- a trainable `EstimatorQNN` pipeline;
- classical LSTM and naive baselines;
- the four-stock bidirectional up/down prediction track using reused ANN, QQBN, and QQTN models;
- a separate GPU-friendly sequence hybrid experiment;
- saved experiment artifacts under `output/`;
- a Streamlit dashboard for result review and lightweight interaction.

## Current Status

- The HQNN-FSP circuit has been reproduced in Qiskit and wrapped into a trainable `EstimatorQNN` without changing the gate sequence.
- The ContextualQNN path runs locally on live `yfinance` data and supports single-asset and two-asset QMTL experiments.
- The qubit/qutrit paper path runs locally as ANN, QQBN, and QQTN direction-classification experiments.
- The bidirectional direction track now benchmarks AAPL, MSFT, GOOGL, and NVDA using the reused ANN/QQBN/QQTN model family, richer lagged features, train-only scaling, and validation-calibrated thresholds.
- The repository also includes a separate optional `BiLSTM-QQTN` experiment for stronger GPU-friendly local hybrid testing.
- The Streamlit dashboard is organized by paper/model track, with the bidirectional direction work and the optional BiLSTM extension on separate tabs.
- The full CustomQNN and HybridQNN1 regression paths still underperform the naive previous-close baseline and their prediction curves remain near-flat under limited simulator training.

## Repository Layout

```text
.
|-- app/
|   `-- streamlit_app.py
|-- docs/
|   |-- local_testing_guide.md
|   |-- bidirectional_direction_prediction.md
|   |-- paper_queue.md
|   `-- progress_log.md
|-- output/
|   |-- bidirectional_direction/
|   |-- contextual_qnn/
|   |-- contextual_qnn_multilevel/
|   |-- qnn_diagnostic_aapl/
|   |-- qnn_full_aapl/
|   |-- sequence_hybrid_aapl/
|   `-- quantum_inspired/
|-- scripts/
|   |-- run_checks.py
|   `-- run_local_suite.py
|-- src/qsp/
|   |-- data.py
|   |-- evaluation.py
|   |-- web_demo.py
|   |-- experiments/
|   `-- models/
|-- custom_qnn_financial_pipeline.py
`-- reproduce_quantum_circuit.py
```

## Environment Setup

### 1. Create the virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 1a. Install the CUDA PyTorch stack for this machine

This project can use the GPU for the PyTorch-based models, but not for the
Qiskit `EstimatorQNN` core itself. On this Windows machine with NVIDIA CUDA
support, install the CUDA build with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu-cu128.txt
```

If you are replacing an existing CPU-only torch install, first close any
running Python, Streamlit, or notebook process that is using `.venv`, then run:

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y torch torchvision torchaudio
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu-cu128.txt
```

Current limitation:

- CUDA is available for the PyTorch-based models.
- The preserved Qiskit `EstimatorQNN` path is still CPU-bound unless a compatible `qiskit-aer` backend is installed separately.

### 2. Set import paths

The local modules are designed to run from the repository root.

```powershell
$env:PYTHONPATH='src;.'
```

Optional editable install:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

## Quick Start

### Fastest useful path

Run the lightweight suite that is practical on a normal CPU laptop:

```powershell
$env:PYTHONPATH='src;.'
.\.venv\Scripts\python.exe scripts\run_local_suite.py --all-light
```

This performs:

- circuit verification;
- one-step QNN sanity test;
- ContextualQNN AAPL run;
- ContextualQNN two-asset QMTL run;
- ANN / QQBN / QQTN AAPL run.

For the four-stock bidirectional up/down prediction track:

```powershell
$env:PYTHONPATH='src;.'
.\.venv\Scripts\python.exe -m qsp.experiments.run_bidirectional_direction --symbols AAPL MSFT GOOGL NVDA --epochs 120 --max-rows 1200 --output-dir output\bidirectional_direction
```

For the optional stronger GPU-friendly sequence hybrid experiment:

```powershell
$env:PYTHONPATH='src;.'
.\.venv\Scripts\python.exe -m qsp.experiments.run_sequence_hybrid --symbol AAPL --epochs 70 --max-rows 1000 --window-size 24 --hidden-dim 96 --learning-rate 0.0008 --output-dir output\sequence_hybrid_aapl
```

### Launch the dashboard

```powershell
$env:PYTHONPATH='src;.'
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

Local dashboard URL:

```text
http://localhost:8501
```

## Web Dashboard Guide

The Streamlit dashboard includes:

- **Overview**: project map and grouped saved-result history;
- **HQNN-FSP**: CustomQNN/HybridQNN1 regression results, circuit diagrams, and saved AAPL plots;
- **ContextualQNN**: binary ContextualQNN, QMTL, and `d=4` return-regime artifacts;
- **QQBN / QQTN**: interim ANN, QQBN, and QQTN direction-classification artifacts;
- **Bidirectional Direction**: the dedicated four-stock up/down benchmark, plots, thresholds, prediction rows, and next-step notes;
- **BiLSTM Extension**: fourth-paper sequence-learning extension kept separate from the first implementation;
- **Interactive Demos**: lightweight binary and `d=4` ContextualQNN demos;
- **Paper Tracker**: progress notes and paper coverage;
- **Docs & Runbook**: commands, local document paths, and external references.

### How to use the interactive prediction page

1. Open the **Interactive Prediction** tab.
2. Select a supported ticker such as `AAPL`, `MSFT`, `GOOGL`, `NVDA`, `TSLA`, `0700.HK`, or `BTC-USD`.
3. Adjust `ContextualQNN epochs` and `Recent samples` if needed.
4. Press **Run prediction**.

The dashboard will:

- download live prices with `yfinance` when available;
- derive binary return labels from recent closes;
- fit a lightweight statevector ContextualQNN;
- show the latest close, naive next close, probability of an up move, and holdout accuracy.

### How to use the advanced ContextualQNN page

1. Open the **Advanced ContextualQNN** tab.
2. Select a supported ticker.
3. Adjust `d=4 epochs` and `d=4 recent samples` if needed.
4. Press **Run d=4 prediction**.

This page uses density-based return buckets with `d=4`, context length `T=2`, and a lightweight statevector model. It predicts the next return regime rather than the exact next close, and it shows the full bucket probability distribution.

### How to use the bidirectional direction page

1. Open the **Bidirectional Direction** tab.
2. Review the saved `ANN`, `QQBN`, and `QQTN` bidirectional direction table.
3. Select a stock and model to view probability-up and confusion-matrix plots.
4. Use the **BiLSTM Extension** tab only as a later extension, not as the first deliverable.

This tab shows saved artifacts rather than live browser-side training. The primary plots show the probability of an up move and the holdout confusion matrix for each stock/model pair. The ensemble row is included as a comparison attempt, but the main model remains QQTN.

If `yfinance` is unavailable, the dashboard falls back to deterministic sample data and labels that result clearly.

## Paper Tracks And Commands

### 1. HQNN-FSP

Paper:
`ref/2503.15403v1.pdf`

Render the reproduced circuits:

```powershell
.\.venv\Scripts\python.exe reproduce_quantum_circuit.py --fold 25
```

Verify preserved architecture and draw trainable/original circuits:

```powershell
.\.venv\Scripts\python.exe custom_qnn_financial_pipeline.py --verify
```

Run the dummy optimizer sanity check:

```powershell
.\.venv\Scripts\python.exe custom_qnn_financial_pipeline.py --dummy
```

Run the heavier AAPL regression benchmark:

```powershell
.\.venv\Scripts\python.exe custom_qnn_financial_pipeline.py --full
```

Important note:
the full regression path is the most expensive part of the repository. The standalone CustomQNN and HybridQNN1 use simulator-based Qiskit gradients and are slow on CPU.

### 2. Contextual QNN

Single-asset AAPL run:

```powershell
$env:PYTHONPATH='src;.'
.\.venv\Scripts\python.exe -m qsp.experiments.run_contextual_qnn --symbol AAPL --epochs 100 --max-samples 128 --num-layers 4 --learning-rate 0.3 --spsa-perturbation 0.01
```

Two-asset QMTL run:

```powershell
$env:PYTHONPATH='src;.'
.\.venv\Scripts\python.exe -m qsp.experiments.run_contextual_qnn --qmtl --qmtl-symbols AAPL MSFT --epochs 400 --max-samples 160 --num-layers 3 --learning-rate 0.1 --spsa-perturbation 0.01
```

Higher-resolution `d=4` AAPL run:

```powershell
$env:PYTHONPATH='src;.'
.\.venv\Scripts\python.exe -m qsp.experiments.run_contextual_qnn_multilevel --symbol AAPL --epochs 240 --max-samples 256 --num-layers 4 --learning-rate 0.05 --spsa-perturbation 0.01
```

Training method:

- binary return quantization with `d=2`;
- context length `T=2`;
- forecast horizon `tau=1`;
- statevector-style simulator in NumPy;
- SPSA-style update;
- time-ordered train/test split.

Higher-resolution extension:

- density-based return quantization with `d=4`;
- two qubits per symbol, so `T=2, tau=1` maps to 6 qubits;
- exact-match multiclass holdout accuracy on the saved AAPL artifact.

Current live-data result:

- `ContextualQNN` AAPL: `0.6923` directional accuracy
- `ContextualQNN-QMTL` AAPL+MSFT: `0.5469` directional accuracy
- `ContextualQNN-d4` AAPL: `0.4231` multiclass accuracy

### 3. ANN / QQBN / QQTN

Run the live AAPL comparison:

```powershell
$env:PYTHONPATH='src;.'
.\.venv\Scripts\python.exe -m qsp.experiments.run_quantum_inspired --symbol AAPL --epochs 60 --hidden-dim 48 --max-samples 420
```

Training method:

- technical indicators from OHLCV;
- MinMax scaling;
- binary next-return target;
- 80/20 time-ordered split;
- Adam optimizer;
- binary cross-entropy via `BCEWithLogitsLoss`.

Model interpretation:

- `ANN`: classical feed-forward baseline;
- `QQBN`: qubit-inspired two-state feature expansion;
- `QQTN`: qutrit-inspired three-state feature expansion.

### 4. Bidirectional Direction Prediction

Run the four-stock up/down prediction benchmark:

```powershell
$env:PYTHONPATH='src;.'
.\.venv\Scripts\python.exe -m qsp.experiments.run_bidirectional_direction --symbols AAPL MSFT GOOGL NVDA --epochs 120 --max-rows 1200 --output-dir output\bidirectional_direction
```

Training method:

- target is next-day direction: `Next_Return > 0`;
- benchmark assets are `AAPL`, `MSFT`, `GOOGL`, and `NVDA`;
- models reuse the interim `ANN`, `QQBN`, and `QQTN` classifiers;
- features add rolling volatility, momentum, SMA ratios, MACD signal/histogram, and volume changes on top of the existing technical indicators;
- scaling is fit on the training split only;
- each stock uses a time-ordered train/validation/test split;
- neural models use validation-calibrated probability thresholds instead of a fixed `0.5`.
- an `ANN+QQBN+QQTN ensemble` row is saved as a comparison attempt, but it does not beat the main QQTN row on the current four-stock average.
- tuned defaults are `hidden_dim=48` and `learning_rate=0.003`, selected from a small local sweep across the reused model family.

Saved artifacts are written to `output/bidirectional_direction/`, including per-stock predictions, thresholds, training logs, probability plots, confusion matrices, and a combined result table.

Presentation reference:
`docs/bidirectional_direction_prediction.md`

Current saved four-stock average:

- `QQTN`: `Directional Accuracy 0.5236`, `F1 0.6789`, `Sharpe Ratio 0.9538`
- `QQTN balanced threshold`: `Directional Accuracy 0.5236`, `F1 0.6275`, `Balanced Accuracy 0.5166`
- `ANN+QQBN+QQTN ensemble`: `Directional Accuracy 0.5111`, `F1 0.6756`
- `Majority baseline`: `Directional Accuracy 0.5111`, `F1 0.6761`
- `Momentum baseline`: `Directional Accuracy 0.5028`, `F1 0.5123`

This is a modest direction-classification improvement, not a claim that the forecasting problem is solved.

### 5. Sequence Hybrid Extension

This is a repository extension rather than an exact paper reproduction. Its purpose is practical: to test whether a stronger temporal encoder combined with a quantum-inspired head can produce a better local hybrid artifact on GPU without modifying the preserved HQNN-FSP circuit.

This is not the first bidirectional deliverable. It remains an optional later extension because the interim presentation did not include BiLSTM as the main model.

Run it with:

```powershell
$env:PYTHONPATH='src;.'
.\.venv\Scripts\python.exe -m qsp.experiments.run_sequence_hybrid --symbol AAPL --epochs 70 --max-rows 1000 --window-size 24 --hidden-dim 96 --learning-rate 0.0008 --output-dir output\sequence_hybrid_aapl
```

Current saved AAPL result:

- `BiLSTM baseline`: `RMSE 3.7144`, `MAE 2.6781`, `Directional Accuracy 0.5250`
- `BiLSTM-QQTN hybrid`: `RMSE 3.7154`, `MAE 2.6731`, `Directional Accuracy 0.5250`

## Saved Outputs

### Main output folders

- `output/qnn_full_aapl/`
  HQNN-FSP full AAPL benchmark, summary, plots, training log, result table
- `output/qnn_diagnostic_aapl/`
  smaller diagnostic rerun for CustomQNN and HybridQNN1 plotting
- `output/contextual_qnn/`
  ContextualQNN and QMTL result tables, losses, predictions
- `output/contextual_qnn_multilevel/`
  higher-resolution `d=4` ContextualQNN artifacts
- `output/quantum_inspired/`
  ANN / QQBN / QQTN result tables and loss curves
- `output/bidirectional_direction/`
  four-stock up/down prediction tables, row-level predictions, thresholds, probability plots, and confusion matrices
- `output/sequence_hybrid_aapl/`
  GPU-friendly sequence hybrid result table, loss curves, and implied price plots

## Why Some QNN Curves Are Flat

The `Standalone CustomQNN` and `HybridQNN1` predicted curves can appear almost flat. This is not a plotting bug. It is a real training result caused by:

- very limited Qiskit simulator training budget;
- a hard next-close regression target;
- the cost of multi-epoch gradient-based optimization on CPU.

The repository keeps these outputs because they are informative. They show that the preserved circuit runs end-to-end, but they do not yet show strong predictive performance.

## What You Can Run Locally

Practical on a normal laptop:

- circuit verification;
- dummy QNN sanity test;
- Streamlit dashboard;
- ContextualQNN AAPL run;
- ContextualQNN two-asset QMTL run;
- ANN / QQBN / QQTN live-data run;
- the bidirectional direction four-stock run;
- GPU-friendly `BiLSTM-QQTN` sequence hybrid run;
- diagnostic CustomQNN plots on smaller subsets.

Expensive or limited locally:

- repeated full-dataset Qiskit regression sweeps;
- long multi-epoch HybridQNN1 training;
- native qutrit hardware execution;
- the largest multi-asset or noise-study variants from the Contextual QNN paper.

## Troubleshooting

### `yfinance` download fails

The repository now routes `yfinance` cache files through a temporary directory because project-local cache paths were causing SQLite `disk I/O error` on this Windows setup.

If you still see failures, try:

```powershell
Remove-Item "$env:TEMP\\quantum_stock_price_prediction" -Recurse -Force -ErrorAction SilentlyContinue
```

Then rerun the command.

### Streamlit page does not update

Stop existing Python/Streamlit processes and relaunch:

```powershell
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
$env:PYTHONPATH='src;.'
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

### Check the environment quickly

```powershell
$env:PYTHONPATH='src;.'
.\.venv\Scripts\python.exe scripts\run_checks.py
```

## Documents

- [Project Progress](D:\coding_workspace\master%20capstone\quantum-stock-price-prediction\docs\progress_log.md)
- [Paper Coverage](D:\coding_workspace\master%20capstone\quantum-stock-price-prediction\docs\paper_queue.md)
- [ the Bidirectional Direction Prediction](D:\coding_workspace\master%20capstone\quantum-stock-price-prediction\docs\bidirectional_direction_prediction.md)
- [Local Testing Guide](D:\coding_workspace\master%20capstone\quantum-stock-price-prediction\docs\local_testing_guide.md)
- [Related Papers](D:\coding_workspace\master%20capstone\quantum-stock-price-prediction\docs\related_papers.md)

## References

- [HQNN-FSP arXiv 2503.15403v1](https://arxiv.org/abs/2503.15403)
- [Contextual Quantum Neural Networks for Stock Price Prediction](https://www.nature.com/articles/s41598-025-34413-5)
- [Quantum Inspired Qubit Qutrit Neural Networks for Real Time Financial Forecasting](https://www.nature.com/articles/s41598-025-09475-0)
- [BLS-QLSTM: a novel hybrid quantum neural network for stock index forecasting](https://www.nature.com/articles/s41599-025-05348-z)
