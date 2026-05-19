# Quantum Stock Price Prediction

This repository contains the implementation workspace for the COMP7705 MSc capstone project, *Comparative Analysis of Quantum-Enhanced Neural Networks in Financial Price Prediction*.

The project compares classical and quantum-enhanced models for financial forecasting. The current code covers three paper tracks:

- *HQNN-FSP: A Hybrid Classical-Quantum Neural Network for Regression-Based Financial Stock Market Prediction*
- *Contextual Quantum Neural Networks for Stock Price Prediction*
- *Quantum Inspired Qubit Qutrit Neural Networks for Real Time Financial Forecasting*

## Environment

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Use the package modules from the repository root:

```powershell
$env:PYTHONPATH='src;.'
```

## HQNN-FSP CustomQNN

Render the reproduced paper circuits:

```powershell
.\.venv\Scripts\python.exe reproduce_quantum_circuit.py --fold 25
```

Verify the preserved trainable circuit:

```powershell
.\.venv\Scripts\python.exe custom_qnn_financial_pipeline.py --verify
```

Run the dummy QNN optimizer check:

```powershell
.\.venv\Scripts\python.exe custom_qnn_financial_pipeline.py --dummy
```

Run the AAPL regression pipeline:

```powershell
.\.venv\Scripts\python.exe custom_qnn_financial_pipeline.py --full
```

Saved outputs are written to `output/qnn_full_aapl`.

## Contextual QNN

```powershell
$env:PYTHONPATH='src;.'
.\.venv\Scripts\python.exe -m qsp.experiments.run_contextual_qnn --symbol AAPL --epochs 30 --max-samples 256
.\.venv\Scripts\python.exe -m qsp.experiments.run_contextual_qnn --qmtl --qmtl-symbols AAPL MSFT --epochs 30 --max-samples 128
```

Saved outputs are written to `output/contextual_qnn`.

## ANN, QQBN, and QQTN

```powershell
$env:PYTHONPATH='src;.'
.\.venv\Scripts\python.exe -m qsp.experiments.run_quantum_inspired --symbol AAPL --epochs 30 --max-samples 420
```

Saved outputs are written to `output/quantum_inspired`.

## Dashboard

```powershell
$env:PYTHONPATH='src;.'
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

The dashboard loads saved result tables, circuit diagrams, loss curves, paper status notes, and a ticker-based ContextualQNN demo.

## Checks

```powershell
$env:PYTHONPATH='src;.'
.\.venv\Scripts\python.exe scripts\run_checks.py
```
