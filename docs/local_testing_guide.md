# Local Testing Guide

This project mixes lightweight simulator code with heavier Qiskit training. The easiest way to approach it locally is to separate what can be tested quickly from what is computationally expensive.

## Safe Local Runs

These commands are practical on a normal laptop:

```powershell
$env:PYTHONPATH='src;.'
.\.venv\Scripts\python.exe scripts\run_checks.py
.\.venv\Scripts\python.exe custom_qnn_financial_pipeline.py --verify
.\.venv\Scripts\python.exe custom_qnn_financial_pipeline.py --dummy
.\.venv\Scripts\python.exe -m qsp.experiments.run_contextual_qnn --symbol AAPL --epochs 100 --max-samples 128
.\.venv\Scripts\python.exe -m qsp.experiments.run_contextual_qnn_multilevel --symbol AAPL --epochs 240 --max-samples 256 --num-layers 4 --learning-rate 0.05
.\.venv\Scripts\python.exe -m qsp.experiments.run_quantum_inspired --symbol AAPL --epochs 60 --hidden-dim 48
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

These runs finish quickly because they use either a tiny Qiskit check, a NumPy-based ContextualQNN simulator, or ordinary PyTorch models. The new `d=4` ContextualQNN run is slower than the binary one, but it is still practical on a normal CPU laptop.

lightweight path:

```powershell
$env:PYTHONPATH='src;.'
.\.venv\Scripts\python.exe scripts\run_local_suite.py --all-light
```

## Heavier Local Runs

The full AAPL CustomQNN regression pipeline is substantially more expensive:

```powershell
.\.venv\Scripts\python.exe custom_qnn_financial_pipeline.py --full
```

This path uses Qiskit Machine Learning with simulator-based gradients. Standalone CustomQNN and HybridQNN1 can take many minutes even at one epoch. Increasing epochs for those models scales badly on CPU.

## What Cannot Be Tested Properly On A Normal Laptop

- large multi-epoch Qiskit CustomQNN sweeps on the full dataset;
- hardware-level qutrit execution, because the QQTN implementation here is qutrit-inspired rather than a native qutrit device run;
- the full Contextual QNN portfolio and noise experiments from the paper without a more deliberate simulation budget.
