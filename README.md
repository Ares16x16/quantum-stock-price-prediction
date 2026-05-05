# quantum-stock-price-prediction

```powershell
python reproduce_quantum_circuit.py --fold 25
```

## Trainable custom QNN pipeline

The trainable QNN/refactored financial pipeline is in:

```powershell
custom_qnn_financial_pipeline.py
```

Recommended Colab setup:

```python
!pip install qiskit qiskit-machine-learning yfinance scikit-learn matplotlib pandas torch
```

Architecture verification and circuit drawings:

```powershell
python custom_qnn_financial_pipeline.py --verify
```

Minimal QNN forward/backward sanity test:

```powershell
python custom_qnn_financial_pipeline.py --dummy
```

Small AAPL smoke run:

```powershell
python custom_qnn_financial_pipeline.py --aapl-smoke
```

Outputs are written under `output/qnn_pipeline`, including circuit diagrams,
loss plots, actual-vs-predicted plots, `training_log.csv`, and
`result_table.csv`.
