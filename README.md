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
!pip install -r requirements.txt
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

The smoke run intentionally limits QNN training to a small sample subset by
default so the simulator test finishes quickly. Use `--max-train-samples` and
`--max-test-samples` to scale up gradually after `--verify` and `--dummy` pass.
