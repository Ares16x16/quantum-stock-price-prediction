import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from qsp.models.custom_qnn import build_custom_qnn_circuit, build_original_custom_qnn_circuit


def test_custom_qnn_architecture_unchanged():
    original = build_original_custom_qnn_circuit()
    refactored, input_params, weight_params = build_custom_qnn_circuit()
    assert original.num_qubits == refactored.num_qubits == 5
    assert dict(original.count_ops()) == dict(refactored.count_ops())
    assert original.depth() == refactored.depth() == 33
    assert len(input_params) == 5
    assert len(weight_params) == 49
