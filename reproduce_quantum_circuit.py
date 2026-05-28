"""Reproduce the paper's QNN circuits with Qiskit.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import fitz
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliProductRotationGate
from qiskit.quantum_info import Pauli
from qiskit.quantum_info import Statevector

DEFAULT_NUM_QUBITS = 5
DEFAULT_OUTPUT_DIR = Path("output/circuits")
DEFAULT_ARXIV_SOURCE_DIR = Path("arxiv_source")


def compute_angle_encoding(
    features: Sequence[float],
    f_transform=None,
    g_transform=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Fig. 4 angles defined in the paper."""

    x = np.asarray(features, dtype=float)
    f_values = x if f_transform is None else np.asarray(f_transform(x), dtype=float)
    g_values = x if g_transform is None else np.asarray(g_transform(x), dtype=float)
    theta = np.arcsin(np.clip(f_values, -1.0, 1.0))
    phi = np.arccos(np.clip(g_values, -1.0, 1.0))
    return theta, phi


def build_angle_encoding_circuit(num_qubits: int = DEFAULT_NUM_QUBITS) -> QuantumCircuit:
    """Build the angle encoding circuit from Fig. 4."""

    circuit = QuantumCircuit(num_qubits, name="fig4_angle_encoding")
    for qubit in range(num_qubits):
        circuit.ry(0, qubit)
        circuit.rz(0, qubit)
    return circuit


def ppr_gate(width: int) -> PauliProductRotationGate:
    """Return the Pauli Product Rotation block shown as `Ppr(0)`."""

    return PauliProductRotationGate(Pauli("Z" * width), angle=0, label="Ppr")


def build_custom_qnn_ansatz_circuit(num_qubits: int = DEFAULT_NUM_QUBITS) -> QuantumCircuit:
    """Transcribe Fig. 5: customized QNN ansatz.

    Reference parts:
    - `arxiv_source/fig5.pdf` supplies the exact visible gate order.
    - The CNOT pairs come from the blue control/target vector geometry.
    - `Ppr(0)` is implemented with Qiskit's Pauli Product Rotation gate using
      a zero angle and a Z-product Pauli string.
    """

    if num_qubits != DEFAULT_NUM_QUBITS:
        raise ValueError("Fig. 5 is a five-qubit circuit.")

    circuit = QuantumCircuit(num_qubits, name="fig5_custom_qnn_ansatz")

    # First line of Fig. 5, ordered by the gate columns in arxiv_source/fig5.pdf.
    circuit.h([0, 1, 2, 3, 4])
    circuit.cx(0, 1)
    circuit.cx(3, 4)
    circuit.rz(0, 1)
    circuit.rz(0, 4)
    circuit.cx(0, 1)
    circuit.cx(3, 4)
    circuit.ry(0, 3)
    circuit.ry(0, 4)
    circuit.cx(0, 2)
    circuit.rz(0, 2)
    circuit.rz(0, 3)
    circuit.rz(0, 4)
    circuit.cx(0, 2)
    circuit.ry(0, 0)
    circuit.cx(1, 2)
    circuit.rz(0, 0)
    circuit.rz(0, 2)
    circuit.rx(0, 0)
    circuit.cx(1, 2)
    circuit.rz(0, 0)
    circuit.ry(0, 1)
    circuit.ry(0, 2)
    circuit.rz(0, 1)
    circuit.rz(0, 2)
    circuit.append(ppr_gate(2), [0, 1])
    circuit.rx(0, 1)
    circuit.rz(0, 1)
    circuit.append(ppr_gate(2), [1, 2])
    circuit.rx(0, 2)
    circuit.rz(0, 2)
    circuit.append(ppr_gate(2), [2, 3])
    circuit.ry(0, 2)
    circuit.rx(0, 3)
    circuit.ry(0, 2)
    circuit.rz(0, 3)
    circuit.append(ppr_gate(2), [3, 4])
    circuit.rz(0, 2)
    circuit.rz(0, 2)
    circuit.rx(0, 4)
    circuit.rz(0, 4)
    circuit.cx(2, 3)
    circuit.append(ppr_gate(5), [0, 1, 2, 3, 4])

    # Continuation line of Fig. 5.
    circuit.ry(0, 0)
    circuit.ry(0, 3)
    circuit.ry(0, 4)
    circuit.ry(0, 0)
    circuit.ry(0, 3)
    circuit.ry(0, 4)
    circuit.rz(0, 0)
    circuit.rz(0, 3)
    circuit.rz(0, 4)
    circuit.rz(0, 0)
    circuit.rz(0, 3)
    circuit.rz(0, 4)
    circuit.cx(0, 1)
    circuit.ry(0, 1)
    circuit.ry(0, 1)
    circuit.rz(0, 1)
    circuit.rz(0, 1)

    return circuit


def build_qnn_regressor_circuit(num_qubits: int = DEFAULT_NUM_QUBITS) -> QuantumCircuit:
    """Transcribe Fig. 6: QNN Regressor circuit used in HybridQNN1.

    Reference parts:
    - The CNOT pairs come from the blue control/target vector geometry
    - The five `Ppr(0)` blocks are implemented as opaque Qiskit gates so the 
      circuit remains runnable.
    """

    if num_qubits != DEFAULT_NUM_QUBITS:
        raise ValueError("Fig. 6 is a five-qubit circuit.")

    circuit = QuantumCircuit(num_qubits, name="fig6_qnn_regressor")

    # First line of the folded Qiskit drawing.
    circuit.h([0, 1, 2, 3, 4])
    circuit.cx(1, 2)
    circuit.cx(0, 4)
    circuit.rz(0, 2)
    circuit.rz(0, 4)
    circuit.cx(1, 2)
    circuit.cx(0, 4)
    circuit.ry(0, 0)
    circuit.ry(0, 4)
    circuit.cx(1, 3)
    circuit.rz(0, 0)
    circuit.rz(0, 3)
    circuit.rz(0, 4)
    circuit.rx(0, 0)
    circuit.cx(1, 3)
    circuit.rz(0, 0)
    circuit.ry(0, 1)
    circuit.cx(2, 3)
    circuit.rz(0, 1)
    circuit.rz(0, 3)
    circuit.append(ppr_gate(2), [0, 1])
    circuit.cx(2, 3)
    circuit.rx(0, 1)
    circuit.ry(0, 2)
    circuit.ry(0, 3)
    circuit.rz(0, 1)
    circuit.rz(0, 2)
    circuit.rz(0, 3)
    circuit.append(ppr_gate(2), [1, 2])
    circuit.rx(0, 2)
    circuit.rz(0, 2)
    circuit.append(ppr_gate(2), [2, 3])
    circuit.ry(0, 2)
    circuit.ry(0, 2)
    circuit.rx(0, 3)
    circuit.rz(0, 3)
    circuit.append(ppr_gate(2), [3, 4])
    circuit.rz(0, 2)
    circuit.rz(0, 2)
    circuit.rx(0, 4)
    circuit.rz(0, 4)
    circuit.cx(2, 3)
    circuit.append(ppr_gate(5), [0, 1, 2, 3, 4])
    circuit.ry(0, 0)
    circuit.ry(0, 3)
    circuit.ry(0, 4)

    # Continuation line
    circuit.ry(0, 0)
    circuit.rz(0, 0)
    circuit.rz(0, 0)
    circuit.cx(0, 1)
    circuit.ry(0, 1)
    circuit.ry(0, 1)
    circuit.rz(0, 1)
    circuit.rz(0, 1)
    circuit.ry(0, 3)
    circuit.rz(0, 3)
    circuit.rz(0, 3)
    circuit.ry(0, 4)
    circuit.rz(0, 4)
    circuit.rz(0, 4)

    return circuit


def build_full_qnn_pipeline(num_qubits: int = DEFAULT_NUM_QUBITS) -> QuantumCircuit:
    """Compose Fig. 4 angle encoding with the Fig. 6 QNN regressor."""

    circuit = build_angle_encoding_circuit(num_qubits).compose(
        build_qnn_regressor_circuit(num_qubits)
    )
    circuit.name = "fig4_plus_fig6_qnn_regressor"
    return circuit


def build_angle_encoding_plus_ansatz(num_qubits: int = DEFAULT_NUM_QUBITS) -> QuantumCircuit:
    """Compose Fig. 4 angle encoding with the Fig. 5 QNN ansatz."""

    circuit = build_angle_encoding_circuit(num_qubits).compose(
        build_custom_qnn_ansatz_circuit(num_qubits)
    )
    circuit.name = "fig4_plus_fig5_custom_qnn_ansatz"
    return circuit


def render_circuit(circuit: QuantumCircuit, output_dir: Path, fold: int) -> None:
    """Write text, PNG, and QPY artifacts for a circuit."""

    output_dir.mkdir(parents=True, exist_ok=True)
    text_path = output_dir / f"{circuit.name}.txt"
    png_path = output_dir / f"{circuit.name}.png"

    text_path.write_text(str(circuit.draw(output="text", fold=fold)), encoding="utf-8")
    figure = circuit.draw(output="mpl", fold=fold)
    figure.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def render_source_figure(source_dir: Path, output_dir: Path, name: str, scale: float = 2.0) -> None:
    source_pdf = source_dir / f"{name}.pdf"
    if not source_pdf.exists():
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(source_pdf)
    pixmap = doc[0].get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    pixmap.save(output_dir / f"{name}_paper_exact.png")


def verify_circuit(circuit: QuantumCircuit) -> None:
    """Run a small statevector check to prove the circuit is executable."""

    Statevector.from_instruction(circuit)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the paper's QNN circuits.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=25,
        help="Qiskit fold value.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_ARXIV_SOURCE_DIR,
    )
    args = parser.parse_args()

    circuits = [
        build_angle_encoding_circuit(),
        build_custom_qnn_ansatz_circuit(),
        build_qnn_regressor_circuit(),
        build_angle_encoding_plus_ansatz(),
        build_full_qnn_pipeline(),
    ]

    for circuit in circuits:
        verify_circuit(circuit)
        render_circuit(circuit, args.output_dir, args.fold)

    render_source_figure(args.source_dir, args.output_dir, "fig5")
    render_source_figure(args.source_dir, args.output_dir, "fig6")
    print(f"Rendered {len(circuits)} circuits to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
