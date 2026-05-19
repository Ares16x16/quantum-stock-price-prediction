"""Run the locally feasible experiment suite for the capstone repo.

This script is intentionally biased toward tasks that are practical on a normal
CPU laptop. It avoids the very expensive full-dataset multi-epoch Qiskit runs
unless the user explicitly invokes those paths elsewhere.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from custom_qnn_financial_pipeline import run_dummy_sanity_test, verify_architecture_unchanged
from qsp.experiments.run_contextual_qnn import run_single_asset, run_two_asset_qmtl
from qsp.experiments.run_contextual_qnn_multilevel import run_multilevel_single_asset
from qsp.experiments.run_quantum_inspired import run_quantum_inspired_experiment


def run_lightweight_suite() -> None:
    """Run everything that is fast enough for ordinary local testing."""

    print("Running circuit verification...")
    verify_architecture_unchanged(output_dir=ROOT / "output" / "qnn_pipeline", draw=True)

    print("Running dummy TorchConnector sanity test...")
    run_dummy_sanity_test()

    print("Running ContextualQNN single-asset AAPL experiment...")
    contextual = run_single_asset(
        symbol="AAPL",
        start="2018-01-01",
        output_dir=ROOT / "output" / "contextual_qnn",
        context_length=2,
        epochs=100,
        max_samples=128,
        num_layers=4,
        learning_rate=0.3,
        spsa_perturbation=0.01,
    )
    print(contextual.to_string(index=False))

    print("Running ContextualQNN two-asset QMTL experiment...")
    qmtl = run_two_asset_qmtl(
        symbols=["AAPL", "MSFT"],
        start="2018-01-01",
        output_dir=ROOT / "output" / "contextual_qnn",
        context_length=2,
        epochs=400,
        max_samples_per_asset=160,
        num_layers=3,
        learning_rate=0.1,
        spsa_perturbation=0.01,
    )
    print(qmtl.to_string(index=False))

    print("Running ContextualQNN d=4 AAPL experiment...")
    contextual_d4 = run_multilevel_single_asset(
        symbol="AAPL",
        start="2018-01-01",
        output_dir=ROOT / "output" / "contextual_qnn_multilevel",
        context_length=2,
        num_levels=4,
        epochs=240,
        max_samples=256,
        num_layers=4,
        learning_rate=0.05,
        spsa_perturbation=0.01,
    )
    print(contextual_d4.to_string(index=False))

    print("Running ANN / QQBN / QQTN AAPL experiment...")
    qi = run_quantum_inspired_experiment(
        symbol="AAPL",
        start="2018-01-01",
        output_dir=ROOT / "output" / "quantum_inspired",
        epochs=60,
        max_samples=420,
        hidden_dim=48,
    )
    print(qi.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the locally feasible capstone experiment suite.")
    parser.add_argument(
        "--all-light",
        action="store_true",
        help="Run circuit verification, dummy sanity test, binary ContextualQNN, QMTL, d=4 ContextualQNN, and ANN/QQBN/QQTN.",
    )
    args = parser.parse_args()

    if args.all_light:
        run_lightweight_suite()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
