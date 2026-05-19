"""Canonical CustomQNN wrappers.

The implementation intentionally delegates to ``custom_qnn_financial_pipeline``
so the paper-preserved circuit has one source of truth.
"""

from custom_qnn_financial_pipeline import (
    build_custom_qnn_circuit,
    build_original_custom_qnn_circuit,
    create_estimator_qnn,
    create_torch_qnn_layer,
    run_dummy_sanity_test,
    verify_architecture_unchanged,
)

__all__ = [
    "build_custom_qnn_circuit",
    "build_original_custom_qnn_circuit",
    "create_estimator_qnn",
    "create_torch_qnn_layer",
    "run_dummy_sanity_test",
    "verify_architecture_unchanged",
]
