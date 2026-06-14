"""Streamlit dashboard for the capstone project.

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qsp.web_demo import (
    SUPPORTED_TICKERS,
    run_multilevel_quick_prediction,
    run_quick_prediction,
)

OUTPUT = ROOT / "output"
DOCS = ROOT / "docs"
FULL_OUTPUT = OUTPUT / "qnn_full_aapl"
DIAGNOSTIC_OUTPUT = OUTPUT / "qnn_diagnostic_aapl"
BIDIRECTIONAL_OUTPUT = OUTPUT / "bidirectional_direction"


st.set_page_config(
    page_title="Quantum Stock Price Prediction",
    page_icon="Q",
    layout="wide",
)

st.title("Quantum-Enhanced Neural Networks for Financial Price Prediction")
st.caption("Paper-aligned dashboard for saved artifacts, the direction track, and lightweight demos.")


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _read_csv(path: Path, warn: bool = True) -> pd.DataFrame | None:
    if not path.exists():
        if warn:
            st.warning(f"Missing artifact: `{_relative(path)}`")
        return None
    return pd.read_csv(path)


def _read_doc(path: Path) -> str:
    if not path.exists():
        return f"Missing document: `{_relative(path)}`"
    return path.read_text(encoding="utf-8")


def _dashboard_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Hide placeholder teammate rows and normalize common numeric columns."""

    cleaned = frame.copy()
    if "Model name" in cleaned.columns:
        teammate_mask = cleaned["Model name"].astype(str).str.contains(
            "teammate|QLSTM placeholder", case=False, na=False
        )
        cleaned = cleaned.loc[~teammate_mask]

    for metric in ["RMSE", "MAE", "Directional Accuracy", "Accuracy", "F1", "Balanced Accuracy"]:
        if metric in cleaned.columns:
            cleaned = cleaned.loc[cleaned[metric].astype(str).str.upper() != "TBD"]
            cleaned[metric] = pd.to_numeric(cleaned[metric], errors="coerce")

    return cleaned.reset_index(drop=True)


def _select_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[[column for column in columns if column in frame.columns]].copy()


def _format_table(frame: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_numeric_dtype(display[column]):
            display[column] = display[column].round(digits)
    return display


def _show_result_table(
    title: str,
    path: Path,
    columns: list[str],
    caption: str | None = None,
    expanded: bool = True,
) -> pd.DataFrame | None:
    frame = _read_csv(path, warn=False)
    if frame is None:
        st.info(f"`{_relative(path)}` has not been generated yet.")
        return None
    frame = _dashboard_table(frame)
    if title and expanded:
        st.subheader(title)
    table = _format_table(_select_columns(frame, columns))
    if expanded:
        st.dataframe(table, use_container_width=True, hide_index=True)
    else:
        with st.expander(title):
            st.dataframe(table, use_container_width=True, hide_index=True)
    if caption:
        st.caption(caption)
    return frame


def _show_image(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Missing image: `{_relative(path)}`")


def _average_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "Dataset / asset" not in frame.columns:
        return pd.DataFrame()
    return frame[frame["Dataset / asset"].astype(str).str.contains("Average", case=False, na=False)].copy()


def _symbol_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "Dataset / asset" not in frame.columns:
        return frame.copy()
    return frame[~frame["Dataset / asset"].astype(str).str.contains("Average", case=False, na=False)].copy()


def _best_row(frame: pd.DataFrame, metric: str) -> pd.Series | None:
    if frame.empty or metric not in frame.columns:
        return None
    numeric = pd.to_numeric(frame[metric], errors="coerce")
    if numeric.dropna().empty:
        return None
    return frame.loc[numeric.idxmax()]


def _normalize_binary_prediction_output(result: tuple) -> tuple:
    """Accept old cached tuples and normalize them to the current 3-item shape."""

    if len(result) == 3:
        return result

    if len(result) == 2:
        prediction, recent_prices = result
        comparison_frame = pd.DataFrame()
        return prediction, recent_prices, comparison_frame

    raise ValueError(
        "Unexpected Interactive Prediction cache payload. "
        f"Expected 2 or 3 items, got {len(result)}."
    )


def _normalize_multilevel_prediction_output(result: tuple) -> tuple:
    """Accept old cached tuples and normalize them to the current 4-item shape."""

    if len(result) == 4:
        return result

    if len(result) == 3:
        prediction, recent_prices, probability_frame = result
        comparison_frame = pd.DataFrame()
        return prediction, recent_prices, probability_frame, comparison_frame

    if len(result) == 2:
        prediction, recent_prices = result
        probability_frame = pd.DataFrame()
        comparison_frame = pd.DataFrame()
        return prediction, recent_prices, probability_frame, comparison_frame

    raise ValueError(
        "Unexpected Advanced ContextualQNN cache payload. "
        f"Expected 2, 3, or 4 items, got {len(result)}."
    )


def _resolve_binary_prediction_fields(
    prediction: object,
    recent_prices: pd.DataFrame,
    comparison_frame: pd.DataFrame,
) -> dict[str, object]:
    """Read current fields from older cached binary prediction objects with safe fallbacks."""

    last_close = float(getattr(prediction, "last_close", recent_prices["Close"].iloc[-1]))
    previous_close = float(
        getattr(
            prediction,
            "previous_close",
            recent_prices["Close"].iloc[-2] if len(recent_prices) >= 2 else last_close,
        )
    )
    naive_next_close = float(getattr(prediction, "naive_next_close", last_close))

    contextual_predicted_close = getattr(prediction, "contextual_predicted_close", None)
    if contextual_predicted_close is None:
        if not comparison_frame.empty and "ContextualQNN predicted" in comparison_frame.columns:
            contextual_predicted_close = float(comparison_frame["ContextualQNN predicted"].iloc[-1])
        else:
            contextual_predicted_close = last_close

    holdout_rmse = getattr(prediction, "holdout_rmse", None)
    holdout_mae = getattr(prediction, "holdout_mae", None)
    if holdout_rmse is None and not comparison_frame.empty:
        predicted_col = "ContextualQNN predicted"
        actual_col = "Actual next close"
        if predicted_col in comparison_frame.columns and actual_col in comparison_frame.columns:
            residual = comparison_frame[actual_col].astype(float) - comparison_frame[predicted_col].astype(float)
            holdout_rmse = float((residual.pow(2).mean()) ** 0.5)
            holdout_mae = float(residual.abs().mean())

    return {
        "symbol": str(getattr(prediction, "symbol", "Unknown")),
        "last_close": last_close,
        "previous_close": previous_close,
        "last_date": str(getattr(prediction, "last_date", recent_prices["Date"].iloc[-1])),
        "naive_next_close": naive_next_close,
        "contextual_predicted_close": float(contextual_predicted_close),
        "latest_context": str(getattr(prediction, "latest_context", "Unavailable")),
        "contextual_probability_up": float(getattr(prediction, "contextual_probability_up", 0.0)),
        "contextual_direction": str(getattr(prediction, "contextual_direction", "Unavailable")),
        "holdout_accuracy": float(getattr(prediction, "holdout_accuracy", 0.0)),
        "holdout_f1": float(getattr(prediction, "holdout_f1", 0.0)),
        "holdout_rmse": float(holdout_rmse if holdout_rmse is not None else 0.0),
        "holdout_mae": float(holdout_mae if holdout_mae is not None else 0.0),
        "train_samples": int(getattr(prediction, "train_samples", 0)),
        "test_samples": int(getattr(prediction, "test_samples", 0)),
        "data_source": str(getattr(prediction, "data_source", "unknown")),
    }


def _resolve_multilevel_prediction_fields(
    prediction: object,
    recent_prices: pd.DataFrame,
    probability_frame: pd.DataFrame,
) -> dict[str, object]:
    """Read current fields from older cached prediction objects with safe fallbacks."""

    last_close = float(getattr(prediction, "last_close", recent_prices["Close"].iloc[-1]))
    previous_close = float(
        getattr(
            prediction,
            "previous_close",
            recent_prices["Close"].iloc[-2] if len(recent_prices) >= 2 else last_close,
        )
    )
    predicted_bucket = int(getattr(prediction, "predicted_bucket", -1))
    predicted_bucket_label = str(getattr(prediction, "predicted_bucket_label", "Unavailable"))

    predicted_next_close = getattr(prediction, "predicted_next_close", None)
    if predicted_next_close is None:
        if not probability_frame.empty and {"bucket", "probability"}.issubset(probability_frame.columns):
            bucket_midpoints = {
                0: -0.030,
                1: -0.010,
                2: 0.010,
                3: 0.030,
            }
            expected_return = 0.0
            for row in probability_frame.itertuples(index=False):
                expected_return += float(getattr(row, "probability")) * bucket_midpoints.get(
                    int(getattr(row, "bucket")),
                    0.0,
                )
            predicted_next_close = last_close * (1.0 + expected_return)
        else:
            predicted_next_close = last_close

    return {
        "symbol": str(getattr(prediction, "symbol", "Unknown")),
        "last_close": last_close,
        "previous_close": previous_close,
        "last_date": str(getattr(prediction, "last_date", recent_prices["Date"].iloc[-1])),
        "latest_context": str(getattr(prediction, "latest_context", "Unavailable")),
        "predicted_bucket": predicted_bucket,
        "predicted_bucket_label": predicted_bucket_label,
        "predicted_next_close": float(predicted_next_close),
        "holdout_accuracy": float(getattr(prediction, "holdout_accuracy", 0.0)),
        "holdout_rmse": float(getattr(prediction, "holdout_rmse", 0.0)),
        "holdout_mae": float(getattr(prediction, "holdout_mae", 0.0)),
        "data_source": str(getattr(prediction, "data_source", "unknown")),
    }


TRACKS = pd.DataFrame(
    [
        {
            "Tab": "HQNN-FSP",
            "Reference": "Paper 1: HQNN-FSP regression",
            "Models": "Naive, LSTM, CustomQNN, HybridQNN1",
            "Task": "AAPL next-close regression",
            "Main output": "output/qnn_full_aapl",
            "Presentation role": "Preserved Qiskit circuit and honest regression baseline.",
        },
        {
            "Tab": "ContextualQNN",
            "Reference": "Paper 2: Contextual QNN",
            "Models": "ContextualQNN, QMTL, d=4",
            "Task": "Return-context direction/regime prediction",
            "Main output": "output/contextual_qnn*",
            "Presentation role": "Paper-aligned return-state modelling.",
        },
        {
            "Tab": "QQBN / QQTN",
            "Reference": "Paper 3: Qubit/Qutrit neural networks",
            "Models": "ANN, QQBN, QQTN",
            "Task": "AAPL next-day direction",
            "Main output": "output/quantum_inspired",
            "Presentation role": "Interim model family reused by the bidirectional track.",
        },
        {
            "Tab": "Bidirectional Direction",
            "Reference": "Bidirectional track using Papers 1-3",
            "Models": "ANN, QQBN, QQTN, calibrated ensemble",
            "Task": "AAPL/MSFT/GOOGL/NVDA next-day up/down",
            "Main output": "output/bidirectional_direction",
            "Presentation role": "Main current deliverable.",
        },
        {
            "Tab": "BiLSTM Extension",
            "Reference": "Paper 4: BLS-QLSTM / sequence-learning reference",
            "Models": "BiLSTM, BiLSTM-QQTN",
            "Task": "Optional AAPL sequence direction/regression",
            "Main output": "output/sequence_hybrid_aapl",
            "Presentation role": "Separate later extension, not the first implementation.",
        },
    ]
)


overview_tab, hqnn_tab, contextual_tab, qqtn_tab, direction_tab, bilstm_tab, demo_tab, papers_tab, docs_tab = st.tabs(
    [
        "Overview",
        "HQNN-FSP",
        "ContextualQNN",
        "QQBN / QQTN",
        "Bidirectional Direction",
        "BiLSTM Extension",
        "Interactive Demos",
        "Paper Tracker",
        "Docs & Runbook",
    ]
)


with overview_tab:
    st.header("Project Map")
    st.write(
        "The dashboard is organized by paper or model track. The bidirectional direction work is now separate "
        "from the optional BiLSTM extension and from the slow CustomQNN regression circuit."
    )
    st.dataframe(TRACKS, use_container_width=True, hide_index=True)

    st.subheader("Current Headline Results")
    hqnn_result = _read_csv(FULL_OUTPUT / "result_table.csv", warn=False)
    contextual_result = _read_csv(OUTPUT / "contextual_qnn" / "AAPL_result_table.csv", warn=False)
    qqtn_result = _read_csv(OUTPUT / "quantum_inspired" / "AAPL_result_table.csv", warn=False)
    direction_result = _read_csv(BIDIRECTIONAL_OUTPUT / "combined_result_table.csv", warn=False)
    bilstm_result = _read_csv(OUTPUT / "sequence_hybrid_aapl" / "AAPL_result_table.csv", warn=False)

    metric_cols = st.columns(5)
    with metric_cols[0]:
        if hqnn_result is not None:
            hqnn_clean = _dashboard_table(hqnn_result)
            best_rmse = hqnn_clean.loc[pd.to_numeric(hqnn_clean["RMSE"], errors="coerce").idxmin()]
            st.metric("Best HQNN-FSP RMSE", f"{float(best_rmse['RMSE']):.4f}", str(best_rmse["Model name"]))
        else:
            st.metric("Best HQNN-FSP RMSE", "N/A")
    with metric_cols[1]:
        if contextual_result is not None:
            contextual_clean = _dashboard_table(contextual_result)
            st.metric(
                "ContextualQNN acc.",
                f"{float(contextual_clean['Directional Accuracy'].iloc[0]):.4f}",
                str(contextual_clean["Dataset / asset"].iloc[0]),
            )
        else:
            st.metric("ContextualQNN acc.", "N/A")
    with metric_cols[2]:
        if qqtn_result is not None:
            qqtn_clean = _dashboard_table(qqtn_result)
            best = _best_row(qqtn_clean, "Directional Accuracy")
            st.metric("Best QQBN/QQTN acc.", f"{float(best['Directional Accuracy']):.4f}", str(best["Model name"]))
        else:
            st.metric("Best QQBN/QQTN acc.", "N/A")
    with metric_cols[3]:
        if direction_result is not None:
            direction_clean = _dashboard_table(direction_result)
            averages = _average_rows(direction_clean)
            best = _best_row(averages if not averages.empty else direction_clean, "Directional Accuracy")
            st.metric("Direction acc.", f"{float(best['Directional Accuracy']):.4f}", str(best["Model name"]))
        else:
            st.metric("Direction acc.", "N/A")
    with metric_cols[4]:
        if bilstm_result is not None:
            bilstm_clean = _dashboard_table(bilstm_result)
            best = _best_row(bilstm_clean, "Directional Accuracy")
            st.metric("BiLSTM extension acc.", f"{float(best['Directional Accuracy']):.4f}", str(best["Model name"]))
        else:
            st.metric("BiLSTM extension acc.", "N/A")

    st.subheader("Saved Result History")
    st.caption("Grouped by track so regression, direction classification, and sequence experiments are not mixed together.")
    history_sources = {
        "Paper 1 - HQNN-FSP regression": [
            ("Full AAPL benchmark", FULL_OUTPUT / "result_table.csv"),
            ("Diagnostic AAPL benchmark", DIAGNOSTIC_OUTPUT / "result_table.csv"),
            ("GPU LSTM refinement", OUTPUT / "lstm_gpu_refined_aapl" / "result_table.csv"),
        ],
        "Paper 2 - ContextualQNN": [
            ("Binary ContextualQNN", OUTPUT / "contextual_qnn" / "AAPL_result_table.csv"),
            ("Two-asset QMTL", OUTPUT / "contextual_qnn" / "qmtl_two_asset_result_table.csv"),
            ("Four-asset QMTL", OUTPUT / "contextual_qnn_multi_asset" / "qmtl_4_asset_result_table.csv"),
            ("d=4 multilevel", OUTPUT / "contextual_qnn_multilevel" / "AAPL_result_table_d4.csv"),
        ],
        "Paper 3 - ANN / QQBN / QQTN": [
            ("Interim AAPL", OUTPUT / "quantum_inspired" / "AAPL_result_table.csv"),
            ("GPU refined AAPL", OUTPUT / "quantum_inspired_gpu_refined" / "AAPL_result_table.csv"),
            ("GPU refined 420 rows", OUTPUT / "quantum_inspired_gpu_refined_420" / "AAPL_result_table.csv"),
        ],
        "Bidirectional direction": [
            ("Four-stock combined", BIDIRECTIONAL_OUTPUT / "combined_result_table.csv"),
        ],
        "Paper 4 extension - BiLSTM": [
            ("Sequence direction", OUTPUT / "sequence_hybrid_aapl" / "AAPL_result_table.csv"),
            ("Sequence regression", OUTPUT / "sequence_hybrid_regression" / "AAPL_result_table.csv"),
        ],
    }
    history_columns = [
        "Model name",
        "Dataset / asset",
        "RMSE",
        "MAE",
        "Directional Accuracy",
        "F1",
        "Balanced Accuracy",
        "Training time",
        "Notes",
    ]
    for group, sources in history_sources.items():
        with st.expander(group, expanded=group.startswith("Bidirectional")):
            for label, path in sources:
                frame = _read_csv(path, warn=False)
                if frame is None:
                    st.info(f"{label}: `{_relative(path)}` not available.")
                    continue
                st.markdown(f"**{label}**")
                frame = _dashboard_table(frame)
                st.dataframe(_format_table(_select_columns(frame, history_columns)), use_container_width=True, hide_index=True)


with hqnn_tab:
    st.header("Paper 1 - HQNN-FSP / CustomQNN Regression")
    st.write(
        "This tab contains the preserved Qiskit circuit and the AAPL next-close regression benchmark. "
        "It is not part of the bidirectional direction model."
    )
    st.info(
        "CustomQNN and HybridQNN1 are shown here because they reproduce the HQNN-FSP track. "
        "They are not used inside the four-stock bidirectional direction run because that run is a fast PyTorch "
        "direction-classification experiment using ANN, QQBN, and QQTN."
    )

    hqnn_frame = _show_result_table(
        "Saved AAPL Regression Results",
        FULL_OUTPUT / "result_table.csv",
        ["Model name", "Dataset / asset", "RMSE", "MAE", "Directional Accuracy", "Training time", "Parameter count", "Notes"],
        "Lower RMSE/MAE is better for this regression track. Directional accuracy here is secondary.",
    )

    if hqnn_frame is not None and {"Model name", "RMSE", "MAE"}.issubset(hqnn_frame.columns):
        numeric = hqnn_frame.copy()
        numeric["RMSE"] = pd.to_numeric(numeric["RMSE"], errors="coerce")
        numeric["MAE"] = pd.to_numeric(numeric["MAE"], errors="coerce")
        st.bar_chart(numeric.set_index("Model name")[["RMSE", "MAE"]])

    st.subheader("Circuit Evidence")
    col1, col2 = st.columns(2)
    with col1:
        _show_image(FULL_OUTPUT / "original_custom_qnn.png", "Original preserved CustomQNN circuit")
    with col2:
        _show_image(FULL_OUTPUT / "custom_trainable_qnn.png", "Parameterized trainable CustomQNN circuit")

    st.subheader("Saved AAPL Curves")
    model = st.selectbox(
        "Saved plot",
        ["lstm", "custom_qnn", "hybrid_qnn1"],
        format_func={
            "lstm": "Classical LSTM",
            "custom_qnn": "Standalone CustomQNN",
            "hybrid_qnn1": "HybridQNN1",
        }.get,
        key="hqnn_saved_plot",
    )
    plot_root = FULL_OUTPUT if model == "lstm" else DIAGNOSTIC_OUTPUT
    col1, col2 = st.columns(2)
    with col1:
        _show_image(plot_root / f"{model}_actual_vs_predicted.png", f"{model} actual vs predicted")
    with col2:
        _show_image(plot_root / f"{model}_loss.png", f"{model} loss curve")
    if model in {"custom_qnn", "hybrid_qnn1"}:
        st.warning("These Qiskit regression curves remain undertrained and near-flat under the current local simulator budget.")


with contextual_tab:
    st.header("Paper 2 - ContextualQNN")
    st.write(
        "This track predicts return states from recent return contexts. The binary path predicts up/down; "
        "the d=4 path predicts a four-bucket return regime."
    )
    contextual_columns = [
        "Model name",
        "Dataset / asset",
        "Number of qubits",
        "VQC layers",
        "Feature set",
        "Directional Accuracy",
        "Training time",
        "Inference time",
        "Notes",
    ]
    col1, col2 = st.columns(2)
    with col1:
        _show_result_table(
            "Binary ContextualQNN",
            OUTPUT / "contextual_qnn" / "AAPL_result_table.csv",
            contextual_columns,
            "d=2 binary return context, useful for the paper-aligned up/down framing.",
        )
        _show_result_table(
            "Two-Asset QMTL",
            OUTPUT / "contextual_qnn" / "qmtl_two_asset_result_table.csv",
            contextual_columns,
            "Shared plus asset-specific contextual parameters.",
        )
    with col2:
        _show_result_table(
            "Four-Asset QMTL",
            OUTPUT / "contextual_qnn_multi_asset" / "qmtl_4_asset_result_table.csv",
            contextual_columns,
            "Multi-asset contextual extension.",
        )
        _show_result_table(
            "d=4 Multilevel ContextualQNN",
            OUTPUT / "contextual_qnn_multilevel" / "AAPL_result_table_d4.csv",
            contextual_columns,
            "Harder four-class return-regime task, so accuracy is not directly comparable to d=2.",
        )

    with st.expander("How to explain this tab"):
        st.markdown(
            """
            - The ContextualQNN paper motivates modelling a distribution over future return states.
            - The binary `d=2` implementation supports the decision to model up/down direction.
            - The `d=4` implementation is a higher-resolution extension, not a replacement for the binary direction result.
            - These models are lightweight NumPy/statevector-style reproductions, not the slow Qiskit `EstimatorQNN` circuit.
            """
        )


with qqtn_tab:
    st.header("Paper 3 - ANN / QQBN / QQTN")
    st.write(
        "This tab contains the interim model family from the qubit/qutrit paper. "
        "The bidirectional direction track reuses this family rather than introducing BiLSTM first."
    )
    st.markdown(
        """
        - `ANN`: classical feed-forward direction classifier.
        - `QQBN`: qubit-inspired two-state feature expansion.
        - `QQTN`: qutrit-inspired three-state feature expansion and the main reused model.
        """
    )
    qi_columns = [
        "Model name",
        "Dataset / asset",
        "Directional Accuracy",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Sharpe ratio",
        "Information coefficient",
        "Training time",
        "Data source",
        "Notes",
    ]
    col1, col2 = st.columns(2)
    with col1:
        _show_result_table(
            "Interim AAPL Direction Run",
            OUTPUT / "quantum_inspired" / "AAPL_result_table.csv",
            qi_columns,
            "This is the simpler AAPL-only predecessor to the four-stock direction benchmark.",
        )
    with col2:
        _show_result_table(
            "GPU Refined AAPL Direction Run",
            OUTPUT / "quantum_inspired_gpu_refined" / "AAPL_result_table.csv",
            qi_columns,
            "Kept as history. The best presentation row remains whichever saved artifact has the stronger validation story.",
        )

    st.subheader("Loss Curves")
    loss_model = st.selectbox("Loss plot", ["ann", "qqbn", "qqtn"], format_func=str.upper, key="qqtn_loss_plot")
    _show_image(OUTPUT / "quantum_inspired" / f"{loss_model}_loss.png", f"{loss_model.upper()} training loss")


with direction_tab:
    st.header("Bidirectional Direction Prediction")
    st.write(
        "This is the current direction deliverable: predict whether the next trading day closes up or down "
        "for AAPL, MSFT, GOOGL, and NVDA."
    )
    st.success(
        "Model ownership: this track reuses `ANN`, `QQBN`, and `QQTN` from the interim qubit/qutrit paper. "
        "It also reports an `ANN+QQBN+QQTN` calibrated ensemble as a lightweight comparison row."
    )
    st.warning(
        "Not CustomQNN and not BiLSTM-first: CustomQNN belongs to the slow HQNN-FSP regression tab, "
        "and BiLSTM is kept in the separate fourth-paper extension tab."
    )

    direction_frame = _read_csv(BIDIRECTIONAL_OUTPUT / "combined_result_table.csv", warn=False)
    if direction_frame is None:
        st.info("Run the bidirectional direction experiment to populate this tab.")
        st.code(
            ".\\.venv\\Scripts\\python.exe -m qsp.experiments.run_bidirectional_direction "
            "--symbols AAPL MSFT GOOGL NVDA --epochs 120 --max-rows 1200 "
            "--output-dir output\\bidirectional_direction",
            language="powershell",
        )
    else:
        direction_frame = _dashboard_table(direction_frame)
        average = _average_rows(direction_frame)
        symbols_only = _symbol_rows(direction_frame)
        direction_columns = [
            "Model name",
            "Dataset / asset",
            "Directional Accuracy",
            "F1",
            "Balanced Accuracy",
            "Precision",
            "Recall",
            "Specificity",
            "Predicted up rate",
            "True up rate",
            "Validation threshold",
            "Threshold objective",
            "Sharpe ratio",
            "Information coefficient",
            "Notes",
        ]

        st.subheader("Four-Stock Average")
        st.dataframe(_format_table(_select_columns(average, direction_columns)), use_container_width=True, hide_index=True)
        best_average = _best_row(average, "Directional Accuracy")
        if best_average is not None:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Best average model", str(best_average["Model name"]))
            c2.metric("Directional accuracy", f"{float(best_average['Directional Accuracy']):.4f}")
            c3.metric("F1", f"{float(best_average['F1']):.4f}")
            c4.metric("Balanced accuracy", f"{float(best_average['Balanced Accuracy']):.4f}")

        st.subheader("Per-Stock Drilldown")
        symbol_options = sorted(symbols_only["Dataset / asset"].dropna().astype(str).unique().tolist())
        model_options = symbols_only["Model name"].dropna().astype(str).unique().tolist()
        plot_model_options = [
            model_name
            for model_name in model_options
            if model_name not in {"Majority baseline", "Momentum baseline"}
        ]
        if not plot_model_options:
            plot_model_options = model_options
        default_model = "QQTN" if "QQTN" in plot_model_options else plot_model_options[0]
        col_symbol, col_model = st.columns(2)
        with col_symbol:
            direction_symbol = st.selectbox("Benchmark stock", symbol_options, key="direction_symbol")
        with col_model:
            direction_model = st.selectbox(
                "Model or threshold view",
                plot_model_options,
                index=plot_model_options.index(default_model),
                key="direction_model",
            )
        selected_rows = symbols_only[symbols_only["Dataset / asset"].astype(str) == direction_symbol]
        st.dataframe(_format_table(_select_columns(selected_rows, direction_columns)), use_container_width=True, hide_index=True)

        plot_key = direction_model.lower().replace("+", "_").replace("/", "_").replace(" ", "_")
        plot_key = "".join(char if char.isalnum() or char == "_" else "_" for char in plot_key).strip("_")
        col1, col2 = st.columns(2)
        with col1:
            _show_image(
                BIDIRECTIONAL_OUTPUT / f"{direction_symbol}_{plot_key}_probability.png",
                f"{direction_symbol} {direction_model} probability-up holdout plot",
            )
        with col2:
            _show_image(
                BIDIRECTIONAL_OUTPUT / f"{direction_symbol}_{plot_key}_confusion_matrix.png",
                f"{direction_symbol} {direction_model} confusion matrix",
            )

        prediction_path = BIDIRECTIONAL_OUTPUT / f"{direction_symbol}_predictions.csv"
        threshold_path = BIDIRECTIONAL_OUTPUT / "thresholds.csv"
        with st.expander("Saved prediction rows"):
            predictions = _read_csv(prediction_path, warn=False)
            if predictions is not None:
                preview_cols = [
                    "Date",
                    "Actual direction",
                    "Next return",
                    "Previous close",
                    "Actual next close",
                    f"{direction_model} Probability up",
                    f"{direction_model} Predicted direction",
                ]
                st.dataframe(_format_table(_select_columns(predictions, preview_cols)), use_container_width=True, hide_index=True)
        with st.expander("Validation thresholds"):
            thresholds = _read_csv(threshold_path, warn=False)
            if thresholds is not None:
                thresholds = thresholds[thresholds["symbol"].astype(str) == direction_symbol]
                st.dataframe(_format_table(thresholds), use_container_width=True, hide_index=True)

        with st.expander("Feature and evaluation design", expanded=True):
            st.markdown(
                """
                - Target: `Next_Return > 0`, where `Next_Return = Close[t + 1] / Close[t] - 1`.
                - Split: time-ordered train, validation, and test sets.
                - Scaling: `MinMaxScaler` is fit on the training split only.
                - Features: OHLCV, RSI, MACD, SMA, ADX, short and medium returns, momentum, rolling volatility, trend ratios, MACD signal/histogram, and volume changes.
                - Baselines: majority-class and previous-day momentum.
                - Thresholds: selected on validation data before test evaluation.
                - Improvement row: `ANN+QQBN+QQTN ensemble` averages probabilities from the reused interim models.
                """
            )

        with st.expander("Presentation interpretation"):
            st.markdown(_read_doc(DOCS / "bidirectional_direction_prediction.md"))

        with st.expander("Next result-improvement steps"):
            st.markdown(
                """
                - Run walk-forward validation instead of one fixed validation/test split, then report mean and variance.
                - Add market-context features such as SPY, QQQ, VIX, or sector ETF returns with strict date alignment.
                - Calibrate probabilities with Platt scaling or isotonic calibration on validation folds.
                - Test multi-horizon labels, for example next-day, 3-day, and 5-day direction.
                - Consider a CustomQNN direction-classification head only as a separate slow experiment, not inside the first direction track.
                - Keep the BiLSTM sequence path separate unless the presentation explicitly moves to the fourth-paper extension.
                """
            )


with bilstm_tab:
    st.header("Paper 4 Extension - BiLSTM / Sequence Hybrid")
    st.write(
        "This tab is intentionally separate. It exists because a fourth sequence-learning reference was found later, "
        "but it should not be presented as the first bidirectional model."
    )
    st.info(
        "Use this as an optional future extension: a stronger temporal encoder plus a qutrit-inspired head. "
        "The main direction tab stays with ANN/QQBN/QQTN to match the interim scope."
    )
    seq_columns = [
        "Model name",
        "Dataset / asset",
        "RMSE",
        "MAE",
        "Directional Accuracy",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Training time",
        "Data source",
        "Notes",
    ]
    _show_result_table(
        "Sequence Direction Artifact",
        OUTPUT / "sequence_hybrid_aapl" / "AAPL_result_table.csv",
        seq_columns,
        "Direction probabilities are converted to an implied next-close curve for plotting.",
    )
    _show_result_table(
        "Sequence Regression Artifact",
        OUTPUT / "sequence_hybrid_regression" / "AAPL_result_table.csv",
        seq_columns,
        "Regression variant kept separate from the direction artifact.",
        expanded=False,
    )
    model_choice = st.selectbox(
        "Sequence plot",
        ["BiLSTM baseline", "BiLSTM-QQTN hybrid"],
        key="bilstm_plot",
    )
    plot_key = model_choice.replace(" ", "_").lower()
    col1, col2 = st.columns(2)
    with col1:
        _show_image(
            OUTPUT / "sequence_hybrid_aapl" / f"{plot_key}_actual_vs_predicted.png",
            f"{model_choice} holdout curve",
        )
    with col2:
        _show_image(
            OUTPUT / "sequence_hybrid_aapl" / f"{plot_key}_loss.png",
            f"{model_choice} training loss",
        )


with demo_tab:
    st.header("Interactive Demos")
    st.write(
        "These demos run lightweight ContextualQNN paths from the dashboard. They do not train the expensive CustomQNN "
        "and they do not regenerate the saved four-stock benchmark."
    )
    binary_demo, d4_demo = st.tabs(["Binary ContextualQNN", "d=4 ContextualQNN"])

    with binary_demo:
        st.subheader("Binary Up/Down ContextualQNN")
        col_a, col_b, col_c = st.columns([1.4, 1.0, 1.0])
        with col_a:
            selected_symbol = st.selectbox(
                "Asset",
                list(SUPPORTED_TICKERS.keys()),
                format_func=lambda symbol: f"{symbol} - {SUPPORTED_TICKERS[symbol]}",
                key="binary_demo_asset",
            )
        with col_b:
            epochs = st.slider("ContextualQNN epochs", 1, 100, 20, key="binary_demo_epochs")
        with col_c:
            max_samples = st.slider("Recent samples", 64, 512, 128, step=64, key="binary_demo_samples")
        run_clicked = st.button("Run binary prediction", type="primary")

        @st.cache_data(show_spinner=False, ttl=3600)
        def _cached_prediction(
            symbol: str,
            epochs_value: int,
            sample_count: int,
            cache_version: str = "interactive_contextual_v3",
        ):
            _ = cache_version
            return run_quick_prediction(symbol=symbol, epochs=epochs_value, max_samples=sample_count)

        if run_clicked:
            with st.spinner("Preparing prices and running ContextualQNN..."):
                raw_result = _cached_prediction(
                    selected_symbol,
                    epochs,
                    max_samples,
                    "interactive_contextual_v3",
                )
                prediction, recent_prices, comparison_frame = _normalize_binary_prediction_output(raw_result)
                resolved_prediction = _resolve_binary_prediction_fields(
                    prediction,
                    recent_prices,
                    comparison_frame,
                )

            if resolved_prediction["data_source"] != "yfinance":
                st.warning(
                    "yfinance was not available, so this run used deterministic sample data. "
                    "Use it to test the interface only; do not report it as a market-data result."
                )
            else:
                st.success("Data source: yfinance live download.")

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Last close", f"{resolved_prediction['last_close']:.2f}", help=f"Date: {resolved_prediction['last_date']}")
            m2.metric("Naive next close", f"{resolved_prediction['naive_next_close']:.2f}")
            m3.metric("Predicted next close", f"{resolved_prediction['contextual_predicted_close']:.2f}")
            m4.metric("P(up)", f"{resolved_prediction['contextual_probability_up']:.3f}")
            m5.metric("Holdout accuracy", f"{resolved_prediction['holdout_accuracy']:.3f}")

            st.write(
                f"Latest binary context `{resolved_prediction['latest_context']}` -> "
                f"ContextualQNN direction: **{resolved_prediction['contextual_direction']}**. "
                f"Holdout F1: `{resolved_prediction['holdout_f1']:.3f}`, "
                f"RMSE: `{resolved_prediction['holdout_rmse']:.3f}`, "
                f"MAE: `{resolved_prediction['holdout_mae']:.3f}` "
                f"({resolved_prediction['train_samples']} train / {resolved_prediction['test_samples']} test samples)."
            )
            if not comparison_frame.empty and "Date" in comparison_frame.columns:
                st.line_chart(comparison_frame.set_index("Date"))
                with st.expander("Holdout comparison table"):
                    st.dataframe(comparison_frame, use_container_width=True, hide_index=True)
            st.markdown("**Recent market curve**")
            st.line_chart(recent_prices.set_index("Date")["Close"])
        else:
            st.info("Choose a ticker, set the sample size, then press Run binary prediction.")

    with d4_demo:
        st.subheader("d=4 Return-Regime ContextualQNN")
        col_a, col_b, col_c = st.columns([1.3, 1.0, 1.0])
        with col_a:
            selected_symbol_d4 = st.selectbox(
                "Asset for d=4 model",
                list(SUPPORTED_TICKERS.keys()),
                format_func=lambda symbol: f"{symbol} - {SUPPORTED_TICKERS[symbol]}",
                key="d4_demo_asset",
            )
        with col_b:
            epochs_d4 = st.slider("d=4 epochs", 10, 240, 80, step=10, key="d4_demo_epochs")
        with col_c:
            max_samples_d4 = st.slider("d=4 recent samples", 64, 320, 160, step=32, key="d4_demo_samples")
        run_multilevel_clicked = st.button("Run d=4 prediction", type="primary")

        @st.cache_data(show_spinner=False, ttl=3600)
        def _cached_multilevel_prediction(
            symbol: str,
            epochs_value: int,
            sample_count: int,
            cache_version: str = "advanced_contextual_v4",
        ):
            _ = cache_version
            return run_multilevel_quick_prediction(
                symbol=symbol,
                epochs=epochs_value,
                max_samples=sample_count,
            )

        if run_multilevel_clicked:
            with st.spinner("Preparing prices and running d=4 ContextualQNN..."):
                raw_result = _cached_multilevel_prediction(
                    selected_symbol_d4, epochs_d4, max_samples_d4, "advanced_contextual_v4"
                )
                prediction_d4, recent_prices_d4, probability_frame, comparison_frame_d4 = _normalize_multilevel_prediction_output(
                    raw_result
                )
                resolved_prediction_d4 = _resolve_multilevel_prediction_fields(
                    prediction_d4,
                    recent_prices_d4,
                    probability_frame,
                )

            if resolved_prediction_d4["data_source"] != "yfinance":
                st.warning(
                    "This run used deterministic sample data because yfinance was not available. "
                    "Treat it as an interface check only."
                )
            else:
                st.success("Data source: yfinance live download.")

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Last close", f"{resolved_prediction_d4['last_close']:.2f}", help=f"Date: {resolved_prediction_d4['last_date']}")
            m2.metric("Previous close", f"{resolved_prediction_d4['previous_close']:.2f}")
            m3.metric("Predicted bucket", str(resolved_prediction_d4["predicted_bucket"]))
            m4.metric("Predicted next close", f"{resolved_prediction_d4['predicted_next_close']:.2f}")
            m5.metric("Holdout accuracy", f"{resolved_prediction_d4['holdout_accuracy']:.3f}")

            st.write(
                f"Latest context `{resolved_prediction_d4['latest_context']}` -> predicted next-return bucket "
                f"`{resolved_prediction_d4['predicted_bucket']}` ({resolved_prediction_d4['predicted_bucket_label']})."
            )
            if not probability_frame.empty and {"label", "probability"}.issubset(probability_frame.columns):
                st.bar_chart(probability_frame.set_index("label")[["probability"]])
            if not comparison_frame_d4.empty and "Date" in comparison_frame_d4.columns:
                st.line_chart(comparison_frame_d4.set_index("Date"))
                with st.expander("Holdout comparison table"):
                    st.dataframe(comparison_frame_d4, use_container_width=True, hide_index=True)
            st.markdown("**Recent market curve**")
            st.line_chart(recent_prices_d4.set_index("Date")["Close"])
        else:
            st.info("Choose a ticker, set the sample size, then press Run d=4 prediction.")


with papers_tab:
    st.header("Paper Tracker")
    st.write("Progress notes and paper coverage are shown together for presentation preparation.")
    st.subheader("Paper Alignment")
    st.dataframe(
        TRACKS[["Reference", "Models", "Task", "Presentation role"]],
        use_container_width=True,
        hide_index=True,
    )
    for doc in [DOCS / "progress_log.md", DOCS / "paper_queue.md", DOCS / "related_papers.md"]:
        with st.expander(doc.name, expanded=doc.name == "progress_log.md"):
            st.markdown(_read_doc(doc))


with docs_tab:
    st.header("Docs & Runbook")
    st.write("Use these commands from the repository root.")
    st.code(
        "$env:PYTHONPATH='src;.'\n"
        ".\\.venv\\Scripts\\python.exe -m qsp.experiments.run_bidirectional_direction "
        "--symbols AAPL MSFT GOOGL NVDA --epochs 120 --max-rows 1200 "
        "--output-dir output\\bidirectional_direction\n"
        ".\\.venv\\Scripts\\python.exe scripts\\run_checks.py\n"
        ".\\.venv\\Scripts\\python.exe -m pytest -q\n"
        ".\\.venv\\Scripts\\python.exe -m streamlit run app\\streamlit_app.py",
        language="powershell",
    )
    st.subheader("Local Documents")
    docs = {
        "Bidirectional direction prediction": DOCS / "bidirectional_direction_prediction.md",
        "Progress log": DOCS / "progress_log.md",
        "Paper queue": DOCS / "paper_queue.md",
        "Local testing guide": DOCS / "local_testing_guide.md",
        "Related papers": DOCS / "related_papers.md",
    }
    for title, path in docs.items():
        status = "available" if path.exists() else "missing"
        st.markdown(f"- **{title}**: `{_relative(path)}` ({status})")
    st.subheader("External References")
    st.markdown("- [HQNN-FSP: A Hybrid Classical-Quantum Neural Network for Regression-Based Financial Stock Market Prediction](https://arxiv.org/abs/2503.15403)")
    st.markdown("- [Contextual Quantum Neural Networks for Stock Price Prediction](https://www.nature.com/articles/s41598-025-34413-5)")
    st.markdown("- [Quantum inspired qubit qutrit neural networks for real time financial forecasting](https://www.nature.com/articles/s41598-025-09475-0)")
    st.markdown("- [BLS-QLSTM: a novel hybrid quantum neural network for stock index forecasting](https://www.nature.com/articles/s41599-025-05348-z)")
