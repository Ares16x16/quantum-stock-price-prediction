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
FULL_OUTPUT = ROOT / "output" / "qnn_full_aapl"
DIAGNOSTIC_OUTPUT = ROOT / "output" / "qnn_diagnostic_aapl"
DOCS = ROOT / "docs"


st.set_page_config(
    page_title="Quantum Stock Price Prediction",
    page_icon="Q",
    layout="wide",
)

st.title("Quantum-Enhanced Neural Networks for Financial Price Prediction")
st.caption("Artifact-driven dashboard for COMP7705 FYP progress monitoring.")

tabs = st.tabs(
    [
        "Model Comparison",
        "All Results",
        "Circuit Viewer",
        "Interactive Prediction",
        "Advanced ContextualQNN",
        "Saved AAPL Results",
        "Paper Tracker",
        "Docs & Links",
    ]
)


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        st.warning(f"Missing artifact: {path.relative_to(ROOT)}")
        return None
    return pd.read_csv(path)


def _dashboard_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Hide placeholder teammate rows in the public dashboard."""

    cleaned = frame.copy()
    if "Model name" in cleaned.columns:
        teammate_mask = cleaned["Model name"].astype(str).str.contains(
            "teammate|QLSTM", case=False, na=False
        )
        cleaned = cleaned.loc[~teammate_mask]

    for metric in ["RMSE", "MAE", "Directional Accuracy"]:
        if metric in cleaned.columns:
            cleaned = cleaned.loc[cleaned[metric].astype(str).str.upper() != "TBD"]

    return cleaned.reset_index(drop=True)


def _show_image(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Missing image: {path.relative_to(ROOT)}")


with tabs[0]:
    st.header("Model Comparison")
    st.write(
        "This page uses saved experiment outputs only. It does not launch expensive Qiskit training."
    )
    result = _read_csv(FULL_OUTPUT / "result_table.csv")
    if result is not None:
        result = _dashboard_table(result)
        st.dataframe(result, use_container_width=True)
        numeric = result.copy()
        for col in ["RMSE", "MAE", "Directional Accuracy"]:
            numeric[col] = pd.to_numeric(numeric[col], errors="coerce")
        st.bar_chart(numeric.set_index("Model name")[["RMSE", "MAE"]])

    summary = FULL_OUTPUT / "full_run_summary.md"
    if summary.exists():
        with st.expander("Full run summary"):
            st.markdown(summary.read_text(encoding="utf-8"))

with tabs[1]:
    st.header("All Results")
    st.write("Consolidated outputs from the current repository. These tables are saved artifacts, not live training jobs.")
    result_sources = {
        "HQNN-FSP CustomQNN": FULL_OUTPUT / "result_table.csv",
        "ContextualQNN AAPL": ROOT / "output" / "contextual_qnn" / "AAPL_result_table.csv",
        "ContextualQNN two-asset QMTL": ROOT / "output" / "contextual_qnn" / "qmtl_two_asset_result_table.csv",
        "ContextualQNN d=4 AAPL": ROOT / "output" / "contextual_qnn_multilevel" / "AAPL_result_table_d4.csv",
        "ANN / QQBN / QQTN": ROOT / "output" / "quantum_inspired" / "AAPL_result_table.csv",
    }
    loaded_tables = []
    for label, path in result_sources.items():
        frame = _read_csv(path)
        if frame is not None:
            frame = _dashboard_table(frame)
            st.subheader(label)
            st.dataframe(frame, use_container_width=True)
            normalized = frame.copy()
            normalized.insert(0, "source", label)
            loaded_tables.append(normalized)

    if loaded_tables:
        st.subheader("Combined view")
        st.dataframe(pd.concat(loaded_tables, ignore_index=True, sort=False), use_container_width=True)

with tabs[2]:
    st.header("Circuit Viewer")
    col1, col2 = st.columns(2)
    with col1:
        _show_image(FULL_OUTPUT / "original_custom_qnn.png", "Original preserved CustomQNN circuit")
    with col2:
        _show_image(FULL_OUTPUT / "custom_trainable_qnn.png", "Parameterized trainable CustomQNN circuit")

with tabs[3]:
    st.header("Interactive Prediction")
    st.write(
        "Select an asset and press Run prediction. The page downloads recent prices when yfinance is available, "
        "then runs the lightweight ContextualQNN direction model. It does not retrain the expensive CustomQNN."
    )
    col_a, col_b, col_c = st.columns([1.4, 1.0, 1.0])
    with col_a:
        selected_symbol = st.selectbox(
            "Asset",
            list(SUPPORTED_TICKERS.keys()),
            format_func=lambda symbol: f"{symbol} - {SUPPORTED_TICKERS[symbol]}",
        )
    with col_b:
        epochs = st.slider("ContextualQNN epochs", 1, 100, 20)
    with col_c:
        max_samples = st.slider("Recent samples", 64, 512, 128, step=64)
    run_clicked = st.button("Run prediction", type="primary")

    @st.cache_data(show_spinner=False, ttl=3600)
    def _cached_prediction(symbol: str, epochs_value: int, sample_count: int):
        return run_quick_prediction(symbol=symbol, epochs=epochs_value, max_samples=sample_count)

    if run_clicked:
        with st.spinner("Preparing prices and running ContextualQNN..."):
            prediction, recent_prices = _cached_prediction(selected_symbol, epochs, max_samples)

        if prediction.data_source != "yfinance":
            st.warning(
                "yfinance was not available, so this run used deterministic sample data. "
                "Use it to test the interface only; do not report it as a market-data result."
            )
        else:
            st.success("Data source: yfinance live download.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Last close", f"{prediction.last_close:.2f}", help=f"Date: {prediction.last_date}")
        m2.metric("Naive next close", f"{prediction.naive_next_close:.2f}")
        m3.metric("P(up)", f"{prediction.contextual_probability_up:.3f}")
        m4.metric("Holdout accuracy", f"{prediction.holdout_accuracy:.3f}")

        st.subheader(f"{prediction.symbol} prediction")
        st.write(
            f"Latest binary context `{prediction.latest_context}` -> "
            f"ContextualQNN direction: **{prediction.contextual_direction}**. "
            f"Holdout F1: `{prediction.holdout_f1:.3f}` "
            f"({prediction.train_samples} train / {prediction.test_samples} test samples)."
        )
        st.caption(
            f"Training method: binary return labels from recent closes, 80/20 time split, "
            f"lightweight statevector ContextualQNN, SPSA-style optimization, data source `{prediction.data_source}`."
        )
        st.line_chart(recent_prices.set_index("Date")["Close"])
        st.caption(
            "This panel follows the Contextual QNN direction-prediction setup with binary return quantization. "
            "It is a demo, not an investment signal."
        )
    else:
        st.info("Choose a ticker, set the sample size, then press Run prediction.")

with tabs[4]:
    st.header("Advanced ContextualQNN")
    st.write(
        "This tab runs the higher-resolution ContextualQNN path with density-based return buckets. "
        "It is the next paper-aligned step after the binary d=2 version."
    )
    col_a, col_b, col_c = st.columns([1.3, 1.0, 1.0])
    with col_a:
        selected_symbol_d4 = st.selectbox(
            "Asset for d=4 model",
            list(SUPPORTED_TICKERS.keys()),
            format_func=lambda symbol: f"{symbol} - {SUPPORTED_TICKERS[symbol]}",
            key="d4_asset",
        )
    with col_b:
        epochs_d4 = st.slider("d=4 epochs", 10, 240, 80, step=10)
    with col_c:
        max_samples_d4 = st.slider("d=4 recent samples", 64, 320, 160, step=32)
    run_multilevel_clicked = st.button("Run d=4 prediction", type="primary")

    @st.cache_data(show_spinner=False, ttl=3600)
    def _cached_multilevel_prediction(symbol: str, epochs_value: int, sample_count: int):
        return run_multilevel_quick_prediction(
            symbol=symbol,
            epochs=epochs_value,
            max_samples=sample_count,
        )

    saved_d4 = _read_csv(ROOT / "output" / "contextual_qnn_multilevel" / "AAPL_result_table_d4.csv")
    if saved_d4 is not None:
        st.caption("Latest saved d=4 artifact")
        st.dataframe(_dashboard_table(saved_d4), use_container_width=True)

    if run_multilevel_clicked:
        with st.spinner("Preparing prices and running d=4 ContextualQNN..."):
            prediction_d4, recent_prices_d4, probability_frame = _cached_multilevel_prediction(
                selected_symbol_d4, epochs_d4, max_samples_d4
            )

        if prediction_d4.data_source != "yfinance":
            st.warning(
                "This run used deterministic sample data because yfinance was not available. "
                "Treat it as an interface check only."
            )
        else:
            st.success("Data source: yfinance live download.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Last close", f"{prediction_d4.last_close:.2f}", help=f"Date: {prediction_d4.last_date}")
        m2.metric("Previous close", f"{prediction_d4.previous_close:.2f}")
        m3.metric("Predicted bucket", str(prediction_d4.predicted_bucket))
        m4.metric("Holdout accuracy", f"{prediction_d4.holdout_accuracy:.3f}")

        st.subheader(f"{prediction_d4.symbol} d=4 regime prediction")
        st.write(
            f"Latest context `{prediction_d4.latest_context}` -> predicted next-return bucket "
            f"`{prediction_d4.predicted_bucket}` ({prediction_d4.predicted_bucket_label})."
        )
        st.caption(
            f"Training method: density-based return quantization with d=4, context length T=2, "
            f"80/20 time split, lightweight statevector ContextualQNN, SPSA-style optimization, "
            f"data source `{prediction_d4.data_source}`."
        )
        prob_chart = probability_frame.set_index("label")[["probability"]]
        st.bar_chart(prob_chart)
        st.line_chart(recent_prices_d4.set_index("Date")["Close"])
        st.caption(
            "This panel predicts the return regime rather than the exact next close. "
            "It is the higher-resolution contextual path that can still run locally on CPU."
        )
    else:
        st.info("Choose a ticker for the d=4 model, then press Run d=4 prediction.")

with tabs[5]:
    st.header("Saved AAPL Results")
    model = st.selectbox(
        "Saved prediction plot",
        ["lstm", "custom_qnn", "hybrid_qnn1"],
        format_func={
            "lstm": "Classical LSTM",
            "custom_qnn": "Standalone CustomQNN",
            "hybrid_qnn1": "HybridQNN1",
        }.get,
    )
    plot_root = FULL_OUTPUT if model == "lstm" else DIAGNOSTIC_OUTPUT
    _show_image(plot_root / f"{model}_actual_vs_predicted.png", f"{model} actual vs predicted")
    _show_image(plot_root / f"{model}_loss.png", f"{model} loss curve")
    if model in {"custom_qnn", "hybrid_qnn1"}:
        st.warning(
            "The nearly flat prediction line is a genuine undertraining result, not a broken chart. "
            "The preserved QNN regressors were run in a smaller diagnostic setting because full multi-epoch Qiskit training is expensive on CPU."
        )
    else:
        st.caption("The LSTM plot is taken from the full AAPL benchmark run.")

with tabs[6]:
    st.header("Paper Reproduction Tracker")
    queue = DOCS / "paper_queue.md"
    progress = DOCS / "progress_log.md"
    for doc in [progress, queue]:
        if doc.exists():
            with st.expander(doc.name, expanded=doc.name == "progress_log.md"):
                st.markdown(doc.read_text(encoding="utf-8"))

    st.subheader("Paper alignment")
    st.markdown(
        """
        - `HQNN-FSP: A Hybrid Classical-Quantum Neural Network for Regression-Based Financial Stock Market Prediction`: preserved Qiskit circuit plus trainable `EstimatorQNN` pipeline.
        - `Contextual Quantum Neural Networks for Stock Price Prediction`: binary context model, d=4 multilevel extension, and two-asset QMTL run.
        - `Quantum Inspired Qubit Qutrit Neural Networks for Real Time Financial Forecasting`: ANN, QQBN, and QQTN direction-classification run.
        """
    )

with tabs[7]:
    st.header("Docs & Links")
    docs = {
        "Progress log": DOCS / "progress_log.md",
        "Paper queue": DOCS / "paper_queue.md",
        "Local testing guide": DOCS / "local_testing_guide.md",
    }
    for title, path in docs.items():
        st.markdown(f"- **{title}**: `{path.relative_to(ROOT)}`")
    st.markdown("### External references")
    st.markdown("- [HQNN-FSP: A Hybrid Classical-Quantum Neural Network for Regression-Based Financial Stock Market Prediction](https://arxiv.org/abs/2503.15403)")
    st.markdown("- [Contextual Quantum Neural Networks for Stock Price Prediction](https://www.nature.com/articles/s41598-025-34413-5)")
    st.markdown("- [Quantum inspired qubit qutrit neural networks for real time financial forecasting](https://www.nature.com/articles/s41598-025-09475-0)")
