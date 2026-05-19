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
        "Experiment Lab",
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
        "Classical LSTM GPU refined": ROOT / "output" / "lstm_gpu_refined_aapl" / "result_table.csv",
        "ContextualQNN AAPL": ROOT / "output" / "contextual_qnn" / "AAPL_result_table.csv",
        "ContextualQNN two-asset QMTL": ROOT / "output" / "contextual_qnn" / "qmtl_two_asset_result_table.csv",
        "ContextualQNN four-asset QMTL": ROOT / "output" / "contextual_qnn_multi_asset" / "qmtl_4_asset_result_table.csv",
        "ContextualQNN d=4 AAPL": ROOT / "output" / "contextual_qnn_multilevel" / "AAPL_result_table_d4.csv",
        "ANN / QQBN / QQTN": ROOT / "output" / "quantum_inspired" / "AAPL_result_table.csv",
        "Sequence hybrid experiment": ROOT / "output" / "sequence_hybrid_aapl" / "AAPL_result_table.csv",
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
    def _cached_prediction(
        symbol: str,
        epochs_value: int,
        sample_count: int,
        cache_version: str = "interactive_contextual_v2",
    ):
        _ = cache_version
        return run_quick_prediction(symbol=symbol, epochs=epochs_value, max_samples=sample_count)

    if run_clicked:
        with st.spinner("Preparing prices and running ContextualQNN..."):
            raw_result = _cached_prediction(
                selected_symbol,
                epochs,
                max_samples,
                "interactive_contextual_v2",
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

        st.subheader(f"{resolved_prediction['symbol']} prediction")
        st.write(
            f"Latest binary context `{resolved_prediction['latest_context']}` -> "
            f"ContextualQNN direction: **{resolved_prediction['contextual_direction']}**. "
            f"Holdout F1: `{resolved_prediction['holdout_f1']:.3f}`, "
            f"RMSE: `{resolved_prediction['holdout_rmse']:.3f}`, "
            f"MAE: `{resolved_prediction['holdout_mae']:.3f}` "
            f"({resolved_prediction['train_samples']} train / {resolved_prediction['test_samples']} test samples)."
        )
        st.caption(
            f"Training method: binary return labels from recent closes, 80/20 time split, "
            f"lightweight statevector ContextualQNN, SPSA-style optimization, data source `{resolved_prediction['data_source']}`."
        )

        if not comparison_frame.empty and "Date" in comparison_frame.columns:
            st.markdown("**Holdout backtest view**")
            st.line_chart(comparison_frame.set_index("Date"))
            with st.expander("Holdout comparison table"):
                st.dataframe(comparison_frame, use_container_width=True)
        else:
            st.info("The holdout comparison curve is unavailable for this cached result. Re-run the prediction to rebuild it.")
        st.markdown("**Recent market curve**")
        st.line_chart(recent_prices.set_index("Date")["Close"])
        st.caption(
            "The holdout chart compares actual next closes against the interactive model's implied next-close path and the naive previous-close baseline. "
            "This panel follows the Contextual QNN direction-prediction setup with binary return quantization. It is a demo, not an investment signal."
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
    def _cached_multilevel_prediction(
        symbol: str,
        epochs_value: int,
        sample_count: int,
        cache_version: str = "advanced_contextual_v3",
    ):
        _ = cache_version
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
            raw_result = _cached_multilevel_prediction(
                selected_symbol_d4, epochs_d4, max_samples_d4, "advanced_contextual_v3"
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

        st.subheader(f"{resolved_prediction_d4['symbol']} d=4 regime prediction")
        st.write(
            f"Latest context `{resolved_prediction_d4['latest_context']}` -> predicted next-return bucket "
            f"`{resolved_prediction_d4['predicted_bucket']}` ({resolved_prediction_d4['predicted_bucket_label']})."
        )
        st.caption(
            f"Training method: density-based return quantization with d=4, context length T=2, "
            f"80/20 time split, lightweight statevector ContextualQNN, SPSA-style optimization, "
            f"data source `{resolved_prediction_d4['data_source']}`."
        )
        st.write(
            f"Holdout RMSE: `{resolved_prediction_d4['holdout_rmse']:.3f}` | "
            f"Holdout MAE: `{resolved_prediction_d4['holdout_mae']:.3f}`"
        )
        if not probability_frame.empty and {"label", "probability"}.issubset(probability_frame.columns):
            prob_chart = probability_frame.set_index("label")[["probability"]]
            st.bar_chart(prob_chart)
        else:
            st.info("Probability bars are unavailable for this cached result. Re-run the prediction to refresh the full output.")

        if not comparison_frame_d4.empty and "Date" in comparison_frame_d4.columns:
            st.markdown("**Holdout backtest view**")
            st.line_chart(comparison_frame_d4.set_index("Date"))
            with st.expander("Holdout comparison table"):
                st.dataframe(comparison_frame_d4, use_container_width=True)
        else:
            st.info("The holdout comparison curve is unavailable for this cached result. Re-run the prediction to rebuild it.")
        st.markdown("**Recent market curve**")
        st.line_chart(recent_prices_d4.set_index("Date")["Close"])
        st.caption(
            "This panel predicts the return regime rather than the exact next close. "
            "The holdout chart converts the predicted return distribution into an implied next-close path for comparison against the actual series and the naive baseline."
        )
    else:
        st.info("Choose a ticker for the d=4 model, then press Run d=4 prediction.")

with tabs[5]:
    st.header("Experiment Lab")
    st.write(
        "This section tracks a stronger local experiment path that stays separate from the preserved HQNN-FSP circuit. "
        "It uses a GPU-friendly temporal encoder and a qutrit-inspired head, so it is suitable for heavier local training."
    )
    st.caption(
        "The current environment has CUDA PyTorch available, but it does not have qiskit-aer installed. "
        "That means the preserved EstimatorQNN path remains CPU-bound, while this experiment tab can use the GPU."
    )
    hybrid_result = _read_csv(ROOT / "output" / "sequence_hybrid_aapl" / "AAPL_result_table.csv")
    if hybrid_result is not None:
        st.dataframe(hybrid_result, use_container_width=True)
        model_choice = st.selectbox(
            "Experiment plot",
            ["BiLSTM baseline", "BiLSTM-QQTN hybrid"],
            key="experiment_lab_plot",
        )
        plot_key = model_choice.replace(" ", "_").lower()
        _show_image(
            ROOT / "output" / "sequence_hybrid_aapl" / f"{plot_key}_actual_vs_predicted.png",
            f"{model_choice} holdout curve",
        )
        _show_image(
            ROOT / "output" / "sequence_hybrid_aapl" / f"{plot_key}_loss.png",
            f"{model_choice} training loss",
        )
        st.caption(
            "These curves are implied next-close paths derived from direction probabilities rather than direct price regression. "
            "That is why the visual comparison can look smoother than the preserved Qiskit regression outputs."
        )
    else:
        st.info("Run the sequence hybrid experiment to populate this tab.")

with tabs[6]:
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
            "undertraining"
        )
    else:
        st.caption("The LSTM plot is taken from the full AAPL benchmark run.")

with tabs[7]:
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

with tabs[8]:
    st.header("Docs & Links")
    docs = {
        "Progress log": DOCS / "progress_log.md",
        "Paper queue": DOCS / "paper_queue.md",
        "Local testing guide": DOCS / "local_testing_guide.md",
        "Related papers": DOCS / "related_papers.md",
    }
    for title, path in docs.items():
        st.markdown(f"- **{title}**: `{path.relative_to(ROOT)}`")
    st.markdown("### External references")
    st.markdown("- [HQNN-FSP: A Hybrid Classical-Quantum Neural Network for Regression-Based Financial Stock Market Prediction](https://arxiv.org/abs/2503.15403)")
    st.markdown("- [Contextual Quantum Neural Networks for Stock Price Prediction](https://www.nature.com/articles/s41598-025-34413-5)")
    st.markdown("- [Quantum inspired qubit qutrit neural networks for real time financial forecasting](https://www.nature.com/articles/s41598-025-09475-0)")
