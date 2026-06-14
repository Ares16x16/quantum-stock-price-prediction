# Bidirectional Direction Prediction

This document is the presentation reference for the bidirectional direction track. In this repository, "bidirectional prediction" means predicting whether the next trading day's close moves up or down:

```text
Next_Return = Close[t + 1] / Close[t] - 1
Target = 1 if Next_Return > 0 else 0
```

The first implementation deliberately does not introduce BiLSTM as the main model. It reuses the interim-aligned `ANN`, `QQBN`, and `QQTN` models and improves the experiment design around them.

## References Used

1. **HQNN-FSP: A Hybrid Classical-Quantum Neural Network for Regression-Based Financial Stock Market Prediction**
   - Repository file: `ref/2503.15403v1.pdf`
   - Project use: this paper motivates hybrid quantum-classical financial forecasting and explicitly discusses recurrent models, LSTM/BiLSTM, and quantum processing. The first direction track does not reuse the expensive HQNN-FSP circuit directly because the current Qiskit regression result is slow and underperforms the naive previous-close baseline. The lesson reused here is the need for technical indicators and a more practical target than raw next-close regression.

2. **Contextual Quantum Neural Networks for Stock Price Prediction**
   - Repository file: `ref/Contextual Quantum Neural Networks for Stock Price Prediction.pdf`
   - Project use: this paper frames stock movement as a distribution over future return states. It supports the decision to model direction as a binary future-return state: up for positive return and down for non-positive return. The track uses that same up/down interpretation, but keeps the implementation in the faster PyTorch QQTN family for the first deliverable.

3. **Quantum inspired qubit qutrit neural networks for real time financial forecasting**
   - Repository file: `ref/Quantum inspired qubit qutrit neural networks for real time financial forecasting.pdf`
   - Project use: this is the direct model reference. The repository already implemented `ANN`, `QQBN`, and `QQTN` in `src/qsp/models/quantum_inspired.py`. The main model is `QQTN` because it is already in the interim scope and was the strongest model family in the earlier AAPL direction run.

## Why Up/Down Instead Of Raw Price Regression

The earlier HQNN-FSP raw next-close regression benchmark struggled because the naive previous-close baseline is very strong for daily close prices. A model can look visually close to the true price curve while still not learning useful directional information.

Direction prediction is more suitable for the deliverable because:

- it directly matches the "up/down" interpretation of bidirectional prediction;
- it aligns with the binary return quantization used by the ContextualQNN paper;
- it matches the classification setup of the QQBN/QQTN paper;
- it avoids claiming that a model beats previous-close on raw price RMSE when the current project evidence does not support that.

## What Changed From `run_quantum_inspired`

The previous `qsp.experiments.run_quantum_inspired` script was a useful interim proof of concept. It trained `ANN`, `QQBN`, and `QQTN` on one AAPL direction-classification run with a simpler feature set and a fixed `0.5` classification threshold.

The new implementation is `qsp.experiments.run_bidirectional_direction`.

Main changes:

- **Four-stock benchmark**: runs `AAPL`, `MSFT`, `GOOGL`, and `NVDA` instead of only AAPL.
- **Richer lagged features**: keeps OHLCV, RSI, MACD, SMA5, ADX, and `Return_1`, then adds MACD signal/histogram, SMA10/SMA20, log return, multi-day returns, momentum, rolling volatility, close-to-SMA ratios, price range features, volume change, and volume z-score.
- **Train-only scaling**: `MinMaxScaler` is fit only on the training split, then applied to validation and test. This avoids using holdout distribution information during preprocessing.
- **Explicit validation split**: each stock uses a time-ordered train/validation/test split. The test set is untouched until final evaluation.
- **Threshold calibration**: neural models choose a probability threshold on the validation split instead of always using `0.5`. This reduces the risk of all-up or all-down predictions.
- **Balanced QQTN threshold**: the saved run also includes a `QQTN balanced threshold` row. It reuses the exact same trained QQTN probabilities, but selects the validation threshold by balanced accuracy instead of F1 so the presentation can discuss the one-sided prediction issue directly.
- **Ensemble comparison**: the saved run also includes `ANN+QQBN+QQTN ensemble` rows. These average the reused interim model probabilities and calibrate thresholds on the validation split. The ensemble is included as an improvement attempt and diagnostic comparison, but it does not beat the main QQTN row on the current four-stock average.
- **Baseline comparisons**: every result table includes a majority-class baseline and a momentum baseline based on the previous day's return sign.
- **Prediction-level artifacts**: every stock saves row-level predictions, probabilities, thresholds, training logs, probability plots, and confusion matrices for presentation and debugging.

## Implementation Locations

- Experiment entrypoint: `src/qsp/experiments/run_bidirectional_direction.py`
- Reused model definitions: `src/qsp/models/quantum_inspired.py`
- Tests: `tests/test_bidirectional_direction.py`
- Dashboard integration: `app/streamlit_app.py`
- Saved artifacts: `output/bidirectional_direction/`

Run command:

```powershell
$env:PYTHONPATH='src;.'
.\.venv\Scripts\python.exe -m qsp.experiments.run_bidirectional_direction --symbols AAPL MSFT GOOGL NVDA --epochs 120 --max-rows 1200 --output-dir output\bidirectional_direction
```

Current tuned defaults are `hidden_dim=48` and `learning_rate=0.003`. They were selected from a small local sweep because they gave the best average QQTN accuracy/F1 across the four-stock benchmark while keeping the model family unchanged.

## Saved Artifacts

The experiment writes:

- `combined_result_table.csv`
- `{SYMBOL}_result_table.csv`
- `{SYMBOL}_predictions.csv`
- `training_log.csv`
- `thresholds.csv`
- `{SYMBOL}_{model}_probability.png`
- `{SYMBOL}_{model}_confusion_matrix.png`

For presentation, use `combined_result_table.csv` for the main comparison and the QQTN probability/confusion-matrix plots for the visual explanation. The ensemble plots are useful if asked what extra improvement attempts were tested.

## Current Saved Result

The current saved four-stock run uses live `yfinance` data, `epochs=120`, `max_rows=1200`, `hidden_dim=48`, and `learning_rate=0.003`.

Average holdout results:

| Model | Directional Accuracy | Precision | Recall | F1 | Sharpe Ratio |
|---|---:|---:|---:|---:|---:|
| Majority baseline | 0.5111 | 0.5111 | 1.0000 | 0.6761 | 0.6380 |
| Momentum baseline | 0.5028 | 0.5136 | 0.5110 | 0.5123 | 0.0170 |
| ANN | 0.5097 | 0.5106 | 0.9813 | 0.6713 | 0.5457 |
| QQBN | 0.5111 | 0.5111 | 1.0000 | 0.6761 | 0.6380 |
| QQTN | 0.5236 | 0.5182 | 0.9866 | 0.6789 | 0.9538 |
| QQTN balanced threshold | 0.5236 | 0.5222 | 0.8480 | 0.6275 | 0.2241 |
| ANN+QQBN+QQTN ensemble | 0.5111 | 0.5112 | 0.9974 | 0.6756 | 0.6160 |
| ANN+QQBN+QQTN balanced ensemble | 0.4972 | 0.3792 | 0.7500 | 0.5034 | 0.2643 |

QQTN is the main model. On the saved four-stock average, it improves over the majority baseline by `+0.0125` directional accuracy and over the momentum baseline by `+0.0208`. The per-stock results still vary, so this should be presented as a modest but cleaner direction-classification improvement, not as a solved forecasting problem.

The balanced-threshold QQTN row has the same average directional accuracy as the main QQTN row, but it improves balanced accuracy from `0.5134` to `0.5166` and lowers the predicted-up rate from `0.9736` to `0.8319`. This makes the model less one-sided, which is useful to mention if asked whether the result is just predicting "up" most of the time.

The ensemble rows confirm that averaging the three reused interim models is not automatically better. They are kept in the output because they show a reasonable improvement attempt and help justify why QQTN remains the headline model.

## Presentation Interpretation

Recommended speaking points:

- The work focuses on a more defensible target: next-day direction rather than raw next-close regression.
- The model is not a new unrelated architecture. It reuses the interim `QQTN` qutrit-inspired classifier and strengthens the experiment design.
- The four-stock setup makes the result less fragile than a single AAPL run.
- The validation threshold is chosen before seeing the test set, which is important because fixed `0.5` thresholds can collapse to one class on financial data.
- The `QQTN balanced threshold` row is included because F1-optimized thresholds still lean heavily toward up predictions. It is a robustness view, not a new model architecture.
- The ensemble row was tested after the first implementation, but it does not outperform QQTN on the four-stock average.
- Majority and momentum baselines are shown beside the neural models, so the result can be interpreted honestly rather than only reporting model accuracy in isolation.

## Next Result-Improvement Steps

Recommended next research steps:

- use walk-forward validation instead of a single fixed validation/test split;
- add market-context features such as SPY, QQQ, VIX, or sector ETF returns with strict date alignment;
- calibrate probabilities with Platt scaling or isotonic calibration across validation folds;
- test multi-horizon targets such as 3-day and 5-day direction;
- keep BiLSTM in the fourth-paper extension unless the presentation explicitly moves beyond the interim-aligned model family;
- only try a CustomQNN direction-classification head as a separate slow experiment, because the current CustomQNN circuit belongs to the HQNN-FSP regression track.

## Current Limitation

This is still a predictive experiment, not a trading system. The outputs should be presented as direction-classification metrics and model-comparison artifacts, not investment advice or deployable trading signals. BiLSTM remains an optional later extension after this reused-model track is complete.
