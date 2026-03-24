# Instructions for Stock_Predictor_Streamlit

## Project overview
- This repo is a **single-file Streamlit app** centered in `stock_app.py`.
- Runtime flow is: UI event (`Predict`) -> data load/fetch -> model load/train -> inference -> chart/table render.
- Persistent artifacts are local and folder-based:
  - `data/<SYMBOL>.csv` for historical OHLCV + indicators
  - `model/<SYMBOL>_model_<timestamp>.h5` and `model/<SYMBOL>_scaler_<timestamp>.pkl`

## Core architecture and data flow
- `get_stock_data(symbol)` is the entry point for market data and is cached with `@st.cache_data(ttl=86400)`.
- If a CSV exists, it is validated for required columns and missing indicators are backfilled (`MA10`, `MA50`, `EMA20`).
- If cache is missing/invalid, `fetch_fresh_data()` pulls Alpha Vantage daily data, normalizes columns, computes indicators, and writes to `data/`.
- `load_or_train_model()` controls model lifecycle: loads latest model/scaler for symbol, retrains if model file is older than 1 day.
- `train_and_save_model()` trains a Bidirectional LSTM on 4 features (`Close`, `MA10`, `MA50`, `EMA20`) with sequence length 30.
- `predict_next_day()` returns `(actual, preds, next_day, rmse)` and uses scaler inverse-transform with zero-padding for non-target features.

## Local dev workflow (Windows PowerShell)
- Create venv: `py -m venv .venv`
- Activate: `.\.venv\Scripts\Activate.ps1`
- Install deps: `pip install -r requirements.txt`
- Run app: `streamlit run stock_app.py`
- No formal test suite exists in this repo; validate by running Streamlit and clicking `Predict` for one or more symbols.

## Project-specific coding conventions
- Keep feature engineering in sync everywhere: training and inference must use the same ordered columns:
  - `['Close', 'MA10', 'MA50', 'EMA20']`
- Preserve folder contracts:
  - `data/` for cached market data
  - `model/` for `.h5` + `.pkl` artifacts
- When modifying model persistence, keep **model and scaler pairing** consistent per symbol/timestamp.
- Prefer minimal, surgical edits in `stock_app.py`; this project is intentionally centralized, not split into packages.
- Keep Streamlit feedback style (`st.info`, `st.warning`, `st.success`, `st.error`) for long operations and failures.

## Integrations and dependencies
- External API: Alpha Vantage via `alpha_vantage.timeseries.TimeSeries` (`get_daily`, full output).
- ML stack: TensorFlow/Keras + scikit-learn `MinMaxScaler` + `joblib` for scaler persistence.
- Visualization/UI: Streamlit + Matplotlib.
- API key is currently hardcoded in `stock_app.py` as `API_KEY`; if refactoring, prefer env var fallback while preserving existing behavior.

## Common pitfalls for agents
- Do not change indicator names or ordering unless updating all dependent code paths.
- Avoid introducing incompatible model/scaler naming that breaks latest-file loading logic.
- Be careful with Streamlit rerun flow (`auto_refresh` + `st.experimental_rerun`) to avoid accidental loop behavior changes.
- Preserve existing CSV schema expectations (`Open`, `High`, `Low`, `Close`, `Volume`).
