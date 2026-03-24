# Stock Predictor Streamlit App

This project predicts next-day stock price using a Bidirectional LSTM model and visualizes actual vs predicted trends in Streamlit.

## How it works
1. **Data fetch**
   - On `Predict`, the app loads `data/<SYMBOL>.csv` if available.
   - If missing/invalid, it fetches full daily history from Alpha Vantage.
2. **Indicator creation**
   - The app computes technical features: `MA10`, `MA50`, `EMA20` from `Close`.
   - Training/inference use feature order: `Close`, `MA10`, `MA50`, `EMA20`.
3. **LSTM training/loading**
   - It loads the latest symbol model from `model/` when available.
   - If model is older than 1 day or missing, it retrains and saves new `.h5` + scaler `.pkl`.
4. **Prediction + RMSE**
   - It predicts historical sequence outputs and one next-day price.
   - It reports RMSE by comparing actual vs predicted historical close prices.

## Run locally
```powershell
cd "C:\Users\user\OneDrive\Desktop\Manikanta\Stock_Predictor_Streamlit"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run stock_app.py
```

## 2-minute demo script (interview)
1. **0:00–0:20 – Problem statement**
   - “This app predicts next-day stock price using historical OHLCV data + moving average indicators.”
2. **0:20–0:40 – Setup**
   - Run Streamlit app and open the browser page.
3. **0:40–1:05 – User flow**
   - Select a symbol (e.g., AAPL) and click **Predict**.
4. **1:05–1:30 – Explain outputs**
   - Show next-day predicted value and RMSE.
   - Explain chart: blue = actual, red dashed = model prediction for recent 200 days.
5. **1:30–2:00 – Limitations + next steps**
   - Limitations: API rate limits, no sentiment/news features, no backtesting module.
   - Next steps: add train/validation split dashboard, hyperparameter tuning, and model comparison.

## Project structure
- `stock_app.py` - main Streamlit app, data processing, model training/loading, prediction
- `requirements.txt` - Python dependencies
- `data/` - cached stock CSV files (generated)
- `model/` - trained model and scaler artifacts (generated)
