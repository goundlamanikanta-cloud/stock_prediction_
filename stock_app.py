# ------------------------------------------------
# 📈 STREAMLIT STOCK PRICE PREDICTOR (PRO VERSION - CLEAN)
# ------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
import matplotlib.pyplot as plt
import math
import os
import time
import joblib
from datetime import datetime

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(page_title="📊 Stock Predictor", layout="centered")

st.title("📈 Stock Price Prediction using Bidirectional LSTM")
st.markdown("Predict next-day stock prices using real-time Alpha Vantage data and advanced deep learning models.")

API_KEY = "IUVCBWTPZWTKZOWI"  # Replace with your own key

# ------------------------------------------------
# FETCH STOCK DATA (Auto-Update + Smart Cache)
# ------------------------------------------------
@st.cache_data(ttl=86400)  # cache expires every 24 hours
def get_stock_data(symbol):
    csv_path = f"data/{symbol}.csv"

    if os.path.exists(csv_path):
        data = pd.read_csv(csv_path, index_col=0)

        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in data.columns for col in required_cols):
            return fetch_fresh_data(symbol, csv_path)

        # Add missing indicators if not present
        if "MA10" not in data.columns:
            data["MA10"] = data["Close"].rolling(window=10).mean()
        if "MA50" not in data.columns:
            data["MA50"] = data["Close"].rolling(window=50).mean()
        if "EMA20" not in data.columns:
            data["EMA20"] = data["Close"].ewm(span=20, adjust=False).mean()

        data = data.dropna()
        data.to_csv(csv_path)
        st.info(f"📁 Loaded and refreshed cached data for {symbol}")
        return data

    return fetch_fresh_data(symbol, csv_path)


# ------------------------------------------------
# HELPER FUNCTION – Fetch Fresh Data
# ------------------------------------------------
def fetch_fresh_data(symbol, csv_path):
    try:
        ts = TimeSeries(key=API_KEY, output_format='pandas')
        data, _ = ts.get_daily(symbol=symbol, outputsize='full')

        data = data.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        })
        data = data.sort_index(ascending=True)
        data["MA10"] = data["Close"].rolling(window=10).mean()
        data["MA50"] = data["Close"].rolling(window=50).mean()
        data["EMA20"] = data["Close"].ewm(span=20, adjust=False).mean()
        data = data.dropna()

        os.makedirs("data", exist_ok=True)
        data.to_csv(csv_path)
        st.success(f"✅ Downloaded and cached new data for {symbol}")
        return data
    except Exception as e:
        st.error(f"❌ Failed to fetch data: {e}")
        raise e


# ------------------------------------------------
# TRAIN AND SAVE MODEL
# ------------------------------------------------
def train_and_save_model(symbol, data):
    st.info(f"⚙️ Training Bidirectional LSTM model for {symbol}... Please wait ⏳")

    features = data[["Close", "MA10", "MA50", "EMA20"]].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(features)

    X, y = [], []
    seq_len = 30
    for i in range(seq_len, len(scaled_data)):
        X.append(scaled_data[i - seq_len:i])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)

    model = Sequential([
        Bidirectional(LSTM(60, return_sequences=True, input_shape=(X.shape[1], X.shape[2]))),
        Dropout(0.3),
        Bidirectional(LSTM(60, return_sequences=False)),
        Dropout(0.3),
        Dense(30, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    progress_bar = st.progress(0)
    for epoch in range(50):
        model.fit(X, y, epochs=1, batch_size=32, verbose=0)
        progress_bar.progress((epoch + 1) / 50)

    os.makedirs("model", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_path = f"model/{symbol}_model_{timestamp}.h5"
    scaler_path = f"model/{symbol}_scaler_{timestamp}.pkl"
    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    st.success(f"💾 Model & scaler for {symbol} saved successfully!")
    return model, scaler


# ------------------------------------------------
# LOAD OR TRAIN MODEL (AUTO REFRESH EVERY 1 DAY)
# ------------------------------------------------
def load_or_train_model(symbol, data):
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)

    model_files = [f for f in os.listdir(model_dir) if f.startswith(symbol) and f.endswith(".h5")]
    scaler_files = [f for f in os.listdir(model_dir) if f.startswith(symbol) and f.endswith(".pkl")]

    if model_files and scaler_files:
        latest_model = sorted(model_files)[-1]
        model_path = os.path.join(model_dir, latest_model)

        # Check if model older than 1 day (86400 seconds)
        file_age = time.time() - os.path.getmtime(model_path)
        if file_age > 86400:
            st.warning(f"♻️ Model for {symbol} is older than 1 day — retraining...")
            return train_and_save_model(symbol, data)

        latest_scaler = sorted(scaler_files)[-1]
        model = load_model(model_path)
        scaler = joblib.load(os.path.join(model_dir, latest_scaler))
        st.info(f"📁 Loaded model for {symbol}: {latest_model}")
        return model, scaler

    st.warning(f"⚙️ No saved model found for {symbol}. Training new one...")
    return train_and_save_model(symbol, data)


# ------------------------------------------------
# PREDICT FUNCTION
# ------------------------------------------------
def predict_next_day(data, model, scaler):
    features = data[["Close", "MA10", "MA50", "EMA20"]].values
    scaled_data = scaler.transform(features)

    X = []
    seq_len = 30
    for i in range(seq_len, len(scaled_data)):
        X.append(scaled_data[i - seq_len:i])
    X = np.array(X)

    preds = model.predict(X)
    preds = scaler.inverse_transform(np.concatenate((preds, np.zeros((preds.shape[0], scaled_data.shape[1] - 1))), axis=1))[:, 0]

    last_seq = scaled_data[-seq_len:]
    next_input = np.reshape(last_seq, (1, seq_len, scaled_data.shape[1]))
    next_day = model.predict(next_input)
    next_day = scaler.inverse_transform(np.concatenate((next_day, np.zeros((1, scaled_data.shape[1] - 1))), axis=1))[0][0]

    actual = scaler.inverse_transform(scaled_data[seq_len:])[:, 0]
    rmse = math.sqrt(mean_squared_error(actual, preds))
    return actual, preds, next_day, rmse


# ------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------
st.sidebar.header("⚙️ Configuration")
symbol = st.sidebar.selectbox(
    "Choose Stock Symbol:",
    ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"],
    index=0
)
auto_refresh = st.sidebar.checkbox("🔁 Auto Refresh every 30 seconds", value=False)

if st.button("Predict") or auto_refresh:
    try:
        data = get_stock_data(symbol)
        model, scaler = load_or_train_model(symbol, data)

        actual, predicted, next_day, rmse = predict_next_day(data, model, scaler)

        st.subheader(f"📈 Predicted Next-Day Price for {symbol}: **${next_day:.2f}**")
        st.write(f"🧮 RMSE (Prediction Error): **{rmse:.2f}**")

        # Plot graph
        st.write("### 📊 Actual vs Predicted Stock Prices")
        fig, ax = plt.subplots(figsize=(12, 6), dpi=120)
        ax.plot(actual[-200:], label="Actual Price", color="blue", linewidth=2)
        ax.plot(predicted[-200:], label="Predicted Price", color="red", linestyle="--", linewidth=2)
        ax.set_title(f"{symbol} - Last 200 Days (Actual vs Predicted)", fontsize=14)
        ax.legend()
        st.pyplot(fig)

        # Last 5 predictions table
        st.write("### 📋 Last 5 Predictions")
        preview_df = pd.DataFrame({
            "Actual Price": actual[-5:],
            "Predicted Price": predicted[-5:]
        })
        st.table(preview_df.round(2))

        if auto_refresh:
            st.info("Auto-refreshing every 30 seconds...")
            time.sleep(30)
            st.experimental_rerun()

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.info("Please verify the stock symbol or your API key.")
