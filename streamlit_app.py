import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

st.title("📈 Stock Price Prediction using LSTM")
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT):", "AAPL")

# Load model
@st.cache_resource
def load_trained_model():
    return load_model("stock_lstm_model.h5")

model = load_trained_model()

# Download data
data = yf.download(ticker, start="2015-01-01", end="2023-12-31")
if data.empty:
    st.error("Failed to fetch data. Please check the stock symbol and your connection.")
    st.stop()

st.subheader(f"Stock Price Data for {ticker}")
st.line_chart(data["Close"])

# Preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
close_data = data[["Close"]].values
scaled_data = scaler.fit_transform(close_data)

# Create sequences
X = []
for i in range(60, len(scaled_data)):
    X.append(scaled_data[i - 60:i])

X = np.array(X)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Predict
predicted_prices = model.predict(X)
predicted_prices = scaler.inverse_transform(predicted_prices)
actual_prices = scaler.inverse_transform(scaled_data[60:])

# Plot
st.subheader("📊 Actual vs Predicted Prices")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(actual_prices, label='Actual')
ax.plot(predicted_prices, label='Predicted')
ax.set_xlabel("Time")
ax.set_ylabel("Stock Price")
ax.set_title(f"{ticker} - Actual vs Predicted Closing Prices")
ax.legend()
st.pyplot(fig)
