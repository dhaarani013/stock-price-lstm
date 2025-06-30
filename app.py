import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

ticker = "AAPL"  
data = yf.download(ticker, start="2015-01-01", end="2023-12-31")

print(data.head())

data["Close"].plot(title=f"{ticker} Closing Price")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.grid(True)
plt.show()
import numpy as np
from sklearn.preprocessing import MinMaxScaler

close_prices = data[['Close']]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)
X = []
y = []
sequence_length = 60
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i - sequence_length:i])
    y.append(scaled_data[i])
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1)) 
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=10, batch_size=32)
predicted_prices = model.predict(X)
predicted_prices = scaler.inverse_transform(predicted_prices)
actual_prices = scaler.inverse_transform(y)
plt.figure(figsize=(10, 5))
plt.plot(actual_prices, label='Actual Prices')
plt.plot(predicted_prices, label='Predicted Prices')
plt.title('LSTM Model - Actual vs Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Stock Price ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(actual_prices, label='Actual Prices')
plt.plot(predicted_prices, label='Predicted Prices')
plt.title('LSTM Model - Actual vs Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Stock Price ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show() 
model.save("stock_lstm_model.h5")

