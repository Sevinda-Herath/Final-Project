import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def get_stock_data(ticker, period='90d'):
    data = yf.download(ticker, period=period, interval='1d')
    return data[['Close']]

def preprocess_data(df, look_back=60):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])

    return np.array(X), np.array(y), scaler

def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_next_days(model, data, scaler, look_back=60, days=7):
    last_sequence = data[-look_back:]
    predictions = []

    for _ in range(days):
        input_seq = last_sequence.reshape(1, look_back, 1)
        pred = model.predict(input_seq)[0][0]
        predictions.append(pred)

        last_sequence = np.append(last_sequence[1:], pred)

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

def train_and_predict(ticker):
    df = get_stock_data(ticker)
    X, y, scaler = preprocess_data(df)

    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = build_model((X.shape[1], 1))
    model.fit(X, y, epochs=5, batch_size=16, verbose=0)

    predictions = predict_next_days(model, y, scaler)
    return predictions.tolist()
