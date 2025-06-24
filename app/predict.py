import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from .model import build_model, prepare_data
import traceback
import pandas as pd

def get_stock_data(symbol, start, end):
    import yfinance as yf
    import os

    print(f"Fetching data for {symbol} from {start} to {end}")
    df = yf.download(symbol, start=start, end=end)

    if df.empty:
        print("No data found.")
        return None

    # Save to CSV in a data/ folder (create it if it doesn't exist)
    os.makedirs("data", exist_ok=True)
    csv_path = f"data/{symbol}_{start}_{end}.csv"
    df.to_csv(csv_path)
    print(f"Saved data to {csv_path}")

    return df['Close'].values
def clean_csv(file_path, output_path=None):
    # Read all rows without interpreting headers
    df_raw = pd.read_csv(file_path, header=None)

    # Use the first row as header
    df_raw.columns = df_raw.iloc[0]
    
    # Drop the first 3 rows (0 = header, 1 and 2 = Ticker, Date)
    df_cleaned = df_raw.iloc[3:].reset_index(drop=True)

    # Save to new file or overwrite original
    if not output_path:
        output_path = file_path

    df_cleaned.to_csv(output_path, index=False)
    print(f"Cleaned CSV saved to: {output_path}")

    return df_cleaned

def clean_csv(file_path, output_path=None):
    # Read all rows without interpreting headers
    df_raw = pd.read_csv(file_path, header=None)

    # Use the first row as header
    df_raw.columns = df_raw.iloc[0]
    
    # Drop the first 3 rows (0 = header, 1 and 2 = Ticker, Date)
    df_cleaned = df_raw.iloc[3:].reset_index(drop=True)

    # Save to new file or overwrite original
    if not output_path:
        output_path = file_path

    df_cleaned.to_csv(output_path, index=False)
    print(f"Cleaned CSV saved to: {output_path}")

    return df_cleaned

def predict_next_7_days(symbol):
    try:
        print(f"Received prediction request for {symbol}")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)

        prices = get_stock_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        if prices is None or len(prices) < 100:
            print("Error: Not enough data or invalid symbol.")
            return {"error": "Not enough data or invalid symbol"}

        print(f"Fetched {len(prices)} closing prices for {symbol}")
        X, y, scaler = prepare_data(prices)
        print("Prepared data, starting model training...")

        model = build_model((X.shape[1], 1))
        model.fit(X, y, epochs=1, batch_size=32, verbose=1)
        print("Model training complete. Predicting next 7 days...")

        last_window = prices[-60:].reshape(-1, 1)
        scaled_window = scaler.transform(last_window)
        predictions = []

        for _ in range(7):
            X_input = scaled_window[-60:].reshape(1, 60, 1)  # Proper LSTM input shape
            pred_scaled = model.predict(X_input, verbose=0)
            pred = scaler.inverse_transform(pred_scaled)[0][0]
            predictions.append(float(pred))

            # Update the window
            new_scaled = scaler.transform(np.array([[pred]]))
            scaled_window = np.append(scaled_window, new_scaled)[-60:]

        print("Prediction complete.")
        return {"symbol": symbol, "predictions": predictions}

    except Exception as e:
        print("ERROR DURING PREDICTION:")
        traceback.print_exc()
        return {"error": str(e)}

