import yfinance as yf
import pandas as pd
import ta

def download_data(ticker, period, interval):
    """Download historical market data."""
    data = yf.download(ticker, period=period, interval=interval)
    data = data.dropna()  # Clean missing values
    return data

def add_features(data):
    """Add technical indicators to the data."""
    # Simple Moving Average (SMA)
    data['SMA_20'] = data['Close'].rolling(window=20).mean()

    # Relative Strength Index (RSI)
    rsi_indicator = ta.momentum.RSIIndicator(data['Close'], window=14)
    rsi_values = rsi_indicator.rsi()

    # Ensure RSI is flattened to a 1D series
    if len(rsi_values.shape) > 1:
        rsi_values = rsi_values.squeeze()

    data['RSI'] = rsi_values

    # Moving Average Convergence Divergence (MACD)
    macd_indicator = ta.trend.MACD(data['Close'])
    macd_values = macd_indicator.macd()

    # Ensure MACD is flattened to a 1D series
    if len(macd_values.shape) > 1:
        macd_values = macd_values.squeeze()

    data['MACD'] = macd_values

    # Drop rows with missing values after adding indicators
    data = data.dropna()
    return data

def save_data(data, path):
    """Save data to a CSV file."""
    data.to_csv(path, index=True)

def load_data(path):
    """Load data from a CSV file."""
    return pd.read_csv(path, index_col=0)
