import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump, load
from config import *
from utils import download_data, add_features, save_data, load_data

# Step 1: Download and Prepare Data
data = download_data(TICKER, PERIOD, INTERVAL)
data = add_features(data)
save_data(data, DATA_PATH)

# Step 2: Train the Random Forest Model
X = data[['SMA_20', 'RSI', 'MACD']]
y = (data['Close'].shift(-1) > data['Close']).astype(int)  # Binary target

# Remove the last row since it will have NaN in 'y'
X = X[:-1]
y = y[:-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Print accuracy to verify model performance
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the trained model
dump(model, MODEL_PATH)
print(f"Model trained and saved at {MODEL_PATH}")

class MiniAIDE:
    def __init__(self, model_path=MODEL_PATH):
        self.model = load(model_path)

    def predict(self, data):
        """Predict whether to buy or hold/sell."""
        X = data[['SMA_20', 'RSI', 'MACD']]
        prediction = self.model.predict(X)
        return prediction[-1]  # Last prediction

    def optimize_trade(self, signal, cash, current_price):
        """Optimize the trading decision."""
        if signal == 1:  # Buy signal
            shares = cash // current_price
            print(f"Mini-AIDE suggests buying {shares} shares.")
        else:
            print("Mini-AIDE suggests holding or selling.")

# Step 3: Load Data and Use Mini-AIDE for Prediction
latest_data = load_data(DATA_PATH)
latest_price = latest_data['Close'].iloc[-1]

mini_aide = MiniAIDE()
trade_signal = mini_aide.predict(latest_data)
mini_aide.optimize_trade(trade_signal, STARTING_CASH, latest_price)
