import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from ta.momentum import RSIIndicator
from ta.trend import MACD
from config import *

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# -----------------------------
# MODEL (MATCH TRAINING)
# -----------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, num_regimes):
        super().__init__()

        self.regime_embed = nn.Embedding(num_regimes, 3)
        self.lstm = nn.LSTM(input_size + 3, 64, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(64, 1)

    def forward(self, x, regime):
        r = self.regime_embed(regime.long())
        x = torch.cat([x, r], dim=2)

        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)

        return self.fc(out).squeeze()


# -----------------------------
# LOAD DATA
# -----------------------------
df = yf.download(DATA_SYMBOL, period=PERIOD, interval=INTERVAL)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
df["volatility"] = df["log_return"].rolling(20).std()

close = df["Close"]

df["rsi"] = RSIIndicator(close).rsi()
df["macd"] = MACD(close).macd()

df["price_range"] = df["High"] - df["Low"]
df["range_strength"] = df["price_range"] / (df["Close"] + 1e-6)

df["momentum"] = df["Close"] - df["Close"].shift(5)

df["ema_fast"] = df["Close"].ewm(span=10).mean()
df["ema_slow"] = df["Close"].ewm(span=50).mean()
df["trend_strength"] = df["ema_fast"] - df["ema_slow"]

df["ema_20"] = df["Close"].ewm(span=20).mean()
df["trend"] = df["Close"] - df["ema_20"]

df["volatility_break"] = df["log_return"].rolling(10).std()

# 🔥 FIXED LIQUIDITY FEATURES
df["volume_pressure"] = df["Volume"] * df["log_return"]
df["volume_volatility"] = df["Volume"].rolling(20).std()
df["turnover"] = df["Close"] * df["Volume"]
df["rel_volume"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-6)

# CLEAN
df = df.replace([np.inf, -np.inf], np.nan)
df = df.ffill().bfill()
df.dropna(inplace=True)

# -----------------------------
# HMM REGIME
# -----------------------------
hmm_model = joblib.load("models/hmm/hmm_model.pkl")
hmm_scaler = joblib.load("models/hmm/scaler.pkl")

X_scaled = hmm_scaler.transform(df[HMM_FEATURES])
df["regime"] = hmm_model.predict(X_scaled)

num_regimes = hmm_model.n_components

# -----------------------------
# INPUT
# -----------------------------
# -----------------------------
# INPUT
# -----------------------------
last_seq = df.iloc[-SEQ_LENGTH:]

# 🔥 LOAD SCALER
scaler = joblib.load("models/lstm/scaler.pkl")

# 🔥 KEEP AS DATAFRAME (IMPORTANT)
X_input = last_seq[FEATURE_COLUMNS]

# SCALE
X_input = scaler.transform(X_input)

# convert to numpy (already numpy but safe)
X_input = np.array(X_input)

r_input = last_seq["regime"].values

# LOAD SCALER (IMPORTANT FIX)
scaler = joblib.load("models/lstm/scaler.pkl")
X_input = scaler.transform(X_input)

# TENSOR
X_input = torch.tensor(X_input, dtype=torch.float32).unsqueeze(0).to(DEVICE)
r_input = torch.tensor(r_input, dtype=torch.long).unsqueeze(0).to(DEVICE)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = LSTMModel(len(FEATURE_COLUMNS), num_regimes).to(DEVICE)
model.load_state_dict(torch.load("models/lstm/final_classification.pt", map_location=DEVICE))
model.eval()

# -----------------------------
# PREDICT
# -----------------------------
with torch.no_grad():
    logits = model(X_input, r_input)
    prob = torch.sigmoid(logits).item()
print("Last timestamp:", df.index[-1])
# -----------------------------
# OUTPUT
# -----------------------------
price = float(df["Close"].iloc[-1])
vol = df["volatility"].iloc[-1]

range_points = price * vol

if prob > 0.6:
    trend = "UP"
elif prob < 0.4:
    trend = "DOWN"
else:
    trend = "SIDEWAYS"

print("\n=== LIVE NIFTY ===")
print(f"Price: {price:.2f}")

print("\n=== PREDICTION ===")
print(f"Signal: {trend}")
print(f"Confidence: {prob:.4f}")
print(f"Range: {price-range_points:.2f} - {price+range_points:.2f}")