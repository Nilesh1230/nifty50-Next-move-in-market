import yfinance as yf
import pandas as pd
import numpy as np
import ta
import os
from config import *


# -----------------------------
# DOWNLOAD DATAs
# -----------------------------
def download_data():
    df = yf.download(DATA_SYMBOL, period=PERIOD, interval=INTERVAL)

    if df.empty:
        raise ValueError("No data downloaded")

    # fix multi-index
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    print("Downloaded shape:", df.shape)

    return df


# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
def feature_engineering(df):

    # -----------------------------
    # BASIC RETURNS
    # -----------------------------
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["volatility"] = df["log_return"].rolling(20).std()

    # -----------------------------
    # INDICATORS
    # -----------------------------
    close = df["Close"]

    df["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    df["macd"] = ta.trend.MACD(close).macd()

    # -----------------------------
    # PRICE STRUCTURE
    # -----------------------------
    df["price_range"] = df["High"] - df["Low"]
    df["range_strength"] = df["price_range"] / (df["Close"] + 1e-6)

    # -----------------------------
    # MOMENTUM
    # -----------------------------
    df["momentum"] = df["Close"] - df["Close"].shift(5)

    # -----------------------------
#  LIQUIDITY FEATURES
# -----------------------------

# volume pressure
    df["volume_pressure"] = df["Volume"] * df["log_return"]

# volume volatility
    df["volume_volatility"] = df["Volume"].rolling(20).std()

# turnover proxy
    df["turnover"] = df["Close"] * df["Volume"]

# relative volume (safe)
    df["rel_volume"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-6)

    # -----------------------------
    # TREND FEATURES (IMPROVED)
    # -----------------------------
    df["ema_fast"] = df["Close"].ewm(span=10).mean()
    df["ema_slow"] = df["Close"].ewm(span=50).mean()

    df["trend_strength"] = df["ema_fast"] - df["ema_slow"]

    df["ema_20"] = df["Close"].ewm(span=20).mean()
    df["trend"] = df["Close"] - df["ema_20"]

    # -----------------------------
    # VOLATILITY DYNAMICS
    # -----------------------------
    df["volatility_change"] = df["volatility"].diff()
    df["volatility_break"] = df["log_return"].rolling(10).std()

    # -----------------------------
    #  REMOVE NOISY / UNSTABLE FEATURES
    # -----------------------------
    # volume_ratio often unstable → drop
    if "volume_ratio" in df.columns:
        df.drop(columns=["volume_ratio"], inplace=True)

    # -----------------------------
    # CLEANING (CRITICAL FIX)
    # -----------------------------
    df = df.replace([np.inf, -np.inf], np.nan)

    # indicator warmup removal (important)
    df = df.iloc[60:]

    # fill safely
    df = df.ffill().bfill()

    # final clean
    df.dropna(inplace=True)

    print("Final data shape:", df.shape)

    # -----------------------------
    # SAVE
    # -----------------------------
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/nifty_features.csv")

    print("Data saved at data/processed/nifty_features.csv")

    return df


# -----------------------------
# RUN PIPELINE
# -----------------------------
if __name__ == "__main__":
    df = download_data()
    df = feature_engineering(df)
    print(" Pipeline complete")