# regime_analysis.py

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
# from config import FEATURE_COLUMNS
from config import HMM_FEATURES

DATA_PATH = "data/processed/nifty_features.csv"

def load_everything():
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    model = joblib.load("models/hmm/hmm_model.pkl")
    scaler = joblib.load("models/hmm/scaler.pkl")
    return df, model, scaler

def add_regimes(df, model, scaler):

    #  required features
    df_features = df[HMM_FEATURES].copy()

    # Drop NaNs BEFORE scaling
    df_clean = df_features.dropna()

    # Scale
    X = scaler.transform(df_clean)

    # Predict regimes
    hidden_states = model.predict(X)

    # Add regimes back only to valid rows
    df.loc[df_clean.index, "regime"] = hidden_states

    return df


def regime_statistics(df):
    print("\n=== Regime Statistics ===")

    for regime in sorted(df["regime"].unique()):
        subset = df[df["regime"] == regime]

        mean_return = subset["log_return"].mean()
        volatility = subset["log_return"].std()
        duration = len(subset)

        print(f"\nRegime {regime}")
        print(f"Mean Return: {mean_return:.6f}")
        print(f"Volatility: {volatility:.6f}")
        print(f"Data Points: {duration}")

def plot_regimes(df):
    plt.figure(figsize=(12,6))

    for regime in sorted(df["regime"].dropna().unique()):
        subset = df[df["regime"] == regime]
        plt.scatter(subset.index, subset["Close"], s=5, label=f"Regime {regime}")

    plt.legend()
    plt.title("NIFTY 50 Regimes")
    plt.show()

if __name__ == "__main__":
    df, model, scaler = load_everything()
    df = add_regimes(df, model, scaler)

    # Print Transition Matrix
    print("\n=== Transition Matrix ===")
    print(model.transmat_)

    regime_statistics(df)
    plot_regimes(df)

