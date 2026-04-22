import pandas as pd
import joblib
from config import BASE_FEATURES

DATA_PATH = "data/processed/nifty_features.csv"

HMM_FEATURES = ["log_return", "volatility", "rsi", "macd"]


def load_data():
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

    # ensure numeric
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    hmm_model = joblib.load("models/hmm/hmm_model.pkl")
    hmm_scaler = joblib.load("models/hmm/scaler.pkl")

    df_clean = df[HMM_FEATURES].dropna()

    X = hmm_scaler.transform(df_clean)
    regimes = hmm_model.predict(X)

    df.loc[df_clean.index, "regime"] = regimes

    df["target"] = df["log_return"].shift(-1)
    df.dropna(inplace=True)

    return df


def main():
    df = load_data()

    print("\n=== REGIME ANALYSIS ===")

    grouped = df.groupby("regime")["log_return"]

    for regime, data in grouped:
        print(f"\nRegime {regime}")
        print(f"Mean Return: {data.mean():.6f}")
        print(f"Volatility: {data.std():.6f}")

    print("\n=== FEATURE CORRELATION ===")
    print(df.corr()["target"].sort_values(ascending=False))


if __name__ == "__main__":
    main()