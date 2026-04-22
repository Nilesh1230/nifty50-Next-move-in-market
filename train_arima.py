import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "data/processed/nifty_features.csv"
SPLIT_RATIO = 0.8


def load_data():
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

    # Use only log return
    series = df["log_return"].dropna()

    return series


def train_arima(series):

    split = int(len(series) * SPLIT_RATIO)

    train = series[:split]
    test = series[split:]

    history = list(train)
    predictions = []

    print("Running ARIMA...")

    for t in range(len(test)):

        model = ARIMA(history, order=(1,0,1))  # simple ARIMA
        model_fit = model.fit()

        yhat = model_fit.forecast()[0]
        predictions.append(yhat)

        history.append(test.iloc[t])

        if t % 100 == 0:
            print(f"Step {t}")

    # metrics
    rmse = np.sqrt(mean_squared_error(test, predictions))
    direction = np.mean(np.sign(predictions) == np.sign(test))

    print("\n=== ARIMA RESULTS ===")
    print(f"RMSE: {rmse:.6f}")
    print(f"Direction Accuracy: {direction:.4f}")

    result_df = pd.DataFrame({
    "actual": test.values,
    "predicted": predictions
    })

    result_df.to_csv("results/arima_predictions.csv", index=False)

if __name__ == "__main__":
    series = load_data()
    train_arima(series)