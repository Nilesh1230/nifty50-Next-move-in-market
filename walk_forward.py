import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")
DATA_PATH = "data/processed/nifty_features.csv"


def load_data():
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    return df["log_return"].dropna()


def walk_forward_arima(series):

    train_size = int(len(series) * 0.8)

    history = list(series[:train_size])
    test = series[train_size:]

    predictions = []

    for t in range(len(test)):

        model = ARIMA(history, order=(1,0,1))
        model_fit = model.fit()

        yhat = model_fit.forecast()[0]
        predictions.append(yhat)

        history.append(test.iloc[t])

    rmse = np.sqrt(np.mean((np.array(predictions) - test.values)**2))

    print(f"Walk Forward RMSE: {rmse:.6f}")


if __name__ == "__main__":
    series = load_data()
    walk_forward_arima(series)