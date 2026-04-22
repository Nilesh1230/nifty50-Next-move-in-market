# baseline_evaluation.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from config import FEATURE_COLUMNS

SEQ_LENGTH = 20
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DATA_PATH = "data/processed/nifty_features.csv"


class GlobalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(GlobalLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze()


def main():

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    df["target"] = df["log_return"].shift(-1)
    df.dropna(inplace=True)

    scaler = StandardScaler()
    df[FEATURE_COLUMNS] = scaler.fit_transform(df[FEATURE_COLUMNS])

    model = GlobalLSTM(len(FEATURE_COLUMNS)).to(DEVICE)
    model.load_state_dict(torch.load("global_model.pt"))
    model.eval()

    predictions = []
    actuals = []

    for i in range(SEQ_LENGTH, len(df)-1):

        window = df.iloc[i-SEQ_LENGTH:i][FEATURE_COLUMNS]
        X_input = torch.tensor(window.values, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred = model(X_input).item()

        predictions.append(pred)
        actuals.append(df.iloc[i]["target"])

    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)

    direction_accuracy = np.mean(
        np.sign(predictions) == np.sign(actuals)
    )

    print("\n=== Baseline Rolling Results ===")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"Direction Accuracy: {direction_accuracy:.4f}")


if __name__ == "__main__":
    main()
