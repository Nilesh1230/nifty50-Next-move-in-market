# train_baseline.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
from sklearn.preprocessing import StandardScaler
from config import FEATURE_COLUMNS

DATA_PATH = "data/processed/nifty_features.csv"
SEQ_LENGTH = 20
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_data():
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    df["target"] = df["log_return"].shift(-1)
    df.dropna(inplace=True)
    return df


def create_sequences(data):
    X_seq, y_seq = [], []

    for i in range(len(data) - SEQ_LENGTH):
        X_seq.append(data.iloc[i:i+SEQ_LENGTH][FEATURE_COLUMNS].values)
        y_seq.append(data.iloc[i+SEQ_LENGTH]["target"])

    return np.array(X_seq), np.array(y_seq)


def train_test_split_time(X, y, split_ratio=0.8):
    split_index = int(len(X) * split_ratio)
    return X[:split_index], y[:split_index], X[split_index:], y[split_index:]


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


if __name__ == "__main__":

    df = load_data()

    scaler = StandardScaler()
    df[FEATURE_COLUMNS] = scaler.fit_transform(df[FEATURE_COLUMNS])

    X, y = create_sequences(df)

    X_train, y_train, X_test, y_test = train_test_split_time(X, y)

    model = GlobalLSTM(len(FEATURE_COLUMNS)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)

    for epoch in range(90):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1} | Train Loss: {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        y_test_t = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)
        test_pred = model(X_test_t)
        test_loss = criterion(test_pred, y_test_t)

    print(f"\nBaseline Test Loss: {test_loss.item():.6f}")
    torch.save(model.state_dict(), "global_model.pt")
    print("Global model saved.")
