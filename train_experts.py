# train_experts.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import os
from sklearn.preprocessing import StandardScaler
from config import BASE_FEATURES

DATA_PATH = "data/processed/nifty_features.csv"
SEQ_LENGTH = 20
SPLIT_RATIO = 0.8
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# -----------------------------
# Load Data + Add Regime
# -----------------------------
def load_data():
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

    hmm_model = joblib.load("models/hmm/hmm_model.pkl")
    hmm_scaler = joblib.load("models/hmm/scaler.pkl")

    # Clean features
    df_clean = df[BASE_FEATURES].dropna()
    # df_clean = df[FEATURE_COLUMNS].dropna()


    # Scale
    X_all = hmm_scaler.transform(df_clean)

    # Predict regimes
    regimes = hmm_model.predict(X_all)

    df.loc[df_clean.index, "regime"] = regimes

    # Target
    df["target"] = df["log_return"].shift(-1)

    df.dropna(inplace=True)

    return df


# -----------------------------
# Sequence Creation
# -----------------------------
def create_sequences(data):
    X_seq, y_seq = [], []

    for i in range(len(data) - SEQ_LENGTH):
        X_seq.append(data.iloc[i:i+SEQ_LENGTH][BASE_FEATURES].values)
        y_seq.append(data.iloc[i+SEQ_LENGTH]["target"])

    return np.array(X_seq), np.array(y_seq)


# -----------------------------
# LSTM Model
# -----------------------------
class ExpertLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(ExpertLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out.squeeze()


# -----------------------------
# Train-Test Split (Time Series)
# -----------------------------
def train_test_split_time(df):
    split = int(len(df) * SPLIT_RATIO)
    return df[:split], df[split:]


# -----------------------------
# Training Function
# -----------------------------
def train_model(X_train, y_train, regime_id, epochs=30):

    model = ExpertLSTM(input_size=len(BASE_FEATURES)).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)

    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()
        outputs = model(X_train)

        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 5 == 0:
            print(f"Regime {regime_id} | Epoch {epoch+1} | Train Loss: {loss.item():.6f}")

    return model


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    df = load_data()

    #  SPLIT DATA (IMPORTANT)
    train_df, test_df = train_test_split_time(df)

    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    os.makedirs("models/lstm", exist_ok=True)

    # -----------------------------
    # TRAIN ONLY ON TRAIN DATA
    # -----------------------------
    for regime in sorted(train_df["regime"].unique()):

        print(f"\n==============================")
        print(f"Training Expert Model for Regime {regime}")
        print(f"==============================")

        df_regime = train_df[train_df["regime"] == regime].copy()

        # Skip small regimes
        if len(df_regime) < SEQ_LENGTH + 100:
            print("Not enough data, skipping...")
            continue

        # Regime-specific scaling
        scaler = StandardScaler()
        df_regime[BASE_FEATURES] = scaler.fit_transform(df_regime[BASE_FEATURES])

        X, y = create_sequences(df_regime)

        model = train_model(X, y, regime_id=regime)

        # Save model + scaler
        torch.save(model.state_dict(), f"models/lstm/expert_regime_{int(regime)}.pt")
        joblib.dump(scaler, f"models/lstm/scaler_regime_{int(regime)}.pkl")

    print("\nAll expert models trained WITHOUT data leakage ")