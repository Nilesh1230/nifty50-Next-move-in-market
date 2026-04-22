# adaptive_evaluation.py (FINAL FIXED)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
from config import HMM_FEATURES

DATA_PATH = "data/processed/nifty_features.csv"
SEQ_LENGTH = 20
SPLIT_RATIO = 0.8
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# -----------------------------
# MODEL
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
# LOAD DATA + REGIMES
# -----------------------------
def load_all():
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

    hmm_model = joblib.load("models/hmm/hmm_model.pkl")
    hmm_scaler = joblib.load("models/hmm/scaler.pkl")

    df_clean = df[HMM_FEATURES].dropna()

    X_all = hmm_scaler.transform(df_clean)
    regimes = hmm_model.predict(X_all)

    df.loc[df_clean.index, "regime"] = regimes

    df["target"] = df["log_return"].shift(-1)
    df.dropna(inplace=True)

    split = int(len(df) * SPLIT_RATIO)
    test_df = df[split:].copy()

    print("Test samples:", len(test_df))

    return test_df


# -----------------------------
# LOAD EXPERTS
# -----------------------------
def load_experts():
    models = {}
    scalers = {}

    for i in range(5):
        try:
            model = ExpertLSTM(len(HMM_FEATURES)).to(DEVICE)
            model.load_state_dict(torch.load(f"models/lstm/expert_regime_{i}.pt"))
            model.eval()

            scaler = joblib.load(f"models/lstm/scaler_regime_{i}.pkl")

            models[i] = model
            scalers[i] = scaler
        except:
            print(f"⚠️ Missing model for regime {i}")

    return models, scalers


# -----------------------------
# MAIN EVALUATION
# -----------------------------
def evaluate():

    test_df = load_all()
    models, scalers = load_experts()

    predictions = []
    actuals = []

    print("\nEvaluating on TEST DATA ONLY...\n")

    for i in range(SEQ_LENGTH, len(test_df)):

        row = test_df.iloc[i]
        regime = int(row["regime"])

        # ✅ FALLBACK LOGIC (IMPORTANT FIX)
        if regime not in models:
            regime = list(models.keys())[0]   # fallback to first available model

        seq = test_df.iloc[i-SEQ_LENGTH:i][HMM_FEATURES].values

        if np.isnan(seq).any():
            continue

        seq = scalers[regime].transform(seq)
        seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred = models[regime](seq).item()

        predictions.append(pred)
        actuals.append(row["target"])

    # ---------------- SAFETY ----------------
    if len(actuals) == 0:
        print("❌ No predictions generated")
        return

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # ---------------- METRICS ----------------
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    direction = np.mean(np.sign(actuals) == np.sign(predictions))

    print("\n=== Adaptive Regime-Aware Results (FINAL) ===")
    print(f"Samples: {len(actuals)}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"Direction Accuracy: {direction:.4f}")


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    evaluate()