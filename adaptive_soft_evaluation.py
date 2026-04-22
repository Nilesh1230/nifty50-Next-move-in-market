# adaptive_soft_evaluation.py (FINAL FIXED - HMM_FEATURES)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
from config import HMM_FEATURES   # ✅ FIXED

# ================= CONFIG =================
SEQ_LENGTH = 60
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DATA_PATH = "data/processed/nifty_features.csv"


# ================= MODEL =================
class ExpertLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(ExpertLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )

        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out.squeeze()


# ================= LOAD =================
def load_models():
    hmm_model = joblib.load("models/hmm/hmm_model.pkl")
    hmm_scaler = joblib.load("models/hmm/scaler.pkl")

    expert_models = {}
    expert_scalers = {}

    for i in range(5):
        model = ExpertLSTM(len(HMM_FEATURES)).to(DEVICE)
        model.load_state_dict(torch.load(f"models/lstm/expert_regime_{i}.pt"))
        model.eval()

        scaler = joblib.load(f"models/lstm/scaler_regime_{i}.pkl")

        expert_models[i] = model
        expert_scalers[i] = scaler

    return hmm_model, hmm_scaler, expert_models, expert_scalers


# ================= MAIN =================
def main():

    print("\n=== Adaptive Soft Evaluation (FINAL FIXED) ===\n")

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

    # Target
    df["target"] = df["log_return"].shift(-1)
    df.dropna(inplace=True)

    print("Total samples:", len(df))

    # Split
    split = int(len(df) * 0.8)
    test_df = df.iloc[split:].copy()

    print("Test samples:", len(test_df))

    if len(test_df) < SEQ_LENGTH:
        print("❌ Not enough test data")
        return

    hmm_model, hmm_scaler, expert_models, expert_scalers = load_models()

    # HMM input
    X_hmm = hmm_scaler.transform(test_df[HMM_FEATURES])
    regime_probs = hmm_model.predict_proba(X_hmm)

    predictions = []
    actuals = []

    # ================= LOOP =================
    for i in range(SEQ_LENGTH, len(test_df) - 1):

        window_raw = test_df.iloc[i-SEQ_LENGTH:i][HMM_FEATURES]

        if window_raw.isnull().values.any():
            continue

        weighted_prediction = 0

        for k in range(5):

            model = expert_models[k]
            scaler = expert_scalers[k]

            try:
                window_scaled = scaler.transform(window_raw)
            except:
                continue

            X_input = torch.tensor(window_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                pred_k = model(X_input).item()

            weight = regime_probs[i][k]
            weighted_prediction += weight * pred_k

        predictions.append(weighted_prediction)
        actuals.append(test_df.iloc[i]["target"])

    # ================= SAFETY =================
    if len(actuals) == 0:
        print("❌ No predictions generated")
        return

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # ================= METRICS =================
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)

    direction_accuracy = np.mean(
        np.sign(predictions) == np.sign(actuals)
    )

    # ================= OUTPUT =================
    print("\n=== RESULTS ===")
    print(f"Samples: {len(actuals)}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"Direction Accuracy: {direction_accuracy:.4f}")


# ================= RUN =================
if __name__ == "__main__":
    main()