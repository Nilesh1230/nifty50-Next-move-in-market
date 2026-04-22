
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import joblib
# import os
# from sklearn.preprocessing import RobustScaler
# from torch.utils.data import TensorDataset, DataLoader
# from sklearn.metrics import classification_report, confusion_matrix
# from config import *

# DATA_PATH = "data/processed/nifty_features.csv"
# DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# # -----------------------------
# # LOAD DATA
# # -----------------------------
# def load_data():
#     df = pd.read_csv(DATA_PATH, index_col=0)

#     print("Loaded data:", df.shape)

#     hmm_model = joblib.load("models/hmm/hmm_model.pkl")
#     hmm_scaler = joblib.load("models/hmm/scaler.pkl")

#     df = df.replace([np.inf, -np.inf], np.nan)
#     df = df.ffill().bfill()

#     # HMM
#     df_hmm = df[HMM_FEATURES].dropna()
#     X = hmm_scaler.transform(df_hmm)

#     df.loc[df_hmm.index, "regime"] = hmm_model.predict(X)
#     df = df[df["regime"].notna()]

#     n_states = hmm_model.n_components

#     print("HMM usable rows:", len(df))

#     # TARGET (simple & stable)
#     df["future_return"] = df["Close"].pct_change(3).shift(-3)
#     df = df.iloc[:-3]

#     df["target"] = (df["future_return"] > 0).astype(int)

#     print("\nTarget distribution:")
#     print(df["target"].value_counts())

#     df = df.replace([np.inf, -np.inf], np.nan)
#     df = df.ffill().bfill()
#     df.dropna(inplace=True)

#     print("Final dataset:", df.shape)

#     return df, n_states


# # -----------------------------
# # CREATE SEQUENCES
# # -----------------------------
# def create_sequences(df):
#     X, y, r = [], [], []

#     for i in range(len(df) - SEQ_LENGTH):
#         X.append(df.iloc[i:i+SEQ_LENGTH][FEATURE_COLUMNS].values)
#         y.append(df.iloc[i+SEQ_LENGTH]["target"])
#         r.append(df.iloc[i:i+SEQ_LENGTH]["regime"].values)

#     return np.array(X), np.array(y), np.array(r)


# # -----------------------------
# # MODEL
# # -----------------------------
# class LSTMModel(nn.Module):
#     def __init__(self, input_size, num_regimes):
#         super().__init__()

#         self.regime_embed = nn.Embedding(num_regimes, 4)
#         self.lstm = nn.LSTM(input_size + 4, 128, num_layers=2, batch_first=True)
#         self.dropout = nn.Dropout(0.2)
#         self.fc = nn.Linear(128, 1)

#     def forward(self, x, regime):
#         r = self.regime_embed(regime.long())
#         x = torch.cat([x, r], dim=2)

#         out, _ = self.lstm(x)
#         out = out[:, -1, :]
#         out = self.dropout(out)

#         return self.fc(out).squeeze()


# # -----------------------------
# # TRAIN
# # -----------------------------
# def train():
#     df, n_states = load_data()

#     split = int(len(df) * 0.8)

#     train_df = df[:split].copy()
#     test_df = df[split:].copy()

#     scaler = RobustScaler()
#     train_df[FEATURE_COLUMNS] = scaler.fit_transform(train_df[FEATURE_COLUMNS])
#     test_df[FEATURE_COLUMNS] = scaler.transform(test_df[FEATURE_COLUMNS])
#     joblib.dump(scaler, "models/lstm/scaler.pkl")
#     X_train, y_train, r_train = create_sequences(train_df)
#     X_test, y_test, r_test = create_sequences(test_df)

#     X_train = np.nan_to_num(X_train)
#     X_test = np.nan_to_num(X_test)

#     print("Train shape:", X_train.shape)

#     model = LSTMModel(len(FEATURE_COLUMNS), n_states).to(DEVICE)

#     #  STABLE OPTIMIZER
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

#     # AUTO BALANCED LOSS
#     pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
#     loss_fn = nn.BCEWithLogitsLoss(
#         pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(DEVICE)
#     )

#     X_train = torch.tensor(X_train, dtype=torch.float32)
#     r_train = torch.tensor(r_train, dtype=torch.long)
#     y_train = torch.tensor(y_train, dtype=torch.float32)

#     loader = DataLoader(
#         TensorDataset(X_train, r_train, y_train),
#         batch_size=64,
#         shuffle=True
#     )

#     # -----------------------------
#     # TRAIN LOOP
#     # -----------------------------
#     for epoch in range(150):
#         model.train()
#         total_loss = 0

#         for xb, rb, yb in loader:
#             xb, rb, yb = xb.to(DEVICE), rb.to(DEVICE), yb.to(DEVICE)

#             optimizer.zero_grad()

#             logits = model(xb, rb)
#             loss = loss_fn(logits, yb)

#             if torch.isnan(loss) or torch.isinf(loss):
#                 continue

#             loss.backward()

#             #  GRADIENT CLIPPING
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

#             optimizer.step()

#             total_loss += loss.item()

#         if (epoch+1) % 10 == 0:
#             print(f"Epoch {epoch+1}, Loss: {total_loss:.6f}")

#     # -----------------------------
#     # EVALUATION
#     # -----------------------------
#     model.eval()

#     X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
#     r_test_t = torch.tensor(r_test, dtype=torch.long).to(DEVICE)

#     with torch.no_grad():
#         logits = model(X_test_t, r_test_t)
#         probs = torch.sigmoid(logits).cpu().numpy()

#     preds = (probs > 0.55).astype(int)

#     accuracy = np.mean(preds == y_test)
#     up_recall = np.sum((preds==1) & (y_test==1)) / np.sum(y_test==1)

#     print("\n=== FINAL MODEL ===")
#     print("Accuracy:", accuracy)
#     print("UP Recall:", up_recall)

#     print("\n=== CLASSIFICATION REPORT ===")
#     print(classification_report(y_test, preds))

#     print("\n=== CONFUSION MATRIX ===")
#     print(confusion_matrix(y_test, preds))

#     # SAVE
#     os.makedirs("models/lstm", exist_ok=True)
#     torch.save(model.state_dict(), "models/lstm/final_classification.pt")

#     print("\nModel saved")


# # -----------------------------
# if __name__ == "__main__":
#     train()



import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import os
from sklearn.preprocessing import RobustScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from config import *

DATA_PATH = "data/processed/nifty_features.csv"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# -----------------------------
# LOAD DATA
# -----------------------------
def load_data():
    df = pd.read_csv(DATA_PATH, index_col=0)

    print("Loaded data:", df.shape)

    hmm_model = joblib.load("models/hmm/hmm_model.pkl")
    hmm_scaler = joblib.load("models/hmm/scaler.pkl")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill()

    # HMM REGIME
    df_hmm = df[HMM_FEATURES].dropna()
    X = hmm_scaler.transform(df_hmm)

    df.loc[df_hmm.index, "regime"] = hmm_model.predict(X)
    df = df[df["regime"].notna()]

    n_states = hmm_model.n_components

    print("HMM usable rows:", len(df))

    # -----------------------------
    # 🔥 TARGET (LESS NOISE)
    # -----------------------------
    df["future_return"] = df["Close"].pct_change(10).shift(-10)
    df = df.iloc[:-10]

    df["target"] = (df["future_return"] > 0).astype(int)

    print("\nTarget distribution:")
    print(df["target"].value_counts())

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill()
    df.dropna(inplace=True)

    print("Final dataset:", df.shape)

    return df, n_states


# -----------------------------
# CREATE SEQUENCES
# -----------------------------
def create_sequences(df):
    X, y, r = [], [], []

    for i in range(len(df) - SEQ_LENGTH):
        X.append(df.iloc[i:i+SEQ_LENGTH][FEATURE_COLUMNS].values)
        y.append(df.iloc[i+SEQ_LENGTH]["target"])
        r.append(df.iloc[i:i+SEQ_LENGTH]["regime"].values)

    return np.array(X), np.array(y), np.array(r)


# -----------------------------
# MODEL (STABLE)
# -----------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, num_regimes):
        super().__init__()

        self.regime_embed = nn.Embedding(num_regimes, 3)

        self.lstm = nn.LSTM(
            input_size + 3,
            64,
            num_layers=1,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(64, 1)

    def forward(self, x, regime):
        r = self.regime_embed(regime.long())
        x = torch.cat([x, r], dim=2)

        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)

        return self.fc(out).squeeze()


# -----------------------------
# TRAIN
# -----------------------------
def train():
    df, n_states = load_data()

    split = int(len(df) * 0.8)

    train_df = df[:split].copy()
    test_df = df[split:].copy()

    scaler = RobustScaler()
    train_df[FEATURE_COLUMNS] = scaler.fit_transform(train_df[FEATURE_COLUMNS])
    test_df[FEATURE_COLUMNS] = scaler.transform(test_df[FEATURE_COLUMNS])
    joblib.dump(scaler, "models/lstm/scaler.pkl")

    X_train, y_train, r_train = create_sequences(train_df)
    X_test, y_test, r_test = create_sequences(test_df)

    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)

    print("Train shape:", X_train.shape)

    model = LSTMModel(len(FEATURE_COLUMNS), n_states).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()

    X_train = torch.tensor(X_train, dtype=torch.float32)
    r_train = torch.tensor(r_train, dtype=torch.long)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X_train, r_train, y_train),
        batch_size=64,
        shuffle=True
    )

    # -----------------------------
    # TRAIN LOOP
    # -----------------------------
    for epoch in range(150):
        model.train()
        total_loss = 0

        for xb, rb, yb in loader:
            xb, rb, yb = xb.to(DEVICE), rb.to(DEVICE), yb.to(DEVICE)

            optimizer.zero_grad()

            logits = model(xb, rb)
            loss = loss_fn(logits, yb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss:.6f}")

    # -----------------------------
    # EVALUATION
    # -----------------------------
    model.eval()

    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    r_test_t = torch.tensor(r_test, dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        logits = model(X_test_t, r_test_t)
        probs = torch.sigmoid(logits).cpu().numpy()

    preds = (probs > 0.5).astype(int)

    accuracy = np.mean(preds == y_test)

    print("\n=== FINAL MODEL ===")
    print("Accuracy:", accuracy)

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, preds))

    print("\n=== CONFUSION MATRIX ===")
    print(confusion_matrix(y_test, preds))

    # SAVE
    os.makedirs("models/lstm", exist_ok=True)
    torch.save(model.state_dict(), "models/lstm/final_classification.pt")

    print("\nModel saved")


# -----------------------------
if __name__ == "__main__":
    train()