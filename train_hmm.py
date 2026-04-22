# import pandas as pd
# import numpy as np
# import os
# import joblib
# from sklearn.preprocessing import StandardScaler
# from hmmlearn.hmm import GaussianHMM
# from config import HMM_FEATURES, HMM_ITERATIONS

# DATA_PATH = "data/processed/nifty_features.csv"
# SPLIT_RATIO = 0.8


# # -----------------------------
# # LOAD DATA
# # -----------------------------
# def load_data():
#     df = pd.read_csv(DATA_PATH, index_col=0)

#     print("Original shape:", df.shape)

#     # 🔥 IMPORTANT: only HMM features
#     df = df[HMM_FEATURES].copy()

#     # clean
#     df.replace([np.inf, -np.inf], np.nan, inplace=True)
#     df.ffill(inplace=True)
#     df.bfill(inplace=True)
#     df.dropna(inplace=True)

#     print("After cleaning:", df.shape)

#     if len(df) < 200:
#         raise ValueError("Too little data for HMM")

#     return df


# # -----------------------------
# # COMPUTE BIC
# # -----------------------------
# def compute_bic(model, X):
#     log_likelihood = model.score(X)
#     n_samples, n_features = X.shape

#     n_components = model.n_components

#     n_params = (
#         n_components - 1 +
#         n_components * (n_components - 1) +
#         n_components * n_features * 2
#     )

#     return -2 * log_likelihood + n_params * np.log(n_samples)


# # -----------------------------
# # TRAIN HMM
# # -----------------------------
# def train_hmm_models(X_train):
#     best_bic = np.inf
#     best_model = None
#     best_n = None

#     print("Training HMM models...")

#     for n in range(2, 16):  # 🔥 REDUCED (more stable)

#         try:
#             model = GaussianHMM(
#                 n_components=n,
#                 covariance_type="diag",
#                 n_iter=HMM_ITERATIONS,
#                 random_state=42
#             )

#             model.fit(X_train)

#             bic = compute_bic(model, X_train)

#             print(f"States: {n}, BIC: {bic:.2f}")

#             if bic < best_bic:
#                 best_bic = bic
#                 best_model = model
#                 best_n = n

#         except Exception as e:
#             print(f"State {n} failed: {e}")
#             continue

#     print(f"\nBest model selected with {best_n} states.")
#     return best_model, best_n


# # -----------------------------
# # SAVE MODEL
# # -----------------------------
# def save_model(model, scaler, n_states):
#     os.makedirs("models/hmm", exist_ok=True)

#     joblib.dump(model, "models/hmm/hmm_model.pkl")
#     joblib.dump(scaler, "models/hmm/scaler.pkl")
#     joblib.dump(n_states, "models/hmm/n_states.pkl")  # 🔥 NEW

#     print("Model saved.")


# # -----------------------------
# # MAIN
# # -----------------------------
# if __name__ == "__main__":

#     df = load_data()

#     split = int(len(df) * SPLIT_RATIO)

#     train_df = df[:split]
#     test_df = df[split:]

#     print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

#     # -----------------------------
#     # SCALING
#     # -----------------------------
#     scaler = StandardScaler()

#     X_train = scaler.fit_transform(train_df)
#     X_test = scaler.transform(test_df)

#     print("Any NaN in train?", np.isnan(X_train).any())
#     print("Any NaN in test?", np.isnan(X_test).any())

#     # -----------------------------
#     # TRAIN
#     # -----------------------------
#     model, n_states = train_hmm_models(X_train)

#     # -----------------------------
#     # SAVE
#     # -----------------------------
#     save_model(model, scaler, n_states)

#     print("\nHMM training completed WITHOUT data leakage")

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from config import HMM_FEATURES, HMM_ITERATIONS

DATA_PATH = "data/processed/nifty_features.csv"
SPLIT_RATIO = 0.8


# -----------------------------
# LOAD DATA
# -----------------------------
def load_data():
    df = pd.read_csv(DATA_PATH, index_col=0)

    print("Original shape:", df.shape)

    df = df[HMM_FEATURES].copy()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.dropna(inplace=True)

    print("After cleaning:", df.shape)

    if len(df) < 200:
        raise ValueError("Too little data for HMM")

    return df


# -----------------------------
# COMPUTE BIC
# -----------------------------
def compute_bic(model, X):
    log_likelihood = model.score(X)
    n_samples, n_features = X.shape

    n_components = model.n_components

    n_params = (
        n_components - 1 +
        n_components * (n_components - 1) +
        n_components * n_features * 2
    )

    return -2 * log_likelihood + n_params * np.log(n_samples)


# -----------------------------
# TRAIN HMM
# -----------------------------
def train_hmm_models(X_train):
    best_bic = np.inf
    best_model = None
    best_n = None

    print("Training HMM models...")

    for n in range(2, 10):   # 🔥 stable range
        try:
            model = GaussianHMM(
                n_components=n,
                covariance_type="diag",
                n_iter=HMM_ITERATIONS,
                random_state=42
            )

            model.fit(X_train)

            bic = compute_bic(model, X_train)

            print(f"States: {n}, BIC: {bic:.2f}")

            if bic < best_bic:
                best_bic = bic
                best_model = model
                best_n = n

        except Exception as e:
            print(f"State {n} failed: {e}")
            continue

    print(f"\nBest model selected with {best_n} states.")
    return best_model, best_n


# -----------------------------
# REGIME ANALYSIS (IMPORTANT)
# -----------------------------
def analyze_regimes(model, scaler, df):

    print("\n=== REGIME ANALYSIS ===")

    X_scaled = scaler.transform(df)
    states = model.predict(X_scaled)

    df["regime"] = states

    analysis = df.groupby("regime")[HMM_FEATURES].mean()
    print(analysis)

    # -----------------------------
    # AUTO LABELING
    # -----------------------------
    print("\n=== REGIME LABELING ===")

    mean_vol = analysis["volatility"].mean()

    for i, row in analysis.iterrows():

        ret = row["log_return"]
        vol = row["volatility"]

        if ret > 0 and vol > mean_vol:
            label = "Strong Bullish"
        elif ret > 0:
            label = "Bullish"

        elif ret < 0 and vol > mean_vol:
            label = "Strong Bearish"
        elif ret < 0:
            label = "Bearish"

        else:
            label = "Sideways"

        print(f"State {i} → {label}")


# -----------------------------
# SAVE MODEL
# -----------------------------
def save_model(model, scaler, n_states):
    os.makedirs("models/hmm", exist_ok=True)

    joblib.dump(model, "models/hmm/hmm_model.pkl")
    joblib.dump(scaler, "models/hmm/scaler.pkl")
    joblib.dump(n_states, "models/hmm/n_states.pkl")

    print("Model saved.")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    df = load_data()

    split = int(len(df) * SPLIT_RATIO)

    train_df = df[:split]
    test_df = df[split:]

    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    # -----------------------------
    # SCALING
    # -----------------------------
    scaler = StandardScaler()

    X_train = scaler.fit_transform(train_df)
    X_test = scaler.transform(test_df)

    print("Any NaN in train?", np.isnan(X_train).any())
    print("Any NaN in test?", np.isnan(X_test).any())

    # -----------------------------
    # TRAIN
    # -----------------------------
    model, n_states = train_hmm_models(X_train)

    # -----------------------------
    # ANALYSIS (🔥 USE FULL DATA)
    # -----------------------------
    analyze_regimes(model, scaler, df.copy())

    # -----------------------------
    # SAVE
    # -----------------------------
    save_model(model, scaler, n_states)

    print("\nHMM training completed WITHOUT data leakage")