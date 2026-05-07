"""
classifier.py
-------------
Trains and runs Random Forest and LSTM classifiers on extracted DNS features.
"""

import numpy as np
import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

FEATURE_COLS = [
    "query_length", "entropy", "subdomain_depth",
    "ngram_deviation", "digit_ratio", "temporal_freq", "response_size_ratio"
]

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RF_PATH = os.path.join(MODEL_DIR, "random_forest.pkl")
LSTM_PATH = os.path.join(MODEL_DIR, "lstm_model.h5")


# ── Random Forest ──────────────────────────────────────────────────────────────

def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(rf, RF_PATH)
    print(f"Random Forest saved to {RF_PATH}")
    return rf


def load_random_forest() -> RandomForestClassifier:
    return joblib.load(RF_PATH)


def predict_rf(X: np.ndarray) -> np.ndarray:
    rf = load_random_forest()
    return rf.predict_proba(X)[:, 1]  # probability of tunneling class


# ── LSTM ───────────────────────────────────────────────────────────────────────

def build_lstm(input_shape: tuple) -> Sequential:
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_lstm(X_train, y_train) -> Sequential:
    # Reshape for LSTM: (samples, timesteps, features) — treat each feature as timestep
    X_seq = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    model = build_lstm((X_seq.shape[1], 1))
    es = EarlyStopping(patience=3, restore_best_weights=True)
    model.fit(X_seq, y_train, epochs=20, batch_size=64, validation_split=0.1,
              callbacks=[es], verbose=1)
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(LSTM_PATH)
    print(f"LSTM model saved to {LSTM_PATH}")
    return model


def predict_lstm(X: np.ndarray) -> np.ndarray:
    model = load_model(LSTM_PATH)
    X_seq = X.reshape((X.shape[0], X.shape[1], 1))
    return model.predict(X_seq, verbose=0).flatten()


# ── Training Pipeline ──────────────────────────────────────────────────────────

def train_all(df: pd.DataFrame, label_col: str = "label"):
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[label_col].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n--- Training Random Forest ---")
    rf = train_random_forest(X_train, y_train)
    rf_preds = (predict_rf(X_test) >= 0.5).astype(int)
    print(f"RF Accuracy: {accuracy_score(y_test, rf_preds):.4f}")

    print("\n--- Training LSTM ---")
    train_lstm(X_train, y_train)
    lstm_preds = (predict_lstm(X_test) >= 0.5).astype(int)
    print(f"LSTM Accuracy: {accuracy_score(y_test, lstm_preds):.4f}")

    print("\n--- Classification Report (RF) ---")
    print(classification_report(y_test, rf_preds, target_names=["Normal", "Tunneling"]))


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python classifier.py <path_to_features_csv>")
        print("CSV must have columns: query_length, entropy, subdomain_depth,")
        print("  ngram_deviation, digit_ratio, temporal_freq, response_size_ratio, label")
    else:
        df = pd.read_csv(sys.argv[1])
        train_all(df)
