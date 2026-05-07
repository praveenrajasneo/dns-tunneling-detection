"""
fusion.py
---------
Weighted decision fusion combining Random Forest, LSTM, and Rule Engine scores.
Final output: risk classification (High / Medium / Low)
"""

import numpy as np
import pandas as pd

# Fusion weights (tuned on validation set)
FUSION_WEIGHTS = {
    "rf": 0.45,
    "lstm": 0.40,
    "rule": 0.15,
}

RISK_THRESHOLDS = {
    "High": 0.65,
    "Medium": 0.35,
}


def fuse_scores(rf_prob: np.ndarray, lstm_prob: np.ndarray, rule_score: np.ndarray) -> np.ndarray:
    """
    Combine RF probability, LSTM probability, and rule-based score
    into a single fused confidence score.
    """
    fused = (
        FUSION_WEIGHTS["rf"] * rf_prob +
        FUSION_WEIGHTS["lstm"] * lstm_prob +
        FUSION_WEIGHTS["rule"] * rule_score
    )
    return np.clip(fused, 0.0, 1.0)


def classify_risk(fused_scores: np.ndarray) -> list:
    """Map fused scores to risk labels."""
    labels = []
    for score in fused_scores:
        if score >= RISK_THRESHOLDS["High"]:
            labels.append("High")
        elif score >= RISK_THRESHOLDS["Medium"]:
            labels.append("Medium")
        else:
            labels.append("Low")
    return labels


def run_fusion(features_df: pd.DataFrame, rf_probs: np.ndarray, lstm_probs: np.ndarray) -> pd.DataFrame:
    """Full fusion pipeline returning results DataFrame."""
    rule_scores = features_df["rule_score"].values if "rule_score" in features_df else np.zeros(len(features_df))

    fused = fuse_scores(rf_probs, lstm_probs, rule_scores)
    risk = classify_risk(fused)

    results = features_df.copy()
    results["rf_prob"] = rf_probs
    results["lstm_prob"] = lstm_probs
    results["fused_score"] = fused
    results["risk"] = risk
    return results
