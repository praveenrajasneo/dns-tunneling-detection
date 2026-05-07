"""
rule_engine.py
--------------
Initial rule-based anomaly scoring before ML classification.
Assigns a suspicion score (0.0 to 1.0) based on threshold rules.
"""

import pandas as pd

# Thresholds derived from CIC-Bell-DNS-2021 dataset analysis
THRESHOLDS = {
    "query_length": 52,       # Queries longer than 52 chars are suspicious
    "entropy": 3.5,           # High entropy indicates encoded payloads
    "subdomain_depth": 4,     # Deep subdomains common in tunneling
    "ngram_deviation": 0.75,  # High deviation = random-looking strings
    "digit_ratio": 0.3,       # Excessive digits in domain
    "temporal_freq": 10.0,    # More than 10 queries/sec from one IP
    "response_size_ratio": 0.05,  # Very low response relative to query
}

WEIGHTS = {
    "query_length": 0.20,
    "entropy": 0.25,
    "subdomain_depth": 0.15,
    "ngram_deviation": 0.20,
    "digit_ratio": 0.10,
    "temporal_freq": 0.05,
    "response_size_ratio": 0.05,
}


def score_record(row: pd.Series) -> float:
    """Compute a weighted anomaly score for a single feature row."""
    score = 0.0
    score += WEIGHTS["query_length"] * (1 if row["query_length"] > THRESHOLDS["query_length"] else 0)
    score += WEIGHTS["entropy"] * (1 if row["entropy"] > THRESHOLDS["entropy"] else 0)
    score += WEIGHTS["subdomain_depth"] * (1 if row["subdomain_depth"] > THRESHOLDS["subdomain_depth"] else 0)
    score += WEIGHTS["ngram_deviation"] * (1 if row["ngram_deviation"] > THRESHOLDS["ngram_deviation"] else 0)
    score += WEIGHTS["digit_ratio"] * (1 if row["digit_ratio"] > THRESHOLDS["digit_ratio"] else 0)
    score += WEIGHTS["temporal_freq"] * (1 if row["temporal_freq"] > THRESHOLDS["temporal_freq"] else 0)
    score += WEIGHTS["response_size_ratio"] * (1 if row["response_size_ratio"] < THRESHOLDS["response_size_ratio"] else 0)
    return round(score, 4)


def apply_rules(features_df: pd.DataFrame) -> pd.DataFrame:
    """Apply rule engine to all records and return scores."""
    features_df = features_df.copy()
    features_df["rule_score"] = features_df.apply(score_record, axis=1)
    return features_df


def classify_risk(score: float) -> str:
    """Map rule score to risk level."""
    if score >= 0.6:
        return "High"
    elif score >= 0.3:
        return "Medium"
    return "Low"
