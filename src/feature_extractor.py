"""
feature_extractor.py
--------------------
Extracts 7 statistical features from DNS query records for ML classification.
"""

import math
import re
import numpy as np
import pandas as pd
import tldextract
from collections import Counter


def compute_entropy(text: str) -> float:
    """Calculate Shannon entropy of a string."""
    if not text:
        return 0.0
    freq = Counter(text)
    length = len(text)
    return -sum((c / length) * math.log2(c / length) for c in freq.values())


def compute_ngram_deviation(text: str, n: int = 3) -> float:
    """
    Measure deviation from expected English n-gram distribution.
    Higher value = more random/encoded = more suspicious.
    """
    # Common English trigrams (simplified)
    common_trigrams = {"the", "and", "ing", "ion", "ent", "com", "net", "www"}
    ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
    if not ngrams:
        return 0.0
    uncommon = sum(1 for g in ngrams if g not in common_trigrams)
    return uncommon / len(ngrams)


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract 7 features from a DNS records DataFrame.

    Returns a new DataFrame with feature columns:
        query_length, entropy, subdomain_depth, ngram_deviation,
        digit_ratio, temporal_freq, response_size_ratio
    """
    features = pd.DataFrame()

    # 1. Query Length
    features["query_length"] = df["query_name"].apply(len)

    # 2. Shannon Entropy
    features["entropy"] = df["query_name"].apply(compute_entropy)

    # 3. Subdomain Depth
    features["subdomain_depth"] = df["query_name"].apply(
        lambda d: len(tldextract.extract(d).subdomain.split(".")) if tldextract.extract(d).subdomain else 0
    )

    # 4. N-gram Deviation
    features["ngram_deviation"] = df["query_name"].apply(
        lambda d: compute_ngram_deviation(d.replace(".", ""))
    )

    # 5. Digit Ratio
    features["digit_ratio"] = df["query_name"].apply(
        lambda d: sum(c.isdigit() for c in d) / max(len(d), 1)
    )

    # 6. Temporal Behavior (queries per second per source IP)
    if "timestamp" in df.columns and "src_ip" in df.columns:
        df = df.sort_values("timestamp")
        df["temporal_freq"] = df.groupby("src_ip")["timestamp"].transform(
            lambda t: t.diff().fillna(1).apply(lambda x: 1 / max(x, 0.001))
        )
        features["temporal_freq"] = df["temporal_freq"].values
    else:
        features["temporal_freq"] = 0.0

    # 7. Response Size Ratio (answer_count as proxy)
    if "answer_count" in df.columns:
        features["response_size_ratio"] = df["answer_count"] / features["query_length"].replace(0, 1)
    else:
        features["response_size_ratio"] = 0.0

    return features


if __name__ == "__main__":
    # Quick test with synthetic data
    sample = pd.DataFrame({
        "query_name": [
            "normal.google.com",
            "aGVsbG8gd29ybGQ.evil.io",
            "XjR2dGhpcyBpcyBhIHRlc3Q.tunnel.net",
            "mail.microsoft.com",
        ],
        "answer_count": [1, 0, 0, 2],
        "timestamp": [1000.0, 1000.1, 1000.2, 1000.5],
        "src_ip": ["192.168.1.1", "10.0.0.5", "10.0.0.5", "192.168.1.1"],
    })

    feats = extract_features(sample)
    print(feats.to_string())
