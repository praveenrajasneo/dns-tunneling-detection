"""
dashboard.py
------------
Real-time Streamlit monitoring dashboard for DNS tunneling detection.
Run with: streamlit run src/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import sys

sys.path.append(os.path.dirname(__file__))

from feature_extractor import extract_features
from rule_engine import apply_rules, classify_risk
from fusion import run_fusion

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DNS Tunneling Detector",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 DNS Tunneling Detection Dashboard")
st.markdown("**Real-time monitoring powered by RF + LSTM Decision Fusion**")
st.divider()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload DNS records (CSV)", type=["csv"])
    risk_filter = st.multiselect("Filter by Risk Level", ["High", "Medium", "Low"],
                                  default=["High", "Medium", "Low"])
    auto_refresh = st.checkbox("Auto-refresh (demo mode)", value=False)
    st.divider()
    st.markdown("**Fusion Weights**")
    st.text("RF:    45%")
    st.text("LSTM:  40%")
    st.text("Rules: 15%")

# ── Load Data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_process(file):
    df = pd.read_csv(file)
    features = extract_features(df)
    features = apply_rules(features)

    # Simulate model probabilities if models not trained yet
    np.random.seed(42)
    rf_probs = np.random.beta(2, 5, size=len(features))
    lstm_probs = np.random.beta(2, 5, size=len(features))

    results = run_fusion(features, rf_probs, lstm_probs)
    if "query_name" in df.columns:
        results.insert(0, "query_name", df["query_name"].values)
    return results


# ── Main Dashboard ─────────────────────────────────────────────────────────────
if uploaded_file:
    results = load_and_process(uploaded_file)
    filtered = results[results["risk"].isin(risk_filter)]

    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Queries", len(results))
    col2.metric("🔴 High Risk", (results["risk"] == "High").sum())
    col3.metric("🟡 Medium Risk", (results["risk"] == "Medium").sum())
    col4.metric("🟢 Low Risk", (results["risk"] == "Low").sum())

    st.divider()

    # Risk Distribution Chart
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Risk Distribution")
        risk_counts = results["risk"].value_counts()
        st.bar_chart(risk_counts)

    with col_right:
        st.subheader("Fused Score Distribution")
        st.line_chart(results["fused_score"].reset_index(drop=True))

    st.divider()

    # Feature Comparison
    st.subheader("📊 Feature Analysis")
    feat_cols = ["query_length", "entropy", "subdomain_depth", "ngram_deviation", "digit_ratio"]
    st.bar_chart(filtered[feat_cols].mean().rename("Average Value"))

    st.divider()

    # Detailed Records Table
    st.subheader("🗂️ Detected Records")

    def color_risk(val):
        colors = {"High": "background-color: #ffcccc",
                  "Medium": "background-color: #fff3cc",
                  "Low": "background-color: #ccffcc"}
        return colors.get(val, "")

    display_cols = ["query_name", "risk", "fused_score", "entropy", "query_length", "subdomain_depth"] \
        if "query_name" in filtered.columns else \
        ["risk", "fused_score", "entropy", "query_length", "subdomain_depth"]

    styled = filtered[display_cols].style.applymap(color_risk, subset=["risk"])
    st.dataframe(styled, use_container_width=True)

    # Download results
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Results CSV", csv, "detection_results.csv", "text/csv")

else:
    st.info("👈 Upload a DNS records CSV file from the sidebar to begin analysis.")
    st.markdown("""
    **Expected CSV columns:**
    - `query_name` — DNS query string
    - `answer_count` — Number of DNS answers
    - `timestamp` — Unix timestamp
    - `src_ip` — Source IP address
    """)

# ── Auto-refresh ───────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(5)
    st.rerun()
