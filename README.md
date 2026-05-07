# 🔍 Detection of DNS Tunneling Attacks

> Using Enhanced Statistical and Machine Learning-Based Traffic Feature Analysis

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat&logo=tensorflow)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-green?style=flat&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=flat&logo=streamlit)

---

## 📌 Overview

DNS tunneling is a cyberattack technique that exploits the DNS protocol to exfiltrate data or maintain covert C2 (Command & Control) channels. Traditional signature-based IDS systems fail to detect such attacks.

This project implements a **5-stage detection pipeline** combining statistical feature engineering, Random Forest, and LSTM models with weighted decision fusion — achieving **97.8% accuracy** and just **1.4% false positive rate**.

---

## 🏗️ System Architecture

```
PCAP / Live DNS Traffic
        ↓
  Packet Parser (Scapy)
        ↓
  CDN Whitelist Filter
        ↓
  Feature Extractor (7 features)
        ↓
  Rule Engine (Anomaly Scoring)
        ↓
  ┌─────────────┐
  │  Random     │  +  │  LSTM  │
  │  Forest     │     │ Model  │
  └─────────────┘
        ↓
  Decision Fusion Layer
        ↓
  Risk Classification: High / Medium / Low
        ↓
  Streamlit Dashboard
```

---

## 🔬 7 Statistical Features Extracted

| Feature | Description |
|---|---|
| Query Length | Length of DNS query string |
| Entropy | Shannon entropy of query |
| Subdomain Depth | Number of subdomain levels |
| N-gram Deviation | Deviation from normal n-gram patterns |
| Digit Ratio | Ratio of digits in the query |
| Temporal Behavior | Query frequency over time window |
| Response Size Ratio | Ratio of response to query size |

---

## 📊 Results

| Metric | Value |
|---|---|
| Accuracy | **97.8%** |
| Precision | 0.98 |
| Recall | 0.97 |
| False Positive Rate | **1.4%** |
| Dataset | CIC-Bell-DNS-2021 (120,000 samples) |

**Improvements over baselines:**
- +3.2% over standalone Random Forest
- +10% over entropy-only methods

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **Scapy** – Packet parsing
- **Scikit-learn** – Random Forest classifier
- **TensorFlow / Keras** – LSTM sequential model
- **Streamlit** – Real-time monitoring dashboard
- **tldextract** – Domain parsing

---

## 📁 Project Structure

```
dns-tunneling-detection/
├── src/
│   ├── parser.py            # Packet parsing with Scapy
│   ├── whitelist.py         # CDN whitelist filtering
│   ├── feature_extractor.py # 7-feature extraction
│   ├── rule_engine.py       # Anomaly rule scoring
│   ├── classifier.py        # RF + LSTM classifiers
│   ├── fusion.py            # Decision fusion layer
│   └── dashboard.py         # Streamlit dashboard
├── models/
│   ├── random_forest.pkl    # Trained RF model
│   └── lstm_model.h5        # Trained LSTM model
├── data/                    # Place dataset here (not tracked)
├── outputs/                 # Results and logs
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/dns-tunneling-detection.git
cd dns-tunneling-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the dataset
Download [CIC-Bell-DNS-2021](https://www.unb.ca/cic/datasets/dns.html) and place CSV files inside the `data/` folder.

### 4. Run the detection pipeline
```bash
python src/classifier.py
```

### 5. Launch the dashboard
```bash
streamlit run src/dashboard.py
```

---

## 👥 Team

| Name | Role |
|---|---|
| Praveen Raj K | ML Pipeline & Feature Engineering |
| Pranav V | Packet Parsing & Dashboard |

**Supervisor:** Mrs. G. Smilarubavathy
**Institution:** St. Joseph's Institute of Technology, Chennai

---

## 📚 References

- CIC-Bell-DNS-2021 Dataset — University of New Brunswick
- Bilge et al., 2011 — Passive DNS Analysis
- Aiello et al., 2015 — Query Length & Frequency
- Sheridan et al., 2015 — Entropy-based Detection
- Buczak & Guven, 2016 — ML-based IDS
- Nadler et al., 2019 — Low-rate DNS Tunneling

---

## 📄 License

This project is for academic purposes only.
