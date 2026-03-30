# Fraud Detection Engine

Production-style fraud detection system built on 284,807 real European credit card transactions. Combines Isolation Forest unsupervised anomaly detection with Random Forest supervised classification and SHAP explainability, deployed as a live two-tab Streamlit investigation dashboard that lets analysts investigate individual flagged transactions with per-feature waterfall explanations.

---

## Results

| Metric | Value |
|---|---|
| Dataset | 284,807 transactions, 0.172% fraud rate |
| Model | Random Forest + Isolation Forest ensemble |
| PR-AUC | **0.874** |
| ROC-AUC | 0.965 |
| Fraud caught (Recall) | **86.7%** of fraudulent transactions |
| Precision at threshold 0.30 | **76.6%** |
| F1 score | 0.813 |
| Net value per test set | **$9,998** (fixed avg) / **$25,869** (MC mean) |
| Net value 90% CI (Monte Carlo, 10K sims) | **$6,822 – $67,899** |
| P(net savings > 0) | **100%** across all simulated scenarios |
| False positive rate | 0.046% |

---

## Architecture

```
creditcard.csv (284K transactions)
      ↓
load_data.py — validate nulls, scale Amount + Time, save .npy scalers
      ↓
build_features.py — hour_of_day, amount_log, night_flag, v_sum, v_mean, high_amount_flag
      ↓
train.py — stratified split → SMOTE → Isolation Forest + Logistic Regression + Random Forest
      ↓
evaluate.py — PR-AUC, confusion matrix, business impact, SHAP analysis
      ↓
dashboard.py — Transaction Feed tab + Investigation Panel tab
```

---

## Three Key Technical Decisions

**1. Why PR-AUC over regular ROC-AUC**

With a 0.172% fraud rate, a model that predicts "legitimate" for every transaction achieves 99.83% accuracy and 0.5 ROC-AUC. ROC-AUC counts the enormous legitimate class — where there is no business value — just as heavily as the fraud class. PR-AUC focuses entirely on minority class performance: how precisely and completely are you catching fraud? Every point of PR-AUC improvement represents real fraud caught or false alarms eliminated.

**2. Why SMOTE at 10% fraud ratio, not 50/50 balance**

Full balance (50% fraud, 50% legitimate via SMOTE) creates 140,000+ synthetic fraud examples from 393 real ones. The model learns the synthetic distribution — a smoothed interpolation of the real fraud examples — rather than actual fraud patterns. SMOTE at 10% brings fraud from 0.17% to 10% of training data: enough signal for the model to learn fraud, not so many synthetic samples that it overfits to interpolated patterns.

**3. Why threshold 0.30 not 0.50**

The default 0.50 threshold is appropriate when the cost of a false positive equals the cost of a false negative. Here they are asymmetric: a missed fraud costs $122 in average losses; an unnecessary analyst review costs $15 in analyst time. At 0.30, the model accepts more false alarms in exchange for catching significantly more real fraud. This is a business decision, not a modeling decision — the threshold should move whenever the cost ratio changes.

---

## Setup

```bash
git clone https://github.com/yourusername/fraud-detection-engine
cd fraud-detection-engine
pip install -r requirements.txt
# Download creditcard.csv from kaggle.com/datasets/mlg-ulb/creditcardfraud
# Place at: data/raw/creditcard.csv
make all
make app
```

`make all` runs: `process` → `features` → `train` → `evaluate` in sequence.

---

## Project Structure

```
fraud-detection-engine/
├── config.py                    # All paths, constants, and hyperparameters in one place
├── requirements.txt             # Reproducible environment (13 packages)
├── Makefile                     # make all runs the full pipeline
├── .gitignore                   # Excludes data/, models/, outputs/ from Git
│
├── data/
│   ├── raw/creditcard.csv       # 144MB Kaggle dataset (not in Git)
│   └── processed/               # Parquet dataset + .npy scalers (written by load_data.py)
│
├── src/
│   ├── data/load_data.py        # Validates CSV, scales Amount+Time, saves parquet
│   ├── features/build_features.py  # Adds 6 engineered features, overwrites parquet
│   ├── models/
│   │   ├── train.py             # Stratified split → SMOTE → trains 3 models → saves .pkl
│   │   └── evaluate.py          # Metrics, 5 charts, business impact, SHAP analysis
│   └── app/dashboard.py         # Streamlit 2-tab dashboard
│
├── models/
│   ├── fraud_model.pkl          # All 3 trained models + X_test + y_test (not in Git)
│   └── shap_explainer.pkl       # TreeExplainer for fast dashboard inference (not in Git)
│
├── outputs/                     # Charts and metrics (not in Git)
│   ├── metrics.json
│   ├── precision_recall_curve.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── shap_summary.png
│   └── shap_bar.png
│
└── notebooks/
    ├── 01_eda.ipynb             # Exploratory analysis — 7 cells, read-only
    ├── 02_modeling.ipynb        # Post-training analysis — 6 cells, reads saved model
    └── 03_monte_carlo.ipynb     # Business impact under uncertainty — 10,000 simulations
```

---

