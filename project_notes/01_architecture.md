# Architecture & Design Decisions

## Project Philosophy

Every design decision in this project follows three rules:

1. **Data flows forward, never backward.** Each script reads exactly what the previous step wrote. No script ever re-reads the raw CSV after `load_data.py` runs. No script ever re-splits the data after `train.py` runs.
2. **One source of truth.** All paths, constants, thresholds, and hyperparameters live in `config.py`. If a number appears in more than one file it is imported from config, never duplicated.
3. **Separate concerns cleanly.** Loading ≠ feature engineering ≠ training ≠ evaluation ≠ presentation. Each of these is its own script. You can swap any one out without touching the others.

---

## Full Directory Layout

```
fraud-detection-engine/
│
├── config.py                          ← single source of truth for all constants
├── requirements.txt                   ← 13 pinned packages
├── Makefile                           ← pipeline automation
├── .gitignore                         ← excludes data/, models/, outputs/
├── README.md                          ← public-facing project summary
├── ML_CONCEPTS.md                     ← deep ML theory reference
│
├── data/
│   ├── raw/
│   │   └── creditcard.csv             ← 144MB Kaggle source (not in Git)
│   └── processed/
│       ├── creditcard_processed.parquet   ← written by load_data.py, read by everything
│       ├── amount_scaler.npy              ← [mean, scale] for Amount StandardScaler
│       └── time_scaler.npy                ← [mean, scale] for Time StandardScaler
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── load_data.py               ← Step 1: validate, scale, save parquet
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py          ← Step 2: engineer 6 new features
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py                   ← Step 3: split, SMOTE, train 3 models
│   │   └── evaluate.py                ← Step 4: metrics, charts, SHAP
│   └── app/
│       ├── __init__.py
│       └── dashboard.py               ← Step 5: Streamlit 2-tab interface
│
├── models/
│   ├── fraud_model.pkl                ← {iso, lr, rf, feature_cols, X_test, y_test, iso_scores}
│   └── shap_explainer.pkl             ← shap.TreeExplainer(rf_model)
│
├── outputs/
│   ├── metrics.json                   ← canonical evaluation numbers
│   ├── monte_carlo_results.json       ← MC simulation results
│   ├── precision_recall_curve.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── shap_summary.png               ← beeswarm: direction + magnitude per feature
│   ├── shap_bar.png                   ← global mean |SHAP| importance
│   └── monte_carlo_distribution.png
│
├── notebooks/
│   ├── 01_eda.ipynb                   ← read-only exploration, never writes files
│   ├── 02_modeling.ipynb              ← post-training analysis, reads saved model
│   └── 03_monte_carlo.ipynb           ← business impact uncertainty simulation
│
└── project_notes/
    ├── 01_architecture.md             ← this file
    ├── 02_data_pipeline.md            ← load_data.py + build_features.py in full detail
    ├── 03_modeling.md                 ← train.py + evaluate.py + SHAP in full detail
    └── 04_dashboard_and_deployment.md ← dashboard.py + deployment options
```

---

## config.py — Every Constant Explained

```python
from pathlib import Path

ROOT_DIR = Path(__file__).parent
```
`__file__` is the absolute path to `config.py` itself. `.parent` is the directory containing it — the project root. Every other path is built relative to this, so the project works from any working directory.

```python
DATA_RAW       = ROOT_DIR / "data/raw/creditcard.csv"
DATA_PROCESSED = ROOT_DIR / "data/processed/creditcard_processed.parquet"
MODEL_PATH     = ROOT_DIR / "models/fraud_model.pkl"
EXPLAINER_PATH = ROOT_DIR / "models/shap_explainer.pkl"
OUTPUTS_DIR    = ROOT_DIR / "outputs"
METRICS_PATH   = ROOT_DIR / "outputs/metrics.json"
```
All I/O paths. Using `pathlib.Path` means path separators work correctly on Windows, Mac, and Linux without string manipulation.

```python
FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount_scaled", "Time_scaled"]
```
The 30 base features used in training: 28 PCA components (V1–V28) plus the two scaled raw features. This list is used in `train.py` as the starting point — the 6 engineered features are appended at runtime.

```python
TARGET_COL = "Class"
```
The label column. Value 1 = fraud, 0 = legitimate.

```python
TEST_SIZE       = 0.2          # 20% of data held out for evaluation
RANDOM_STATE    = 42           # all random operations seeded here — reproducibility
FRAUD_THRESHOLD = 0.3          # decision boundary: flag if fraud_prob >= 0.3
CONTAMINATION   = 0.0017       # Isolation Forest: expected fraud fraction
```

```python
AVG_FRAUD_LOSS      = 122.21   # mean fraud transaction value in the dataset ($)
ANALYST_REVIEW_COST = 15.0     # cost per false-positive review ($)
```
These two constants define the asymmetric cost structure. The ratio (122.21 / 15.0 ≈ 8.1) determines that the model should accept up to 8 false alarms to avoid 1 missed fraud. This ratio directly justifies the 0.30 threshold instead of 0.50.

---

## Makefile — Pipeline Automation

```makefile
setup:    pip install -r requirements.txt
process:  python src/data/load_data.py
features: python src/features/build_features.py
train:    python src/models/train.py
evaluate: python src/models/evaluate.py
app:      streamlit run src/app/dashboard.py
all:      process features train evaluate
```

`make all` is idempotent — running it twice produces the same result. Each target overwrites its outputs. This means if you change `FRAUD_THRESHOLD` in config.py, `make evaluate` re-runs evaluation with the new threshold without retraining.

---

## Data Flow Diagram (detailed)

```
creditcard.csv (284,807 rows × 31 cols)
    │
    ▼ load_data.py
    ├── validates: row count, fraud rate, nulls, amount range, time range
    ├── scales Amount → Amount_scaled (StandardScaler)
    ├── scales Time   → Time_scaled   (StandardScaler)
    ├── saves amount_scaler.npy = [mean, scale]
    ├── saves time_scaler.npy   = [mean, scale]
    └── saves creditcard_processed.parquet (284,807 × 31 cols)
                            │
                            ▼ build_features.py
                            ├── loads parquet + .npy scalers
                            ├── reconstructs original Amount and Time
                            ├── builds: hour_of_day, amount_log, v_sum,
                            │          v_mean, high_amount_flag, night_flag
                            └── overwrites parquet (284,807 × 37 cols)
                                            │
                                            ▼ train.py
                                            ├── stratified 80/20 split
                                            │   → X_train (227,845), X_test (56,962)
                                            ├── Isolation Forest on X_train only
                                            ├── SMOTE on X_train only (10% ratio)
                                            │   → X_train_sm (250,196 rows)
                                            ├── Logistic Regression on X_train_sm
                                            ├── Random Forest on X_train_sm
                                            └── saves fraud_model.pkl
                                                {iso, lr, rf, feature_cols,
                                                 X_test, y_test, iso_scores}
                                                            │
                                                            ▼ evaluate.py
                                                            ├── loads pkl (never touches raw data)
                                                            ├── computes metrics for all 3 models
                                                            ├── generates 5 charts → outputs/
                                                            ├── computes business impact
                                                            ├── saves metrics.json
                                                            ├── runs SHAP on 500-sample
                                                            └── saves shap_explainer.pkl
                                                                        │
                                                    ┌───────────────────┘
                                                    ▼ dashboard.py
                                                    ├── loads: fraud_model.pkl
                                                    ├── loads: shap_explainer.pkl
                                                    ├── loads: metrics.json
                                                    ├── loads: creditcard_processed.parquet
                                                    ├── Tab 1: transaction feed
                                                    └── Tab 2: investigation panel
```

---

## The Test Set Rule

The test set is created **once** in `train.py` and saved inside `fraud_model.pkl`. It is never recreated, re-sampled, or modified by any other script.

This matters because:
- If `evaluate.py` re-split the data, it might get a different random split and metrics would not be comparable across runs
- If any preprocessing were applied to the test set (SMOTE, scaling changes), the evaluation would measure performance on artificially modified data
- The same `X_test` and `y_test` objects appear in the dashboard, the notebooks, and `evaluate.py` — they are always the identical held-out set

The pattern is: **split once, save, load everywhere.**

---

## sys.path Pattern

Every script adds the project root to the Python path:

```python
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
```

This allows `from config import ...` to work regardless of which directory the script is run from. Without this, running `python src/data/load_data.py` from the project root would fail to find `config.py` because Python only searches `sys.path`, not the project root by default.

The `.parent.parent.parent` chain:
- `__file__` = `/project/src/data/load_data.py`
- `.parent` = `/project/src/data/`
- `.parent` = `/project/src/`
- `.parent` = `/project/` ← the project root where `config.py` lives

For `src/app/dashboard.py` it is also three `.parent` levels because `dashboard.py` is two levels deep under `src/`.

---

## Environment

Virtual environment at `.venv/` using Python 3.14.2. The `.venv/` directory is excluded from Git via `.gitignore`.

Key version constraints that matter for this project:
- `shap>=0.46` changed the return format of `TreeExplainer.shap_values()` from a list to a 3D ndarray. The code handles both formats explicitly.
- `scikit-learn>=1.8` deprecated `n_jobs` in `LogisticRegression`. The code accepts the FutureWarning without breaking.
- `pandas>=2.0` changed some default dtypes. The code uses explicit `.astype()` calls where type stability matters.
