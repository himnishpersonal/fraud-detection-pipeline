from pathlib import Path

ROOT_DIR = Path(__file__).parent

DATA_RAW       = ROOT_DIR / "data/raw/creditcard.csv"
DATA_PROCESSED = ROOT_DIR / "data/processed/creditcard_processed.parquet"
MODEL_PATH     = ROOT_DIR / "models/fraud_model.pkl"
EXPLAINER_PATH = ROOT_DIR / "models/shap_explainer.pkl"
OUTPUTS_DIR    = ROOT_DIR / "outputs"
METRICS_PATH   = ROOT_DIR / "outputs/metrics.json"

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount_scaled", "Time_scaled"]
TARGET_COL   = "Class"

TEST_SIZE        = 0.2
RANDOM_STATE     = 42
FRAUD_THRESHOLD  = 0.3
CONTAMINATION    = 0.0017

AVG_FRAUD_LOSS      = 122.21
ANALYST_REVIEW_COST = 15.0
