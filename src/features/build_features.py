from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import DATA_PROCESSED, FEATURE_COLS, TARGET_COL  # noqa: E402


ENGINEERED_FEATURES = [
    "hour_of_day",
    "amount_log",
    "v_sum",
    "v_mean",
    "high_amount_flag",
    "night_flag",
]


def main() -> None:
    df = pd.read_parquet(DATA_PROCESSED)

    processed_dir = Path(DATA_PROCESSED).parent
    amount_mean, amount_scale = np.load(processed_dir / "amount_scaler.npy")
    time_mean, time_scale = np.load(processed_dir / "time_scaler.npy")

    original_time = df["Time_scaled"].to_numpy(dtype=float) * float(time_scale) + float(time_mean)
    hour_of_day = (original_time % 86400) / 3600
    df["hour_of_day"] = hour_of_day.astype(float)

    original_amount = df["Amount_scaled"].to_numpy(dtype=float) * float(amount_scale) + float(amount_mean)
    df["amount_log"] = np.log1p(original_amount).astype(float)

    v_cols = [f"V{i}" for i in range(1, 11)]
    df["v_sum"] = df[v_cols].abs().sum(axis=1).astype(float)
    df["v_mean"] = df[v_cols].abs().mean(axis=1).astype(float)

    df["high_amount_flag"] = (original_amount > 500).astype(int)
    df["night_flag"] = ((df["hour_of_day"] >= 0) & (df["hour_of_day"] <= 6)).astype(int)

    print("=== Engineered Feature Preview (first 5 values) ===")
    for f in ENGINEERED_FEATURES:
        print(f"{f}: {df[f].head(5).to_list()}")

    print("\n=== Engineered Feature Correlations with Class ===")
    corrs = {}
    for f in ENGINEERED_FEATURES:
        corrs[f] = float(df[f].corr(df[TARGET_COL]))
    corr_df = (
        pd.DataFrame({"feature": list(corrs.keys()), "corr": list(corrs.values())})
        .assign(abs_corr=lambda d: d["corr"].abs())
        .sort_values("abs_corr", ascending=False)
        .drop(columns=["abs_corr"])
    )
    print(corr_df.to_string(index=False))

    print("\n=== Null counts for engineered features (should be zero) ===")
    print(df[ENGINEERED_FEATURES].isna().sum().to_string())

    feature_cols = FEATURE_COLS + ENGINEERED_FEATURES
    missing = [c for c in feature_cols + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns after feature engineering: {missing}")

    df.to_parquet(DATA_PROCESSED, index=False)
    print(f"\nFinal shape: {df.shape}")


if __name__ == "__main__":
    main()
