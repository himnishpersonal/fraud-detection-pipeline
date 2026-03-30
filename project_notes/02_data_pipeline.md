# Data Pipeline — load_data.py & build_features.py

## The Raw Dataset

**Source:** Kaggle — Credit Card Fraud Detection (ULB Machine Learning Group)
**File:** `data/raw/creditcard.csv`
**Size:** ~144MB on disk, 284,807 rows × 31 columns

### Column inventory

| Column | Type | Description |
|---|---|---|
| `Time` | float64 | Seconds elapsed since the first transaction in the dataset (0 to 172,792) |
| `V1`–`V28` | float64 | PCA-transformed features. Original features are confidential; PCA was applied by the dataset creators to anonymise them |
| `Amount` | float64 | Transaction value in euros (0.00 to 25,691.16) |
| `Class` | int64 | Target label: 1 = fraud, 0 = legitimate |

### Why PCA was already applied

The dataset creators anonymised the original transaction features (merchant, location, card type, etc.) using Principal Component Analysis before releasing the data publicly. The result is 28 orthogonal components that capture the variance of the original feature space but are not individually interpretable. This means:

- V1–V28 are already zero-mean and have comparable variances — they do not need standardisation
- The components are uncorrelated by construction (orthogonality of PCA)
- Feature importance (SHAP values, RF importance) tells you *which* PCA component mattered, but not *what* it represents in human terms
- We can still engineer new features from `Time` and `Amount` which were NOT PCA-transformed

### Class imbalance

```
Class 0 (legitimate): 284,315  →  99.827%
Class 1 (fraud):          492  →   0.173%
Ratio: 578:1
```

This is one of the most extreme class imbalances in any public ML dataset. It makes accuracy worthless as a metric and requires explicit handling in both training (SMOTE) and evaluation (PR-AUC).

---

## load_data.py — Line by Line

### sys.path setup

```python
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
```
Inserts the project root at position 0 in Python's module search path, giving `from config import ...` the highest priority resolution.

### Reading the CSV

```python
df = pd.read_csv(DATA_RAW)
```
Reads all 284,807 rows into memory. At ~144MB on disk this expands to ~350MB in RAM as a pandas DataFrame (dtype overhead). Takes ~15-20 seconds on a standard laptop due to CSV parsing — this is why we save to parquet immediately after.

### Validation block

Six checks run before any transformation:

**1. Row count**
```python
total_rows = len(df)  # expect 284,807
```
Catches truncated downloads. If you downloaded a partial file this fails immediately instead of silently training on incomplete data.

**2. Fraud count and rate**
```python
fraud_count = int((df[TARGET_COL] == 1).sum())
fraud_pct   = fraud_count / total_rows * 100
# expect 492 and 0.172%
```

**3. Null count**
```python
nulls_per_col = df.isna().sum()
# expect all zeros — this dataset has no missing values
```
Any non-zero here would require imputation strategy decisions before proceeding. This dataset is clean.

**4. Amount range**
```python
amount_min    = df["Amount"].min()    # $0.00
amount_median = df["Amount"].median() # $22.00
amount_max    = df["Amount"].max()    # $25,691.16
```
The extreme max ($25,691) vs median ($22) confirms heavy right-skew. This directly motivates `amount_log` in feature engineering.

**5. Time range**
```python
time_min = df["Time"].min()  # 0 seconds
time_max = df["Time"].max()  # 172,792 seconds ≈ 48 hours
```
The dataset covers two days of transactions. Raw Time is not useful as a feature because it encodes position in the 48-hour recording window, not time of day. This motivates the `hour_of_day` feature in build_features.py.

**6. Class distribution with percentages**
```python
class_counts = df[TARGET_COL].value_counts(dropna=False)
class_pcts   = df[TARGET_COL].value_counts(normalize=True) * 100
```

### Scaling Amount

```python
amount_scaler = StandardScaler()
df["Amount_scaled"] = amount_scaler.fit_transform(df[["Amount"]]).astype(float)
df = df.drop(columns=["Amount"])
```

`StandardScaler` computes: `z = (x - mean) / std`

For Amount:
- `mean` ≈ $88.35
- `std`  ≈ $250.12

A $500 transaction becomes: `(500 - 88.35) / 250.12 ≈ 1.645`

**Why scale Amount but not V1–V28?** V1–V28 are already PCA-transformed and have unit-scale variance. Amount ranges from $0 to $25,691 — on a raw basis it would dominate distance calculations and gradient steps in scale-sensitive models (Logistic Regression, Isolation Forest) simply due to its magnitude, not its predictive content.

### Saving the scaler parameters

```python
np.save(processed_dir / "amount_scaler.npy",
        np.array([amount_scaler.mean_[0], amount_scaler.scale_[0]]))
```

Saves a 2-element array: `[mean, scale]`. This is the minimal information needed to **invert** the scaling later:

```python
original_amount = Amount_scaled × scale + mean
```

Why not pickle the entire StandardScaler object? The `.npy` approach is:
- Smaller (16 bytes vs ~1KB)
- Transparent (you can inspect the values directly)
- Dependency-free (no sklearn needed to load it)
- Explicit (forces build_features.py to show the inverse transform formula)

### Scaling Time

Identical process to Amount. Time ranges 0–172,792 seconds:
- `mean` ≈ 94,813 seconds
- `std`  ≈ 47,488 seconds

### Saving to Parquet

```python
df.to_parquet(DATA_PROCESSED, index=False)
```

**Why parquet over CSV?**

| Property | CSV | Parquet |
|---|---|---|
| Read speed | ~15-20s | ~0.3s |
| Write size | ~144MB | ~25MB |
| Type preservation | No (everything is string on read) | Yes (float64 stays float64) |
| Column pruning | No (reads all columns) | Yes (can read only needed columns) |

Every subsequent script (`build_features.py`, `train.py`, `dashboard.py`) reads from this parquet. The 50x read speed improvement means the dashboard doesn't freeze on load.

---

## build_features.py — The 6 Engineered Features

### The inverse transform pattern

Before building features that depend on original values, we undo the StandardScaler:

```python
amount_mean, amount_scale = np.load(processed_dir / "amount_scaler.npy")
time_mean,   time_scale   = np.load(processed_dir / "time_scaler.npy")

original_time   = df["Time_scaled"]   * time_scale   + time_mean
original_amount = df["Amount_scaled"] * amount_scale + amount_mean
```

This is the exact inverse of `StandardScaler.transform()`. The result:
- `original_time` ranges 0 to 172,792 (verified by checking first few rows against raw CSV)
- `original_amount` ranges 0 to 25,691

### Feature 1: `hour_of_day`

```python
hour_of_day = (original_time % 86400) / 3600
```

**Step by step:**
1. `original_time % 86400` — modulo 86,400 (seconds in a day) wraps time into a 24-hour window. A transaction at 100,000 seconds: `100000 % 86400 = 13600` (second day, 3.78 hours in)
2. `/ 3600` — converts seconds to hours. Result is a float in [0.0, 24.0)

**Why this matters:** Raw `Time` is position in the 48-hour recording window. `hour_of_day` extracts the *time of day* — the circadian signal. Fraud clusters heavily at 2–4am. The model cannot learn this from raw Time without this transformation.

**Type:** float, not integer. Hour 3.5 means 3:30am — the continuous value carries more information than rounding to the nearest hour.

### Feature 2: `amount_log`

```python
amount_log = np.log1p(original_amount)
```

**Why `log1p` not `log`:** `np.log1p(x)` computes `log(1 + x)`. This handles `x=0` gracefully (`log1p(0) = 0`). Some transactions have Amount = 0.00 (card-testing probes); `np.log(0)` would return `-inf`.

**Why log-transform amount:** The raw amount distribution is extremely right-skewed:
- Median: $22
- Mean: $88
- Max: $25,691

After log1p:
- $0 → 0.0
- $22 → 3.14
- $100 → 4.62
- $25,691 → 10.15

The model can now distinguish between $5 and $50 as clearly as it distinguishes between $5,000 and $50,000. Without this, the high-value end of the distribution dominates the feature's contribution.

### Feature 3: `v_sum`

```python
v_sum = df[['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10']].abs().sum(axis=1)
```

**What it captures:** The aggregate magnitude of behavioral anomaly across the first 10 PCA components. If a transaction is unusual in *multiple* dimensions simultaneously, `v_sum` will be large even if no single V feature is extreme.

**Why absolute values:** PCA components can be positive or negative; a strongly negative V3 is equally anomalous as a strongly positive V3. Without `abs()`, large positive and negative values would cancel and the sum would understate the overall anomaly.

**Why V1–V10 not all 28:** The first PCA components capture the most variance in the original data (by definition of PCA). V11–V28 capture progressively less variance and add more noise than signal to an aggregate measure.

### Feature 4: `v_mean`

```python
v_mean = df[['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10']].abs().mean(axis=1)
```

`v_mean = v_sum / 10`. These are perfectly correlated on this dataset (same columns, linear transformation), so they provide identical information in that sense — but they are both kept because:
- Tree-based models sometimes make different splits on the sum vs the mean depending on other feature scales
- Keeping both adds minimal overhead and gives the model flexibility to find the more useful threshold

### Feature 5: `high_amount_flag`

```python
high_amount_flag = (original_amount > 500).astype(int)
```

**Why a binary flag when we already have `amount_log`?** `amount_log` provides a smooth, continuous signal across the full range. `high_amount_flag` provides an explicit categorical signal at the $500 boundary — where the fraud pattern visibly changes from card-testing small charges to high-value purchase fraud. A tree-based model can find this boundary itself, but making it explicit helps:
1. Logistic Regression, which needs explicit non-linearity
2. Speed up Random Forest tree building (one less split to discover)

**Why $500 and not some other value?** The EDA Cell 3 plot shows the amount distribution by class. The crossover point where fraud starts appearing at meaningful rates for larger amounts is around $500. This is a domain-informed threshold.

### Feature 6: `night_flag`

```python
night_flag = ((df["hour_of_day"] >= 0) & (df["hour_of_day"] <= 6)).astype(int)
```

**Why this is the most important engineered feature:** The EDA time pattern shows that the fraud rate between midnight and 6am is 3–4× the daytime rate. `hour_of_day` captures this continuously, but `night_flag` makes the overnight window an explicit binary — giving the model a direct handle on the highest-risk period without needing to discover the 0–6 boundary through splits.

**In practice:** SHAP analysis confirms that `night_flag=1` consistently pushes fraud probability up, and it ranks in the top 5 global features by mean |SHAP| in most runs of this dataset.

### Post-feature validation

```python
for f in ENGINEERED_FEATURES:
    print(f"{f}: {df[f].head(5).to_list()}")
```
Visual check: `hour_of_day` values at the start of the dataset should all be near 0 (transactions starting at midnight). `night_flag` should be 1 for most early rows.

```python
corrs = {f: float(df[f].corr(df[TARGET_COL])) for f in ENGINEERED_FEATURES}
```
Correlation with Class. Sorted by absolute value:
- `v_sum` and `v_mean` rank highest (~0.23) — they are the most discriminative of the 6
- `night_flag` ranks third (~0.025) — small absolute correlation but high SHAP because it interacts with other features
- `hour_of_day`, `amount_log`, `high_amount_flag` follow

Correlation measures linear relationship only. SHAP values (computed later) better capture non-linear feature importance for tree models.

### Final parquet overwrite

```python
df.to_parquet(DATA_PROCESSED, index=False)
print(f"Final shape: {df.shape}")  # (284807, 37)
```

The processed parquet now has 37 columns: 28 V features + 2 scaled raw features + 1 target + 6 engineered features. This is the version used by every downstream script.

---

## Why These Specific 6 Features

The 6 features address gaps that V1–V28 cannot fill:

| Gap | Why V1–V28 can't fill it | Feature that fills it |
|---|---|---|
| Time of day | `Time` is 48-hour position, not clock time | `hour_of_day` |
| Overnight window | Continuous hour doesn't highlight 0–6 explicitly | `night_flag` |
| Amount skew | `Amount_scaled` is linear, right tail dominates | `amount_log` |
| High-value threshold | No explicit categorical boundary | `high_amount_flag` |
| Multi-component anomaly | Individual V features miss co-occurring anomalies | `v_sum`, `v_mean` |

None of these introduce data leakage — they are all derived from features that would be available at transaction time in a real deployment.
