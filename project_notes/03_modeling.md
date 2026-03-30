# Modeling — train.py, evaluate.py, SHAP, Monte Carlo

## train.py

### Feature set

```python
ENGINEERED_FEATURES = [
    "hour_of_day", "amount_log", "v_sum",
    "v_mean", "high_amount_flag", "night_flag"
]
feature_cols = FEATURE_COLS + ENGINEERED_FEATURES
# FEATURE_COLS = [V1..V28, Amount_scaled, Time_scaled]  → 30 features
# + 6 engineered                                         → 36 features total
```

Note: the model uses **36** features, not 37. The 37th column is `Class` (the target) which is never passed as a feature.

---

### Step 1 — Stratified Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

**Output sizes:**
```
Train: 227,845 rows  (394 fraud,    227,451 legitimate)
Test:   56,962 rows  ( 98 fraud,     56,864 legitimate)
```

**Why stratify=y:** With 492 total fraud cases, a random split has non-trivial variance in how many fraud cases end up in each set. Stratification guarantees exactly 80%/20% of fraud cases in each split (394 train / 98 test), matching the overall dataset fraud rate of 0.172% in both sets.

**Why random_state=42:** Full reproducibility. Every run of `train.py` produces identical splits, identical models, identical metrics. Without this, metrics would drift between runs and you could not detect regressions.

**The golden rule:** `X_test` and `y_test` are saved into `fraud_model.pkl` and loaded by `evaluate.py`. The test set is never recreated. Split happens once, here, and never again.

---

### Step 2 — Isolation Forest (Unsupervised Baseline)

```python
iso_model = IsolationForest(
    contamination=CONTAMINATION,  # 0.0017
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
iso_model.fit(X_train)  # no y_train — completely unsupervised
iso_scores = iso_model.decision_function(X_test)
```

**How Isolation Forest works:**

For each of the 100 trees:
1. Draw a subsample of 256 rows from `X_train` (default `max_samples`)
2. Randomly select a feature
3. Randomly select a split value between the feature's min and max
4. Recurse until every point is in its own leaf

Points that isolate in fewer splits = more anomalous. The `decision_function` returns the average path length normalised to a score:
- Score near +0.5 = definitely normal
- Score near 0 = ambiguous
- Score near -0.5 = definitely anomalous

**`contamination=0.0017`:** Sets the threshold such that 0.17% of the training set is classified as anomalous — matching the known fraud rate. This affects the internal threshold used by `predict()`, but `decision_function()` returns raw scores regardless.

**Training on X_train only:** Isolation Forest sees no labels. It detects statistical rarity, not learned fraud patterns. This is the correct setup — it gives a genuine unsupervised baseline. If it saw X_test during fitting, evaluation scores would be inflated by distribution leakage.

**Expected output:**
```
Avg anomaly score (fraud):      more negative (more anomalous)
Avg anomaly score (legitimate): closer to +0.25 (more normal)
```

If fraud scores are NOT more negative than legitimate, the features do not contain sufficient anomaly signal and feature engineering needs revisiting.

---

### Step 3 — SMOTE

```python
smote = SMOTE(
    sampling_strategy=0.1,  # bring fraud to 10% of training
    k_neighbors=5,
    random_state=42
)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
```

**Before SMOTE:**
```
Class 0 (legitimate): 227,451
Class 1 (fraud):          394
Ratio: 578:1
```

**After SMOTE:**
```
Class 0 (legitimate): 227,451  (unchanged)
Class 1 (fraud):       22,745  (394 real + 22,351 synthetic)
Ratio: 10:1
```

**How SMOTE creates synthetic fraud cases:**

For each of the 394 real fraud examples:
1. Find the 5 nearest fraud neighbors in 36-dimensional feature space (Euclidean distance)
2. Randomly select one neighbor
3. Generate a new point anywhere on the line segment between the original and that neighbor:
   ```
   synthetic = original + rand(0,1) × (neighbor - original)
   ```

The synthetic points are **interpolations** within the fraud region of feature space. They are not copies, they are not extrapolations, they are linear combinations of real fraud feature vectors.

**Why 10% not 50/50:**

At `sampling_strategy=1.0` (50/50 balance): 22,745 synthetic samples would need to become 227,451 — that's 576 synthetic examples per real fraud case. The model learns the smooth interpolated manifold of the 394 real cases, not actual fraud patterns. PR-AUC degrades.

At `sampling_strategy=0.1` (10%): 57 synthetic examples per real case. Enough to give the model fraud signal, not so many that synthetic artifacts dominate.

**Critical rule:** SMOTE is applied to `X_train` only. `X_test` remains the original unmodified held-out data. Applying SMOTE to test data would mean evaluating on artificial data, which would inflate recall metrics.

---

### Step 4 — Logistic Regression Baseline

```python
lr_model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42,
    n_jobs=-1
)
lr_model.fit(X_train_sm, y_train_sm)
```

**Role:** Interpretable linear baseline. If Random Forest cannot significantly beat Logistic Regression on PR-AUC, it suggests either:
- The fraud signal is approximately linear in feature space (rare)
- The features don't have enough non-linear structure for RF to exploit
- Something is wrong with training (data leakage, incorrect labels, etc.)

**`class_weight='balanced'`:** Even after SMOTE the data is not 50/50. This parameter makes the loss function count each fraud example `(n_samples / (n_classes × fraud_count))` times more than each legitimate example. It complements SMOTE rather than replacing it.

**`max_iter=1000`:** Default is 100, which is often insufficient for convergence on high-dimensional data with 36 features. 1000 iterations ensures the solver converges.

**Validation AUC:** ~0.974. Note this is ROC-AUC — Logistic Regression achieves high ROC-AUC because it correctly clears almost all legitimate transactions. Its PR-AUC is much lower (0.785) because it generates many false alarms when threshold is lowered to catch fraud.

---

### Step 5 — Random Forest

```python
rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_sm, y_train_sm)
```

**How Random Forest builds each tree:**

1. Bootstrap sample: draw 227,845 rows WITH replacement from X_train_sm → ~63% unique rows
2. At each node split: consider only `sqrt(36)` ≈ 6 randomly chosen features
3. Find the best split among those 6 features (maximise Gini impurity reduction)
4. Recurse until leaves are pure or minimum sample threshold

Repeat for all 100 trees.

**Prediction:**
```python
fraud_probability = (number of trees voting fraud) / 100
```

**Why it beats Logistic Regression here:**
- Fraud patterns are non-linear in feature space (combinations of features matter)
- The interaction between `night_flag` and `v_sum` is not captured by a linear model
- Random Forest discovers these interactions through hierarchical splits

**`n_jobs=-1`:** Parallelises tree building across all CPU cores. Each tree is independent — trivially parallelisable.

**Validation AUC:** ~0.965 (ROC-AUC). RF's PR-AUC (0.874) substantially exceeds LR's PR-AUC (0.785), confirming non-linear structure in the fraud patterns.

---

### Step 6 — Saving the Model Bundle

```python
bundle = {
    'isolation_forest':    iso_model,
    'logistic_regression': lr_model,
    'random_forest':       rf_model,
    'feature_cols':        feature_cols,   # list of 36 feature names
    'X_test':              X_test,         # 56,962 × 36 DataFrame
    'y_test':              y_test,         # 56,962 Series
    'iso_scores_test':     iso_scores      # 56,962 float array
}
joblib.dump(bundle, MODEL_PATH)
```

**Why one dict, not separate files:**
- Atomic: loading one file gives you everything needed for evaluation or inference
- Consistent: X_test and y_test are guaranteed to match the models they were evaluated with
- Portable: copying one file transfers the complete model state

**`joblib` not `pickle`:** joblib uses memory-mapped numpy arrays which are dramatically faster for large numpy arrays (X_test is a 56,962 × 36 float64 array = ~16MB).

---

## evaluate.py

### Loading and extracting

```python
bundle       = joblib.load(MODEL_PATH)
rf_model     = bundle["random_forest"]
lr_model     = bundle["logistic_regression"]
iso_model    = bundle["isolation_forest"]
feature_cols = bundle["feature_cols"]
X_test       = bundle["X_test"]
y_test       = bundle["y_test"]
iso_scores   = bundle["iso_scores_test"]
```

This is the only file evaluate.py touches. It never reads parquet, never reads the raw CSV, never touches the training data.

### Isolation Forest score conversion

```python
def _iso_probs(iso_scores):
    shifted = -iso_scores         # flip sign: more anomalous → higher value
    return (shifted - shifted.min()) / (shifted.max() - shifted.min() + 1e-12)
```

Converts raw IF scores ([-0.5, 0.5]) to pseudo-probabilities [0, 1] via min-max normalisation. These are NOT calibrated probabilities — they are monotone rank-preserving transforms used only for threshold comparison and PR curve plotting. The `1e-12` prevents division by zero if all scores are identical.

### Core metrics computation

```python
def _metrics_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rec, prec)
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "pr_auc":    pr_auc,
        "roc_auc":   roc_auc_score(y_true, y_prob),
    }
```

**Threshold-dependent metrics** (precision, recall, F1): computed by thresholding `y_prob` at `FRAUD_THRESHOLD=0.30`
**Threshold-independent metrics** (PR-AUC, ROC-AUC): computed from the full probability curve, summarise performance across all possible thresholds

### Results at threshold 0.30

```
Model                Precision   Recall    F1      PR-AUC   ROC-AUC
Random Forest          0.766     0.867   0.813     0.874     0.965
Logistic Regression    0.028     0.918   0.055     0.785     0.974
Isolation Forest       0.017     0.878   0.033     0.155     0.954
```

**Interpreting LR vs RF:**
- LR has higher ROC-AUC (0.974) but much lower precision (0.028)
- At threshold 0.30, LR flags almost everything as fraud — high recall, catastrophic false alarm rate
- RF has lower ROC-AUC but dramatically higher precision — it is much more selective
- PR-AUC tells the real story: RF (0.874) >> LR (0.785) — RF maintains high precision across recall levels

**Interpreting IF:**
- PR-AUC of 0.155 vs random baseline of 0.00172 — better than random but far below supervised models
- Expected: unsupervised anomaly detection without labels will lose to supervised learning given 393 labelled fraud cases

### Confusion matrix (RF at threshold 0.30)

```
                    Predicted Legit    Predicted Fraud
Actual Legitimate       56,838 (TN)          26 (FP)
Actual Fraud                13 (FN)          85 (TP)
```

- **85 true positives:** fraud flagged correctly → analyst investigates → money recovered
- **13 false negatives:** fraud missed → transaction processes → average $122 lost each
- **26 false positives:** legitimate transactions flagged → analyst reviews → $15 wasted each
- **56,838 true negatives:** legitimate cleared → no cost

**False positive rate:** 26 / (26 + 56,838) = **0.046%** — extremely low. An analyst queue of 111 flagged transactions per day contains ~97% real fraud.

### Business impact calculation

```python
value_caught = TP * AVG_FRAUD_LOSS   = 85 × $122.21 = $10,387.85
value_missed = FN * AVG_FRAUD_LOSS   = 13 × $122.21 = $1,588.73
review_cost  = FP * ANALYST_REVIEW_COST = 26 × $15.00 = $390.00
net_value    = value_caught - review_cost = $10,387.85 - $390.00 = $9,997.85
```

**Limitation of fixed-average calculation:** `AVG_FRAUD_LOSS = $122.21` is the dataset-wide average across all 492 fraud cases. The actual caught fraud cases (85 specific transactions) have their own amount distribution. The Monte Carlo simulation (notebook 03) addresses this by sampling from the actual fraud amount distribution rather than using the global average.

### metrics.json

```json
{
  "pr_auc":           0.8736,
  "roc_auc":          0.9646,
  "precision":        0.7658,
  "recall":           0.8673,
  "f1":               0.8134,
  "threshold":        0.3,
  "true_positives":   85,
  "false_positives":  26,
  "false_negatives":  13,
  "true_negatives":   56838,
  "value_caught":     10387.85,
  "value_missed":     1588.73,
  "review_cost":      390.0,
  "net_value":        9997.85
}
```

The dashboard loads this file to populate the metric cards. It is the single canonical source of model performance numbers.

---

## SHAP Analysis

### Why SHAP over standard feature importance

Random Forest's built-in feature importance answers: "across all 100 trees, how much did splits on this feature reduce Gini impurity?"

This has known problems:
- Biased toward high-cardinality continuous features
- Does not show direction (does high `night_flag` push toward fraud or away?)
- Cannot explain individual predictions
- Correlated features split importance arbitrarily

SHAP (SHapley Additive exPlanations) answers: "for this specific transaction, how much did each feature's value push the fraud probability above or below the average prediction?"

### TreeSHAP — computational approach

`shap.TreeExplainer(rf_model)` encodes the Random Forest structure for exact SHAP computation. For each tree, TreeSHAP computes exact Shapley values in O(TLD²) time where:
- T = number of trees (100)
- L = max leaves per tree
- D = max depth per tree

This is polynomial (not exponential like brute-force Shapley), making it feasible for Random Forest.

### Stratified 500-sample

```python
fraud_idx = np.where(y_test == 1)[0]   # 98 fraud cases
legit_idx = np.where(y_test == 0)[0]   # 56,864 legitimate cases

n_fraud = len(fraud_idx)               # 98
n_legit = 500 - n_fraud                # 402
legit_sample = rng.choice(legit_idx, size=n_legit, replace=False)
sample_idx = np.concatenate([fraud_idx, legit_sample])
```

**Why 500 not all 56,962:** SHAP computation on Random Forest is O(n_samples × trees × depth). On 56,962 samples with 100 trees this takes 20+ minutes. On 500 samples: ~30 seconds. The global importance picture (beeswarm, bar chart) is stable at 500 samples — the mean |SHAP| values converge well before 500.

**Why include all fraud cases:** The 98 fraud cases are rare and all informative for understanding the fraud SHAP signature. We include all of them and fill the remaining 402 slots with random legitimate cases to get a representative mixed sample.

### SHAP return format — version handling

```python
shap_values = explainer.shap_values(X_sample)

# SHAP ≥0.46 returns ndarray (n_samples, n_features, n_classes)
# Older versions return list [class0_array, class1_array]
if isinstance(shap_values, list):
    sv_fraud = shap_values[1]             # list: index 1 = fraud class
elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    sv_fraud = shap_values[:, :, 1]       # ndarray: last dim 1 = fraud class
else:
    sv_fraud = shap_values
```

SHAP 0.51.0 (installed in this project) uses the 3D ndarray format. The branching logic ensures the code works if shap is upgraded or downgraded.

### Top 5 features by mean |SHAP|

```
V14       0.077450   ← strongest global predictor
V4        0.067917
V12       0.062919
V10       0.059623
V11       0.038804
```

Notably absent from top 5: `night_flag`. Its global mean |SHAP| is lower because it applies only to ~20% of transactions (those at night). For those transactions, its SHAP value is high — but averaged across all 500 samples it is diluted. The beeswarm plot makes this clear: `night_flag` has a cluster of high positive SHAP values (red dots) at the right side.

### `explain_transaction()` — individual transaction explanation

```python
def explain_transaction(transaction_df, rf_model, explainer, feature_cols):
    x = transaction_df[feature_cols]
    fraud_prob = float(rf_model.predict_proba(x)[0][1])

    sv = explainer.shap_values(x)
    # [version-compatible extraction of fraud-class SHAP values]
    sv_fraud = ...  # shape: (n_features,)

    shap_df = pd.DataFrame({"feature": feature_cols, "shap_value": sv_fraud})
    shap_df = shap_df.sort_values("shap_value", ascending=False)

    top3 = shap_df[shap_df["shap_value"] > 0].head(3)["feature"].tolist()
    risk_factors = [reasons_map.get(f, f"Feature {f} shows anomalous value") for f in top3]

    return {
        "fraud_probability": round(fraud_prob * 100, 1),
        "risk_factors":      risk_factors,
        "shap_values_dict":  dict(zip(shap_df["feature"], shap_df["shap_value"])),
    }
```

**The `reasons_map` dict:** Maps feature names to plain English explanations for the dashboard. Top 3 *positive* SHAP features are selected — these are the features actively pushing toward fraud for this specific transaction. Features with negative SHAP are pulling *away* from fraud and are shown as mitigating factors in the waterfall chart rather than risk factors.

---

## Monte Carlo Business Impact Simulation

### Why the fixed estimate is wrong

`metrics.json` reports `net_value = $9,997.85`. This uses:
- 85 caught fraud × $122.21 (dataset-wide average) = $10,387.85
- 26 false alarms × $15.00 (assumed fixed) = $390.00

**Problem:** The 85 caught fraud cases have their own amount distribution. Some caught cases are $5 card-testing probes. Others are $2,000 legitimate-looking transactions. Using the global average hides this variance entirely.

### Log-normal distribution fit

Fraud amounts are strictly positive and right-skewed — the log-normal distribution is the natural parametric choice:

```python
log_amounts = np.log(fraud_amounts)
log_mean    = log_amounts.mean()    # fitted μ of log-normal
log_std     = log_amounts.std()     # fitted σ of log-normal
```

The log-normal PDF is: `f(x) = (1/xσ√2π) × exp(-(ln(x)-μ)²/2σ²)`

Implied mean of fitted distribution: `exp(μ + σ²/2)` — this accounts for the asymmetric heavy right tail.

### Simulation structure

```python
# 10,000 independent scenarios
for each simulation i:
    fraud_loss_i   ~ LogNormal(log_mean, log_std) × TP samples → sum
    review_cost_i  ~ Normal(15, 3) × FP samples                → sum
    net_value_i    = fraud_loss_i - review_cost_i
```

Each scenario independently samples the full uncertainty of both cost inputs. The output is 10,000 net value realisations.

### Results interpretation

```
Expected (mean):   $25,869   ← higher than $9,998 because log-normal mean > arithmetic mean
Median:            $17,301   ← median is more robust to large outlier catches
P5:                 $6,822   ← worst realistic scenario (5th percentile)
P95:               $67,899   ← best realistic scenario (95th percentile)
P(positive):         100%    ← model always recovers positive value
```

**Why MC mean ($25,869) >> fixed estimate ($9,998):**

The fixed estimate uses $122.21 (arithmetic mean of ALL 492 fraud cases including many small card-testing transactions). The log-normal simulation samples from the distribution of actual fraud amounts weighted by the log-normal fit — which has a heavier right tail than the simple average implies. The 85 caught cases are a non-random sample; the model tends to catch higher-value anomalous transactions.

**Why the std deviation is so large ($43,044):**

Fraud loss is inherently volatile. A single $25,000 transaction caught or missed can swing the outcome by $25k. The wide CI is not a model quality problem — it accurately reflects the economic reality of fraud loss variance.

### Sensitivity decomposition

The notebook's Cell 5 sensitivity chart shows:
- **Fraud loss variance** is the dominant driver of outcome uncertainty
- **Review cost variance** (Normal(15,3)) contributes minimally — $3 std on $15 mean is low relative to fraud loss variance

This tells a stakeholder: the main uncertainty in the business case is how much each fraud transaction is worth, not how long analysts take to review cases.
