# ML Concepts — Fraud Detection Engine

A thorough reference for every machine learning concept used in this project. Written so that both a beginner who wants to understand why each decision was made and an interviewer who wants to probe technical depth will find what they need.

---

## Table of Contents

1. [Class Imbalance — Why Accuracy Fails](#1-class-imbalance--why-accuracy-fails)
2. [Evaluation Metrics — Precision, Recall, F1, PR-AUC, ROC-AUC](#2-evaluation-metrics--precision-recall-f1-pr-auc-roc-auc)
3. [Stratified Train/Test Split](#3-stratified-traintest-split)
4. [StandardScaler — Why Amount and Time Need Scaling](#4-standardscaler--why-amount-and-time-need-scaling)
5. [Feature Engineering — The 6 New Features](#5-feature-engineering--the-6-new-features)
6. [SMOTE — Synthetic Minority Oversampling](#6-smote--synthetic-minority-oversampling)
7. [Isolation Forest — Unsupervised Anomaly Detection](#7-isolation-forest--unsupervised-anomaly-detection)
8. [Random Forest — Bagging, Feature Subsampling, Voting](#8-random-forest--bagging-feature-subsampling-voting)
9. [SHAP Values — Game Theory Meets Model Explanation](#9-shap-values--game-theory-meets-model-explanation)
10. [The Precision-Recall Tradeoff and Threshold as a Business Decision](#10-the-precision-recall-tradeoff-and-threshold-as-a-business-decision)

---

## 1. Class Imbalance — Why Accuracy Fails

### The problem

The dataset has 284,807 transactions. Only 492 (0.172%) are fraudulent.

Write this single-line model:

```python
def predict(transaction):
    return "legitimate"
```

Its accuracy: **99.828%**. It has never seen a fraud case, learned nothing, and would cost the business hundreds of thousands of dollars in undetected losses — yet accuracy calls it nearly perfect.

### Why this happens

Accuracy = (correct predictions) / (total predictions). When one class is 583 times larger than the other, predicting the majority class always dominates the numerator. The metric cannot distinguish between "the model learned to identify fraud" and "the model learned to ignore fraud."

### The formal definition of why this breaks

For a dataset with fraud rate π = 0.00172:

- A model that always predicts legitimate gets accuracy = 1 - π = 99.828%
- A perfect fraud detector gets accuracy = 1 - π + π = 100%, only 0.172% better
- The accuracy range the entire fraud detection problem lives in is [99.828%, 100%]
- This 0.172% range is invisible to accuracy as a signal

### What the right frame is

Every fraud transaction has a dollar value (average $122 in this dataset). The business does not care about accuracy; it cares about:

- How much fraud did we catch (in dollars)?
- How much analyst time did we waste on false alarms (in hours × cost)?

This reframes the problem as an expected-value calculation, not a classification accuracy problem. See Section 10 for how the threshold translates model scores into this business calculation.

---

## 2. Evaluation Metrics — Precision, Recall, F1, PR-AUC, ROC-AUC

### The confusion matrix — the foundation

Every classification result falls into one of four cells:

```
                    Predicted Legitimate    Predicted Fraud
Actual Legitimate       True Negative (TN)     False Positive (FP)
Actual Fraud            False Negative (FN)    True Positive (TP)
```

- **True Positive (TP):** Fraud correctly flagged. Business value: ~$122 saved.
- **False Positive (FP):** Legitimate transaction wrongly flagged. Cost: ~$15 analyst review.
- **False Negative (FN):** Fraud missed entirely. Cost: ~$122 lost.
- **True Negative (TN):** Legitimate transaction correctly cleared. Value: operational efficiency.

### Precision

```
Precision = TP / (TP + FP)
```

Of all the transactions the model flagged as fraud, what fraction actually were fraud?

High precision = few false alarms. An analyst receiving 100 flags trusts that most are real fraud.
Low precision = many false alarms. Analysts start ignoring the system because it cries wolf.

In this project at threshold 0.30: precision tells you what fraction of the investigation queue is genuinely fraudulent.

### Recall (Sensitivity)

```
Recall = TP / (TP + FN)
```

Of all the actual fraud cases that occurred, what fraction did the model catch?

High recall = catches most fraud. Low recall = misses most fraud.

The recall-precision tradeoff: lowering the threshold catches more fraud (higher recall) but also flags more legitimate transactions (lower precision). There is no free lunch.

### F1 Score

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

The harmonic mean of precision and recall. Useful as a single-number summary when you want to balance both, but it gives equal weight to precision and recall. In fraud detection, the costs of FP ($15) and FN ($122) are not equal — so F1 can be misleading as the primary metric. Use it as a sanity check, not a decision metric.

### ROC-AUC

ROC (Receiver Operating Characteristic) curve plots True Positive Rate (= Recall) against False Positive Rate at every possible threshold from 0 to 1. The Area Under this Curve is ROC-AUC.

```
True Positive Rate  = TP / (TP + FN)    [same as Recall]
False Positive Rate = FP / (FP + TN)
```

ROC-AUC of 1.0 = perfect. ROC-AUC of 0.5 = random coin flip.

**The problem with ROC-AUC for imbalanced data:**

FPR = FP / (FP + TN). When TN is enormous (284,315 legitimate transactions), even a large absolute number of false positives produces a tiny FPR. ROC-AUC is dominated by how well the model handles the majority class, which is not where the business value lives.

A model that correctly clears 99% of legitimate transactions but catches only 20% of fraud can still achieve ROC-AUC > 0.9 on this dataset.

### PR-AUC (the correct primary metric here)

Precision-Recall curve plots Precision against Recall at every threshold. The area under this curve is PR-AUC.

**Why PR-AUC is better here:**

- It does not involve True Negatives at all. The enormous legitimate class cannot inflate the score.
- The baseline (a random classifier) achieves PR-AUC ≈ fraud_rate = 0.00172 — near zero.
- A perfect classifier achieves PR-AUC = 1.0.
- The entire usable range of the metric is [0.00172, 1.0], which gives fine-grained signal.
- Every improvement in PR-AUC represents a real improvement in catching fraud or reducing false alarms.

In this project, PR-AUC is the headline metric in the results table, the metric used to compare models, and the number on the resume bullet.

---

## 3. Stratified Train/Test Split

### What stratification means

A standard (random) 80/20 split on 284,807 rows will produce approximately 393 fraud cases in training and 99 in testing — but "approximately" is the problem. With only 492 fraud cases, random sampling has real variance.

In an unlucky split you might get:
- 420 fraud cases in training, 72 in testing (over-represents training fraud)
- Or worse: imbalanced sub-splits that cause certain fraud patterns to be absent from one set

Stratified splitting guarantees the fraud rate in both training and test sets matches the overall dataset rate (0.172%). With 492 fraud cases:
- Training (80%): exactly 393 fraud cases
- Test (20%): exactly 99 fraud cases

### Why this matters practically

The test set is the held-out ground truth for all evaluation metrics. If the fraud rate in the test set differs from the true rate, every metric — PR-AUC, precision, recall, business value calculations — will be biased. Stratification eliminates this variance.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y          # this line is the entire mechanism
)
```

### The golden rule: split once, never again

The test set is sacred. In this project:
- `train.py` performs the split and saves `X_test` and `y_test` inside the model bundle
- `evaluate.py` loads those saved test sets — it never reloads or re-splits the data
- The notebooks load the same saved test sets

This ensures that everything evaluated after training uses the identical held-out set.

---

## 4. StandardScaler — Why Amount and Time Need Scaling

### What StandardScaler does

StandardScaler transforms a column to have mean = 0 and standard deviation = 1:

```
z = (x - mean) / std
```

For Amount: mean ≈ $88, std ≈ $250, so a $500 transaction becomes (500 - 88) / 250 ≈ 1.65.
For Time: mean ≈ 94,813 seconds, std ≈ 47,488 seconds.

### Why V1–V28 don't need scaling

The 28 PCA-transformed features are already standardised. PCA produces components with unit variance by convention. They have comparable magnitudes.

### Why Amount and Time do need scaling

- **Amount** ranges from $0 to $25,691. The raw scale is in dollars.
- **Time** ranges from 0 to 172,792 seconds (about 48 hours).

If you feed raw Amount and Time into a model alongside V1–V28 (which range from about -10 to +10), the model's internal distance calculations and gradient steps are dominated by Amount and Time simply because of their larger numeric range — not because they are more informative.

### Which models care about this and which don't

| Model | Needs scaling? | Why |
|---|---|---|
| Logistic Regression | Yes | Gradient descent convergence and regularisation depend on feature scale |
| Isolation Forest | Somewhat | Uses feature ranges to determine split points; extreme ranges bias the splits |
| Random Forest | No | Split-based; only the ranking of values within a feature matters |
| Distance-based models (KNN, SVM) | Yes | Euclidean distance is directly scale-dependent |

In this project, scaling is applied globally for consistency across all three models.

### Why the scalers are saved as .npy files

`build_features.py` needs to reverse the scaling to reconstruct original Amount and Time values for feature engineering:

```python
original_time   = Time_scaled   × time_scale   + time_mean
original_amount = Amount_scaled × amount_scale + amount_mean
```

Saving `[mean, scale]` as a 2-element numpy array is the lightweight mechanism to carry the scaler parameters forward without pickling the entire sklearn object.

---

## 5. Feature Engineering — The 6 New Features

Feature engineering is the process of creating new input variables that make patterns in the data more explicit and learnable. V1–V28 are PCA components — they capture variance efficiently but the components themselves are not interpretable. The 6 engineered features add domain-specific signals.

### `hour_of_day` — Continuous time-of-day signal

```python
original_time = Time_scaled × time_scale + time_mean
hour_of_day = (original_time % 86400) / 3600   # float in [0.0, 24.0)
```

**Why:** The raw `Time` column is seconds elapsed since the first transaction — a number from 0 to 172,792. This encodes when in the 48-hour recording period a transaction occurred, but it does not encode what time of day it occurred at. Fraud clusters at 2–4am. `hour_of_day` makes this temporal pattern directly available to the model.

**The `% 86400`:** Wraps seconds within a 24-hour window (86,400 = 60 × 60 × 24).

### `amount_log` — Log-transformed transaction amount

```python
original_amount = Amount_scaled × amount_scale + amount_mean
amount_log = np.log1p(original_amount)
```

**Why:** Transaction amounts have a heavily right-skewed distribution. Most transactions are under $100; a few are $25,000+. This skew means the raw amount provides weaker signal — the extreme outliers dominate the scale. `log1p` (log(1 + x)) compresses the right tail, bringing all amounts into a [0, ~10] range where differences between $5 and $50 are as visible as differences between $5,000 and $50,000.

**Why `log1p` not `log`:** `log(0)` is undefined; `log1p(0) = 0`. Some transactions have Amount = 0.

### `v_sum` — Aggregate behavioral deviation

```python
v_sum = df[['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10']].abs().sum(axis=1)
```

**Why:** Each V feature captures a different PCA component of transaction behavior. Individually, one anomalous V feature might be noise. When multiple V features are simultaneously anomalous, it is a stronger signal. `v_sum` aggregates the absolute deviations of the first 10 components into a single "how unusual is this transaction overall" signal.

**Why absolute values:** PCA components can be positive or negative; a large negative value is equally anomalous as a large positive one. Summing without `abs()` would have cancellation.

### `v_mean` — Average behavioral deviation

```python
v_mean = df[['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10']].abs().mean(axis=1)
```

**Why:** The mean of absolute V values (rather than sum) normalises for the number of features, making the signal scale-invariant. `v_sum` and `v_mean` are correlated but not identical — they give the model two views of the same underlying "behavioral anomaly magnitude" concept.

### `high_amount_flag` — Binary large-transaction indicator

```python
high_amount_flag = 1 if original_amount > 500 else 0
```

**Why:** Large transactions ($500+) exhibit a fraud pattern distinct from small transactions. Card testers probe with tiny amounts; a different fraud category involves large one-time purchases on stolen cards. The log transform `amount_log` captures the overall distribution but smooths over this categorical boundary. The binary flag makes the threshold explicit and learnable as a decision boundary.

**Why $500:** The amount distribution shows a natural gap around $500 where the fraud pattern shifts. This can be tuned — see the EDA notebook Cell 3.

### `night_flag` — Binary overnight indicator

```python
night_flag = 1 if 0 <= hour_of_day <= 6 else 0
```

**Why:** Fraud rate during midnight–6am is 3–4× the daytime rate on this dataset. `hour_of_day` gives the model continuous time-of-day information, but the overnight window is so predictive that an explicit binary encoding gives the model a direct handle on it. In most SHAP analyses on this dataset, `night_flag` ranks among the top 5 features.

---

## 6. SMOTE — Synthetic Minority Oversampling

### The problem SMOTE solves

After the 80/20 stratified split, training data has:
- ~227,000 legitimate transactions
- ~393 fraud transactions

If you train Random Forest directly on this, it learns that predicting "legitimate" for everything achieves 99.83% accuracy on training data. The gradient or split criterion never sees enough fraud to learn its patterns. The resulting model has near-zero recall.

### How SMOTE works

SMOTE (Synthetic Minority Oversampling TEchnique) creates synthetic fraud examples, not copies:

1. For each real fraud example, find its k nearest fraud neighbors in feature space (k=5 in this project)
2. Pick one neighbor at random
3. Generate a new synthetic example at a random point on the line segment between the original and that neighbor:

```
synthetic = original + random(0,1) × (neighbor - original)
```

4. Repeat until the desired class ratio is reached

The synthetic examples are interpolations of real fraud — they live in the same region of feature space as real fraud but add diversity.

### Why SMOTE at 10%, not 50/50

| Strategy | Training fraud after SMOTE | Synthetic fraud examples |
|---|---|---|
| 50/50 balance | ~227,000 | ~226,607 synthetic from 393 real |
| 10% ratio | ~25,222 | ~24,829 synthetic from 393 real |

At 50/50: the model is trained on 577 synthetic examples for every 1 real fraud example. The synthetic distribution — however well-constructed — is not real fraud data. With that ratio, the model learns the synthetic distribution's geometry rather than the real fraud signal. Overfitting to interpolated patterns generalises poorly to the real test set.

At 10%: synthetic examples support learning without overwhelming the real signal. This balance was chosen empirically — PR-AUC is higher at 10% than at 50/50 on this specific dataset.

### SMOTE is applied only to training data — never to test data

This is a hard rule. Applying SMOTE to test data would:
1. Contaminate the evaluation — you would be measuring performance on artificial data
2. Violate the independence assumption — test data must represent real-world distribution

The test set is always the original, unmodified held-out data.

### What SMOTE cannot fix

SMOTE helps with training imbalance but it does not:
- Add information that was not in the original 393 fraud cases
- Compensate for features that have no predictive power
- Help if the fraud pattern shifts between training period and deployment

---

## 7. Isolation Forest — Unsupervised Anomaly Detection

### The core idea

Isolation Forest detects anomalies by asking: how easy is it to isolate this point from the rest of the data?

Normal points: they cluster together, so you need many random splits to isolate one from the crowd. Each split has many other points nearby.

Anomalous points: they are unusual values, far from the cluster. A single well-placed split often isolates them immediately.

### How it works mechanically

1. Draw a random sample of the data (typically 256 points per tree)
2. Randomly select a feature
3. Randomly select a split value between the feature's min and max
4. Recurse: split the data at this value, creating two branches
5. Repeat until each point is isolated in its own leaf node
6. Record the depth at which each point was isolated

Points isolated at shallow depth (few splits needed) = anomalies.
Points isolated at deep depth (many splits needed) = normal.

Build 100 such trees (this project uses `n_estimators=100`). Average the isolation depth across trees. The final anomaly score:

```
score ∈ (-1, 1)
More negative → more anomalous → more likely fraud
More positive → more normal → less likely fraud
```

The `decision_function()` returns this score. More negative = flag it.

### Why it works without labels

Isolation Forest is completely unsupervised — it sees only the feature values, not the Class label. It detects anything statistically unusual relative to the bulk of the data. Fraud transactions happen to be unusual (by design — they represent abnormal spending behavior), so they tend to isolate at shallow depth.

This makes Isolation Forest valuable as:
- A sanity check: "Do our features actually separate fraud from legitimate?"
- A complementary signal: catching fraud cases that slip past the supervised model
- A zero-day detector: catching novel fraud patterns not seen in training labels

### Contamination parameter

`contamination=0.0017` tells Isolation Forest to set the anomaly threshold such that 0.17% of training data is classified as anomalous — matching the known fraud rate. If you set contamination too low, it misses rare fraud. Too high, and everything looks anomalous.

### In this project

Isolation Forest is trained on X_train only (no labels) and scored on X_test. Its `decision_function` scores are saved in the model bundle. The dashboard compares whether both models flagged a transaction — transactions flagged by both are the highest-conviction fraud cases.

---

## 8. Random Forest — Bagging, Feature Subsampling, Voting

### Decision trees — the building block

A single decision tree asks a sequence of yes/no questions about features:

```
Is V14 < -4.2?
  Yes → Is hour_of_day < 6?
          Yes → FRAUD (high confidence)
          No  → check more features...
  No  → LEGITIMATE (probably)
```

The tree finds the split point at each node that best separates fraud from legitimate (maximising information gain or Gini impurity reduction).

**Problem with a single deep tree:** It memorises the training data. On a new transaction it has never seen, it may perform poorly. A tree of depth 30 on 200,000 training rows has essentially memorised the training set.

### Bagging (Bootstrap Aggregating)

Random Forest builds `n_estimators` independent trees (100 in this project), each trained on a different bootstrap sample of the training data:

```
Bootstrap sample = sample n rows WITH replacement from n training rows
→ Each sample includes ~63% of unique rows, ~37% duplicated
→ Each tree sees a different subset of the training data
→ Each tree makes different errors
```

**Why this helps:** If 100 trees each make uncorrelated errors on different transactions, the majority vote is almost always correct. The individual tree errors cancel out.

### Feature subsampling

At each split in each tree, Random Forest considers only `sqrt(n_features)` randomly chosen features (about 6 out of 34 in this project). This ensures the trees are not all splitting on the same dominant features (V14, night_flag) — each tree develops expertise in a different subset of features.

This is the key difference between Random Forest and Bagging on decision trees. Without feature subsampling, all trees would make similar splits and their errors would be correlated — the averaging would not help.

### Voting for classification

For a test transaction, each of the 100 trees produces a class prediction. The final prediction is the majority vote. `predict_proba` returns the fraction of trees that voted for each class:

```
fraud_probability = (trees voting fraud) / 100
```

This fraction is the fraud probability used as the score for threshold comparison.

### `class_weight='balanced'`

Even after SMOTE, the training set is still not 50/50. `class_weight='balanced'` makes the model's splitting criterion count each fraud example more heavily than each legitimate example, proportional to class frequency:

```
weight[class] = n_samples / (n_classes × count[class])
```

This gives the minority class more influence on split decisions, complementing SMOTE.

### `n_jobs=-1`

Uses all available CPU cores in parallel. Random Forest is embarrassingly parallel — each tree is independent. On a machine with 8 cores, training 100 trees takes roughly the time of training 13 sequential trees.

---

## 9. SHAP Values — Game Theory Meets Model Explanation

### The problem with standard feature importance

Random Forest feature importance (used in the feature importance chart) asks: on average, how much did splits on this feature reduce impurity across all trees? This gives a global ranking but tells you nothing about individual predictions.

It also does not tell you:
- Was the feature pushing the probability up or down?
- For this specific transaction, which feature was most important?
- Do high values of night_flag increase or decrease fraud probability?

### Shapley values — the game theory foundation

SHAP is grounded in Shapley values from cooperative game theory (1953, Lloyd Shapley, Nobel Prize in Economics 2012). The setup:

- Players = features in the model
- Game = predicting fraud probability for one transaction
- Payout = the model's prediction
- Shapley value of feature i = the average marginal contribution of feature i across all possible orderings of features

Concretely: how much does knowing feature i's value change the prediction, compared to the average prediction over the training data?

The Shapley value has a rigorous axiomatic justification — it is the unique allocation that satisfies efficiency (values sum to the prediction minus the base rate), symmetry (identical features get identical values), and linearity (explanations compose additively).

### TreeSHAP — exact and fast for tree models

Computing exact Shapley values requires iterating over all 2^n feature subsets (n=34 features = 17 billion subsets — impractical). TreeSHAP (Lundberg et al., 2018) exploits the tree structure to compute exact Shapley values in polynomial time — typically seconds for thousands of examples.

For this project:
- A single `shap.TreeExplainer(rf_model)` encodes the Random Forest structure
- `explainer.shap_values(X_sample)` returns exact SHAP values for all transactions
- `shap_values[1]` is the SHAP values for the fraud class (class index 1)

### Reading SHAP values

For a specific transaction:
- `shap_value[night_flag] = +0.15` means: night_flag's value (+1) pushed the fraud probability 0.15 higher than the baseline (average) prediction
- `shap_value[V4] = -0.03` means: V4's value pushed fraud probability 0.03 lower
- The sum of all SHAP values = `model_prediction - base_rate`

### The beeswarm plot (shap_summary.png)

Each dot = one transaction. Position on x-axis = SHAP value (positive = pushed toward fraud). Color = feature value (red = high, blue = low). The spread of dots shows how consistent the feature's effect is.

Reading it: "High values of night_flag (red dots) appear at large positive SHAP values — night_flag=1 consistently pushes fraud probability up."

### The bar plot (shap_bar.png)

Shows `mean(|SHAP value|)` per feature across all sampled transactions — the global importance ranking. This is the "overall" answer to "which feature matters most."

### In the dashboard

`explain_transaction()` computes SHAP values for one selected transaction and:
1. Returns the raw SHAP values for the waterfall bar chart
2. Maps the top 3 positive SHAP features to plain English explanations (the `reasons_map` dict)
3. Returns the fraud probability

The waterfall bar chart shows, for the specific transaction under investigation, exactly which features are pushing the probability up (red bars) and which are pulling it down (blue bars) — the most interpretable fraud explanation a model can produce.

---

## 10. The Precision-Recall Tradeoff and Threshold as a Business Decision

### What the threshold is

`predict_proba()` returns a continuous score between 0 and 1 for every transaction. The threshold converts this continuous score into a binary decision:

```
if fraud_prob >= threshold: flag as FRAUD
else: clear as LEGITIMATE
```

The default threshold of 0.5 assumes you treat a false positive and a false negative as equally costly. They are not.

### The asymmetry in this problem

| Error type | What happened | Dollar cost |
|---|---|---|
| False Negative (missed fraud) | Transaction went through, money stolen | $122.21 average loss |
| False Positive (false alarm) | Analyst spends time reviewing a legitimate transaction | $15.00 analyst review cost |

Ratio: missing fraud is 8.1× more costly than a false alarm. This means you should accept false alarms to avoid missing fraud — shift the threshold below 0.5.

### How the threshold moves precision and recall

```
Lower threshold → flag more transactions as fraud
→ More true positives (catch more fraud)       → recall goes UP
→ More false positives (more false alarms)     → precision goes DOWN

Higher threshold → flag fewer transactions as fraud
→ Fewer false positives (fewer false alarms)   → precision goes UP
→ More false negatives (miss more fraud)       → recall goes DOWN
```

### Computing the optimal threshold from business costs

The expected net value at threshold t:

```
net_value(t) = TP(t) × $122.21 − FP(t) × $15.00
```

Maximising this over the threshold sweep (see notebook Cell 2) gives the economically optimal threshold. In this project it lands near 0.25–0.35, which is why 0.30 was chosen.

The threshold sweep chart in `02_modeling.ipynb` Cell 2 plots this directly — it is the single most business-relevant chart in the entire project, because it shows any stakeholder exactly what they are trading when they move the threshold.

### The threshold is not a modeling decision

It cannot be determined from the training data alone, because it depends on:
- The cost of analyst review time (changes with staffing)
- The average fraud loss (changes with fraud patterns)
- The regulatory requirement to report fraud above a certain rate
- The analyst team's capacity — too many flags means some go uninvestigated

The Streamlit dashboard's threshold slider makes this tangible: drag it left, watch the flagged count increase and the value-at-risk metric change in real time.

### The PR curve — visualising all thresholds at once

The Precision-Recall curve plots (precision, recall) at every threshold from 0 to 1. The area under this curve (PR-AUC) summarises model quality across all possible business configurations — it is the metric that does not require committing to a specific threshold.

A model with high PR-AUC gives you more precision at every level of recall. More fraud caught for the same number of false alarms. This is what model improvement means in this problem.
