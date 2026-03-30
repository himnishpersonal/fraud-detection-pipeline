# Dashboard & Deployment — dashboard.py

## Overview

`src/app/dashboard.py` is a two-tab Streamlit application that turns the trained model into an operational fraud investigation tool. It is the public face of the project — what an analyst would open at the start of a shift.

**Two tabs, two distinct purposes:**
- **Tab 1 — Transaction Feed:** operational queue. Shows all sampled transactions sorted by risk, filtered by threshold/amount, color-coded by risk level.
- **Tab 2 — Investigation Panel:** forensic drill-down. For a selected transaction, shows fraud probability, SHAP waterfall chart, plain-English risk factors, and Isolation Forest agreement.

---

## Startup and sys.path

```python
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
```

`dashboard.py` is at `src/app/dashboard.py` — three levels down from the project root. The triple `.parent` chain resolves to the project root where `config.py` lives.

```python
from config import (
    ANALYST_REVIEW_COST, AVG_FRAUD_LOSS, DATA_PROCESSED,
    FRAUD_THRESHOLD, METRICS_PATH, MODEL_PATH,
)
from src.models.evaluate import explain_transaction
```

Importing `explain_transaction` from `evaluate.py` is the key pattern that keeps SHAP logic in one place. The dashboard does not re-implement any explanation logic — it just calls the function.

---

## Caching Strategy

Streamlit reruns the entire script top-to-bottom every time the user interacts with any widget. Without caching, every slider drag would reload the 144MB parquet and retrain predictions. Caching prevents this.

### `@st.cache_resource` — for objects that should not be duplicated

```python
@st.cache_resource
def load_models():
    return joblib.load(MODEL_PATH)
```

`cache_resource` is for heavyweight objects (ML models, database connections) that should be shared across all user sessions and all reruns. The model bundle (~50MB in memory) is loaded once and reused. If two browser tabs have the app open, they share the same model object.

**vs `cache_data`:** `cache_data` serialises and copies the return value — fine for DataFrames, wrong for large model objects because it would copy the entire RF model on every cache hit.

### `@st.cache_data` — for DataFrames and serialisable objects

```python
@st.cache_data
def load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)

@st.cache_data
def load_sample_transactions():
    df = pd.read_parquet(DATA_PROCESSED)
    fraud = df[df["Class"] == 1]
    legit = df[df["Class"] == 0].sample(n=min(2000, len(df[df["Class"] == 0])), random_state=42)
    return pd.concat([fraud, legit]).sample(frac=1, random_state=42).reset_index(drop=True)
```

`cache_data` is for functions that return DataFrames, dicts, or other serialisable objects. It stores a serialised copy and returns a new copy on each cache hit — preventing mutation bugs where one rerun modifies data that another rerun depends on.

**Sample composition:** All 492 fraud cases + 2,000 randomly sampled legitimate transactions = 2,492 rows maximum. This provides enough variety to demonstrate the threshold slider while keeping the table responsive.

---

## Prediction Generation

```python
rf_probs = rf_model.predict_proba(X_sample)[:, 1]
```

`predict_proba()` returns an `(n_samples, 2)` array. Column 0 = probability of Class 0 (legitimate). Column 1 = probability of Class 1 (fraud). `[:, 1]` extracts the fraud probabilities — floats between 0 and 1.

**Why not `predict()`:** `predict()` returns binary labels (0 or 1) using the default 0.5 threshold. The entire dashboard is built around the adjustable threshold slider — using `predict()` would make every row either 0% or 100% probability, breaking all color-coding and filtering.

```python
def _risk_level(p: float, threshold: float) -> str:
    if p >= threshold:           return "High Risk"
    if p >= threshold * 0.6:     return "Suspicious"
    return "Clear"
```

Risk level bands are relative to the current threshold:
- `High Risk`: at or above the action threshold
- `Suspicious`: between 60% and 100% of the threshold (approaching the flag zone)
- `Clear`: below 60% of the threshold

This means when the user drags the threshold slider, all three categories update simultaneously — the risk level is always expressed relative to the current operating point.

---

## Sidebar Controls

```python
threshold = st.slider("Fraud Threshold", min_value=0.10, max_value=0.90,
                       value=float(FRAUD_THRESHOLD), step=0.05)
show_flagged_only = st.checkbox("Show flagged only", value=False)
amount_range = st.slider("Amount filter ($)", ...)
```

**Threshold slider:** The most important control. Dragging left catches more fraud (lower threshold = more flagged) but increases false alarms. Dragging right reduces false alarms but misses more fraud. The metric cards and table update instantly. This makes the precision-recall tradeoff tangible and interactive — you can *see* the flagged count change in real time.

**Why sidebar:** Streamlit convention places persistent controls in the sidebar. It remains visible regardless of which tab is active.

---

## Tab 1 — Transaction Feed

### KPI Metric Cards

```python
c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions monitored", "284,807")
c2.metric("Flagged this session", flagged_count, delta=f"+{flagged_count}", delta_color="inverse")
c3.metric("Estimated value at risk", f"${value_at_risk:,.0f}")
c4.metric("Net model value", f"${net_val:,.0f}")
```

- **Transactions monitored (284,807):** hardcoded — represents the full dataset, not the 2,492-row sample
- **Flagged this session:** updates with threshold slider. `delta_color="inverse"` makes the delta red (more flagged = higher risk shown in red)
- **Estimated value at risk:** `flagged_count × $122.21`. Rough upper bound on exposure in the flagged queue
- **Net model value:** loaded from `metrics.json` — the $9,997.85 calculated at evaluation time

### Row Styling

```python
def _row_style(row):
    risk = row["Risk Level"]
    color = "#FCEBEB" if risk == "High Risk" else ("#FAEEDA" if risk == "Suspicious" else "")
    bg = f"background-color: {color}; " if color else ""
    return [f"{bg}color: #1a1a1a; font-weight: 500"] * len(row)

st.dataframe(show_cols.style.apply(_row_style, axis=1), ...)
```

- `#FCEBEB` — light red for High Risk rows
- `#FAEEDA` — light amber for Suspicious rows
- `color: #1a1a1a` — dark charcoal text on all rows (prevents light-on-light unreadability)
- `font-weight: 500` — medium weight for legibility

The styling applies to the pandas Styler before passing to `st.dataframe`, which renders it as an HTML table with inline CSS.

### Investigate button

```python
if st.button("Investigate →"):
    st.session_state.selected_txn_idx = int(display_df.index[sel_idx])
    st.info("Switched — open the Investigation Panel tab to see the analysis.")
```

`st.session_state` persists values across reruns within a session. Setting `selected_txn_idx` here allows Tab 2 to read it on the next render. This is the Streamlit pattern for cross-tab communication — there is no direct tab switching API, so the user must click the tab manually after clicking Investigate.

---

## Tab 2 — Investigation Panel

### Session state check

```python
if st.session_state.selected_txn_idx is None:
    st.info("Select a transaction from the Transaction Feed tab to investigate.")
    st.stop()
```

`st.stop()` halts script execution at this point — nothing below renders until a transaction is selected.

### SHAP explainer loading

```python
explainer = joblib.load(Path(MODEL_PATH).parent / "shap_explainer.pkl")
explanation = explain_transaction(txn, rf_model, explainer, feature_cols)
```

The explainer is loaded fresh each time Tab 2 renders for a new transaction. This is slightly inefficient — ideally it would be cached with `@st.cache_resource`. An improvement for future versions.

### SHAP Waterfall Chart

```python
shap_series = pd.Series(shap_dict).sort_values(key=abs, ascending=False).head(10)
colors = ["#E24B4A" if v > 0 else "#378ADD" for v in shap_series.values]

fig = go.Figure(go.Bar(
    x=shap_series.index.tolist(),
    y=shap_series.values.tolist(),
    marker_color=colors,
))
fig.add_hline(y=0, line_width=1, line_color="black")
```

**Color encoding:**
- Red (`#E24B4A`): positive SHAP — this feature is pushing fraud probability **up**
- Blue (`#378ADD`): negative SHAP — this feature is pushing fraud probability **down**

**The horizontal line at y=0** is the visual anchor — bars above push toward fraud, bars below push toward legitimate. The sum of all SHAP values equals `fraud_probability - base_rate`.

**Top 10 by absolute value:** Sorted by `key=abs` so the most impactful features (regardless of direction) appear first.

### Isolation Forest Agreement

```python
test_match = (X_test == txn[feature_cols].iloc[0]).all(axis=1)
match_positions = np.where(test_match.to_numpy())[0]
if len(match_positions) > 0:
    iso_score_val = float(iso_scores[match_positions[0]])
```

Looks up the Isolation Forest score for this specific transaction by matching its feature vector against X_test row by row. If found (transaction was in the test set), shows whether the IF score is below -0.1 (anomalous — both models flagged) or above -0.1 (only RF flagged). If not found (transaction was from the full parquet sample, not the test set), shows "not available."

**Threshold -0.1:** Isolation Forest scores near 0 are ambiguous. -0.1 is a conservative anomaly threshold — only clearly anomalous transactions qualify as "both models flagged." This avoids false confidence from borderline IF scores.

---

## Pre-flight Check

```python
if metrics is None:
    st.error("outputs/metrics.json not found. Run `make evaluate` first to generate it.")
    st.stop()
```

The dashboard checks for `metrics.json` before doing anything else. If the user runs `streamlit run` before running `make evaluate`, they get a clear error message instead of a cryptic KeyError deep in the rendering code.

---

## Deployment Options

### Option 1 — Streamlit Community Cloud (recommended for portfolio)

**What it is:** Streamlit's free hosting platform at share.streamlit.io

**Steps:**
1. Push repo to GitHub (already done)
2. Go to share.streamlit.io → connect GitHub → select repo → set file path to `src/app/dashboard.py`
3. Set Python version to 3.11 (more stable than 3.14 for deployment)
4. Deploy

**The model file problem:** `fraud_model.pkl` is ~50MB+ and excluded from Git by `.gitignore`. Options:
- **Git LFS:** track `.pkl` files with Git Large File Storage — free up to 1GB on GitHub
- **Cloud storage:** store models in S3/GCS, download in `load_models()` on first run
- **Rebuild on deploy:** add a `startup.sh` that runs `make all` on deployment (slow but self-contained)

**Free tier limitations:** App sleeps after 7 days of inactivity (wakes on visit, ~30 second delay). Fine for a portfolio project.

### Option 2 — Railway or Render (always-on)

**What it is:** Container hosting platforms with free tiers that never sleep.

Add a `Procfile` at the project root:
```
web: streamlit run src/app/dashboard.py --server.port=$PORT --server.address=0.0.0.0
```

Push to GitHub → connect Railway/Render → auto-deploys on every push. The `$PORT` variable is injected by the platform.

**Model files:** Same problem as above — needs external storage or Git LFS.

### Option 3 — Docker (local or cloud)

Add a `Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "src/app/dashboard.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t fraud-detection .
docker run -p 8501:8501 fraud-detection
```

Deploy to any container platform: AWS ECS, Google Cloud Run, Fly.io, etc.

**Advantage:** Fully reproducible environment. The container includes Python version, all dependencies, and the app. No "works on my machine" issues.

### Model file strategy for deployment

The cleanest approach for a portfolio project:

1. Run `make all` locally (already done — models are in `models/`)
2. Upload `fraud_model.pkl` and `shap_explainer.pkl` to an S3 bucket (free tier: 5GB)
3. In `dashboard.py`, modify `load_models()`:

```python
@st.cache_resource
def load_models():
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        import urllib.request
        model_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(
            "https://your-bucket.s3.amazonaws.com/fraud_model.pkl",
            model_path
        )
    return joblib.load(model_path)
```

This downloads the model on first run and caches it locally. Subsequent runs use the cached file.

---

## Performance Notes

**Dashboard load time breakdown:**
- `load_models()`: ~2-3s (one-time, cached)
- `load_sample_transactions()`: ~0.3s (parquet read, cached)
- `predict_proba(X_sample)`: ~0.1s (2,492 rows, 100 trees)
- Tab 2 SHAP computation: ~1-2s per transaction (single-row inference)

**Memory footprint:**
- Model bundle in memory: ~200-400MB (100 RF trees × 36 features)
- Sample DataFrame: ~2MB
- SHAP explainer: ~100MB

Total resident memory: ~500MB. Within the free tier limits of all major hosting platforms (typically 1GB).

**Bottleneck for scaling:** If the sample size were increased from 2,492 to the full 284,807 rows, `predict_proba` would take ~5 seconds and the table render would be slow. The 2,492-row stratified sample (all fraud + 2,000 legitimate) is the right balance for interactivity.
