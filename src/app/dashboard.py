from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (  # noqa: E402
    ANALYST_REVIEW_COST,
    AVG_FRAUD_LOSS,
    DATA_PROCESSED,
    FRAUD_THRESHOLD,
    METRICS_PATH,
    MODEL_PATH,
)
from src.models.evaluate import explain_transaction  # noqa: E402


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Fraud Detection Engine")


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if not Path(METRICS_PATH).exists():
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)


@st.cache_data
def load_sample_transactions():
    df = pd.read_parquet(DATA_PROCESSED)
    fraud = df[df["Class"] == 1]
    legit = df[df["Class"] == 0].sample(
        n=min(2000, len(df[df["Class"] == 0])), random_state=42
    )
    return (
        pd.concat([fraud, legit])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Initialise session state
# ---------------------------------------------------------------------------
if "selected_txn_idx" not in st.session_state:
    st.session_state.selected_txn_idx = None
if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0


# ---------------------------------------------------------------------------
# Load data / models
# ---------------------------------------------------------------------------
metrics = load_metrics()

if metrics is None:
    st.error("outputs/metrics.json not found. Run `make evaluate` first to generate it.")
    st.stop()

bundle       = load_models()
rf_model     = bundle["random_forest"]
iso_scores   = bundle["iso_scores_test"]
feature_cols = bundle["feature_cols"]

sample_df = load_sample_transactions()
X_sample  = sample_df[feature_cols]

rf_probs = rf_model.predict_proba(X_sample)[:, 1]
sample_df = sample_df.copy()
sample_df["fraud_prob"] = rf_probs.astype(float)
sample_df["txn_id"] = [f"TXN-{i:05d}" for i in sample_df.index]

# Reconstruct original Amount for display (Amount_scaled is in the df)
# We show Amount_scaled as a proxy; load .npy to reconstruct true dollar amount
processed_dir = Path(DATA_PROCESSED).parent
try:
    amount_mean, amount_scale = np.load(processed_dir / "amount_scaler.npy")
    sample_df["amount_orig"] = (
        sample_df["Amount_scaled"].astype(float) * float(amount_scale) + float(amount_mean)
    )
except FileNotFoundError:
    sample_df["amount_orig"] = sample_df["Amount_scaled"]


def _risk_level(p: float, threshold: float) -> str:
    if p >= threshold:
        return "High Risk"
    if p >= threshold * 0.6:
        return "Suspicious"
    return "Clear"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Controls")
    threshold = st.slider(
        "Fraud Threshold",
        min_value=0.10,
        max_value=0.90,
        value=float(FRAUD_THRESHOLD),
        step=0.05,
    )
    show_flagged_only = st.checkbox("Show flagged only", value=False)
    amount_min_val = float(sample_df["amount_orig"].min())
    amount_max_val = float(sample_df["amount_orig"].max())
    amount_range = st.slider(
        "Amount filter ($)",
        min_value=amount_min_val,
        max_value=amount_max_val,
        value=(amount_min_val, amount_max_val),
        step=1.0,
    )

sample_df["risk_level"] = sample_df["fraud_prob"].apply(
    lambda p: _risk_level(p, threshold)
)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_feed, tab_invest = st.tabs(["Transaction Feed", "Investigation Panel"])


# ============================================================
# TAB 1 — Transaction Feed
# ============================================================
with tab_feed:
    st.title("Fraud Detection Engine")
    st.caption(f"Monitoring 284,807 transactions | Threshold: {threshold:.2f}")

    flagged_count = int((sample_df["fraud_prob"] >= threshold).sum())
    value_at_risk = flagged_count * AVG_FRAUD_LOSS
    net_val       = metrics.get("net_value", 0.0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions monitored", "284,807")
    c2.metric("Flagged this session", flagged_count, delta=f"+{flagged_count}", delta_color="inverse")
    c3.metric("Estimated value at risk", f"${value_at_risk:,.0f}")
    c4.metric("Net model value", f"${net_val:,.0f}")

    # Apply filters
    display_df = sample_df[
        (sample_df["amount_orig"] >= amount_range[0])
        & (sample_df["amount_orig"] <= amount_range[1])
    ].copy()
    if show_flagged_only:
        display_df = display_df[display_df["fraud_prob"] >= threshold]

    display_df = display_df.sort_values("fraud_prob", ascending=False).reset_index(drop=True)

    # Format display columns
    def _fmt_hour(h: float) -> str:
        total_min = int(h * 60)
        hh = total_min // 60 % 24
        mm = total_min % 60
        return f"{hh:02d}:{mm:02d}"

    show_cols = display_df[["txn_id", "amount_orig", "hour_of_day", "fraud_prob", "risk_level", "Class"]].copy()
    show_cols.columns = ["Transaction ID", "Amount", "Hour", "Fraud Probability", "Risk Level", "Actual Label"]
    show_cols["Amount"]           = show_cols["Amount"].apply(lambda x: f"${x:.2f}")
    show_cols["Hour"]             = show_cols["Hour"].apply(_fmt_hour)
    show_cols["Fraud Probability"] = show_cols["Fraud Probability"].apply(lambda x: f"{x*100:.1f}%")
    show_cols["Actual Label"]     = show_cols["Actual Label"].apply(lambda x: "Fraud" if x == 1 else "Legitimate")

    def _row_style(row):
        risk = row["Risk Level"]
        color = "#FCEBEB" if risk == "High Risk" else ("#FAEEDA" if risk == "Suspicious" else "")
        bg = f"background-color: {color}; " if color else ""
        return [f"{bg}color: #1a1a1a; font-weight: 500"] * len(row)

    st.dataframe(
        show_cols.style.apply(_row_style, axis=1),
        use_container_width=True,
        height=500,
    )

    st.markdown("---")
    st.subheader("Investigate a Transaction")
    st.caption("Select a row index from the table above and click Investigate.")

    sel_idx = st.number_input(
        "Row index to investigate (0-based from table above):",
        min_value=0,
        max_value=max(0, len(display_df) - 1),
        value=0,
        step=1,
    )
    if st.button("Investigate →"):
        st.session_state.selected_txn_idx = int(display_df.index[sel_idx])
        st.info("Switched — open the Investigation Panel tab to see the analysis.")


# ============================================================
# TAB 2 — Investigation Panel
# ============================================================
with tab_invest:
    st.title("Investigation Panel")

    if st.session_state.selected_txn_idx is None:
        st.info("Select a transaction from the Transaction Feed tab to investigate.")
        st.stop()

    idx = st.session_state.selected_txn_idx
    txn = sample_df.loc[[idx]]

    try:
        explainer = joblib.load(
            Path(MODEL_PATH).parent / "shap_explainer.pkl"
        )
        explanation = explain_transaction(txn, rf_model, explainer, feature_cols)
    except Exception as e:
        st.error(f"Could not load SHAP explainer: {e}. Run `make evaluate` first.")
        st.stop()

    fraud_pct = explanation["fraud_probability"]
    risk_factors = explanation["risk_factors"]
    shap_dict = explanation["shap_values_dict"]

    col1, col2 = st.columns([1.3, 1])

    with col1:
        prob_color = "inverse" if fraud_pct / 100 >= threshold else "normal"
        st.metric(
            label="Fraud Probability",
            value=f"{fraud_pct:.1f}%",
            delta="ABOVE threshold" if fraud_pct / 100 >= threshold else "below threshold",
            delta_color="inverse" if fraud_pct / 100 >= threshold else "normal",
        )

        detail_cols = st.columns(2)
        detail_cols[0].metric("Amount", f"${float(txn['amount_orig'].iloc[0]):.2f}")
        detail_cols[1].metric(
            "Hour",
            f"{int(float(txn['hour_of_day'].iloc[0])):02d}:{int((float(txn['hour_of_day'].iloc[0]) % 1)*60):02d}"
        )
        detail_cols2 = st.columns(2)
        detail_cols2[0].metric("Transaction ID", txn["txn_id"].iloc[0])
        detail_cols2[1].metric(
            "Actual Label",
            "Fraud" if int(txn["Class"].iloc[0]) == 1 else "Legitimate"
        )

        # SHAP waterfall bar chart
        shap_series = pd.Series(shap_dict).sort_values(key=abs, ascending=False).head(10)
        colors = ["#E24B4A" if v > 0 else "#378ADD" for v in shap_series.values]

        fig = go.Figure(go.Bar(
            x=shap_series.index.tolist(),
            y=shap_series.values.tolist(),
            marker_color=colors,
        ))
        fig.add_hline(y=0, line_width=1, line_color="black")
        fig.update_layout(
            title="Feature contributions to fraud probability",
            xaxis_title="Feature",
            yaxis_title="SHAP value",
            height=380,
            margin=dict(l=20, r=20, t=50, b=80),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Risk Factors")
        for reason in risk_factors:
            st.error(reason)

        st.subheader("Model Metrics")
        m_cols = st.columns(2)
        m_cols[0].metric("PR-AUC",    f"{metrics.get('pr_auc', 0):.3f}")
        m_cols[1].metric("Threshold", f"{metrics.get('threshold', threshold):.2f}")
        m_cols2 = st.columns(2)
        m_cols2[0].metric("Precision", f"{metrics.get('precision', 0):.3f}")
        m_cols2[1].metric("Recall",    f"{metrics.get('recall', 0):.3f}")

        st.subheader("Model Agreement")
        # Find the index of this transaction in the test set to look up iso_scores
        y_test   = bundle["y_test"]
        X_test   = bundle["X_test"]
        # iso_scores_test aligns with X_test — try to find a matching row
        # Fallback: use the sample_df index if test set alignment is unclear
        try:
            test_match = (X_test == txn[feature_cols].iloc[0]).all(axis=1)
            match_positions = np.where(test_match.to_numpy())[0]
            if len(match_positions) > 0:
                iso_score_val = float(iso_scores[match_positions[0]])
            else:
                iso_score_val = None
        except Exception:
            iso_score_val = None

        if iso_score_val is not None:
            if iso_score_val < -0.1:
                st.markdown(
                    '<p style="color:red;font-weight:bold">Both models flagged this transaction</p>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<p style="color:#b8860b;font-weight:bold">Only RF flagged (IF did not flag)</p>',
                    unsafe_allow_html=True,
                )
            st.caption(f"Isolation Forest score: {iso_score_val:.4f} (more negative = more anomalous)")
        else:
            st.caption("Isolation Forest score not available for this transaction (not in test set).")
