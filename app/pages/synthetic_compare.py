# app/pages/synthetic_compare.py
"""
Dual model scoring: Production ML model + Synthetic benchmark.

Upload application banking JSON (or use the last run from Main), then see:
  - **Production model** score (current model.pkl / scaler.pkl)
  - **Synthetic benchmark** PD (transparent rule-based score from the synthetic engine)

The synthetic benchmark is for comparison and sensitivity only — not used for
production decisions. It uses the same 13 features and a fixed rule-based formula.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from app.ui_theme import (
    apply_ui_theme,
    render_compact_page_title,
    render_empty_state_no_run,
    sidebar_section,
    sidebar_subsection,
)

# Repo root for build_training_dataset and synthetic_data
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Feature building from training pipeline (no changes to that script)
from build_training_dataset import (
    _flatten_transactions,
    build_mca_features,
    derive_ml_features,
)

# Synthetic rule-based PD (transparent benchmark)
try:
    from synthetic_data.outcomes import compute_rule_based_pd
    SYNTHETIC_AVAILABLE = True
except ImportError:
    SYNTHETIC_AVAILABLE = False
    compute_rule_based_pd = None

# App settings for production model path
try:
    from app.config.settings import settings
    MODEL_PATH = Path(settings.BASE_DIR) / settings.MODEL_PATH
    SCALER_PATH = Path(settings.BASE_DIR) / settings.SCALER_PATH
except Exception:
    MODEL_PATH = REPO_ROOT / "app" / "models" / "model_artifacts" / "model.pkl"
    SCALER_PATH = REPO_ROOT / "app" / "models" / "model_artifacts" / "scaler.pkl"

FEATURE_NAMES = [
    "Directors Score", "Total Revenue", "Total Debt", "Debt-to-Income Ratio",
    "Operating Margin", "Debt Service Coverage Ratio", "Cash Flow Volatility",
    "Revenue Growth Rate", "Average Month-End Balance",
    "Average Negative Balance Days per Month", "Number of Bounced Payments",
    "Company Age (Months)", "Sector_Risk",
]


def _get_features_from_last_run(run):
    """Build 13-feature dict from Main page last_run (metrics + params)."""
    metrics = run.get("metrics", {})
    params = run.get("params", {})
    industry = params.get("industry", "Other")
    sector_risk = 1 if industry in {"Restaurants and Cafes", "Bars and Pubs", "Construction Firms"} else 0
    return {
        "Directors Score": params.get("directors_score", 50),
        "Total Revenue": metrics.get("Total Revenue", 0),
        "Total Debt": metrics.get("Total Debt", 0),
        "Debt-to-Income Ratio": metrics.get("Debt-to-Income Ratio", 0),
        "Operating Margin": metrics.get("Operating Margin", 0),
        "Debt Service Coverage Ratio": metrics.get("Debt Service Coverage Ratio", 0),
        "Cash Flow Volatility": metrics.get("Cash Flow Volatility", 0.5),
        "Revenue Growth Rate": metrics.get("Revenue Growth Rate", 0),
        "Average Month-End Balance": metrics.get("Average Month-End Balance", 0),
        "Average Negative Balance Days per Month": metrics.get("Average Negative Balance Days per Month", 0),
        "Number of Bounced Payments": metrics.get("Number of Bounced Payments", 0),
        "Company Age (Months)": params.get("company_age_months", 12),
        "Sector_Risk": sector_risk,
    }


def _get_features_from_json(json_data, directors_score, company_age_months, industry):
    """Build 13-feature dict from uploaded JSON using build_training_dataset logic."""
    if isinstance(json_data, list):
        transactions = json_data
    elif isinstance(json_data, dict):
        transactions = json_data.get("transactions", json_data.get("transaction", []))
        if not transactions and "data" in json_data:
            transactions = json_data.get("data", [])
        if not isinstance(transactions, list):
            transactions = []
    else:
        transactions = []
    if not transactions:
        return None
    txns = _flatten_transactions(transactions)
    mca_feats = build_mca_features(txns)
    if not mca_feats:
        return None
    app_data = {
        "directors_score": directors_score,
        "company_age_months": company_age_months,
        "industry": industry,
        "total_debt": 0,
    }
    return derive_ml_features(mca_feats, app_data)


def _score_production(features_dict):
    """Return (score_pct, error_msg). Score is repayment probability as percentage."""
    try:
        import joblib
        if not MODEL_PATH.exists() or not SCALER_PATH.exists():
            return None, f"Model not found: {MODEL_PATH} or {SCALER_PATH}"
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        row = [features_dict.get(k, 0) for k in FEATURE_NAMES]
        X = pd.DataFrame([row], columns=FEATURE_NAMES)
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X = X.fillna(0)
        X_scaled = scaler.transform(X)
        prob = model.predict_proba(X_scaled)[0, 1]
        return round(prob * 100, 2), None
    except Exception as e:
        return None, str(e)


def _score_synthetic_benchmark(features_dict):
    """Return (pd_pct, error_msg). PD = probability of default (synthetic rule-based)."""
    if not SYNTHETIC_AVAILABLE or compute_rule_based_pd is None:
        return None, "Synthetic engine not available"
    try:
        row = pd.Series(features_dict)
        pd_val = compute_rule_based_pd(row)
        return round(pd_val * 100, 2), None
    except Exception as e:
        return None, str(e)


st.set_page_config(
    page_title="Dual model scoring | MCA Scorecard",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get Help": None, "Report a bug": None, "About": None},
)
apply_ui_theme()
render_compact_page_title(
    "Dual model scoring",
    "Production ML vs synthetic benchmark (comparison only; synthetic is not used for production decisions).",
    eyebrow="MCA Scorecard",
)

# Sidebar: data source and params
sidebar_section("Data source")
use_last_run = st.sidebar.checkbox("Use last run from Main page", value=True)
last_run = st.session_state.get("last_run")

if use_last_run and last_run:
    features = _get_features_from_last_run(last_run)
    data_source_label = "Last run from Main"
else:
    features = None
    data_source_label = None
    sidebar_section("Or upload banking JSON")
    uploaded = st.sidebar.file_uploader("Upload transaction JSON", type=["json"], key="synthetic_compare_json")
    if uploaded:
        try:
            raw = uploaded.getvalue().decode("utf-8", errors="replace")
            json_data = json.loads(raw)
        except Exception as e:
            st.sidebar.error(f"Invalid JSON: {e}")
            json_data = None
        if json_data is not None:
            sidebar_subsection("Parameters for uploaded file")
            directors_score = st.sidebar.number_input("Directors Score", 0, 100, 50, key="dc_ds")
            company_age_months = st.sidebar.number_input("Company Age (Months)", 0, 600, 12, key="dc_age")
            industry = st.sidebar.selectbox(
                "Industry",
                ["Other", "Restaurants and Cafes", "Bars and Pubs", "Construction Firms"],
                key="dc_ind",
            )
            features = _get_features_from_json(json_data, directors_score, company_age_months, industry)
            if features is None:
                st.sidebar.warning("Could not derive features from this JSON (no or invalid transactions).")
            else:
                data_source_label = f"Uploaded: {uploaded.name}"

if features is None and not (use_last_run and last_run):
    render_empty_state_no_run(
        "Dual model scoring",
        "Upload banking JSON in the sidebar, or tick “Use last run from Main page” after you score on Main.",
    )
    st.stop()

if features is None:
    render_empty_state_no_run(
        "Dual model scoring",
        "Could not build features from the current inputs. Upload a valid banking JSON or use last run from Main.",
    )
    st.stop()

# Score with both models
prod_score, prod_err = _score_production(features)
syn_pd, syn_err = _score_synthetic_benchmark(features)

# Display
st.subheader("Scores")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Data source", data_source_label or "—")
with col2:
    if prod_err:
        st.metric("Production ML (repayment %)", "—")
        st.caption(f"Error: {prod_err}")
    else:
        st.metric("Production ML (repayment %)", f"{prod_score}%")
        st.caption("Current model.pkl / scaler.pkl")
with col3:
    if syn_err:
        st.metric("Synthetic benchmark (PD %)", "—")
        st.caption(syn_err)
    else:
        st.metric("Synthetic benchmark (PD %)", f"{syn_pd}%")
        st.caption("Rule-based PD (comparison only)")

st.divider()
st.subheader("Feature values used")
st.dataframe(pd.DataFrame([features]).T.rename(columns={0: "Value"}), use_container_width=True)
