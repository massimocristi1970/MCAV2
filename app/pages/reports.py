# app/pages/reports.py
"""Report generation and export functions for business finance dashboard."""

import json
from datetime import datetime
from typing import Any, Dict

import pandas as pd
import streamlit as st

from app.services.dashboard_export import DashboardExporter, compute_loans_analysis
from app.ui_theme import (
    apply_ui_theme,
    render_compact_page_title,
    render_empty_state_no_run,
)

st.set_page_config(
    page_title="Reports | MCA Scorecard",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get Help": None, "Report a bug": None, "About": None},
)
apply_ui_theme()
render_compact_page_title(
    "Reports / export",
    "Download HTML, JSON, CSV, or PDF from your latest scored run.",
    eyebrow="MCA Scorecard",
)

run = st.session_state.get("last_run")
if not run:
    render_empty_state_no_run(
        "Reports",
        "Exports unlock after you score a case on Main.",
    )
    st.stop()

company_name = run["company_name"]
params = run["params"]
metrics = run["metrics"]
scores = run["scores"]
analysis_period = run["analysis_period"]
revenue_insights = run.get("revenue_insights", {})
filtered_df = run.get("filtered_df")
manual_balances = run.get("manual_outstanding_debt_balances") or st.session_state.get(
    "manual_outstanding_debt_balances", {}
) or {}
loans_analysis = compute_loans_analysis(filtered_df, analysis_period, manual_balances)

st.subheader("Export")
exporter = DashboardExporter()
exporter.create_export_buttons(
    company_name=company_name,
    params=params,
    metrics=metrics,
    scores=scores,
    analysis_period=analysis_period,
    revenue_insights=revenue_insights,
    loans_analysis=loans_analysis,
    filtered_df=filtered_df,
    manual_debt_balances=manual_balances,
)


def get_score_summary_text(scores: Dict[str, Any]) -> str:
    """Generate a text summary of scores for display."""
    subprime = scores.get("subprime_score", 0)
    mca_rule = scores.get("mca_rule_score", 0)
    ml = scores.get("adjusted_ml_score", scores.get("ml_score", 0)) or 0
    tier = scores.get("subprime_tier", "N/A")
    recommendation = scores.get("subprime_recommendation", "N/A")
    final = scores.get("final_decision", "N/A")
    return (
        f"Final: {final} | Subprime: {subprime:.1f}/100 ({tier}) | "
        f"MCA: {mca_rule:.0f}/100 | ML: {ml:.1f}% | {recommendation}"
    )


def format_metrics_for_display(metrics: Dict[str, Any]) -> Dict[str, str]:
    """Format metric values for human-readable display."""
    formatted: Dict[str, str] = {}
    for key, value in metrics.items():
        if key == "monthly_summary" or isinstance(value, pd.DataFrame):
            continue
        if isinstance(value, float):
            formatted[key] = f"{value:.2f}"
        elif isinstance(value, int):
            formatted[key] = str(value)
        else:
            formatted[key] = str(value)
    return formatted
