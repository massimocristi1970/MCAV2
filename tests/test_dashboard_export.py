"""Tests for unified dashboard export payload and reports."""

from __future__ import annotations

import pandas as pd
import pytest

from app.services.dashboard_export import (
    build_export_payload,
    generate_html_report,
    generate_pdf_report,
)


@pytest.fixture
def sample_run_context() -> dict:
    return {
        "company_name": "Acme Trading Ltd",
        "params": {
            "company_name": "Acme Trading Ltd",
            "industry": "Retail",
            "requested_loan": 10000.0,
            "directors_score": 75,
            "company_age_months": 24,
            "business_ccj": True,
            "poor_or_no_online_presence": False,
            "uses_generic_email": False,
            "mca_rule_score": 72,
            "mca_rule_decision": "APPROVE",
            "mca_rule_reasons": ["Strong DSCR", "Stable revenue"],
            "tu_director_score": 680,
            "tu_director_decision": "APPROVE",
        },
        "metrics": {
            "Total Revenue": 250000.0,
            "Monthly Average Revenue": 20000.0,
            "Net Income": 15000.0,
            "Operating Margin": 0.12,
            "Revenue Growth Rate": 0.08,
            "Debt Service Coverage Ratio": 2.5,
            "Cash Flow Volatility": 0.15,
            "Average Month-End Balance": 5000.0,
            "OB History Months": 12,
            "OB Transaction Count": 500,
            "Open Banking Insights Used In Score": "No - analysis/export only",
            "Card Processing Insight Layer": "Not available",
        },
        "scores": {
            "subprime_score": 68.5,
            "subprime_tier": "B",
            "subprime_recommendation": "Approve with monitoring",
            "mca_rule_score": 72,
            "mca_rule_decision": "APPROVE",
            "mca_rule_reasons": ["Strong DSCR", "Stable revenue"],
            "adjusted_ml_score": 71.2,
            "loan_risk": "Moderate",
            "final_decision": "APPROVE",
            "final_decision_reasons": ["Ensemble APPROVE", "TU overlay: APPROVE"],
            "ensemble": {
                "decision": "APPROVE",
                "combined_score": 70.4,
                "confidence": 82,
                "score_convergence": "Good agreement",
                "primary_reason": "MCA and Subprime align on approve",
                "contributing_scores": {"mca_score": 72, "subprime_score": 68.5},
                "risk_factors": ["Elevated CCJ flag"],
                "positive_factors": ["Strong DSCR"],
                "recommendations": ["Standard monitoring"],
                "pricing_guidance": {
                    "factor_rate": "1.28",
                    "max_term": "12 months",
                    "max_multiple": "1.2x",
                    "collection_frequency": "Daily",
                },
            },
        },
        "analysis_period": "12",
        "revenue_insights": {
            "unique_revenue_sources": 8,
            "avg_daily_revenue_amount": 850.0,
            "total_revenue_days": 220,
            "avg_revenue_transactions_per_day": 3.2,
        },
        "filtered_df": pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-15", "2024-02-10", "2024-03-05"]),
                "amount": [5000.0, -1200.0, 8000.0],
                "name": ["Lender A", "Lender A Repay", "Revenue Co"],
                "subcategory": ["Loans", "Debt Repayments", "Income"],
            }
        ),
        "manual_outstanding_debt_balances": {"Lender B": 2500.0},
    }


def test_export_payload_includes_dashboard_decision_fields(sample_run_context):
    payload = build_export_payload(
        company_name=sample_run_context["company_name"],
        params=sample_run_context["params"],
        metrics=sample_run_context["metrics"],
        scores=sample_run_context["scores"],
        analysis_period=sample_run_context["analysis_period"],
        revenue_insights=sample_run_context["revenue_insights"],
        filtered_df=sample_run_context["filtered_df"],
        manual_debt_balances=sample_run_context["manual_outstanding_debt_balances"],
    )

    sr = payload["scoring_results"]
    assert sr["final_decision"] == "APPROVE"
    assert sr["mca_rule_decision"] == "APPROVE"
    assert sr["mca_rule_reasons"] == ["Strong DSCR", "Stable revenue"]
    assert sr["ensemble"]["combined_score"] == 70.4

    rf = payload["business_parameters"]["risk_factors"]
    assert rf["business_ccj"] is True

    assert payload["open_banking_insights"]
    assert payload["card_processing_insights"]
    assert payload["score_impact"]
    assert payload["evidence_quality"]
    assert payload["metrics_thresholds"]
    assert payload["underwriting"]
    assert "advance_holdback" in payload["underwriting"]


def test_export_html_contains_final_decision(sample_run_context):
    payload = build_export_payload(
        company_name=sample_run_context["company_name"],
        params=sample_run_context["params"],
        metrics=sample_run_context["metrics"],
        scores=sample_run_context["scores"],
        analysis_period=sample_run_context["analysis_period"],
        revenue_insights=sample_run_context["revenue_insights"],
        filtered_df=sample_run_context["filtered_df"],
        manual_debt_balances=sample_run_context["manual_outstanding_debt_balances"],
    )
    html = generate_html_report(payload)
    assert "APPROVE" in html
    assert "Acme Trading Ltd" in html
    assert "Open banking derived insights" in html
    assert "Underwriting workspace" in html
    assert "Business CCJs" in html


def test_export_pdf_generates_bytes(sample_run_context):
    pytest.importorskip("reportlab")
    payload = build_export_payload(
        company_name=sample_run_context["company_name"],
        params=sample_run_context["params"],
        metrics=sample_run_context["metrics"],
        scores=sample_run_context["scores"],
        analysis_period=sample_run_context["analysis_period"],
        revenue_insights=sample_run_context["revenue_insights"],
        filtered_df=sample_run_context["filtered_df"],
        manual_debt_balances=sample_run_context["manual_outstanding_debt_balances"],
    )
    pdf = generate_pdf_report(payload)
    assert pdf[:4] == b"%PDF"
    assert len(pdf) > 1000
