"""Tests for underwriting workspace helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from app.services.underwriting_insights import (
    apply_data_quality_gate,
    assess_data_quality,
    build_advance_holdback_guidance,
    build_decision_caps_detail,
    build_underwriting_alerts,
    build_underwriting_package,
)


def _sample_metrics() -> dict:
    return {
        "Total Revenue": 120000.0,
        "Monthly Average Revenue": 10000.0,
        "OB Weakest Month Revenue": 8000.0,
        "OB History Months": 6,
        "OB Transaction Count": 120,
        "total_revenue_days": 110,
        "OB Top Revenue Source Percentage": 75.0,
        "OB Non-Revenue Inflow Ratio": 0.1,
        "Debt Service Coverage Ratio": 1.5,
        "OB Debt Repayment Burden": 0.1,
    }


def _sample_params() -> dict:
    return {
        "requested_loan": 15000.0,
        "tu_parse_status": "parsed",
        "tu_director_score": 680,
        "tu_director_decision": "APPROVE",
        "business_ccj": False,
    }


def _sample_scores() -> dict:
    return {
        "subprime_score": 68.0,
        "subprime_tier": "Tier 2",
        "mca_rule_score": 72,
        "mca_rule_decision": "APPROVE",
        "ensemble": {"decision": "APPROVE"},
    }


def _sample_df() -> pd.DataFrame:
    dates = pd.date_range(end=pd.Timestamp.now().normalize(), periods=40, freq="7D")
    return pd.DataFrame({"date": dates, "amount": [100.0] * 40})


def test_assess_data_quality_passes_with_sufficient_data():
    result = assess_data_quality(_sample_df(), _sample_metrics(), _sample_params(), "6")
    assert result["overall"] == "Pass"
    assert result["gate_action"] == "none"
    assert result["fail_count"] == 0


def test_assess_data_quality_fails_without_tu_xml():
    params = dict(_sample_params())
    params.pop("tu_director_score", None)
    params["tu_parse_status"] = "missing"
    result = assess_data_quality(_sample_df(), _sample_metrics(), params, "6")
    assert result["overall"] == "Fail"
    assert result["gate_action"] == "refer"


def test_apply_data_quality_gate_caps_approve_to_refer():
    scores = {"final_decision": "APPROVE", "final_decision_reasons": ["Engine approve"]}
    params = {"final_decision": "APPROVE"}
    dq = {"gate_action": "refer", "summary": "Insufficient data quality"}
    apply_data_quality_gate(scores, params, dq)
    assert scores["final_decision"] == "REFER"
    assert any("Data quality gate" in r for r in scores["final_decision_reasons"])


def test_advance_holdback_guidance_flags_above_capacity():
    params = dict(_sample_params())
    params["requested_loan"] = 35000.0
    guidance = build_advance_holdback_guidance(_sample_metrics(), params, _sample_scores())
    assert guidance["recommended_max_advance"] > 0
    assert guidance["request_status"] == "above_capacity"
    assert guidance["illustrative_holdback_pct_of_revenue"] is not None


def test_underwriting_alerts_include_revenue_concentration():
    alerts = build_underwriting_alerts(_sample_metrics(), _sample_params(), _sample_scores(), _sample_df())
    assert any(a["alert"] == "Revenue concentration" for a in alerts)


def test_decision_caps_detail_includes_subprime_gate():
    scores = dict(_sample_scores())
    scores["subprime_score"] = 58.0
    detail = build_decision_caps_detail(_sample_metrics(), _sample_params(), scores)
    assert detail["has_active_caps"] is True
    assert any(c["type"] == "Subprime gate" for c in detail["decision_caps"])


def test_build_underwriting_package_structure():
    package = build_underwriting_package(
        metrics=_sample_metrics(),
        params=_sample_params(),
        scores=_sample_scores(),
        filtered_df=_sample_df(),
        analysis_period="6",
        manual_debt_balances={"Hidden Lender": 5000.0},
    )
    assert "data_quality" in package
    assert "advance_holdback" in package
    assert "underwriting_alerts" in package
    assert "decision_caps" in package
    assert "lender_stacking" in package
    assert any(r.get("source") == "Underwriter entered balance" for r in package["lender_stacking"])
