"""Tests for scoring calibration fixes."""

from __future__ import annotations

import pandas as pd
import pytest

from app.config.scoring_thresholds import get_thresholds
from app.services.business_risk_signals import calculate_business_metrics
from app.services.ensemble_scorer import get_ensemble_recommendation
from app.services.scoring_alignment import align_scoring_metrics
from app.services.subprime_scoring_system import SubprimeScoring
from mca_scorecard_rules import decide_application


@pytest.mark.unit
def test_dscr_neutral_when_no_repayments_observed():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=30, freq="D"),
            "amount": [-100.0] * 30,
            "name": ["Customer"] * 30,
            "subcategory": ["Income"] * 30,
        }
    )
    metrics = calculate_business_metrics(df, company_age_months=12)
    assert metrics["DSCR Repayments Observed"] is False
    assert metrics["Debt Service Coverage Ratio"] == pytest.approx(1.15)


@pytest.mark.unit
def test_align_scoring_metrics_unifies_volatility_with_mca_cv():
    metrics = {"Cash Flow Volatility": 0.35}
    params = {"mca_rule_signals": {"inflow_cv": 0.82, "inflow_days_30d": 12}}
    align_scoring_metrics(metrics, params)
    assert metrics["Cash Flow Volatility"] == 0.82
    assert metrics.get("Cash Flow Volatility Unified") is True


@pytest.mark.unit
def test_subprime_uses_centralized_directors_threshold():
    thresholds = get_thresholds()
    scorer = SubprimeScoring()
    metrics = {
        "Debt Service Coverage Ratio": 1.9,
        "Revenue Growth Rate": 0.12,
        "Average Month-End Balance": 3000,
        "Cash Flow Volatility": 0.25,
        "Operating Margin": 0.08,
        "Net Income": 5000,
        "Average Negative Balance Days per Month": 1,
        "Number of Bounced Payments": 0,
    }
    params = {"directors_score": thresholds.DIRECTORS.full_points, "company_age_months": 24, "industry": "Retail"}
    points, pct, status = scorer._score_metric_points("Directors Score", params["directors_score"], metrics)
    assert status == "PASS"
    assert points == thresholds.DIRECTORS.weight


@pytest.mark.unit
def test_subprime_scores_bounced_payments():
    scorer = SubprimeScoring()
    metrics = {
        "Debt Service Coverage Ratio": 1.5,
        "Revenue Growth Rate": 0.05,
        "Average Month-End Balance": 2000,
        "Cash Flow Volatility": 0.4,
        "Operating Margin": 0.05,
        "Net Income": 1000,
        "Average Negative Balance Days per Month": 2,
        "Number of Bounced Payments": 0,
    }
    params = {"directors_score": 70, "company_age_months": 18, "industry": "Retail"}
    clean = scorer.calculate_subprime_score(metrics, params)["subprime_score"]
    metrics["Number of Bounced Payments"] = 5
    dirty = scorer.calculate_subprime_score(metrics, params)["subprime_score"]
    assert dirty < clean


@pytest.mark.unit
def test_mca_approve_requires_min_score():
    features = {
        "inflow_days_30d": 20,
        "max_inflow_gap_days": 4,
        "inflow_cv": 0.5,
        "months_covered": 6,
        "txn_count_avg_month": 50,
    }
    decision, score, _ = decide_application(features)
    assert decision == "APPROVE"
    assert score >= 75


@pytest.mark.unit
def test_ensemble_returns_decision_alignment():
    result = get_ensemble_recommendation(
        scores={"mca_score": 85, "mca_decision": "APPROVE", "subprime_score": 72},
        metrics={"Debt Service Coverage Ratio": 1.8, "Cash Flow Volatility": 0.3},
        params={"directors_score": 72},
    )
    assert result["decision_alignment"] in ("Aligned", "Mixed", "Complementary", "Opposed")
    assert result["numeric_score_gap"] == 13.0


@pytest.mark.unit
def test_subprime_unavailable_does_not_force_zero_decline():
    result = get_ensemble_recommendation(
        scores={"mca_score": 80, "mca_decision": "APPROVE", "subprime_score": None},
        metrics={"Debt Service Coverage Ratio": 1.8, "Cash Flow Volatility": 0.3},
        params={"directors_score": 72},
    )
    assert result["combined_score"] == 80.0
    assert result["decision"] == "APPROVE"
