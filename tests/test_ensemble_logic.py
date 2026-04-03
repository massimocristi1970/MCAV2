import pytest

from app.main import adjust_ml_score_for_growth_business
from app.services.ensemble_scorer import get_ensemble_recommendation


def _strong_metrics():
    return {
        "Debt Service Coverage Ratio": 1.8,
        "Cash Flow Volatility": 0.28,
        "Average Negative Balance Days per Month": 1,
        "Operating Margin": 0.08,
        "Average Month-End Balance": 3000,
        "Revenue Growth Rate": 0.12,
        "Number of Bounced Payments": 0,
    }


def _strong_params(**overrides):
    params = {
        "directors_score": 76,
        "company_age_months": 24,
        "business_ccj": False,
        "business_ccj_count": 0,
    }
    params.update(overrides)
    return params


@pytest.mark.unit
def test_single_business_ccj_is_not_an_automatic_hard_decline():
    result = get_ensemble_recommendation(
        scores={
            "mca_score": 85,
            "mca_decision": "APPROVE",
            "subprime_score": 80,
            "ml_score": 78,
        },
        metrics=_strong_metrics(),
        params=_strong_params(business_ccj=True, business_ccj_count=1),
    )

    assert result["decision"] == "APPROVE"
    assert result["combined_score"] > 75


@pytest.mark.unit
def test_multiple_ccjs_still_trigger_hard_decline():
    result = get_ensemble_recommendation(
        scores={
            "mca_score": 85,
            "mca_decision": "APPROVE",
            "subprime_score": 80,
            "ml_score": 78,
        },
        metrics=_strong_metrics(),
        params=_strong_params(business_ccj=True, business_ccj_count=2),
    )

    assert result["decision"] == "DECLINE"
    assert result["combined_score"] == 0


@pytest.mark.unit
def test_senior_review_band_is_reachable():
    result = get_ensemble_recommendation(
        scores={
            "mca_score": 62,
            "subprime_score": 62,
            "ml_score": 62,
        },
        metrics=_strong_metrics(),
        params=_strong_params(),
    )

    assert result["decision"] == "SENIOR_REVIEW"
    assert result["combined_score"] == 62


@pytest.mark.unit
def test_growth_business_ml_adjustment_caps_at_85_not_50():
    adjusted = adjust_ml_score_for_growth_business(
        raw_ml_score=82,
        metrics={
            "Debt Service Coverage Ratio": 3.2,
            "Operating Margin": -0.05,
            "Revenue Growth Rate": 0.20,
            "Total Revenue": 144000,
            "Monthly Average Revenue": 12000,
        },
        params={
            "directors_score": 82,
            "company_age_months": 24,
        },
    )

    assert adjusted == 85

@pytest.mark.unit
def test_zero_scores_are_treated_as_real_inputs_not_missing_data():
    result = get_ensemble_recommendation(
        scores={
            "mca_score": 0,
            "mca_decision": "REFER",
            "subprime_score": 80,
            "ml_score": 80,
        },
        metrics=_strong_metrics(),
        params=_strong_params(),
    )

    assert result["combined_score"] == 40
    assert result["contributing_scores"]["mca_score"] == 0


@pytest.mark.unit
def test_missing_scores_still_renormalize_weights():
    result = get_ensemble_recommendation(
        scores={
            "mca_score": 80,
            "mca_decision": "APPROVE",
            "subprime_score": 60,
        },
        metrics=_strong_metrics(),
        params=_strong_params(),
    )

    assert result["detailed_breakdown"]["raw_combined_score"] == 70
    assert result["combined_score"] == 68
    assert "ml_score" not in result["contributing_scores"]



