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

    assert result["decision"] == "REFER"
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

    assert result["combined_score"] == 29.0
    assert result["contributing_scores"]["mca_score"] == 0
    assert "ml_score" not in result["contributing_scores"]


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

    assert result["detailed_breakdown"]["raw_combined_score"] == 72
    assert result["combined_score"] == 71
    assert "ml_score" not in result["contributing_scores"]


@pytest.mark.unit
def test_ml_score_is_informational_only_and_does_not_drive_decline():
    result = get_ensemble_recommendation(
        scores={
            "mca_score": 80,
            "mca_decision": "APPROVE",
            "subprime_score": 82,
            "ml_score": 0,
        },
        metrics={
            **_strong_metrics(),
            "Cash Flow Volatility": 0.62,
        },
        params=_strong_params(directors_score=48),
    )

    assert result["detailed_breakdown"]["raw_combined_score"] == 80.8
    assert result["combined_score"] == 80.8
    assert result["decision"] == "APPROVE"
    assert "ml_score" not in result["contributing_scores"]
    assert result["detailed_breakdown"]["informational_scores"]["ml_score"] == 0
    assert result["primary_reason"].startswith("Weighted MCA/Subprime score")
    assert "High cash flow volatility" not in result["primary_reason"]


@pytest.mark.unit
def test_mca_approve_cannot_override_weak_subprime_score():
    result = get_ensemble_recommendation(
        scores={
            "mca_score": 100,
            "mca_decision": "APPROVE",
            "subprime_score": 62,
            "ml_score": None,
        },
        metrics=_strong_metrics(),
        params=_strong_params(),
    )

    assert result["combined_score"] == 82.8
    assert result["decision"] == "REFER"
    assert "MCA score alone is not enough" in result["primary_reason"]


@pytest.mark.unit
def test_full_approval_requires_subprime_gate_even_with_perfect_mca():
    result = get_ensemble_recommendation(
        scores={
            "mca_score": 100,
            "mca_decision": "APPROVE",
            "subprime_score": 64,
            "ml_score": None,
        },
        metrics=_strong_metrics(),
        params=_strong_params(),
    )

    assert result["combined_score"] == 83.6
    assert result["decision"] == "REFER"
    assert "caps the case at referral" in result["primary_reason"]


@pytest.mark.unit
def test_primary_reason_explains_threshold_not_first_risk_factor():
    result = get_ensemble_recommendation(
        scores={
            "mca_score": 55,
            "mca_decision": "APPROVE",
            "subprime_score": 55,
            "ml_score": 90,
        },
        metrics={
            **_strong_metrics(),
            "Cash Flow Volatility": 0.75,
        },
        params=_strong_params(directors_score=48),
    )

    assert result["decision"] == "DECLINE"
    assert result["combined_score"] == 55
    assert "below the referral threshold" in result["primary_reason"]
    assert any("High cash flow volatility" in r for r in result["risk_factors"])
    assert "High cash flow volatility" not in result["primary_reason"]


@pytest.mark.unit
def test_calibrated_overlay_caps_approval_to_referral():
    result = get_ensemble_recommendation(
        scores={
            "mca_score": 86,
            "mca_decision": "APPROVE",
            "subprime_score": 82,
            "ml_score": None,
        },
        metrics={
            **_strong_metrics(),
            "Net Income": -2500,
            "Operating Margin": 0.08,
        },
        params=_strong_params(),
    )

    assert result["decision"] == "REFER"
    assert result["combined_score"] >= 75
    assert "calibrated risk overlays cap" in result["primary_reason"].lower()
    assert result["detailed_breakdown"]["calibration_risk_overlays"]["severe_count"] == 1


@pytest.mark.unit
def test_stacked_calibrated_overlays_decline():
    result = get_ensemble_recommendation(
        scores={
            "mca_score": 88,
            "mca_decision": "APPROVE",
            "subprime_score": 84,
            "ml_score": None,
        },
        metrics={
            **_strong_metrics(),
            "Net Income": -2500,
            "Operating Margin": -0.05,
        },
        params=_strong_params(),
    )

    assert result["decision"] == "DECLINE"
    assert "Calibrated risk overlays trigger decline" in result["primary_reason"]
    assert result["detailed_breakdown"]["calibration_risk_overlays"]["decline_overlay"] is True


@pytest.mark.unit
def test_single_mca_serious_signal_is_not_hard_stop():
    result = get_ensemble_recommendation(
        scores={
            "mca_score": 75,
            "mca_decision": "REFER",
            "mca_rule_signals": {
                "inflow_days_30d": 6,
                "max_inflow_gap_days": 4,
                "inflow_cv": 0.4,
            },
            "subprime_score": 82,
            "ml_score": None,
        },
        metrics=_strong_metrics(),
        params=_strong_params(),
    )

    assert result["decision"] == "REFER"
    assert result["combined_score"] > 75
    assert result["score_convergence"] != "N/A - Hard Stop"

