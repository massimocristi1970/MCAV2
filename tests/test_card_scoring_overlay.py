import pandas as pd
import pytest

from app.services.card_terminal_ingestion import CardTerminalIngestionService
from app.services.ensemble_scorer import get_ensemble_recommendation
from app.services.subprime_scoring_system import SubprimeScoring


def _base_metrics():
    return {
        "Debt Service Coverage Ratio": 1.6,
        "Revenue Growth Rate": 0.06,
        "Average Month-End Balance": 2500,
        "Cash Flow Volatility": 0.25,
        "Operating Margin": 0.08,
        "Net Income": 2000,
        "Average Negative Balance Days per Month": 1,
    }


def _base_params():
    return {
        "directors_score": 70,
        "company_age_months": 24,
        "industry": "Retail",
        "business_ccj": False,
    }


@pytest.mark.unit
def test_card_processing_insight_adjustment_rewards_strong_evidence():
    service = CardTerminalIngestionService()
    parsed = pd.DataFrame(
        [
            {"filename": "jan.pdf", "statement_end": "2026-01-31", "gross_card_sales": 10000.0, "refunds_amount": 100.0, "chargebacks_amount": 0.0, "fees_total": 200.0, "transaction_count": 200, "confidence": 0.95},
            {"filename": "feb.pdf", "statement_end": "2026-02-28", "gross_card_sales": 10500.0, "refunds_amount": 100.0, "chargebacks_amount": 0.0, "fees_total": 210.0, "transaction_count": 210, "confidence": 0.95},
            {"filename": "mar.pdf", "statement_end": "2026-03-31", "gross_card_sales": 11000.0, "refunds_amount": 100.0, "chargebacks_amount": 0.0, "fees_total": 220.0, "transaction_count": 220, "confidence": 0.95},
        ]
    )
    monthly = service.summarize_by_month(parsed)
    comparison = pd.DataFrame(
        [
            {"year_month": "2026-01", "gross_card_sales": 10000.0, "bank_revenue_inflows": 10200.0},
            {"year_month": "2026-02", "gross_card_sales": 10500.0, "bank_revenue_inflows": 10700.0},
            {"year_month": "2026-03", "gross_card_sales": 11000.0, "bank_revenue_inflows": 11200.0},
        ]
    )

    insights = service.derive_card_processing_insights(
        parsed,
        monthly,
        {"comparison": comparison, "summary": {"reconciliation_quality": "Good"}},
    )

    assert insights["Card Processing Insights Used In Score"] == "Yes - capped score overlay"
    assert insights["Card Processing Score Adjustment"] == 5.0
    assert any("overlay capped" in reason for reason in insights["Card Processing Score Adjustment Reasons"])


@pytest.mark.unit
def test_card_processing_insight_adjustment_penalises_weak_evidence():
    service = CardTerminalIngestionService()
    parsed = pd.DataFrame(
        [
            {"filename": "apr.pdf", "statement_end": "2026-04-30", "gross_card_sales": 10000.0, "refunds_amount": 1500.0, "chargebacks_amount": 250.0, "fees_total": 500.0, "transaction_count": 100, "confidence": 0.6},
        ]
    )
    monthly = service.summarize_by_month(parsed)
    comparison = pd.DataFrame(
        [{"year_month": "2026-04", "gross_card_sales": 10000.0, "bank_revenue_inflows": 4000.0}]
    )

    insights = service.derive_card_processing_insights(
        parsed,
        monthly,
        {"comparison": comparison, "summary": {"reconciliation_quality": "Poor"}},
    )

    assert insights["Card Processing Score Adjustment"] == -8.0
    assert insights["Card MCA Suitability"] == "Weak"
    assert any("overlay capped" in reason for reason in insights["Card Processing Score Adjustment Reasons"])


@pytest.mark.unit
def test_subprime_score_uses_card_processing_overlay():
    scorer = SubprimeScoring()
    metrics = _base_metrics()
    params = _base_params()

    baseline = scorer.calculate_subprime_score(dict(metrics), dict(params))
    with_card = scorer.calculate_subprime_score(
        {
            **metrics,
            "Card Processing Insight Layer": "Available",
            "Card Processing Score Adjustment": 5.0,
            "Card Processing Score Adjustment Reasons": ["Card sales reconcile well to bank revenue: +2.0"],
        },
        dict(params),
    )

    assert with_card["subprime_score"] == pytest.approx(baseline["subprime_score"] + 5.0)
    assert with_card["card_processing_score_adjustment"] == 5.0
    assert any("Card Processing Overlay: +5.0" in line for line in with_card["breakdown"])


@pytest.mark.unit
def test_ensemble_exposes_card_processing_risk_and_positive_factors():
    strong = get_ensemble_recommendation(
        scores={"mca_score": 75, "subprime_score": 75, "ml_score": 75},
        metrics={**_base_metrics(), "Card Processing Insight Layer": "Available", "Card Processing Score Adjustment": 4.0, "Card Processing Positive Signals": ["Card sales reconcile well to bank revenue"]},
        params=_base_params(),
    )
    weak = get_ensemble_recommendation(
        scores={"mca_score": 75, "subprime_score": 75, "ml_score": 75},
        metrics={**_base_metrics(), "Card Processing Insight Layer": "Available", "Card Processing Score Adjustment": -6.0, "Card MCA Suitability": "Weak", "Card Processing Concerns": ["Poor reconciliation between card statements and bank revenue"]},
        params=_base_params(),
    )

    assert any("Card processing overlay" in factor for factor in strong["positive_factors"])
    assert any("Card processing overlay" in factor for factor in weak["risk_factors"])