import pandas as pd
import pytest

from app.services.card_terminal_ingestion import CardTerminalIngestionService


@pytest.mark.unit
def test_card_processing_insights_capture_quality_and_reconciliation():
    service = CardTerminalIngestionService()
    parsed = pd.DataFrame(
        [
            {
                "filename": "jan.pdf",
                "provider": "Stripe",
                "statement_end": "2026-01-31",
                "gross_card_sales": 10000.0,
                "refunds_amount": 200.0,
                "chargebacks_amount": 0.0,
                "fees_total": 250.0,
                "transaction_count": 200,
            },
            {
                "filename": "feb.pdf",
                "provider": "Stripe",
                "statement_end": "2026-02-28",
                "gross_card_sales": 12000.0,
                "refunds_amount": 300.0,
                "chargebacks_amount": 0.0,
                "fees_total": 280.0,
                "transaction_count": 240,
            },
            {
                "filename": "mar.pdf",
                "provider": "Stripe",
                "statement_end": "2026-03-31",
                "gross_card_sales": 11000.0,
                "refunds_amount": 250.0,
                "chargebacks_amount": 0.0,
                "fees_total": 260.0,
                "transaction_count": 220,
            },
        ]
    )
    monthly = service.summarize_by_month(parsed)
    comparison = pd.DataFrame(
        [
            {"year_month": "2026-01", "gross_card_sales": 10000.0, "bank_revenue_inflows": 11000.0},
            {"year_month": "2026-02", "gross_card_sales": 12000.0, "bank_revenue_inflows": 12500.0},
            {"year_month": "2026-03", "gross_card_sales": 11000.0, "bank_revenue_inflows": 11800.0},
        ]
    )

    insights = service.derive_card_processing_insights(
        parsed,
        monthly,
        {"comparison": comparison, "summary": {"reconciliation_quality": "Good"}},
    )

    assert insights["Card Processing Insight Layer"] == "Available"
    assert insights["Card Sales Total"] == 33000.0
    assert insights["Card Refund Ratio"] == pytest.approx(0.023, abs=0.001)
    assert insights["Card Chargeback Ratio"] == 0.0
    assert insights["Card MCA Suitability"] == "Strong"
    assert "Card sales reconcile well to bank revenue" in insights["Card Processing Positive Signals"]


@pytest.mark.unit
def test_card_processing_insights_flag_chargebacks_and_missing_bank_evidence():
    service = CardTerminalIngestionService()
    parsed = pd.DataFrame(
        [
            {
                "filename": "apr.pdf",
                "provider": "Clover",
                "statement_end": "2026-04-30",
                "gross_card_sales": 10000.0,
                "refunds_amount": 1200.0,
                "chargebacks_amount": 250.0,
                "fees_total": 400.0,
                "transaction_count": 100,
            }
        ]
    )
    monthly = service.summarize_by_month(parsed)
    comparison = pd.DataFrame(
        [{"year_month": "2026-04", "gross_card_sales": 10000.0, "bank_revenue_inflows": 5000.0}]
    )

    insights = service.derive_card_processing_insights(
        parsed,
        monthly,
        {"comparison": comparison, "summary": {"reconciliation_quality": "Poor"}},
    )

    assert insights["Card MCA Suitability"] == "Weak"
    assert insights["Card Chargeback Ratio"] == pytest.approx(0.025)
    assert insights["Card Unmatched Sales Shortfall"] == 5000.0
    assert "Elevated chargeback ratio" in insights["Card Processing Concerns"]
    assert "Material card sales shortfall versus bank revenue evidence" in insights["Card Processing Concerns"]
