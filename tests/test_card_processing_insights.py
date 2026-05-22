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


@pytest.mark.unit
def test_paypal_transaction_history_pdf_parser_counts_completed_customer_payments_once():
    service = CardTerminalIngestionService()
    text = """
Recovia Ltd
ben.portman@recoviaholdings.co.uk
Date Description Status Currency Gross Fee Net
15/04/2026 Express Checkout Payment: Simon Nicholls
 ID: 2A0855447R198114B Completed GBP 280.00 -3.66 276.34
15/04/2026 User Initiated Withdrawal
 ID: 92E124305L773220J Completed GBP -276.34 0.00 -276.34
15/04/2026 Express Checkout Payment: Christi King
 ID: 19H593285U064313D Completed GBP 170.00 -2.34 167.66
15/04/2026 General Hold
 ID: 1MA35002B6305420C Completed GBP -51.47 0.00 -51.47
15/04/2026 Express Checkout Payment: Pat Clark
 ID: 6RB872464B609293S Completed GBP 16.07 -0.49 15.58
15/04/2026 General Hold
 ID: 7GG44000D7962471U Completed GBP -15.58 0.00 -15.58
15/04/2026 Express Checkout Payment: Tony Doyle
 ID: 6PH96523J3958210M Completed GBP 190.00 -2.58 187.42
15/04/2026 Express Checkout Payment: Pat Clark
 ID: 6RB872464B609293S Completed GBP 16.07 0.00 16.07
15/04/2026 Express Checkout Payment: Christi King
 ID: 19H593285U064313D Completed GBP 170.00 0.00 170.00
15/04/2026 Express Checkout Payment: Tony Doyle
 ID: 6PH96523J3958210M Completed GBP 190.00 0.00 190.00
15/04/2026 General Authorisation: Rev Corp
 ID: 2XH16143M7150631T Pending GBP -19.92 0.00 -19.92
Transaction History
April 15, 2026 through May 21, 2026
Date Description Status Currency Gross Fee Net
16/04/2026 General PayPal Debit Card Transaction: Rev Corp
 ID: 2SH99623Y97494353 Completed GBP -19.92 0.00 -19.92
20/04/2026 General Credit Card Deposit
 ID: 0PU38611B5174110S Refused GBP 17.90 0.00 17.90
22/04/2026 Debit Card Cashback Bonus
 ID: 3WS134362D303713N Completed GBP 0.10 0.00 0.10
24/04/2026 Express Checkout Payment: Be World Class
 ID: 5M516281M3623654W Completed GBP 170.00 -5.23 164.77
07/05/2026 Express Checkout Payment: Darren Adamson
 ID: 1CD60508FF463414H Completed GBP 759.00 -9.41 749.59
Transaction History
April 15, 2026 through May 21, 2026
"""

    assert service._is_paypal_transaction_history_pdf(text)
    result = service._parse_paypal_transaction_history_pdf("Download-3.PDF", text)

    assert result.provider == "PayPal"
    assert result.parser == "paypal_transaction_history_pdf_v1"
    assert result.statement_start.isoformat() == "2026-04-15"
    assert result.statement_end.isoformat() == "2026-05-21"
    assert result.merchant_id == "ben.portman@recoviaholdings.co.uk"
    assert result.gross_card_sales == pytest.approx(1585.07)
    assert result.fees_total == pytest.approx(23.71)
    assert result.transaction_count == 6
    assert result.confidence == pytest.approx(0.9)
