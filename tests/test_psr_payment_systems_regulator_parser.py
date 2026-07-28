from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services.card_terminal_ingestion import CardTerminalIngestionService
from app.services.payment_provider_registry import provider_catalog


PSR_DIR = Path(
    r"C:\Users\Massimo Cristi\OneDrive - Savvy Loan Products Ltd\Merchant Cash Advance (MCA)\Applications\UNDERWRITING\Revline Service Centre Ltd"
)


@pytest.mark.unit
def test_psr_parser_extracts_authoritative_monthly_totals():
    text = """
    Your card payments summary
    We are required to provide this information by the Payment Systems Regulator
    Monthly transaction statement 01-06-2026 to 30-06-2026
    Merchant number: GB0000000193348
    RHK MOT LTD | GB0000000193348
    EVO Payments International
    Monthly activity for other fees
    Fee type Transactions Fees in GBP
    Refunds 2 -1,00
    Total -1,00
    Monthly activity of other fees
    Total Visa and MasterCard activity by transaction type plus DCC sales
    Total 30 479,24230 0,000 0,00% 0,00
    Daily Settlement Report
    Total 30 479,24 -251,01 236,77 0,00 0,00 0,00 0,00 30 725,16
    """
    service = CardTerminalIngestionService()

    assert service._is_psr_payment_systems_regulator_statement(text)
    result = service._parse_psr_payment_systems_regulator_pdf("JUNE.pdf", text)

    assert result.provider == "PSR Payment Systems Regulator"
    assert result.parser == "psr_payment_systems_regulator_pdf_v1"
    assert result.merchant_id == "GB0000000193348"
    assert result.raw_summary["merchant_name"] == "RHK MOT LTD"
    assert result.statement_start == date(2026, 6, 1)
    assert result.statement_end == date(2026, 6, 30)
    assert result.gross_card_sales == pytest.approx(30479.24)
    assert result.refunds_amount == pytest.approx(251.01)
    assert result.chargebacks_amount == 0
    assert result.fees_total == pytest.approx(237.77)
    assert result.transaction_count == 230
    assert result.confidence == 0.98
    assert result.extraction_diagnostics["fallback_used"] is False


@pytest.mark.unit
def test_psr_provider_is_listed_as_native():
    item = next(
        row for row in provider_catalog()
        if row["provider"] == "PSR Payment Systems Regulator"
    )
    assert item["parser_support"] == "Native profile"


@pytest.mark.unit
@pytest.mark.parametrize(
    "filename,start,end,gross,refunds,chargebacks,fees,transactions",
    [
        ("639208483997260607 JAN.pdf", date(2026, 1, 1), date(2026, 1, 31), 33856.06, 43.82, 0.0, 293.86, 189),
        ("639208483799833295 FEB.pdf", date(2026, 2, 1), date(2026, 2, 28), 22056.65, 396.00, 150.0, 123.49, 157),
        ("639208483573490412 MAR.pdf", date(2026, 3, 1), date(2026, 3, 31), 16032.72, 84.00, 0.0, 112.16, 165),
        ("639208482916297396 APR.pdf", date(2026, 4, 1), date(2026, 4, 30), 10368.02, 144.00, 0.0, 70.04, 109),
        ("639208482758217730 MAY.pdf", date(2026, 5, 1), date(2026, 5, 31), 8559.18, 26.50, 216.0, 66.74, 93),
        ("639208482441132533 JUNE.pdf", date(2026, 6, 1), date(2026, 6, 30), 30479.24, 251.01, 0.0, 237.77, 230),
    ],
)
def test_real_psr_statements_parse_when_available(
    filename, start, end, gross, refunds, chargebacks, fees, transactions
):
    path = PSR_DIR / filename
    if not path.exists():
        pytest.skip("Real PSR statement fixture is not available on this machine")

    service = CardTerminalIngestionService()
    result = service._parse_single(
        SimpleNamespace(name=filename, getvalue=path.read_bytes)
    )

    assert result.provider == "PSR Payment Systems Regulator"
    assert result.statement_start == start
    assert result.statement_end == end
    assert result.gross_card_sales == pytest.approx(gross)
    assert result.refunds_amount == pytest.approx(refunds)
    assert result.chargebacks_amount == pytest.approx(chargebacks)
    assert result.fees_total == pytest.approx(fees)
    assert result.transaction_count == transactions
    assert result.confidence == 0.98

