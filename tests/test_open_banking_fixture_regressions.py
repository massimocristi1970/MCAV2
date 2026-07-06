import json
from pathlib import Path

import pandas as pd
import pytest

from app.services.business_risk_signals import calculate_business_metrics, categorize_business_transactions
from app.services.open_banking_adapter import normalize_open_banking_payload, transactions_to_dataframe


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "open_banking"


def _fixture_df(name: str) -> pd.DataFrame:
    payload = json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))
    ob_payload = normalize_open_banking_payload(payload)
    return transactions_to_dataframe(ob_payload.transactions, ob_payload.accounts)


@pytest.mark.unit
def test_processor_revenue_fixture_preserves_revenue_despite_plaid_transfer_labels():
    df = _fixture_df("business_revenue_processors.json")
    categorized = categorize_business_transactions(df)
    metrics = calculate_business_metrics(df, company_age_months=24)

    assert categorized.loc[categorized["name"].str.contains("Stripe", case=False), "subcategory"].iloc[0] == "Income"
    assert categorized.loc[categorized["name"].str.contains("SumUp", case=False), "subcategory"].iloc[0] == "Income"
    assert metrics["Total Revenue"] == 2000.0
    assert metrics["Balance Source"] == "reconstructed"
    assert metrics["Balance Confidence"] == "medium"


@pytest.mark.unit
def test_transfer_funding_special_inflow_fixture_excludes_non_revenue_credits():
    df = _fixture_df("transfers_funding_and_special_inflows.json")
    metrics = calculate_business_metrics(df, company_age_months=18)

    assert metrics["Total Revenue"] == 1000.0
    assert metrics["Internal Transfer Inflow Total"] == 300.0
    assert metrics["Funding Inflow Total"] == 400.0
    assert metrics["Special Inflow Total"] == 250.0
    assert metrics["OB Non-Revenue Inflows"] == 950.0
    assert metrics["Balance Source"] == "estimated"
    assert metrics["Balance Confidence"] == "low"


@pytest.mark.unit
def test_lender_repayment_fixture_keeps_debt_and_bank_charge_signals_separate():
    df = _fixture_df("lender_repayments_and_charges.json")
    metrics = calculate_business_metrics(df, company_age_months=12)

    assert metrics["Total Revenue"] == 2000.0
    assert metrics["Total Debt"] == 750.0
    assert metrics["Total Debt Repayments"] == 150.0
    assert metrics["Bank Charge Count"] == 1
    assert metrics["Number of Bounced Payments"] == 1

