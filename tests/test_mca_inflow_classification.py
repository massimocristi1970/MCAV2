import pandas as pd
import pytest

from app.services.business_risk_signals import categorize_business_transactions
from build_training_dataset import build_mca_features


@pytest.mark.unit
def test_special_inflows_do_not_count_as_revenue_in_business_categorizer():
    df = pd.DataFrame(
        [
            {"date": "2026-02-01", "amount": -500, "amount_original": -500, "name_y": "Transfer from savings", "personal_finance_category.detailed": "transfer_in_account_transfer"},
            {"date": "2026-02-02", "amount": -200, "amount_original": -200, "name_y": "Tax refund", "personal_finance_category.detailed": "income_other_income"},
            {"date": "2026-02-03", "amount": -900, "amount_original": -900, "name_y": "Stripe payout", "personal_finance_category.detailed": "income_other_income"},
        ]
    )

    categorized = categorize_business_transactions(df)

    assert categorized.loc[0, "subcategory"] == "Transfer In"
    assert bool(categorized.loc[0, "is_revenue"]) is False
    assert categorized.loc[1, "subcategory"] == "Special Inflow"
    assert bool(categorized.loc[1, "is_revenue"]) is False
    assert bool(categorized.loc[2, "is_revenue"]) is True


@pytest.mark.unit
def test_mca_features_exclude_transfers_and_loans_from_inflow_consistency():
    transactions = [
        {"date": "2026-01-01", "amount": -1000, "name": "Stripe payout", "personal_finance_category": {"primary": "INCOME", "detailed": "income_other_income"}},
        {"date": "2026-01-10", "amount": -800, "name": "Invoice payment", "personal_finance_category": {"primary": "INCOME", "detailed": "income_other_income"}},
        {"date": "2026-01-15", "amount": -500, "name": "Transfer from savings", "personal_finance_category": {"primary": "TRANSFER_IN", "detailed": "transfer_in_account_transfer"}},
        {"date": "2026-01-20", "amount": -700, "name": "Iwoca disbursement", "personal_finance_category": {"primary": "TRANSFER_IN", "detailed": "transfer_in_cash_advances_and_loans"}},
        {"date": "2026-02-01", "amount": -1100, "name": "Stripe payout", "personal_finance_category": {"primary": "INCOME", "detailed": "income_other_income"}},
        {"date": "2026-02-10", "amount": -900, "name": "Invoice payment", "personal_finance_category": {"primary": "INCOME", "detailed": "income_other_income"}},
        {"date": "2026-02-18", "amount": 600, "name": "Rent", "personal_finance_category": {"primary": "GENERAL_SERVICES", "detailed": "general_services_rent_and_utilities"}},
    ]

    features = build_mca_features(transactions)

    assert features["inflow_days_30d"] == 2
    assert features["max_inflow_gap_days"] == 22
    assert features["non_revenue_inflow_total"] == 1200
