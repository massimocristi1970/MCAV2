import pandas as pd
import pytest

from app.services.business_risk_signals import categorize_business_transactions
from build_training_dataset import build_mca_features
from mca_scorecard_rules import decide_application


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
    base = pd.Timestamp.now().normalize() - pd.Timedelta(days=20)
    d = lambda offset: (base + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
    transactions = [
        {"date": d(0), "amount": -1000, "name": "Stripe payout", "personal_finance_category": {"primary": "INCOME", "detailed": "income_other_income"}},
        {"date": d(9), "amount": -800, "name": "Invoice payment", "personal_finance_category": {"primary": "INCOME", "detailed": "income_other_income"}},
        {"date": d(14), "amount": -500, "name": "Transfer from savings", "personal_finance_category": {"primary": "TRANSFER_IN", "detailed": "transfer_in_account_transfer"}},
        {"date": d(19), "amount": -700, "name": "Iwoca disbursement", "personal_finance_category": {"primary": "TRANSFER_IN", "detailed": "transfer_in_cash_advances_and_loans"}},
        {"date": d(25), "amount": -1100, "name": "Stripe payout", "personal_finance_category": {"primary": "INCOME", "detailed": "income_other_income"}},
        {"date": d(34), "amount": -900, "name": "Invoice payment", "personal_finance_category": {"primary": "INCOME", "detailed": "income_other_income"}},
        {"date": d(42), "amount": 600, "name": "Rent", "personal_finance_category": {"primary": "GENERAL_SERVICES", "detailed": "general_services_rent_and_utilities"}},
    ]

    features = build_mca_features(transactions)

    assert features["inflow_days_30d"] >= 2
    assert features["non_revenue_inflow_total"] == 1200


@pytest.mark.unit
def test_single_low_inflow_days_signal_refers_not_declines():
    decision, score, reasons = decide_application(
        {
            "inflow_days_30d": 6,
            "max_inflow_gap_days": 4,
            "inflow_cv": 0.4,
            "months_covered": 3,
            "txn_count_avg_month": 80,
        }
    )

    assert decision == "REFER"
    assert score >= 50
    assert any("inflow_days_30d<=" in reason for reason in reasons)


@pytest.mark.unit
def test_stacked_mca_consistency_failures_decline():
    decision, score, reasons = decide_application(
        {
            "inflow_days_30d": 6,
            "max_inflow_gap_days": 24,
            "inflow_cv": 0.4,
            "months_covered": 3,
            "txn_count_avg_month": 80,
        }
    )

    assert decision == "DECLINE"
    assert score < 50
    assert any("MCA decline" in reason for reason in reasons)
