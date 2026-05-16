import pandas as pd
import pytest

from app.services.business_risk_signals import calculate_business_metrics, categorize_business_transactions
from app.services.data_processor import TransactionCategorizer
from app.services.open_banking_insights import derive_open_banking_insights


@pytest.fixture
def business_transactions():
    rows = [
        {"date": "2026-01-05", "amount": -1000, "name_y": "Stripe payout", "balances.available": 1200},
        {"date": "2026-01-08", "amount": -500, "name_y": "Invoice payment ACME", "balances.available": 1700},
        {"date": "2026-01-12", "amount": -300, "name_y": "Director loan capital introduced", "balances.available": 2000},
        {"date": "2026-01-14", "amount": -200, "name_y": "Transfer from savings", "balances.available": 2200},
        {"date": "2026-01-14", "amount": 200, "name_y": "Transfer to current account", "balances.available": 2000},
        {"date": "2026-01-20", "amount": -600, "name_y": "Iwoca disbursement", "balances.available": 2600},
        {"date": "2026-01-22", "amount": 150, "name_y": "Iwoca repayment", "balances.available": 2450},
        {"date": "2026-01-24", "amount": 20, "name_y": "Monthly account fee", "balances.available": 2430},
        {"date": "2026-01-28", "amount": 400, "name_y": "Office rent", "balances.available": 2030},
        {"date": "2026-02-05", "amount": -1000, "name_y": "Stripe payout", "balances.available": 3030},
        {"date": "2026-02-08", "amount": -500, "name_y": "Invoice payment ACME", "balances.available": 3530},
        {"date": "2026-02-12", "amount": -300, "name_y": "Shareholder loan injection", "balances.available": 3830},
        {"date": "2026-02-14", "amount": -200, "name_y": "Transfer from savings", "balances.available": 4030},
        {"date": "2026-02-14", "amount": 200, "name_y": "Transfer to current account", "balances.available": 3830},
        {"date": "2026-02-22", "amount": 150, "name_y": "Iwoca repayment", "balances.available": 3680},
        {"date": "2026-02-24", "amount": 20, "name_y": "Monthly account fee", "balances.available": 3660},
        {"date": "2026-02-27", "amount": 35, "name_y": "Unpaid DD fee", "balances.available": 3625},
        {"date": "2026-02-28", "amount": 400, "name_y": "Office rent", "balances.available": 3225},
        {"date": "2026-03-05", "amount": -1000, "name_y": "Stripe payout", "balances.available": 4225},
        {"date": "2026-03-08", "amount": -500, "name_y": "Invoice payment ACME", "balances.available": 4725},
        {"date": "2026-03-14", "amount": -200, "name_y": "Transfer from savings", "balances.available": 4925},
        {"date": "2026-03-14", "amount": 200, "name_y": "Transfer to current account", "balances.available": 4725},
        {"date": "2026-03-22", "amount": 150, "name_y": "Iwoca repayment", "balances.available": 4575},
        {"date": "2026-03-24", "amount": 20, "name_y": "Monthly account fee", "balances.available": 4555},
        {"date": "2026-03-28", "amount": 400, "name_y": "Office rent", "balances.available": 4155},
        {"date": "2026-04-05", "amount": -1000, "name_y": "Stripe payout", "balances.available": 5155},
        {"date": "2026-04-08", "amount": -500, "name_y": "Invoice payment ACME", "balances.available": 5655},
        {"date": "2026-04-14", "amount": -200, "name_y": "Transfer from savings", "balances.available": 5855},
        {"date": "2026-04-14", "amount": 200, "name_y": "Transfer to current account", "balances.available": 5655},
        {"date": "2026-04-22", "amount": 150, "name_y": "Iwoca repayment", "balances.available": 5505},
        {"date": "2026-04-24", "amount": 20, "name_y": "Monthly account fee", "balances.available": 5485},
        {"date": "2026-04-28", "amount": 400, "name_y": "Office rent", "balances.available": 5085},
    ]
    return pd.DataFrame(rows)


@pytest.mark.unit
def test_business_metrics_separate_revenue_from_transfers_and_funding(business_transactions):
    metrics = calculate_business_metrics(business_transactions, company_age_months=24)

    assert metrics["Total Revenue"] == 6000.0
    assert metrics["Funding Inflow Total"] == 600.0
    assert metrics["Total Debt"] == 600.0
    assert metrics["Internal Transfer Inflow Total"] == 800.0
    assert metrics["Internal Transfer Outflow Total"] == 800.0
    assert metrics["Bank Charge Count"] == 4
    assert metrics["Number of Bounced Payments"] == 1
    assert metrics["Funding Reliance Ratio"] == pytest.approx(0.091, abs=0.001)
    assert metrics["Bank Charge Burden"] == pytest.approx(0.0133, abs=0.0002)
    assert metrics["Active Lenders Detected"] >= 1
    assert metrics["Debt Stacking Risk"] == "Low"
    assert metrics["Revenue Source Count"] >= 2
    assert metrics["Revenue Regularity Score"] >= 40


@pytest.mark.unit
def test_transaction_categorizer_classifies_business_specific_inflows_and_charges():
    categorizer = TransactionCategorizer()

    funding_category, _ = categorizer.categorize_transaction({"amount": -250, "name_y": "Director loan capital introduced"})
    transfer_category, _ = categorizer.categorize_transaction({"amount": -250, "name_y": "Transfer from savings"})
    bank_charge_category, _ = categorizer.categorize_transaction({"amount": 25, "name_y": "Monthly account fee"})
    loan_category, _ = categorizer.categorize_transaction({"amount": -1000, "name_y": "Iwoca disbursement"})

    assert funding_category == "Funding Inflow"
    assert transfer_category == "Transfer In"
    assert bank_charge_category == "Bank Charge"
    assert loan_category == "Loans"


@pytest.mark.unit
def test_open_banking_insights_are_derived_without_scoring_flag(business_transactions):
    categorized = categorize_business_transactions(business_transactions)
    insights = derive_open_banking_insights(categorized, requested_loan=3000)

    assert insights["Open Banking Insights Used In Score"] == "No - analysis/export only"
    assert insights["OB True Revenue"] == 6000.0
    assert insights["OB Non-Revenue Inflows"] >= 2000.0
    assert insights["OB Debt Repayment Burden"] > 0
    assert insights["OB Recent Loan Credits 30D"] >= 0.0
    assert insights["OB Requested Loan To Monthly Revenue"] == 2.0
