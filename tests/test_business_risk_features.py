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
    youlend_sales_category, _ = categorizer.categorize_transaction(
        {
            "amount": -50.91,
            "name_y": "YouLend Limited YL60406663OUT",
            "merchant_name": "YouLend",
            "personal_finance_category.detailed": "TRANSFER_IN_CASH_ADVANCES_AND_LOANS",
        }
    )
    youlend_funding_category, _ = categorizer.categorize_transaction(
        {
            "amount": -840.00,
            "name_y": "YL III Limited YL60406663FND",
            "personal_finance_category.detailed": "INCOME_WAGES",
        }
    )

    assert funding_category == "Funding Inflow"
    assert transfer_category == "Transfer In"
    assert bank_charge_category == "Bank Charge"
    assert loan_category == "Loans"
    assert youlend_sales_category == "Income"
    assert youlend_funding_category == "Loans"


@pytest.mark.unit
def test_youlend_out_counts_as_revenue_but_fnd_counts_as_loan():
    df = pd.DataFrame(
        [
            {
                "date": "2026-05-17",
                "amount": -50.91,
                "name_y": "YouLend Limited YL60406663OUT",
                "merchant_name": "YouLend",
                "personal_finance_category.detailed": "TRANSFER_IN_CASH_ADVANCES_AND_LOANS",
            },
            {
                "date": "2026-03-19",
                "amount": -840.00,
                "name_y": "YL III Limited YL60406663FND",
                "personal_finance_category.detailed": "INCOME_WAGES",
            },
        ]
    )

    categorized = categorize_business_transactions(df)

    assert categorized.loc[0, "subcategory"] == "Income"
    assert bool(categorized.loc[0, "is_revenue"]) is True
    assert categorized.loc[1, "subcategory"] == "Loans"
    assert bool(categorized.loc[1, "is_revenue"]) is False
    assert bool(categorized.loc[1, "is_debt"]) is True


@pytest.mark.unit
def test_payment_processor_credits_beat_plaid_transfer_labels():
    df = pd.DataFrame(
        [
            {
                "date": "2026-02-01",
                "amount": -2500,
                "name_y": "Stripe Payments UK Ltd / Ref: Stripe",
                "personal_finance_category.detailed": "TRANSFER_IN_ACCOUNT_TRANSFER",
            },
            {
                "date": "2026-02-02",
                "amount": -900,
                "name_y": "Payment from Paypal Code 2060 - PAYPAL CODE 2060",
                "personal_finance_category.detailed": "TRANSFER_IN_ACCOUNT_TRANSFER",
            },
            {
                "date": "2026-02-03",
                "amount": -1100,
                "name_y": "Adyen N.V. Adyen",
                "personal_finance_category.detailed": "TRANSFER_IN_ACCOUNT_TRANSFER",
            },
            {
                "date": "2026-02-04",
                "amount": -450,
                "name_y": "Payment from The Gluten Free World Ltd - sent from SumUp",
                "personal_finance_category.detailed": "TRANSFER_IN_ACCOUNT_TRANSFER",
            },
        ]
    )

    categorized = categorize_business_transactions(df)

    assert categorized["subcategory"].tolist() == ["Income", "Income", "Income", "Income"]
    assert categorized["is_revenue"].tolist() == [True, True, True, True]


@pytest.mark.unit
def test_tax_and_pot_credits_do_not_count_as_revenue():
    df = pd.DataFrame(
        [
            {
                "date": "2026-02-01",
                "amount": -10000,
                "name_y": "HMRC VAT 436137208",
                "personal_finance_category.detailed": "INCOME_WAGES",
            },
            {
                "date": "2026-02-02",
                "amount": -500,
                "name_y": "Wages Pot Transfer",
                "personal_finance_category.detailed": "INCOME_WAGES",
            },
        ]
    )

    categorized = categorize_business_transactions(df)

    assert categorized.loc[0, "subcategory"] == "Special Inflow"
    assert bool(categorized.loc[0, "is_revenue"]) is False
    assert categorized.loc[1, "subcategory"] == "Transfer In"
    assert bool(categorized.loc[1, "is_revenue"]) is False


@pytest.mark.unit
def test_known_finance_repayments_are_debt_repayments_not_expenses():
    categorizer = TransactionCategorizer()

    examples = [
        "82442615 Close Brother Premium Finance",
        "GC Couture - Finbiz Funding Lim Direct Debit",
        "Mercedes Benz Fin Mercedes-Benz Finance",
        "Motonovo Finance L 21609437",
        "Card payment to BMW Finance on 24-05-2024",
    ]

    for name in examples:
        category, _ = categorizer.categorize_transaction({"amount": 250, "name_y": name})
        assert category == "Debt Repayments"


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
