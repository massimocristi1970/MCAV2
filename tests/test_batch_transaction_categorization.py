import pandas as pd
import pytest

from MCAV2_BatchProcessor.batch_processor_standalone import categorize_transactions, map_transaction_category


@pytest.mark.unit
def test_batch_uses_shared_categorizer_for_transfer_and_youlend_cases():
    assert map_transaction_category(
        {
            "amount": -500,
            "amount_original": -500,
            "name": "Transfer from savings",
            "personal_finance_category.detailed": "transfer_in_account_transfer",
        }
    ) == "Transfer In"

    assert map_transaction_category(
        {
            "amount": -50.91,
            "amount_original": -50.91,
            "name": "YouLend Limited YL60406663OUT",
            "merchant_name": "YouLend",
            "personal_finance_category.detailed": "TRANSFER_IN_CASH_ADVANCES_AND_LOANS",
        }
    ) == "Income"

    assert map_transaction_category(
        {
            "amount": -840.00,
            "amount_original": -840.00,
            "name": "YL III Limited YL60406663FND",
            "personal_finance_category.detailed": "INCOME_WAGES",
        }
    ) == "Loans"


@pytest.mark.unit
def test_batch_does_not_count_special_inflows_or_transfers_as_revenue():
    df = pd.DataFrame(
        [
            {
                "date": "2026-02-01",
                "amount": -500,
                "amount_original": -500,
                "name": "Transfer from savings",
                "personal_finance_category.detailed": "transfer_in_account_transfer",
            },
            {
                "date": "2026-02-02",
                "amount": -200,
                "amount_original": -200,
                "name": "Tax refund",
                "personal_finance_category.detailed": "income_other_income",
            },
            {
                "date": "2026-02-03",
                "amount": -900,
                "amount_original": -900,
                "name": "Stripe payout",
                "personal_finance_category.detailed": "income_other_income",
            },
        ]
    )

    categorized = categorize_transactions(df)

    assert categorized.loc[0, "subcategory"] == "Transfer In"
    assert bool(categorized.loc[0, "is_revenue"]) is False
    assert categorized.loc[1, "subcategory"] == "Special Inflow"
    assert bool(categorized.loc[1, "is_revenue"]) is False
    assert categorized.loc[2, "subcategory"] == "Income"
    assert bool(categorized.loc[2, "is_revenue"]) is True
