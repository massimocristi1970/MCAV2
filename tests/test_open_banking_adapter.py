import pandas as pd
import pytest

from app.services.open_banking_adapter import (
    AMOUNT_CONVENTION_API_TYPED,
    AMOUNT_CONVENTION_BANK_SIGNED,
    AMOUNT_CONVENTION_PLAID_SIGNED,
    detect_amount_convention,
    normalize_open_banking_payload,
)
from app.services.business_risk_signals import categorize_business_transactions
from build_training_dataset import _flatten_transactions


@pytest.mark.unit
def test_plaid_signed_payload_preserves_negative_credit_amounts():
    payload = {
        "accounts": [{"account_id": "acc_1"}],
        "transactions": [
            {
                "transaction_id": "txn_1",
                "account_id": "acc_1",
                "date": "2026-02-01",
                "amount": -2500,
                "name": "Stripe payout",
                "personal_finance_category": {
                    "primary": "TRANSFER_IN",
                    "detailed": "TRANSFER_IN_ACCOUNT_TRANSFER",
                },
            }
        ],
    }

    ob_payload = normalize_open_banking_payload(payload)

    assert ob_payload.metadata["amount_convention"] == AMOUNT_CONVENTION_PLAID_SIGNED
    assert ob_payload.transactions[0]["amount"] == -2500


@pytest.mark.unit
def test_api_typed_positive_credit_debit_is_normalized_to_plaid_signs():
    txns = [
        {"date": "2026-01-01", "amount": 100, "type": "CREDIT", "name": "Customer payment"},
        {"date": "2026-01-02", "amount": 40, "type": "DEBIT", "name": "Rent"},
    ]

    ob_payload = normalize_open_banking_payload(txns)

    assert ob_payload.metadata["amount_convention"] == AMOUNT_CONVENTION_API_TYPED
    assert [txn["amount"] for txn in ob_payload.transactions] == [-100.0, 40.0]


@pytest.mark.unit
def test_bank_signed_export_is_inverted_and_warned():
    txns = [{"date": f"2026-01-{day:02d}", "amount": -10, "name": "Card spend"} for day in range(1, 9)]
    txns += [{"date": "2026-01-09", "amount": 100, "name": "Deposit"}]

    convention, warning = detect_amount_convention(txns)
    ob_payload = normalize_open_banking_payload(txns)

    assert convention == AMOUNT_CONVENTION_BANK_SIGNED
    assert warning
    assert ob_payload.transactions[0]["amount"] == 10.0
    assert ob_payload.transactions[-1]["amount"] == -100.0
    assert "amount_convention_warning" in ob_payload.metadata


@pytest.mark.unit
def test_nested_payload_drops_junk_and_duplicates():
    payload = {
        "wrapper": {
            "data": {
                "transactions": [
                    {},
                    {"date": "2026-01-01", "amount": -50, "name": "Invoice payment"},
                    {"date": "2026-01-01", "amount": -50, "name": "Invoice payment"},
                ]
            }
        }
    }

    ob_payload = normalize_open_banking_payload(payload)

    assert len(ob_payload.transactions) == 1
    assert ob_payload.metadata["dropped_junk_transaction_count"] == 1
    assert ob_payload.metadata["dropped_duplicate_transaction_count"] == 1


@pytest.mark.unit
def test_flatten_transactions_uses_canonical_adapter():
    txns = _flatten_transactions(
        [{"date": "2026-01-01", "amount": 100, "type": "CREDIT", "name": "Customer payment"}]
    )

    assert txns[0]["amount"] == -100.0


@pytest.mark.unit
def test_business_categorizer_still_preserves_processor_revenue_over_plaid_transfer():
    ob_payload = normalize_open_banking_payload(
        {
            "transactions": [
                {
                    "date": "2026-02-01",
                    "amount": -2500,
                    "name": "Stripe Payments UK Ltd / Ref: Stripe",
                    "personal_finance_category": {
                        "primary": "TRANSFER_IN",
                        "detailed": "TRANSFER_IN_ACCOUNT_TRANSFER",
                    },
                }
            ]
        }
    )

    categorized = categorize_business_transactions(pd.DataFrame(ob_payload.transactions))

    assert categorized.loc[0, "subcategory"] == "Income"
    assert bool(categorized.loc[0, "is_revenue"]) is True

