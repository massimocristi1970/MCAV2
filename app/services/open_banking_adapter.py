"""Canonical Open Banking/Plaid payload adapter for business-account analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


AMOUNT_CONVENTION_PLAID_SIGNED = "plaid_signed"
AMOUNT_CONVENTION_API_TYPED = "api_typed"
AMOUNT_CONVENTION_BANK_SIGNED = "bank_signed"
AMOUNT_CONVENTION_MIXED = "mixed"


@dataclass
class OpenBankingPayload:
    """Normalized open banking payload plus ingestion quality metadata."""

    transactions: List[Dict[str, Any]]
    accounts: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_json_data(self) -> Dict[str, Any]:
        return {"accounts": self.accounts, "transactions": self.transactions}


def _raw_amount(txn: Dict[str, Any]) -> float:
    try:
        return float(txn.get("amount") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _txn_text(txn: Dict[str, Any]) -> str:
    fields = ("name", "name_y", "transaction_name", "merchant_name", "description", "original_description")
    return " ".join(str(txn.get(field) or "") for field in fields).strip()


def transaction_is_meaningful(txn: Dict[str, Any]) -> bool:
    """Return True when a row has at least one signal worth keeping."""
    if not isinstance(txn, dict):
        return False
    if str(txn.get("date") or txn.get("authorized_date") or "").strip():
        return True
    if _txn_text(txn):
        return True
    return _raw_amount(txn) != 0.0


def select_meaningful_transactions(txns: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    meaningful = [t for t in txns if transaction_is_meaningful(t)]
    return meaningful, len(txns) - len(meaningful)


def _pfc_primary(txn: Dict[str, Any]) -> str:
    pfc = txn.get("personal_finance_category") or {}
    if isinstance(pfc, dict):
        return str(pfc.get("primary") or "").upper()
    return str(txn.get("personal_finance_category.primary") or "").upper()


def _pfc_detailed(txn: Dict[str, Any]) -> str:
    pfc = txn.get("personal_finance_category") or {}
    if isinstance(pfc, dict):
        return str(pfc.get("detailed") or "").upper()
    return str(txn.get("personal_finance_category.detailed") or "").upper()


def _has_plaid_category_evidence(txn: Dict[str, Any]) -> bool:
    return bool(_pfc_primary(txn) or _pfc_detailed(txn) or txn.get("transaction_id"))


def detect_amount_convention(txns: List[Dict[str, Any]]) -> Tuple[str, Optional[str]]:
    """Infer how amounts are signed and return (convention, optional warning)."""
    if not txns:
        return AMOUNT_CONVENTION_PLAID_SIGNED, None

    n = len(txns)
    typed = sum(1 for t in txns if str(t.get("type") or "").strip())
    negative = sum(1 for t in txns if _raw_amount(t) < 0)
    plaid_evidence = sum(1 for t in txns if _has_plaid_category_evidence(t))
    typed_ratio = typed / n
    neg_ratio = negative / n
    plaid_ratio = plaid_evidence / n

    if typed_ratio >= 0.80:
        if neg_ratio >= 0.10:
            return (
                AMOUNT_CONVENTION_MIXED,
                "Mixed signs with mostly typed rows. Amounts were normalized using CREDIT/DEBIT type where present.",
            )
        return AMOUNT_CONVENTION_API_TYPED, None

    if plaid_ratio >= 0.25:
        return AMOUNT_CONVENTION_PLAID_SIGNED, None

    if neg_ratio >= 0.10:
        return (
            AMOUNT_CONVENTION_BANK_SIGNED,
            "Upload looks like a bank-signed export rather than Plaid-signed data. Amount signs were normalized.",
        )

    return AMOUNT_CONVENTION_PLAID_SIGNED, None


def normalize_amount_to_plaid_signed(
    txn: Dict[str, Any],
    *,
    amount_convention: str = AMOUNT_CONVENTION_PLAID_SIGNED,
) -> float:
    """Return Plaid convention amount: negative = money in, positive = money out."""
    amount = _raw_amount(txn)
    txn_type = str(txn.get("type") or "").upper().strip()

    if amount_convention == AMOUNT_CONVENTION_BANK_SIGNED:
        if amount < 0:
            return abs(amount)
        if amount > 0:
            return -abs(amount)
        return 0.0

    if amount_convention in {AMOUNT_CONVENTION_API_TYPED, AMOUNT_CONVENTION_MIXED}:
        if txn_type == "CREDIT":
            return -abs(amount)
        if txn_type == "DEBIT":
            return abs(amount)

    return amount


def _extract_accounts(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, dict) and isinstance(obj.get("accounts"), list):
        return [a for a in obj["accounts"] if isinstance(a, dict)]
    return []


def _extract_transactions(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        return [t for t in obj if isinstance(t, dict)]

    txns: List[Dict[str, Any]] = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            for key in ("transactions", "transaction", "data", "items", "results"):
                value = x.get(key)
                if isinstance(value, list):
                    txns.extend(t for t in value if isinstance(t, dict))

            for value in x.values():
                walk(value)
        elif isinstance(x, list):
            for value in x:
                walk(value)

    walk(obj)
    return txns


def _dedupe_transactions(txns: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    seen: set[Tuple[str, str, str, str]] = set()
    out: List[Dict[str, Any]] = []
    for txn in txns:
        txn_id = str(txn.get("transaction_id") or "")
        key = (
            txn_id,
            str(txn.get("date") or txn.get("authorized_date") or ""),
            str(txn.get("amount") or ""),
            str(txn.get("account_id") or "") + "|" + _txn_text(txn),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(txn)
    return out, len(txns) - len(out)


def normalize_open_banking_payload(raw: Any) -> OpenBankingPayload:
    """Extract, clean, dedupe, and sign-normalize transactions from a payload."""
    accounts = _extract_accounts(raw)
    extracted = _extract_transactions(raw)
    meaningful, dropped_junk = select_meaningful_transactions(extracted)
    deduped, dropped_duplicates = _dedupe_transactions(meaningful)
    convention, warning = detect_amount_convention(deduped)

    normalized: List[Dict[str, Any]] = []
    for txn in deduped:
        row = dict(txn)
        row["amount"] = normalize_amount_to_plaid_signed(row, amount_convention=convention)
        row.setdefault("amount_original", row["amount"])
        normalized.append(row)

    metadata: Dict[str, Any] = {
        "raw_transaction_count": len(extracted),
        "meaningful_transaction_count": len(meaningful),
        "transaction_count": len(normalized),
        "dropped_junk_transaction_count": dropped_junk,
        "dropped_duplicate_transaction_count": dropped_duplicates,
        "account_count": len(accounts),
        "amount_convention": convention,
    }
    if warning:
        metadata["amount_convention_warning"] = warning
    if dropped_junk:
        metadata["junk_transactions_warning"] = (
            f"Dropped {dropped_junk} empty row(s) with no date, amount, or description/merchant."
        )
    if dropped_duplicates:
        metadata["duplicate_transactions_warning"] = f"Dropped {dropped_duplicates} duplicate transaction row(s)."

    return OpenBankingPayload(transactions=normalized, accounts=accounts, metadata=metadata)


def _account_balance_lookup(accounts: List[Dict[str, Any]] | None) -> Dict[str, float]:
    lookup: Dict[str, float] = {}
    for account in accounts or []:
        if not isinstance(account, dict):
            continue
        account_id = account.get("account_id")
        if not account_id:
            continue
        balances = account.get("balances") or {}
        if not isinstance(balances, dict):
            continue
        raw_balance = balances.get("current")
        if raw_balance is None:
            raw_balance = balances.get("available")
        try:
            lookup[str(account_id)] = float(raw_balance)
        except (TypeError, ValueError):
            continue
    return lookup


def _add_reconstructed_balances(df: pd.DataFrame, accounts: List[Dict[str, Any]] | None) -> pd.DataFrame:
    account_balances = _account_balance_lookup(accounts)
    if df.empty or not account_balances or "account_id" not in df.columns:
        return df

    pieces: List[pd.DataFrame] = []
    for account_id, group in df.groupby(df["account_id"].astype(str), dropna=False):
        group = group.sort_values("date", ascending=False).copy()
        current_balance = account_balances.get(str(account_id))
        if current_balance is None:
            pieces.append(group)
            continue

        reconstructed = []
        balance = current_balance
        for _, row in group.iterrows():
            reconstructed.append(balance)
            try:
                balance += float(row.get("amount") or 0.0)
            except (TypeError, ValueError):
                pass
        group["calculated_balance"] = reconstructed
        pieces.append(group)

    if not pieces:
        return df
    return pd.concat(pieces, ignore_index=True).sort_values("date", ascending=False).reset_index(drop=True)


def add_balance_provenance(df: pd.DataFrame, accounts: List[Dict[str, Any]] | None = None) -> pd.DataFrame:
    """Annotate balance source/confidence and reconstruct when account balances allow it."""
    if df.empty:
        return df

    df = df.copy()
    provided_balance_cols = [col for col in ("balances.available", "balances.current") if col in df.columns]
    if provided_balance_cols:
        coverage = max(pd.to_numeric(df[col], errors="coerce").notna().mean() for col in provided_balance_cols)
        if coverage > 0:
            df["balance_source"] = "provided"
            df["balance_confidence"] = "high" if coverage >= 0.8 else "medium"
            df["balance_warning"] = ""
            return df

    df = _add_reconstructed_balances(df, accounts)
    if "calculated_balance" in df.columns and pd.to_numeric(df["calculated_balance"], errors="coerce").notna().any():
        df["balance_source"] = "reconstructed"
        df["balance_confidence"] = "medium"
        df["balance_warning"] = "Balances reconstructed from latest account balance and Plaid-signed transactions."
        return df

    df["balance_source"] = "unavailable"
    df["balance_confidence"] = "low"
    df["balance_warning"] = "No reliable balance history was supplied; balance metrics may use cashflow estimates."
    return df


def transactions_to_dataframe(txns: List[Dict[str, Any]], accounts: List[Dict[str, Any]] | None = None) -> pd.DataFrame:
    """Normalize transactions into the DataFrame shape used by MCAV2 analysis."""
    df = pd.json_normalize(txns)
    if df.empty:
        return df
    if "name" not in df.columns and "description" in df.columns:
        df["name"] = df["description"]
    if "name_y" not in df.columns:
        df["name_y"] = df.get("name", "Unknown Transaction")
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
    df["amount"] = pd.to_numeric(df.get("amount"), errors="coerce")
    if "amount_original" not in df.columns:
        df["amount_original"] = df["amount"]
    df = df.dropna(subset=["date", "amount"]).copy()
    return add_balance_provenance(df, accounts)

