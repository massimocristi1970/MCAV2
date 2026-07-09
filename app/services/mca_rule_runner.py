"""Run MCA rule scoring on transaction lists with optional period filtering."""

from __future__ import annotations

from typing import Any

import pandas as pd

from build_training_dataset import build_mca_features
from mca_scorecard_rules import Thresholds, decide_application


def filter_transactions_by_period(transactions: list[dict], period_months: str | int) -> list[dict]:
    """Filter raw transaction dicts to match the dashboard analysis period."""
    if not transactions or period_months == "All":
        return list(transactions)

    dated: list[tuple[pd.Timestamp, dict]] = []
    for txn in transactions:
        dt_raw = txn.get("date") or txn.get("authorized_date")
        if not dt_raw:
            continue
        try:
            dt = pd.to_datetime(dt_raw)
        except Exception:
            continue
        dated.append((dt, txn))

    if not dated:
        return []

    latest = max(dt for dt, _ in dated)
    start = latest - pd.DateOffset(months=int(period_months))
    return [txn for dt, txn in dated if dt >= start]


def dataframe_to_mca_transactions(df: pd.DataFrame) -> list[dict]:
    """Convert a transaction DataFrame back into MCA feature-builder input rows."""
    if df is None or df.empty:
        return []

    records: list[dict] = []
    for _, row in df.iterrows():
        date_val = row.get("date")
        if hasattr(date_val, "isoformat"):
            date_val = date_val.isoformat()
        records.append(
            {
                "date": date_val,
                "amount": row.get("amount"),
                "name": row.get("name", ""),
                "merchant_name": row.get("merchant_name", ""),
                "transaction_type": row.get("transaction_type", ""),
                "personal_finance_category": row.get("personal_finance_category"),
            }
        )
    return records


def run_mca_rule_scoring(transactions: list[dict], analysis_period: str | int) -> dict[str, Any]:
    """Score MCA rule features for the selected analysis period."""
    scoped = filter_transactions_by_period(transactions, analysis_period)
    features = build_mca_features(scoped)
    if not features:
        decision, score, reasons = "REFER", 0, ["No usable transactions in analysis period"]
    else:
        decision, score, reasons = decide_application(features, t=Thresholds())

    return {
        "mca_rule_decision": decision,
        "mca_rule_score": score,
        "mca_rule_reasons": reasons,
        "mca_rule_signals": {
            "inflow_days_30d": features.get("inflow_days_30d"),
            "max_inflow_gap_days": features.get("max_inflow_gap_days"),
            "inflow_cv": features.get("inflow_cv"),
            "months_covered": features.get("months_covered"),
            "txn_count_avg_month": features.get("txn_count_avg_month"),
        },
    }


def apply_mca_rule_to_params(params: dict[str, Any], transactions: list[dict], analysis_period: str | int) -> dict[str, Any]:
    """Update params in-place with MCA rule outputs and return params."""
    params.update(run_mca_rule_scoring(transactions, analysis_period))
    return params
