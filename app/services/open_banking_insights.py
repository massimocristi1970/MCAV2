"""Derived open-banking insights for underwriting review.

These fields are intentionally analysis/export signals. They do not alter the
score unless a scoring module explicitly consumes them.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def _first_existing(df: pd.DataFrame, columns: list[str], default: str = "") -> pd.Series:
    values = pd.Series([default] * len(df), index=df.index, dtype="object")
    for column in columns:
        if column in df.columns:
            candidate = df[column].fillna("").astype(str).str.strip()
            values = values.where(values.astype(str).str.len() > 0, candidate)
    return values.fillna(default)


def _safe_numeric(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def derive_open_banking_insights(data: pd.DataFrame, requested_loan: float | None = None) -> Dict[str, Any]:
    """Derive review-friendly transaction signals from categorized bank data."""
    if data is None or data.empty:
        return {
            "Open Banking Insight Layer": "Not available",
            "Open Banking Insights Used In Score": "No - analysis/export only",
        }

    df = data.copy()
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    if df.empty:
        return {
            "Open Banking Insight Layer": "No valid transaction dates",
            "Open Banking Insights Used In Score": "No - analysis/export only",
        }

    amount_source = df["signed_amount"] if "signed_amount" in df.columns else df.get("amount")
    df["signed_amount"] = _safe_numeric(amount_source)
    df["abs_amount"] = df["signed_amount"].abs()
    df["narrative"] = _first_existing(df, ["name", "name_y", "transaction_name", "merchant_name"])
    df["year_month"] = df["date"].dt.to_period("M")
    months_count = max(int(df["year_month"].nunique()), 1)
    history_days = max(int((df["date"].max() - df["date"].min()).days) + 1, 1)

    for flag in [
        "is_revenue",
        "is_special_inflow",
        "is_transfer_in",
        "is_transfer_out",
        "is_funding_injection",
        "is_debt",
        "is_debt_repayment",
        "is_failed_payment",
        "is_bank_charge",
    ]:
        if flag not in df.columns:
            df[flag] = False
        df[flag] = df[flag].fillna(False).astype(bool)

    credits = df["signed_amount"] < 0
    debits = df["signed_amount"] > 0
    revenue = df[df["is_revenue"]].copy()
    debt_repayments = df[df["is_debt_repayment"]].copy()
    loan_credits = df[df["is_debt"]].copy()
    non_revenue_inflows = df[credits & ~df["is_revenue"]].copy()

    total_revenue = float(revenue["abs_amount"].sum())
    total_credits = float(df.loc[credits, "abs_amount"].sum())
    total_non_revenue_inflows = float(non_revenue_inflows["abs_amount"].sum())
    total_debt_repayments = float(debt_repayments["abs_amount"].sum())
    total_loan_credits = float(loan_credits["abs_amount"].sum())

    monthly_revenue = revenue.groupby("year_month")["abs_amount"].sum()
    weakest_month_revenue = float(monthly_revenue.min()) if not monthly_revenue.empty else 0.0
    strongest_month_revenue = float(monthly_revenue.max()) if not monthly_revenue.empty else 0.0
    revenue_active_days = int(revenue["date"].dt.date.nunique()) if not revenue.empty else 0

    if not revenue.empty:
        source_totals = revenue.groupby(revenue["narrative"].str.lower().str[:80])["abs_amount"].sum().sort_values(ascending=False)
        top_source_pct = _ratio(float(source_totals.iloc[0]), total_revenue) if len(source_totals) else 0.0
        source_count = int(len(source_totals))
        processor_mask = revenue["narrative"].str.contains(
            r"stripe|sumup|zettle|square|worldpay|paypal|shopify|take payments|barclaycard|elavon|adyen|teya|fresha|treatwell",
            case=False,
            na=False,
        )
        card_processor_revenue = float(revenue.loc[processor_mask, "abs_amount"].sum())
    else:
        top_source_pct = 0.0
        source_count = 0
        card_processor_revenue = 0.0

    if "balances.available" in df.columns and not df["balances.available"].isna().all():
        balances = pd.to_numeric(df["balances.available"], errors="coerce")
        daily_balance = pd.Series(balances.values, index=df["date"]).sort_index().resample("D").last().ffill().dropna()
    else:
        daily_balance = pd.Series(dtype="float64")

    if not daily_balance.empty:
        avg_daily_balance = float(daily_balance.mean())
        min_daily_balance = float(daily_balance.min())
        low_balance_days_100 = int((daily_balance < 100).sum())
        low_balance_days_500 = int((daily_balance < 500).sum())
        low_balance_days_1000 = int((daily_balance < 1000).sum())
        negative_days = int((daily_balance < 0).sum())
    else:
        avg_daily_balance = 0.0
        min_daily_balance = 0.0
        low_balance_days_100 = 0
        low_balance_days_500 = 0
        low_balance_days_1000 = 0
        negative_days = 0

    recent_cutoff = df["date"].max() - pd.Timedelta(days=30)
    recent_loan_credits = float(loan_credits.loc[loan_credits["date"] >= recent_cutoff, "abs_amount"].sum()) if not loan_credits.empty else 0.0
    recent_failed_payments = int(df.loc[(df["date"] >= recent_cutoff) & df["is_failed_payment"]].shape[0])

    requested_loan_value = float(requested_loan or 0)
    avg_monthly_revenue = total_revenue / months_count if months_count else 0.0

    return {
        "Open Banking Insight Layer": "Available",
        "Open Banking Insights Used In Score": "No - analysis/export only",
        "OB Transaction Count": int(len(df)),
        "OB History Days": history_days,
        "OB History Months": months_count,
        "OB True Revenue": round(total_revenue, 2),
        "OB Non-Revenue Inflows": round(total_non_revenue_inflows, 2),
        "OB Non-Revenue Inflow Ratio": round(_ratio(total_non_revenue_inflows, total_credits), 3),
        "OB Revenue Active Day Rate": round(_ratio(revenue_active_days, history_days), 3),
        "OB Revenue Source Count": source_count,
        "OB Top Revenue Source Percentage": round(top_source_pct * 100, 1),
        "OB Card Processor Revenue Share": round(_ratio(card_processor_revenue, total_revenue), 3),
        "OB Weakest Month Revenue": round(weakest_month_revenue, 2),
        "OB Strongest Month Revenue": round(strongest_month_revenue, 2),
        "OB Debt Repayment Burden": round(_ratio(total_debt_repayments, total_revenue), 3),
        "OB Monthly Debt Repayment Estimate": round(total_debt_repayments / months_count, 2),
        "OB Recent Loan Credits 30D": round(recent_loan_credits, 2),
        "OB Loan Credits Total": round(total_loan_credits, 2),
        "OB Recent Failed Payments 30D": recent_failed_payments,
        "OB Avg Daily Balance": round(avg_daily_balance, 2),
        "OB Min Daily Balance": round(min_daily_balance, 2),
        "OB Low Balance Days <100": low_balance_days_100,
        "OB Low Balance Days <500": low_balance_days_500,
        "OB Low Balance Days <1000": low_balance_days_1000,
        "OB Negative Balance Days": negative_days,
        "OB Requested Loan To Monthly Revenue": round(_ratio(requested_loan_value, avg_monthly_revenue), 2),
        "OB Requested Loan To Weakest Month Revenue": round(_ratio(requested_loan_value, weakest_month_revenue), 2),
    }
