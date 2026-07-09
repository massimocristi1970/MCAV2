"""Loans and debt repayment analysis (shared by dashboard UI and exports)."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from app.services.business_risk_signals import categorize_business_transactions


def analyze_loans_and_repayments(df: pd.DataFrame) -> Dict[str, Any]:
    """Comprehensive analysis of loans received and debt repayments."""
    if df.empty:
        return {}

    categorized_data = categorize_business_transactions(df.copy())

    loans_data = categorized_data[categorized_data["subcategory"] == "Loans"].copy()
    repayments_data = categorized_data[categorized_data["subcategory"] == "Debt Repayments"].copy()

    for data in [loans_data, repayments_data]:
        if not data.empty:
            data["date"] = pd.to_datetime(data["date"])
            data["month"] = data["date"].dt.to_period("M")
            data["amount_abs"] = abs(data["amount"])

    analysis: Dict[str, Any] = {}

    if not loans_data.empty:
        analysis["total_loans_received"] = loans_data["amount_abs"].sum()
        analysis["loan_count"] = len(loans_data)
        analysis["avg_loan_amount"] = loans_data["amount_abs"].mean()
        analysis["largest_loan"] = loans_data["amount_abs"].max()
        analysis["smallest_loan"] = loans_data["amount_abs"].min()
        analysis["loans_by_month"] = loans_data.groupby("month")["amount_abs"].agg(["count", "sum"]).reset_index()
        analysis["loans_by_month"]["month_str"] = analysis["loans_by_month"]["month"].astype(str)
        loans_data["lender_clean"] = loans_data["name"].str.lower().str.strip()
        analysis["loans_by_lender"] = (
            loans_data.groupby("lender_clean")["amount_abs"].agg(["count", "sum"]).reset_index().sort_values("sum", ascending=False)
        )
    else:
        analysis.update(
            {
                "total_loans_received": 0,
                "loan_count": 0,
                "avg_loan_amount": 0,
                "largest_loan": 0,
                "smallest_loan": 0,
                "loans_by_month": pd.DataFrame(),
                "loans_by_lender": pd.DataFrame(),
            }
        )

    if not repayments_data.empty:
        analysis["total_repayments_made"] = repayments_data["amount_abs"].sum()
        analysis["repayment_count"] = len(repayments_data)
        analysis["avg_repayment_amount"] = repayments_data["amount_abs"].mean()
        analysis["largest_repayment"] = repayments_data["amount_abs"].max()
        analysis["smallest_repayment"] = repayments_data["amount_abs"].min()
        analysis["repayments_by_month"] = repayments_data.groupby("month")["amount_abs"].agg(["count", "sum"]).reset_index()
        analysis["repayments_by_month"]["month_str"] = analysis["repayments_by_month"]["month"].astype(str)
        repayments_data["recipient_clean"] = repayments_data["name"].str.lower().str.strip()
        analysis["repayments_by_recipient"] = (
            repayments_data.groupby("recipient_clean")["amount_abs"]
            .agg(["count", "sum"])
            .reset_index()
            .sort_values("sum", ascending=False)
        )
    else:
        analysis.update(
            {
                "total_repayments_made": 0,
                "repayment_count": 0,
                "avg_repayment_amount": 0,
                "largest_repayment": 0,
                "smallest_repayment": 0,
                "repayments_by_month": pd.DataFrame(),
                "repayments_by_recipient": pd.DataFrame(),
            }
        )

    analysis["net_borrowing"] = analysis["total_loans_received"] - analysis["total_repayments_made"]
    analysis["repayment_ratio"] = (
        analysis["total_repayments_made"] / analysis["total_loans_received"]
        if analysis["total_loans_received"] > 0
        else None
    )
    analysis["repayments_without_visible_loan"] = bool(analysis["loan_count"] == 0 and analysis["repayment_count"] > 0)

    if not analysis["repayments_by_recipient"].empty:
        possible_lenders = analysis["repayments_by_recipient"].copy()
        possible_lenders["possible_lender"] = possible_lenders["recipient_clean"].str.title()
        visible_lenders = set()
        if not analysis["loans_by_lender"].empty:
            visible_lenders = set(analysis["loans_by_lender"]["lender_clean"].astype(str).str.lower().str.strip())
        possible_lenders["loan_credit_seen"] = possible_lenders["recipient_clean"].isin(visible_lenders)
        possible_lenders["review_reason"] = possible_lenders["loan_credit_seen"].map(
            {
                True: "Repayment recipient with visible loan credit",
                False: "Repayments found but no matching loan credit in selected bank data",
            }
        )
        possible_lenders = possible_lenders.rename(columns={"count": "repayment_count", "sum": "total_repaid_in_period"})
        analysis["possible_lenders_from_repayments"] = possible_lenders[
            [
                "possible_lender",
                "recipient_clean",
                "repayment_count",
                "total_repaid_in_period",
                "loan_credit_seen",
                "review_reason",
            ]
        ]
    else:
        analysis["possible_lenders_from_repayments"] = pd.DataFrame()

    if not loans_data.empty or not repayments_data.empty:
        all_months = set()
        if not loans_data.empty:
            all_months.update(loans_data["month"].unique())
        if not repayments_data.empty:
            all_months.update(repayments_data["month"].unique())

        monthly_net = []
        for month in sorted(all_months):
            month_loans = loans_data[loans_data["month"] == month]["amount_abs"].sum() if not loans_data.empty else 0
            month_repayments = (
                repayments_data[repayments_data["month"] == month]["amount_abs"].sum() if not repayments_data.empty else 0
            )
            monthly_net.append(
                {
                    "month": month,
                    "month_str": str(month),
                    "loans": month_loans,
                    "repayments": month_repayments,
                    "net_borrowing": month_loans - month_repayments,
                }
            )
        analysis["monthly_net_borrowing"] = pd.DataFrame(monthly_net)
    else:
        analysis["monthly_net_borrowing"] = pd.DataFrame()

    return analysis
