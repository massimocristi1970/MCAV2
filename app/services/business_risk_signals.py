"""Business-account transaction signals used by scoring and validation."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from .advanced_metrics import calculate_advanced_metrics
from .data_processor import TransactionCategorizer
from .open_banking_insights import derive_open_banking_insights


_CATEGORIZER = TransactionCategorizer()


def _coalesce_text_columns(df: pd.DataFrame) -> pd.Series:
    text_columns = [
        "name_y",
        "transaction_name",
        "name",
        "merchant_name",
        "account_name",
    ]
    text = pd.Series([""] * len(df), index=df.index, dtype="object")
    for column in text_columns:
        if column in df.columns:
            values = df[column].fillna("").astype(str).str.strip()
            text = text.where(text != "", values)
    return text.fillna("")


def categorize_business_transactions(data: pd.DataFrame) -> pd.DataFrame:
    """Apply business-focused transaction categories and boolean flags."""
    if data.empty:
        return data.copy()

    df = data.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    amount_source = df["amount_original"] if "amount_original" in df.columns else df["amount"]
    df["signed_amount"] = pd.to_numeric(amount_source, errors="coerce").fillna(0.0)
    df["amount"] = df["signed_amount"]
    df["name"] = _coalesce_text_columns(df)
    if "name_y" not in df.columns:
        df["name_y"] = df["name"]

    categories = []
    confidences = []
    for _, row in df.iterrows():
        payload = row.to_dict()
        payload["amount_original"] = row["signed_amount"]
        category, confidence = _CATEGORIZER.categorize_transaction(payload)
        categories.append(category)
        confidences.append(confidence)

    df["subcategory"] = categories
    df["categorization_confidence"] = confidences
    df["abs_amount"] = df["signed_amount"].abs()

    df["is_revenue"] = df["subcategory"].eq("Income")
    df["is_special_inflow"] = df["subcategory"].eq("Special Inflow")
    df["is_expense"] = df["subcategory"].isin(["Expenses", "Special Outflow", "Bank Charge"])
    df["is_debt_repayment"] = df["subcategory"].eq("Debt Repayments")
    df["is_debt"] = df["subcategory"].eq("Loans")
    df["is_failed_payment"] = df["subcategory"].eq("Failed Payment")
    df["is_transfer_in"] = df["subcategory"].eq("Transfer In")
    df["is_transfer_out"] = df["subcategory"].eq("Transfer Out")
    df["is_internal_transfer"] = df["is_transfer_in"] | df["is_transfer_out"]
    df["is_funding_injection"] = df["subcategory"].eq("Funding Inflow")
    df["is_bank_charge"] = df["subcategory"].eq("Bank Charge")
    df["is_non_revenue_inflow"] = (
        df["is_special_inflow"] | df["is_transfer_in"] | df["is_funding_injection"] | df["is_debt"]
    )

    return df


def calculate_business_metrics(data: pd.DataFrame, company_age_months: int | float | None = None) -> Dict[str, Any]:
    """Calculate core MCA metrics plus business-account behavioral signals."""
    if data.empty:
        return {}

    df = categorize_business_transactions(data)
    if df["date"].isna().all():
        return {}

    df = df.dropna(subset=["date"]).copy()
    df["year_month"] = df["date"].dt.to_period("M")
    months_count = max(df["year_month"].nunique(), 1)

    total_revenue = float(df.loc[df["is_revenue"], "abs_amount"].sum())
    total_expenses = float(df.loc[df["is_expense"], "abs_amount"].sum())
    total_debt_repayments = float(df.loc[df["is_debt_repayment"], "abs_amount"].sum())
    total_debt = float(df.loc[df["is_debt"], "abs_amount"].sum())
    funding_inflows = float(df.loc[df["is_funding_injection"], "abs_amount"].sum())
    special_inflows = float(df.loc[df["is_special_inflow"], "abs_amount"].sum())
    transfer_inflows = float(df.loc[df["is_transfer_in"], "abs_amount"].sum())
    transfer_outflows = float(df.loc[df["is_transfer_out"], "abs_amount"].sum())
    bank_charge_amount = float(df.loc[df["is_bank_charge"], "abs_amount"].sum())
    bank_charge_count = int(df["is_bank_charge"].sum())
    failed_payment_count = int(df["is_failed_payment"].sum())

    gross_inflows = float(df.loc[df["signed_amount"] < 0, "abs_amount"].sum())
    gross_outflows = float(df.loc[df["signed_amount"] > 0, "abs_amount"].sum())
    net_income = total_revenue - total_expenses
    monthly_avg_revenue = total_revenue / months_count if months_count else 0.0
    revenue_for_ratios = max(total_revenue, 1.0)

    debt_to_income_ratio = min(total_debt / revenue_for_ratios, 10.0)
    expense_to_revenue_ratio = total_expenses / revenue_for_ratios
    operating_margin = max(-1.0, min(1.0, net_income / revenue_for_ratios))

    if total_debt_repayments > 0:
        debt_service_coverage_ratio = total_revenue / total_debt_repayments
        dscr_repayments_observed = True
    elif total_debt > 0:
        estimated_annual_payment = total_debt * 0.1
        debt_service_coverage_ratio = total_revenue / estimated_annual_payment if estimated_annual_payment > 0 else 0.0
        dscr_repayments_observed = False
    else:
        debt_service_coverage_ratio = 1.15
        dscr_repayments_observed = False
    debt_service_coverage_ratio = min(float(debt_service_coverage_ratio), 50.0)

    monthly_summary = pd.DataFrame(
        [
            {
                "year_month": year_month,
                "monthly_revenue": float(group.loc[group["is_revenue"], "abs_amount"].sum()),
                "monthly_expenses": float(group.loc[group["is_expense"], "abs_amount"].sum()),
                "monthly_failed_payments": int(group["is_failed_payment"].sum()),
                "monthly_bank_charges": float(group.loc[group["is_bank_charge"], "abs_amount"].sum()),
            }
            for year_month, group in df.groupby("year_month")
        ]
    ).set_index("year_month").round(2)

    if len(monthly_summary) > 1:
        revenue_values = monthly_summary["monthly_revenue"]
        revenue_mean = revenue_values.mean()
        cash_flow_volatility = min(float(revenue_values.std() / revenue_mean), 2.0) if revenue_mean > 0 else 0.5
        growth_changes = revenue_values.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        revenue_growth_rate = float(growth_changes.median()) if len(growth_changes) > 0 else 0.0
        revenue_growth_rate = max(-0.5, min(0.5, revenue_growth_rate))
        gross_burn_rate = float(monthly_summary["monthly_expenses"].mean())
    else:
        cash_flow_volatility = 0.1
        revenue_growth_rate = 0.0
        gross_burn_rate = total_expenses / months_count if months_count else total_expenses

    balance_source = str(df.get("balance_source", pd.Series(["estimated"])).dropna().iloc[0]) if "balance_source" in df.columns and not df["balance_source"].dropna().empty else "estimated"
    balance_confidence = str(df.get("balance_confidence", pd.Series(["low"])).dropna().iloc[0]) if "balance_confidence" in df.columns and not df["balance_confidence"].dropna().empty else "low"
    balance_warning = str(df.get("balance_warning", pd.Series([""])).dropna().iloc[0]) if "balance_warning" in df.columns and not df["balance_warning"].dropna().empty else ""

    balance_column = None
    if "balances.available" in df.columns and pd.to_numeric(df["balances.available"], errors="coerce").notna().any():
        balance_column = "balances.available"
        balance_source = "provided"
        if balance_confidence not in {"high", "medium"}:
            balance_confidence = "high"
    elif "balances.current" in df.columns and pd.to_numeric(df["balances.current"], errors="coerce").notna().any():
        balance_column = "balances.current"
        balance_source = "provided"
        if balance_confidence not in {"high", "medium"}:
            balance_confidence = "high"
    elif "calculated_balance" in df.columns and pd.to_numeric(df["calculated_balance"], errors="coerce").notna().any():
        balance_column = "calculated_balance"
        balance_source = "reconstructed"
        if balance_confidence not in {"medium", "low"}:
            balance_confidence = "medium"

    if balance_column:
        balances = pd.to_numeric(df[balance_column], errors="coerce")
        avg_month_end_balance = float(balances.dropna().mean())
        daily_balance = pd.Series(balances.values, index=df["date"]).sort_index().resample("D").last().ffill().dropna()
        avg_negative_days = int(round((daily_balance < 0).sum() / months_count)) if len(daily_balance) > 0 else 0
    else:
        balance_source = "estimated"
        balance_confidence = "low"
        balance_warning = balance_warning or "No reliable balance history was supplied; balance metrics are estimated from cashflow."
        monthly_net = (total_revenue - total_expenses) / months_count if months_count else 0.0
        avg_month_end_balance = float(max(500.0, monthly_net * 0.35))
        if cash_flow_volatility > 0.3:
            avg_negative_days = min(10, int(cash_flow_volatility * 10))
        elif operating_margin < 0:
            avg_negative_days = 3
        else:
            avg_negative_days = 0

    advanced_metrics = calculate_advanced_metrics(df)
    debt_repayment_recipients = []
    possible_lenders_from_repayments = []
    if total_debt_repayments > 0 and "name" in df.columns:
        repayment_recipients = (
            df.loc[df["is_debt_repayment"]]
            .assign(recipient_clean=lambda x: x["name"].fillna("").astype(str).str.lower().str.strip())
            .groupby("recipient_clean")["abs_amount"]
            .agg(["count", "sum"])
            .reset_index()
            .sort_values("sum", ascending=False)
        )
        visible_lenders = set(
            df.loc[df["is_debt"], "name"].fillna("").astype(str).str.lower().str.strip().unique()
        )
        for row in repayment_recipients.head(10).to_dict("records"):
            lender_name = str(row["recipient_clean"])
            debt_repayment_recipients.append(lender_name.title())
            if lender_name not in visible_lenders:
                possible_lenders_from_repayments.append(
                    {
                        "possible_lender": lender_name.title(),
                        "repayment_count": int(row["count"]),
                        "total_repaid_in_period": round(float(row["sum"]), 2),
                        "review_reason": "Repayments found but no matching loan credit in selected bank data",
                    }
                )

    revenue_plus_funding = total_revenue + funding_inflows
    total_external_funding = funding_inflows + total_debt
    gross_cash_movement = max(gross_inflows + gross_outflows, 1.0)

    metrics: Dict[str, Any] = {
        "Total Revenue": round(total_revenue, 2),
        "Monthly Average Revenue": round(monthly_avg_revenue, 2),
        "Total Expenses": round(total_expenses, 2),
        "Net Income": round(net_income, 2),
        "Total Debt Repayments": round(total_debt_repayments, 2),
        "Total Debt": round(total_debt, 2),
        "Potential Lenders From Repayments": possible_lenders_from_repayments,
        "Potential Lenders From Repayments Count": len(possible_lenders_from_repayments),
        "Debt Repayment Recipients": debt_repayment_recipients,
        "Repayments Without Visible Loan": bool(total_debt <= 0 and total_debt_repayments > 0),
        "Debt-to-Income Ratio": round(debt_to_income_ratio, 3),
        "Expense-to-Revenue Ratio": round(expense_to_revenue_ratio, 3),
        "Operating Margin": round(operating_margin, 3),
        "Debt Service Coverage Ratio": round(debt_service_coverage_ratio, 2),
        "DSCR Repayments Observed": dscr_repayments_observed,
        "Gross Burn Rate": round(gross_burn_rate, 2),
        "Cash Flow Volatility": round(cash_flow_volatility, 3),
        "Revenue Growth Rate": round(revenue_growth_rate, 2),
        "Average Month-End Balance": round(avg_month_end_balance, 2),
        "Average Negative Balance Days per Month": avg_negative_days,
        "Balance Source": balance_source,
        "Balance Confidence": balance_confidence,
        "Balance Warning": balance_warning,
        "Number of Bounced Payments": failed_payment_count,
        "Funding Inflow Total": round(funding_inflows, 2),
        "Funding Reliance Ratio": round(funding_inflows / max(revenue_plus_funding, 1.0), 3),
        "External Funding Reliance Ratio": round(total_external_funding / max(total_revenue + total_external_funding, 1.0), 3),
        "Internal Transfer Inflow Total": round(transfer_inflows, 2),
        "Internal Transfer Outflow Total": round(transfer_outflows, 2),
        "Internal Transfer Activity Ratio": round((transfer_inflows + transfer_outflows) / gross_cash_movement, 3),
        "Special Inflow Total": round(special_inflows, 2),
        "Bank Charge Count": bank_charge_count,
        "Bank Charge Amount": round(bank_charge_amount, 2),
        "Bank Charge Burden": round(bank_charge_amount / revenue_for_ratios, 4),
        "monthly_summary": monthly_summary,
    }

    metrics.update(
        {
            "Revenue Regularity Score": advanced_metrics.get("transaction_regularity", 0),
            "Deposit Frequency Score": advanced_metrics.get("deposit_frequency_score", 0),
            "Average Inflow Gap Days": round(float(advanced_metrics.get("avg_inflow_gap_days", 0) or 0), 1),
            "Max Inflow Gap Days": round(float(advanced_metrics.get("max_inflow_gap_days", 0) or 0), 1),
            "Revenue Source Count": int(advanced_metrics.get("unique_revenue_sources", 0) or 0),
            "Top Revenue Source Percentage": round(float(advanced_metrics.get("top_source_percentage", 0) or 0), 1),
            "Revenue Concentration Ratio": round(float(advanced_metrics.get("revenue_concentration_ratio", 0) or 0), 3),
            "Revenue Concentration Risk": advanced_metrics.get("concentration_risk", "Unknown"),
            "NSF Count 90D": int(advanced_metrics.get("nsf_count_90d", 0) or 0),
            "Days Since Last NSF": advanced_metrics.get("days_since_last_nsf"),
            "Days Since Last Negative Balance": advanced_metrics.get("days_since_last_negative"),
            "Balance Trend": advanced_metrics.get("balance_trend"),
            "Balance Improving": advanced_metrics.get("balance_improving"),
            "Active Lenders Detected": int(advanced_metrics.get("active_lenders_detected", 0) or 0),
            "Debt Stacking Risk": advanced_metrics.get("debt_stacking_risk", "Unknown"),
            "Monthly Debt Obligations": round(float(advanced_metrics.get("monthly_debt_obligations", 0) or 0), 2),
            "Advanced Risk Score": round(float(advanced_metrics.get("advanced_risk_score", 0) or 0), 1),
        }
    )

    metrics.update(derive_open_banking_insights(df))

    if company_age_months is not None:
        metrics["Company Age (Months)"] = int(company_age_months)

    return metrics

