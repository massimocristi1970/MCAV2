from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.services.business_risk_signals import categorize_business_transactions


REVENUE_CANDIDATE_RE = re.compile(
    r"\b("
    r"stripe|sumup|zettle|izettle|square|worldpay|paypal|shopify|barclaycard|"
    r"elavon|adyen|take\s*payments|paymentsense|dojo|valitor|paypoint|mypos|"
    r"teya|fresha|treatwell|just\s*eat|deliveroo|ubereats|uber|bolt|"
    r"invoice|customer|sales|takings|settlement|payout|card\s+payment"
    r")\b",
    re.IGNORECASE,
)

LENDER_CANDIDATE_RE = re.compile(
    r"\b("
    r"iwoca|capify|funding\s*circle|liberis|you\s*lend|youlend|\byl\b|"
    r"fleximize|marketfinance|merchant\s*money|capital\s*on\s*tap|kriya|"
    r"uncapped|cubefunder|swishfund|loan|lending|finance|funding|advance|"
    r"disbursement|repay|repayment|instalment|installment|debt"
    r")\b",
    re.IGNORECASE,
)

NON_REVENUE_INCOME_WARNING_RE = re.compile(
    r"\b("
    r"transfer|trf|savings|refund|rebate|cashback|director|shareholder|"
    r"capital\s+introduced|capital\s+injection|loan|funding|fnd|advance|"
    r"disbursement|hmrc|tax"
    r")\b",
    re.IGNORECASE,
)

REQUIRED_TRANSACTION_FIELDS = ["date", "amount", "name"]
USEFUL_PLAID_FIELDS = [
    "personal_finance_category.detailed",
    "personal_finance_category.primary",
    "personal_finance_category.confidence_level",
    "merchant_name",
    "transaction_type",
]


def _load_json_from_zip(zf: zipfile.ZipFile, name: str) -> Any:
    with zf.open(name) as fh:
        return json.loads(fh.read().decode("utf-8-sig"))


def _transactions_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        txns = payload.get("transactions")
        if isinstance(txns, list):
            return [row for row in txns if isinstance(row, dict)]
        if any(key in payload for key in ("transaction_id", "amount", "date")):
            return [payload]
    return []


def _coalesce_name(row: pd.Series) -> str:
    for column in ("name", "name_y", "transaction_name", "merchant_name"):
        value = row.get(column)
        if pd.notna(value) and str(value).strip():
            return str(value).strip()
    return ""


def _normalize_transactions(transactions: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.json_normalize(transactions)
    if df.empty:
        return df

    if "amount" not in df.columns:
        df["amount"] = 0
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["amount_original"] = df["amount"]

    if "date" not in df.columns:
        df["date"] = pd.NaT
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "name_y" not in df.columns:
        df["name_y"] = df.apply(_coalesce_name, axis=1)
    if "name" not in df.columns:
        df["name"] = df["name_y"]

    for column in USEFUL_PLAID_FIELDS:
        if column not in df.columns:
            df[column] = ""

    return df


def _safe_str(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def _exception_rows(file_name: str, categorized: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if categorized.empty:
        return rows

    for _, row in categorized.iterrows():
        narrative = _coalesce_name(row)
        amount = float(row.get("signed_amount", row.get("amount", 0)) or 0)
        abs_amount = abs(amount)
        subcategory = _safe_str(row.get("subcategory"))
        plaid = _safe_str(row.get("personal_finance_category.detailed"))
        is_credit = amount < 0

        reasons: list[str] = []
        if subcategory == "Uncategorised":
            reasons.append("uncategorised")
        if is_credit and not bool(row.get("is_revenue")) and REVENUE_CANDIDATE_RE.search(narrative):
            reasons.append("possible_revenue_not_counted")
        if bool(row.get("is_revenue")) and NON_REVENUE_INCOME_WARNING_RE.search(narrative):
            reasons.append("income_contains_non_revenue_keyword")
        if is_credit and subcategory == "Loans" and abs_amount <= 100:
            reasons.append("small_credit_marked_loan")
        if subcategory == "Expenses" and LENDER_CANDIDATE_RE.search(narrative):
            reasons.append("possible_debt_repayment_marked_expense")
        if is_credit and subcategory in {"Transfer In", "Special Inflow", "Funding Inflow", "Loans"} and abs_amount >= 1000:
            reasons.append("high_value_non_revenue_credit")
        if subcategory == "Failed Payment" and re.search(r"transfer", narrative, re.IGNORECASE):
            reasons.append("failed_payment_contains_transfer")

        if reasons:
            rows.append(
                {
                    "file": file_name,
                    "date": row.get("date"),
                    "name": narrative,
                    "merchant_name": row.get("merchant_name"),
                    "amount": round(amount, 2),
                    "abs_amount": round(abs_amount, 2),
                    "subcategory": subcategory,
                    "is_revenue": bool(row.get("is_revenue")),
                    "plaid_detailed": plaid,
                    "plaid_primary": row.get("personal_finance_category.primary"),
                    "exception_reasons": "; ".join(reasons),
                }
            )
    return rows


def audit_zip(zip_path: Path, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    file_summaries: list[dict[str, Any]] = []
    exception_rows: list[dict[str, Any]] = []
    pattern_counter: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()
    plaid_vs_ours: Counter[tuple[str, str]] = Counter()
    missing_field_counts: Counter[str] = Counter()
    error_rows: list[dict[str, Any]] = []

    with zipfile.ZipFile(zip_path) as zf:
        json_names = [name for name in zf.namelist() if name.lower().endswith(".json") and not name.endswith("/")]

        for name in json_names:
            try:
                payload = _load_json_from_zip(zf, name)
                transactions = _transactions_from_payload(payload)
                df = _normalize_transactions(transactions)
                if df.empty:
                    file_summaries.append({"file": name, "transaction_count": 0, "status": "no_transactions"})
                    continue

                for field in REQUIRED_TRANSACTION_FIELDS + USEFUL_PLAID_FIELDS:
                    if field not in df.columns or df[field].isna().all() or (df[field].fillna("").astype(str).str.strip() == "").all():
                        missing_field_counts[field] += 1

                categorized = categorize_business_transactions(df)
                exceptions = _exception_rows(name, categorized)
                exception_rows.extend(exceptions)

                category_counts = categorized["subcategory"].value_counts(dropna=False).to_dict()
                for category, count in category_counts.items():
                    category_counter[str(category)] += int(count)

                for _, row in categorized.iterrows():
                    ours = _safe_str(row.get("subcategory"))
                    plaid = _safe_str(row.get("personal_finance_category.detailed")).lower()
                    plaid_vs_ours[(plaid, ours)] += 1
                    if ours == "Uncategorised":
                        text = re.sub(r"\s+", " ", _coalesce_name(row).lower()).strip()
                        pattern_counter[text[:120]] += 1

                revenue_total = float(categorized.loc[categorized["is_revenue"], "abs_amount"].sum())
                non_revenue_credit_total = float(
                    categorized.loc[(categorized["signed_amount"] < 0) & ~categorized["is_revenue"], "abs_amount"].sum()
                )
                file_summaries.append(
                    {
                        "file": name,
                        "status": "ok",
                        "transaction_count": int(len(categorized)),
                        "date_min": categorized["date"].min(),
                        "date_max": categorized["date"].max(),
                        "revenue_total": round(revenue_total, 2),
                        "non_revenue_credit_total": round(non_revenue_credit_total, 2),
                        "uncategorised_count": int((categorized["subcategory"] == "Uncategorised").sum()),
                        "failed_payment_count": int((categorized["subcategory"] == "Failed Payment").sum()),
                        "loan_credit_count": int((categorized["subcategory"] == "Loans").sum()),
                        "debt_repayment_count": int((categorized["subcategory"] == "Debt Repayments").sum()),
                        "exception_count": len(exceptions),
                    }
                )
            except Exception as exc:
                error_rows.append({"file": name, "error": repr(exc)})

    file_summary_df = pd.DataFrame(file_summaries)
    exception_df = pd.DataFrame(exception_rows)
    errors_df = pd.DataFrame(error_rows)
    category_df = pd.DataFrame(
        [{"subcategory": key, "transaction_count": value} for key, value in category_counter.most_common()]
    )
    missing_df = pd.DataFrame(
        [{"field": key, "files_missing_or_blank": value} for key, value in missing_field_counts.most_common()]
    )
    uncategorised_patterns_df = pd.DataFrame(
        [{"pattern": key, "count": value} for key, value in pattern_counter.most_common(200)]
    )
    plaid_vs_ours_df = pd.DataFrame(
        [
            {"plaid_detailed": plaid, "subcategory": ours, "count": count}
            for (plaid, ours), count in plaid_vs_ours.most_common()
        ]
    )

    paths = {
        "file_summary": output_dir / "file_level_quality_summary.csv",
        "exceptions": output_dir / "categorisation_exceptions.csv",
        "category_summary": output_dir / "category_summary.csv",
        "missing_fields": output_dir / "missing_fields_summary.csv",
        "uncategorised_patterns": output_dir / "uncategorised_patterns.csv",
        "plaid_vs_ours": output_dir / "plaid_vs_ours_summary.csv",
        "errors": output_dir / "files_with_errors.csv",
        "report": output_dir / "audit_report.md",
    }

    file_summary_df.to_csv(paths["file_summary"], index=False)
    exception_df.to_csv(paths["exceptions"], index=False)
    category_df.to_csv(paths["category_summary"], index=False)
    missing_df.to_csv(paths["missing_fields"], index=False)
    uncategorised_patterns_df.to_csv(paths["uncategorised_patterns"], index=False)
    plaid_vs_ours_df.to_csv(paths["plaid_vs_ours"], index=False)
    errors_df.to_csv(paths["errors"], index=False)

    ok_files = int((file_summary_df.get("status") == "ok").sum()) if not file_summary_df.empty else 0
    total_txns = int(file_summary_df.get("transaction_count", pd.Series(dtype=int)).sum()) if not file_summary_df.empty else 0
    total_revenue = float(file_summary_df.get("revenue_total", pd.Series(dtype=float)).sum()) if not file_summary_df.empty else 0.0
    total_uncat = int(category_counter.get("Uncategorised", 0))
    total_exceptions = int(len(exception_df))

    top_exception_reasons = []
    if not exception_df.empty:
        reason_counts: Counter[str] = Counter()
        for value in exception_df["exception_reasons"].fillna(""):
            for reason in str(value).split("; "):
                if reason:
                    reason_counts[reason] += 1
        top_exception_reasons = reason_counts.most_common(10)

    report = [
        "# Open Banking Categorisation Audit",
        "",
        f"Source zip: `{zip_path}`",
        f"Created: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Summary",
        f"- JSON files found: {len(file_summaries) + len(error_rows)}",
        f"- Files processed successfully: {ok_files}",
        f"- Files with errors: {len(error_rows)}",
        f"- Transactions processed: {total_txns:,}",
        f"- Categorised revenue total: £{total_revenue:,.2f}",
        f"- Uncategorised transactions: {total_uncat:,}",
        f"- Exception rows flagged for review: {total_exceptions:,}",
        "",
        "## Top Categories",
    ]
    for row in category_df.head(15).to_dict("records"):
        report.append(f"- {row['subcategory']}: {row['transaction_count']:,}")

    report.extend(["", "## Top Exception Reasons"])
    for reason, count in top_exception_reasons:
        report.append(f"- {reason}: {count:,}")

    report.extend(["", "## Output Files"])
    for label, path in paths.items():
        if label != "report":
            report.append(f"- {label}: `{path}`")

    paths["report"].write_text("\n".join(report), encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_outputs") / "categorisation_audit")
    args = parser.parse_args()

    paths = audit_zip(args.zip_path, args.output_dir)
    print(paths["report"])


if __name__ == "__main__":
    main()
