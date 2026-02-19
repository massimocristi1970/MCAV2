"""
Build ML Training Dataset

Combines two data sources:
  1. Transaction JSON files  (one per application, in data/JsonExport/)
  2. Application spreadsheet (data/training_dataset.xlsx)

The spreadsheet must have these columns:
  application_id        - JSON filename (with or without .json)
  requested_loan        - Loan amount requested
  company_age_months    - Business age in months
  directors_score       - Director credit score (0-100)
  outcome               - 1 = repaid, 0 = defaulted (blank = unfunded)

Outputs:
  data/mca_training_dataset.csv   - MCA transaction features (for MCA Rule scoring)
  data/ml_training_dataset.csv    - ML model features (ready for train_improved_model.py)

Usage:
  python build_training_dataset.py

  Override paths with environment variables:
    TRAINING_OUTCOMES_XLSX  - Path to application spreadsheet
    TRAINING_JSON_ROOT      - Directory containing JSON transaction files
    TRAINING_OUTPUT_DIR     - Directory for output files
"""

import json
import os
import re
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

_BASE_DIR = os.environ.get('TRAINING_DATA_DIR', os.getcwd())

OUTCOMES_XLSX = os.environ.get(
    'TRAINING_OUTCOMES_XLSX',
    os.path.join(_BASE_DIR, 'data', 'training_dataset.xlsx')
)

JSON_ROOT = os.environ.get(
    'TRAINING_JSON_ROOT',
    os.path.join(_BASE_DIR, 'data', 'JsonExport')
)

_OUTPUT_DIR = os.environ.get('TRAINING_OUTPUT_DIR', os.path.join(_BASE_DIR, 'data'))
OUTPUT_MCA_CSV = os.path.join(_OUTPUT_DIR, 'mca_training_dataset.csv')
OUTPUT_ML_CSV = os.path.join(_OUTPUT_DIR, 'ml_training_dataset.csv')

# Kept for backward compat (other modules import these names)
OUTPUT_CSV = OUTPUT_MCA_CSV
OUTPUT_XLSX = os.path.join(_OUTPUT_DIR, 'mca_training_dataset.xlsx')

# High-risk industries that map to Sector_Risk = 1
HIGH_RISK_INDUSTRIES = {'Restaurants and Cafes', 'Bars and Pubs', 'Construction Firms'}


# ============================================================
# Helpers
# ============================================================
def _safe_float(x):
    try:
        if x is None:
            return np.nan
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip().replace(",", "")
        if not s:
            return np.nan
        return float(s)
    except Exception:
        return np.nan


def _parse_date(s):
    if not s:
        return None
    if isinstance(s, datetime):
        return s
    s = str(s).strip()
    for fmt in (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%fZ",
    ):
        try:
            return datetime.strptime(s[: len(fmt)], fmt)
        except Exception:
            pass
    try:
        dt = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.to_pydatetime()
    except Exception:
        return None


def _month_key(dt: datetime) -> str:
    return f"{dt.year:04d}-{dt.month:02d}"


def _normalise_filename_cell(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if not s:
        return ""
    s = s.replace("\\", "/")
    return s.split("/")[-1].strip()


def _strip_json_ext(name: str) -> str:
    return name[:-5] if name.lower().endswith(".json") else name


def _flatten_transactions(obj):
    if isinstance(obj, list):
        return [t for t in obj if isinstance(t, dict)]

    txns = []

    def walk(x):
        nonlocal txns
        if isinstance(x, dict):
            for k in ["transactions", "transaction", "data", "items", "results"]:
                if isinstance(x.get(k), list):
                    txns.extend(x[k])
                elif isinstance(x.get(k), dict):
                    vv = x[k].get("transactions")
                    if isinstance(vv, list):
                        txns.extend(vv)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(obj)

    seen, out = set(), []
    for t in txns:
        if not isinstance(t, dict):
            continue
        key = (
            str(t.get("date") or ""),
            str(t.get("amount") or ""),
            str(t.get("name") or ""),
        )
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


# ============================================================
# Spreadsheet loader
# ============================================================
def load_application_data(path_xlsx):
    """
    Load the application spreadsheet.

    Expected columns:
        application_id, requested_loan, company_age_months,
        directors_score, outcome

    Optional columns:
        industry, total_debt

    Returns:
        dict mapping normalised filename -> row dict
    """
    df = pd.read_excel(path_xlsx)

    required = ["application_id", "outcome"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Spreadsheet missing required columns: {missing}")

    df["application_id"] = df["application_id"].apply(_normalise_filename_cell)
    df["outcome"] = pd.to_numeric(df["outcome"], errors="coerce").astype("Int64")

    # Optional columns with defaults
    if "directors_score" not in df.columns:
        df["directors_score"] = 50
    if "company_age_months" not in df.columns:
        df["company_age_months"] = 12
    if "requested_loan" not in df.columns:
        df["requested_loan"] = 0
    if "total_debt" not in df.columns:
        df["total_debt"] = 0
    if "industry" not in df.columns:
        df["industry"] = "Other"

    mapping = {}
    for _, r in df.iterrows():
        row_data = r.to_dict()
        k = row_data["application_id"]
        mapping[k] = row_data
        mapping[_strip_json_ext(k)] = row_data

    return mapping


# ============================================================
# MCA Feature builder (unchanged — used by MCA Rule scoring)
# ============================================================
def build_mca_features(transactions):
    rows = []

    for t in transactions:
        amt = _safe_float(t.get("amount"))
        dt = _parse_date(t.get("date") or t.get("authorized_date"))
        name = t.get("name") or t.get("merchant_name") or ""

        pfc = t.get("personal_finance_category") or {}
        pfc_primary = str(pfc.get("primary") or "").upper() if isinstance(pfc, dict) else ""

        if dt is None or np.isnan(amt):
            continue

        rows.append(
            {
                "date": dt,
                "month": _month_key(dt),
                "amount": amt,
                "name": name,
                "pfc": pfc_primary,
            }
        )

    if not rows:
        return {}

    df = pd.DataFrame(rows).sort_values("date")

    # Direction logic
    if df["amount"].min() >= 0:
        inflow_keys = {"INCOME", "TRANSFER_IN"}
        df["inflow"] = np.where(df["pfc"].isin(inflow_keys), df["amount"], 0.0)
        df["outflow"] = np.where(~df["pfc"].isin(inflow_keys), df["amount"], 0.0)
        sign_label = "CATEGORY(primary)"
    else:
        inflow_A = df["amount"].clip(upper=0).abs()
        outflow_A = df["amount"].clip(lower=0)
        inflow_B = df["amount"].clip(lower=0)
        outflow_B = df["amount"].clip(upper=0).abs()

        use_B = inflow_B.sum() > inflow_A.sum()
        df["inflow"] = inflow_B if use_B else inflow_A
        df["outflow"] = outflow_B if use_B else outflow_A
        sign_label = "B(+inflow)" if use_B else "A(-inflow)"

    # Monthly aggregation
    m = df.groupby("month").agg(
        inflow=("inflow", "sum"),
        outflow=("outflow", "sum"),
        txn_count=("amount", "count"),
    )

    avg_in = m["inflow"].mean()
    avg_out = m["outflow"].mean()
    inflow_vol = m["inflow"].std(ddof=0)
    outflow_vol = m["outflow"].std(ddof=0)

    inflow_cv = inflow_vol / avg_in if avg_in > 0 else np.nan
    outflow_cv = outflow_vol / avg_out if avg_out > 0 else np.nan

    # Revenue-day consistency
    anchor = df["date"].max()
    inflow_days_30d = (
        df[(df["inflow"] > 0) & (df["date"] >= anchor - pd.Timedelta(days=30))]
        ["date"]
        .dt.date.nunique()
    )

    inflow_dates = df[df["inflow"] > 0]["date"].dt.normalize().drop_duplicates().sort_values()
    max_gap = inflow_dates.diff().dt.days.max() if len(inflow_dates) > 1 else np.nan

    # Running balance for Average Month-End Balance and Negative Balance Days
    df_sorted = df.sort_values("date").copy()
    df_sorted["signed"] = df_sorted["inflow"] - df_sorted["outflow"]
    df_sorted["running_balance"] = df_sorted["signed"].cumsum()

    # Month-end balances
    df_sorted["month_key"] = df_sorted["date"].dt.to_period("M")
    month_end_balances = df_sorted.groupby("month_key")["running_balance"].last()
    avg_month_end_balance = month_end_balances.mean() if len(month_end_balances) > 0 else 0

    # Negative balance days: estimate from daily resampled balance
    daily = df_sorted.set_index("date")["running_balance"].resample("D").last().ffill()
    if len(daily) > 0:
        neg_days_total = (daily < 0).sum()
        n_months = max(1, len(m))
        avg_neg_days_per_month = neg_days_total / n_months
    else:
        avg_neg_days_per_month = 0

    # Bounced payments: look for common bounced-payment indicators in names
    bounced_patterns = re.compile(
        r"(bounced|returned|unpaid|failed payment|dd return|rejected|refer to drawer)",
        re.IGNORECASE,
    )
    bounced_count = sum(1 for t in transactions if bounced_patterns.search(
        t.get("name") or t.get("merchant_name") or ""
    ))

    # Revenue growth rate from first half vs second half of monthly inflows
    if len(m) >= 4:
        mid = len(m) // 2
        first_half_avg = m["inflow"].iloc[:mid].mean()
        second_half_avg = m["inflow"].iloc[mid:].mean()
        if first_half_avg > 0:
            revenue_growth_rate = (second_half_avg - first_half_avg) / first_half_avg
        else:
            revenue_growth_rate = 0
    elif len(m) >= 2:
        if m["inflow"].iloc[0] > 0:
            revenue_growth_rate = (m["inflow"].iloc[-1] - m["inflow"].iloc[0]) / m["inflow"].iloc[0]
        else:
            revenue_growth_rate = 0
    else:
        revenue_growth_rate = 0

    return {
        # Original MCA features
        "months_covered": len(m),
        "txn_count_total": len(df),
        "txn_count_avg_month": m["txn_count"].mean(),
        "avg_monthly_inflow": avg_in,
        "avg_monthly_outflow": avg_out,
        "avg_monthly_net": avg_in - avg_out,
        "outflow_to_inflow_ratio": avg_out / avg_in if avg_in > 0 else np.nan,
        "inflow_volatility_monthly": inflow_vol,
        "outflow_volatility_monthly": outflow_vol,
        "inflow_cv": inflow_cv,
        "outflow_cv": outflow_cv,
        "inflow_days_30d": inflow_days_30d,
        "max_inflow_gap_days": max_gap,
        "first_txn_date": df["date"].min().date().isoformat(),
        "last_txn_date": df["date"].max().date().isoformat(),
        "sign_convention_used": sign_label,
        # Additional features needed by the ML model
        "avg_month_end_balance": avg_month_end_balance,
        "avg_neg_days_per_month": avg_neg_days_per_month,
        "bounced_count": bounced_count,
        "revenue_growth_rate": revenue_growth_rate,
    }


# ============================================================
# ML Feature derivation
# ============================================================
def derive_ml_features(mca_feats, app_data):
    """
    Convert MCA transaction features + spreadsheet data into the 13
    features expected by train_improved_model.py.

    Args:
        mca_feats: dict from build_mca_features()
        app_data:  dict from spreadsheet row (directors_score, etc.)

    Returns:
        dict with the 13 ML columns + outcome
    """
    months = mca_feats.get("months_covered", 1) or 1
    avg_in = mca_feats.get("avg_monthly_inflow", 0) or 0
    avg_out = mca_feats.get("avg_monthly_outflow", 0) or 0
    avg_net = mca_feats.get("avg_monthly_net", 0) or 0

    total_revenue = avg_in * months
    total_debt = _safe_float(app_data.get("total_debt", 0)) or 0

    # Operating Margin = net / inflow
    operating_margin = avg_net / avg_in if avg_in > 0 else 0

    # Debt-to-Income Ratio
    debt_to_income = total_debt / total_revenue if total_revenue > 0 else 0

    # DSCR: if we know debt, estimate monthly repayment over 12 months
    if total_debt > 0:
        monthly_debt_payment = total_debt / 12
        dscr = avg_net / monthly_debt_payment if monthly_debt_payment > 0 else 10
    else:
        dscr = 10  # No debt = excellent coverage

    # Cash Flow Volatility from inflow CV
    cash_flow_volatility = mca_feats.get("inflow_cv", 0.5) or 0.5

    # Industry → Sector_Risk
    industry = str(app_data.get("industry", "Other"))
    sector_risk = 1 if industry in HIGH_RISK_INDUSTRIES else 0

    return {
        "Directors Score": _safe_float(app_data.get("directors_score", 50)) or 50,
        "Total Revenue": total_revenue,
        "Total Debt": total_debt,
        "Debt-to-Income Ratio": debt_to_income,
        "Operating Margin": operating_margin,
        "Debt Service Coverage Ratio": dscr,
        "Cash Flow Volatility": cash_flow_volatility,
        "Revenue Growth Rate": mca_feats.get("revenue_growth_rate", 0) or 0,
        "Average Month-End Balance": mca_feats.get("avg_month_end_balance", 0) or 0,
        "Average Negative Balance Days per Month": mca_feats.get("avg_neg_days_per_month", 0) or 0,
        "Number of Bounced Payments": mca_feats.get("bounced_count", 0) or 0,
        "Company Age (Months)": _safe_float(app_data.get("company_age_months", 12)) or 12,
        "Sector_Risk": sector_risk,
    }


# ============================================================
# IO / Main
# ============================================================
def iter_json_files(root):
    yield from Path(root).rglob("*.json")


def main():
    print(f"Loading spreadsheet: {OUTCOMES_XLSX}")
    app_data_map = load_application_data(OUTCOMES_XLSX)
    print(f"  Found {len(app_data_map) // 2} applications in spreadsheet")

    print(f"Scanning JSON files: {JSON_ROOT}")
    mca_rows = []
    ml_rows = []
    skipped = 0

    for fp in iter_json_files(JSON_ROOT):
        key = fp.stem
        app_data = app_data_map.get(fp.name) or app_data_map.get(key)
        if app_data is None:
            skipped += 1
            continue

        outcome = app_data.get("outcome")
        if pd.isna(outcome):
            skipped += 1
            continue

        with open(fp, "r", encoding="utf-8") as f:
            txns = _flatten_transactions(json.load(f))

        mca_feats = build_mca_features(txns)
        if not mca_feats:
            print(f"  WARNING: No transactions extracted from {fp.name}")
            skipped += 1
            continue

        # MCA dataset row (original format)
        mca_row = dict(mca_feats)
        mca_row.update({
            "application_file": fp.name,
            "application_key": key,
            "outcome": int(outcome),
            "txn_extracted_count": len(txns),
        })
        mca_rows.append(mca_row)

        # ML dataset row (13 features + outcome)
        ml_row = derive_ml_features(mca_feats, app_data)
        ml_row["outcome"] = int(outcome)
        ml_row["application_id"] = fp.name
        ml_rows.append(ml_row)

    # Save MCA dataset
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    mca_df = pd.DataFrame(mca_rows)
    mca_df.to_csv(OUTPUT_MCA_CSV, index=False)
    mca_df.to_excel(OUTPUT_XLSX, index=False)
    print(f"\nMCA dataset: {len(mca_df)} rows -> {OUTPUT_MCA_CSV}")

    # Save ML dataset
    ml_df = pd.DataFrame(ml_rows)
    ml_df.to_csv(OUTPUT_ML_CSV, index=False)
    print(f"ML dataset:  {len(ml_df)} rows -> {OUTPUT_ML_CSV}")

    if skipped:
        print(f"\nSkipped {skipped} files (no matching spreadsheet row or no outcome)")

    # Summary
    if len(ml_df) > 0:
        print(f"\nML Dataset Summary:")
        print(f"  Total rows:  {len(ml_df)}")
        outcome_counts = ml_df["outcome"].value_counts()
        for val, count in outcome_counts.items():
            label = "repaid" if val == 1 else "defaulted"
            print(f"  Outcome {val} ({label}): {count}")
        print(f"\nTo train the model, run:")
        print(f'  python train_improved_model.py --data "{OUTPUT_ML_CSV}"')


if __name__ == "__main__":
    main()
