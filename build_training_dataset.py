import json
import os
import re
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# ============================================================
# CONFIG - Uses environment variables with fallback defaults
# Set these environment variables or create a .env file:
#   TRAINING_OUTCOMES_XLSX - Path to training outcomes Excel file
#   TRAINING_JSON_ROOT - Directory containing JSON transaction files
#   TRAINING_OUTPUT_DIR - Directory for output files (optional)
# ============================================================

# Get base directory from environment or use current directory
_BASE_DIR = os.environ.get('TRAINING_DATA_DIR', os.getcwd())

# Input paths - can be overridden with environment variables
OUTCOMES_XLSX = os.environ.get(
    'TRAINING_OUTCOMES_XLSX',
    os.path.join(_BASE_DIR, 'data', 'training_dataset.xlsx')
)

JSON_ROOT = os.environ.get(
    'TRAINING_JSON_ROOT',
    os.path.join(_BASE_DIR, 'data', 'JsonExport')
)

# Output paths
_OUTPUT_DIR = os.environ.get('TRAINING_OUTPUT_DIR', os.path.join(_BASE_DIR, 'data'))
OUTPUT_CSV = os.path.join(_OUTPUT_DIR, 'mca_training_dataset.csv')
OUTPUT_XLSX = os.path.join(_OUTPUT_DIR, 'mca_training_dataset.xlsx')


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
# Outcomes loader
# ============================================================
def load_outcomes(path_xlsx):
    df = pd.read_excel(path_xlsx)

    file_col = "application_id"
    out_col = "outcome"

    df = df[[file_col, out_col]].copy()
    df[file_col] = df[file_col].apply(_normalise_filename_cell)
    df[out_col] = pd.to_numeric(df[out_col], errors="coerce").astype("Int64")

    mapping = {}
    for _, r in df.iterrows():
        if pd.isna(r[out_col]):
            continue
        k = r[file_col]
        mapping[k] = int(r[out_col])
        mapping[_strip_json_ext(k)] = int(r[out_col])

    return mapping, file_col, out_col


# ============================================================
# Feature builder
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

    return {
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
    }


# ============================================================
# IO / Main
# ============================================================
def iter_json_files(root):
    yield from Path(root).rglob("*.json")


def main():
    outcomes_map, _, _ = load_outcomes(OUTCOMES_XLSX)

    rows = []
    for fp in iter_json_files(JSON_ROOT):
        key = fp.stem
        outcome = outcomes_map.get(fp.name) or outcomes_map.get(key)
        if outcome is None:
            continue

        with open(fp, "r", encoding="utf-8") as f:
            txns = _flatten_transactions(json.load(f))

        feats = build_mca_features(txns)
        feats.update(
            {
                "application_file": fp.name,
                "application_key": key,
                "outcome": outcome,
                "txn_extracted_count": len(txns),
            }
        )
        rows.append(feats)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    df.to_excel(OUTPUT_XLSX, index=False)

    print(f"Saved {len(df)} rows → {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
