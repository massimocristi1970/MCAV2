import json
import re
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# ============================================================
# CONFIG (Windows / OneDrive paths for MCA Scorecard)
# ============================================================
OUTCOMES_XLSX = r"C:\Users\massi\OneDrive - Savvy Loan Products Ltd\Merchant Cash Advance (MCA)\Scorecard Development\training_dataset.xlsx"

JSON_ROOT = r"C:\Users\massi\OneDrive - Savvy Loan Products Ltd\Merchant Cash Advance (MCA)\Scorecard Development\JsonExport"

OUTPUT_CSV = r"C:\Users\massi\OneDrive - Savvy Loan Products Ltd\Merchant Cash Advance (MCA)\Scorecard Development\mca_training_dataset.csv"
OUTPUT_XLSX = r"C:\Users\massi\OneDrive - Savvy Loan Products Ltd\Merchant Cash Advance (MCA)\Scorecard Development\mca_training_dataset.xlsx"


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
    """
    Takes whatever is in the Excel 'application_id' column and returns a clean filename-like key:
    - If it's a full path, keep only the last path component
    - Preserve extension if present
    """
    if x is None:
        return ""
    s = str(x).strip()
    if not s:
        return ""
    s = s.replace("\\", "/")
    s = s.split("/")[-1]
    return s.strip()


def _strip_json_ext(name: str) -> str:
    n = str(name).strip()
    return n[:-5] if n.lower().endswith(".json") else n


def _flatten_transactions(obj):
    """
    Extract transactions list from many JSON shapes.
    Returns list[dict].

    FIX: Handles top-level list JSON (array of transactions).
    """
    # ✅ Case 1: top-level list of transactions
    if isinstance(obj, list):
        return [t for t in obj if isinstance(t, dict)]

    txns = []

    def walk(x):
        nonlocal txns
        if isinstance(x, dict):
            # common keys
            for k in ["transactions", "transaction", "data", "items", "results"]:
                if k in x and isinstance(x[k], list):
                    cand = x[k]
                    if cand and isinstance(cand[0], dict) and any(
                        z in cand[0] for z in ["amount", "date", "name", "merchant_name", "original_description"]
                    ):
                        txns.extend(cand)

                # nested dict case: data -> transactions
                if k in x and isinstance(x[k], dict):
                    vv = x[k].get("transactions")
                    if isinstance(vv, list) and vv and isinstance(vv[0], dict):
                        txns.extend(vv)

            # recurse
            for v in x.values():
                walk(v)

        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(obj)

    # Deduplicate by (date, amount, name)
    seen = set()
    out = []
    for t in txns:
        if not isinstance(t, dict):
            continue
        key = (
            str(t.get("date") or t.get("authorized_date") or ""),
            str(t.get("amount") or ""),
            str(t.get("name") or t.get("merchant_name") or t.get("original_description") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


# ============================================================
# Outcomes loader (hard-locked to your sheet columns)
# ============================================================
def load_outcomes(path_xlsx):
    df = pd.read_excel(path_xlsx)

    expected_file_col = "application_id"
    expected_out_col = "outcome"

    cols_lc = {str(c).strip().lower(): c for c in df.columns}

    file_col = cols_lc.get(expected_file_col)
    out_col = cols_lc.get(expected_out_col)

    if file_col is None or out_col is None:
        raise ValueError(
            "Outcomes Excel missing required columns.\n"
            f"Expected columns: {expected_file_col}, {expected_out_col}\n"
            f"Columns found: {list(df.columns)}"
        )

    df = df[[file_col, out_col]].copy()

    # Clean / normalise keys (supports full path or just filename; with or without .json)
    df[file_col] = df[file_col].apply(_normalise_filename_cell)

    # Ensure outcomes are numeric 0/1
    df[out_col] = pd.to_numeric(df[out_col], errors="coerce").astype("Int64")

    mapping = {}
    for _, row in df.iterrows():
        key = str(row[file_col]).strip()
        oc = row[out_col]
        if not key or pd.isna(oc):
            continue

        oc = int(oc)

        # Map both "RECORD_123.json" and "RECORD_123"
        mapping[key] = oc
        mapping[_strip_json_ext(key)] = oc

    return mapping, file_col, out_col


# ============================================================
# Feature builder (MCA / business-loan oriented)
# ============================================================
def build_mca_features(transactions):
    """
    MCA/business-loan style features:
      - monthly inflow proxy (turnover)
      - volatility and trend
      - concentration (top counterparties)
      - activity / density
      - recent (30/60/90d) splits

    Fix: If all amounts are >= 0 (common in Plaid exports), infer inflow/outflow
         using personal_finance_category.primary when available.
    """
    rows = []
    for t in transactions:
        amt = _safe_float(t.get("amount"))
        dt = _parse_date(t.get("date") or t.get("authorized_date"))
        name = t.get("name") or t.get("merchant_name") or t.get("original_description") or ""

        pfc = t.get("personal_finance_category") or {}
        pfc_primary = ""
        if isinstance(pfc, dict):
            pfc_primary = str(pfc.get("primary") or "").strip().upper()

        if dt is None or np.isnan(amt):
            continue

        rows.append(
            {
                "date": dt,
                "month": _month_key(dt),
                "amount_raw": float(amt),
                "name": str(name),
                "pfc_primary": pfc_primary,
            }
        )

    if not rows:
        return {}

    df = pd.DataFrame(rows).sort_values("date")

    # ------------------------------------------------------------
    # Direction inference
    # If ALL amounts are >= 0, infer inflow/outflow using category.primary.
    # Otherwise use sign-based conventions with a guardrail.
    # ------------------------------------------------------------
    if df["amount_raw"].min() >= 0:
        inflow_primaries = {"INCOME", "TRANSFER_IN"}
        df["inflow"] = np.where(df["pfc_primary"].isin(inflow_primaries), df["amount_raw"], 0.0)
        df["outflow"] = np.where(~df["pfc_primary"].isin(inflow_primaries), df["amount_raw"], 0.0)
        sign_label = "CATEGORY(primary)"
    else:
        # Convention A: + = outflow (Plaid typical), - = inflow
        inflow_A = df["amount_raw"].clip(upper=0).abs()
        outflow_A = df["amount_raw"].clip(lower=0)

        # Convention B: + = inflow, - = outflow
        inflow_B = df["amount_raw"].clip(lower=0)
        outflow_B = df["amount_raw"].clip(upper=0).abs()

        # Choose the convention with the larger inflow total
        use_B = inflow_B.sum() > inflow_A.sum()
        inflow = inflow_B if use_B else inflow_A
        outflow = outflow_B if use_B else outflow_A

        # Guardrail: if chosen convention yields near-zero outflow but the other doesn't, flip it
        if outflow.sum() < 1e-6 and (outflow_A.sum() > 0 or outflow_B.sum() > 0):
            use_B = not use_B
            inflow = inflow_B if use_B else inflow_A
            outflow = outflow_B if use_B else outflow_A

        df["inflow"] = inflow
        df["outflow"] = outflow
        sign_label = "B(+inflow)" if use_B else "A(-inflow)"

    # Monthly rollups
    m = df.groupby("month").agg(
        txn_count=("amount_raw", "size"),
        inflow_sum=("inflow", "sum"),
        outflow_sum=("outflow", "sum"),
    )

    months = int(m.shape[0])
    avg_monthly_inflow = float(m["inflow_sum"].mean()) if months else 0.0
    avg_monthly_outflow = float(m["outflow_sum"].mean()) if months else 0.0
    avg_monthly_net = avg_monthly_inflow - avg_monthly_outflow

    inflow_vol = float(m["inflow_sum"].std(ddof=0)) if months > 1 else 0.0
    outflow_vol = float(m["outflow_sum"].std(ddof=0)) if months > 1 else 0.0

    def slope(series):
        if len(series) < 2:
            return 0.0
        y = series.values.astype(float)
        x = np.arange(len(y), dtype=float)
        xm = x.mean()
        ym = y.mean()
        denom = ((x - xm) ** 2).sum()
        if denom == 0:
            return 0.0
        return float(((x - xm) * (y - ym)).sum() / denom)

    inflow_trend = slope(m["inflow_sum"])
    outflow_trend = slope(m["outflow_sum"])

    # Concentration: top counterparties by inflow (proxy revenue sources)
    inflow_df = df[df["inflow"] > 0].copy()
    if not inflow_df.empty:
        top_inflow = inflow_df.groupby("name")["inflow"].sum().sort_values(ascending=False)
        top_inflow_total = float(top_inflow.sum())
        top1_share = float(top_inflow.iloc[0] / top_inflow_total) if top_inflow_total > 0 else np.nan
        top3_share = float(top_inflow.head(3).sum() / top_inflow_total) if top_inflow_total > 0 else np.nan
        top5_names = top_inflow.head(5).index.tolist()
    else:
        top1_share = np.nan
        top3_share = np.nan
        top5_names = []

    txn_count_total = int(df.shape[0])
    txn_count_avg_month = float(m["txn_count"].mean()) if months else 0.0

    anchor = df["date"].max()

    def recent_sum(days, col):
        cutoff = anchor - pd.Timedelta(days=days)
        return float(df.loc[df["date"] >= cutoff, col].sum())

    outflow_to_inflow = (avg_monthly_outflow / avg_monthly_inflow) if avg_monthly_inflow > 0 else np.nan

    return {
        "months_covered": months,
        "txn_count_total": txn_count_total,
        "txn_count_avg_month": txn_count_avg_month,
        "avg_monthly_inflow": avg_monthly_inflow,  # turnover proxy
        "avg_monthly_outflow": avg_monthly_outflow,
        "avg_monthly_net": float(avg_monthly_net),
        "outflow_to_inflow_ratio": float(outflow_to_inflow) if pd.notna(outflow_to_inflow) else np.nan,
        "inflow_volatility_monthly": inflow_vol,
        "outflow_volatility_monthly": outflow_vol,
        "inflow_trend_slope": float(inflow_trend),
        "outflow_trend_slope": float(outflow_trend),
        "top1_inflow_share": float(top1_share) if pd.notna(top1_share) else np.nan,
        "top3_inflow_share": float(top3_share) if pd.notna(top3_share) else np.nan,
        "top_inflow_name_1": top5_names[0] if len(top5_names) > 0 else "",
        "top_inflow_name_2": top5_names[1] if len(top5_names) > 1 else "",
        "top_inflow_name_3": top5_names[2] if len(top5_names) > 2 else "",
        "top_inflow_name_4": top5_names[3] if len(top5_names) > 3 else "",
        "top_inflow_name_5": top5_names[4] if len(top5_names) > 4 else "",
        "recent_inflow_30d": recent_sum(30, "inflow"),
        "recent_outflow_30d": recent_sum(30, "outflow"),
        "recent_inflow_60d": recent_sum(60, "inflow"),
        "recent_outflow_60d": recent_sum(60, "outflow"),
        "recent_inflow_90d": recent_sum(90, "inflow"),
        "recent_outflow_90d": recent_sum(90, "outflow"),
        "first_txn_date": df["date"].min().date().isoformat(),
        "last_txn_date": df["date"].max().date().isoformat(),
        "sign_convention_used": sign_label,
    }


# ============================================================
# IO
# ============================================================
def iter_json_files(root):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"JSON root folder not found: {root}")
    yield from root.rglob("*.json")


def main():
    outcomes_map, file_col, out_col = load_outcomes(OUTCOMES_XLSX)
    print(f"Loaded outcomes from: {OUTCOMES_XLSX}")
    print(f"Using columns: {file_col=} {out_col=}")
    print(f"Outcome keys loaded (incl. stem variants): {len(outcomes_map)}")

    json_files = list(iter_json_files(JSON_ROOT))
    print(f"Found JSON files under: {JSON_ROOT}")
    print(f"JSON file count: {len(json_files)}")

    records = []
    matched = 0
    missing_outcome = 0
    failed = 0

    for fp in json_files:
        filename = fp.name  # e.g. RECORD_123.json or South Coast_categorized.json
        stem = fp.stem

        outcome = outcomes_map.get(filename)
        if outcome is None:
            outcome = outcomes_map.get(stem)

        if outcome is None:
            missing_outcome += 1
            continue

        try:
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)

            txns = _flatten_transactions(obj)
            feats = build_mca_features(txns)

            row = {
                "application_file": filename,
                "application_key": stem,
                "outcome": int(outcome),
                "json_file_path": str(fp),
                "txn_extracted_count": int(len(txns)),
            }
            row.update(feats)
            records.append(row)
            matched += 1

        except Exception as e:
            failed += 1
            records.append(
                {
                    "application_file": filename,
                    "application_key": stem,
                    "outcome": int(outcome),
                    "json_file_path": str(fp),
                    "txn_extracted_count": np.nan,
                    "error": str(e),
                }
            )

    df = pd.DataFrame(records)

    # Stable ordering
    core_cols = [
        "application_file",
        "application_key",
        "outcome",
        "json_file_path",
        "txn_extracted_count",
        "months_covered",
        "first_txn_date",
        "last_txn_date",
        "sign_convention_used",
    ]
    for c in core_cols:
        if c not in df.columns:
            df[c] = np.nan
    other_cols = [c for c in df.columns if c not in core_cols]
    df = df[core_cols + other_cols]

    print("\n===================================================")
    print(f"Matched rows built: {matched}")
    print(f"JSONs skipped (no matching outcome): {missing_outcome}")
    print(f"Failed parses (kept with error column): {failed}")
    print(f"Final dataset rows: {len(df)}")
    print("===================================================\n")

    df.to_csv(OUTPUT_CSV, index=False)
    df.to_excel(OUTPUT_XLSX, index=False)

    print(f"Saved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
