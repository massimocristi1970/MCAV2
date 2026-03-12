"""
Build Full Feature Dataset (Stage 1 of Reject Inference Pipeline)

Reads the same spreadsheet and JSON data as build_training_dataset.py and produces
a single dataset for ALL matched applications:
  - Funded/labelled rows: Outcome is present (1 = repaid, 0 = defaulted)
  - Rejected/unfunded rows: Outcome is blank

Uses the same feature derivation logic as build_training_dataset.py via imports.
Does NOT modify build_training_dataset.py or its behaviour.

Output: full_feature_dataset.csv with ML features, identifiers, Outcome, and
        is_labelled / is_rejected_or_unfunded flags.

Usage:
  python build_full_feature_dataset.py

  Override paths with environment variables (same as build_training_dataset.py):
    TRAINING_OUTCOMES_XLSX  - Path to application spreadsheet
    TRAINING_JSON_ROOT     - Directory containing JSON transaction files
    TRAINING_OUTPUT_DIR    - Directory for output files
"""

import json
import os
from pathlib import Path

import pandas as pd

# Reuse feature-building logic from existing script (no changes to that file)
from build_training_dataset import (
    load_application_data,
    _flatten_transactions,
    build_mca_features,
    derive_ml_features,
    iter_json_files,
)

# ============================================================
# CONFIG (same env vars as build_training_dataset.py)
# ============================================================

_BASE_DIR = os.environ.get("TRAINING_DATA_DIR", os.getcwd())
OUTCOMES_XLSX = os.environ.get(
    "TRAINING_OUTCOMES_XLSX",
    os.path.join(_BASE_DIR, "data", "training_dataset.xlsx"),
)
JSON_ROOT = os.environ.get(
    "TRAINING_JSON_ROOT",
    os.path.join(_BASE_DIR, "data", "JsonExport"),
)
_OUTPUT_DIR = os.environ.get("TRAINING_OUTPUT_DIR", os.path.join(_BASE_DIR, "data"))
OUTPUT_FULL_CSV = os.path.join(_OUTPUT_DIR, "full_feature_dataset.csv")

# Required columns for downstream pipeline (fail loudly if missing)
ML_FEATURE_COLS = [
    "Directors Score",
    "Total Revenue",
    "Total Debt",
    "Debt-to-Income Ratio",
    "Operating Margin",
    "Debt Service Coverage Ratio",
    "Cash Flow Volatility",
    "Revenue Growth Rate",
    "Average Month-End Balance",
    "Average Negative Balance Days per Month",
    "Number of Bounced Payments",
    "Company Age (Months)",
    "Sector_Risk",
]
REQUIRED_NON_FEATURE = ["company_name", "Outcome", "is_labelled", "is_rejected_or_unfunded"]


def main():
    """
    Build full feature dataset for all matched applications (funded + rejected).
    Outputs full_feature_dataset.csv. Does not skip rows with blank Outcome.
    """
    print(f"Loading spreadsheet: {OUTCOMES_XLSX}")
    app_data_map = load_application_data(OUTCOMES_XLSX)
    unique_apps = len({id(v) for v in app_data_map.values()})
    print(f"  Found {unique_apps} applications in spreadsheet")

    print(f"Scanning JSON files: {JSON_ROOT}")
    rows = []
    matched = 0
    skipped_no_match = 0
    skipped_no_txns = 0

    for fp in iter_json_files(JSON_ROOT):
        key = fp.stem
        app_data = (
            app_data_map.get(fp.name)
            or app_data_map.get(key)
            or app_data_map.get(key.lower())
        )
        if app_data is None:
            skipped_no_match += 1
            continue

        with open(fp, "r", encoding="utf-8") as f:
            txns = _flatten_transactions(json.load(f))

        mca_feats = build_mca_features(txns)
        if not mca_feats:
            print(f"  WARNING: No transactions extracted from {fp.name}")
            skipped_no_txns += 1
            continue

        matched += 1
        outcome_raw = app_data.get("Outcome")
        is_labelled = not pd.isna(outcome_raw)
        is_rejected_or_unfunded = pd.isna(outcome_raw)

        ml_row = derive_ml_features(mca_feats, app_data)
        ml_row["company_name"] = app_data.get("company_name", key)
        ml_row["application_file"] = fp.name
        ml_row["Outcome"] = outcome_raw
        ml_row["is_labelled"] = is_labelled
        ml_row["is_rejected_or_unfunded"] = is_rejected_or_unfunded
        rows.append(ml_row)

    if not rows:
        raise RuntimeError(
            "No matched applications with valid transaction data. "
            "Check spreadsheet and JSON_ROOT."
        )

    df = pd.DataFrame(rows)

    # Ensure column order: features first, then identifiers and flags
    id_cols = ["company_name", "application_file", "Outcome", "is_labelled", "is_rejected_or_unfunded"]
    for c in id_cols:
        if c not in df.columns:
            raise ValueError(f"Required column missing after build: {c}")
    for c in ML_FEATURE_COLS:
        if c not in df.columns:
            raise ValueError(f"Required ML feature column missing: {c}")

    out_cols = ML_FEATURE_COLS + id_cols
    df = df[out_cols]

    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FULL_CSV, index=False)
    print(f"\nFull feature dataset: {len(df)} rows -> {OUTPUT_FULL_CSV}")

    n_labelled = df["is_labelled"].sum()
    n_rejected = df["is_rejected_or_unfunded"].sum()
    print(f"  Labelled (funded):     {n_labelled}")
    print(f"  Rejected/unfunded:     {n_rejected}")
    print(f"  Skipped (no match):    {skipped_no_match}")
    print(f"  Skipped (no txns):     {skipped_no_txns}")
    return OUTPUT_FULL_CSV


if __name__ == "__main__":
    main()
