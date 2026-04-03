"""Validate business-account risk signals against labelled outcomes."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from app.services.business_risk_signals import calculate_business_metrics
from build_training_dataset import _flatten_transactions, iter_json_files, load_application_data


_BASE_DIR = os.environ.get("TRAINING_DATA_DIR", os.getcwd())
OUTCOMES_XLSX = os.environ.get(
    "TRAINING_OUTCOMES_XLSX",
    os.path.join(_BASE_DIR, "data", "training_dataset.xlsx"),
)
JSON_ROOT = os.environ.get(
    "TRAINING_JSON_ROOT",
    os.path.join(_BASE_DIR, "data", "JsonExport"),
)
OUTPUT_DIR = Path(os.environ.get("TRAINING_OUTPUT_DIR", os.path.join(_BASE_DIR, "data")))

SIGNAL_COLUMNS = [
    "Funding Reliance Ratio",
    "External Funding Reliance Ratio",
    "Internal Transfer Activity Ratio",
    "Bank Charge Count",
    "Bank Charge Burden",
    "Revenue Regularity Score",
    "Deposit Frequency Score",
    "Top Revenue Source Percentage",
    "Revenue Concentration Ratio",
    "NSF Count 90D",
    "Active Lenders Detected",
    "Average Inflow Gap Days",
    "Max Inflow Gap Days",
    "Advanced Risk Score",
]


def _prepare_transactions(transactions: List[Dict]) -> pd.DataFrame:
    rows = []
    for txn in transactions:
        amount = txn.get("amount")
        txn_date = txn.get("date") or txn.get("authorized_date")
        if amount is None or txn_date is None:
            continue

        pfc = txn.get("personal_finance_category") or {}
        rows.append(
            {
                "date": txn_date,
                "amount": amount,
                "amount_original": amount,
                "name": txn.get("name") or txn.get("merchant_name") or "",
                "name_y": txn.get("name") or txn.get("merchant_name") or "",
                "merchant_name": txn.get("merchant_name"),
                "personal_finance_category.detailed": pfc.get("detailed", ""),
                "balances.available": txn.get("balances", {}).get("available"),
            }
        )

    return pd.DataFrame(rows)


def _signal_summary(df: pd.DataFrame) -> pd.DataFrame:
    good = df[df["Outcome"] == 1]
    bad = df[df["Outcome"] == 0]
    rows = []
    for column in SIGNAL_COLUMNS:
        if column not in df.columns:
            continue
        series = pd.to_numeric(df[column], errors="coerce")
        good_mean = pd.to_numeric(good[column], errors="coerce").mean()
        bad_mean = pd.to_numeric(bad[column], errors="coerce").mean()
        pooled_std = series.std(ddof=0)
        effect_size = 0.0 if pd.isna(pooled_std) or pooled_std == 0 else (good_mean - bad_mean) / pooled_std
        rows.append(
            {
                "signal": column,
                "good_mean": round(float(good_mean), 4) if pd.notna(good_mean) else np.nan,
                "bad_mean": round(float(bad_mean), 4) if pd.notna(bad_mean) else np.nan,
                "mean_gap": round(float(good_mean - bad_mean), 4) if pd.notna(good_mean) and pd.notna(bad_mean) else np.nan,
                "effect_size": round(float(effect_size), 4),
            }
        )

    return pd.DataFrame(rows).sort_values("effect_size", key=lambda s: s.abs(), ascending=False)


def main() -> int:
    app_data_map = load_application_data(OUTCOMES_XLSX)
    rows = []

    for fp in iter_json_files(JSON_ROOT):
        key = fp.stem
        app_data = app_data_map.get(fp.name) or app_data_map.get(key) or app_data_map.get(key.lower())
        if app_data is None or pd.isna(app_data.get("Outcome")):
            continue

        with open(fp, "r", encoding="utf-8") as handle:
            transactions = _flatten_transactions(json.load(handle))

        tx_df = _prepare_transactions(transactions)
        if tx_df.empty:
            continue

        metrics = calculate_business_metrics(tx_df, app_data.get("company_age_months", 12))
        if not metrics:
            continue

        row = {
            "company_name": app_data.get("company_name", key),
            "application_file": fp.name,
            "Outcome": int(app_data["Outcome"]),
        }
        for column in SIGNAL_COLUMNS:
            row[column] = metrics.get(column)
        rows.append(row)

    if not rows:
        raise RuntimeError("No labelled applications with usable transaction data were found.")

    signal_df = pd.DataFrame(rows)
    summary_df = _signal_summary(signal_df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    signal_df.to_csv(OUTPUT_DIR / "business_risk_signals.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "business_risk_signal_validation.csv", index=False)

    bad_rate = 1 - signal_df["Outcome"].mean()
    print(f"Labelled applications analysed: {len(signal_df)}")
    print(f"Bad rate: {bad_rate:.1%}")
    print("Top separating signals:")
    print(summary_df.head(10).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
