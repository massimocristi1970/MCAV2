import json
from pathlib import Path
import pandas as pd
import numpy as np

from mca_scorecard_rules import decide_application, Thresholds

# ------------------------------------------------------------
# MUST MATCH your build_training_dataset.py paths
# ------------------------------------------------------------
JSON_ROOT = r"C:\Users\massi\OneDrive - Savvy Loan Products Ltd\Merchant Cash Advance (MCA)\Scorecard Development\JsonExport"

OUT_DIR = r"C:\Users\massi\OneDrive - Savvy Loan Products Ltd\Merchant Cash Advance (MCA)\Scorecard Development"
OUT_CSV = str(Path(OUT_DIR) / "mca_scorecard_decisions_all_apps.csv")
OUT_XLSX = str(Path(OUT_DIR) / "mca_scorecard_decisions_all_apps.xlsx")


# ------------------------------------------------------------
# Import the exact functions from your existing builder file
# (keeps feature calculations consistent)
# ------------------------------------------------------------
from build_training_dataset import _flatten_transactions, build_mca_features  # noqa: E402


def iter_json_files(root):
    rootp = Path(root)
    yield from rootp.rglob("*.json")


def main():
    t = Thresholds()  # initial thresholds (edit in mca_scorecard_rules.py)

    rows = []
    json_files = list(iter_json_files(JSON_ROOT))
    print(f"Found {len(json_files)} json files under {JSON_ROOT}")

    for fp in json_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)

            txns = _flatten_transactions(obj)
            feats = build_mca_features(txns)

            decision, score, reasons = decide_application(feats, t=t)

            rows.append(
                {
                    "application_file": fp.name,
                    "application_key": fp.stem,
                    "json_file_path": str(fp),
                    "txn_extracted_count": len(txns),
                    "decision": decision,
                    "score": score,
                    "reasons": " | ".join(reasons),
                    # include the core features for easy review
                    "months_covered": feats.get("months_covered"),
                    "txn_count_avg_month": feats.get("txn_count_avg_month"),
                    "avg_monthly_inflow": feats.get("avg_monthly_inflow"),
                    "avg_monthly_outflow": feats.get("avg_monthly_outflow"),
                    "inflow_days_30d": feats.get("inflow_days_30d"),
                    "max_inflow_gap_days": feats.get("max_inflow_gap_days"),
                    "inflow_cv": feats.get("inflow_cv"),
                    "sign_convention_used": feats.get("sign_convention_used"),
                    "first_txn_date": feats.get("first_txn_date"),
                    "last_txn_date": feats.get("last_txn_date"),
                }
            )

        except Exception as e:
            rows.append(
                {
                    "application_file": fp.name,
                    "application_key": fp.stem,
                    "json_file_path": str(fp),
                    "txn_extracted_count": np.nan,
                    "decision": "ERROR",
                    "score": np.nan,
                    "reasons": str(e),
                }
            )

    df = pd.DataFrame(rows)

    # Basic ordering
    preferred = [
        "application_file",
        "application_key",
        "decision",
        "score",
        "reasons",
        "txn_extracted_count",
        "months_covered",
        "txn_count_avg_month",
        "avg_monthly_inflow",
        "avg_monthly_outflow",
        "inflow_days_30d",
        "max_inflow_gap_days",
        "inflow_cv",
        "sign_convention_used",
        "first_txn_date",
        "last_txn_date",
        "json_file_path",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]

    df.to_csv(OUT_CSV, index=False)
    df.to_excel(OUT_XLSX, index=False)

    print(f"Saved decisions CSV:  {OUT_CSV}")
    print(f"Saved decisions XLSX: {OUT_XLSX}")

    # Quick summary to console
    print("\nDecision counts:")
    print(df["decision"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
