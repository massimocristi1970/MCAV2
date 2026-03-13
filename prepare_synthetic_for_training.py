"""
Prepare synthetic dataset for training-format and optionally run a validation train.

Reads data/synthetic/synthetic_dataset.csv (or a path you give). If it has
synthetic_outcome, creates a CSV with the 13 ML features + 'outcome' so you can
run train_improved_model.py. Optionally runs training to a SEPARATE output folder
so the production model (app/models/model_artifacts/) is never overwritten.

Use this only for pipeline validation and testing, NOT for production calibration.

Usage:
  python prepare_synthetic_for_training.py
  python prepare_synthetic_for_training.py --synthetic data/synthetic/synthetic_dataset.csv --run-validation
  python prepare_synthetic_for_training.py --synthetic data/synthetic/run_adverse/synthetic_dataset.csv --run-validation --output-dir app/models/model_artifacts_synthetic_test
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SYNTHETIC = SCRIPT_DIR / "data" / "synthetic" / "synthetic_dataset.csv"
DEFAULT_TRAINING_CSV = SCRIPT_DIR / "data" / "synthetic" / "synthetic_training_format.csv"
VALIDATION_OUTPUT_DIR = SCRIPT_DIR / "app" / "models" / "model_artifacts_synthetic_test"

FEATURE_COLS = [
    "Directors Score", "Total Revenue", "Total Debt", "Debt-to-Income Ratio",
    "Operating Margin", "Debt Service Coverage Ratio", "Cash Flow Volatility",
    "Revenue Growth Rate", "Average Month-End Balance",
    "Average Negative Balance Days per Month", "Number of Bounced Payments",
    "Company Age (Months)", "Sector_Risk",
]


def main():
    parser = argparse.ArgumentParser(description="Prepare synthetic CSV for training format; optional validation train.")
    parser.add_argument("--synthetic", type=Path, default=DEFAULT_SYNTHETIC, help="Path to synthetic_dataset.csv")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_TRAINING_CSV, help="Path to write training-format CSV")
    parser.add_argument("--run-validation", action="store_true", help="Run train_improved_model.py on the prepared CSV to a test folder")
    parser.add_argument("--output-dir", type=Path, default=VALIDATION_OUTPUT_DIR, help="Output dir for validation train (only if --run-validation)")
    args = parser.parse_args()

    if not args.synthetic.exists():
        print(f"Error: synthetic file not found: {args.synthetic}")
        print("Generate synthetic data first (e.g. with --generate-outcomes) so synthetic_outcome exists.")
        return 1

    df = pd.read_csv(args.synthetic)
    if "synthetic_outcome" not in df.columns:
        print("Error: synthetic_dataset has no 'synthetic_outcome' column.")
        print("Re-run the synthetic engine with --generate-outcomes so outcomes exist.")
        return 1

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"Error: synthetic dataset missing feature columns: {missing}")
        return 1

    out = df[FEATURE_COLS].copy()
    out["outcome"] = df["synthetic_outcome"].astype(int)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"Written: {args.output_csv} ({len(out)} rows)")

    if args.run_validation:
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "train_improved_model.py"),
            "--data", str(args.output_csv),
            "--output-dir", str(args.output_dir),
        ]
        print(f"\nRunning validation train (output to {args.output_dir}, production model unchanged):")
        print(" ", " ".join(cmd))
        code = subprocess.call(cmd)
        if code != 0:
            return code
        print("\nValidation train finished. Production artefacts in app/models/model_artifacts/ were not modified.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
