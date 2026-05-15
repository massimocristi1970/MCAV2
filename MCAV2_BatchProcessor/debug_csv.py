import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = REPO_ROOT / "data" / "training_dataset.csv"


def main():
    parser = argparse.ArgumentParser(description="Print a quick diagnostic summary for a CSV file.")
    parser.add_argument("csv", nargs="?", default=DEFAULT_CSV_PATH, type=Path, help="CSV file to inspect.")
    parser.add_argument("--rows", type=int, default=10, help="Number of rows to preview.")
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    preview_rows = max(args.rows, 1)

    print("CSV loaded successfully")
    print(f"Path: {args.csv}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst {preview_rows} rows:")
    print(df.head(preview_rows))

    if len(df.columns) >= 1:
        print(f"\nCOLUMN A DATA (first {preview_rows}):")
        for i, value in enumerate(df.iloc[:preview_rows, 0]):
            print(f"Row {i + 1}: {value!r}")

    if len(df.columns) >= 2:
        print(f"\nCOLUMN B DATA (first {preview_rows}):")
        for i, value in enumerate(df.iloc[:preview_rows, 1]):
            print(f"Row {i + 1}: {value!r}")

    print("\nDATA TYPES:")
    print(df.dtypes)

    print("\nNULL VALUES:")
    print(df.isnull().sum())


if __name__ == "__main__":
    main()
