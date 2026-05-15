import argparse
import re
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = REPO_ROOT / "data" / "training_dataset.csv"
DEFAULT_JSON_FOLDER = REPO_ROOT / "data" / "JsonExport"


def normalize_name(name):
    if pd.isna(name) or name is None:
        return ""

    normalized = str(name).lower().strip()
    normalized = re.sub(r"\.(json|txt|pdf|csv|xlsx?)$", "", normalized)
    normalized = re.sub(r"\s+transaction\s+reports?\b.*$", "", normalized)
    normalized = re.sub(r"[^\w\s&]", " ", normalized)
    replacements = {
        "limited": "ltd",
        "incorporated": "inc",
        "corporation": "corp",
        "company": "co",
    }
    for old, new in replacements.items():
        normalized = re.sub(rf"\b{old}\b", new, normalized)

    return re.sub(r"\s+", " ", normalized).strip()


def read_csv_with_fallback(csv_path):
    for encoding in ("utf-8-sig", "utf-8", "ISO-8859-1", "cp1252"):
        try:
            return pd.read_csv(csv_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(csv_path)


def get_identifier_column(df, requested_column=None):
    if requested_column:
        if requested_column not in df.columns:
            raise ValueError(f"Column '{requested_column}' not found. Available columns: {list(df.columns)}")
        return requested_column

    for candidate in ("application_id", "company_name", "Company Name", "company", "business_name", "Business Name"):
        if candidate in df.columns:
            return candidate

    raise ValueError("No company identifier column found. Add application_id/company_name or pass --column.")


def main():
    parser = argparse.ArgumentParser(description="Compare scorecard CSV company identifiers against JSON filenames.")
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH, type=Path, help="Path to the parameter/training CSV.")
    parser.add_argument("--json-folder", default=DEFAULT_JSON_FOLDER, type=Path, help="Folder containing JSON exports.")
    parser.add_argument("--column", help="CSV identifier column. Defaults to application_id/company_name when present.")
    parser.add_argument("--threshold", type=float, default=90, help="Minimum match score.")
    parser.add_argument("--output", default=REPO_ROOT / "missing_companies.txt", type=Path)
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")
    if not args.json_folder.exists():
        raise FileNotFoundError(f"JSON folder not found: {args.json_folder}")

    df = read_csv_with_fallback(args.csv)
    identifier_column = get_identifier_column(df, args.column)
    companies = df[identifier_column].dropna().astype(str).tolist()
    files = sorted(path.name for path in args.json_folder.glob("*.json"))

    if not files:
        raise ValueError(f"No JSON files found in {args.json_folder}")

    normalized_files = [(file_name, normalize_name(file_name)) for file_name in files]
    missing_companies = []

    print(f"CSV: {args.csv}")
    print(f"Identifier column: {identifier_column}")
    print(f"JSON folder: {args.json_folder}")
    print(f"JSON files found: {len(files)}")
    print("\nStarting comparison...")

    for company in companies:
        norm_company = normalize_name(company)
        best_file = ""
        best_score = 0

        for file_name, norm_file in normalized_files:
            score = max(fuzz.ratio(norm_company, norm_file), fuzz.token_set_ratio(norm_company, norm_file))
            if score > best_score:
                best_score = score
                best_file = file_name

        if best_score < args.threshold:
            missing_companies.append(company)
            print(f"MISSING: {company} (best guess: {best_file} - {best_score:.1f}%)")

    print("\nSUMMARY:")
    print(f"Total IDs in CSV: {len(companies)}")
    print(f"Total JSON files: {len(files)}")
    print(f"Missing below threshold: {len(missing_companies)}")

    args.output.write_text("\n".join(missing_companies), encoding="utf-8")
    print(f"\nMissing IDs saved to {args.output}")


if __name__ == "__main__":
    main()
