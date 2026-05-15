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

    normalized = str(name).strip().lower()
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
    parser = argparse.ArgumentParser(description="Write a CSV report comparing CSV companies with JSON files.")
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH, type=Path, help="Path to the parameter/training CSV.")
    parser.add_argument("--json-folder", default=DEFAULT_JSON_FOLDER, type=Path, help="Folder containing JSON exports.")
    parser.add_argument("--column", help="CSV identifier column. Defaults to application_id/company_name when present.")
    parser.add_argument("--threshold", type=float, default=90, help="Minimum match score.")
    parser.add_argument("--output", default=REPO_ROOT / "matching_results.csv", type=Path)
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")
    if not args.json_folder.exists():
        raise FileNotFoundError(f"JSON folder not found: {args.json_folder}")

    df = pd.read_csv(args.csv)
    identifier_column = get_identifier_column(df, args.column)
    companies = df[identifier_column].dropna().astype(str).tolist()
    json_files = sorted(path.name for path in args.json_folder.glob("*.json"))

    if not json_files:
        raise ValueError(f"No JSON files found in {args.json_folder}")

    normalized_files = [(file_name, normalize_name(file_name)) for file_name in json_files]
    rows = []

    print(f"Matching '{identifier_column}' with JSON files (threshold: {args.threshold:.1f}%)...")

    for company in companies:
        norm_company = normalize_name(company)
        best_file = None
        best_score = 0

        for file_name, norm_file in normalized_files:
            score = max(
                fuzz.ratio(norm_company, norm_file),
                fuzz.partial_ratio(norm_company, norm_file),
                fuzz.token_set_ratio(norm_company, norm_file),
            )
            if score > best_score:
                best_score = score
                best_file = file_name

        matched = best_score >= args.threshold
        rows.append(
            {
                "identifier": company,
                "file": best_file if matched else "MISSING",
                "best_guess": best_file,
                "score": round(best_score, 1),
                "matched": matched,
            }
        )
        status = "MATCH" if matched else "MISSING"
        print(f"{status}: {company} -> {best_file} ({best_score:.1f}%)")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(args.output, index=False)

    print("\nSUMMARY:")
    print(f"Total entries: {len(companies)}")
    print(f"Matches found: {int(results_df['matched'].sum())}")
    print(f"Missing files: {int((~results_df['matched']).sum())}")
    print(f"Match rate: {results_df['matched'].mean() * 100:.1f}%")
    print(f"\nDetailed results saved to {args.output}")


if __name__ == "__main__":
    main()
