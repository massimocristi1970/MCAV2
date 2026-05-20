from __future__ import annotations

import argparse
import re
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from rapidfuzz import fuzz, process


REPORT_SUFFIX_PATTERNS = [
    r"\s*-\s*GetReportByID\s*.*$",
    r"\s*-\s*Transaction\s+Reports?\s*.*$",
    r"\s+Transaction\s+Reports?\s*.*$",
    r"\s+Transactions?\s*.*$",
    r"\s+Ob\s*\d+\s*$",
    r"\s+App\s*\d+\s*$",
    r"\s+app\s*\d+\s*$",
    r"\s+Original\s*$",
    r"\s+Orginal\s+Bank\s+acc\s*$",
    r"\s+Clover\s+Acc\s*$",
    r"\s*-\s*\d{6}\s*v?\d*\s*$",
    r"\s*-\s*\d{2}\s*$",
    r"\s+\d+(?:st|nd|rd|th)\s+app\s*$",
    r"\s*\(\d+\)\s*$",
    r"\s*\d+,\d+\s*$",
    r"\s+categorized\s*$",
    r"\s+categorised\s*$",
    r"\s*-\s*$",
]


def clean_json_company_stem(stem: str) -> str:
    cleaned = str(stem or "").replace("_", " ").strip()
    previous = None
    while previous != cleaned:
        previous = cleaned
        for pattern in REPORT_SUFFIX_PATTERNS:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ._-")
    cleaned = re.sub(r"\bLTD\b", "Ltd", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bLIMITED\b", "Limited", cleaned, flags=re.IGNORECASE)
    return cleaned or str(stem or "").strip()


def normalize_for_match(value: Any) -> str:
    if pd.isna(value) or value is None:
        return ""
    normalized = str(value).lower().strip()
    normalized = re.sub(r"\.(json|txt|pdf|csv|xlsx?)$", "", normalized)
    normalized = normalized.replace("_", " ")
    normalized = clean_json_company_stem(normalized)
    normalized = normalized.lower()
    normalized = re.sub(r"[^\w\s&]", " ", normalized)
    replacements = {
        "limited": "ltd",
        "incorporated": "inc",
        "corporation": "corp",
        "company": "co",
    }
    for old, new in replacements.items():
        normalized = re.sub(rf"\b{old}\b", new, normalized)
    normalized = re.sub(r"\bthe\b", "", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def display_case_name(value: str) -> str:
    if not value:
        return value
    words = []
    for word in value.split(" "):
        if word.upper() in {"UK", "GB", "MCA", "ET", "IT"}:
            words.append(word.upper())
        elif word.lower() in {"ltd", "limited"}:
            words.append("Ltd" if word.lower() == "ltd" else "Limited")
        elif word.isupper() and len(word) <= 4:
            words.append(word)
        else:
            words.append(word[:1].upper() + word[1:])
    return " ".join(words)


def unique_target_names(files: list[Path]) -> pd.DataFrame:
    base_rows = []
    grouped: defaultdict[str, list[Path]] = defaultdict(list)
    for path in files:
        clean_name = display_case_name(clean_json_company_stem(path.stem))
        grouped[clean_name.lower()].append(path)
        base_rows.append({"path": path, "clean_company_name": clean_name})

    group_counts = {key: len(value) for key, value in grouped.items()}
    occurrence_counter: Counter[str] = Counter()
    rows = []
    for item in base_rows:
        path = item["path"]
        clean_name = item["clean_company_name"]
        key = clean_name.lower()
        occurrence_counter[key] += 1
        collision_count = group_counts[key]

        if collision_count == 1:
            final_name = f"{clean_name}.json"
            collision_status = "unique"
        else:
            final_name = f"{clean_name} - {occurrence_counter[key]:02d}.json"
            collision_status = "collision_numbered"

        rows.append(
            {
                "original_filename": path.name,
                "clean_company_name": clean_name,
                "proposed_filename": final_name,
                "collision_group_size": collision_count,
                "collision_status": collision_status,
                "would_rename": path.name != final_name,
                "file_size": path.stat().st_size,
                "last_modified": path.stat().st_mtime,
            }
        )
    return pd.DataFrame(rows)


def read_mapping(mapping_path: Path) -> pd.DataFrame:
    suffix = mapping_path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(mapping_path, sheet_name="Export")
    return pd.read_csv(mapping_path)


def find_customer_column(df: pd.DataFrame) -> str:
    for candidate in ("CustomerName", "company_name", "Company Name", "company", "business_name", "Business Name"):
        if candidate in df.columns:
            return candidate
    raise ValueError(f"No customer/company column found. Columns: {list(df.columns)}")


def parse_approved_matches(values: list[str] | None) -> dict[str, str]:
    approved: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Approved match must be in CustomerName=JsonCompanyName format: {value}")
        customer_name, json_company_name = value.split("=", 1)
        customer_key = normalize_for_match(customer_name)
        json_value = json_company_name.strip()
        if not customer_key or not json_value:
            raise ValueError(f"Approved match has a blank side: {value}")
        approved[customer_key] = json_value
    return approved


def parse_rejected_matches(values: list[str] | None) -> set[tuple[str, str]]:
    rejected: set[tuple[str, str]] = set()
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Rejected match must be in CustomerName=JsonCompanyName format: {value}")
        customer_name, json_company_name = value.split("=", 1)
        customer_key = normalize_for_match(customer_name)
        json_key = normalize_for_match(json_company_name)
        if not customer_key or not json_key:
            raise ValueError(f"Rejected match has a blank side: {value}")
        rejected.add((customer_key, json_key))
    return rejected


def compare_mapping(
    rename_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    threshold: float,
    approved_matches: dict[str, str] | None = None,
    rejected_matches: set[tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    approved_matches = approved_matches or {}
    rejected_matches = rejected_matches or set()
    customer_column = find_customer_column(mapping_df)
    valid_mapping = mapping_df.copy()
    valid_mapping[customer_column] = valid_mapping[customer_column].astype(str).str.strip()
    valid_mapping = valid_mapping[
        valid_mapping[customer_column].ne("")
        & ~valid_mapping[customer_column].str.fullmatch(r"\+?\d+", na=False)
    ].copy()

    json_choices = {
        normalize_for_match(row["clean_company_name"]): row["clean_company_name"]
        for _, row in rename_df.drop_duplicates("clean_company_name").iterrows()
        if normalize_for_match(row["clean_company_name"])
    }
    json_norm_names = list(json_choices.keys())

    rows = []
    for _, row in valid_mapping.iterrows():
        customer = row[customer_column]
        norm_customer = normalize_for_match(customer)
        best_norm = ""
        score = 0.0
        strategy = "none"

        if norm_customer in json_choices:
            best_norm = norm_customer
            score = 100.0
            strategy = "exact_normalized"
        elif norm_customer in approved_matches:
            approved_json_name = approved_matches[norm_customer]
            approved_json_norm = normalize_for_match(approved_json_name)
            if approved_json_norm not in json_choices:
                raise ValueError(
                    f"Approved JSON company name does not exist in cleaned JSON list: {approved_json_name}"
                )
            best_norm = approved_json_norm
            score = 100.0
            strategy = "approved_override"
        elif json_norm_names:
            match = process.extractOne(
                norm_customer,
                json_norm_names,
                scorer=lambda a, b, **_: max(fuzz.ratio(a, b), fuzz.token_set_ratio(a, b), fuzz.partial_ratio(a, b)),
            )
            if match:
                best_norm, score, _idx = match
                strategy = "fuzzy"

        is_rejected_match = (norm_customer, best_norm) in rejected_matches
        if is_rejected_match:
            strategy = "rejected_override"

        rows.append(
            {
                "AppID": row.get("AppID"),
                "CustomerName": customer,
                "normalized_customer": norm_customer,
                "best_json_company": json_choices.get(best_norm, ""),
                "match_score": round(float(score), 1),
                "match_strategy": strategy,
                "matched": (strategy == "approved_override" or float(score) >= threshold) and not is_rejected_match,
                "OB Signed": row.get("OB Signed"),
            }
        )

    comparison_df = pd.DataFrame(rows)

    matched_json_names = set(comparison_df.loc[comparison_df["matched"], "best_json_company"].dropna().astype(str))
    json_without_mapping = rename_df[~rename_df["clean_company_name"].isin(matched_json_names)].copy()
    return comparison_df, json_without_mapping


def apply_renames(rename_df: pd.DataFrame, json_folder: Path, output_dir: Path) -> Path:
    moving_df = rename_df[rename_df["would_rename"]].copy()
    target_counts = moving_df["proposed_filename"].value_counts()
    duplicate_targets = target_counts[target_counts > 1]
    if not duplicate_targets.empty:
        raise ValueError(f"Duplicate proposed target filenames found: {duplicate_targets.to_dict()}")

    all_original_names = set(rename_df["original_filename"].astype(str))
    all_original_names_casefolded = {name.casefold() for name in all_original_names}
    all_proposed_names = set(rename_df["proposed_filename"].astype(str))
    conflicts = []
    for proposed_name in sorted(all_proposed_names):
        target = json_folder / proposed_name
        if target.exists() and proposed_name.casefold() not in all_original_names_casefolded:
            conflicts.append(str(target))
    if conflicts:
        raise FileExistsError("Proposed target already exists outside the rename set:\n" + "\n".join(conflicts[:20]))

    batch_id = uuid.uuid4().hex[:12]
    temp_rows = []
    for _, row in moving_df.iterrows():
        source = json_folder / str(row["original_filename"])
        if not source.exists():
            raise FileNotFoundError(f"Source file missing before rename: {source}")
        temp = json_folder / f".codex-renaming-{batch_id}-{source.name}"
        if temp.exists():
            raise FileExistsError(f"Temporary file already exists: {temp}")
        temp_rows.append((row, source, temp, json_folder / str(row["proposed_filename"])))

    log_rows = []
    for row, source, temp, target in temp_rows:
        source.rename(temp)
        log_rows.append(
            {
                "original_filename": row["original_filename"],
                "temporary_filename": temp.name,
                "proposed_filename": row["proposed_filename"],
                "stage": "temporary",
            }
        )

    for row, _source, temp, target in temp_rows:
        temp.rename(target)
        log_rows.append(
            {
                "original_filename": row["original_filename"],
                "temporary_filename": temp.name,
                "proposed_filename": row["proposed_filename"],
                "stage": "final",
            }
        )

    log_path = output_dir / "json_rename_apply_log.csv"
    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    return log_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run JsonExport filename cleanup and mapping comparison.")
    parser.add_argument("--json-folder", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_outputs") / "json_filename_cleanup")
    parser.add_argument("--threshold", type=float, default=88.0)
    parser.add_argument(
        "--approved-match",
        action="append",
        default=[],
        help="Approved weak match in CustomerName=JsonCompanyName format. May be passed more than once.",
    )
    parser.add_argument(
        "--rejected-match",
        action="append",
        default=[],
        help="Rejected fuzzy match in CustomerName=JsonCompanyName format. May be passed more than once.",
    )
    parser.add_argument("--apply", action="store_true", help="Rename JSON files in place after writing the dry-run files.")
    args = parser.parse_args()

    files = sorted(args.json_folder.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {args.json_folder}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rename_df = unique_target_names(files)
    mapping_df = read_mapping(args.mapping)
    approved_matches = parse_approved_matches(args.approved_match)
    rejected_matches = parse_rejected_matches(args.rejected_match)
    comparison_df, json_without_mapping_df = compare_mapping(
        rename_df,
        mapping_df,
        args.threshold,
        approved_matches,
        rejected_matches,
    )

    rename_path = args.output_dir / "json_rename_dry_run.csv"
    comparison_path = args.output_dir / "data_mapping_comparison.csv"
    missing_path = args.output_dir / "mapping_rows_without_json_match.csv"
    extras_path = args.output_dir / "json_files_without_mapping_match.csv"
    summary_path = args.output_dir / "summary.txt"

    rename_df.to_csv(rename_path, index=False)
    comparison_df.to_csv(comparison_path, index=False)
    comparison_df[~comparison_df["matched"]].to_csv(missing_path, index=False)
    json_without_mapping_df.to_csv(extras_path, index=False)

    apply_log_path = None
    if args.apply:
        apply_log_path = apply_renames(rename_df, args.json_folder, args.output_dir)

    summary = [
        f"JSON files: {len(rename_df)}",
        f"Files that would be renamed: {int(rename_df['would_rename'].sum())}",
        f"Collision-numbered files: {int((rename_df['collision_status'] == 'collision_numbered').sum())}",
        f"Collision groups: {int((rename_df.groupby(rename_df['clean_company_name'].str.lower()).size() > 1).sum())}",
        f"Mapping rows compared: {len(comparison_df)}",
        f"Mapping rows matched >= {args.threshold}: {int(comparison_df['matched'].sum())}",
        f"Mapping rows without JSON match: {int((~comparison_df['matched']).sum())}",
        f"Approved weak matches: {len(approved_matches)}",
        f"Rejected fuzzy matches: {len(rejected_matches)}",
        f"Unique cleaned JSON company names: {rename_df['clean_company_name'].nunique()}",
        f"Cleaned JSON company names not matched to mapping: {json_without_mapping_df['clean_company_name'].nunique()}",
        f"Apply mode: {'yes' if args.apply else 'no'}",
        "",
        f"Rename dry run: {rename_path}",
        f"Mapping comparison: {comparison_path}",
        f"Missing mapping rows: {missing_path}",
        f"JSON extras: {extras_path}",
    ]
    if apply_log_path:
        summary.append(f"Apply log: {apply_log_path}")
    summary_path.write_text("\n".join(summary), encoding="utf-8")
    print(summary_path)
    print("\n".join(summary[:9]))


if __name__ == "__main__":
    main()
