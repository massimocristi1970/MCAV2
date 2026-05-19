"""
Sync newly downloaded application files into the batch processor input folders.

Source:
  OneDrive - Savvy Loan Products Ltd/Merchant Cash Advance (MCA)/Applications

Targets:
  Scorecard Development/JsonExport
  Scorecard Development/Arrears_Jsons
  Scorecard Development/Repaid_Jsons
  Scorecard Development/Capitalised_Credit_reports
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path


ONEDRIVE_ROOT = Path(r"C:\Users\Massimo Cristi\OneDrive - Savvy Loan Products Ltd")
APPLICATIONS_DIR = ONEDRIVE_ROOT / "Merchant Cash Advance (MCA)" / "Applications"
SCORECARD_DEV_DIR = ONEDRIVE_ROOT / "Merchant Cash Advance (MCA)" / "Scorecard" / "Scorecard Development"

TARGETS = {
    "all_jsons": SCORECARD_DEV_DIR / "JsonExport",
    "arrears_jsons": SCORECARD_DEV_DIR / "Arrears_Jsons",
    "repaid_jsons": SCORECARD_DEV_DIR / "Repaid_Jsons",
    "capital_reports": SCORECARD_DEV_DIR / "Capitalised_Credit_reports",
}


@dataclass
class CopyAction:
    source: Path
    target: Path
    category: str
    status: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_existing_hashes(target_dir: Path) -> set[str]:
    hashes: set[str] = set()
    if not target_dir.exists():
        return hashes
    for path in target_dir.rglob("*"):
        if path.is_file():
            try:
                hashes.add(file_sha256(path))
            except OSError:
                continue
    return hashes


def unique_target_path(target_dir: Path, filename: str) -> Path:
    candidate = target_dir / filename
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    for index in range(2, 1000):
        next_candidate = target_dir / f"{stem}_{index}{suffix}"
        if not next_candidate.exists():
            return next_candidate
    raise RuntimeError(f"Could not create a unique filename for {candidate}")


def copy_if_new(
    source: Path,
    target_dir: Path,
    category: str,
    dry_run: bool,
    hash_cache: dict[Path, set[str]],
    deep_dedupe: bool,
) -> CopyAction:
    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)

    same_name_target = target_dir / source.name
    if same_name_target.exists() and same_name_target.is_file():
        return CopyAction(source=source, target=same_name_target, category=category, status="duplicate_name_skipped")

    if deep_dedupe:
        source_hash = file_sha256(source)
        existing_hashes = hash_cache.setdefault(target_dir, build_existing_hashes(target_dir))
        if source_hash in existing_hashes:
            return CopyAction(source=source, target=target_dir / source.name, category=category, status="duplicate_content_skipped")
    else:
        existing_hashes = set()

    target = unique_target_path(target_dir, source.name)
    if not dry_run:
        shutil.copy2(source, target)
    if deep_dedupe:
        existing_hashes.add(source_hash)
    return CopyAction(source=source, target=target, category=category, status="copied" if not dry_run else "would_copy")


def is_under_named_folder(path: Path, folder_name: str) -> bool:
    return any(part.lower() == folder_name.lower() for part in path.parts)


def discover_actions(dry_run: bool, deep_dedupe: bool) -> list[CopyAction]:
    if not APPLICATIONS_DIR.exists():
        raise FileNotFoundError(f"Applications folder not found: {APPLICATIONS_DIR}")

    actions: list[CopyAction] = []
    hash_cache: dict[Path, set[str]] = {}
    for source in APPLICATIONS_DIR.rglob("*"):
        if not source.is_file():
            continue

        suffix = source.suffix.lower()
        name_lower = source.name.lower()

        if suffix == ".json":
            actions.append(copy_if_new(source, TARGETS["all_jsons"], "JsonExport", dry_run, hash_cache, deep_dedupe))

            if is_under_named_folder(source, "COLLECTIONS"):
                actions.append(copy_if_new(source, TARGETS["arrears_jsons"], "Arrears_Jsons", dry_run, hash_cache, deep_dedupe))

            if is_under_named_folder(source, "REPAID"):
                actions.append(copy_if_new(source, TARGETS["repaid_jsons"], "Repaid_Jsons", dry_run, hash_cache, deep_dedupe))

        if suffix == ".pdf" and "capital_report" in name_lower:
            actions.append(copy_if_new(source, TARGETS["capital_reports"], "Capitalised_Credit_reports", dry_run, hash_cache, deep_dedupe))

    return actions


def print_summary(actions: list[CopyAction], verbose: bool) -> None:
    if not actions:
        print("No matching JSON or capital_report PDF files found.")
        return

    counts: dict[tuple[str, str], int] = {}
    for action in actions:
        key = (action.category, action.status)
        counts[key] = counts.get(key, 0) + 1

    print("\nSync summary")
    print("============")
    for (category, status), count in sorted(counts.items()):
        print(f"{category}: {status}: {count}")

    if verbose:
        print("\nDetails")
        print("=======")
        for action in actions:
            print(f"[{action.status}] {action.category}: {action.source} -> {action.target}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync batch processor input files from Applications OneDrive folders.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be copied without copying files.")
    parser.add_argument(
        "--deep-dedupe",
        action="store_true",
        help="Scan existing target folders by content hash to skip duplicates with different filenames. Slower on large folders.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print one line for every discovered file action.")
    args = parser.parse_args()

    print(f"Source: {APPLICATIONS_DIR}")
    print(f"Target root: {SCORECARD_DEV_DIR}")
    if args.dry_run:
        print("Mode: dry run")
    if args.deep_dedupe:
        print("Duplicate mode: deep content scan")

    actions = discover_actions(dry_run=args.dry_run, deep_dedupe=args.deep_dedupe)
    print_summary(actions, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
