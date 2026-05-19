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
import re
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
DUPLICATE_ARCHIVE_DIR = SCORECARD_DEV_DIR / "_Duplicate_batch_inputs_do_not_upload"

STATUS_FOLDER_NAMES = {
    "applications",
    "collections",
    "declined applications",
    "funded applications",
    "repaid",
    "underwriting",
}


@dataclass
class CopyAction:
    source: Path
    target: Path
    category: str
    status: str


class TargetFingerprintIndex:
    """Fast duplicate lookup: size first, then hashes only for same-size files."""

    def __init__(self) -> None:
        self._size_index: dict[Path, dict[int, list[Path]]] = {}
        self._hash_index: dict[tuple[Path, int], dict[str, Path]] = {}

    def _build_size_index(self, target_dir: Path) -> dict[int, list[Path]]:
        if target_dir not in self._size_index:
            by_size: dict[int, list[Path]] = {}
            if target_dir.exists():
                for path in target_dir.glob("*.json"):
                    if path.is_file():
                        by_size.setdefault(path.stat().st_size, []).append(path)
            self._size_index[target_dir] = by_size
        return self._size_index[target_dir]

    def find_duplicate(self, source: Path, target_dir: Path) -> Path | None:
        source_size = source.stat().st_size
        candidates = self._build_size_index(target_dir).get(source_size, [])
        if not candidates:
            return None

        key = (target_dir, source_size)
        if key not in self._hash_index:
            hashes: dict[str, Path] = {}
            for candidate in candidates:
                try:
                    hashes.setdefault(file_sha256(candidate), candidate)
                except OSError:
                    continue
            self._hash_index[key] = hashes

        try:
            return self._hash_index[key].get(file_sha256(source))
        except OSError:
            return None

    def add(self, path: Path) -> None:
        if path.suffix.lower() != ".json" or not path.exists():
            return
        target_dir = path.parent
        size_index = self._build_size_index(target_dir)
        size_index.setdefault(path.stat().st_size, []).append(path)
        self._hash_index.pop((target_dir, path.stat().st_size), None)


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


def safe_filename_part(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]+', " ", str(value or ""))
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ._-")
    return cleaned or "Unknown Business"


def is_generic_json_name(path: Path) -> bool:
    stem = path.stem.strip()
    compact = re.sub(r"[\s_().-]+", "", stem).lower()
    return bool(
        re.fullmatch(r"\d+[a-z0-9]*", compact)
        or compact.startswith("getreport")
        or compact.startswith("reportid")
        or compact in {"report", "transactions", "transactionreport"}
    )


def business_folder_for_source(source: Path) -> str | None:
    """Use the nearest non-status parent folder as the business name."""
    try:
        relative_parent = source.parent.relative_to(APPLICATIONS_DIR)
    except ValueError:
        return None

    for part in reversed(relative_parent.parts):
        if part.lower().strip() not in STATUS_FOLDER_NAMES:
            return safe_filename_part(part)
    return None


def target_filename_for_source(source: Path) -> str:
    if source.suffix.lower() != ".json" or not is_generic_json_name(source):
        return source.name

    business_name = business_folder_for_source(source)
    if not business_name:
        return source.name

    original_stem = safe_filename_part(source.stem)
    return f"{business_name} - {original_stem}{source.suffix}"


def copy_if_new(
    source: Path,
    target_dir: Path,
    category: str,
    dry_run: bool,
    hash_cache: dict[Path, set[str]],
    deep_dedupe: bool,
    fingerprint_index: TargetFingerprintIndex,
) -> CopyAction:
    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)

    target_name = target_filename_for_source(source)
    same_name_target = target_dir / target_name
    if same_name_target.exists() and same_name_target.is_file():
        return CopyAction(source=source, target=same_name_target, category=category, status="duplicate_name_skipped")

    old_generic_target = target_dir / source.name
    if dry_run and target_name != source.name and old_generic_target.exists() and old_generic_target.is_file():
        return CopyAction(source=source, target=same_name_target, category=category, status="would_use_renamed_existing")

    if source.suffix.lower() == ".json":
        duplicate = fingerprint_index.find_duplicate(source, target_dir)
        if duplicate:
            return CopyAction(source=source, target=duplicate, category=category, status="duplicate_content_skipped")

    if deep_dedupe:
        source_hash = file_sha256(source)
        existing_hashes = hash_cache.setdefault(target_dir, build_existing_hashes(target_dir))
        if source_hash in existing_hashes:
            return CopyAction(source=source, target=target_dir / target_name, category=category, status="duplicate_content_skipped")
    else:
        existing_hashes = set()

    target = unique_target_path(target_dir, target_name)
    if not dry_run:
        shutil.copy2(source, target)
        fingerprint_index.add(target)
    if deep_dedupe:
        existing_hashes.add(source_hash)
    return CopyAction(source=source, target=target, category=category, status="copied" if not dry_run else "would_copy")


def is_under_named_folder(path: Path, folder_name: str) -> bool:
    return any(part.lower() == folder_name.lower() for part in path.parts)


def target_categories_for_source(source: Path) -> list[tuple[Path, str]]:
    suffix = source.suffix.lower()
    name_lower = source.name.lower()
    categories: list[tuple[Path, str]] = []

    if suffix == ".json":
        categories.append((TARGETS["all_jsons"], "JsonExport"))

        if is_under_named_folder(source, "COLLECTIONS"):
            categories.append((TARGETS["arrears_jsons"], "Arrears_Jsons"))

        if is_under_named_folder(source, "REPAID"):
            categories.append((TARGETS["repaid_jsons"], "Repaid_Jsons"))

    if suffix == ".pdf" and "capital_report" in name_lower:
        categories.append((TARGETS["capital_reports"], "Capitalised_Credit_reports"))

    return categories


def rename_existing_generic_targets(dry_run: bool, verbose: bool) -> list[CopyAction]:
    """Rename already-copied generic JSON files when they match a source file."""
    actions: list[CopyAction] = []

    for source in APPLICATIONS_DIR.rglob("*.json"):
        if not source.is_file() or not is_generic_json_name(source):
            continue

        new_name = target_filename_for_source(source)
        if new_name == source.name:
            continue

        source_hash: str | None = None
        for target_dir, category in target_categories_for_source(source):
            old_target = target_dir / source.name
            if not old_target.exists() or not old_target.is_file():
                continue

            try:
                if old_target.stat().st_size != source.stat().st_size:
                    actions.append(CopyAction(source=old_target, target=target_dir / new_name, category=category, status="rename_ambiguous_skipped"))
                    continue
                if source_hash is None:
                    source_hash = file_sha256(source)
                if file_sha256(old_target) != source_hash:
                    actions.append(CopyAction(source=old_target, target=target_dir / new_name, category=category, status="rename_ambiguous_skipped"))
                    continue
            except OSError:
                actions.append(CopyAction(source=old_target, target=target_dir / new_name, category=category, status="rename_error_skipped"))
                continue

            new_target = unique_target_path(target_dir, new_name)
            if not dry_run:
                old_target.rename(new_target)
            actions.append(
                CopyAction(
                    source=old_target,
                    target=new_target,
                    category=category,
                    status="would_rename_existing" if dry_run else "renamed_existing",
                )
            )
    return actions


def duplicate_keeper_rank(path: Path) -> tuple[int, int, int, int, int, str]:
    name = path.name.lower()
    return (
        1 if "all_json_files" in name else 0,
        1 if is_generic_json_name(path) else 0,
        1 if re.search(r"\(\d+\)(?=\.json$)", path.name, re.IGNORECASE) else 0,
        1 if "categorized" in name else 0,
        1 if "_0,0" in name else 0,
        path.name.lower(),
    )


def tidy_existing_duplicate_jsons(dry_run: bool) -> list[CopyAction]:
    """Move exact duplicate JSONs out of live input folders into an archive."""
    actions: list[CopyAction] = []
    live_targets = [
        (TARGETS["all_jsons"], "JsonExport"),
        (TARGETS["arrears_jsons"], "Arrears_Jsons"),
        (TARGETS["repaid_jsons"], "Repaid_Jsons"),
    ]

    for target_dir, category in live_targets:
        if not target_dir.exists():
            continue

        by_size: dict[int, list[Path]] = {}
        for path in target_dir.glob("*.json"):
            if path.is_file():
                by_size.setdefault(path.stat().st_size, []).append(path)

        for same_size_files in by_size.values():
            if len(same_size_files) < 2:
                continue

            by_hash: dict[str, list[Path]] = {}
            for path in same_size_files:
                try:
                    by_hash.setdefault(file_sha256(path), []).append(path)
                except OSError:
                    continue

            for duplicate_group in by_hash.values():
                if len(duplicate_group) < 2:
                    continue

                keep = sorted(duplicate_group, key=duplicate_keeper_rank)[0]
                archive_dir = DUPLICATE_ARCHIVE_DIR / category
                if not dry_run:
                    archive_dir.mkdir(parents=True, exist_ok=True)

                for duplicate in sorted(duplicate_group, key=duplicate_keeper_rank)[1:]:
                    archive_target = unique_target_path(archive_dir, duplicate.name)
                    if not dry_run:
                        duplicate.rename(archive_target)
                    actions.append(
                        CopyAction(
                            source=duplicate,
                            target=archive_target,
                            category=category,
                            status="would_archive_duplicate" if dry_run else "archived_duplicate",
                        )
                    )

                actions.append(CopyAction(source=keep, target=keep, category=category, status="kept_duplicate_group"))

    return actions


def discover_actions(dry_run: bool, deep_dedupe: bool) -> list[CopyAction]:
    if not APPLICATIONS_DIR.exists():
        raise FileNotFoundError(f"Applications folder not found: {APPLICATIONS_DIR}")

    actions: list[CopyAction] = []
    hash_cache: dict[Path, set[str]] = {}
    fingerprint_index = TargetFingerprintIndex()
    for source in APPLICATIONS_DIR.rglob("*"):
        if not source.is_file():
            continue

        for target_dir, category in target_categories_for_source(source):
            actions.append(copy_if_new(source, target_dir, category, dry_run, hash_cache, deep_dedupe, fingerprint_index))

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
    parser.add_argument(
        "--skip-rename-existing",
        action="store_true",
        help="Do not rename generic JSON files that were already copied into target folders.",
    )
    parser.add_argument(
        "--skip-tidy-existing",
        action="store_true",
        help="Do not move exact duplicate JSONs from live target folders into the duplicate archive.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print one line for every discovered file action.")
    args = parser.parse_args()

    print(f"Source: {APPLICATIONS_DIR}")
    print(f"Target root: {SCORECARD_DEV_DIR}")
    if args.dry_run:
        print("Mode: dry run")
    if args.deep_dedupe:
        print("Duplicate mode: deep content scan")

    actions: list[CopyAction] = []
    if not args.skip_tidy_existing:
        actions.extend(tidy_existing_duplicate_jsons(dry_run=args.dry_run))
    if not args.skip_rename_existing:
        actions.extend(rename_existing_generic_targets(dry_run=args.dry_run, verbose=args.verbose))
    actions.extend(discover_actions(dry_run=args.dry_run, deep_dedupe=args.deep_dedupe))
    print_summary(actions, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
