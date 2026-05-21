"""
Persist completed scorecard runs under data/app_runs for monitoring and modelling.

Stores JSON snapshots (no raw PII blobs beyond what is already in metrics/params).
Enable/disable via SAVE_APP_RUNS env (default: true).
"""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from app.config.settings import settings


def _slug(name: str, max_len: int = 48) -> str:
    s = re.sub(r"[^\w\s-]", "", (name or "company")).strip().lower()
    s = re.sub(r"[\s]+", "_", s)
    return (s or "company")[:max_len]


def _jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, str, int, float)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            return float(obj)
    return str(obj)


def runs_root() -> Path:
    d = settings.BASE_DIR / "data" / "app_runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def reloadable_runs_default_root() -> Path:
    """Default folder for user-reloadable app runs."""
    configured = getattr(settings, "APP_SAVED_RUNS_DIR", None)
    d = Path(configured).expanduser() if configured else settings.BASE_DIR / "data" / "app_saved_runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def choose_directory(*, title: str, initial_dir: Optional[str] = None) -> Optional[str]:
    """Open a native folder picker when Streamlit is running locally on desktop."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(
            title=title,
            initialdir=initial_dir or str(reloadable_runs_default_root()),
            mustexist=True,
        )
        root.destroy()
        return selected or None
    except Exception:
        return None


def _safe_run_folder_name(run_name: str, company_name: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = _slug(run_name or company_name or "scorecard_run", max_len=64)
    return f"{ts}_{label}"


def _write_dataframe(path: Path, df: Any) -> None:
    if isinstance(df, pd.DataFrame):
        df.to_json(path, orient="table", date_format="iso", indent=2, default_handler=str)
    else:
        pd.DataFrame().to_json(path, orient="table", date_format="iso", indent=2)


def _read_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_json(path, orient="table")


def save_reloadable_scorecard_run(
    *,
    run: Dict[str, Any],
    save_root: str | Path,
    run_name: str,
) -> Path:
    """Save a full main-app run so it can be reloaded without reprocessing."""
    root = Path(save_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    company_name = str(run.get("company_name") or run.get("params", {}).get("company_name") or "Company")
    run_dir = root / _safe_run_folder_name(run_name, company_name)
    run_dir.mkdir(parents=True, exist_ok=False)

    _write_dataframe(run_dir / "transactions.json", run.get("df"))
    _write_dataframe(run_dir / "filtered_transactions.json", run.get("filtered_df"))

    payload = {
        "company_name": run.get("company_name"),
        "analysis_period": run.get("analysis_period"),
        "params": _jsonable(run.get("params") or {}),
        "metrics": _jsonable(run.get("metrics") or {}),
        "scores": _jsonable(run.get("scores") or {}),
        "revenue_insights": _jsonable(run.get("revenue_insights") or {}),
        "card_processing_payload": _jsonable(run.get("card_processing_payload") or {}),
        "source_upload_name": run.get("source_upload_name"),
    }
    (run_dir / "run.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "run_name": run_name or company_name,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "app_name": settings.APP_NAME,
        "app_version": settings.APP_VERSION,
        "company_name": company_name,
        "analysis_period": run.get("analysis_period"),
        "source_upload_name": run.get("source_upload_name"),
        "files": {
            "run": "run.json",
            "transactions": "transactions.json",
            "filtered_transactions": "filtered_transactions.json",
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return run_dir


def list_reloadable_scorecard_runs(root: str | Path) -> list[dict[str, Any]]:
    """Return valid saved run manifests under a selected folder, newest first."""
    base = Path(root).expanduser()
    if not base.exists() or not base.is_dir():
        return []

    manifests: list[dict[str, Any]] = []
    for manifest_path in base.glob("*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != 1:
                continue
            run_file = manifest_path.parent / manifest.get("files", {}).get("run", "run.json")
            if not run_file.exists():
                continue
            manifest["path"] = str(manifest_path.parent)
            manifests.append(manifest)
        except Exception:
            continue

    return sorted(manifests, key=lambda row: row.get("saved_at_utc", ""), reverse=True)


def load_reloadable_scorecard_run(manifest_or_path: dict[str, Any] | str | Path) -> Dict[str, Any]:
    """Reconstruct st.session_state['last_run'] from a saved run folder."""
    if isinstance(manifest_or_path, dict):
        run_dir = Path(manifest_or_path["path"])
        manifest = manifest_or_path
    else:
        run_dir = Path(manifest_or_path).expanduser()
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    files = manifest.get("files", {})
    payload_path = run_dir / files.get("run", "run.json")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))

    return {
        "company_name": payload.get("company_name") or manifest.get("company_name"),
        "analysis_period": payload.get("analysis_period") or manifest.get("analysis_period"),
        "df": _read_dataframe(run_dir / files.get("transactions", "transactions.json")),
        "filtered_df": _read_dataframe(run_dir / files.get("filtered_transactions", "filtered_transactions.json")),
        "params": payload.get("params") or {},
        "metrics": payload.get("metrics") or {},
        "scores": payload.get("scores") or {},
        "revenue_insights": payload.get("revenue_insights") or {},
        "card_terminal_files": None,
        "card_processing_payload": payload.get("card_processing_payload") or {},
        "source_upload_name": payload.get("source_upload_name"),
        "loaded_saved_run_path": str(run_dir),
    }


def make_reloadable_run_package(run_dir: str | Path) -> Path:
    """Create a zip package for moving a saved run between machines."""
    source = Path(run_dir)
    archive_base = source.parent / source.name
    archive_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=source))
    return archive_path


def persist_scorecard_run(
    *,
    company_name: str,
    analysis_period: str,
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    scores: Dict[str, Any],
    revenue_insights: Dict[str, Any],
    source_upload_name: Optional[str],
    txn_row_count: int,
    date_min_iso: Optional[str],
    date_max_iso: Optional[str],
) -> Optional[str]:
    """
    Write one JSON snapshot and append a line to runs_index.jsonl.
    Returns path to detail file, or None if disabled / error.
    """
    if not settings.SAVE_APP_RUNS:
        return None

    try:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        slug = _slug(company_name)
        run_id = f"{ts}_{slug}"
        root = runs_root()
        detail_path = root / f"{run_id}.json"

        payload = {
            "run_id": run_id,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "app_name": settings.APP_NAME,
            "app_version": settings.APP_VERSION,
            "company_name": company_name,
            "analysis_period": analysis_period,
            "inputs": {
                "transaction_json_filename": source_upload_name,
                "transaction_row_count": txn_row_count,
                "date_min": date_min_iso,
                "date_max": date_max_iso,
            },
            "params": _jsonable(params),
            "metrics": _jsonable({k: v for k, v in metrics.items() if k != "monthly_summary"}),
            "scores": _jsonable(scores),
            "revenue_insights": _jsonable(revenue_insights),
            "summary": {
                "final_decision": scores.get("final_decision"),
                "subprime_score": scores.get("subprime_score"),
                "mca_rule_score": scores.get("mca_rule_score"),
                "adjusted_ml_score": scores.get("adjusted_ml_score"),
                "subprime_tier": scores.get("subprime_tier"),
            },
        }

        detail_path.write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )

        index_line = {
            "run_id": run_id,
            "saved_at_utc": payload["saved_at_utc"],
            "company_name": company_name,
            "detail_file": detail_path.name,
            "final_decision": scores.get("final_decision"),
            "subprime_score": scores.get("subprime_score"),
            "source_upload_name": source_upload_name,
            "txn_row_count": txn_row_count,
        }
        index_path = root / "runs_index.jsonl"
        with index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(index_line, default=str) + "\n")

        return str(detail_path)
    except Exception:
        return None
