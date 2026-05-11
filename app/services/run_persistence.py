"""
Persist completed scorecard runs under data/app_runs for monitoring and modelling.

Stores JSON snapshots (no raw PII blobs beyond what is already in metrics/params).
Enable/disable via SAVE_APP_RUNS env (default: true).
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

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
