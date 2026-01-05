from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
import math


@dataclass(frozen=True)
class Thresholds:
    """
    Initial thresholds based on your tiny labelled batch.
    These are intentionally simple and adjustable.
    """
    # Core MCA consistency signals
    inflow_days_30d_approve_min: int = 18
    inflow_days_30d_decline_max: int = 8

    max_inflow_gap_days_approve_max: int = 6
    max_inflow_gap_days_decline_min: int = 21

    inflow_cv_approve_max: float = 0.90
    inflow_cv_decline_min: float = 1.30

    # Optional guardrails (soft)
    months_covered_min: int = 2
    txn_count_avg_month_min: float = 40.0


def _is_nan(x: Any) -> bool:
    try:
        return x is None or (isinstance(x, float) and math.isnan(x))
    except Exception:
        return True


def decide_application(features: Dict[str, Any], t: Thresholds = Thresholds()) -> Tuple[str, int, List[str]]:
    """
    Transparent, rule-based decision:
      - DECLINE: clear evidence of sparse/fragile cashflow
      - APPROVE: strong consistency signals + basic sufficiency checks
      - REFER: everything in between (manual review / alternative sizing)

    Returns: (decision, score, reasons)
      decision: "APPROVE" | "REFER" | "DECLINE"
      score: simple points score (0–100)
      reasons: list of triggered rules (for auditability)
    """
    reasons: List[str] = []

    inflow_days_30d = features.get("inflow_days_30d")
    max_gap = features.get("max_inflow_gap_days")
    inflow_cv = features.get("inflow_cv")

    months_covered = features.get("months_covered")
    txn_count_avg_month = features.get("txn_count_avg_month")

    # ---- Data sufficiency (kept soft: REFER not DECLINE) ----
    if _is_nan(months_covered) or int(months_covered) < t.months_covered_min:
        reasons.append(f"Data sufficiency: months_covered<{t.months_covered_min} (REFER)")
    if _is_nan(txn_count_avg_month) or float(txn_count_avg_month) < t.txn_count_avg_month_min:
        reasons.append(f"Low activity: txn_count_avg_month<{t.txn_count_avg_month_min:g} (REFER)")

    # ---- Hard DECLINE rules (core MCA consistency) ----
    # If inflow days are very low OR gaps are very large OR CV very high → fragile revenue
    if not _is_nan(inflow_days_30d) and int(inflow_days_30d) <= t.inflow_days_30d_decline_max:
        return "DECLINE", 10, [f"inflow_days_30d<= {t.inflow_days_30d_decline_max}"]

    if not _is_nan(max_gap) and float(max_gap) >= t.max_inflow_gap_days_decline_min:
        return "DECLINE", 10, [f"max_inflow_gap_days>= {t.max_inflow_gap_days_decline_min}"]

    if not _is_nan(inflow_cv) and float(inflow_cv) >= t.inflow_cv_decline_min:
        return "DECLINE", 10, [f"inflow_cv>= {t.inflow_cv_decline_min:g}"]

    # ---- APPROVE rules (strong consistency) ----
    approve_hits = 0

    if not _is_nan(inflow_days_30d) and int(inflow_days_30d) >= t.inflow_days_30d_approve_min:
        approve_hits += 1
    else:
        reasons.append(f"inflow_days_30d<{t.inflow_days_30d_approve_min} (not strong)")

    if not _is_nan(max_gap) and float(max_gap) <= t.max_inflow_gap_days_approve_max:
        approve_hits += 1
    else:
        reasons.append(f"max_inflow_gap_days>{t.max_inflow_gap_days_approve_max} (not strong)")

    if not _is_nan(inflow_cv) and float(inflow_cv) <= t.inflow_cv_approve_max:
        approve_hits += 1
    else:
        reasons.append(f"inflow_cv>{t.inflow_cv_approve_max:g} (not strong)")

    # Score (simple, explainable points)
    score = 0
    if not _is_nan(inflow_days_30d):
        # 0–40 points
        score += max(0, min(40, int(inflow_days_30d) * 2))
    if not _is_nan(max_gap):
        # 0–30 points (smaller gap = better)
        score += max(0, min(30, int(30 - float(max_gap) * 2)))
    if not _is_nan(inflow_cv):
        # 0–30 points (lower cv = better)
        score += max(0, min(30, int(30 - float(inflow_cv) * 20)))

    # If core signals are strong and there are no sufficiency flags → APPROVE
    has_soft_flags = any("(REFER)" in r for r in reasons)

    if approve_hits == 3 and not has_soft_flags:
        return "APPROVE", min(100, score), ["Strong inflow consistency across all 3 core signals"]

    # Otherwise REFER (review / different terms / request docs)
    return "REFER", min(100, score), reasons


def thresholds_as_dict(t: Thresholds) -> Dict[str, Any]:
    return {
        "inflow_days_30d_approve_min": t.inflow_days_30d_approve_min,
        "inflow_days_30d_decline_max": t.inflow_days_30d_decline_max,
        "max_inflow_gap_days_approve_max": t.max_inflow_gap_days_approve_max,
        "max_inflow_gap_days_decline_min": t.max_inflow_gap_days_decline_min,
        "inflow_cv_approve_max": t.inflow_cv_approve_max,
        "inflow_cv_decline_min": t.inflow_cv_decline_min,
        "months_covered_min": t.months_covered_min,
        "txn_count_avg_month_min": t.txn_count_avg_month_min,
    }
