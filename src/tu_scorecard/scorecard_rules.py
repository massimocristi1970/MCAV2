from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ScoreResult:
    score: int
    decision: str
    reasons: List[str]
    recovery_flags: List[str]


def _cap(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# -----------------------------
# Recovery flag thresholds (configurable via environment variables)
# -----------------------------
# RECOVERY_ACTIVE_ACCOUNTS_Q75: Top quartile threshold for active accounts
# Can be set via environment variable or computed from training data
# Default: 17 (based on initial calibration)
def _get_recovery_threshold() -> int:
    """Get the recovery active accounts threshold from environment or use default."""
    env_val = os.environ.get("TU_RECOVERY_ACTIVE_ACCOUNTS_Q75")
    if env_val is not None:
        try:
            return int(env_val)
        except ValueError:
            pass
    return 17  # Default value from initial calibration


RECOVERY_ACTIVE_ACCOUNTS_Q75 = _get_recovery_threshold()


def score_tu_features(feats: Dict[str, Any]) -> ScoreResult:
    """
    TU-only rule scorecard (v2d).

    Adds two explicit recovery flags (do not auto-approve):
    1) CREDIT_VETERAN_REBOUNDER
    2) SELECTIVE_OPPORTUNIST
    """

    reasons: List[str] = []
    recovery_flags: List[str] = []

    # -----------------
    # Extract core fields safely
    # -----------------
    bureau_score = int(feats.get("bureau_score", 0) or 0)

    accounts_total = int(feats.get("accounts_total", 0) or 0)
    accounts_active = int(feats.get("accounts_active_total", 0) or 0)
    accounts_settled = int(feats.get("accounts_settled_total", 0) or 0)

    opened_6m = int(feats.get("accounts_opened_6m_total", 0) or 0)

    searches_30d = int(feats.get("searches_30d", 0) or 0)
    searches_90d = int(feats.get("searches_90d", 0) or 0)

    missed_3m = int(feats.get("missed_months_3m", 0) or 0)
    missed_6m = int(feats.get("missed_months_6m", 0) or 0)
    missed_12m = int(feats.get("missed_months_12m", 0) or 0)

    worst_ah_12m = int(feats.get("worst_ah_pay_12m", 0) or 0)

    defaults_12m = int(feats.get("defaults_12m_total", 0) or 0)
    delinq_12m = int(feats.get("delinq_12m_total", 0) or 0)

    current_bai_record = bool(feats.get("current_bai_record", False))
    active_iva_or_admin = bool(feats.get("active_iva_or_admin_order", False))
    iva_or_admin_36m = bool(feats.get("iva_or_admin_order_36m", False))
    active_bankruptcy = bool(feats.get("active_bankruptcy_or_sequestration", False))
    bankruptcy_36m = bool(feats.get("bankruptcy_or_sequestration_36m", False))
    active_judgments = int(feats.get("active_judgments_total", feats.get("ccj_active_total", 0)) or 0)

    # Utilisation is noisy; cap hard
    util_raw = float(feats.get("utilisation_pct", 0.0) or 0.0)
    util = _cap(util_raw, 0.0, 500.0)

    # -----------------
    # Recovery flags (do NOT auto-approve)
    # -----------------
    # 1) Credit Veteran / Rebounder
    if accounts_settled >= 25 and missed_12m <= 1:
        recovery_flags.append("CREDIT_VETERAN_REBOUNDER")

    # 2) Selective Opportunist
    if searches_30d >= 2 and accounts_active >= RECOVERY_ACTIVE_ACCOUNTS_Q75 and missed_12m == 0:
        recovery_flags.append("SELECTIVE_OPPORTUNIST")

    # -----------------
    # Hard-stop declines (few and severe)
    # -----------------
    public_record_reasons = feats.get("public_record_reasons") or []

    if active_iva_or_admin or active_bankruptcy or current_bai_record:
        reasons = list(public_record_reasons)
        if not reasons:
            reasons = ["Current active BAI/public insolvency record"]
        return ScoreResult(
            score=0,
            decision="DECLINE",
            reasons=reasons,
            recovery_flags=recovery_flags,
        )

    if iva_or_admin_36m:
        return ScoreResult(
            score=0,
            decision="DECLINE",
            reasons=list(public_record_reasons) or ["IVA/administration order recorded within 36 months"],
            recovery_flags=recovery_flags,
        )

    if bankruptcy_36m:
        return ScoreResult(
            score=0,
            decision="DECLINE",
            reasons=list(public_record_reasons) or ["Bankruptcy/sequestration recorded within 36 months"],
            recovery_flags=recovery_flags,
        )

    if active_judgments > 0:
        return ScoreResult(
            score=0,
            decision="DECLINE",
            reasons=[f"Active judgment/public record count: {active_judgments}"],
            recovery_flags=recovery_flags,
        )

    if worst_ah_12m >= 3:
        return ScoreResult(
            score=0,
            decision="DECLINE",
            reasons=[f"Severe arrears in account history last 12m (worst pay={worst_ah_12m})"],
            recovery_flags=recovery_flags,
        )

    if missed_3m >= 3:
        return ScoreResult(
            score=0,
            decision="DECLINE",
            reasons=[f"Persistent arrears months in last 3m (count={missed_3m})"],
            recovery_flags=recovery_flags,
        )

    # -----------------
    # Score build (0-100)
    # -----------------
    score = 100

    # A) Thin / weak file penalties
    if accounts_total <= 20:
        score -= 18
        reasons.append(f"Thin credit file: accounts_total={accounts_total} (<=20)")
    elif accounts_total <= 31:
        score -= 10
        reasons.append(f"Lower credit file depth: accounts_total={accounts_total} (<=31)")

    if accounts_settled <= 10:
        score -= 18
        reasons.append(f"Low settled-account history: accounts_settled_total={accounts_settled} (<=10)")
    elif accounts_settled <= 16:
        score -= 10
        reasons.append(f"Lower settled-account history: accounts_settled_total={accounts_settled} (<=16)")

    if bureau_score <= 520:
        score -= 16
        reasons.append(f"Low bureau score: {bureau_score} (<=520)")
    elif bureau_score <= 551:
        score -= 9
        reasons.append(f"Below-average bureau score: {bureau_score} (<=551)")

    # B) Recent arrears indicators (penalty unless severe)
    if missed_3m == 2:
        score -= 18
        reasons.append("Arrears months in last 3m: 2")
    elif missed_3m == 1:
        score -= 10
        reasons.append("Arrears months in last 3m: 1")

    if missed_6m >= 3:
        score -= 12
        reasons.append(f"Multiple arrears months in last 6m: {missed_6m}")
    elif missed_6m >= 1:
        score -= 6
        reasons.append(f"Arrears months in last 6m: {missed_6m}")

    if missed_12m >= 4:
        score -= 8
        reasons.append(f"Repeated arrears months in last 12m: {missed_12m}")

    # C) Searches / credit hunger
    if searches_30d >= 3:
        score -= 12
        reasons.append(f"High searches in last 30d: {searches_30d}")
    elif searches_30d >= 1:
        score -= 6
        reasons.append(f"Recent searches in last 30d: {searches_30d}")

    if searches_90d >= 8:
        score -= 8
        reasons.append(f"High searches in last 90d: {searches_90d}")
    elif searches_90d >= 4:
        score -= 4
        reasons.append(f"Moderate searches in last 90d: {searches_90d}")

    # D) Credit velocity (opened accounts)
    if opened_6m >= 12:
        score -= 8
        reasons.append(f"Very high accounts opened in last 6m: {opened_6m}")
    elif opened_6m >= 6:
        score -= 4
        reasons.append(f"Elevated accounts opened in last 6m: {opened_6m}")

    # E) Defaults / delinquency counts (incremental only)
    if defaults_12m >= 2:
        score -= 10
        reasons.append(f"Multiple defaults recorded in last 12m: {defaults_12m}")
    elif defaults_12m == 1:
        score -= 5
        reasons.append("Default recorded in last 12m: 1")

    if delinq_12m >= 3:
        score -= 6
        reasons.append(f"Multiple delinquencies in last 12m: {delinq_12m}")
    elif delinq_12m >= 1:
        score -= 3
        reasons.append(f"Delinquencies in last 12m: {delinq_12m}")

    # F) Utilisation (noisy) — light penalty only
    if util >= 200:
        score -= 4
        reasons.append(f"Very high utilisation (capped): {util:.0f}%")
    elif util >= 120:
        score -= 2
        reasons.append(f"High utilisation (capped): {util:.0f}%")

    score = max(0, min(100, int(score)))

    # Decision bands
    if score >= 60:
        decision = "APPROVE"
    elif score >= 45:
        decision = "REFER"
    else:
        decision = "DECLINE"

    if not reasons:
        reasons.append("No adverse bureau indicators triggered (v2d ruleset)")

    return ScoreResult(score=score, decision=decision, reasons=reasons, recovery_flags=recovery_flags)
