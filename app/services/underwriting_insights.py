"""
Underwriting support: data quality gates, advance capacity, alerts, decision caps, lender stacking.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app.services.ensemble_scorer import EnsembleScorer
from app.services.loans_analysis import analyze_loans_and_repayments


TIER_MAX_MULTIPLES = {
    "Tier 1": 4.0,
    "Tier 2": 3.0,
    "Tier 3": 2.5,
    "Tier 4": 2.0,
    "Decline": 0.0,
}

DEFAULT_FACTOR_RATES = {
    "Tier 1": 1.28,
    "Tier 2": 1.35,
    "Tier 3": 1.42,
    "Tier 4": 1.50,
    "Decline": 0.0,
}

DEFAULT_TERMS_DAYS = {
    "Tier 1": 180,
    "Tier 2": 150,
    "Tier 3": 120,
    "Tier 4": 90,
    "Decline": 0,
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _tier_key(subprime_tier: str | None) -> str:
    tier = (subprime_tier or "Decline").strip()
    for key in TIER_MAX_MULTIPLES:
        if tier.startswith(key):
            return key
    return "Decline"


def get_calibration_overlays(metrics: dict, params: dict) -> dict[str, Any]:
    """Public wrapper for ensemble calibration overlays used as decision caps."""
    return EnsembleScorer()._get_calibration_risk_overlays(metrics, params)


def assess_data_quality(
    filtered_df: pd.DataFrame,
    metrics: dict,
    params: dict,
    analysis_period: str | int,
) -> dict[str, Any]:
    """Assess whether bank data is sufficient for reliable underwriting."""
    checks: list[dict[str, str]] = []
    fail_count = 0
    warn_count = 0

    txn_count = len(filtered_df) if filtered_df is not None else 0
    history_months = int(metrics.get("OB History Months") or 0)
    revenue_days = int(metrics.get("total_revenue_days") or 0)

    total_revenue = _safe_float(metrics.get("Total Revenue") or metrics.get("OB True Revenue"))
    monthly_revenue = _safe_float(metrics.get("Monthly Average Revenue"))

    min_txns = 30 if str(analysis_period) != "3" else 15
    min_months = 3 if str(analysis_period) == "All" else max(2, min(int(analysis_period) if str(analysis_period).isdigit() else 3, 3) - 1)

    if txn_count < min_txns:
        fail_count += 1
        checks.append(
            {
                "status": "Fail",
                "check": "Transaction volume",
                "detail": f"{txn_count} rows in period (need at least {min_txns})",
            }
        )
    else:
        checks.append(
            {
                "status": "Pass",
                "check": "Transaction volume",
                "detail": f"{txn_count} rows in selected period",
            }
        )

    if history_months < min_months:
        fail_count += 1
        checks.append(
            {
                "status": "Fail",
                "check": "History depth",
                "detail": f"{history_months} months covered (need at least {min_months})",
            }
        )
    else:
        checks.append(
            {
                "status": "Pass",
                "check": "History depth",
                "detail": f"{history_months} months of bank history",
            }
        )

    if total_revenue <= 0 or monthly_revenue <= 0:
        fail_count += 1
        checks.append(
            {
                "status": "Fail",
                "check": "Identified revenue",
                "detail": "No meaningful trading revenue detected after categorisation",
            }
        )
    else:
        checks.append(
            {
                "status": "Pass",
                "check": "Identified revenue",
                "detail": f"£{total_revenue:,.0f} total / £{monthly_revenue:,.0f} monthly average",
            }
        )

    stale_days = None
    if filtered_df is not None and not filtered_df.empty and "date" in filtered_df.columns:
        latest = pd.to_datetime(filtered_df["date"]).max()
        if getattr(latest, "tzinfo", None) is not None:
            latest = latest.tz_localize(None)
        stale_days = int((pd.Timestamp.now().normalize() - latest.normalize()).days)
        if stale_days > 45:
            warn_count += 1
            checks.append(
                {
                    "status": "Warn",
                    "check": "Data freshness",
                    "detail": f"Latest transaction is {stale_days} days old",
                }
            )
        else:
            checks.append(
                {
                    "status": "Pass",
                    "check": "Data freshness",
                    "detail": f"Latest transaction {stale_days} days ago",
                }
            )

    ob_txn_count = int(metrics.get("OB Transaction Count") or txn_count)
    if ob_txn_count < 20:
        warn_count += 1
        checks.append(
            {
                "status": "Warn",
                "check": "Sparse file",
                "detail": f"Only {ob_txn_count} categorised transactions — metrics may be volatile",
            }
        )

    if params.get("tu_parse_status") != "parsed" and params.get("tu_director_score") is None:
        fail_count += 1
        checks.append(
            {
                "status": "Fail",
                "check": "Director TU XML",
                "detail": "Valid director TransUnion XML is required",
            }
        )
    else:
        checks.append(
            {
                "status": "Pass",
                "check": "Director TU XML",
                "detail": f"TU score {params.get('tu_director_score')} / decision {params.get('tu_director_decision')}",
            }
        )

    if fail_count > 0:
        gate_action = "refer"
        summary = "Insufficient data quality — decision capped to REFER for manual review"
        overall = "Fail"
    elif warn_count > 0:
        gate_action = "none"
        summary = "Data quality acceptable with warnings — review before approving"
        overall = "Warn"
    else:
        gate_action = "none"
        summary = "Data quality acceptable for automated scoring"
        overall = "Pass"

    return {
        "overall": overall,
        "gate_action": gate_action,
        "summary": summary,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "checks": checks,
        "stale_days": stale_days,
    }


def apply_data_quality_gate(scores: dict, params: dict, data_quality: dict) -> None:
    """Cap approve paths when data quality fails."""
    if data_quality.get("gate_action") != "refer":
        return

    current = str(scores.get("final_decision") or "").upper()
    if current in ("APPROVE", "CONDITIONAL_APPROVE"):
        scores["final_decision"] = "REFER"
        reasons = list(scores.get("final_decision_reasons") or [])
        reasons.append(f"Data quality gate: {data_quality.get('summary')}")
        scores["final_decision_reasons"] = reasons
        params["final_decision"] = scores["final_decision"]
        params["final_decision_reasons"] = reasons


def build_advance_holdback_guidance(metrics: dict, params: dict, scores: dict) -> dict[str, Any]:
    """Suggested advance capacity and illustrative daily holdback."""
    monthly = _safe_float(metrics.get("Monthly Average Revenue"))
    weakest = _safe_float(metrics.get("OB Weakest Month Revenue"), monthly)
    requested = _safe_float(params.get("requested_loan"))
    tier = _tier_key(scores.get("subprime_tier"))
    multiple = TIER_MAX_MULTIPLES.get(tier, 0.0)
    factor = DEFAULT_FACTOR_RATES.get(tier, 0.0)
    term_days = DEFAULT_TERMS_DAYS.get(tier, 0)

    capacity_avg = monthly * multiple if monthly and multiple else 0.0
    capacity_weakest = weakest * multiple if weakest and multiple else 0.0
    recommended_max = min(capacity_avg, capacity_weakest) if capacity_avg and capacity_weakest else max(capacity_avg, capacity_weakest)

    multiples = []
    for label, base in (("1.0× monthly", monthly), ("1.2× monthly", monthly * 1.2), (f"{multiple:.1f}× monthly (tier max)", capacity_avg)):
        if base > 0:
            multiples.append({"label": label, "amount": round(base, 2)})

    if requested > 0 and monthly > 0:
        vs_monthly = requested / monthly
        vs_weakest = requested / weakest if weakest else None
    else:
        vs_monthly = None
        vs_weakest = None

    if recommended_max <= 0:
        request_status = "unknown"
    elif requested <= 0:
        request_status = "not_set"
    elif requested <= recommended_max * 0.85:
        request_status = "within_capacity"
    elif requested <= recommended_max:
        request_status = "at_ceiling"
    else:
        request_status = "above_capacity"

    trading_days = 22.0
    daily_revenue = monthly / trading_days if monthly else 0.0
    holdback_pct = None
    if requested > 0 and daily_revenue > 0 and factor > 0 and term_days > 0:
        total_repay = requested * factor
        daily_repay = total_repay / term_days
        holdback_pct = round((daily_repay / daily_revenue) * 100, 1)

    return {
        "tier": tier,
        "monthly_average_revenue": round(monthly, 2),
        "weakest_month_revenue": round(weakest, 2),
        "requested_loan": round(requested, 2),
        "tier_max_multiple": multiple,
        "recommended_max_advance": round(recommended_max, 2),
        "capacity_vs_monthly": round(capacity_avg, 2),
        "capacity_vs_weakest": round(capacity_weakest, 2),
        "advance_scenarios": multiples,
        "requested_vs_monthly": round(vs_monthly, 2) if vs_monthly is not None else None,
        "requested_vs_weakest": round(vs_weakest, 2) if vs_weakest is not None else None,
        "request_status": request_status,
        "illustrative_factor_rate": factor,
        "illustrative_term_days": term_days,
        "illustrative_holdback_pct_of_revenue": holdback_pct,
        "pricing_from_ensemble": (scores.get("ensemble") or {}).get("pricing_guidance") or scores.get("subprime_pricing"),
    }


def build_lender_stacking_rows(
    loans_analysis: dict,
    manual_debt_balances: dict | None = None,
) -> list[dict[str, Any]]:
    """Combine visible lenders, repayment recipients, and entered balances."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_row(name: str, source: str, borrowed: float, repaid: float, outstanding: float, notes: str):
        key = name.strip().lower()
        if not key or key in seen:
            return
        seen.add(key)
        rows.append(
            {
                "lender_or_counterparty": name,
                "source": source,
                "loan_credits": round(borrowed, 2),
                "repayments_seen": round(repaid, 2),
                "outstanding_or_entered": round(outstanding, 2),
                "notes": notes,
            }
        )

    loans_by_lender = loans_analysis.get("loans_by_lender")
    if isinstance(loans_by_lender, pd.DataFrame):
        for _, row in loans_by_lender.iterrows():
            name = str(row.get("lender_clean", "")).title() or "Unknown lender"
            add_row(
                name,
                "Visible loan credit",
                _safe_float(row.get("sum")),
                0.0,
                0.0,
                f"{int(row.get('count', 0))} loan credit(s) in bank data",
            )

    possible = loans_analysis.get("possible_lenders_from_repayments")
    if isinstance(possible, pd.DataFrame):
        for _, row in possible.iterrows():
            name = str(row.get("possible_lender") or row.get("recipient_clean", "")).title()
            seen_flag = bool(row.get("loan_credit_seen"))
            add_row(
                name,
                "Repayment recipient",
                0.0,
                _safe_float(row.get("total_repaid_in_period")),
                0.0,
                "Matching loan credit seen" if seen_flag else "Repayments without matching loan credit in period",
            )

    for label, amount in (manual_debt_balances or {}).items():
        val = _safe_float(amount)
        if val > 0:
            add_row(str(label), "Underwriter entered balance", 0.0, 0.0, val, "Confirmed outstanding balance")

    if not rows and _safe_float(loans_analysis.get("manual_outstanding_debt")) <= 0:
        rows.append(
            {
                "lender_or_counterparty": "—",
                "source": "Analysis",
                "loan_credits": 0.0,
                "repayments_seen": 0.0,
                "outstanding_or_entered": 0.0,
                "notes": "No external lenders or manual balances identified",
            }
        )

    return sorted(rows, key=lambda r: r.get("outstanding_or_entered", 0) + r.get("loan_credits", 0), reverse=True)


def build_underwriting_alerts(
    metrics: dict,
    params: dict,
    scores: dict,
    filtered_df: pd.DataFrame | None,
    loans_analysis: dict | None = None,
) -> list[dict[str, str]]:
    """Rule-based underwriting alerts from existing metrics."""
    alerts: list[dict[str, str]] = []
    loans = loans_analysis or {}

    def add(severity: str, alert: str, detail: str):
        alerts.append({"severity": severity, "alert": alert, "detail": detail})

    concentration = _safe_float(metrics.get("OB Top Revenue Source Percentage"))
    if concentration >= 70:
        add("High", "Revenue concentration", f"Largest source is {concentration:.1f}% of revenue")

    non_rev = _safe_float(metrics.get("OB Non-Revenue Inflow Ratio"))
    if non_rev >= 0.35:
        add("High", "High non-revenue inflows", f"{non_rev * 100:.1f}% of inflows are non-trading")

    if loans.get("repayments_without_visible_loan"):
        add("High", "Hidden borrowing risk", "Repayments found without matching loan credits in this bank data")

    manual_debt = _safe_float(loans.get("manual_outstanding_debt"))
    if manual_debt > 0 and int(loans.get("loan_count", 0) or 0) == 0:
        add("Medium", "Manual debt only", f"£{manual_debt:,.0f} outstanding balance entered without visible loan credits")

    card_layer = metrics.get("Card Processing Insight Layer")
    if card_layer == "Not available":
        card_share = _safe_float(metrics.get("OB Card Processor Revenue Share"))
        if card_share >= 40:
            add("Medium", "Card statements missing", f"{card_share:.1f}% card-processor revenue share — upload terminal statements")
    elif metrics.get("Card Reconciliation Quality") in ("Poor", "Weak"):
        add("High", "Card vs bank mismatch", str(metrics.get("Card Reconciliation Quality")))

    if params.get("business_ccj"):
        count = params.get("business_ccj_count")
        add("High", "Business CCJ", f"{'Multiple CCJs' if count and count >= 2 else 'CCJ'} on business bureau")

    tu_decision = str(params.get("tu_director_decision") or "").upper()
    ensemble_decision = str((scores.get("ensemble") or {}).get("decision") or "").upper()
    if tu_decision == "DECLINE" and ensemble_decision == "APPROVE":
        add("High", "TU / engine disagreement", "Ensemble approves but director TU is DECLINE")

    if params.get("primary_account_assessment", {}).get("is_potential_non_primary"):
        add("Medium", "Primary account doubt", params["primary_account_assessment"].get("note", "Review account selection"))

    dscr = _safe_float(metrics.get("Debt Service Coverage Ratio"))
    if 0 < dscr < 1.0:
        add("High", "Weak debt service cover", f"DSCR {dscr:.2f}")

    repayment_burden = _safe_float(metrics.get("OB Debt Repayment Burden"))
    if repayment_burden >= 0.25:
        add("Medium", "Heavy debt repayments", f"Repayments are {repayment_burden * 100:.1f}% of revenue")

    recent_loans = _safe_float(metrics.get("OB Recent Loan Credits 30D"))
    if recent_loans > 0:
        add("Medium", "Recent funding inflows", f"£{recent_loans:,.0f} loan/funding credits in last 30 days")

    subprime = _safe_float(scores.get("subprime_score"))
    mca = _safe_float(scores.get("mca_rule_score") or params.get("mca_rule_score"))
    if subprime and mca and abs(subprime - mca) >= 25:
        add("Medium", "Score divergence", f"Subprime {subprime:.0f} vs MCA rule {mca:.0f}")

    return alerts


def build_decision_caps_detail(metrics: dict, params: dict, scores: dict) -> dict[str, Any]:
    """Explain overlays and gates that may cap the automated decision."""
    overlays = get_calibration_overlays(metrics, params)
    caps: list[dict[str, str]] = []

    for item in overlays.get("items", []):
        caps.append(
            {
                "type": "Calibration overlay",
                "severity": str(item.get("severity", "")).title(),
                "label": str(item.get("label", "")),
                "detail": f"Value {item.get('value')} vs threshold {item.get('threshold')}",
                "effect": "Decline overlay" if overlays.get("decline_overlay") else ("Refer cap" if overlays.get("refer_overlay") else "Advisory"),
            }
        )

    if overlays.get("decline_overlay"):
        caps.insert(
            0,
            {
                "type": "Calibration summary",
                "severity": "Severe",
                "label": "Decline overlay active",
                "detail": f"{overlays.get('severe_count', 0)} severe / {overlays.get('total_count', 0)} total overlays",
                "effect": "Can force DECLINE",
            },
        )
    elif overlays.get("refer_overlay"):
        caps.insert(
            0,
            {
                "type": "Calibration summary",
                "severity": "Moderate",
                "label": "Refer overlay active",
                "detail": f"{overlays.get('moderate_count', 0)} moderate / {overlays.get('total_count', 0)} total overlays",
                "effect": "Caps APPROVE to REFER",
            },
        )

    subprime = _safe_float(scores.get("subprime_score"))
    if subprime and subprime < 60:
        caps.append(
            {
                "type": "Subprime gate",
                "severity": "Severe",
                "label": "Subprime below 60",
                "detail": f"Subprime score {subprime:.1f}",
                "effect": "DECLINE",
            }
        )
    elif subprime and subprime < 65:
        caps.append(
            {
                "type": "Subprime gate",
                "severity": "Moderate",
                "label": "Subprime below 65",
                "detail": f"Subprime score {subprime:.1f}",
                "effect": "Cannot fully APPROVE",
            }
        )

    mca_decision = str(scores.get("mca_rule_decision") or params.get("mca_rule_decision") or "").upper()
    if mca_decision == "DECLINE":
        caps.append(
            {
                "type": "MCA rule cap",
                "severity": "Moderate",
                "label": "MCA rule DECLINE",
                "detail": f"MCA score {scores.get('mca_rule_score')}",
                "effect": "Caps maximum decision",
            }
        )
    elif mca_decision == "REFER":
        caps.append(
            {
                "type": "MCA rule cap",
                "severity": "Moderate",
                "label": "MCA rule REFER",
                "detail": f"MCA score {scores.get('mca_rule_score')}",
                "effect": "Caps maximum decision",
            }
        )

    dq = params.get("data_quality") or {}
    if dq.get("gate_action") == "refer":
        caps.append(
            {
                "type": "Data quality gate",
                "severity": "Severe",
                "label": "Insufficient bank data",
                "detail": dq.get("summary", ""),
                "effect": "Caps APPROVE to REFER",
            }
        )

    return {
        "calibration_overlays": overlays,
        "decision_caps": caps,
        "has_active_caps": len(caps) > 0,
    }


def _compute_loans_for_package(
    filtered_df: pd.DataFrame | None,
    analysis_period: str | int,
    manual_debt_balances: dict | None,
) -> dict:
    from app.services.dashboard_export import apply_manual_debt_to_loans_analysis, filter_data_by_period as export_filter

    if filtered_df is None or filtered_df.empty:
        return {}
    period_df = export_filter(filtered_df, analysis_period)
    analysis = analyze_loans_and_repayments(period_df)
    if not analysis:
        return {}
    return apply_manual_debt_to_loans_analysis(analysis, manual_debt_balances)


def build_underwriting_package(
    *,
    metrics: dict,
    params: dict,
    scores: dict,
    filtered_df: pd.DataFrame | None,
    analysis_period: str | int,
    manual_debt_balances: dict | None = None,
) -> dict[str, Any]:
    """Build the full underwriting support payload for UI and export."""
    loans_analysis = _compute_loans_for_package(filtered_df, analysis_period, manual_debt_balances)
    data_quality = assess_data_quality(filtered_df, metrics, params, analysis_period)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_quality": data_quality,
        "advance_holdback": build_advance_holdback_guidance(metrics, params, scores),
        "underwriting_alerts": build_underwriting_alerts(metrics, params, scores, filtered_df, loans_analysis),
        "decision_caps": build_decision_caps_detail(metrics, params, scores),
        "lender_stacking": build_lender_stacking_rows(loans_analysis, manual_debt_balances),
        "loans_analysis_snapshot": {
            k: v for k, v in loans_analysis.items() if not isinstance(v, pd.DataFrame)
        },
    }


def render_underwriting_workspace(
    underwriting: dict[str, Any],
    *,
    expanded_alerts: bool = True,
) -> None:
    """Render the underwriting workspace section in Streamlit."""
    import streamlit as st

    if not underwriting:
        return

    st.markdown("---")
    st.subheader("Underwriting workspace")

    dq = underwriting.get("data_quality") or {}
    overall = dq.get("overall", "Pass")
    if overall == "Fail":
        st.error(f"**Data quality:** {dq.get('summary')}")
    elif overall == "Warn":
        st.warning(f"**Data quality:** {dq.get('summary')}")
    else:
        st.success(f"**Data quality:** {dq.get('summary')}")

    with st.expander("Data quality checks", expanded=overall != "Pass"):
        st.dataframe(pd.DataFrame(dq.get("checks", [])), use_container_width=True, hide_index=True)

    advance = underwriting.get("advance_holdback") or {}
    st.markdown("#### Suggested advance & holdback")
    adv_cols = st.columns(4)
    with adv_cols[0]:
        st.metric("Recommended max", f"£{advance.get('recommended_max_advance', 0):,.0f}")
    with adv_cols[1]:
        st.metric("Requested", f"£{advance.get('requested_loan', 0):,.0f}")
    with adv_cols[2]:
        vs_m = advance.get("requested_vs_monthly")
        st.metric("Requested / monthly", f"{vs_m:.2f}×" if vs_m is not None else "N/A")
    with adv_cols[3]:
        hb = advance.get("illustrative_holdback_pct_of_revenue")
        st.metric("Illustrative holdback", f"{hb:.1f}% of daily revenue" if hb is not None else "N/A")

    status = advance.get("request_status")
    if status == "within_capacity":
        st.success("Requested amount is within tier-based capacity.")
    elif status == "at_ceiling":
        st.warning("Requested amount is at the top of tier-based capacity — review carefully.")
    elif status == "above_capacity":
        st.error("Requested amount exceeds tier-based capacity — consider a lower advance or REFER.")
    elif status == "not_set":
        st.info("Set a requested loan amount in the sidebar to compare against capacity.")

    if advance.get("advance_scenarios"):
        st.caption("Advance scenarios")
        st.dataframe(pd.DataFrame(advance["advance_scenarios"]), use_container_width=True, hide_index=True)

    alerts = underwriting.get("underwriting_alerts") or []
    with st.expander(f"Underwriting alerts ({len(alerts)})", expanded=expanded_alerts and bool(alerts)):
        if alerts:
            st.dataframe(pd.DataFrame(alerts), use_container_width=True, hide_index=True)
        else:
            st.info("No automated alerts — still apply standard manual review.")

    caps = (underwriting.get("decision_caps") or {}).get("decision_caps") or []
    with st.expander(f"Decision caps & overlays ({len(caps)})", expanded=bool(caps)):
        if caps:
            st.dataframe(pd.DataFrame(caps), use_container_width=True, hide_index=True)
        else:
            st.info("No active calibration overlays or hard gates beyond the headline decision.")

    lenders = underwriting.get("lender_stacking") or []
    with st.expander("Lender & stacking view", expanded=True):
        st.caption("Visible loan credits, repayment recipients, and underwriter-entered balances.")
        st.dataframe(pd.DataFrame(lenders), use_container_width=True, hide_index=True)
