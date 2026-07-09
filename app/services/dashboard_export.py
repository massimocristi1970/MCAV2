"""
Unified dashboard export: HTML, JSON, CSV, and PDF reports aligned with the live UI.
"""

from __future__ import annotations

import json
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional

import pandas as pd

from app.config.industry_config import get_industry_thresholds
from app.config.settings import settings
from app.services.loans_analysis import analyze_loans_and_repayments

GENERATED_BY = f"{settings.APP_NAME} v{settings.APP_VERSION}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _clean_dict_for_json(data: dict) -> dict:
    clean: dict = {}
    for key, value in data.items():
        if isinstance(value, pd.DataFrame):
            continue
        clean[key] = _jsonable(value)
    return clean


def filter_data_by_period(df: pd.DataFrame, period_months: str | int) -> pd.DataFrame:
    if df is None or df.empty or period_months == "All":
        return df if df is not None else pd.DataFrame()
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    latest_date = out["date"].max()
    start_date = latest_date - pd.DateOffset(months=int(period_months))
    return out[out["date"] >= start_date]


def manual_debt_total_from_balances(balances: dict | None) -> float:
    total = 0.0
    for value in (balances or {}).values():
        try:
            total += max(float(value or 0), 0.0)
        except (TypeError, ValueError):
            continue
    return round(total, 2)


def apply_manual_debt_to_loans_analysis(
    analysis: dict,
    balances: dict | None,
) -> dict:
    """Reflect underwriter-entered balances in loan/repayment export metrics."""
    manual_total = manual_debt_total_from_balances(balances)
    analysis["manual_outstanding_debt"] = manual_total
    if manual_total <= 0:
        analysis["total_known_borrowing"] = analysis.get("total_loans_received", 0)
        analysis["known_outstanding_balance"] = max(float(analysis.get("net_borrowing", 0) or 0), 0.0)
        return analysis

    visible_loans = float(analysis.get("total_loans_received", 0) or 0)
    repayments = float(analysis.get("total_repayments_made", 0) or 0)

    if visible_loans > 0:
        total_known_borrowing = visible_loans + manual_total
    else:
        total_known_borrowing = repayments + manual_total

    analysis["total_known_borrowing"] = round(total_known_borrowing, 2)
    analysis["known_outstanding_balance"] = manual_total
    analysis["net_borrowing"] = manual_total
    analysis["repayment_ratio"] = repayments / total_known_borrowing if total_known_borrowing > 0 else None

    monthly_net = analysis.get("monthly_net_borrowing", pd.DataFrame())
    if isinstance(monthly_net, pd.DataFrame) and monthly_net.empty:
        today_month = pd.Timestamp.today().to_period("M")
        monthly_net = pd.DataFrame(
            [
                {
                    "month": today_month,
                    "month_str": str(today_month),
                    "loans": 0.0,
                    "repayments": 0.0,
                    "manual_balance_adjustment": manual_total,
                    "net_borrowing": manual_total,
                }
            ]
        )
    elif isinstance(monthly_net, pd.DataFrame):
        monthly_net = monthly_net.copy()
        monthly_net["manual_balance_adjustment"] = 0.0
        current_final_position = float(monthly_net["net_borrowing"].sum())
        balance_adjustment = manual_total - current_final_position
        monthly_net.loc[monthly_net.index[0], "manual_balance_adjustment"] = balance_adjustment
        monthly_net["net_borrowing"] = (
            monthly_net["loans"] + monthly_net["manual_balance_adjustment"] - monthly_net["repayments"]
        )

    analysis["monthly_net_borrowing"] = monthly_net
    return analysis


def compute_loans_analysis(
    filtered_df: pd.DataFrame | None,
    analysis_period: str | int,
    manual_debt_balances: dict | None = None,
) -> dict:
    if filtered_df is None or filtered_df.empty:
        return {}
    period_df = filter_data_by_period(filtered_df, analysis_period)
    analysis = analyze_loans_and_repayments(period_df)
    if not analysis:
        return {}
    return apply_manual_debt_to_loans_analysis(analysis, manual_debt_balances)


def _clean_loans_analysis(analysis: dict | None) -> dict:
    if not analysis:
        return {}
    return _clean_dict_for_json(analysis)


# ---------------------------------------------------------------------------
# Insight / evidence builders (shared with dashboard UI)
# ---------------------------------------------------------------------------

def build_open_banking_insight_rows(metrics: dict, params: dict) -> list[dict[str, object]]:
    requested = float(params.get("requested_loan") or 0)
    monthly_revenue = float(metrics.get("Monthly Average Revenue") or 0)
    weakest_revenue = float(metrics.get("OB Weakest Month Revenue") or 0)
    rows = [
        ("Scoring impact", metrics.get("Open Banking Insights Used In Score", "No - analysis/export only"), "New derived fields are displayed/exported only"),
        ("History", f"{metrics.get('OB History Months', 0)} months / {metrics.get('OB Transaction Count', 0)} transactions", "Coverage and file depth"),
        ("True revenue", f"£{float(metrics.get('OB True Revenue', metrics.get('Total Revenue', 0)) or 0):,.0f}", "Revenue after categorisation"),
        ("Non-revenue inflows", f"{float(metrics.get('OB Non-Revenue Inflow Ratio', 0) or 0) * 100:.1f}%", "Transfers, funding injections, loans and other non-trading inflows"),
        ("Revenue concentration", f"{float(metrics.get('OB Top Revenue Source Percentage', 0) or 0):.1f}%", "Largest payer or processor share of revenue"),
        ("Card processor share", f"{float(metrics.get('OB Card Processor Revenue Share', 0) or 0) * 100:.1f}%", "Revenue from recognised card/payment processors"),
        ("Weakest month revenue", f"£{weakest_revenue:,.0f}", "Lowest observed trading revenue month"),
        ("Debt repayment burden", f"{float(metrics.get('OB Debt Repayment Burden', 0) or 0) * 100:.1f}%", "Debt repayments as share of trading revenue"),
        ("Recent loan credits", f"£{float(metrics.get('OB Recent Loan Credits 30D', 0) or 0):,.0f}", "Funding credits in the latest 30 days"),
        ("Low balance days", f"{int(metrics.get('OB Low Balance Days <1000', 0) or 0)} below £1k", "Daily balance pressure"),
        ("Failed payments 30D", int(metrics.get("OB Recent Failed Payments 30D", 0) or 0), "Recent returned or failed payment markers"),
    ]
    if requested > 0:
        rows.append(
            (
                "Requested amount cover",
                f"{requested / monthly_revenue:.2f}x monthly / {requested / weakest_revenue:.2f}x weakest"
                if monthly_revenue and weakest_revenue
                else "N/A",
                "Requested loan versus normal and weakest-month revenue",
            )
        )
    return [{"signal": name, "value": value, "meaning": meaning} for name, value, meaning in rows]


def build_card_processing_insight_rows(metrics: dict) -> list[dict[str, object]]:
    if metrics.get("Card Processing Insight Layer") != "Available":
        return [
            {
                "signal": "Card processing insight layer",
                "value": metrics.get("Card Processing Insight Layer", "Not available"),
                "meaning": "Upload card terminal statements to derive these signals",
            }
        ]

    rows = [
        ("Scoring impact", metrics.get("Card Processing Insights Used In Score", "No - analysis/export only"), "Capped overlay applied to Subprime score when statements are parsed"),
        ("Score overlay", f"{float(metrics.get('Card Processing Score Adjustment', 0) or 0):+.1f} pts", "Bounded card statement impact on score"),
        ("Statements", f"{int(metrics.get('Card Processor Statements Parsed', 0) or 0)} files / {int(metrics.get('Card Processor Months Present', 0) or 0)} months", "Coverage from uploaded processor statements"),
        ("Card sales total", f"£{float(metrics.get('Card Sales Total', 0) or 0):,.0f}", "Gross card sales from statements"),
        ("Average card sales", f"£{float(metrics.get('Card Sales Monthly Average', 0) or 0):,.0f}", "Monthly card sales average"),
        ("Weakest card month", f"£{float(metrics.get('Card Weakest Month Sales', 0) or 0):,.0f}", "Lowest observed card sales month"),
        ("Card sales volatility", f"{float(metrics.get('Card Sales Volatility', 0) or 0) * 100:.1f}%", "Month-to-month card sales stability"),
        ("Latest month drop", f"{float(metrics.get('Card Latest Month Drop Pct', 0) or 0) * 100:.1f}%", "Latest card month versus prior average"),
        ("Refund ratio", f"{float(metrics.get('Card Refund Ratio', 0) or 0) * 100:.1f}%", "Refunds as share of gross card sales"),
        ("Chargeback ratio", f"{float(metrics.get('Card Chargeback Ratio', 0) or 0) * 100:.1f}%", "Chargebacks as share of gross card sales"),
        ("Fee ratio", f"{float(metrics.get('Card Fee Ratio', 0) or 0) * 100:.1f}%", "Processor fees as share of gross card sales"),
        ("Average transaction value", f"£{float(metrics.get('Card Average Transaction Value', 0) or 0):,.2f}", "Gross card sales divided by transaction count"),
        ("Card vs OB revenue", f"{float(metrics.get('Card vs OB Revenue Ratio', 0) or 0) * 100:.1f}%", "Card sales as share of open banking revenue evidence"),
        ("Unmatched card shortfall", f"£{float(metrics.get('Card Unmatched Sales Shortfall', 0) or 0):,.0f} / {float(metrics.get('Card Unmatched Sales Shortfall Pct', 0) or 0) * 100:.1f}%", "Card sales not covered by bank revenue in matching months"),
        ("Reconciliation quality", metrics.get("Card Reconciliation Quality", "N/A"), "Monthly card sales versus bank revenue"),
        ("MCA suitability", metrics.get("Card MCA Suitability", "N/A"), "Initial card-led underwriting view"),
    ]
    concerns = metrics.get("Card Processing Concerns") or []
    positives = metrics.get("Card Processing Positive Signals") or []
    if positives:
        rows.append(("Positive signals", "; ".join(positives[:4]), "Supportive card processor evidence"))
    if concerns:
        rows.append(("Concerns", "; ".join(concerns[:4]), "Items for underwriting review"))
    return [{"signal": name, "value": value, "meaning": meaning} for name, value, meaning in rows]


def build_evidence_quality(
    params: dict,
    scores: dict,
    df: pd.DataFrame | None = None,
) -> list[dict[str, str]]:
    row_count = len(df) if df is not None else 0
    tu_score = params.get("tu_director_score")
    tu_status = params.get("tu_parse_status", "parsed" if tu_score is not None else "missing")
    bureau_band = params.get("bureau_band")
    bureau_status = params.get("business_bureau_parse_status", "parsed" if bureau_band else "missing")
    evidence = [
        {
            "evidence": "Bank transactions",
            "status": "Present" if row_count else "Missing",
            "detail": f"{row_count:,} transaction rows" if row_count else "No transaction JSON processed",
        },
        {
            "evidence": "MCA rule signals",
            "status": "Present" if params.get("mca_rule_score") is not None else "Missing",
            "detail": f"Score {params.get('mca_rule_score')} / {params.get('mca_rule_decision')}" if params.get("mca_rule_score") is not None else "No MCA rule output",
        },
        {
            "evidence": "Director TU XML",
            "status": "Present" if tu_status == "parsed" else ("Failed" if tu_status == "failed" else "Missing"),
            "detail": (
                f"Score {tu_score} / {params.get('tu_director_decision')}"
                if tu_status == "parsed"
                else (params.get("tu_parse_error") or "No director TU XML parsed")
            ),
        },
        {
            "evidence": "Business credit PDF",
            "status": "Present" if bureau_status == "parsed" else ("Failed" if bureau_status == "failed" else "Missing"),
            "detail": (
                f"{bureau_band}"
                if bureau_status == "parsed" and bureau_band
                else (params.get("business_bureau_parse_error") or "No business credit PDF parsed")
            ),
        },
        {
            "evidence": "Business bureau score",
            "status": "Suppressed" if params.get("business_credit_score_suppressed") else ("Present" if params.get("business_credit_score_range") or params.get("business_credit_score") is not None else "Missing"),
            "detail": (
                "Score suppressed by bureau"
                if params.get("business_credit_score_suppressed")
                else (
                    f"Score {params.get('business_credit_score_range')}"
                    if params.get("business_credit_score_range")
                    else (f"Score {params.get('business_credit_score')}" if params.get("business_credit_score") is not None else "No usable bureau score")
                )
            ),
        },
    ]
    if scores.get("ensemble"):
        evidence.append(
            {
                "evidence": "Decision engine",
                "status": "Present",
                "detail": f"Confidence {scores['ensemble'].get('confidence', 0):.0f}%",
            }
        )
    return evidence


def build_score_impact_rows(params: dict, metrics: dict, scores: dict) -> list[dict[str, object]]:
    ensemble = scores.get("ensemble") or {}
    detailed = ensemble.get("detailed_breakdown", {}) or {}
    contributing = ensemble.get("contributing_scores", {}) or {}
    rows: list[dict[str, object]] = []

    raw_combined = detailed.get("raw_combined_score")
    combined = ensemble.get("combined_score")
    if contributing:
        rows.append(
            {
                "component": "MCA rule",
                "value": contributing.get("mca_score", params.get("mca_rule_score")),
                "impact": "60% weighted input",
                "decision_effect": params.get("mca_rule_decision") or "N/A",
            }
        )
        rows.append(
            {
                "component": "Subprime score",
                "value": contributing.get("subprime_score", scores.get("subprime_score")),
                "impact": "40% weighted input",
                "decision_effect": scores.get("subprime_tier") or "N/A",
            }
        )
    if raw_combined is not None:
        rows.append(
            {
                "component": "Raw weighted score",
                "value": round(float(raw_combined), 1),
                "impact": "MCA/Subprime before numeric-gap penalty",
                "decision_effect": ensemble.get("score_convergence", "N/A"),
            }
        )
    if raw_combined is not None and combined is not None:
        penalty = round(float(raw_combined) - float(combined), 1)
        rows.append(
            {
                "component": "Convergence adjustment",
                "value": -penalty,
                "impact": "Penalty for wide numeric gap" if penalty else "No penalty",
                "decision_effect": ensemble.get("score_convergence", "N/A"),
            }
        )

    alignment = ensemble.get("decision_alignment") or detailed.get("decision_alignment")
    if alignment:
        rows.append(
            {
                "component": "Decision alignment",
                "value": ensemble.get("numeric_score_gap") or detailed.get("numeric_score_gap"),
                "impact": ensemble.get("decision_alignment_detail") or detailed.get("decision_alignment_detail"),
                "decision_effect": alignment,
            }
        )

    for penalty in params.get("_applied_risk_penalties", []) or []:
        rows.append(
            {
                "component": "Risk penalty",
                "value": "",
                "impact": penalty,
                "decision_effect": "Included in Subprime score",
            }
        )

    tu_decision = params.get("tu_director_decision")
    if tu_decision:
        rows.append(
            {
                "component": "Director TU overlay",
                "value": params.get("tu_director_score"),
                "impact": f"TU decision {tu_decision}",
                "decision_effect": "May cap approve to refer",
            }
        )

    bureau_band = params.get("bureau_band")
    if bureau_band:
        rows.append(
            {
                "component": "Business credit PDF",
                "value": bureau_band,
                "impact": "; ".join((params.get("bureau_band_reasons") or [])[:3]),
                "decision_effect": "Feeds bureau risk penalties",
            }
        )

    rows.append(
        {
            "component": "Final decision",
            "value": scores.get("final_decision") or ensemble.get("decision"),
            "impact": "After MCA/Subprime decision and TU overlay",
            "decision_effect": "Current displayed recommendation",
        }
    )
    return rows


def build_metrics_threshold_rows(metrics: dict, industry: str) -> list[dict[str, str]]:
    """Detailed financial metrics with industry threshold pass/fail (matches dashboard table)."""
    rows: list[dict[str, str]] = []
    industry_thresholds = get_industry_thresholds(industry)

    for metric, value in metrics.items():
        if metric not in industry_thresholds or metric == "monthly_summary":
            continue
        threshold = industry_thresholds[metric]
        if metric in ["Cash Flow Volatility", "Average Negative Balance Days per Month", "Number of Bounced Payments"]:
            meets_threshold = value <= threshold
            comparison = "≤"
        else:
            meets_threshold = value >= threshold
            comparison = "≥"

        if isinstance(value, float):
            if metric in ["Operating Margin", "Debt-to-Income Ratio", "Expense-to-Revenue Ratio"]:
                formatted_value = f"{value:.3f} ({value * 100:.1f}%)"
            elif metric in ["Revenue Growth Rate"]:
                formatted_value = f"{value:.3f} ({value:.1f}%)"
            else:
                formatted_value = f"{value:.2f}"
        elif isinstance(value, int):
            formatted_value = str(value)
        else:
            formatted_value = str(value)

        rows.append(
            {
                "metric": metric,
                "actual_value": formatted_value,
                "threshold": f"{comparison} {threshold}",
                "status": "Pass" if meets_threshold else "Fail",
            }
        )
    return rows


def build_export_payload(
    *,
    company_name: str,
    params: dict,
    metrics: dict,
    scores: dict,
    analysis_period: str | int,
    revenue_insights: dict,
    filtered_df: pd.DataFrame | None = None,
    manual_debt_balances: dict | None = None,
    export_timestamp: datetime | None = None,
    underwriting: dict | None = None,
) -> dict:
    """Build a complete export payload aligned with the live dashboard."""
    ts = export_timestamp or datetime.now()
    clean_metrics = _clean_dict_for_json(metrics)
    clean_revenue = _clean_dict_for_json(revenue_insights)
    ensemble = scores.get("ensemble") or {}
    loans_raw = compute_loans_analysis(filtered_df, analysis_period, manual_debt_balances)
    txn_df = filtered_df if filtered_df is not None else pd.DataFrame()

    final_decision = scores.get("final_decision") or params.get("final_decision") or ensemble.get("decision")
    mca_rule_decision = scores.get("mca_rule_decision") or params.get("mca_rule_decision")
    mca_rule_reasons = scores.get("mca_rule_reasons") or params.get("mca_rule_reasons") or []
    final_decision_reasons = scores.get("final_decision_reasons") or params.get("final_decision_reasons") or []

    if underwriting is None:
        from app.services.underwriting_insights import build_underwriting_package

        underwriting = build_underwriting_package(
            metrics=metrics,
            params=params,
            scores=scores,
            filtered_df=filtered_df,
            analysis_period=analysis_period,
            manual_debt_balances=manual_debt_balances,
        )

    return {
        "export_info": {
            "company_name": company_name,
            "export_timestamp": ts.isoformat(),
            "analysis_period": str(analysis_period),
            "generated_by": GENERATED_BY,
        },
        "business_parameters": {
            "industry": params.get("industry"),
            "requested_loan": params.get("requested_loan"),
            "directors_score": params.get("directors_score"),
            "company_age_months": params.get("company_age_months"),
            "tu_director_score": params.get("tu_director_score"),
            "tu_director_decision": params.get("tu_director_decision"),
            "tu_director_flags": params.get("tu_director_flags", []),
            "tu_director_reasons": params.get("tu_director_reasons", []),
            "overall_decision": params.get("overall_decision"),
            "mca_main_decision": params.get("mca_main_decision"),
            "primary_account_assessment": params.get("primary_account_assessment"),
            "risk_factors": {
                "business_ccj": params.get("business_ccj", False),
                "poor_or_no_online_presence": params.get("poor_or_no_online_presence", False),
                "uses_generic_email": params.get("uses_generic_email", False),
            },
        },
        "financial_metrics": clean_metrics,
        "scoring_results": {
            "subprime_score": scores.get("subprime_score"),
            "subprime_tier": scores.get("subprime_tier"),
            "subprime_recommendation": scores.get("subprime_recommendation"),
            "mca_rule_score": scores.get("mca_rule_score", params.get("mca_rule_score", 0)),
            "mca_rule_decision": mca_rule_decision,
            "mca_rule_reasons": list(mca_rule_reasons),
            "ml_score": scores.get("ml_score"),
            "adjusted_ml_score": scores.get("adjusted_ml_score"),
            "industry_score": scores.get("industry_score"),
            "loan_risk": scores.get("loan_risk"),
            "primary_account_assessment": scores.get("primary_account_assessment", params.get("primary_account_assessment")),
            "final_decision": final_decision,
            "final_decision_reasons": list(final_decision_reasons),
            "ensemble": _clean_dict_for_json(ensemble) if ensemble else {},
        },
        "revenue_insights": clean_revenue,
        "loans_analysis": _clean_loans_analysis(loans_raw),
        "open_banking_insights": build_open_banking_insight_rows(metrics, params),
        "card_processing_insights": build_card_processing_insight_rows(metrics),
        "score_impact": build_score_impact_rows(params, metrics, scores),
        "evidence_quality": build_evidence_quality(params, scores, txn_df),
        "metrics_thresholds": build_metrics_threshold_rows(metrics, params.get("industry", "")),
        "underwriting": _clean_dict_for_json(underwriting),
    }


def build_export_payload_from_run(run: dict) -> dict:
    """Convenience wrapper for a session ``last_run`` dict."""
    return build_export_payload(
        company_name=run.get("company_name") or run.get("params", {}).get("company_name", "Company"),
        params=run.get("params") or {},
        metrics=run.get("metrics") or {},
        scores=run.get("scores") or {},
        analysis_period=run.get("analysis_period", "All"),
        revenue_insights=run.get("revenue_insights") or {},
        filtered_df=run.get("filtered_df"),
        manual_debt_balances=run.get("manual_outstanding_debt_balances") or {},
        underwriting=run.get("underwriting"),
    )


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _score_class(score) -> str:
    if score is None:
        return "low"
    try:
        score = float(score)
    except (TypeError, ValueError):
        return "low"
    if score >= 70:
        return "high"
    if score >= 40:
        return "medium"
    return "low"


def _format_score(score, suffix="/100", precision=1) -> str:
    if score is None:
        return "N/A"
    try:
        val = float(score)
    except (TypeError, ValueError):
        return "N/A"
    if precision == 0:
        return f"{val:.0f}{suffix}"
    return f"{val:.{precision}f}{suffix}"


def _decision_badge_class(decision: str | None) -> str:
    d = (decision or "").upper()
    if d in ("APPROVE", "CONDITIONAL_APPROVE"):
        return "badge-approve"
    if d == "DECLINE":
        return "badge-decline"
    return "badge-refer"


def _html_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "<p><em>No data available.</em></p>"
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f'<div class="table-responsive"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _underwriting_html_section(uw: dict) -> str:
    """Render underwriting workspace block for HTML export."""
    if not uw:
        return ""

    dq = uw.get("data_quality") or {}
    advance = uw.get("advance_holdback") or {}
    alerts = uw.get("underwriting_alerts") or []
    caps = (uw.get("decision_caps") or {}).get("decision_caps") or []
    lenders = uw.get("lender_stacking") or []

    dq_rows = [[c.get("status", ""), c.get("check", ""), c.get("detail", "")] for c in dq.get("checks", [])]
    alert_rows = [[a.get("severity", ""), a.get("alert", ""), a.get("detail", "")] for a in alerts]
    cap_rows = [[c.get("type", ""), c.get("severity", ""), c.get("label", ""), c.get("effect", "")] for c in caps]
    lender_rows = [
        [
            r.get("lender_or_counterparty", ""),
            r.get("source", ""),
            f"£{float(r.get('outstanding_or_entered') or 0):,.0f}",
            r.get("notes", ""),
        ]
        for r in lenders
    ]

    hb = advance.get("illustrative_holdback_pct_of_revenue")
    hb_display = f"{hb:.1f}% of daily revenue" if hb is not None else "N/A"
    vs_m = advance.get("requested_vs_monthly")
    vs_m_display = f"{vs_m:.2f}×" if vs_m is not None else "N/A"

    return f"""
    <div class="section">
        <h2>Underwriting workspace</h2>
        <p><strong>Data quality ({dq.get('overall', 'N/A')}):</strong> {dq.get('summary', '')}</p>
        <h3>Data quality checks</h3>
        {_html_table(['Status', 'Check', 'Detail'], dq_rows)}
        <h3>Suggested advance &amp; holdback</h3>
        <div class="metric-grid">
            <div class="metric-card"><h4>Recommended max</h4><div>£{float(advance.get('recommended_max_advance') or 0):,.0f}</div></div>
            <div class="metric-card"><h4>Requested</h4><div>£{float(advance.get('requested_loan') or 0):,.0f}</div></div>
            <div class="metric-card"><h4>Requested / monthly</h4><div>{vs_m_display}</div></div>
            <div class="metric-card"><h4>Illustrative holdback</h4><div>{hb_display}</div></div>
        </div>
        <h3>Underwriting alerts</h3>
        {_html_table(['Severity', 'Alert', 'Detail'], alert_rows or [['—', 'None', 'No automated alerts']])}
        <h3>Decision caps &amp; overlays</h3>
        {_html_table(['Type', 'Severity', 'Label', 'Effect'], cap_rows or [['—', '—', 'None', 'No active caps']])}
        <h3>Lender &amp; stacking view</h3>
        {_html_table(['Lender / counterparty', 'Source', 'Outstanding', 'Notes'], lender_rows)}
    </div>"""


def generate_html_report(export_data: dict) -> str:
    """Generate a comprehensive HTML report matching the live dashboard."""
    info = export_data.get("export_info", {})
    bp = export_data.get("business_parameters", {}) or {}
    sr = export_data.get("scoring_results", {}) or {}
    fm = export_data.get("financial_metrics", {}) or {}
    ri = export_data.get("revenue_insights", {}) or {}
    loans = export_data.get("loans_analysis", {}) or {}
    rf = bp.get("risk_factors", {}) or {}
    ensemble = sr.get("ensemble", {}) or {}

    subprime_raw = sr.get("subprime_score")
    mca_raw = sr.get("mca_rule_score")
    adjusted_raw = sr.get("adjusted_ml_score") or sr.get("ml_score")
    final_decision = sr.get("final_decision", "N/A")
    combined_score = ensemble.get("combined_score")
    confidence = ensemble.get("confidence")
    convergence = ensemble.get("score_convergence", "")
    primary_reason = ensemble.get("primary_reason", "")

    ts = datetime.fromisoformat(info["export_timestamp"]).strftime("%B %d, %Y at %I:%M %p")

    loans_section = ""
    if loans and (loans.get("loan_count", 0) > 0 or loans.get("repayment_count", 0) > 0 or loans.get("manual_outstanding_debt", 0) > 0):
        ratio = loans.get("repayment_ratio")
        ratio_display = f"{(ratio or 0) * 100:.1f}%" if ratio is not None else "N/A"
        loans_section = f"""
        <div class="section">
            <h2>Loans &amp; debt analysis</h2>
            <div class="metric-grid">
                <div class="metric-card"><h4>Total known borrowing</h4><div>£{loans.get('total_known_borrowing', loans.get('total_loans_received', 0)):,.0f}</div></div>
                <div class="metric-card"><h4>Total repayments</h4><div>£{loans.get('total_repayments_made', 0):,.0f}</div></div>
                <div class="metric-card"><h4>Outstanding balance</h4><div>£{loans.get('known_outstanding_balance', loans.get('net_borrowing', 0)):,.0f}</div></div>
                <div class="metric-card"><h4>Repayment ratio</h4><div>{ratio_display}</div></div>
            </div>
        </div>"""

    score_impact_rows = [
        [str(r.get("component", "")), str(r.get("value", "")), str(r.get("impact", "")), str(r.get("decision_effect", ""))]
        for r in export_data.get("score_impact", [])
    ]
    evidence_rows = [
        [r.get("evidence", ""), r.get("status", ""), r.get("detail", "")]
        for r in export_data.get("evidence_quality", [])
    ]
    ob_rows = [
        [r.get("signal", ""), str(r.get("value", "")), r.get("meaning", "")]
        for r in export_data.get("open_banking_insights", [])
    ]
    card_rows = [
        [r.get("signal", ""), str(r.get("value", "")), r.get("meaning", "")]
        for r in export_data.get("card_processing_insights", [])
    ]
    threshold_rows = [
        [r.get("metric", ""), r.get("actual_value", ""), r.get("threshold", ""), r.get("status", "")]
        for r in export_data.get("metrics_thresholds", [])
    ]

    pricing = ensemble.get("pricing_guidance", {}) or {}
    pricing_html = ""
    if pricing and pricing.get("factor_rate") not in (None, "N/A"):
        pricing_html = f"""
        <p><strong>Factor rate:</strong> {pricing.get('factor_rate', 'N/A')} &nbsp;|&nbsp;
        <strong>Max term:</strong> {pricing.get('max_term', 'N/A')} &nbsp;|&nbsp;
        <strong>Max amount:</strong> {pricing.get('max_multiple', 'N/A')} &nbsp;|&nbsp;
        <strong>Collection:</strong> {pricing.get('collection_frequency', 'N/A')}</p>"""

    risk_factors = ensemble.get("risk_factors", []) or []
    positive_factors = ensemble.get("positive_factors", []) or []
    recommendations = ensemble.get("recommendations", []) or []

    def _list_html(items: list) -> str:
        if not items:
            return "<p><em>None identified.</em></p>"
        return "<ul>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>"

    mca_reasons_html = "".join(f"<li>{r}</li>" for r in sr.get("mca_rule_reasons", []))
    final_reasons_html = "".join(f"<li>{r}</li>" for r in sr.get("final_decision_reasons", []))
    underwriting_section = _underwriting_html_section(export_data.get("underwriting") or {})

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8"/>
    <title>Business Finance Scorecard — {info.get('company_name', 'Report')}</title>
    <style>
        :root {{
            --ink: #0f172a;
            --accent: #0e7490;
            --accent-deep: #0f766e;
            --muted: #475569;
            --line: #e2e8f0;
            --surface: #f8fafc;
            --approve: #15803d;
            --refer: #b45309;
            --decline: #b91c1c;
        }}
        * {{ box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; margin: 0; color: var(--ink); background: #eef2f7; line-height: 1.55; }}
        .page {{ max-width: 960px; margin: 0 auto; padding: 32px 24px 48px; }}
        .header {{ background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #0f766e 100%); color: #f8fafc; padding: 32px; border-radius: 12px; margin-bottom: 24px; }}
        .header h1 {{ margin: 0 0 8px; font-size: 1.75rem; font-weight: 600; }}
        .header h2 {{ margin: 0 0 16px; font-size: 1.35rem; font-weight: 500; opacity: 0.95; }}
        .header p {{ margin: 4px 0; color: #cbd5e1; font-size: 0.95rem; }}
        .decision-banner {{ padding: 20px 24px; border-radius: 10px; margin-bottom: 24px; background: #fff; border: 1px solid var(--line); }}
        .decision-banner h2 {{ margin: 0 0 8px; font-size: 1.4rem; }}
        .badge {{ display: inline-block; padding: 4px 12px; border-radius: 999px; font-weight: 600; font-size: 0.85rem; }}
        .badge-approve {{ background: #dcfce7; color: var(--approve); }}
        .badge-refer {{ background: #fef3c7; color: var(--refer); }}
        .badge-decline {{ background: #fee2e2; color: var(--decline); }}
        .section {{ background: #fff; border: 1px solid var(--line); border-radius: 10px; padding: 20px 24px; margin-bottom: 20px; }}
        .section h2 {{ margin: 0 0 16px; font-size: 1.15rem; color: var(--accent-deep); border-bottom: 2px solid var(--accent); padding-bottom: 8px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
        .metric-card {{ background: var(--surface); padding: 14px; border-radius: 8px; text-align: center; border: 1px solid var(--line); }}
        .metric-card h3, .metric-card h4 {{ margin: 0 0 6px; font-size: 0.8rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.03em; }}
        .metric-card div {{ font-size: 1.25rem; font-weight: 600; }}
        .score-high {{ color: var(--approve); }}
        .score-medium {{ color: var(--refer); }}
        .score-low {{ color: var(--decline); }}
        .table-responsive {{ overflow-x: auto; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
        th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--line); }}
        th {{ background: var(--surface); font-weight: 600; color: var(--muted); font-size: 0.8rem; text-transform: uppercase; }}
        .footer {{ margin-top: 32px; padding-top: 20px; border-top: 1px solid var(--line); color: var(--muted); font-size: 0.85rem; }}
        @media print {{ body {{ background: #fff; }} .page {{ padding: 0; }} }}
    </style>
</head>
<body>
<div class="page">
    <div class="header">
        <h1>Business Finance Scorecard</h1>
        <h2>{info.get('company_name', '')}</h2>
        <p><strong>Generated:</strong> {ts}</p>
        <p><strong>Analysis period:</strong> {info.get('analysis_period', '')} &nbsp;|&nbsp; <strong>Industry:</strong> {bp.get('industry', 'N/A')}</p>
    </div>

    <div class="decision-banner">
        <h2>Recommendation: <span class="badge {_decision_badge_class(final_decision)}">{final_decision}</span></h2>
        <p><strong>Combined score:</strong> {_format_score(combined_score) if combined_score is not None else 'N/A'}
        &nbsp;|&nbsp; <strong>Confidence:</strong> {f'{confidence:.0f}%' if confidence is not None else 'N/A'}
        &nbsp;|&nbsp; <strong>Convergence:</strong> {convergence or 'N/A'}</p>
        <p><em>{primary_reason}</em></p>
        {pricing_html}
    </div>

    <div class="section">
        <h2>Executive summary</h2>
        <div class="metric-grid">
            <div class="metric-card"><h3>Subprime score</h3><div class="score-{_score_class(subprime_raw)}">{_format_score(subprime_raw)}</div><p>{sr.get('subprime_tier', '')}</p></div>
            <div class="metric-card"><h3>MCA rule (60%)</h3><div class="score-{_score_class(mca_raw)}">{_format_score(mca_raw, precision=0)}</div></div>
            <div class="metric-card"><h3>ML score (info)</h3><div class="score-{_score_class(adjusted_raw)}">{_format_score(adjusted_raw, suffix='%')}</div></div>
            <div class="metric-card"><h3>Requested loan</h3><div>£{float(bp.get('requested_loan') or 0):,.0f}</div><p>{sr.get('loan_risk', '')}</p></div>
        </div>
        <h3>Decision stack</h3>
        {_html_table(['Layer', 'Result'], [
            ['FINAL decision', str(final_decision)],
            ['MCA rule', f"{sr.get('mca_rule_decision', 'N/A')} (score {sr.get('mca_rule_score', 'N/A')})"],
            ['Subprime', f"{sr.get('subprime_recommendation', 'N/A')} (tier {sr.get('subprime_tier', 'N/A')})"],
            ['ML score (info)', _format_score(adjusted_raw, suffix='%')],
        ])}
        <h3>MCA rule reasons</h3>
        <ul>{mca_reasons_html or '<li>None recorded.</li>'}</ul>
        <h3>Final decision path</h3>
        <ul>{final_reasons_html or '<li>None recorded.</li>'}</ul>
    </div>

    {underwriting_section}

    <div class="section">
        <h2>Score impact</h2>
        {_html_table(['Component', 'Value', 'Impact', 'Decision effect'], score_impact_rows)}
    </div>

    <div class="section">
        <h2>Evidence quality</h2>
        {_html_table(['Evidence', 'Status', 'Detail'], evidence_rows)}
    </div>

    <div class="section">
        <h2>Financial performance</h2>
        {_html_table(['Metric', 'Value'], [
            ['Total revenue', f"£{float(fm.get('Total Revenue', 0) or 0):,.2f}"],
            ['Monthly average revenue', f"£{float(fm.get('Monthly Average Revenue', 0) or 0):,.2f}"],
            ['Net income', f"£{float(fm.get('Net Income', 0) or 0):,.2f}"],
            ['Operating margin', f"{float(fm.get('Operating Margin', 0) or 0) * 100:.1f}%"],
            ['Revenue growth rate', f"{float(fm.get('Revenue Growth Rate', 0) or 0) * 100:.1f}%"],
            ['Debt service coverage ratio', f"{float(fm.get('Debt Service Coverage Ratio', 0) or 0):.2f}"],
            ['Cash flow volatility', f"{float(fm.get('Cash Flow Volatility', 0) or 0):.3f}"],
            ['Average month-end balance', f"£{float(fm.get('Average Month-End Balance', 0) or 0):,.2f}"],
        ])}
    </div>

    <div class="section">
        <h2>Detailed metrics vs industry thresholds</h2>
        {_html_table(['Metric', 'Actual', 'Threshold', 'Status'], threshold_rows)}
    </div>

    <div class="section">
        <h2>Revenue insights</h2>
        <div class="metric-grid">
            <div class="metric-card"><h4>Revenue sources</h4><div>{ri.get('unique_revenue_sources', 0)}</div></div>
            <div class="metric-card"><h4>Avg daily revenue</h4><div>£{float(ri.get('avg_daily_revenue_amount', 0) or 0):,.2f}</div></div>
            <div class="metric-card"><h4>Revenue active days</h4><div>{ri.get('total_revenue_days', 0)}</div></div>
            <div class="metric-card"><h4>Transactions/day</h4><div>{float(ri.get('avg_revenue_transactions_per_day', 0) or 0):.1f}</div></div>
        </div>
    </div>

    <div class="section">
        <h2>Open banking derived insights</h2>
        {_html_table(['Signal', 'Value', 'Meaning'], ob_rows)}
    </div>

    <div class="section">
        <h2>Card processor derived insights</h2>
        {_html_table(['Signal', 'Value', 'Meaning'], card_rows)}
    </div>

    {loans_section}

    <div class="section">
        <h2>Risk &amp; positive factors</h2>
        <h3>Risk factors</h3>
        {_list_html(risk_factors)}
        <h3>Positive factors</h3>
        {_list_html(positive_factors)}
        <h3>Recommendations</h3>
        {_list_html(recommendations)}
    </div>

    <div class="section">
        <h2>Business risk flags</h2>
        {_html_table(['Risk factor', 'Status'], [
            ['Business CCJs', 'Yes' if rf.get('business_ccj') else 'No'],
            ['Poor/no online presence', 'Yes' if rf.get('poor_or_no_online_presence') else 'No'],
            ['Generic email', 'Yes' if rf.get('uses_generic_email') else 'No'],
        ])}
    </div>

    <div class="section">
        <h2>Business information</h2>
        {_html_table(['Parameter', 'Value'], [
            ['Company name', info.get('company_name', '')],
            ['Industry', str(bp.get('industry', ''))],
            ['Company age', f"{bp.get('company_age_months', '')} months"],
            ['Directors score', f"{bp.get('directors_score', '')}/100"],
            ['TU director score', str(bp.get('tu_director_score', 'N/A'))],
            ['TU director decision', str(bp.get('tu_director_decision', 'N/A'))],
            ['Requested loan', f"£{float(bp.get('requested_loan') or 0):,.0f}"],
        ])}
    </div>

    <div class="footer">
        <p><strong>Report generated by:</strong> {info.get('generated_by', GENERATED_BY)}</p>
        <p><strong>Disclaimer:</strong> This report is for informational purposes only and should not be considered financial advice.
        All lending decisions should involve comprehensive due diligence and risk assessment.</p>
    </div>
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# PDF report (ReportLab)
# ---------------------------------------------------------------------------

def generate_pdf_report(export_data: dict) -> bytes:
    """Generate a polished multi-page PDF report."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=0.65 * inch,
        rightMargin=0.65 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        title=f"Scorecard — {export_data.get('export_info', {}).get('company_name', 'Report')}",
    )

    styles = getSampleStyleSheet()
    ink = colors.HexColor("#0f172a")
    accent = colors.HexColor("#0e7490")
    muted = colors.HexColor("#475569")
    surface = colors.HexColor("#f8fafc")

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontSize=20,
        textColor=colors.white,
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=12,
        textColor=colors.HexColor("#cbd5e1"),
        spaceAfter=2,
    )
    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=accent,
        spaceBefore=14,
        spaceAfter=8,
        borderPadding=4,
    )
    body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=9.5, textColor=ink, leading=13)
    small_style = ParagraphStyle("Small", parent=body_style, fontSize=8.5, textColor=muted)

    info = export_data.get("export_info", {})
    bp = export_data.get("business_parameters", {}) or {}
    sr = export_data.get("scoring_results", {}) or {}
    fm = export_data.get("financial_metrics", {}) or {}
    ensemble = sr.get("ensemble", {}) or {}
    final_decision = sr.get("final_decision", "N/A")

    def _p(text: str, style=body_style):
        return Paragraph(str(text).replace("&", "&amp;"), style)

    def _section(title: str):
        return _p(title, section_style)

    def _table(headers: list[str], rows: list[list[str]], col_widths: list[float] | None = None):
        data = [headers] + [[str(c) for c in row] for row in rows]
        if not col_widths:
            col_widths = [doc.width / len(headers)] * len(headers)
        t = Table(data, colWidths=col_widths, repeatRows=1)
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), surface),
                    ("TEXTCOLOR", (0, 0), (-1, 0), muted),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, surface]),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        return t

    ts = datetime.fromisoformat(info["export_timestamp"]).strftime("%d %B %Y, %H:%M")
    story: list = []

    header_data = [
        [_p("Business Finance Scorecard", title_style)],
        [_p(info.get("company_name", ""), ParagraphStyle("Co", parent=title_style, fontSize=16))],
        [_p(f"Generated {ts}  |  Period: {info.get('analysis_period', '')}  |  Industry: {bp.get('industry', '')}", subtitle_style)],
    ]
    header_table = Table(header_data, colWidths=[doc.width])
    header_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), ink),
                ("BOX", (0, 0), (-1, -1), 0, ink),
                ("LEFTPADDING", (0, 0), (-1, -1), 16),
                ("RIGHTPADDING", (0, 0), (-1, -1), 16),
                ("TOPPADDING", (0, 0), (-1, -1), 14),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
            ]
        )
    )
    story.append(header_table)
    story.append(Spacer(1, 14))

    combined = ensemble.get("combined_score")
    confidence = ensemble.get("confidence")
    story.append(
        _p(
            f"<b>Recommendation: {final_decision}</b> &nbsp;|&nbsp; "
            f"Combined score: {_format_score(combined)} &nbsp;|&nbsp; "
            f"Confidence: {f'{confidence:.0f}%' if confidence is not None else 'N/A'} &nbsp;|&nbsp; "
            f"Convergence: {ensemble.get('score_convergence', 'N/A')}",
            body_style,
        )
    )
    if ensemble.get("primary_reason"):
        story.append(_p(f"<i>{ensemble.get('primary_reason')}</i>", small_style))
    story.append(Spacer(1, 10))

    story.append(_section("Executive summary"))
    story.append(
        _table(
            ["Metric", "Value"],
            [
                ["Subprime score", f"{_format_score(sr.get('subprime_score'))} ({sr.get('subprime_tier', '')})"],
                ["MCA rule (60%)", _format_score(sr.get("mca_rule_score"), precision=0)],
                ["ML score (info)", _format_score(sr.get("adjusted_ml_score") or sr.get("ml_score"), suffix="%")],
                ["MCA decision", str(sr.get("mca_rule_decision", "N/A"))],
                ["Subprime recommendation", str(sr.get("subprime_recommendation", "N/A"))],
                ["Requested loan", f"£{float(bp.get('requested_loan') or 0):,.0f}"],
                ["Loan risk", str(sr.get("loan_risk", "N/A"))],
            ],
            [2.2 * inch, doc.width - 2.2 * inch],
        )
    )
    story.append(Spacer(1, 8))

    if sr.get("mca_rule_reasons"):
        story.append(_p("<b>MCA rule reasons:</b>", body_style))
        for reason in sr.get("mca_rule_reasons", []):
            story.append(_p(f"• {reason}", small_style))

    uw = export_data.get("underwriting") or {}
    if uw:
        story.append(_section("Underwriting workspace"))
        dq = uw.get("data_quality") or {}
        story.append(_p(f"<b>Data quality ({dq.get('overall', 'N/A')}):</b> {dq.get('summary', '')}", body_style))
        dq_rows = [[c.get("status", ""), c.get("check", ""), c.get("detail", "")] for c in dq.get("checks", [])]
        if dq_rows:
            story.append(_table(["Status", "Check", "Detail"], dq_rows, [0.7 * inch, 1.3 * inch, doc.width - 2.0 * inch]))
        advance = uw.get("advance_holdback") or {}
        story.append(
            _table(
                ["Metric", "Value"],
                [
                    ["Recommended max advance", f"£{float(advance.get('recommended_max_advance') or 0):,.0f}"],
                    ["Requested loan", f"£{float(advance.get('requested_loan') or 0):,.0f}"],
                    ["Tier max multiple", str(advance.get("tier_max_multiple", "N/A"))],
                    [
                        "Illustrative holdback",
                        f"{advance.get('illustrative_holdback_pct_of_revenue'):.1f}%"
                        if advance.get("illustrative_holdback_pct_of_revenue") is not None
                        else "N/A",
                    ],
                ],
                [2.5 * inch, doc.width - 2.5 * inch],
            )
        )
        alerts = uw.get("underwriting_alerts") or []
        if alerts:
            story.append(_p("<b>Underwriting alerts</b>", body_style))
            story.append(
                _table(
                    ["Severity", "Alert", "Detail"],
                    [[a.get("severity", ""), a.get("alert", ""), a.get("detail", "")] for a in alerts],
                    [0.8 * inch, 1.5 * inch, doc.width - 2.3 * inch],
                )
            )
        caps = (uw.get("decision_caps") or {}).get("decision_caps") or []
        if caps:
            story.append(_p("<b>Decision caps</b>", body_style))
            story.append(
                _table(
                    ["Type", "Severity", "Label", "Effect"],
                    [[c.get("type", ""), c.get("severity", ""), c.get("label", ""), c.get("effect", "")] for c in caps],
                )
            )
        lenders = uw.get("lender_stacking") or []
        if lenders:
            story.append(_p("<b>Lender stacking</b>", body_style))
            story.append(
                _table(
                    ["Lender", "Source", "Outstanding", "Notes"],
                    [
                        [
                            r.get("lender_or_counterparty", ""),
                            r.get("source", ""),
                            f"£{float(r.get('outstanding_or_entered') or 0):,.0f}",
                            r.get("notes", ""),
                        ]
                        for r in lenders
                    ],
                    [1.4 * inch, 1.2 * inch, 1.0 * inch, doc.width - 3.6 * inch],
                )
            )

    story.append(_section("Score impact"))
    impact_rows = [
        [str(r.get("component", "")), str(r.get("value", "")), str(r.get("impact", "")), str(r.get("decision_effect", ""))]
        for r in export_data.get("score_impact", [])
    ]
    story.append(_table(["Component", "Value", "Impact", "Effect"], impact_rows or [["—", "—", "—", "—"]]))

    story.append(_section("Evidence quality"))
    evidence_rows = [
        [r.get("evidence", ""), r.get("status", ""), r.get("detail", "")]
        for r in export_data.get("evidence_quality", [])
    ]
    story.append(_table(["Evidence", "Status", "Detail"], evidence_rows or [["—", "—", "—"]], [1.4 * inch, 0.9 * inch, doc.width - 2.3 * inch]))

    story.append(_section("Financial performance"))
    story.append(
        _table(
            ["Metric", "Value"],
            [
                ["Total revenue", f"£{float(fm.get('Total Revenue', 0) or 0):,.2f}"],
                ["Monthly avg revenue", f"£{float(fm.get('Monthly Average Revenue', 0) or 0):,.2f}"],
                ["Net income", f"£{float(fm.get('Net Income', 0) or 0):,.2f}"],
                ["Operating margin", f"{float(fm.get('Operating Margin', 0) or 0) * 100:.1f}%"],
                ["DSCR", f"{float(fm.get('Debt Service Coverage Ratio', 0) or 0):.2f}"],
                ["Cash flow volatility", f"{float(fm.get('Cash Flow Volatility', 0) or 0):.3f}"],
            ],
            [2.5 * inch, doc.width - 2.5 * inch],
        )
    )

    thresholds = export_data.get("metrics_thresholds", [])
    if thresholds:
        story.append(_section("Metrics vs industry thresholds"))
        story.append(
            _table(
                ["Metric", "Actual", "Threshold", "Status"],
                [[r["metric"], r["actual_value"], r["threshold"], r["status"]] for r in thresholds],
            )
        )

    loans = export_data.get("loans_analysis", {}) or {}
    if loans and (loans.get("loan_count", 0) > 0 or loans.get("repayment_count", 0) > 0 or loans.get("manual_outstanding_debt", 0) > 0):
        story.append(_section("Loans & debt analysis"))
        ratio = loans.get("repayment_ratio")
        ratio_s = f"{(ratio or 0) * 100:.1f}%" if ratio is not None else "N/A"
        story.append(
            _table(
                ["Metric", "Value"],
                [
                    ["Total known borrowing", f"£{loans.get('total_known_borrowing', loans.get('total_loans_received', 0)):,.0f}"],
                    ["Total repayments", f"£{loans.get('total_repayments_made', 0):,.0f}"],
                    ["Outstanding balance", f"£{loans.get('known_outstanding_balance', loans.get('net_borrowing', 0)):,.0f}"],
                    ["Repayment ratio", ratio_s],
                ],
                [2.5 * inch, doc.width - 2.5 * inch],
            )
        )

    ob_rows = [[r.get("signal", ""), str(r.get("value", "")), r.get("meaning", "")] for r in export_data.get("open_banking_insights", [])]
    if ob_rows:
        story.append(_section("Open banking insights"))
        story.append(_table(["Signal", "Value", "Meaning"], ob_rows, [1.5 * inch, 1.3 * inch, doc.width - 2.8 * inch]))

    card_rows = [[r.get("signal", ""), str(r.get("value", "")), r.get("meaning", "")] for r in export_data.get("card_processing_insights", [])]
    if card_rows:
        story.append(_section("Card processor insights"))
        story.append(_table(["Signal", "Value", "Meaning"], card_rows, [1.5 * inch, 1.3 * inch, doc.width - 2.8 * inch]))

    rf = bp.get("risk_factors", {}) or {}
    story.append(_section("Business risk flags"))
    story.append(
        _table(
            ["Risk factor", "Status"],
            [
                ["Business CCJs", "Yes" if rf.get("business_ccj") else "No"],
                ["Poor/no online presence", "Yes" if rf.get("poor_or_no_online_presence") else "No"],
                ["Generic email", "Yes" if rf.get("uses_generic_email") else "No"],
            ],
            [2.5 * inch, doc.width - 2.5 * inch],
        )
    )

    story.append(Spacer(1, 16))
    story.append(
        _p(
            f"<i>Generated by {info.get('generated_by', GENERATED_BY)}. "
            "For informational purposes only — not financial advice.</i>",
            small_style,
        )
    )

    doc.build(story)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Streamlit export UI
# ---------------------------------------------------------------------------

class DashboardExporter:
    """Dashboard export system shared by main app and reports page."""

    def __init__(self):
        self.export_timestamp = datetime.now()

    def export_dashboard_data(
        self,
        company_name: str,
        params: dict,
        metrics: dict,
        scores: dict,
        analysis_period: str | int,
        revenue_insights: dict,
        loans_analysis: dict | None = None,
        filtered_df: pd.DataFrame | None = None,
        manual_debt_balances: dict | None = None,
    ) -> dict:
        payload = build_export_payload(
            company_name=company_name,
            params=params,
            metrics=metrics,
            scores=scores,
            analysis_period=analysis_period,
            revenue_insights=revenue_insights,
            filtered_df=filtered_df,
            manual_debt_balances=manual_debt_balances,
            export_timestamp=self.export_timestamp,
        )
        if loans_analysis:
            payload["loans_analysis"] = _clean_loans_analysis(loans_analysis)
        return payload

    def generate_html_report(self, export_data: dict) -> str:
        return generate_html_report(export_data)

    def generate_pdf_report(self, export_data: dict) -> bytes:
        return generate_pdf_report(export_data)

    def generate_csv_metrics(self, export_data: dict, company_name: str) -> str:
        rows: list[dict[str, object]] = []
        for key, value in export_data.get("financial_metrics", {}).items():
            if isinstance(value, (int, float)):
                rows.append({"Metric": key, "Value": value})
            elif value is not None and not isinstance(value, (dict, list)):
                rows.append({"Metric": key, "Value": str(value)})

        for row in export_data.get("metrics_thresholds", []):
            rows.append(
                {
                    "Metric": f"{row.get('metric')} (threshold)",
                    "Value": f"{row.get('actual_value')} vs {row.get('threshold')} — {row.get('status')}",
                }
            )

        sr = export_data.get("scoring_results", {})
        for label, key in [
            ("Final decision", "final_decision"),
            ("MCA rule decision", "mca_rule_decision"),
            ("Subprime tier", "subprime_tier"),
            ("Combined score", None),
        ]:
            if key:
                val = sr.get(key)
            else:
                val = (sr.get("ensemble") or {}).get("combined_score")
            if val is not None:
                rows.append({"Metric": label, "Value": val})

        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame([{"Metric": "No metrics", "Value": ""}])
        df.insert(0, "Company", company_name)
        df.insert(1, "Export Date", self.export_timestamp.strftime("%Y-%m-%d %H:%M:%S"))
        return df.to_csv(index=False)

    def create_export_buttons(
        self,
        company_name: str,
        params: dict,
        metrics: dict,
        scores: dict,
        analysis_period: str | int,
        revenue_insights: dict,
        loans_analysis: dict | None = None,
        filtered_df: pd.DataFrame | None = None,
        manual_debt_balances: dict | None = None,
    ) -> None:
        import streamlit as st

        st.markdown("---")
        st.subheader("Export dashboard report")

        export_data = self.export_dashboard_data(
            company_name=company_name,
            params=params,
            metrics=metrics,
            scores=scores,
            analysis_period=analysis_period,
            revenue_insights=revenue_insights,
            loans_analysis=loans_analysis,
            filtered_df=filtered_df,
            manual_debt_balances=manual_debt_balances,
        )

        slug = company_name.replace(" ", "_")
        stamp = datetime.now().strftime("%Y%m%d_%H%M")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.download_button(
                label="Export HTML report",
                data=self.generate_html_report(export_data),
                file_name=f"{slug}_financial_report_{stamp}.html",
                mime="text/html",
                help="Full dashboard report as a web page",
                type="primary",
            )

        with col2:
            st.download_button(
                label="Export JSON data",
                data=json.dumps(export_data, indent=2, default=str),
                file_name=f"{slug}_data_{stamp}.json",
                mime="application/json",
                help="Complete structured export for integration",
            )

        with col3:
            st.download_button(
                label="Export CSV metrics",
                data=self.generate_csv_metrics(export_data, company_name),
                file_name=f"{slug}_metrics_{stamp}.csv",
                mime="text/csv",
                help="Key metrics and scoring summary for spreadsheets",
            )

        with col4:
            st.download_button(
                label="Export PDF report",
                data=self.generate_pdf_report(export_data),
                file_name=f"{slug}_financial_report_{stamp}.pdf",
                mime="application/pdf",
                help="Production PDF report matching the dashboard",
            )

        st.info(
            "**Export options**\n"
            "- **HTML report** — full dashboard in a browser-ready page\n"
            "- **JSON data** — complete structured data for analysis or integration\n"
            "- **CSV metrics** — key numbers and threshold results for spreadsheets\n"
            "- **PDF report** — polished printable report\n\n"
            "**Includes:** unified recommendation, scoring, financial metrics, revenue insights, "
            "loans analysis, open banking & card insights, risk factors, and business parameters."
        )
