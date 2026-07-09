"""Streamlit UI helpers for complementary MCA / Subprime scoring views."""

from __future__ import annotations

from typing import Any


def _mca_flow_summary(params: dict, metrics: dict) -> str:
    signals = params.get("mca_rule_signals") or {}
    parts = []
    if signals.get("inflow_days_30d") is not None:
        parts.append(f"{signals['inflow_days_30d']} revenue days in last 30d")
    if signals.get("max_inflow_gap_days") is not None:
        parts.append(f"max gap {signals['max_inflow_gap_days']}d")
    if signals.get("inflow_cv") is not None:
        parts.append(f"inflow CV {float(signals['inflow_cv']):.2f}")
    if not parts and metrics.get("MCA Inflow CV") is not None:
        parts.append(f"inflow CV {metrics['MCA Inflow CV']}")
    decision = params.get("mca_rule_decision") or "N/A"
    return f"MCA decision **{decision}** — " + (", ".join(parts) if parts else "transaction flow signals not available")


def _subprime_summary(scores: dict, params: dict, metrics: dict) -> str:
    tier = scores.get("subprime_tier", "N/A")
    sub = scores.get("subprime_score")
    dscr = metrics.get("Debt Service Coverage Ratio")
    dscr_note = ""
    if not metrics.get("DSCR Repayments Observed", True):
        dscr_note = " (neutral — no repayments seen in bank data)"
    return (
        f"Subprime **{sub:.1f}** ({tier}) — DSCR {dscr}{dscr_note}, "
        f"director TU {params.get('directors_score', 'N/A')}"
    )


def render_complementary_scoring_assessment(
    ensemble: dict,
    scores: dict,
    params: dict,
    metrics: dict,
) -> None:
    """Show MCA vs Subprime as complementary views, not a failure to agree."""
    import streamlit as st

    if not ensemble:
        return

    alignment = ensemble.get("decision_alignment") or ensemble.get("detailed_breakdown", {}).get("decision_alignment")
    alignment_detail = ensemble.get("decision_alignment_detail") or ensemble.get("detailed_breakdown", {}).get(
        "decision_alignment_detail"
    )
    convergence = ensemble.get("score_convergence", "Unknown")
    gap = ensemble.get("numeric_score_gap") or ensemble.get("detailed_breakdown", {}).get("numeric_score_gap")

    st.markdown("#### Complementary scoring assessment")
    st.caption(
        "MCA measures **daily collection feasibility** from bank flow. "
        "Subprime measures **business and director creditworthiness**. "
        "They are not expected to match numerically."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**MCA Rule (collection pattern)**")
        st.write(_mca_flow_summary(params, metrics))
        mca_score = (ensemble.get("contributing_scores") or {}).get("mca_score", params.get("mca_rule_score"))
        if mca_score is not None:
            st.metric("MCA score", f"{float(mca_score):.0f}")

    with col_b:
        st.markdown("**Subprime (business profile)**")
        if scores.get("subprime_score") is not None:
            st.write(_subprime_summary(scores, params, metrics))
            st.metric("Subprime score", f"{float(scores['subprime_score']):.1f}")
        else:
            st.info("Subprime score unavailable for this run.")

    if alignment:
        if alignment == "Aligned":
            st.success(f"**Decision alignment:** {alignment} — {alignment_detail}")
        elif alignment == "Mixed":
            st.warning(f"**Decision alignment:** {alignment} — {alignment_detail}")
        elif alignment == "Opposed":
            st.error(f"**Decision alignment:** {alignment} — {alignment_detail}")
        else:
            st.info(f"**Decision alignment:** {alignment} — {alignment_detail}")

    gap_text = f"{gap:.0f} points" if gap is not None else "n/a"
    if convergence == "High Convergence":
        st.success(f"**Numeric gap:** {gap_text} — {convergence}")
    elif convergence in ("Good Convergence", "Moderate Convergence"):
        st.info(f"**Numeric gap:** {gap_text} — {convergence} (expected when models measure different risks)")
    elif convergence == "Single Score Available":
        st.warning(f"**Numeric gap:** {gap_text} — only one decision score was available")
    else:
        st.info(f"**Numeric gap:** {gap_text} — {convergence} (review both models separately)")
