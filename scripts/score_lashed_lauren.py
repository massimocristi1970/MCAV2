"""Headless score run for Lashed by Lauren king LTD application."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("DEBUG", "false")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

APP = Path(
    r"C:\Users\Massimo Cristi\OneDrive - Savvy Loan Products Ltd"
    r"\Merchant Cash Advance (MCA)\Applications\UNDERWRITING\Lashed by Lauren king LTD"
)
OB = APP / "Lashed by Lauren king LTD.json"
TU = APP / "101922.xml"
CARD = APP / "Sum up payments .csv"

from app.main import (
    _score_tu_xml_bytes,
    apply_manual_outstanding_debt,
    assess_primary_account_signal,
    calculate_all_scores_enhanced,
    calculate_financial_metrics,
    calculate_revenue_insights,
    combine_mca_and_tu_decisions,
    derive_card_processing_payload,
    filter_data_by_period,
)
from app.services.scoring_alignment import align_scoring_metrics
from app.services.subprime_scoring_system import SubprimeScoring
from app.workflows.application_analysis import AnalysisCallbacks, analyse_open_banking_application


def main() -> None:
    with OB.open("r", encoding="utf-8") as f:
        json_data = json.load(f)

    tu = _score_tu_xml_bytes(TU.read_bytes(), "lashed_lauren")
    params = {
        "company_name": "Lashed by Lauren king LTD",
        "industry": "Beauty Salons and Spas",
        "requested_loan": 5000.0,
        "company_age_months": 60,
        "business_ccj": False,
        "business_ccj_count": 0,
        "poor_or_no_online_presence": False,
        "uses_generic_email": False,
        "business_credit_score": 80,
        "business_credit_limit": 600,
        "business_max_recommended_credit": 600,
        "business_negative_impact_count": 1,
        "business_enquiries_3m": 19,
        "business_bureau_parse_status": "parsed",
        "bureau_band": "B (Moderate Risk)",
        "tu_parse_status": "parsed",
        "tu_director_score": tu["score"],
        "tu_director_decision": tu["decision"],
        "tu_director_flags": tu.get("flags", []),
        "tu_director_reasons": tu.get("reasons", []),
        "tu_director_features": tu.get("features", {}),
        "directors_score": tu["score"],
    }

    card_files = None
    if CARD.exists():

        class CardFile:
            name = CARD.name

            def getvalue(self) -> bytes:
                return CARD.read_bytes()

        card_files = [CardFile()]

    result = analyse_open_banking_application(
        json_data=json_data,
        params=params,
        analysis_period="All",
        card_terminal_files=card_files,
        callbacks=AnalysisCallbacks(
            filter_data_by_period=filter_data_by_period,
            assess_primary_account_signal=assess_primary_account_signal,
            calculate_financial_metrics=calculate_financial_metrics,
            apply_manual_outstanding_debt=apply_manual_outstanding_debt,
            derive_card_processing_payload=derive_card_processing_payload,
            calculate_all_scores_enhanced=calculate_all_scores_enhanced,
            combine_mca_and_tu_decisions=combine_mca_and_tu_decisions,
            calculate_revenue_insights=calculate_revenue_insights,
        ),
    )

    scores = result.scores
    metrics = result.metrics
    params = result.params
    ens = scores.get("ensemble") or {}

    print("=== CURRENT RUN (post scoring fixes) ===")
    print(f"MCA: {params.get('mca_rule_score')} / {params.get('mca_rule_decision')}")
    print(f"Subprime: {scores.get('subprime_score')} / {scores.get('subprime_tier')}")
    print(f"Final: {scores.get('final_decision')}")
    print(f"Ensemble: {ens.get('combined_score')} / {ens.get('decision')}")
    print(f"Alignment: {ens.get('decision_alignment')} (gap {ens.get('numeric_score_gap')})")
    print(f"Convergence: {ens.get('score_convergence')}")
    print(f"Primary: {ens.get('primary_reason')}")
    print()
    for k in (
        "Debt Service Coverage Ratio",
        "DSCR Repayments Observed",
        "Cash Flow Volatility",
        "Cash Flow Volatility Before MCA Align",
        "MCA Inflow CV",
        "Revenue Growth Rate",
        "Average Month-End Balance",
        "Balance Source",
        "Operating Margin",
        "Number of Bounced Payments",
    ):
        print(f"  {k}: {metrics.get(k)}")
    print()
    print("Subprime breakdown:")
    for line in scores.get("subprime_breakdown", []):
        print(f"  {line}")

    scorer = SubprimeScoring()
    diag = scorer.calculate_subprime_score(metrics, dict(params)).get("diagnostics", {})
    print()
    print("Metric points:")
    for m in diag.get("metric_breakdown", []):
        print(
            f"  {m['metric']}: {m['actual_value']} -> "
            f"{m['points_earned']}/{m['points_possible']} ({m['status']})"
        )

    # Counterfactual: old saved metrics without MCA align
    saved_metrics = {
        "Debt Service Coverage Ratio": 1.73,
        "Revenue Growth Rate": -0.1,
        "Average Month-End Balance": 256.65,
        "Cash Flow Volatility": 0.358,
        "Operating Margin": 0.496,
        "Net Income": 25140.56,
        "Average Negative Balance Days per Month": 3,
        "Number of Bounced Payments": 0,
        "Card Processing Insight Layer": "Available",
        "Card Processing Score Adjustment": -1.0,
    }
    saved_params = dict(params)
    saved_params.pop("_applied_risk_penalties", None)
    saved_params.pop("_card_processing_score_adjustment", None)
    old = scorer.calculate_subprime_score(saved_metrics, saved_params)
    print()
    print("=== COUNTERFACTUAL: saved-run metrics, no MCA align ===")
    print(f"Subprime: {old['subprime_score']}")
    for line in old["breakdown"][:14]:
        print(f"  {line}")

    aligned = dict(saved_metrics)
    align_scoring_metrics(aligned, {"mca_rule_signals": params.get("mca_rule_signals")})
    aligned_res = scorer.calculate_subprime_score(aligned, saved_params)
    print()
    print("=== COUNTERFACTUAL: saved metrics + MCA vol align only ===")
    print(f"Volatility: {aligned.get('Cash Flow Volatility')}")
    print(f"Subprime: {aligned_res['subprime_score']}")
    for line in aligned_res["breakdown"][:14]:
        print(f"  {line}")


    run_variants()


def run_variants() -> None:
    """Explore configurations that might produce a lower score."""
    scenarios = [
        ("All+card+TU", "All", True, True),
        ("6m+card+TU", "6", True, True),
        ("3m+card+TU", "3", True, True),
        ("All no card", "All", False, True),
        ("6m no card", "6", False, True),
        ("All no TU", "All", True, False),
    ]
    for label, period, use_card, use_tu in scenarios:
        with OB.open("r", encoding="utf-8") as f:
            json_data = json.load(f)
        tu = _score_tu_xml_bytes(TU.read_bytes(), "lashed_lauren") if use_tu else None
        params = {
            "company_name": "Lashed by Lauren king LTD",
            "industry": "Beauty Salons and Spas",
            "requested_loan": 5000.0,
            "company_age_months": 60,
            "business_ccj": False,
            "business_ccj_count": 0,
            "business_credit_limit": 600,
            "business_max_recommended_credit": 600,
            "business_negative_impact_count": 1,
            "business_enquiries_3m": 19,
            "tu_parse_status": "parsed" if use_tu else "missing",
        }
        if tu:
            params.update(
                {
                    "tu_director_score": tu["score"],
                    "tu_director_decision": tu["decision"],
                    "tu_director_features": tu.get("features", {}),
                    "directors_score": tu["score"],
                }
            )
        else:
            params["directors_score"] = 50

        card_files = None
        if use_card and CARD.exists():

            class CardFile:
                name = CARD.name

                def getvalue(self) -> bytes:
                    return CARD.read_bytes()

            card_files = [CardFile()]

        result = analyse_open_banking_application(
            json_data=json_data,
            params=params,
            analysis_period=period,
            card_terminal_files=card_files,
            callbacks=AnalysisCallbacks(
                filter_data_by_period=filter_data_by_period,
                assess_primary_account_signal=assess_primary_account_signal,
                calculate_financial_metrics=calculate_financial_metrics,
                apply_manual_outstanding_debt=apply_manual_outstanding_debt,
                derive_card_processing_payload=derive_card_processing_payload,
                calculate_all_scores_enhanced=calculate_all_scores_enhanced,
                combine_mca_and_tu_decisions=combine_mca_and_tu_decisions,
                calculate_revenue_insights=calculate_revenue_insights,
            ),
        )
        s = result.scores
        e = s.get("ensemble") or {}
        m = result.metrics
        print(
            f"{label:14} subprime={s.get('subprime_score')} tier={s.get('subprime_tier')} "
            f"final={s.get('final_decision')} ens={e.get('decision')} "
            f"dscr={m.get('Debt Service Coverage Ratio')} vol={m.get('Cash Flow Volatility')} "
            f"growth={m.get('Revenue Growth Rate')}"
        )


if __name__ == "__main__":
    main()
    print()
    run_variants()
