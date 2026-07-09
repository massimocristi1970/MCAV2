"""Application analysis workflow orchestration.

This module owns the non-UI path from raw Open Banking JSON to the run payload
stored by the Streamlit app. UI rendering, persistence, and Streamlit widgets
stay in ``app.main``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

import pandas as pd

from app.services.open_banking_adapter import normalize_open_banking_payload, transactions_to_dataframe
from app.services.mca_rule_runner import run_mca_rule_scoring
from app.services.tu_director_params import apply_tu_features_to_params_and_metrics
from app.services.scoring_alignment import align_scoring_metrics
from app.services.underwriting_insights import (
    apply_data_quality_gate,
    assess_data_quality,
    build_underwriting_package,
)


@dataclass
class AnalysisCallbacks:
    filter_data_by_period: Callable[[pd.DataFrame, str], pd.DataFrame]
    assess_primary_account_signal: Callable[[pd.DataFrame], dict]
    calculate_financial_metrics: Callable[[pd.DataFrame, int | float], dict]
    apply_manual_outstanding_debt: Callable[[dict], dict]
    derive_card_processing_payload: Callable[[pd.DataFrame, Any], dict]
    calculate_all_scores_enhanced: Callable[[dict, dict], dict]
    combine_mca_and_tu_decisions: Callable[[str, Any], str]
    calculate_revenue_insights: Callable[[pd.DataFrame], dict]


@dataclass
class AnalysisResult:
    df: pd.DataFrame
    filtered_df: pd.DataFrame
    params: Dict[str, Any]
    metrics: Dict[str, Any]
    scores: Dict[str, Any]
    revenue_insights: Dict[str, Any]
    card_processing_payload: Dict[str, Any]
    ingestion_metadata: Dict[str, Any]
    date_min_iso: str | None
    date_max_iso: str | None
    run: Dict[str, Any]


def _base_decision_from_subprime(recommendation_text: str) -> str:
    text = (recommendation_text or "").upper()
    if "CONDITIONAL" in text or "SENIOR REVIEW" in text or "REVIEW" in text:
        return "REFER"
    if "APPROVE" in text:
        return "APPROVE"
    return "DECLINE"


def _apply_final_decision(scores: Dict[str, Any], params: Dict[str, Any], combine_decisions: Callable[[str, Any], str]) -> None:
    ensemble = scores.get("ensemble") or {}
    ensemble_decision = ensemble.get("decision")
    if ensemble_decision is not None:
        ensemble_decision = str(ensemble_decision).upper().strip()
        params["mca_main_decision"] = ensemble_decision

    if ensemble_decision in ("DECLINE", "REFER", "APPROVE", "SENIOR_REVIEW", "CONDITIONAL_APPROVE"):
        final_decision = combine_decisions(ensemble_decision, params.get("tu_director_decision"))
        final_reasons = [
            f"Decision from weighted MCA/Subprime engine: {ensemble_decision}",
            f"Final TU overlay: {ensemble_decision} -> {final_decision}",
            f"Reason: {ensemble.get('primary_reason', 'n/a')}",
        ]
    else:
        base_decision = _base_decision_from_subprime(scores.get("subprime_recommendation", ""))
        final_decision = combine_decisions(base_decision, params.get("tu_director_decision"))
        final_reasons = [
            f"Fallback decision from Subprime: {base_decision}",
            f"Final TU overlay: {base_decision} -> {final_decision}",
        ]

    scores["final_decision"] = final_decision
    scores["final_decision_reasons"] = final_reasons
    params["final_decision"] = final_decision
    params["final_decision_reasons"] = final_reasons


def analyse_open_banking_application(
    *,
    json_data: Any,
    params: Dict[str, Any],
    analysis_period: str,
    card_terminal_files: Any,
    callbacks: AnalysisCallbacks,
    source_upload_name: str | None = None,
) -> AnalysisResult:
    """Run the canonical upload analysis path without UI side effects."""
    params = dict(params)
    ob_payload = normalize_open_banking_payload(json_data)
    transactions = ob_payload.transactions
    if not transactions:
        raise ValueError("No usable transactions found in JSON file")

    params["open_banking_ingestion"] = ob_payload.metadata

    df = transactions_to_dataframe(transactions, ob_payload.accounts)
    if df.empty:
        raise ValueError("No valid transactions after cleaning")

    filtered_df = callbacks.filter_data_by_period(df, analysis_period)

    mca = run_mca_rule_scoring(transactions, analysis_period)
    params.update(mca)

    primary_account_assessment = callbacks.assess_primary_account_signal(filtered_df)
    params["primary_account_assessment"] = primary_account_assessment

    metrics = callbacks.calculate_financial_metrics(filtered_df, params["company_age_months"])
    metrics = callbacks.apply_manual_outstanding_debt(metrics)
    apply_tu_features_to_params_and_metrics(params, metrics, params.get("tu_director_features"))
    align_scoring_metrics(metrics, params)
    card_processing_payload = callbacks.derive_card_processing_payload(filtered_df, card_terminal_files)
    metrics.update(card_processing_payload.get("insights") or {})

    scores = callbacks.calculate_all_scores_enhanced(metrics, params)
    scores["mca_rule_decision"] = params.get("mca_rule_decision")
    scores["mca_rule_score"] = params.get("mca_rule_score")
    scores["mca_rule_reasons"] = params.get("mca_rule_reasons", [])
    scores["primary_account_assessment"] = primary_account_assessment

    data_quality = assess_data_quality(filtered_df, metrics, params, analysis_period)
    params["data_quality"] = data_quality

    _apply_final_decision(scores, params, callbacks.combine_mca_and_tu_decisions)
    apply_data_quality_gate(scores, params, data_quality)

    revenue_insights = callbacks.calculate_revenue_insights(filtered_df)
    date_min_iso = str(df["date"].min()) if "date" in df.columns and not df.empty else None
    date_max_iso = str(df["date"].max()) if "date" in df.columns and not df.empty else None

    underwriting = build_underwriting_package(
        metrics=metrics,
        params=params,
        scores=scores,
        filtered_df=filtered_df,
        analysis_period=analysis_period,
        manual_debt_balances=params.get("manual_outstanding_debt_balances") or {},
    )

    run = {
        "company_name": params.get("company_name"),
        "analysis_period": analysis_period,
        "df": df,
        "filtered_df": filtered_df,
        "params": params,
        "metrics": metrics,
        "scores": scores,
        "revenue_insights": revenue_insights,
        "underwriting": underwriting,
        "manual_outstanding_debt_balances": params.get("manual_outstanding_debt_balances") or {},
        "card_terminal_files": None,
        "card_processing_payload": card_processing_payload,
        "source_upload_name": source_upload_name,
    }

    return AnalysisResult(
        df=df,
        filtered_df=filtered_df,
        params=params,
        metrics=metrics,
        scores=scores,
        revenue_insights=revenue_insights,
        card_processing_payload=card_processing_payload,
        ingestion_metadata=ob_payload.metadata,
        date_min_iso=date_min_iso,
        date_max_iso=date_max_iso,
        run=run,
    )

