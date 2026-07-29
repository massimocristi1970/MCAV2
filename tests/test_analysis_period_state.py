"""Regression tests for changing the period on a cached analysis."""

from __future__ import annotations

import pandas as pd

import app.main as main


def test_cached_analysis_can_be_reprocessed_for_a_new_period(monkeypatch):
    periods_seen: list[str] = []
    dataframe = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-06-01"]),
            "amount": [100.0, 200.0],
        }
    )
    session_state = {
        "last_run": {
            "analysis_period": "All",
            "df": dataframe,
            "filtered_df": dataframe,
            "params": {"company_age_months": 24},
            "metrics": {},
            "scores": {},
        },
        "manual_outstanding_debt_balances": {},
    }
    monkeypatch.setattr(main.st, "session_state", session_state)

    def filter_for_period(df, period):
        periods_seen.append(period)
        return df.iloc[-1:].copy()

    monkeypatch.setattr(main, "filter_data_by_period", filter_for_period)
    monkeypatch.setattr(main, "assess_primary_account_signal", lambda df: {})
    monkeypatch.setattr(main, "calculate_financial_metrics", lambda df, age: {"rows": len(df)})
    monkeypatch.setattr(main, "apply_manual_outstanding_debt", lambda metrics: metrics)
    monkeypatch.setattr(main, "apply_tu_features_to_params_and_metrics", lambda *args: None)
    monkeypatch.setattr(main, "align_scoring_metrics", lambda *args: None)
    monkeypatch.setattr(main, "derive_card_processing_payload", lambda *args: {"insights": {}})
    monkeypatch.setattr(main, "dataframe_to_mca_transactions", lambda df: [])

    def apply_mca(params, transactions, period):
        periods_seen.append(period)
        params.update(
            {
                "mca_rule_decision": "APPROVE",
                "mca_rule_score": 90,
                "mca_rule_reasons": [],
            }
        )

    monkeypatch.setattr(main, "apply_mca_rule_to_params", apply_mca)
    monkeypatch.setattr(
        main,
        "calculate_all_scores_enhanced",
        lambda metrics, params: {
            "ensemble": {"decision": "APPROVE", "primary_reason": "test"},
        },
    )
    monkeypatch.setattr(main, "combine_mca_and_tu_decisions", lambda decision, tu: decision)
    monkeypatch.setattr(main, "assess_data_quality", lambda *args: {})
    monkeypatch.setattr(main, "apply_data_quality_gate", lambda *args: None)
    monkeypatch.setattr(main, "calculate_revenue_insights", lambda df: {"rows": len(df)})
    monkeypatch.setattr(main, "build_underwriting_package", lambda **kwargs: {})

    assert main.rerun_last_analysis_with_manual_debt("6") is True

    updated = session_state["last_run"]
    assert periods_seen == ["6", "6"]
    assert updated["analysis_period"] == "6"
    assert len(updated["filtered_df"]) == 1
    assert updated["metrics"]["rows"] == 1
