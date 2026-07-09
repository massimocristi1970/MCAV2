import pandas as pd
import pytest

from app.services.business_risk_signals import calculate_business_metrics
from app.workflows.application_analysis import AnalysisCallbacks, analyse_open_banking_application


@pytest.mark.unit
def test_application_analysis_workflow_builds_run_payload_without_ui_side_effects():
    payload = {
        "transactions": [
            {"transaction_id": "txn_workflow_1", "date": "2026-05-01", "amount": -1000, "name": "Stripe payout", "personal_finance_category": {"primary": "INCOME", "detailed": "INCOME_OTHER_INCOME"}},
            {"transaction_id": "txn_workflow_2", "date": "2026-05-03", "amount": 250, "name": "Office rent", "personal_finance_category": {"primary": "RENT_AND_UTILITIES", "detailed": "RENT_AND_UTILITIES_RENT"}},
        ]
    }
    params = {
        "company_name": "Fixture Ltd",
        "industry": "Other",
        "requested_loan": 1000,
        "directors_score": 70,
        "company_age_months": 24,
        "tu_director_decision": "APPROVE",
        "tu_director_score": 70,
        "tu_director_features": {"defaults_36m_total": 0, "defaults_12m_total": 0, "ccj_active_total": 0},
        "tu_parse_status": "parsed",
    }

    callbacks = AnalysisCallbacks(
        filter_data_by_period=lambda df, period: df,
        assess_primary_account_signal=lambda df: {"status": "not_assessed"},
        calculate_financial_metrics=calculate_business_metrics,
        apply_manual_outstanding_debt=lambda metrics: metrics,
        derive_card_processing_payload=lambda df, files: {"insights": {}, "error": None},
        calculate_all_scores_enhanced=lambda metrics, params: {
            "subprime_recommendation": "APPROVE",
            "ensemble": {"decision": "APPROVE", "primary_reason": "fixture"},
        },
        combine_mca_and_tu_decisions=lambda decision, tu_decision: decision,
        calculate_revenue_insights=lambda df: {"avg_revenue_transactions_per_day": 1},
    )

    result = analyse_open_banking_application(
        json_data=payload,
        params=params,
        analysis_period="All",
        card_terminal_files=None,
        callbacks=callbacks,
        source_upload_name="fixture.json",
    )

    assert isinstance(result.df, pd.DataFrame)
    assert result.metrics["Total Revenue"] == 1000
    assert result.run["underwriting"]
    assert result.run["underwriting"]["data_quality"]["overall"] == "Fail"
    assert result.scores["final_decision"] == "REFER"
    assert any("Data quality gate" in r for r in result.scores.get("final_decision_reasons", []))
    assert result.run["source_upload_name"] == "fixture.json"
    assert result.run["params"]["open_banking_ingestion"]["transaction_count"] == 2

