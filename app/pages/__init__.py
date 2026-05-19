"""
Pages package.

Streamlit page modules render UI when executed as pages, so this package keeps
imports lazy to avoid executing one page while another page is loading.
"""

_EXPORTS = {
    # Scoring
    "WEIGHTS": "app.pages.scoring",
    "PENALTIES": "app.pages.scoring",
    "calculate_weighted_scores": "app.pages.scoring",
    "load_models": "app.pages.scoring",
    "calculate_ml_score": "app.pages.scoring",
    "calculate_subprime_score": "app.pages.scoring",
    "adjust_ml_score_for_growth_business": "app.pages.scoring",
    "get_ml_score_interpretation": "app.pages.scoring",
    "determine_loan_risk_level": "app.pages.scoring",
    "render_scoring_page": "app.pages.scoring",
    # Charts
    "create_score_charts": "app.pages.charts",
    "create_financial_charts": "app.pages.charts",
    "create_threshold_chart": "app.pages.charts",
    "create_monthly_breakdown_charts": "app.pages.charts",
    "create_loans_repayments_charts": "app.pages.charts",
    "create_score_comparison_chart": "app.pages.charts",
    "create_cashflow_trend_chart": "app.pages.charts",
    "create_risk_gauge_chart": "app.pages.charts",
    # Transactions
    "map_transaction_category": "app.pages.transactions",
    "categorize_transactions": "app.pages.transactions",
    "filter_data_by_period": "app.pages.transactions",
    "calculate_financial_metrics": "app.pages.transactions",
    "calculate_revenue_insights": "app.pages.transactions",
    "create_monthly_breakdown": "app.pages.transactions",
    "create_categorized_csv": "app.pages.transactions",
    "analyze_loans_and_repayments": "app.pages.transactions",
    # Reports
    "DashboardExporter": "app.pages.reports",
    "get_score_summary_text": "app.pages.reports",
    "format_metrics_for_display": "app.pages.reports",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module = import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
