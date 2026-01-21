# app/pages/__init__.py
"""
Pages module containing extracted components from main.py.

This module provides organized access to:
- Scoring functions (scoring.py)
- Chart/visualization functions (charts.py)
- Transaction processing functions (transactions.py)
- Report generation functions (reports.py)
"""

# Scoring module exports
from .scoring import (
    WEIGHTS,
    PENALTIES,
    calculate_weighted_scores,
    load_models,
    calculate_ml_score,
    calculate_subprime_score,
    adjust_ml_score_for_growth_business,
    get_ml_score_interpretation,
    determine_loan_risk_level,
)

# Charts module exports
from .charts import (
    create_score_charts,
    create_financial_charts,
    create_threshold_chart,
    create_monthly_breakdown_charts,
    create_loans_repayments_charts,
    create_score_comparison_chart,
    create_cashflow_trend_chart,
    create_risk_gauge_chart,
)

# Transactions module exports
from .transactions import (
    map_transaction_category,
    categorize_transactions,
    filter_data_by_period,
    calculate_financial_metrics,
    calculate_revenue_insights,
    create_monthly_breakdown,
    create_categorized_csv,
    analyze_loans_and_repayments,
)

# Reports module exports
from .reports import (
    DashboardExporter,
    get_score_summary_text,
    format_metrics_for_display,
)

__all__ = [
    # Scoring
    'WEIGHTS',
    'PENALTIES',
    'calculate_weighted_scores',
    'load_models',
    'calculate_ml_score',
    'calculate_subprime_score',
    'adjust_ml_score_for_growth_business',
    'get_ml_score_interpretation',
    'determine_loan_risk_level',
    # Charts
    'create_score_charts',
    'create_financial_charts',
    'create_threshold_chart',
    'create_monthly_breakdown_charts',
    'create_loans_repayments_charts',
    'create_score_comparison_chart',
    'create_cashflow_trend_chart',
    'create_risk_gauge_chart',
    # Transactions
    'map_transaction_category',
    'categorize_transactions',
    'filter_data_by_period',
    'calculate_financial_metrics',
    'calculate_revenue_insights',
    'create_monthly_breakdown',
    'create_categorized_csv',
    'analyze_loans_and_repayments',
    # Reports
    'DashboardExporter',
    'get_score_summary_text',
    'format_metrics_for_display',
]
