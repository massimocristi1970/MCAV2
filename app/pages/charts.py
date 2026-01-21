# app/pages/charts.py
"""Chart and visualization functions for business finance dashboard."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional, Tuple, List


def create_score_charts(scores: Dict[str, Any], metrics: Dict[str, Any]) -> go.Figure:
    """
    Create bar chart comparing different scoring methods.
    
    Args:
        scores: Dictionary of all calculated scores
        metrics: Financial metrics dictionary
        
    Returns:
        Plotly figure object
    """
    fig_scores = go.Figure()
    
    score_data = {
        'Subprime Score': scores.get('subprime_score', 0),
        'V2 Weighted': scores.get('weighted_score', 0),
        'Adjusted ML': scores.get('adjusted_ml_score', scores.get('ml_score', 0) or 0)
    }
    
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e']  # Green, Blue, Orange
    
    fig_scores.add_trace(go.Bar(
        x=list(score_data.keys()),
        y=list(score_data.values()),
        marker_color=colors,
        text=[f"{v:.1f}" + ("%" if k == "Adjusted ML" else "/100") for k, v in score_data.items()],
        textposition='outside'
    ))
    
    fig_scores.update_layout(
        title="Primary Scoring Methods Comparison",
        yaxis_title="Score",
        showlegend=False,
        height=400,
        yaxis=dict(range=[0, 100])
    )
    
    return fig_scores


def create_financial_charts(metrics: Dict[str, Any]) -> Tuple[go.Figure, Optional[go.Figure]]:
    """
    Create financial performance charts.
    
    Args:
        metrics: Financial metrics dictionary
        
    Returns:
        Tuple of (financial_overview_chart, monthly_trend_chart)
    """
    # Financial metrics bar chart
    key_metrics = {
        'Total Revenue': metrics.get('Total Revenue', 0),
        'Total Expenses': metrics.get('Total Expenses', 0),
        'Net Income': metrics.get('Net Income', 0),
        'Total Debt': metrics.get('Total Debt', 0),
        'Debt Repayments': metrics.get('Total Debt Repayments', 0)
    }
    
    fig_financial = go.Figure()
    
    colors = ['green' if v >= 0 else 'red' for v in key_metrics.values()]
    
    fig_financial.add_trace(go.Bar(
        x=list(key_metrics.keys()),
        y=list(key_metrics.values()),
        marker_color=colors,
        text=[f"£{v:,.0f}" for v in key_metrics.values()],
        textposition='outside'
    ))
    
    fig_financial.update_layout(
        title="Financial Overview",
        yaxis_title="Amount (£)",
        showlegend=False,
        height=400
    )
    
    # Monthly trend chart if data available
    fig_trend = None
    if 'monthly_summary' in metrics and metrics['monthly_summary'] is not None and not metrics['monthly_summary'].empty:
        monthly_data = metrics['monthly_summary'].reset_index()
        monthly_data['date'] = monthly_data['year_month'].astype(str)
        
        fig_trend = go.Figure()
        
        fig_trend.add_trace(go.Scatter(
            x=monthly_data['date'],
            y=monthly_data['monthly_revenue'],
            mode='lines+markers',
            name='Revenue',
            line=dict(color='green', width=3)
        ))
        
        fig_trend.add_trace(go.Scatter(
            x=monthly_data['date'],
            y=monthly_data['monthly_expenses'],
            mode='lines+markers',
            name='Expenses',
            line=dict(color='red', width=3)
        ))
        
        fig_trend.update_layout(
            title="Monthly Revenue vs Expenses Trend",
            xaxis_title="Month",
            yaxis_title="Amount (£)",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
    
    return fig_financial, fig_trend


def create_threshold_chart(score_breakdown: Dict[str, Dict[str, Any]]) -> go.Figure:
    """
    Create threshold comparison chart showing actual vs threshold values.
    
    Args:
        score_breakdown: Dictionary of metrics with actual values and thresholds
        
    Returns:
        Plotly figure object
    """
    if not score_breakdown:
        return go.Figure()
    
    metrics_list = list(score_breakdown.keys())
    actual_values = []
    threshold_values = []
    meets_threshold = []
    
    for metric, data in score_breakdown.items():
        actual_values.append(data.get('actual', 0))
        threshold_values.append(data.get('threshold', 0))
        meets_threshold.append(data.get('meets', False))
    
    fig = make_subplots(rows=1, cols=1)
    
    # Actual values
    colors = ['green' if meets else 'red' for meets in meets_threshold]
    
    fig.add_trace(go.Bar(
        name='Actual',
        x=metrics_list,
        y=actual_values,
        marker_color=colors,
        text=[f"{v:.2f}" for v in actual_values],
        textposition='outside'
    ))
    
    # Threshold line
    fig.add_trace(go.Scatter(
        name='Threshold',
        x=metrics_list,
        y=threshold_values,
        mode='lines+markers',
        line=dict(color='black', width=2, dash='dash'),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="Metrics vs Industry Thresholds",
        xaxis_title="Metric",
        yaxis_title="Value",
        height=500,
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def create_monthly_breakdown_charts(
    pivot_counts: pd.DataFrame, 
    pivot_amounts: pd.DataFrame
) -> Tuple[go.Figure, go.Figure]:
    """
    Create monthly breakdown charts by category.
    
    Args:
        pivot_counts: DataFrame of transaction counts by month and category
        pivot_amounts: DataFrame of transaction amounts by month and category
        
    Returns:
        Tuple of (count_chart, amount_chart)
    """
    # Transaction count chart
    months = [str(month) for month in pivot_counts.index]
    
    fig_counts = go.Figure()
    colors = ['#22c55e', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4']
    
    for i, category in enumerate(pivot_counts.columns):
        fig_counts.add_trace(go.Bar(
            name=category,
            x=months,
            y=pivot_counts[category],
            marker_color=colors[i % len(colors)]
        ))
    
    fig_counts.update_layout(
        title="Monthly Transaction Counts by Category",
        xaxis_title="Month",
        yaxis_title="Number of Transactions",
        barmode='stack',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Amount chart
    fig_amounts = go.Figure()
    
    for i, category in enumerate(pivot_amounts.columns):
        fig_amounts.add_trace(go.Bar(
            name=category,
            x=months,
            y=pivot_amounts[category],
            marker_color=colors[i % len(colors)]
        ))
    
    fig_amounts.update_layout(
        title="Monthly Transaction Amounts by Category",
        xaxis_title="Month",
        yaxis_title="Amount (£)",
        barmode='stack',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig_counts, fig_amounts


def create_loans_repayments_charts(analysis: Dict[str, Any]) -> Tuple[Optional[go.Figure], Optional[go.Figure]]:
    """
    Create charts for loans and repayments analysis.
    
    Args:
        analysis: Dictionary containing loans and repayments analysis data
        
    Returns:
        Tuple of (loans_chart, repayments_chart)
    """
    fig_loans = None
    fig_repayments = None
    
    # Loans by month chart
    loans_by_month = analysis.get('loans_by_month')
    if loans_by_month is not None and not loans_by_month.empty:
        fig_loans = go.Figure()
        
        fig_loans.add_trace(go.Bar(
            x=loans_by_month['month_str'],
            y=loans_by_month['sum'],
            name='Loan Amount',
            marker_color='#22c55e',
            text=[f"£{v:,.0f}" for v in loans_by_month['sum']],
            textposition='outside'
        ))
        
        fig_loans.update_layout(
            title="Loans Received by Month",
            xaxis_title="Month",
            yaxis_title="Amount (£)",
            height=400,
            showlegend=False
        )
    
    # Repayments by month chart
    repayments_by_month = analysis.get('repayments_by_month')
    if repayments_by_month is not None and not repayments_by_month.empty:
        fig_repayments = go.Figure()
        
        fig_repayments.add_trace(go.Bar(
            x=repayments_by_month['month_str'],
            y=repayments_by_month['sum'],
            name='Repayment Amount',
            marker_color='#ef4444',
            text=[f"£{v:,.0f}" for v in repayments_by_month['sum']],
            textposition='outside'
        ))
        
        fig_repayments.update_layout(
            title="Debt Repayments by Month",
            xaxis_title="Month",
            yaxis_title="Amount (£)",
            height=400,
            showlegend=False
        )
    
    return fig_loans, fig_repayments


def create_score_comparison_chart(scores: Dict[str, Any]) -> go.Figure:
    """
    Create a comprehensive score comparison chart for subprime context.
    
    Args:
        scores: Dictionary of all calculated scores
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    score_data = {
        'V2 Weighted': scores.get('weighted_score', 0),
        'ML Probability': scores.get('ml_score', 0) or 0,
        'Subprime Optimized': scores.get('subprime_score', 0)
    }
    
    colors = ['#1f77b4', '#d62728', '#2ca02c']
    
    fig.add_trace(go.Bar(
        x=list(score_data.keys()),
        y=list(score_data.values()),
        marker_color=colors,
        text=[f"{v:.1f}%" if k == "ML Probability" else f"{v:.1f}" for k, v in score_data.items()],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Scoring Methods for Subprime Lending",
        yaxis_title="Score",
        showlegend=False,
        height=350,
        yaxis=dict(range=[0, 100])
    )
    
    return fig


def create_cashflow_trend_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    """
    Create a cashflow trend chart showing inflows and outflows over time.
    
    Args:
        df: DataFrame with transaction data
        
    Returns:
        Plotly figure or None if data is insufficient
    """
    if df.empty or 'date' not in df.columns:
        return None
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')
    
    # Ensure categorization columns exist
    if 'is_revenue' not in df.columns or 'is_expense' not in df.columns:
        return None
    
    monthly = df.groupby('month').agg({
        'amount': [
            lambda x: abs(x[df.loc[x.index, 'is_revenue']].sum()) if df.loc[x.index, 'is_revenue'].any() else 0,
            lambda x: abs(x[df.loc[x.index, 'is_expense']].sum()) if df.loc[x.index, 'is_expense'].any() else 0
        ]
    }).round(2)
    
    monthly.columns = ['inflow', 'outflow']
    monthly['net'] = monthly['inflow'] - monthly['outflow']
    monthly = monthly.reset_index()
    monthly['month_str'] = monthly['month'].astype(str)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Inflow',
        x=monthly['month_str'],
        y=monthly['inflow'],
        marker_color='#22c55e'
    ))
    
    fig.add_trace(go.Bar(
        name='Outflow',
        x=monthly['month_str'],
        y=monthly['outflow'],
        marker_color='#ef4444'
    ))
    
    fig.add_trace(go.Scatter(
        name='Net Cashflow',
        x=monthly['month_str'],
        y=monthly['net'],
        mode='lines+markers',
        line=dict(color='#3b82f6', width=3)
    ))
    
    fig.update_layout(
        title="Monthly Cashflow Analysis",
        xaxis_title="Month",
        yaxis_title="Amount (£)",
        barmode='group',
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def create_risk_gauge_chart(score: float, title: str = "Risk Score") -> go.Figure:
    """
    Create a gauge chart for visualizing risk scores.
    
    Args:
        score: Score value (0-100)
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    # Determine color based on score
    if score >= 70:
        color = "green"
    elif score >= 50:
        color = "orange"
    elif score >= 30:
        color = "darkorange"
    else:
        color = "red"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#ffcccb'},
                {'range': [30, 50], 'color': '#fff4cc'},
                {'range': [50, 70], 'color': '#ffffcc'},
                {'range': [70, 100], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig
