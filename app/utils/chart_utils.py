# app/utils/chart_utils.py
"""Interactive chart utilities using Plotly for enhanced visualizations."""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

from ..core.logger import get_logger
from ..core.cache import CacheManager

logger = get_logger("charts")

class ChartGenerator:
    """Generate interactive charts for financial analysis."""
    
    def __init__(self):
        self.color_palette = {
            'revenue': '#2E8B57',      # Sea Green
            'expenses': '#DC143C',      # Crimson
            'profit': '#4169E1',        # Royal Blue
            'debt': '#FF6347',          # Tomato
            'balance': '#9370DB',       # Medium Purple
            'growth': '#32CD32',        # Lime Green
            'decline': '#FF4500',       # Orange Red
            'neutral': '#708090'        # Slate Gray
        }
        
        self.chart_theme = {
            'template': 'plotly_white',
            'font_family': 'Arial, sans-serif',
            'font_size': 12,
            'title_font_size': 16,
            'margin': dict(l=50, r=50, t=80, b=50)
        }
    
    @CacheManager.cache_data(ttl=900)  # 15 minutes
    def create_revenue_expense_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create interactive revenue vs expense chart."""
        
        try:
            # Prepare monthly data
            monthly_data = self._prepare_monthly_data(data)
            
            if monthly_data.empty:
                return self._create_empty_chart("No data available for revenue/expense analysis")
            
            # Create the chart
            fig = go.Figure()
            
            # Add revenue line
            fig.add_trace(go.Scatter(
                x=monthly_data.index,
                y=monthly_data['revenue'],
                mode='lines+markers',
                name='Revenue',
                line=dict(color=self.color_palette['revenue'], width=3),
                marker=dict(size=8),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Date: %{x}<br>' +
                             'Amount: £%{y:,.2f}<extra></extra>'
            ))
            
            # Add expenses line
            fig.add_trace(go.Scatter(
                x=monthly_data.index,
                y=monthly_data['expenses'],
                mode='lines+markers',
                name='Expenses',
                line=dict(color=self.color_palette['expenses'], width=3),
                marker=dict(size=8),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Date: %{x}<br>' +
                             'Amount: £%{y:,.2f}<extra></extra>'
            ))
            
            # Add profit/loss area
            fig.add_trace(go.Scatter(
                x=monthly_data.index,
                y=monthly_data['profit'],
                mode='lines',
                name='Net Profit/Loss',
                line=dict(color=self.color_palette['profit'], width=2, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(65, 105, 225, 0.1)',
                hovertemplate='<b>Net Profit/Loss</b><br>' +
                             'Date: %{x}<br>' +
                             'Amount: £%{y:,.2f}<extra></extra>'
            ))
            
            # Update layout
            fig.update_layout(
                title='Revenue vs Expenses Over Time',
                xaxis_title='Date',
                yaxis_title='Amount (£)',
                hovermode='x unified',
                **self.chart_theme
            )
            
            # Add annotations for key insights
            self._add_revenue_insights(fig, monthly_data)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating revenue/expense chart: {str(e)}")
            return self._create_error_chart("Error creating revenue/expense chart")
    
    @CacheManager.cache_data(ttl=900)
    def create_cash_flow_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create interactive cash flow chart with balance tracking."""
        
        try:
            # Prepare daily cash flow data
            daily_data = self._prepare_daily_cashflow(data)
            
            if daily_data.empty:
                return self._create_empty_chart("No data available for cash flow analysis")
            
            # Create subplot with secondary y-axis
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Daily Cash Flow', 'Account Balance'),
                vertical_spacing=0.1,
                row_heights=[0.6, 0.4]
            )
            
            # Cash flow chart (top)
            fig.add_trace(
                go.Bar(
                    x=daily_data.index,
                    y=daily_data['net_flow'],
                    name='Net Cash Flow',
                    marker_color=np.where(daily_data['net_flow'] >= 0, 
                                        self.color_palette['revenue'], 
                                        self.color_palette['expenses']),
                    hovertemplate='Date: %{x}<br>' +
                                 'Net Flow: £%{y:,.2f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Balance chart (bottom)
            if 'balance' in daily_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=daily_data.index,
                        y=daily_data['balance'],
                        mode='lines',
                        name='Account Balance',
                        line=dict(color=self.color_palette['balance'], width=2),
                        hovertemplate='Date: %{x}<br>' +
                                     'Balance: £%{y:,.2f}<extra></extra>'
                    ),
                    row=2, col=1
                )
                
                # Add zero line for balance
                fig.add_hline(y=0, line_dash="dash", line_color="red", 
                             annotation_text="Zero Balance", row=2, col=1)
            
            # Update layout
            fig.update_layout(
                title='Cash Flow and Balance Analysis',
                height=600,
                showlegend=True,
                **self.chart_theme
            )
            
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Cash Flow (£)", row=1, col=1)
            fig.update_yaxes(title_text="Balance (£)", row=2, col=1)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating cash flow chart: {str(e)}")
            return self._create_error_chart("Error creating cash flow chart")
    
    @CacheManager.cache_data(ttl=900)
    def create_transaction_category_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create interactive transaction category breakdown."""
        
        try:
            if 'subcategory' not in data.columns:
                return self._create_empty_chart("No category data available")
            
            # Prepare category data
            category_data = data.groupby('subcategory')['amount'].sum().sort_values(ascending=False)
            
            if category_data.empty:
                return self._create_empty_chart("No transaction categories found")
            
            # Create pie chart
            fig = go.Figure(data=[go.Pie(
                labels=category_data.index,
                values=category_data.values,
                hole=0.4,
                hovertemplate='<b>%{label}</b><br>' +
                             'Amount: £%{value:,.2f}<br>' +
                             'Percentage: %{percent}<extra></extra>',
                textinfo='label+percent',
                textposition='outside'
            )])
            
            # Update layout
            fig.update_layout(
                title='Transaction Categories Breakdown',
                annotations=[dict(text='Total<br>Transactions', x=0.5, y=0.5, 
                                font_size=14, showarrow=False)],
                **self.chart_theme
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating category chart: {str(e)}")
            return self._create_error_chart("Error creating category chart")
    
    @CacheManager.cache_data(ttl=900)
    def create_seasonal_analysis_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create seasonal analysis chart."""
        
        try:
            if len(data) < 12:  # Need at least some data for seasonal analysis
                return self._create_empty_chart("Insufficient data for seasonal analysis")
            
            # Prepare seasonal data
            seasonal_data = self._prepare_seasonal_data(data)
            
            # Create subplot for monthly and quarterly patterns
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Monthly Revenue Pattern', 'Quarterly Comparison'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Monthly pattern (left)
            fig.add_trace(
                go.Bar(
                    x=seasonal_data['monthly'].index,
                    y=seasonal_data['monthly']['revenue'],
                    name='Monthly Revenue',
                    marker_color=self.color_palette['revenue'],
                    hovertemplate='Month: %{x}<br>Revenue: £%{y:,.2f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Quarterly pattern (right)
            fig.add_trace(
                go.Bar(
                    x=seasonal_data['quarterly'].index,
                    y=seasonal_data['quarterly']['revenue'],
                    name='Quarterly Revenue',
                    marker_color=self.color_palette['profit'],
                    hovertemplate='Quarter: Q%{x}<br>Revenue: £%{y:,.2f}<extra></extra>'
                ),
                row=1, col=2
            )
            
            # Update layout
            fig.update_layout(
                title='Seasonal Revenue Analysis',
                height=400,
                showlegend=False,
                **self.chart_theme
            )
            
            fig.update_xaxes(title_text="Month", row=1, col=1)
            fig.update_xaxes(title_text="Quarter", row=1, col=2)
            fig.update_yaxes(title_text="Revenue (£)", row=1, col=1)
            fig.update_yaxes(title_text="Revenue (£)", row=1, col=2)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating seasonal chart: {str(e)}")
            return self._create_error_chart("Error creating seasonal chart")
    
    @CacheManager.cache_data(ttl=900)
    def create_financial_metrics_dashboard(self, metrics: Dict[str, Any]) -> go.Figure:
        """Create a comprehensive metrics dashboard."""
        
        try:
            # Create a 2x2 subplot dashboard
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Profitability Metrics',
                    'Liquidity Metrics', 
                    'Debt Management',
                    'Growth Indicators'
                ),
                specs=[[{"type": "bar"}, {"type": "indicator"}],
                       [{"type": "bar"}, {"type": "scatter"}]]
            )
            
            # Profitability metrics (top-left)
            profitability_metrics = {
                'Operating Margin': metrics.get('Operating Margin', 0) * 100,
                'Net Income': metrics.get('Net Income', 0),
                'Gross Profit': metrics.get('Total Revenue', 0) - metrics.get('Total Expenses', 0)
            }
            
            fig.add_trace(
                go.Bar(
                    x=list(profitability_metrics.keys()),
                    y=list(profitability_metrics.values()),
                    name='Profitability',
                    marker_color=self.color_palette['profit']
                ),
                row=1, col=1
            )
            
            # Liquidity indicator (top-right)
            current_ratio = metrics.get('Average Month-End Balance', 0) / max(metrics.get('Gross Burn Rate', 1), 1)
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=current_ratio,
                    title={'text': "Months of Runway"},
                    gauge={'axis': {'range': [None, 12]},
                           'bar': {'color': self.color_palette['balance']},
                           'steps': [
                               {'range': [0, 3], 'color': "lightgray"},
                               {'range': [3, 6], 'color': "yellow"},
                               {'range': [6, 12], 'color': "lightgreen"}
                           ],
                           'threshold': {'line': {'color': "red", 'width': 4},
                                       'thickness': 0.75, 'value': 6}}
                ),
                row=1, col=2
            )
            
            # Debt management (bottom-left)
            debt_metrics = {
                'DSCR': metrics.get('Debt Service Coverage Ratio', 0),
                'Debt/Income': metrics.get('Debt-to-Income Ratio', 0),
                'Total Debt': metrics.get('Total Debt', 0) / 1000  # Convert to thousands
            }
            
            fig.add_trace(
                go.Bar(
                    x=list(debt_metrics.keys()),
                    y=list(debt_metrics.values()),
                    name='Debt Management',
                    marker_color=self.color_palette['debt']
                ),
                row=2, col=1
            )
            
            # Growth trend (bottom-right)
            growth_rate = metrics.get('Revenue Growth Rate', 0)
            months = list(range(1, 7))  # Sample 6 months
            projected_growth = [growth_rate * i for i in months]
            
            fig.add_trace(
                go.Scatter(
                    x=months,
                    y=projected_growth,
                    mode='lines+markers',
                    name='Growth Projection',
                    line=dict(color=self.color_palette['growth'] if growth_rate > 0 else self.color_palette['decline'])
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                title='Financial Metrics Dashboard',
                height=600,
                showlegend=False,
                **self.chart_theme
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating metrics dashboard: {str(e)}")
            return self._create_error_chart("Error creating metrics dashboard")
    
    def create_risk_assessment_chart(self, metrics: Dict[str, Any], industry_benchmarks: Dict[str, Any]) -> go.Figure:
        """Create risk assessment radar chart."""
        
        try:
            # Define risk categories and their values
            risk_categories = [
                'Profitability', 'Liquidity', 'Debt Management', 
                'Cash Flow Stability', 'Growth Potential', 'Payment Reliability'
            ]
            
            # Calculate risk scores (0-100, where 100 is best)
            risk_scores = self._calculate_risk_scores(metrics, industry_benchmarks)
            
            # Create radar chart
            fig = go.Figure()
            
            # Add actual performance
            fig.add_trace(go.Scatterpolar(
                r=list(risk_scores.values()),
                theta=risk_categories,
                fill='toself',
                name='Current Performance',
                line_color=self.color_palette['profit']
            ))
            
            # Add industry benchmark
            benchmark_scores = [75] * len(risk_categories)  # Sample benchmark
            fig.add_trace(go.Scatterpolar(
                r=benchmark_scores,
                theta=risk_categories,
                fill='toself',
                name='Industry Benchmark',
                line_color=self.color_palette['neutral'],
                opacity=0.6
            ))
            
            # Update layout
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                showlegend=True,
                title='Risk Assessment Profile',
                **self.chart_theme
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating risk assessment chart: {str(e)}")
            return self._create_error_chart("Error creating risk assessment chart")
    
    def create_transaction_timeline(self, data: pd.DataFrame) -> go.Figure:
        """Create interactive transaction timeline."""
        
        try:
            # Prepare timeline data
            timeline_data = self._prepare_timeline_data(data)
            
            if timeline_data.empty:
                return self._create_empty_chart("No transaction data available for timeline")
            
            # Create timeline chart
            fig = go.Figure()
            
            # Add different transaction types
            for category, cat_data in timeline_data.groupby('subcategory'):
                fig.add_trace(go.Scatter(
                    x=cat_data['date'],
                    y=cat_data['amount'],
                    mode='markers',
                    name=category,
                    marker=dict(
                        size=cat_data['amount'] / cat_data['amount'].max() * 20 + 5,
                        opacity=0.7
                    ),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                 'Date: %{x}<br>' +
                                 'Amount: £%{y:,.2f}<br>' +
                                 'Description: %{text}<extra></extra>',
                    text=cat_data['transaction_name']
                ))
            
            # Update layout
            fig.update_layout(
                title='Transaction Timeline',
                xaxis_title='Date',
                yaxis_title='Amount (£)',
                hovermode='closest',
                **self.chart_theme
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating transaction timeline: {str(e)}")
            return self._create_error_chart("Error creating transaction timeline")
    
    def _prepare_monthly_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare monthly aggregated data."""
        
        if data.empty or 'date' not in data.columns:
            return pd.DataFrame()
        
        # Ensure we have the required boolean columns
        if 'is_revenue' not in data.columns:
            data['is_revenue'] = data.get('subcategory', '').isin(['Income', 'Special Inflow'])
        if 'is_expense' not in data.columns:
            data['is_expense'] = data.get('subcategory', '').isin(['Expenses', 'Special Outflow'])
        
        # Group by month
        monthly_data = data.groupby(data['date'].dt.to_period('M')).agg({
            'amount': [
                lambda x: x[data.loc[x.index, 'is_revenue']].sum(),  # Revenue
                lambda x: x[data.loc[x.index, 'is_expense']].sum(),  # Expenses
                'count'  # Transaction count
            ]
        }).round(2)
        
        # Flatten column names
        monthly_data.columns = ['revenue', 'expenses', 'transaction_count']
        monthly_data['profit'] = monthly_data['revenue'] - monthly_data['expenses']
        
        return monthly_data
    
    def _prepare_daily_cashflow(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare daily cash flow data."""
        
        if data.empty or 'date' not in data.columns:
            return pd.DataFrame()
        
        # Ensure we have the required boolean columns
        if 'is_revenue' not in data.columns:
            data['is_revenue'] = data.get('subcategory', '').isin(['Income', 'Special Inflow'])
        if 'is_expense' not in data.columns:
            data['is_expense'] = data.get('subcategory', '').isin(['Expenses', 'Special Outflow'])
        
        # Group by day
        daily_data = data.groupby(data['date'].dt.date).agg({
            'amount': [
                lambda x: x[data.loc[x.index, 'is_revenue']].sum(),  # Daily revenue
                lambda x: x[data.loc[x.index, 'is_expense']].sum(),  # Daily expenses
            ]
        }).round(2)
        
        # Flatten column names
        daily_data.columns = ['inflow', 'outflow']
        daily_data['net_flow'] = daily_data['inflow'] - daily_data['outflow']
        
        # Add balance if available
        if 'calculated_balance' in data.columns:
            balance_data = data.groupby(data['date'].dt.date)['calculated_balance'].last()
            daily_data['balance'] = balance_data
        
        return daily_data
    
    def _prepare_seasonal_data(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Prepare seasonal analysis data."""
        
        if data.empty or 'date' not in data.columns:
            return {'monthly': pd.DataFrame(), 'quarterly': pd.DataFrame()}
        
        # Ensure we have revenue data
        if 'is_revenue' not in data.columns:
            data['is_revenue'] = data.get('subcategory', '').isin(['Income', 'Special Inflow'])
        
        revenue_data = data[data['is_revenue']]
        
        # Monthly patterns
        monthly_data = revenue_data.groupby(revenue_data['date'].dt.month).agg({
            'amount': 'sum'
        }).round(2)
        monthly_data.columns = ['revenue']
        monthly_data.index = [
            'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
        ][:len(monthly_data)]
        
        # Quarterly patterns
        quarterly_data = revenue_data.groupby(revenue_data['date'].dt.quarter).agg({
            'amount': 'sum'
        }).round(2)
        quarterly_data.columns = ['revenue']
        
        return {
            'monthly': monthly_data,
            'quarterly': quarterly_data
        }
    
    def _calculate_risk_scores(self, metrics: Dict[str, Any], benchmarks: Dict[str, Any]) -> Dict[str, float]:
        """Calculate risk scores for radar chart."""
        
        scores = {}
        
        # Profitability (Operating Margin)
        operating_margin = metrics.get('Operating Margin', 0)
        scores['Profitability'] = min(100, max(0, operating_margin * 1000))  # Scale to 0-100
        
        # Liquidity (Months of runway)
        runway = metrics.get('Average Month-End Balance', 0) / max(metrics.get('Gross Burn Rate', 1), 1)
        scores['Liquidity'] = min(100, runway * 10)  # Scale to 0-100
        
        # Debt Management (DSCR)
        dscr = metrics.get('Debt Service Coverage Ratio', 0)
        scores['Debt Management'] = min(100, dscr * 40)  # Scale to 0-100
        
        # Cash Flow Stability (inverse of volatility)
        volatility = metrics.get('Cash Flow Volatility', 1)
        scores['Cash Flow Stability'] = max(0, 100 - volatility * 200)  # Scale to 0-100
        
        # Growth Potential
        growth_rate = metrics.get('Revenue Growth Rate', 0)
        scores['Growth Potential'] = min(100, max(0, 50 + growth_rate * 5))  # Scale to 0-100
        
        # Payment Reliability
        bounced_payments = metrics.get('Number of Bounced Payments', 0)
        scores['Payment Reliability'] = max(0, 100 - bounced_payments * 20)  # Scale to 0-100
        
        return scores
    
    def _prepare_timeline_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for transaction timeline."""
        
        if data.empty:
            return pd.DataFrame()
        
        # Select relevant columns
        timeline_cols = ['date', 'amount', 'subcategory']
        if 'transaction_name' in data.columns:
            timeline_cols.append('transaction_name')
        elif 'name_y' in data.columns:
            timeline_cols.append('name_y')
            data = data.rename(columns={'name_y': 'transaction_name'})
        
        # Filter and clean data
        timeline_data = data[timeline_cols].copy()
        timeline_data = timeline_data.dropna(subset=['date', 'amount'])
        
        # Limit to recent transactions for performance
        if len(timeline_data) > 1000:
            timeline_data = timeline_data.head(1000)
        
        return timeline_data
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create an empty chart with a message."""
        
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            **self.chart_theme
        )
        return fig
    
    def _create_error_chart(self, message: str) -> go.Figure:
        """Create an error chart with a message."""
        
        fig = go.Figure()
        fig.add_annotation(
            text=f"⚠️ {message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            **self.chart_theme
        )
        return fig
    
    def _add_revenue_insights(self, fig: go.Figure, monthly_data: pd.DataFrame) -> None:
        """Add insights annotations to revenue chart."""
        
        try:
            # Find best and worst months
            best_month = monthly_data['revenue'].idxmax()
            worst_month = monthly_data['revenue'].idxmin()
            
            # Add annotations
            fig.add_annotation(
                x=best_month,
                y=monthly_data.loc[best_month, 'revenue'],
                text=f"Peak: £{monthly_data.loc[best_month, 'revenue']:,.0f}",
                showarrow=True,
                arrowhead=2,
                arrowcolor="green",
                bgcolor="lightgreen",
                bordercolor="green"
            )
            
            fig.add_annotation(
                x=worst_month,
                y=monthly_data.loc[worst_month, 'revenue'],
                text=f"Low: £{monthly_data.loc[worst_month, 'revenue']:,.0f}",
                showarrow=True,
                arrowhead=2,
                arrowcolor="red",
                bgcolor="lightcoral",
                bordercolor="red"
            )
            
        except Exception as e:
            logger.warning(f"Could not add revenue insights: {str(e)}")

# Global chart generator instance
chart_generator = ChartGenerator()

# Streamlit integration functions
def display_chart(fig: go.Figure, use_container_width: bool = True) -> None:
    """Display Plotly chart in Streamlit with error handling."""
    
    try:
        st.plotly_chart(fig, use_container_width=use_container_width)
    except Exception as e:
        logger.error(f"Error displaying chart: {str(e)}")
        st.error("Error displaying chart. Please try again.")

def create_metric_cards(metrics: Dict[str, Any]) -> None:
    """Create metric cards for dashboard display."""
    
    try:
        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            revenue = metrics.get('Total Revenue', 0)
            revenue_growth = metrics.get('Revenue Growth Rate', 0)
            st.metric(
                label="Total Revenue",
                value=f"£{revenue:,.2f}",
                delta=f"{revenue_growth:+.1f}%"
            )
        
        with col2:
            operating_margin = metrics.get('Operating Margin', 0)
            st.metric(
                label="Operating Margin",
                value=f"{operating_margin*100:.1f}%",
                delta=None
            )
        
        with col3:
            dscr = metrics.get('Debt Service Coverage Ratio', 0)
            st.metric(
                label="Debt Service Coverage",
                value=f"{dscr:.2f}",
                delta=None
            )
        
        with col4:
            cash_flow_vol = metrics.get('Cash Flow Volatility', 0)
            st.metric(
                label="Cash Flow Volatility",
                value=f"{cash_flow_vol:.3f}",
                delta=None,
                delta_color="inverse"  # Lower is better
            )
    
    except Exception as e:
        logger.error(f"Error creating metric cards: {str(e)}")
        st.error("Error displaying metrics. Please try again.")