# app/services/financial_analyzer.py
"""Enhanced financial analysis service with advanced metrics and benchmarking."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
from collections import defaultdict

from ..core.exceptions import InsufficientDataError, DataValidationError
from ..core.logger import get_logger, log_performance
from ..core.cache import CacheManager, DataCache
from ..core.validators import DataValidator
from ..config.industry_config import INDUSTRY_THRESHOLDS

logger = get_logger("financial_analyzer")

class FinancialAnalyzer:
    """Enhanced financial analysis with advanced metrics and benchmarking."""
    
    def __init__(self):
        self.data_validator = DataValidator()
    
    @log_performance(logger)
    @CacheManager.cache_data(ttl=1800)  # 30 minutes
    def calculate_comprehensive_metrics(
        self, 
        data: pd.DataFrame, 
        company_age_months: int,
        include_advanced: bool = True
    ) -> Dict[str, Any]:
        """Calculate comprehensive financial metrics with advanced analytics."""
        
        # Validate input data
        data = self.data_validator.validate_transaction_data(data)
        
        if data.empty:
            raise InsufficientDataError("No transaction data available for analysis")
        
        # Basic metrics
        basic_metrics = self._calculate_basic_metrics(data, company_age_months)
        
        # Advanced metrics
        advanced_metrics = {}
        if include_advanced:
            advanced_metrics = self._calculate_advanced_metrics(data)
        
        # Risk indicators
        risk_metrics = self._calculate_risk_indicators(data)
        
        # Seasonal analysis
        seasonal_metrics = self._calculate_seasonal_metrics(data)
        
        # Trend analysis
        trend_metrics = self._calculate_trend_analysis(data)
        
        # Combine all metrics
        all_metrics = {
            **basic_metrics,
            **advanced_metrics,
            **risk_metrics,
            **seasonal_metrics,
            **trend_metrics,
            'analysis_date': datetime.now().isoformat(),
            'data_period': self._get_data_period(data)
        }
        
        logger.info(f"Calculated {len(all_metrics)} financial metrics")
        return all_metrics
    
    def _calculate_basic_metrics(self, data: pd.DataFrame, company_age_months: int) -> Dict[str, Any]:
        """Calculate basic financial metrics."""
        
        # Ensure boolean columns exist
        if 'is_revenue' not in data.columns:
            data = self._categorize_transactions(data)
        
        # Basic financial totals
        total_revenue = round(data.loc[data['is_revenue'], 'amount'].sum() or 0, 2)
        total_expenses = round(data.loc[data['is_expense'], 'amount'].sum() or 0, 2)
        net_income = round(total_revenue - total_expenses, 2)
        total_debt_repayments = round(data.loc[data['is_debt_repayment'], 'amount'].sum() or 0, 2)
        total_debt = round(data.loc[data['is_debt'], 'amount'].sum() or 0, 2)
        
        # Financial ratios
        debt_to_income_ratio = round(total_debt / total_revenue if total_revenue != 0 else 0, 2)
        expense_to_revenue_ratio = round(total_expenses / total_revenue if total_revenue != 0 else 0, 2)
        operating_income = total_revenue - total_expenses
        operating_margin = round(operating_income / total_revenue if total_revenue != 0 else 0, 2)
        debt_service_coverage_ratio = round(total_revenue / total_debt_repayments if total_debt_repayments != 0 else 0, 2)
        
        # Time-based analysis
        data['date'] = pd.to_datetime(data['date'])
        data['year_month'] = data['date'].dt.to_period('M')
        
        # Monthly summaries
        monthly_summary = data.groupby('year_month').agg(
            net_cashflow=('amount', lambda x: x[data['is_revenue']].sum() - x[data['is_expense']].sum()),
            monthly_revenue=('amount', lambda x: x[data['is_revenue']].sum()),
            monthly_expenses=('amount', lambda x: x[data['is_expense']].sum())
        ).reset_index()
        
        # Calculate derived metrics
        total_months = len(monthly_summary)
        gross_burn_rate = round(monthly_summary['monthly_expenses'].sum() / total_months if total_months > 0 else 0, 2)
        
        # Cash flow volatility
        cash_flow_mean = monthly_summary['net_cashflow'].mean()
        cash_flow_std = monthly_summary['net_cashflow'].std()
        cash_flow_volatility = round((cash_flow_std / cash_flow_mean) if cash_flow_mean != 0 else 0, 2)
        
        # Revenue growth rate
        revenue_growth_rate = round(monthly_summary['monthly_revenue'].pct_change().median() * 100, 2)
        
        # Monthly average revenue
        months_in_data = data['date'].dt.to_period('M').nunique()
        monthly_average_revenue = round(total_revenue / months_in_data, 2) if months_in_data else 0
        
        # Balance analysis
        balance_metrics = self._calculate_balance_metrics(data)
        
        # Bounced payments
        bounced_payments_count = self._count_bounced_payments(data)
        
        return {
            "Total Revenue": total_revenue,
            "Monthly Average Revenue": monthly_average_revenue,
            "Total Expenses": total_expenses,
            "Net Income": net_income,
            "Total Debt Repayments": total_debt_repayments,
            "Total Debt": total_debt,
            "Debt-to-Income Ratio": debt_to_income_ratio,
            "Expense-to-Revenue Ratio": expense_to_revenue_ratio,
            "Operating Margin": operating_margin,
            "Debt Service Coverage Ratio": debt_service_coverage_ratio,
            "Gross Burn Rate": gross_burn_rate,
            "Cash Flow Volatility": cash_flow_volatility,
            "Revenue Growth Rate": revenue_growth_rate,
            **balance_metrics,
            "Number of Bounced Payments": bounced_payments_count
        }
    
    def _calculate_advanced_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate advanced financial metrics."""
        
        advanced_metrics = {}
        
        try:
            # Working capital analysis
            advanced_metrics.update(self._calculate_working_capital_metrics(data))
            
            # Liquidity ratios
            advanced_metrics.update(self._calculate_liquidity_ratios(data))
            
            # Efficiency metrics
            advanced_metrics.update(self._calculate_efficiency_metrics(data))
            
            # Profitability analysis
            advanced_metrics.update(self._calculate_profitability_metrics(data))
            
            # Customer concentration analysis
            advanced_metrics.update(self._analyze_customer_concentration(data))
            
        except Exception as e:
            logger.warning(f"Error calculating advanced metrics: {str(e)}")
            # Continue with basic metrics if advanced calculation fails
        
        return advanced_metrics
    
    def _calculate_working_capital_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate working capital related metrics."""
        
        # Estimate working capital from cash flow patterns
        monthly_data = data.groupby(data['date'].dt.to_period('M')).agg(
            revenue=('amount', lambda x: x[data['is_revenue']].sum()),
            expenses=('amount', lambda x: x[data['is_expense']].sum())
        )
        
        # Working capital proxy (revenue - expenses variability)
        working_capital_volatility = monthly_data['revenue'].std() / monthly_data['revenue'].mean() if monthly_data['revenue'].mean() > 0 else 0
        
        return {
            "Working Capital Volatility": round(working_capital_volatility, 3),
            "Average Monthly Working Capital": round((monthly_data['revenue'] - monthly_data['expenses']).mean(), 2)
        }
    
    def _calculate_liquidity_ratios(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate liquidity-related ratios."""
        
        # Quick ratio proxy using balance data
        if 'balances.available' in data.columns:
            avg_balance = data['balances.available'].mean()
            monthly_expenses = data.loc[data['is_expense'], 'amount'].sum() / max(1, data['date'].dt.to_period('M').nunique())
            
            months_of_runway = avg_balance / monthly_expenses if monthly_expenses > 0 else 0
            
            return {
                "Average Available Balance": round(avg_balance, 2),
                "Months of Runway": round(months_of_runway, 1)
            }
        
        return {}
    
    def _calculate_efficiency_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate operational efficiency metrics."""
        
        # Transaction frequency analysis
        daily_transactions = data.groupby(data['date'].dt.date).size()
        avg_daily_transactions = daily_transactions.mean()
        
        # Revenue per transaction
        revenue_transactions = data[data['is_revenue']]
        if not revenue_transactions.empty:
            avg_revenue_per_transaction = revenue_transactions['amount'].mean()
            revenue_transaction_frequency = len(revenue_transactions) / data['date'].dt.to_period('D').nunique()
        else:
            avg_revenue_per_transaction = 0
            revenue_transaction_frequency = 0
        
        return {
            "Average Daily Transactions": round(avg_daily_transactions, 1),
            "Average Revenue per Transaction": round(avg_revenue_per_transaction, 2),
            "Revenue Transaction Frequency": round(revenue_transaction_frequency, 2)
        }
    
    def _calculate_profitability_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate detailed profitability metrics."""
        
        # Monthly profitability analysis
        monthly_profit = data.groupby(data['date'].dt.to_period('M')).agg(
            revenue=('amount', lambda x: x[data['is_revenue']].sum()),
            expenses=('amount', lambda x: x[data['is_expense']].sum())
        )
        
        monthly_profit['profit'] = monthly_profit['revenue'] - monthly_profit['expenses']
        monthly_profit['profit_margin'] = monthly_profit['profit'] / monthly_profit['revenue']
        
        # Profitability consistency
        profit_consistency = 1 - (monthly_profit['profit_margin'].std() / monthly_profit['profit_margin'].mean()) if monthly_profit['profit_margin'].mean() != 0 else 0
        
        # Months profitable
        profitable_months = (monthly_profit['profit'] > 0).sum()
        total_months = len(monthly_profit)
        profitability_ratio = profitable_months / total_months if total_months > 0 else 0
        
        return {
            "Profit Consistency Score": round(max(0, profit_consistency), 3),
            "Profitable Months Ratio": round(profitability_ratio, 3),
            "Average Monthly Profit": round(monthly_profit['profit'].mean(), 2),
            "Best Month Profit": round(monthly_profit['profit'].max(), 2),
            "Worst Month Profit": round(monthly_profit['profit'].min(), 2)
        }
    
    def _analyze_customer_concentration(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze customer/revenue source concentration."""
        
        revenue_data = data[data['is_revenue']]
        
        if revenue_data.empty or 'name_y' not in revenue_data.columns:
            return {"Revenue Source Concentration": "Unable to analyze"}
        
        # Revenue by source
        revenue_by_source = revenue_data.groupby('name_y')['amount'].sum().sort_values(ascending=False)
        
        if len(revenue_by_source) == 0:
            return {"Revenue Source Concentration": "No revenue sources identified"}
        
        total_revenue = revenue_by_source.sum()
        
        # Concentration metrics
        top_1_concentration = revenue_by_source.iloc[0] / total_revenue if len(revenue_by_source) > 0 else 0
        top_3_concentration = revenue_by_source.head(3).sum() / total_revenue if len(revenue_by_source) > 0 else 0
        
        # Herfindahl-Hirschman Index (HHI) for concentration
        market_shares = revenue_by_source / total_revenue
        hhi = (market_shares ** 2).sum()
        
        return {
            "Number of Revenue Sources": len(revenue_by_source),
            "Top Customer Concentration": round(top_1_concentration, 3),
            "Top 3 Customers Concentration": round(top_3_concentration, 3),
            "Revenue Concentration Index (HHI)": round(hhi, 3),
            "Revenue Diversification Score": round(1 - hhi, 3)  # Higher is better
        }
    
    def _calculate_risk_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk-related indicators."""
        
        risk_metrics = {}
        
        # Payment failure analysis
        if 'name_y' in data.columns:
            risk_metrics.update(self._analyze_payment_failures(data))
        
        # Cash flow stress testing
        risk_metrics.update(self._calculate_stress_indicators(data))
        
        # Transaction pattern anomalies
        risk_metrics.update(self._detect_transaction_anomalies(data))
        
        return risk_metrics
    
    def _analyze_payment_failures(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze payment failures and bounced transactions."""
        
        # Keywords indicating payment failures
        failure_keywords = [
            'unpaid', 'returned', 'bounced', 'insufficient', 'failed', 
            'declined', 'reversed', 'chargeback', 'nsf', 'unp'
        ]
        
        descriptions = data['name_y'].fillna('').str.lower()
        
        failed_payments = 0
        for keyword in failure_keywords:
            failed_payments += descriptions.str.contains(keyword, na=False).sum()
        
        total_transactions = len(data)
        failure_rate = failed_payments / total_transactions if total_transactions > 0 else 0
        
        return {
            "Payment Failure Count": failed_payments,
            "Payment Failure Rate": round(failure_rate, 4),
            "Payment Reliability Score": round(1 - failure_rate, 4)
        }
    
    def _calculate_stress_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate financial stress indicators."""
        
        # Daily cash flow analysis
        daily_cashflow = data.groupby(data['date'].dt.date).agg(
            inflow=('amount', lambda x: x[data['is_revenue']].sum()),
            outflow=('amount', lambda x: x[data['is_expense']].sum())
        )
        
        daily_cashflow['net_flow'] = daily_cashflow['inflow'] - daily_cashflow['outflow']
