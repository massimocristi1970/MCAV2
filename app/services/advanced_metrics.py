# app/services/advanced_metrics.py
"""
Advanced Risk Metrics Calculator

Provides additional risk signals not covered by basic financial metrics:
- Transaction patterns and regularity
- Customer/revenue concentration
- Seasonal adjustment
- Banking behavior indicators
- Debt stacking detection
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import re


class AdvancedMetricsCalculator:
    """
    Calculates advanced risk metrics from transaction data.
    
    These metrics complement traditional financial ratios by examining
    transaction patterns and behavioral signals.
    """
    
    def __init__(self):
        # Configuration for metric calculations
        self.config = {
            'min_transactions_for_analysis': 20,
            'lookback_days': 90,
            'seasonality_min_months': 6,
            'concentration_threshold': 0.5,  # 50% from single source = high concentration
        }
    
    def calculate_all_metrics(
        self,
        transactions_df: pd.DataFrame,
        company_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate all advanced metrics from transaction data.
        
        Args:
            transactions_df: DataFrame with transaction data
            company_info: Optional company information
            
        Returns:
            Dictionary of all calculated advanced metrics
        """
        if transactions_df.empty or len(transactions_df) < self.config['min_transactions_for_analysis']:
            return self._empty_metrics()
        
        # Ensure date column is datetime
        df = transactions_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        metrics = {}
        
        # Transaction pattern metrics
        metrics.update(self.calculate_transaction_patterns(df))
        
        # Revenue concentration metrics
        metrics.update(self.calculate_revenue_concentration(df))
        
        # Seasonality metrics
        metrics.update(self.calculate_seasonality(df))
        
        # Banking behavior metrics
        metrics.update(self.calculate_banking_behavior(df))
        
        # Debt stacking detection
        metrics.update(self.calculate_debt_indicators(df))
        
        # Overall risk score from advanced metrics
        metrics['advanced_risk_score'] = self._calculate_advanced_risk_score(metrics)
        
        return metrics
    
    def calculate_transaction_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze transaction patterns for MCA suitability.
        
        Key indicators:
        - Deposit frequency and regularity
        - Transaction velocity
        - Inflow consistency
        """
        metrics = {}
        
        # Identify credit transactions (deposits/income)
        if 'amount' in df.columns:
            df['is_credit'] = df['amount'] < 0  # Negative = money in
            credits = df[df['is_credit']].copy()
        else:
            return {'transaction_pattern_error': 'Missing amount column'}
        
        if credits.empty:
            return {
                'deposit_frequency_score': 0,
                'transaction_regularity': 0,
                'inflow_days_30d': 0,
                'max_inflow_gap_days': 30,
                'inflow_cv': 2.0
            }
        
        # Last 30 days inflow analysis
        latest_date = df['date'].max()
        last_30_days = credits[credits['date'] >= (latest_date - timedelta(days=30))]
        
        metrics['inflow_days_30d'] = last_30_days['date'].dt.date.nunique()
        
        # Calculate deposit frequency score (0-100)
        # 20+ days with deposits in 30 days = excellent (100)
        # 10-20 days = good (50-100)
        # <10 days = poor (0-50)
        inflow_days = metrics['inflow_days_30d']
        if inflow_days >= 20:
            metrics['deposit_frequency_score'] = 100
        elif inflow_days >= 15:
            metrics['deposit_frequency_score'] = 80
        elif inflow_days >= 10:
            metrics['deposit_frequency_score'] = 60
        elif inflow_days >= 5:
            metrics['deposit_frequency_score'] = 40
        else:
            metrics['deposit_frequency_score'] = max(0, inflow_days * 4)
        
        # Max gap between inflow days
        inflow_dates = credits['date'].dt.normalize().drop_duplicates().sort_values()
        if len(inflow_dates) > 1:
            gaps = inflow_dates.diff().dt.days.dropna()
            metrics['max_inflow_gap_days'] = float(gaps.max()) if len(gaps) > 0 else 0
            metrics['avg_inflow_gap_days'] = float(gaps.mean()) if len(gaps) > 0 else 0
        else:
            metrics['max_inflow_gap_days'] = 30
            metrics['avg_inflow_gap_days'] = 30
        
        # Inflow coefficient of variation (consistency)
        df['year_month'] = df['date'].dt.to_period('M')
        monthly_inflows = credits.groupby('year_month')['amount'].sum().abs()
        
        if len(monthly_inflows) > 1:
            mean_inflow = monthly_inflows.mean()
            std_inflow = monthly_inflows.std()
            metrics['inflow_cv'] = float(std_inflow / mean_inflow) if mean_inflow > 0 else 2.0
        else:
            metrics['inflow_cv'] = 0.5  # Single month - neutral
        
        # Transaction regularity score based on consistency
        cv = metrics['inflow_cv']
        if cv <= 0.3:
            metrics['transaction_regularity'] = 100
        elif cv <= 0.5:
            metrics['transaction_regularity'] = 80
        elif cv <= 0.7:
            metrics['transaction_regularity'] = 60
        elif cv <= 1.0:
            metrics['transaction_regularity'] = 40
        else:
            metrics['transaction_regularity'] = max(0, 100 - int(cv * 50))
        
        return metrics
    
    def calculate_revenue_concentration(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze revenue source concentration.
        
        High concentration from few sources = higher risk
        Diverse revenue sources = lower risk
        """
        metrics = {}
        
        # Get credit transactions
        credits = df[df['amount'] < 0].copy() if 'amount' in df.columns else pd.DataFrame()
        
        if credits.empty or 'name' not in credits.columns:
            return {
                'revenue_concentration_ratio': 0.5,
                'unique_revenue_sources': 0,
                'top_source_percentage': 100,
                'concentration_risk': 'Unknown'
            }
        
        # Clean and group by source
        credits['source_clean'] = credits['name'].str.lower().str.strip()
        
        # Group similar sources
        credits['source_grouped'] = credits['source_clean'].apply(self._group_revenue_source)
        
        source_totals = credits.groupby('source_grouped')['amount'].sum().abs()
        total_revenue = source_totals.sum()
        
        if total_revenue == 0:
            return {
                'revenue_concentration_ratio': 0.5,
                'unique_revenue_sources': 0,
                'top_source_percentage': 100,
                'concentration_risk': 'Unknown'
            }
        
        # Unique sources
        metrics['unique_revenue_sources'] = len(source_totals)
        
        # Top source percentage
        top_source_pct = (source_totals.max() / total_revenue) * 100
        metrics['top_source_percentage'] = round(top_source_pct, 1)
        
        # Herfindahl-Hirschman Index (HHI) for concentration
        market_shares = source_totals / total_revenue
        hhi = (market_shares ** 2).sum()
        metrics['revenue_concentration_ratio'] = round(float(hhi), 3)
        
        # Concentration risk classification
        if hhi > 0.5:  # Very concentrated
            metrics['concentration_risk'] = 'High'
        elif hhi > 0.25:  # Moderately concentrated
            metrics['concentration_risk'] = 'Medium'
        else:  # Diversified
            metrics['concentration_risk'] = 'Low'
        
        # Top 3 sources breakdown
        top_sources = source_totals.nlargest(3)
        metrics['top_3_sources'] = [
            {'source': src, 'amount': float(amt), 'percentage': round(amt/total_revenue*100, 1)}
            for src, amt in top_sources.items()
        ]
        
        return metrics
    
    def _group_revenue_source(self, source: str) -> str:
        """Group similar revenue sources together."""
        if pd.isna(source):
            return 'Unknown'
        
        source = str(source).lower()
        
        # Payment processors
        payment_processors = ['stripe', 'paypal', 'square', 'sumup', 'zettle', 'worldpay', 
                            'barclaycard', 'elavon', 'adyen', 'checkout', 'gocardless']
        for processor in payment_processors:
            if processor in source:
                return f'Payment Processor ({processor.title()})'
        
        # Delivery platforms
        delivery = ['ubereats', 'deliveroo', 'just eat', 'justeat']
        for platform in delivery:
            if platform in source:
                return 'Delivery Platform'
        
        # Generic groupings
        if 'transfer' in source:
            return 'Bank Transfer'
        if 'cash' in source:
            return 'Cash Deposit'
        
        return source[:30]  # Truncate long names
    
    def calculate_seasonality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect and quantify seasonal patterns in revenue.
        
        Seasonal businesses may need different underwriting approaches.
        """
        metrics = {}
        
        df['year_month'] = df['date'].dt.to_period('M')
        df['month'] = df['date'].dt.month
        
        # Need at least 6 months of data for seasonality
        unique_months = df['year_month'].nunique()
        if unique_months < self.config['seasonality_min_months']:
            return {
                'seasonality_coefficient': 0,
                'peak_month': None,
                'trough_month': None,
                'seasonal_adjustment_needed': False
            }
        
        # Get credit transactions for revenue analysis
        credits = df[df['amount'] < 0] if 'amount' in df.columns else pd.DataFrame()
        
        if credits.empty:
            return {
                'seasonality_coefficient': 0,
                'peak_month': None,
                'trough_month': None,
                'seasonal_adjustment_needed': False
            }
        
        # Monthly average by calendar month
        monthly_by_calendar = credits.groupby('month')['amount'].sum().abs()
        
        if len(monthly_by_calendar) < 3:
            return {
                'seasonality_coefficient': 0,
                'peak_month': None,
                'trough_month': None,
                'seasonal_adjustment_needed': False
            }
        
        # Calculate seasonality coefficient (CV of monthly averages)
        mean_monthly = monthly_by_calendar.mean()
        std_monthly = monthly_by_calendar.std()
        
        metrics['seasonality_coefficient'] = round(
            float(std_monthly / mean_monthly) if mean_monthly > 0 else 0, 
            3
        )
        
        # Peak and trough months
        metrics['peak_month'] = int(monthly_by_calendar.idxmax())
        metrics['trough_month'] = int(monthly_by_calendar.idxmin())
        
        # Peak to trough ratio
        peak_value = monthly_by_calendar.max()
        trough_value = monthly_by_calendar.min()
        metrics['peak_to_trough_ratio'] = round(
            float(peak_value / trough_value) if trough_value > 0 else 0,
            2
        )
        
        # Seasonal adjustment needed if CV > 0.3 or peak/trough ratio > 2
        metrics['seasonal_adjustment_needed'] = (
            metrics['seasonality_coefficient'] > 0.3 or
            metrics.get('peak_to_trough_ratio', 1) > 2
        )
        
        return metrics
    
    def calculate_banking_behavior(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze banking behavior patterns.
        
        Indicators of financial stress or good management.
        """
        metrics = {}
        
        # Balance analysis
        if 'balances.available' in df.columns:
            df['balance'] = pd.to_numeric(df['balances.available'], errors='coerce')
        elif 'balance' in df.columns:
            df['balance'] = pd.to_numeric(df['balance'], errors='coerce')
        else:
            df['balance'] = np.nan
        
        if df['balance'].notna().sum() > 0:
            # Days since last negative balance
            negative_balance_dates = df[df['balance'] < 0]['date']
            if not negative_balance_dates.empty:
                last_negative = negative_balance_dates.max()
                days_since_negative = (df['date'].max() - last_negative).days
                metrics['days_since_last_negative'] = int(days_since_negative)
            else:
                metrics['days_since_last_negative'] = 999  # Never negative
            
            # Balance trend (is it improving?)
            df_sorted = df.sort_values('date')
            if len(df_sorted) >= 10:
                first_half_avg = df_sorted['balance'].iloc[:len(df_sorted)//2].mean()
                second_half_avg = df_sorted['balance'].iloc[len(df_sorted)//2:].mean()
                
                if first_half_avg > 0:
                    metrics['balance_trend'] = round(
                        (second_half_avg - first_half_avg) / first_half_avg * 100, 
                        1
                    )
                else:
                    metrics['balance_trend'] = 100 if second_half_avg > first_half_avg else -100
                
                metrics['balance_improving'] = second_half_avg > first_half_avg
            else:
                metrics['balance_trend'] = 0
                metrics['balance_improving'] = None
        else:
            metrics['days_since_last_negative'] = None
            metrics['balance_trend'] = None
            metrics['balance_improving'] = None
        
        # NSF/Bounced payment recency
        if 'name' in df.columns:
            nsf_keywords = ['nsf', 'insufficient', 'bounced', 'returned', 'unpaid', 'failed']
            df['is_nsf'] = df['name'].str.lower().str.contains('|'.join(nsf_keywords), na=False)
            
            nsf_dates = df[df['is_nsf']]['date']
            if not nsf_dates.empty:
                last_nsf = nsf_dates.max()
                metrics['days_since_last_nsf'] = int((df['date'].max() - last_nsf).days)
                metrics['nsf_count_90d'] = len(
                    nsf_dates[nsf_dates >= df['date'].max() - timedelta(days=90)]
                )
            else:
                metrics['days_since_last_nsf'] = 999
                metrics['nsf_count_90d'] = 0
        else:
            metrics['days_since_last_nsf'] = None
            metrics['nsf_count_90d'] = None
        
        return metrics
    
    def calculate_debt_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect debt stacking and existing loan obligations.
        
        Multiple concurrent loans = higher risk of default.
        """
        metrics = {}
        
        # Known lender patterns
        lender_patterns = [
            'iwoca', 'capify', 'fundbox', 'funding circle', 'fleximize',
            'marketfinance', 'liberis', 'esme', 'kriya', 'uncapped',
            'merchant money', 'capital on tap', 'youlend', 'yl ltd',
            'boost capital', 'lendingcrowd', 'bizcap'
        ]
        
        if 'name' not in df.columns:
            return {
                'active_lenders_detected': 0,
                'debt_stacking_risk': 'Unknown',
                'monthly_debt_obligations': 0
            }
        
        # Detect lender-related transactions
        df_lower = df.copy()
        df_lower['name_lower'] = df_lower['name'].str.lower()
        
        active_lenders = set()
        lender_transactions = []
        
        for pattern in lender_patterns:
            matches = df_lower[df_lower['name_lower'].str.contains(pattern, na=False)]
            if not matches.empty:
                # Check if there are both inflows (loan) and outflows (repayments)
                recent_matches = matches[matches['date'] >= df['date'].max() - timedelta(days=90)]
                if not recent_matches.empty:
                    active_lenders.add(pattern)
                    lender_transactions.extend(matches.to_dict('records'))
        
        metrics['active_lenders_detected'] = len(active_lenders)
        metrics['lender_names'] = list(active_lenders)
        
        # Debt stacking risk
        if len(active_lenders) >= 3:
            metrics['debt_stacking_risk'] = 'High'
        elif len(active_lenders) == 2:
            metrics['debt_stacking_risk'] = 'Medium'
        elif len(active_lenders) == 1:
            metrics['debt_stacking_risk'] = 'Low'
        else:
            metrics['debt_stacking_risk'] = 'None Detected'
        
        # Estimate monthly debt obligations (outflows to lenders)
        if lender_transactions:
            lender_df = pd.DataFrame(lender_transactions)
            # Outflows (repayments) are positive amounts
            monthly_payments = lender_df[lender_df['amount'] > 0]['amount'].sum()
            months_covered = max(1, df['date'].dt.to_period('M').nunique())
            metrics['monthly_debt_obligations'] = round(monthly_payments / months_covered, 2)
        else:
            metrics['monthly_debt_obligations'] = 0
        
        return metrics
    
    def _calculate_advanced_risk_score(self, metrics: Dict[str, Any]) -> float:
        """
        Calculate an overall risk score from advanced metrics.
        
        Returns a score from 0-100 where higher = lower risk.
        """
        score = 50  # Start neutral
        
        # Transaction patterns (+/- 15 points)
        deposit_freq = metrics.get('deposit_frequency_score', 50)
        score += (deposit_freq - 50) * 0.15
        
        regularity = metrics.get('transaction_regularity', 50)
        score += (regularity - 50) * 0.10
        
        # Revenue concentration (+/- 10 points)
        concentration_risk = metrics.get('concentration_risk', 'Medium')
        if concentration_risk == 'Low':
            score += 10
        elif concentration_risk == 'High':
            score -= 10
        
        # Banking behavior (+/- 15 points)
        days_since_nsf = metrics.get('days_since_last_nsf', 999)
        if days_since_nsf >= 90:
            score += 10
        elif days_since_nsf >= 30:
            score += 5
        elif days_since_nsf < 14:
            score -= 15
        
        balance_improving = metrics.get('balance_improving')
        if balance_improving is True:
            score += 5
        elif balance_improving is False:
            score -= 5
        
        # Debt stacking (-20 to 0 points)
        debt_risk = metrics.get('debt_stacking_risk', 'None Detected')
        if debt_risk == 'High':
            score -= 20
        elif debt_risk == 'Medium':
            score -= 10
        elif debt_risk == 'Low':
            score -= 5
        
        return max(0, min(100, round(score, 1)))
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure when data is insufficient."""
        return {
            'deposit_frequency_score': 0,
            'transaction_regularity': 0,
            'inflow_days_30d': 0,
            'max_inflow_gap_days': 0,
            'inflow_cv': 0,
            'revenue_concentration_ratio': 0,
            'unique_revenue_sources': 0,
            'top_source_percentage': 0,
            'concentration_risk': 'Unknown',
            'seasonality_coefficient': 0,
            'seasonal_adjustment_needed': False,
            'days_since_last_negative': None,
            'days_since_last_nsf': None,
            'active_lenders_detected': 0,
            'debt_stacking_risk': 'Unknown',
            'advanced_risk_score': 0,
            'data_quality': 'Insufficient data for analysis'
        }


def calculate_advanced_metrics(
    transactions_df: pd.DataFrame,
    company_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to calculate all advanced metrics.
    
    Args:
        transactions_df: DataFrame with transaction data
        company_info: Optional company information
        
    Returns:
        Dictionary of advanced metrics
    """
    calculator = AdvancedMetricsCalculator()
    return calculator.calculate_all_metrics(transactions_df, company_info)
