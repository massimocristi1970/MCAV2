# app/utils/feature_alignment.py
"""
Feature Alignment Utility

Maps features between different systems:
- Transaction-derived features (from build_training_dataset.py)
- ML model features
- Rule-based scoring features

This ensures consistency when using different scoring approaches
with the same underlying data.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime


class FeatureAligner:
    """
    Aligns features between different systems in the scoring pipeline.
    
    The codebase has multiple feature sets:
    1. Transaction features: inflow_cv, inflow_days_30d, max_inflow_gap_days
    2. ML features: DSCR, Operating Margin, Directors Score, etc.
    3. Rule-based features: Similar to ML but with different names
    
    This class provides mappings and derivation logic to convert between them.
    """
    
    # Feature mappings between systems
    TRANSACTION_TO_ML_MAPPING = {
        # Transaction feature -> ML feature (with derivation method)
        'inflow_cv': ('Cash Flow Volatility', 'direct'),
        'inflow_volatility_monthly': ('Cash Flow Volatility', 'scale'),
        'avg_monthly_inflow': ('Total Revenue', 'annualize'),
        'avg_monthly_net': ('Net Income', 'annualize'),
        'outflow_to_inflow_ratio': ('Expense-to-Revenue Ratio', 'direct'),
        'inflow_days_30d': (None, 'advanced_metric'),  # Not directly in ML
        'max_inflow_gap_days': (None, 'advanced_metric'),  # Not directly in ML
    }
    
    ML_FEATURE_NAMES = [
        'Directors Score',
        'Total Revenue',
        'Total Debt',
        'Debt-to-Income Ratio',
        'Operating Margin',
        'Debt Service Coverage Ratio',
        'Cash Flow Volatility',
        'Revenue Growth Rate',
        'Average Month-End Balance',
        'Average Negative Balance Days per Month',
        'Number of Bounced Payments',
        'Company Age (Months)',
        'Sector_Risk'
    ]
    
    RULE_FEATURE_NAMES = [
        'Debt Service Coverage Ratio',
        'Net Income',
        'Operating Margin',
        'Revenue Growth Rate',
        'Cash Flow Volatility',
        'Gross Burn Rate',
        'Company Age (Months)',
        'Directors Score',
        'Sector Risk',
        'Average Month-End Balance',
        'Average Negative Balance Days per Month',
        'Number of Bounced Payments',
    ]
    
    def __init__(self):
        """Initialize feature aligner with default configuration."""
        self.feature_defaults = {
            'Directors Score': 50,
            'Total Revenue': 0,
            'Total Debt': 0,
            'Debt-to-Income Ratio': 0,
            'Operating Margin': 0,
            'Debt Service Coverage Ratio': 1.0,
            'Cash Flow Volatility': 0.5,
            'Revenue Growth Rate': 0,
            'Average Month-End Balance': 0,
            'Average Negative Balance Days per Month': 0,
            'Number of Bounced Payments': 0,
            'Company Age (Months)': 12,
            'Sector_Risk': 0,
            'Sector Risk': 0,
        }
    
    def transaction_to_ml_features(
        self,
        transaction_features: Dict[str, Any],
        directors_score: int = 50,
        company_age_months: int = 12,
        sector_risk: int = 0,
        total_debt: float = 0
    ) -> Dict[str, Any]:
        """
        Convert transaction-derived features to ML model features.
        
        Args:
            transaction_features: Features from build_training_dataset.py
            directors_score: Director credit score
            company_age_months: Company age in months
            sector_risk: Sector risk indicator (0 or 1)
            total_debt: Known debt amount
            
        Returns:
            Dictionary of ML-compatible features
        """
        ml_features = self.feature_defaults.copy()
        
        # Direct mappings
        if 'inflow_cv' in transaction_features:
            ml_features['Cash Flow Volatility'] = min(2.0, transaction_features['inflow_cv'])
        
        # Revenue from monthly inflow
        if 'avg_monthly_inflow' in transaction_features:
            monthly = transaction_features['avg_monthly_inflow']
            months = transaction_features.get('months_covered', 1)
            ml_features['Total Revenue'] = monthly * months
        
        # Net income from monthly net
        if 'avg_monthly_net' in transaction_features:
            monthly_net = transaction_features['avg_monthly_net']
            months = transaction_features.get('months_covered', 1)
            ml_features['Net Income'] = monthly_net * months
        
        # Operating margin estimation
        if 'avg_monthly_inflow' in transaction_features and 'avg_monthly_net' in transaction_features:
            inflow = transaction_features['avg_monthly_inflow']
            net = transaction_features['avg_monthly_net']
            if inflow > 0:
                ml_features['Operating Margin'] = net / inflow
        
        # DSCR estimation (if debt info available)
        if total_debt > 0 and 'avg_monthly_net' in transaction_features:
            monthly_net = transaction_features['avg_monthly_net']
            # Assume debt is paid over 12 months for DSCR calc
            monthly_debt_payment = total_debt / 12
            if monthly_debt_payment > 0:
                ml_features['Debt Service Coverage Ratio'] = monthly_net / monthly_debt_payment
        elif 'avg_monthly_net' in transaction_features:
            # No debt = excellent DSCR
            ml_features['Debt Service Coverage Ratio'] = 10.0
        
        # Growth rate estimation (if we have enough data)
        if 'avg_monthly_inflow' in transaction_features:
            # This would ideally come from comparing periods
            ml_features['Revenue Growth Rate'] = 0  # Default neutral
        
        # Debt to income ratio
        if total_debt > 0 and ml_features['Total Revenue'] > 0:
            ml_features['Debt-to-Income Ratio'] = min(10, total_debt / ml_features['Total Revenue'])
            ml_features['Total Debt'] = total_debt
        
        # External parameters
        ml_features['Directors Score'] = directors_score
        ml_features['Company Age (Months)'] = company_age_months
        ml_features['Sector_Risk'] = sector_risk
        
        # Month-end balance (if available in transaction data)
        # This would typically come from balance data in transactions
        
        return ml_features
    
    def ml_to_rule_features(
        self,
        ml_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert ML features to rule-based scoring features.
        
        Mostly a renaming exercise with some calculations.
        """
        rule_features = {}
        
        # Direct mappings (same names)
        direct_mappings = [
            'Debt Service Coverage Ratio',
            'Operating Margin',
            'Revenue Growth Rate',
            'Cash Flow Volatility',
            'Average Month-End Balance',
            'Average Negative Balance Days per Month',
            'Number of Bounced Payments',
            'Company Age (Months)',
            'Directors Score',
        ]
        
        for feature in direct_mappings:
            if feature in ml_features:
                rule_features[feature] = ml_features[feature]
        
        # Net Income (if available)
        if 'Net Income' in ml_features:
            rule_features['Net Income'] = ml_features['Net Income']
        elif 'Total Revenue' in ml_features and 'Operating Margin' in ml_features:
            rule_features['Net Income'] = ml_features['Total Revenue'] * ml_features['Operating Margin']
        
        # Gross Burn Rate estimation
        if 'Total Revenue' in ml_features and 'Operating Margin' in ml_features:
            revenue = ml_features['Total Revenue']
            margin = ml_features['Operating Margin']
            expenses = revenue * (1 - margin)
            months = 12  # Assume annual data
            rule_features['Gross Burn Rate'] = expenses / months
        
        # Sector Risk (rename)
        if 'Sector_Risk' in ml_features:
            rule_features['Sector Risk'] = ml_features['Sector_Risk']
        
        return rule_features
    
    def align_for_ensemble(
        self,
        raw_features: Dict[str, Any],
        transaction_features: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Align features for use in ensemble scoring.
        
        Ensures all scoring systems receive compatible feature sets.
        
        Args:
            raw_features: Primary features (usually from financial analysis)
            transaction_features: Optional transaction-derived features
            params: Business parameters
            
        Returns:
            Unified feature dictionary for all scoring systems
        """
        params = params or {}
        
        # Start with defaults
        aligned = self.feature_defaults.copy()
        
        # Update with raw features
        for key, value in raw_features.items():
            if key in aligned:
                aligned[key] = value
        
        # Supplement with transaction features if available
        if transaction_features:
            # Cash flow volatility from transaction data is often more accurate
            if 'inflow_cv' in transaction_features and transaction_features['inflow_cv'] > 0:
                aligned['Cash Flow Volatility'] = min(2.0, transaction_features['inflow_cv'])
        
        # Apply business parameters
        if 'directors_score' in params:
            aligned['Directors Score'] = params['directors_score']
        if 'company_age_months' in params:
            aligned['Company Age (Months)'] = params['company_age_months']
        if 'industry' in params:
            # Simple sector risk mapping
            high_risk_industries = ['Restaurants and Cafes', 'Bars and Pubs', 'Construction Firms']
            aligned['Sector_Risk'] = 1 if params['industry'] in high_risk_industries else 0
            aligned['Sector Risk'] = aligned['Sector_Risk']
        
        return aligned
    
    def validate_features(
        self,
        features: Dict[str, Any],
        target_system: str = 'ml'
    ) -> Tuple[bool, List[str]]:
        """
        Validate that features are complete for a target system.
        
        Args:
            features: Feature dictionary to validate
            target_system: 'ml' or 'rule'
            
        Returns:
            Tuple of (is_valid, list of missing/invalid features)
        """
        issues = []
        
        if target_system == 'ml':
            required = self.ML_FEATURE_NAMES
        else:
            required = self.RULE_FEATURE_NAMES
        
        for feature_name in required:
            if feature_name not in features:
                issues.append(f"Missing: {feature_name}")
            elif features[feature_name] is None:
                issues.append(f"Null value: {feature_name}")
            elif isinstance(features[feature_name], float) and np.isnan(features[feature_name]):
                issues.append(f"NaN value: {feature_name}")
        
        return len(issues) == 0, issues
    
    def fill_missing(
        self,
        features: Dict[str, Any],
        strategy: str = 'default'
    ) -> Dict[str, Any]:
        """
        Fill missing features with appropriate values.
        
        Args:
            features: Feature dictionary with potential gaps
            strategy: 'default' (use defaults), 'conservative' (use pessimistic values)
            
        Returns:
            Features with missing values filled
        """
        filled = features.copy()
        
        if strategy == 'conservative':
            # Use pessimistic values for missing features
            conservative_defaults = {
                'Directors Score': 35,
                'Debt Service Coverage Ratio': 0.8,
                'Cash Flow Volatility': 1.0,
                'Revenue Growth Rate': -0.05,
                'Operating Margin': -0.02,
                'Average Month-End Balance': 200,
                'Average Negative Balance Days per Month': 5,
                'Number of Bounced Payments': 2,
            }
            defaults_to_use = {**self.feature_defaults, **conservative_defaults}
        else:
            defaults_to_use = self.feature_defaults
        
        for feature, default_value in defaults_to_use.items():
            if feature not in filled or filled[feature] is None:
                filled[feature] = default_value
            elif isinstance(filled[feature], float) and np.isnan(filled[feature]):
                filled[feature] = default_value
        
        return filled


def align_features_for_scoring(
    financial_metrics: Dict[str, Any],
    params: Dict[str, Any],
    transaction_features: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function to align features for scoring.
    
    Args:
        financial_metrics: Calculated financial metrics
        params: Business parameters
        transaction_features: Optional raw transaction features
        
    Returns:
        Aligned features ready for all scoring systems
    """
    aligner = FeatureAligner()
    
    aligned = aligner.align_for_ensemble(
        raw_features=financial_metrics,
        transaction_features=transaction_features,
        params=params
    )
    
    # Fill any missing values
    aligned = aligner.fill_missing(aligned, strategy='default')
    
    return aligned


def validate_scoring_input(
    features: Dict[str, Any],
    system: str = 'ml'
) -> Tuple[bool, List[str]]:
    """
    Validate features before scoring.
    
    Args:
        features: Feature dictionary
        system: 'ml' or 'rule'
        
    Returns:
        Tuple of (is_valid, issues)
    """
    aligner = FeatureAligner()
    return aligner.validate_features(features, system)
