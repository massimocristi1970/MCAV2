# app/pages/scoring.py
"""Scoring calculation functions for business finance assessment."""

import numpy as np
import joblib
from typing import Dict, Any, Tuple, List, Optional

# Import the subprime scoring system
try:
    from ..services.subprime_scoring_system import SubprimeScoring
    SUBPRIME_SCORING_AVAILABLE = True
except ImportError:
    SUBPRIME_SCORING_AVAILABLE = False
    SubprimeScoring = None

# Scoring weights for V2 weighted scoring
WEIGHTS = {
    'Debt Service Coverage Ratio': 19, 'Net Income': 13, 'Operating Margin': 9,
    'Revenue Growth Rate': 5, 'Cash Flow Volatility': 12, 'Gross Burn Rate': 3,
    'Company Age (Months)': 4, 'Directors Score': 18, 'Sector Risk': 3,
    'Average Month-End Balance': 5, 'Average Negative Balance Days per Month': 6,
    'Number of Bounced Payments': 3,
}

PENALTIES = {
    "business_ccj": 5, "director_ccj": 3,
    'poor_or_no_online_presence': 3, 'uses_generic_email': 1
}


def calculate_weighted_scores(
    metrics: Dict[str, Any], 
    params: Dict[str, Any], 
    industry_thresholds: Dict[str, Any]
) -> int:
    """
    Calculate weighted score based on industry thresholds.
    
    Args:
        metrics: Financial metrics dictionary
        params: Business parameters dictionary
        industry_thresholds: Industry-specific thresholds
        
    Returns:
        Weighted score (0-100)
    """
    weighted_score = 0
    
    for metric, weight in WEIGHTS.items():
        if metric == 'Company Age (Months)':
            if params['company_age_months'] >= 6:
                weighted_score += weight
        elif metric == 'Directors Score':
            if params['directors_score'] >= industry_thresholds['Directors Score']:
                weighted_score += weight
        elif metric == 'Sector Risk':
            sector_risk = industry_thresholds['Sector Risk']
            if sector_risk <= industry_thresholds['Sector Risk']:
                weighted_score += weight
        elif metric in metrics:
            threshold = industry_thresholds.get(metric, 0)
            # These metrics are "lower is better"
            if metric in ['Cash Flow Volatility', 'Average Negative Balance Days per Month', 'Number of Bounced Payments']:
                if metrics[metric] <= threshold:
                    weighted_score += weight
            else:
                if metrics[metric] >= threshold:
                    weighted_score += weight
    
    # Apply penalties
    penalties = 0
    for flag, penalty in PENALTIES.items():
        if params.get(flag, False):
            penalties += penalty
    
    weighted_score = max(0, weighted_score - penalties)
    return weighted_score


def load_models() -> Tuple[Optional[Any], Optional[Any]]:
    """
    Load ML models for prediction.
    
    Returns:
        Tuple of (model, scaler) or (None, None) if loading fails
    """
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception:
        return None, None


def calculate_ml_score(
    metrics: Dict[str, Any], 
    params: Dict[str, Any],
    model: Any,
    scaler: Any
) -> Optional[float]:
    """
    Calculate ML-based probability score.
    
    Args:
        metrics: Financial metrics dictionary
        params: Business parameters dictionary
        model: Trained ML model
        scaler: Feature scaler
        
    Returns:
        ML score as percentage, or None if calculation fails
    """
    if model is None or scaler is None:
        return None
    
    try:
        import pandas as pd
        
        # Prepare features in the exact order expected by the model
        features = {
            'Directors Score': params.get('directors_score', 0),
            'Total Revenue': metrics.get("Total Revenue", 0),
            'Total Debt': metrics.get("Total Debt", 0),
            'Debt-to-Income Ratio': metrics.get("Debt-to-Income Ratio", 0),
            'Operating Margin': metrics.get("Operating Margin", 0),
            'Debt Service Coverage Ratio': metrics.get("Debt Service Coverage Ratio", 0),
            'Cash Flow Volatility': metrics.get("Cash Flow Volatility", 0),
            'Revenue Growth Rate': metrics.get("Revenue Growth Rate", 0),
            'Average Month-End Balance': metrics.get("Average Month-End Balance", 0),
            'Average Negative Balance Days per Month': metrics.get("Average Negative Balance Days per Month", 0),
            'Number of Bounced Payments': metrics.get("Number of Bounced Payments", 0),
            'Company Age (Months)': params.get('company_age_months', 0),
            'Sector_Risk': 1 if params.get('industry', '') in ['Restaurants and Cafes', 'Bars and Pubs', 'Other'] else 0
        }
        
        features_df = pd.DataFrame([features])
        features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        features_df.fillna(0, inplace=True)
        
        # Scale and predict
        features_scaled = scaler.transform(features_df)
        probability = model.predict_proba(features_scaled)[:, 1][0]
        
        return round(probability * 100, 2)
        
    except Exception as e:
        print(f"ML scoring error: {e}")
        return None


def calculate_subprime_score(
    metrics: Dict[str, Any], 
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate subprime lending score using the specialized scoring system.
    
    Args:
        metrics: Financial metrics dictionary
        params: Business parameters dictionary
        
    Returns:
        Dictionary containing subprime score results
    """
    if not SUBPRIME_SCORING_AVAILABLE or SubprimeScoring is None:
        return {
            'subprime_score': 0,
            'risk_tier': 'Not Available',
            'pricing_guidance': {'suggested_rate': 'N/A'},
            'recommendation': 'Subprime scoring module not available',
            'breakdown': ['Module not loaded'],
            'diagnostics': {}
        }
    
    try:
        scorer = SubprimeScoring()
        result = scorer.calculate_subprime_score(metrics, params)
        return result
    except Exception as e:
        return {
            'subprime_score': 0,
            'risk_tier': 'Error',
            'pricing_guidance': {'suggested_rate': 'N/A'},
            'recommendation': f'Subprime scoring failed: {str(e)}',
            'breakdown': [f'Error: {str(e)}'],
            'diagnostics': {}
        }


def adjust_ml_score_for_growth_business(
    raw_ml_score: float, 
    metrics: Dict[str, Any], 
    params: Dict[str, Any]
) -> float:
    """
    Adjust ML score for growth business characteristics.
    
    Args:
        raw_ml_score: Original ML score
        metrics: Financial metrics dictionary
        params: Business parameters dictionary
        
    Returns:
        Adjusted ML score
    """
    if raw_ml_score is None:
        return 0.0
    
    adjustment = 0
    
    # Growth rate bonus
    growth_rate = metrics.get('Revenue Growth Rate', 0)
    if growth_rate > 0.10:  # >10% growth
        adjustment += min(15, growth_rate * 50)
    elif growth_rate > 0.05:  # >5% growth
        adjustment += min(8, growth_rate * 40)
    elif growth_rate > 0:  # Any positive growth
        adjustment += min(3, growth_rate * 20)
    
    # DSCR bonus
    dscr = metrics.get('Debt Service Coverage Ratio', 0)
    if dscr > 2.0:
        adjustment += 5
    elif dscr > 1.5:
        adjustment += 3
    elif dscr > 1.2:
        adjustment += 1
    
    # Directors score bonus
    directors_score = params.get('directors_score', 50)
    if directors_score >= 80:
        adjustment += 5
    elif directors_score >= 70:
        adjustment += 3
    elif directors_score >= 60:
        adjustment += 1
    
    # Company age consideration (young but growing)
    company_age = params.get('company_age_months', 0)
    if company_age < 12 and growth_rate > 0.05:
        adjustment += 5  # Bonus for young but growing companies
    
    # Penalty adjustments
    if params.get('business_ccj', False):
        adjustment -= 5
    if params.get('director_ccj', False):
        adjustment -= 3
    
    adjusted_score = min(95, max(5, raw_ml_score + adjustment))
    return round(adjusted_score, 1)


def get_ml_score_interpretation(adjusted_score: float, raw_score: float) -> str:
    """
    Provide interpretation of the adjusted ML score.
    
    Args:
        adjusted_score: Score after growth adjustments
        raw_score: Original ML score
        
    Returns:
        Human-readable interpretation string
    """
    improvement = adjusted_score - (raw_score or 0)
    
    if adjusted_score >= 75:
        interpretation = "**Excellent** repayment probability"
    elif adjusted_score >= 60:
        interpretation = "**Good** repayment probability"
    elif adjusted_score >= 45:
        interpretation = "**Moderate** repayment probability"
    elif adjusted_score >= 30:
        interpretation = "**Below average** repayment probability"
    else:
        interpretation = "**High risk** - enhanced monitoring recommended"
    
    if improvement >= 15:
        interpretation += f"\n📈 **Significant upward adjustment** (+{improvement:.1f}) for growth business profile"
    elif improvement >= 8:
        interpretation += f"\n📈 **Notable upward adjustment** (+{improvement:.1f}) for growth characteristics"
    elif improvement >= 3:
        interpretation += f"\n📈 **Minor upward adjustment** (+{improvement:.1f}) for positive factors"
    else:
        interpretation += f"\n➡️ **Minimal adjustment** (+{improvement:.1f}) - standard risk profile"
    
    return interpretation


def determine_loan_risk_level(
    scores: Dict[str, Any], 
    requested_loan: float, 
    monthly_revenue: float
) -> str:
    """
    Determine overall loan risk level based on scores and loan amount.
    
    Args:
        scores: Dictionary of all calculated scores
        requested_loan: Requested loan amount
        monthly_revenue: Average monthly revenue
        
    Returns:
        Risk level string
    """
    # Calculate loan-to-revenue ratio
    loan_to_revenue = requested_loan / monthly_revenue if monthly_revenue > 0 else float('inf')
    
    subprime_score = scores.get('subprime_score', 0)
    weighted_score = scores.get('weighted_score', 0)
    ml_score = scores.get('adjusted_ml_score', scores.get('ml_score', 0)) or 0
    
    # Average of available scores
    available_scores = [s for s in [subprime_score, weighted_score, ml_score] if s > 0]
    avg_score = sum(available_scores) / len(available_scores) if available_scores else 0
    
    if avg_score >= 70 and loan_to_revenue <= 3:
        return "Low Risk"
    elif avg_score >= 55 and loan_to_revenue <= 4:
        return "Moderate Risk"
    elif avg_score >= 40 and loan_to_revenue <= 5:
        return "Higher Risk"
    elif avg_score >= 30:
        return "High Risk - Enhanced Monitoring"
    else:
        return "Very High Risk - Senior Review Required"
