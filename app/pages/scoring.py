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

# Import centralized thresholds
try:
    from ..config.scoring_thresholds import get_thresholds
    THRESHOLDS_AVAILABLE = True
except ImportError:
    THRESHOLDS_AVAILABLE = False
    get_thresholds = None

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

# Continuous scoring configuration
# Maps how close to threshold earns partial credit
CONTINUOUS_SCORING_TIERS = [
    (1.0, 1.0),    # >= 100% of threshold = full points
    (0.8, 0.7),    # >= 80% of threshold = 70% of points
    (0.6, 0.4),    # >= 60% of threshold = 40% of points
    (0.4, 0.15),   # >= 40% of threshold = 15% of points
]


def _score_continuous(
    value: float,
    threshold: float,
    weight: int,
    lower_is_better: bool = False
) -> float:
    """
    Calculate continuous score with partial credit.
    
    Instead of binary pass/fail, gives partial credit for values
    approaching the threshold.
    
    Args:
        value: Actual metric value
        threshold: Target threshold value
        weight: Maximum points available
        lower_is_better: True for metrics like volatility
        
    Returns:
        Points earned (0 to weight)
    """
    if threshold == 0:
        return weight if value == 0 else 0
    
    if lower_is_better:
        # For metrics where lower is better (volatility, negative days)
        # Being at or below threshold = full points
        # Being higher = partial credit if within reasonable range
        if value <= threshold:
            return weight
        
        ratio = threshold / value  # How close are we (inverted)
        for min_ratio, credit_pct in CONTINUOUS_SCORING_TIERS:
            if ratio >= min_ratio:
                return weight * credit_pct
        return 0
    else:
        # For metrics where higher is better (DSCR, growth)
        if value >= threshold:
            return weight
        
        ratio = value / threshold  # How close are we
        for min_ratio, credit_pct in CONTINUOUS_SCORING_TIERS:
            if ratio >= min_ratio:
                return weight * credit_pct
        return 0


def calculate_weighted_scores(
    metrics: Dict[str, Any], 
    params: Dict[str, Any], 
    industry_thresholds: Dict[str, Any],
    use_continuous: bool = True
) -> int:
    """
    Calculate weighted score based on industry thresholds.
    
    Args:
        metrics: Financial metrics dictionary
        params: Business parameters dictionary
        industry_thresholds: Industry-specific thresholds
        use_continuous: If True, use continuous scoring with partial credit
        
    Returns:
        Weighted score (0-100)
    """
    weighted_score = 0
    
    # Metrics where lower values are better
    lower_is_better_metrics = {
        'Cash Flow Volatility', 
        'Average Negative Balance Days per Month', 
        'Number of Bounced Payments',
        'Gross Burn Rate'
    }
    
    for metric, weight in WEIGHTS.items():
        if metric == 'Company Age (Months)':
            age = params.get('company_age_months', 0)
            if use_continuous:
                # Continuous scoring for company age
                age_threshold = 6  # 6 months minimum
                if age >= 18:
                    weighted_score += weight
                elif age >= 12:
                    weighted_score += weight * 0.8
                elif age >= age_threshold:
                    weighted_score += weight * 0.5
                elif age >= 3:
                    weighted_score += weight * 0.2
            else:
                if age >= 6:
                    weighted_score += weight
                    
        elif metric == 'Directors Score':
            directors = params.get('directors_score', 0)
            threshold = industry_thresholds.get('Directors Score', 75)
            if use_continuous:
                weighted_score += _score_continuous(directors, threshold, weight, lower_is_better=False)
            else:
                if directors >= threshold:
                    weighted_score += weight
                    
        elif metric == 'Sector Risk':
            sector_risk = industry_thresholds.get('Sector Risk', 0)
            # Sector risk is pre-determined, binary
            if sector_risk <= industry_thresholds.get('Sector Risk', 0):
                weighted_score += weight
                
        elif metric in metrics:
            value = metrics.get(metric, 0)
            threshold = industry_thresholds.get(metric, 0)
            lower_is_better = metric in lower_is_better_metrics
            
            if use_continuous:
                weighted_score += _score_continuous(value, threshold, weight, lower_is_better)
            else:
                # Original binary scoring
                if lower_is_better:
                    if value <= threshold:
                        weighted_score += weight
                else:
                    if value >= threshold:
                        weighted_score += weight
    
    # Apply penalties
    penalties = 0
    for flag, penalty in PENALTIES.items():
        if params.get(flag, False):
            penalties += penalty
    
    weighted_score = max(0, weighted_score - penalties)
    return int(round(weighted_score))


def calculate_weighted_scores_detailed(
    metrics: Dict[str, Any], 
    params: Dict[str, Any], 
    industry_thresholds: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate weighted score with detailed breakdown.
    
    Args:
        metrics: Financial metrics dictionary
        params: Business parameters dictionary
        industry_thresholds: Industry-specific thresholds
        
    Returns:
        Dictionary with score and detailed breakdown
    """
    breakdown = []
    total_score = 0
    max_possible = sum(WEIGHTS.values())
    
    lower_is_better_metrics = {
        'Cash Flow Volatility', 
        'Average Negative Balance Days per Month', 
        'Number of Bounced Payments',
        'Gross Burn Rate'
    }
    
    for metric, weight in WEIGHTS.items():
        entry = {
            'metric': metric,
            'weight': weight,
            'points_earned': 0,
            'percentage': 0,
            'status': 'FAIL'
        }
        
        if metric == 'Company Age (Months)':
            age = params.get('company_age_months', 0)
            entry['value'] = age
            entry['threshold'] = 6
            
            if age >= 18:
                entry['points_earned'] = weight
                entry['percentage'] = 100
                entry['status'] = 'PASS'
            elif age >= 12:
                entry['points_earned'] = weight * 0.8
                entry['percentage'] = 80
                entry['status'] = 'PARTIAL'
            elif age >= 6:
                entry['points_earned'] = weight * 0.5
                entry['percentage'] = 50
                entry['status'] = 'PARTIAL'
            elif age >= 3:
                entry['points_earned'] = weight * 0.2
                entry['percentage'] = 20
                entry['status'] = 'PARTIAL'
                
        elif metric == 'Directors Score':
            directors = params.get('directors_score', 0)
            threshold = industry_thresholds.get('Directors Score', 75)
            entry['value'] = directors
            entry['threshold'] = threshold
            
            points = _score_continuous(directors, threshold, weight, lower_is_better=False)
            entry['points_earned'] = points
            entry['percentage'] = (points / weight * 100) if weight > 0 else 0
            entry['status'] = 'PASS' if points == weight else ('PARTIAL' if points > 0 else 'FAIL')
            
        elif metric == 'Sector Risk':
            sector_risk = industry_thresholds.get('Sector Risk', 0)
            entry['value'] = sector_risk
            entry['threshold'] = sector_risk
            entry['points_earned'] = weight
            entry['percentage'] = 100
            entry['status'] = 'PASS'
            
        elif metric in metrics:
            value = metrics.get(metric, 0)
            threshold = industry_thresholds.get(metric, 0)
            lower_is_better = metric in lower_is_better_metrics
            
            entry['value'] = value
            entry['threshold'] = threshold
            entry['lower_is_better'] = lower_is_better
            
            points = _score_continuous(value, threshold, weight, lower_is_better)
            entry['points_earned'] = points
            entry['percentage'] = (points / weight * 100) if weight > 0 else 0
            entry['status'] = 'PASS' if points == weight else ('PARTIAL' if points > 0 else 'FAIL')
        
        total_score += entry['points_earned']
        breakdown.append(entry)
    
    # Apply penalties
    penalty_details = []
    total_penalty = 0
    for flag, penalty in PENALTIES.items():
        if params.get(flag, False):
            total_penalty += penalty
            penalty_details.append({'flag': flag, 'penalty': penalty})
    
    final_score = max(0, total_score - total_penalty)
    
    return {
        'score': int(round(final_score)),
        'raw_score': round(total_score, 1),
        'max_possible': max_possible,
        'penalties_applied': total_penalty,
        'penalty_details': penalty_details,
        'breakdown': breakdown
    }


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


_ML_FEATURE_BOUNDS = {
    'Directors Score': (0, 100),
    'Total Revenue': (0, 5_000_000),
    'Total Debt': (0, 2_000_000),
    'Debt-to-Income Ratio': (0, 10),
    'Operating Margin': (-1.0, 1.0),
    'Debt Service Coverage Ratio': (0, 500_000),
    'Cash Flow Volatility': (0, 100),
    'Revenue Growth Rate': (-500, 500),
    'Average Month-End Balance': (-500_000, 500_000),
    'Average Negative Balance Days per Month': (0, 31),
    'Number of Bounced Payments': (0, 100),
    'Company Age (Months)': (0, 600),
    'Sector_Risk': (0, 1),
}


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
        
        # Clip features to prevent extrapolation on unseen ranges
        for col, (lo, hi) in _ML_FEATURE_BOUNDS.items():
            if col in features_df.columns:
                features_df[col] = features_df[col].clip(lo, hi)
        
        features_scaled = scaler.transform(features_df)
        # The retrained model is already wrapped in CalibratedClassifierCV
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
    mca_rule_score = scores.get('mca_rule_score', 0)
    ml_score = scores.get('adjusted_ml_score', scores.get('ml_score', 0)) or 0
    
    # Average of available scores
    available_scores = [s for s in [subprime_score, mca_rule_score, ml_score] if s > 0]
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
