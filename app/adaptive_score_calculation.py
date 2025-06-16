"""
Save this as: adaptive_score_calculation.py
Place it in the same directory as your app.py file
"""

import pandas as pd
import numpy as np

# Define months_threshold locally to avoid config import issues
months_threshold = 6

def safe_get_metric(metrics, key, default=0):
    """Safely get a metric value with a default fallback"""
    return metrics.get(key, default)

def calculate_adaptive_weighted_score(metrics, directors_score, sector_risk, thresholds, company_age_months, 
                                    personal_default_12m=False, business_ccj=False, director_ccj=False, 
                                    website_or_social_outdated=False, uses_generic_email=False, 
                                    no_online_presence=False, penalties=None):
    """
    Improved weighted score that uses continuous scoring instead of binary thresholds
    to better match ML model behavior
    """
    
    # New continuous scoring weights (redistributed to match ML feature importance)
    continuous_weights = {
        'Debt Service Coverage Ratio': 19,     # High importance (matches ML)
        'Directors Score': 18,                 # High importance (matches ML) 
        'Cash Flow Volatility': 12,           # Medium-high importance
        'Operating Margin': 9,                # Medium importance
        'Average Negative Balance Days per Month': 6,  # Medium importance
        'Average Month-End Balance': 5,       # Medium importance
        'Revenue Growth Rate': 5,             # Medium importance
        'Company Age': 4,                     # Low-medium importance
        'Number of Bounced Payments': 3,      # Low-medium importance
        'Sector Risk': 3,                     # Low importance
        'Total Revenue': 8,                   # New: Revenue scale matters
        'Total Debt': 7,                      # New: Debt level matters
        'Debt-to-Income Ratio': 6            # New: Direct debt burden measure
    }
    
    weighted_score = 0
    scoring_details = []
    
    # 1. DEBT SERVICE COVERAGE RATIO (19 points) - Continuous scoring
    dscr = safe_get_metric(metrics, "Debt Service Coverage Ratio", 0)
    threshold = thresholds.get("Debt Service Coverage Ratio", 1.4)
    if dscr >= threshold:
        score = continuous_weights['Debt Service Coverage Ratio']
    elif dscr >= threshold * 0.8:  # Partial credit for 80%+ of threshold
        score = continuous_weights['Debt Service Coverage Ratio'] * 0.7
    elif dscr >= threshold * 0.6:  # Partial credit for 60%+ of threshold
        score = continuous_weights['Debt Service Coverage Ratio'] * 0.4
    elif dscr > 0:  # At least some coverage
        score = continuous_weights['Debt Service Coverage Ratio'] * 0.1
    else:
        score = 0
    weighted_score += score
    scoring_details.append(f"DSCR: {dscr:.2f} vs {threshold:.2f} → {score:.1f} pts")
    
    # 2. DIRECTORS SCORE (18 points) - Continuous scoring
    dir_threshold = thresholds.get("Directors Score", 75)
    if directors_score >= dir_threshold:
        score = continuous_weights['Directors Score']
    elif directors_score >= dir_threshold * 0.9:
        score = continuous_weights['Directors Score'] * 0.8
    elif directors_score >= dir_threshold * 0.8:
        score = continuous_weights['Directors Score'] * 0.5
    elif directors_score >= dir_threshold * 0.6:
        score = continuous_weights['Directors Score'] * 0.2
    else:
        score = 0
    weighted_score += score
    scoring_details.append(f"Directors: {directors_score} vs {dir_threshold} → {score:.1f} pts")
    
    # 3. CASH FLOW VOLATILITY (12 points) - Inverse continuous scoring (lower is better)
    volatility = safe_get_metric(metrics, "Cash Flow Volatility", 0.1)
    vol_threshold = thresholds.get("Cash Flow Volatility", 0.2)
    if volatility <= vol_threshold:
        score = continuous_weights['Cash Flow Volatility']
    elif volatility <= vol_threshold * 1.25:  # Within 25% of threshold
        score = continuous_weights['Cash Flow Volatility'] * 0.7
    elif volatility <= vol_threshold * 1.5:   # Within 50% of threshold
        score = continuous_weights['Cash Flow Volatility'] * 0.4
    elif volatility <= vol_threshold * 2.0:   # Within 100% of threshold
        score = continuous_weights['Cash Flow Volatility'] * 0.1
    else:
        score = 0
    weighted_score += score
    scoring_details.append(f"Volatility: {volatility:.3f} vs {vol_threshold:.3f} → {score:.1f} pts")
    
    # 4. OPERATING MARGIN (9 points) - Continuous scoring
    margin = safe_get_metric(metrics, "Operating Margin", 0)
    margin_threshold = thresholds.get("Operating Margin", 0.08)
    if margin >= margin_threshold:
        score = continuous_weights['Operating Margin']
    elif margin >= margin_threshold * 0.75:
        score = continuous_weights['Operating Margin'] * 0.7
    elif margin >= margin_threshold * 0.5:
        score = continuous_weights['Operating Margin'] * 0.4
    elif margin >= 0:  # At least positive
        score = continuous_weights['Operating Margin'] * 0.2
    else:  # Negative margin
        score = 0
    weighted_score += score
    scoring_details.append(f"Op Margin: {margin:.3f} vs {margin_threshold:.3f} → {score:.1f} pts")
    
    # 5. TOTAL REVENUE (8 points) - NEW: Scale-based scoring
    revenue = safe_get_metric(metrics, "Total Revenue", 0)
    # Dynamic thresholds based on company age
    if company_age_months >= 24:  # Mature company
        revenue_benchmarks = [15000, 30000, 60000, 100000]
    else:  # Younger company
        revenue_benchmarks = [8000, 20000, 40000, 80000]
    
    if revenue >= revenue_benchmarks[3]:
        score = continuous_weights['Total Revenue']
    elif revenue >= revenue_benchmarks[2]:
        score = continuous_weights['Total Revenue'] * 0.8
    elif revenue >= revenue_benchmarks[1]:
        score = continuous_weights['Total Revenue'] * 0.6
    elif revenue >= revenue_benchmarks[0]:
        score = continuous_weights['Total Revenue'] * 0.3
    elif revenue > 0:
        score = continuous_weights['Total Revenue'] * 0.1
    else:
        score = 0
    weighted_score += score
    scoring_details.append(f"Revenue: £{revenue:,.0f} → {score:.1f} pts")
    
    # 6. TOTAL DEBT (7 points) - NEW: Inverse scale scoring (lower debt is better)
    debt = safe_get_metric(metrics, "Total Debt", 0)
    debt_ratio = debt / revenue if revenue > 0 else 0
    if debt_ratio <= 0.1:      # Very low debt
        score = continuous_weights['Total Debt']
    elif debt_ratio <= 0.25:   # Moderate debt
        score = continuous_weights['Total Debt'] * 0.8
    elif debt_ratio <= 0.5:    # High debt
        score = continuous_weights['Total Debt'] * 0.4
    elif debt_ratio <= 1.0:    # Very high debt
        score = continuous_weights['Total Debt'] * 0.1
    else:                      # Extreme debt
        score = 0
    weighted_score += score
    scoring_details.append(f"Debt Ratio: {debt_ratio:.3f} → {score:.1f} pts")
    
    # 7. DEBT-TO-INCOME RATIO (6 points) - NEW: Direct ratio scoring
    debt_income_ratio = safe_get_metric(metrics, "Debt-to-Income Ratio", 0)
    if debt_income_ratio <= 0.15:
        score = continuous_weights['Debt-to-Income Ratio']
    elif debt_income_ratio <= 0.3:
        score = continuous_weights['Debt-to-Income Ratio'] * 0.7
    elif debt_income_ratio <= 0.5:
        score = continuous_weights['Debt-to-Income Ratio'] * 0.4
    elif debt_income_ratio <= 1.0:
        score = continuous_weights['Debt-to-Income Ratio'] * 0.1
    else:
        score = 0
    weighted_score += score
    scoring_details.append(f"Debt/Income: {debt_income_ratio:.3f} → {score:.1f} pts")
    
    # 8. AVERAGE NEGATIVE BALANCE DAYS (6 points) - Inverse continuous
    neg_days = safe_get_metric(metrics, "Average Negative Balance Days per Month", 0)
    neg_threshold = thresholds.get("Average Negative Balance Days per Month", 2)
    if neg_days <= neg_threshold:
        score = continuous_weights['Average Negative Balance Days per Month']
    elif neg_days <= neg_threshold + 2:
        score = continuous_weights['Average Negative Balance Days per Month'] * 0.6
    elif neg_days <= neg_threshold + 5:
        score = continuous_weights['Average Negative Balance Days per Month'] * 0.3
    elif neg_days <= neg_threshold + 10:
        score = continuous_weights['Average Negative Balance Days per Month'] * 0.1
    else:
        score = 0
    weighted_score += score
    scoring_details.append(f"Neg Days: {neg_days:.1f} vs {neg_threshold} → {score:.1f} pts")
    
    # 9. AVERAGE MONTH-END BALANCE (5 points) - Continuous scoring
    balance = safe_get_metric(metrics, "Average Month-End Balance", 0)
    balance_threshold = thresholds.get("Average Month-End Balance", 1000)
    if balance >= balance_threshold:
        score = continuous_weights['Average Month-End Balance']
    elif balance >= balance_threshold * 0.7:
        score = continuous_weights['Average Month-End Balance'] * 0.7
    elif balance >= balance_threshold * 0.4:
        score = continuous_weights['Average Month-End Balance'] * 0.4
    elif balance >= 0:  # At least positive
        score = continuous_weights['Average Month-End Balance'] * 0.2
    else:  # Negative balance
        score = 0
    weighted_score += score
    scoring_details.append(f"Avg Balance: £{balance:,.0f} vs £{balance_threshold:,.0f} → {score:.1f} pts")
    
    # 10. REVENUE GROWTH RATE (5 points) - Continuous scoring
    growth = safe_get_metric(metrics, "Revenue Growth Rate", 0)
    growth_threshold = thresholds.get("Revenue Growth Rate", 0.04)
    if growth >= growth_threshold:
        score = continuous_weights['Revenue Growth Rate']
    elif growth >= growth_threshold * 0.5:
        score = continuous_weights['Revenue Growth Rate'] * 0.7
    elif growth >= 0:  # At least not declining
        score = continuous_weights['Revenue Growth Rate'] * 0.4
    elif growth >= -0.05:  # Small decline acceptable
        score = continuous_weights['Revenue Growth Rate'] * 0.2
    else:  # Significant decline
        score = 0
    weighted_score += score
    scoring_details.append(f"Growth: {growth:.1%} vs {growth_threshold:.1%} → {score:.1f} pts")
    
    # 11. COMPANY AGE (4 points) - Continuous scoring
    if company_age_months >= months_threshold:
        score = continuous_weights['Company Age']
    elif company_age_months >= months_threshold * 0.75:  # 75% of threshold
        score = continuous_weights['Company Age'] * 0.7
    elif company_age_months >= months_threshold * 0.5:   # 50% of threshold
        score = continuous_weights['Company Age'] * 0.4
    elif company_age_months >= months_threshold * 0.25:  # 25% of threshold
        score = continuous_weights['Company Age'] * 0.2
    else:
        score = 0
    weighted_score += score
    scoring_details.append(f"Age: {company_age_months} months vs {months_threshold} → {score:.1f} pts")
    
    # 12. NUMBER OF BOUNCED PAYMENTS (3 points) - Inverse continuous
    bounced = safe_get_metric(metrics, "Number of Bounced Payments", 0)
    bounced_threshold = thresholds.get("Number of Bounced Payments", 0)
    if bounced <= bounced_threshold:
        score = continuous_weights['Number of Bounced Payments']
    elif bounced <= bounced_threshold + 2:
        score = continuous_weights['Number of Bounced Payments'] * 0.5
    elif bounced <= bounced_threshold + 5:
        score = continuous_weights['Number of Bounced Payments'] * 0.2
    else:
        score = 0
    weighted_score += score
    scoring_details.append(f"Bounced: {bounced} vs {bounced_threshold} → {score:.1f} pts")
    
    # 13. SECTOR RISK (3 points) - Binary (matches ML)
    sector_threshold = thresholds.get("Sector Risk", 0)
    if sector_risk <= sector_threshold:  # Low risk
        score = continuous_weights['Sector Risk']
    else:
        score = 0
    weighted_score += score
    scoring_details.append(f"Sector Risk: {sector_risk} → {score:.1f} pts")
    
    # Apply penalties (same as before but adjusted impact)
    penalty_total = 0
    if penalties:
        if personal_default_12m:
            penalty = penalties.get("personal_default_12m", 0)
            penalty_total += penalty
            scoring_details.append(f"Personal Default Penalty: -{penalty} pts")
        if business_ccj:
            penalty = penalties.get("business_ccj", 0)
            penalty_total += penalty
            scoring_details.append(f"Business CCJ Penalty: -{penalty} pts")
        if director_ccj:
            penalty = penalties.get("director_ccj", 0)
            penalty_total += penalty
            scoring_details.append(f"Director CCJ Penalty: -{penalty} pts")
        if website_or_social_outdated:
            penalty = penalties.get("website_or_social_outdated", 0)
            penalty_total += penalty
            scoring_details.append(f"Outdated Web Presence Penalty: -{penalty} pts")
        if uses_generic_email:
            penalty = penalties.get("uses_generic_email", 0)
            penalty_total += penalty
            scoring_details.append(f"Generic Email Penalty: -{penalty} pts")
        if no_online_presence:
            penalty = penalties.get("no_online_presence", 0)
            penalty_total += penalty
            scoring_details.append(f"No Online Presence Penalty: -{penalty} pts")
    
    final_score = max(0, weighted_score - penalty_total)  # Don't go below 0
    
    return final_score, scoring_details

def calculate_ml_aligned_score_percentage(adaptive_score, max_possible_score=105):
    """
    Convert adaptive weighted score to percentage that aligns better with ML probability
    """
    # Convert to percentage
    percentage = (adaptive_score / max_possible_score) * 100
    
    # Apply sigmoid-like transformation to better match ML probability ranges
    # This compresses extreme values and expands middle ranges
    def sigmoid_transform(x):
        # Transform 0-100 to roughly 15-85 range (more realistic business probability range)
        normalized = x / 100.0  # 0 to 1
        transformed = 1 / (1 + np.exp(-6 * (normalized - 0.5)))  # Sigmoid centered at 0.5
        return 15 + (transformed * 70)  # Scale to 15-85 range
    
    ml_aligned_score = sigmoid_transform(percentage)
    
    return round(ml_aligned_score, 1)

def get_improved_weighted_score(metrics, directors_score, sector_risk, industry_thresholds, weights,
                               company_age_months, personal_default_12m=False, business_ccj=False, 
                               director_ccj=False, website_or_social_outdated=False, 
                               uses_generic_email=False, no_online_presence=False, penalties=None):
    """
    Drop-in replacement for your existing calculate_weighted_score function
    Returns ML-aligned percentage for consistency with ML model output
    """
    
    adaptive_score, details = calculate_adaptive_weighted_score(
        metrics, directors_score, sector_risk, industry_thresholds, company_age_months,
        personal_default_12m, business_ccj, director_ccj, website_or_social_outdated,
        uses_generic_email, no_online_presence, penalties
    )
    
    # Return ML-aligned percentage for consistency with ML model output
    ml_aligned_percentage = calculate_ml_aligned_score_percentage(adaptive_score)
    
    return ml_aligned_percentage

def get_detailed_scoring_breakdown(metrics, directors_score, sector_risk, industry_thresholds, 
                                  company_age_months, personal_default_12m=False, business_ccj=False, 
                                  director_ccj=False, website_or_social_outdated=False, 
                                  uses_generic_email=False, no_online_presence=False, penalties=None):
    """
    Get both the score and detailed breakdown for display purposes
    """
    
    adaptive_score, details = calculate_adaptive_weighted_score(
        metrics, directors_score, sector_risk, industry_thresholds, company_age_months,
        personal_default_12m, business_ccj, director_ccj, website_or_social_outdated,
        uses_generic_email, no_online_presence, penalties
    )
    
    ml_aligned_percentage = calculate_ml_aligned_score_percentage(adaptive_score)
    
    return ml_aligned_percentage, adaptive_score, details

# Test function to verify the module works
def test_module():
    """Test function to verify the module is working correctly"""
    print("✅ Adaptive scoring module loaded successfully!")
    
    # Complete test metrics that match what the main app provides
    test_metrics = {
        'Debt Service Coverage Ratio': 1.5,
        'Net Income': 5000,
        'Operating Margin': 0.1,
        'Revenue Growth Rate': 0.05,
        'Cash Flow Volatility': 0.15,
        'Gross Burn Rate': 10000,
        'Average Month-End Balance': 5000,
        'Average Negative Balance Days per Month': 1,
        'Number of Bounced Payments': 0,
        'Total Revenue': 50000,
        'Total Debt': 5000,
        'Debt-to-Income Ratio': 0.1,
        'Total Expenses': 45000,
        'Total Debt Repayments': 1000,
        'Expense-to-Revenue Ratio': 0.9
    }
    
    test_thresholds = {
        'Debt Service Coverage Ratio': 1.4,
        'Net Income': 1000,
        'Operating Margin': 0.08,
        'Revenue Growth Rate': 0.04,
        'Cash Flow Volatility': 0.2,
        'Gross Burn Rate': 12000,
        'Directors Score': 75,
        'Sector Risk': 0,
        'Average Month-End Balance': 1000,
        'Average Negative Balance Days per Month': 2,
        'Number of Bounced Payments': 0
    }
    
    test_penalties = {
        "personal_default_12m": 3,
        "business_ccj": 5,
        "director_ccj": 3,
        'website_or_social_outdated': 3,
        'uses_generic_email': 1,
        'no_online_presence': 2
    }
    
    # Test the function
    score, raw, details = get_detailed_scoring_breakdown(
        test_metrics, 75, 0, test_thresholds, 24,
        False, False, False, False, False, False, test_penalties
    )
    
    print(f"Test results:")
    print(f"  ML-Aligned Score: {score:.1f}%")
    print(f"  Raw Score: {raw:.1f}/105")
    print(f"  Breakdown items: {len(details)}")
    print(f"  Sample details:")
    for detail in details[:3]:  # Show first 3 details
        print(f"    {detail}")
    
    return True

if __name__ == "__main__":
    test_module()