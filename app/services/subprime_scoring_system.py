# subprime_scoring_system.py
"""
Subprime Business Finance Scoring System
Designed for alternative lenders serving subprime business market
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from datetime import datetime

class SubprimeScoring:
    """Enhanced scoring system specifically designed for subprime business lending."""
    
    def __init__(self):
        # Subprime-optimized weights - focus on ability to pay and growth trajectory
        self.subprime_weights = {
            'Debt Service Coverage Ratio': 28,      # PRIMARY - Current ability to service debt
            'Revenue Growth Rate': 20,              # HIGH - Growth trajectory toward profitability
            'Directors Score': 16,                  # HIGH - Personal reliability crucial in subprime
            'Average Month-End Balance': 12,        # HIGH - Liquidity buffer essential
            'Cash Flow Volatility': 8,              # MODERATE - Some volatility expected
            'Operating Margin': 6,                  # LOW - Current losses more acceptable
            'Net Income': 4,                        # LOW - Growth more important than current profit
            'Average Negative Balance Days per Month': 4,  # Monitor but don't over-penalize
            'Company Age (Months)': 2,              # MINIMAL - Less relevant for growth businesses
        }
        
        # Risk tolerance thresholds for subprime market
        self.subprime_thresholds = {
            'minimum_dscr': 1.2,                   # Lower than traditional (was 1.4+)
            'maximum_volatility': 1.0,             # Higher than traditional (was 0.12)
            'minimum_growth': -0.1,                # Can accept some decline if other factors strong
            'minimum_balance': 500,                # Lower liquidity requirement
            'maximum_negative_days': 5,            # More tolerance for cash flow gaps
        }
        
        # Industry risk adjustments for subprime context
        self.industry_multipliers = {
            # Lower risk industries (bonus)
            'Medical Practices (GPs, Clinics, Dentists)': 1.1,
            'IT Services and Support Companies': 1.1,
            'Business Consultants': 1.05,
            'Education': 1.05,
            
            # Standard risk (no adjustment)
            'Manufacturing': 1.0,
            'Retail': 1.0,
            'Food Service': 1.0,
            'Tradesman': 1.0,
            
            # Higher risk but still acceptable with pricing
            'Restaurants and Cafes': 0.9,
            'Construction Firms': 0.9,
            'Beauty Salons and Spas': 0.9,
            'Bars and Pubs': 0.85,
            'Event Planning and Management Firms': 0.8,
        }
    
    def calculate_subprime_score(self, metrics: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate comprehensive subprime business score.
        
        Returns detailed scoring breakdown with risk tier and pricing guidance.
        """
        
        # Calculate base weighted score using subprime weights
        base_score = self._calculate_base_subprime_score(metrics, params)
        
        # Apply industry adjustment
        industry_adjusted_score = self._apply_industry_adjustment(base_score, params.get('industry'))
        
        # Calculate growth momentum bonus
        growth_bonus = self._calculate_growth_momentum_bonus(metrics)
        
        # Calculate stability penalty
        stability_penalty = self._calculate_stability_penalty(metrics)
        
        # Final score calculation
        final_score = max(0, min(100, industry_adjusted_score + growth_bonus - stability_penalty))
        
        # Determine risk tier and pricing
        risk_tier, pricing_guidance = self._determine_risk_tier(final_score, metrics, params)
        
        # Generate detailed breakdown
        breakdown = self._generate_scoring_breakdown(
            base_score, industry_adjusted_score, growth_bonus, 
            stability_penalty, final_score, metrics, params
        )
        
        return {
            'subprime_score': round(final_score, 1),
            'risk_tier': risk_tier,
            'pricing_guidance': pricing_guidance,
            'breakdown': breakdown,
            'recommendation': self._generate_recommendation(risk_tier, metrics, params)
        }
    
    def _calculate_base_subprime_score(self, metrics: Dict[str, Any], params: Dict[str, Any]) -> float:
        """Calculate base score using subprime-optimized weights."""
        
        score = 0
        max_possible = sum(self.subprime_weights.values())
        
        # Debt Service Coverage Ratio (28 points)
        dscr = metrics.get('Debt Service Coverage Ratio', 0)
        if dscr >= 3.0:
            score += self.subprime_weights['Debt Service Coverage Ratio']
        elif dscr >= 2.0:
            score += self.subprime_weights['Debt Service Coverage Ratio'] * 0.9
        elif dscr >= 1.5:
            score += self.subprime_weights['Debt Service Coverage Ratio'] * 0.8
        elif dscr >= 1.2:  # Minimum threshold for subprime
            score += self.subprime_weights['Debt Service Coverage Ratio'] * 0.6
        elif dscr >= 1.0:
            score += self.subprime_weights['Debt Service Coverage Ratio'] * 0.3
        # Below 1.0 gets 0 points
        
        # Revenue Growth Rate (20 points)
        growth = metrics.get('Revenue Growth Rate', 0)
        if growth >= 0.3:  # 30%+ growth
            score += self.subprime_weights['Revenue Growth Rate']
        elif growth >= 0.2:  # 20-30% growth
            score += self.subprime_weights['Revenue Growth Rate'] * 0.9
        elif growth >= 0.1:  # 10-20% growth
            score += self.subprime_weights['Revenue Growth Rate'] * 0.7
        elif growth >= 0.05:  # 5-10% growth
            score += self.subprime_weights['Revenue Growth Rate'] * 0.5
        elif growth >= 0:  # Flat to 5% growth
            score += self.subprime_weights['Revenue Growth Rate'] * 0.3
        elif growth >= -0.1:  # Small decline acceptable
            score += self.subprime_weights['Revenue Growth Rate'] * 0.1
        # Worse than -10% gets 0 points
        
        # Directors Score (16 points)
        directors_score = params.get('directors_score', 0)
        if directors_score >= 85:
            score += self.subprime_weights['Directors Score']
        elif directors_score >= 75:
            score += self.subprime_weights['Directors Score'] * 0.9
        elif directors_score >= 65:
            score += self.subprime_weights['Directors Score'] * 0.7
        elif directors_score >= 55:  # Lower threshold for subprime
            score += self.subprime_weights['Directors Score'] * 0.5
        elif directors_score >= 45:
            score += self.subprime_weights['Directors Score'] * 0.3
        # Below 45 gets 0 points
        
        # Average Month-End Balance (12 points)
        balance = metrics.get('Average Month-End Balance', 0)
        if balance >= 10000:
            score += self.subprime_weights['Average Month-End Balance']
        elif balance >= 5000:
            score += self.subprime_weights['Average Month-End Balance'] * 0.8
        elif balance >= 2000:
            score += self.subprime_weights['Average Month-End Balance'] * 0.6
        elif balance >= 500:  # Minimum for subprime
            score += self.subprime_weights['Average Month-End Balance'] * 0.4
        # Below £500 gets 0 points
        
        # Cash Flow Volatility (8 points) - Inverse scoring
        volatility = metrics.get('Cash Flow Volatility', 1.0)
        if volatility <= 0.15:
            score += self.subprime_weights['Cash Flow Volatility']
        elif volatility <= 0.3:
            score += self.subprime_weights['Cash Flow Volatility'] * 0.8
        elif volatility <= 0.5:
            score += self.subprime_weights['Cash Flow Volatility'] * 0.6
        elif volatility <= 0.8:
            score += self.subprime_weights['Cash Flow Volatility'] * 0.4
        elif volatility <= 1.0:  # Maximum tolerance for subprime
            score += self.subprime_weights['Cash Flow Volatility'] * 0.2
        # Above 1.0 gets 0 points
        
        # Operating Margin (6 points) - More tolerant of losses
        margin = metrics.get('Operating Margin', 0)
        if margin >= 0.1:
            score += self.subprime_weights['Operating Margin']
        elif margin >= 0.05:
            score += self.subprime_weights['Operating Margin'] * 0.8
        elif margin >= 0:
            score += self.subprime_weights['Operating Margin'] * 0.6
        elif margin >= -0.05:  # Small losses acceptable
            score += self.subprime_weights['Operating Margin'] * 0.4
        elif margin >= -0.1:  # Moderate losses with strong growth
            score += self.subprime_weights['Operating Margin'] * 0.2
        # Worse than -10% gets 0 points
        
        # Net Income (4 points) - Very tolerant for growth businesses
        net_income = metrics.get('Net Income', 0)
        if net_income >= 5000:
            score += self.subprime_weights['Net Income']
        elif net_income >= 0:
            score += self.subprime_weights['Net Income'] * 0.8
        elif net_income >= -10000:  # Losses acceptable if growth strong
            score += self.subprime_weights['Net Income'] * 0.5
        elif net_income >= -25000:
            score += self.subprime_weights['Net Income'] * 0.2
        # Worse than -£25k gets 0 points
        
        # Negative Balance Days (4 points)
        neg_days = metrics.get('Average Negative Balance Days per Month', 0)
        if neg_days <= 1:
            score += self.subprime_weights['Average Negative Balance Days per Month']
        elif neg_days <= 3:
            score += self.subprime_weights['Average Negative Balance Days per Month'] * 0.7
        elif neg_days <= 5:  # Higher tolerance
            score += self.subprime_weights['Average Negative Balance Days per Month'] * 0.4
        # More than 5 days gets 0 points
        
        # Company Age (2 points) - Minimal weight
        age_months = params.get('company_age_months', 0)
        if age_months >= 24:
            score += self.subprime_weights['Company Age (Months)']
        elif age_months >= 12:
            score += self.subprime_weights['Company Age (Months)'] * 0.7
        elif age_months >= 6:
            score += self.subprime_weights['Company Age (Months)'] * 0.4
        # Less than 6 months gets 0 points
        
        # Convert to percentage
        return (score / max_possible) * 100
    
    def _apply_industry_adjustment(self, base_score: float, industry: str) -> float:
        """Apply industry-specific risk adjustments."""
        
        multiplier = self.industry_multipliers.get(industry, 0.95)  # Default slight penalty for unknown
        return base_score * multiplier
    
    def _calculate_growth_momentum_bonus(self, metrics: Dict[str, Any]) -> float:
        """Calculate bonus for strong growth momentum."""
        
        bonus = 0
        growth = metrics.get('Revenue Growth Rate', 0)
        dscr = metrics.get('Debt Service Coverage Ratio', 0)
        
        # Strong growth + strong coverage = bonus
        if growth >= 0.25 and dscr >= 2.0:
            bonus += 5  # "Rocket ship" bonus
        elif growth >= 0.15 and dscr >= 1.5:
            bonus += 3  # Strong momentum bonus
        elif growth >= 0.1 and dscr >= 1.2:
            bonus += 1  # Modest momentum bonus
        
        return bonus
    
    def _calculate_stability_penalty(self, metrics: Dict[str, Any]) -> float:
        """Calculate penalty for extreme instability."""
        
        penalty = 0
        volatility = metrics.get('Cash Flow Volatility', 0)
        operating_margin = metrics.get('Operating Margin', 0)
        neg_days = metrics.get('Average Negative Balance Days per Month', 0)
        
        # Extreme volatility penalty
        if volatility > 1.0:
            penalty += (volatility - 1.0) * 10  # 10 points per unit over 1.0
        
        # Severe losses penalty
        if operating_margin < -0.15:  # More than 15% losses
            penalty += abs(operating_margin - (-0.15)) * 50  # Escalating penalty
        
        # Excessive negative balance days
        if neg_days > 5:
            penalty += (neg_days - 5) * 2  # 2 points per day over 5
        
        return min(penalty, 20)  # Cap penalty at 20 points
    
    def _determine_risk_tier(self, score: float, metrics: Dict[str, Any], params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Determine risk tier and pricing guidance based on score and factors."""
        
        dscr = metrics.get('Debt Service Coverage Ratio', 0)
        growth = metrics.get('Revenue Growth Rate', 0)
        directors_score = params.get('directors_score', 0)
        
        # Tier 1: Premium Subprime (65+ score with strong fundamentals)
        if (score >= 65 and dscr >= 2.0 and growth >= 0.15 and directors_score >= 75):
            return "Tier 1", {
                "risk_level": "Premium Subprime",
                "suggested_rate": "18-24% APR",
                "max_loan_multiple": "6x monthly revenue",
                "term_range": "12-24 months",
                "monitoring": "Quarterly reviews",
                "approval_probability": "Very High"
            }
        
        # Tier 2: Standard Subprime (50-65 score)
        elif (score >= 50 and dscr >= 1.5):
            return "Tier 2", {
                "risk_level": "Standard Subprime", 
                "suggested_rate": "24-36% APR",
                "max_loan_multiple": "4x monthly revenue",
                "term_range": "6-18 months",
                "monitoring": "Monthly reviews",
                "approval_probability": "High"
            }
        
        # Tier 3: High-Risk Subprime (35-50 score)
        elif (score >= 35 and dscr >= 1.2 and directors_score >= 55):
            return "Tier 3", {
                "risk_level": "High-Risk Subprime",
                "suggested_rate": "36-48% APR", 
                "max_loan_multiple": "3x monthly revenue",
                "term_range": "6-12 months",
                "monitoring": "Bi-weekly reviews",
                "approval_probability": "Moderate"
            }
        
        # Tier 4: Enhanced Monitoring Required
        elif (score >= 25 and dscr >= 1.0):
            return "Tier 4", {
                "risk_level": "Enhanced Monitoring Required",
                "suggested_rate": "48-60% APR",
                "max_loan_multiple": "2x monthly revenue", 
                "term_range": "3-9 months",
                "monitoring": "Weekly reviews + daily balance monitoring",
                "approval_probability": "Low - Senior review required"
            }
        
        # Decline
        else:
            return "Decline", {
                "risk_level": "Decline",
                "suggested_rate": "N/A",
                "max_loan_multiple": "N/A",
                "term_range": "N/A", 
                "monitoring": "N/A",
                "approval_probability": "Decline - Risk too high"
            }
    
    def _generate_scoring_breakdown(self, base_score, industry_score, growth_bonus, 
                                  stability_penalty, final_score, metrics, params) -> List[str]:
        """Generate detailed scoring breakdown."""
        
        breakdown = [
            f"Base Subprime Score: {base_score:.1f}/100",
            f"Industry Adjustment: {industry_score - base_score:+.1f} points",
            f"Growth Momentum Bonus: +{growth_bonus:.1f} points",
            f"Stability Penalty: -{stability_penalty:.1f} points",
            f"Final Score: {final_score:.1f}/100",
            "",
            "Key Factors:",
            f"• DSCR: {metrics.get('Debt Service Coverage Ratio', 0):.2f}",
            f"• Revenue Growth: {metrics.get('Revenue Growth Rate', 0)*100:.1f}%",
            f"• Directors Score: {params.get('directors_score', 0)}/100", 
            f"• Cash Flow Volatility: {metrics.get('Cash Flow Volatility', 0):.3f}",
            f"• Operating Margin: {metrics.get('Operating Margin', 0)*100:.1f}%"
        ]
        
        return breakdown
    
    def _generate_recommendation(self, risk_tier: str, metrics: Dict[str, Any], params: Dict[str, Any]) -> str:
        """Generate lending recommendation based on risk tier."""
        
        if risk_tier == "Tier 1":
            return "APPROVE - Excellent subprime candidate with strong fundamentals and growth trajectory."
        elif risk_tier == "Tier 2": 
            return "APPROVE - Good subprime candidate. Standard monitoring and pricing recommended."
        elif risk_tier == "Tier 3":
            return "CONDITIONAL APPROVE - Acceptable with enhanced terms and close monitoring."
        elif risk_tier == "Tier 4":
            return "SENIOR REVIEW - High risk but potentially acceptable with strict conditions."
        else:
            return "DECLINE - Risk profile exceeds acceptable parameters for subprime lending."
    
    def compare_scoring_methods(self, traditional_score: float, adaptive_score: float, 
                              ml_score: float, subprime_score: float) -> Dict[str, Any]:
        """Compare all scoring methods and provide unified guidance."""
        
        # Calculate score convergence
        scores = [traditional_score, adaptive_score, ml_score, subprime_score]
        score_range = max(scores) - min(scores)
        
        if score_range <= 15:
            convergence = "High"
        elif score_range <= 30:
            convergence = "Moderate" 
        else:
            convergence = "Low"
        
        # Primary recommendation based on subprime score (most relevant)
        if subprime_score >= 50:
            primary_rec = "Approve with appropriate subprime pricing"
        elif subprime_score >= 35:
            primary_rec = "Conditional approval with enhanced monitoring"
        else:
            primary_rec = "Decline - risk too high even for subprime"
        
        return {
            "score_convergence": convergence,
            "score_range": score_range,
            "primary_recommendation": primary_rec,
            "scores_summary": {
                "Traditional": traditional_score,
                "Adaptive": adaptive_score,
                "ML Model": ml_score,
                "Subprime": subprime_score
            },
            "most_relevant": "Subprime score most relevant for your business model"
        }


# Example usage and testing
def test_subprime_scoring():
    """Test the subprime scoring system with your example data."""
    
    # Your example business metrics
    test_metrics = {
        'Revenue Growth Rate': 0.245,  # 24.5%
        'Operating Margin': -0.1,      # -10%
        'Number of Bounced Payments': 0,
        'Net Income': -13398,
        'Gross Burn Rate': 11693.32,
        'Debt Service Coverage Ratio': 10.52,
        'Cash Flow Volatility': 0.66,
        'Average Negative Balance Days per Month': 2.0,
        'Average Month-End Balance': 13861.52
    }
    
    test_params = {
        'directors_score': 75,
        'company_age_months': 18,
        'industry': 'IT Services and Support Companies'  # Example
    }
    
    scorer = SubprimeScoring()
    result = scorer.calculate_subprime_score(test_metrics, test_params)
    
    print("=== SUBPRIME SCORING ANALYSIS ===")
    print(f"Subprime Score: {result['subprime_score']}/100")
    print(f"Risk Tier: {result['risk_tier']}")
    print(f"Recommendation: {result['recommendation']}")
    print("\nPricing Guidance:")
    for key, value in result['pricing_guidance'].items():
        print(f"  {key}: {value}")
    
    print(f"\nDetailed Breakdown:")
    for line in result['breakdown']:
        print(f"  {line}")
    
    return result

if __name__ == "__main__":
    test_result = test_subprime_scoring()