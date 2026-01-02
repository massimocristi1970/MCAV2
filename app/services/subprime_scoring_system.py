# Enhanced subprime_scoring_system.py with risk factor penalties
"""
Subprime Business Finance Scoring System - ENHANCED VERSION
Now includes risk factor penalties that match the V2 weighted system
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from datetime import datetime

from sklearn import metrics

class SubprimeScoring:
    """Enhanced scoring system specifically designed for subprime business lending with risk factor penalties."""
    
    def __init__(self):
    # TIGHTENED subprime weights for £1-10k short-term lending (6-9 months)
    # Emphasizes cash flow stability over growth for short-term repayment ability
        self.subprime_weights = {
            'Debt Service Coverage Ratio': 28,      # PRIMARY - Current ability to service debt
            'Average Month-End Balance': 18,        # INCREASED - Critical for short-term loans
            'Directors Score': 16,                  # HIGH - Personal reliability crucial in subprime
            'Cash Flow Volatility': 12,             # INCREASED - Stability crucial for short terms
            'Revenue Growth Rate': 10,              # REDUCED - Less relevant for 6-9 month loans
            'Operating Margin': 6,                  # LOW - Current losses more acceptable
            'Net Income': 4,                        # LOW - Growth more important than current profit
            'Average Negative Balance Days per Month': 4,  # Monitor but don't over-penalize
            'Company Age (Months)': 2,              # MINIMAL - Less relevant for growth businesses
        }
    
    # TIGHTENED risk tolerance thresholds for short-term subprime lending
        self.subprime_thresholds = {
            'minimum_dscr': 1.3,                   # RAISED from 1.2 - need margin for loan payment
            'maximum_volatility': 0.6,             # LOWERED from 1.0 - stability critical for short terms
            'minimum_growth': -0.05,               # TIGHTENED from -0.1 - less tolerance for decline
            'minimum_balance': 1500,               # RAISED from 500 - need buffer for payments
            'maximum_negative_days': 3,            # LOWERED from 5 - less tolerance for cash gaps
        }

    # NEW: Risk factor penalties specifically calibrated for subprime market
    self.risk_factor_penalties = {
        "business_ccj": 12,  # Severe penalty - business litigation risk
        "director_ccj": 8,  # High penalty - director financial issues
        'poor_or_no_online_presence': 4,  # Moderate penalty - business viability/operational concerns
        'uses_generic_email': 2  # Very minor penalty - professionalism
    }
        
        # Industry risk adjustments for subprime context
        self.industry_multipliers = {
            # Lower risk industries (bonus)
            'Medical Practices (GPs, Clinics, Dentists)': 1.1,
            'IT Services and Support Companies': 1.1,
            'Business Consultants': 1.05,
            'Education': 1.05,
            'Engineering': 1.05,
            'Telecommunications': 1.05,
            
            # Standard risk (no adjustment)
            'Manufacturing': 1.0,
            'Retail': 1.0,
            'Food Service': 1.0,
            'Tradesman': 1.0,
            'Courier Services (Independent and Regional Operators)': 1.0,
            'Grocery Stores and Mini-Markets': 1.0,
            'Estate Agent': 1.0,
            'Import / Export': 1.0,
            'Marketing / Advertising / Design': 1.0,
            'Off-Licence Business': 1.0,
            'Wholesaler / Distributor': 1.0,
            'Auto Repair Shops': 1.0,
            'Printing / Publishing': 1.0,
            'Recruitment': 1.0,
            'Personal Services': 1.0,
            'E-Commerce Retailers': 1.0,
            'Fitness Centres and Gyms': 1.0,
            'Other': 1.0,
            
            # Higher risk but still acceptable with pricing
            'Restaurants and Cafes': 0.9,
            'Construction Firms': 0.9,
            'Beauty Salons and Spas': 0.9,
            'Bars and Pubs': 0.85,
            'Event Planning and Management Firms': 0.8,
        }
    
    def calculate_subprime_score(self, metrics: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate comprehensive subprime business score with risk factor penalties.
        
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
        
        # NEW: Apply risk factor penalties
        risk_factor_penalty = self._calculate_risk_factor_penalties(params)
        
        # Final score calculation - NOW INCLUDES RISK FACTOR PENALTIES
        pre_penalty_score = industry_adjusted_score + growth_bonus - stability_penalty
        final_score = max(0, min(100, pre_penalty_score - risk_factor_penalty))
        
        # Determine risk tier and pricing
        risk_tier, pricing_guidance = self._determine_risk_tier(final_score, metrics, params)
        
        # Generate detailed breakdown
        breakdown = self._generate_scoring_breakdown(
            base_score, industry_adjusted_score, growth_bonus, 
            stability_penalty, risk_factor_penalty, final_score, metrics, params
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
        # Debt Service Coverage Ratio (28 points) - TIGHTENED for short-term lending
        dscr = metrics.get('Debt Service Coverage Ratio', 0)
        if dscr >= 3.0:
            score += self.subprime_weights['Debt Service Coverage Ratio']
        elif dscr >= 2.5:
            score += self.subprime_weights['Debt Service Coverage Ratio'] * 0.85
        elif dscr >= 2.0:
            score += self.subprime_weights['Debt Service Coverage Ratio'] * 0.7
        elif dscr >= 1.5:  # RAISED minimum threshold
            score += self.subprime_weights['Debt Service Coverage Ratio'] * 0.5
        elif dscr >= 1.3:  # RAISED from 1.0
            score += self.subprime_weights['Debt Service Coverage Ratio'] * 0.25
        # Below 1.3 gets 0 points (was 1.0) 
        
        # Revenue Growth Rate (10 points) - REDUCED weight for short-term lending
        growth = metrics.get('Revenue Growth Rate', 0)
        if growth >= 0.25:  # 25%+ growth
            score += self.subprime_weights['Revenue Growth Rate']
        elif growth >= 0.15:  # 15-25% growth
            score += self.subprime_weights['Revenue Growth Rate'] * 0.8
        elif growth >= 0.08:  # 8-15% growth
            score += self.subprime_weights['Revenue Growth Rate'] * 0.6
        elif growth >= 0.03:  # 3-8% growth
            score += self.subprime_weights['Revenue Growth Rate'] * 0.4
        elif growth >= 0:  # Flat to 3% growth
            score += self.subprime_weights['Revenue Growth Rate'] * 0.2
        # Below 0% gets 0 points (TIGHTENED - no credit for decline)
        
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
        
        # Average Month-End Balance (18 points) - INCREASED weight, TIGHTENED thresholds
        balance = metrics.get('Average Month-End Balance', 0)
        if balance >= 12000:
            score += self.subprime_weights['Average Month-End Balance']
        elif balance >= 8000:
            score += self.subprime_weights['Average Month-End Balance'] * 0.8
        elif balance >= 5000:
            score += self.subprime_weights['Average Month-End Balance'] * 0.6
        elif balance >= 3000:
            score += self.subprime_weights['Average Month-End Balance'] * 0.4
        elif balance >= 1500:  # RAISED minimum from £500
            score += self.subprime_weights['Average Month-End Balance'] * 0.2
        # Below £1500 gets 0 points (was £500)
        
        # Cash Flow Volatility (12 points) - INCREASED weight, TIGHTENED thresholds
        volatility = metrics.get('Cash Flow Volatility', 1.0)
        if volatility <= 0.15:
            score += self.subprime_weights['Cash Flow Volatility']
        elif volatility <= 0.25:
            score += self.subprime_weights['Cash Flow Volatility'] * 0.8
        elif volatility <= 0.35:
            score += self.subprime_weights['Cash Flow Volatility'] * 0.6
        elif volatility <= 0.45:
            score += self.subprime_weights['Cash Flow Volatility'] * 0.4
        elif volatility <= 0.6:  # LOWERED max tolerance from 1.0
            score += self.subprime_weights['Cash Flow Volatility'] * 0.2
        # Above 0.6 gets 0 points (was 1.0)
        
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
        
        # Negative Balance Days (4 points) - TIGHTENED thresholds
        neg_days = metrics.get('Average Negative Balance Days per Month', 0)
        if neg_days == 0:
            score += self.subprime_weights['Average Negative Balance Days per Month']
        elif neg_days <= 1:
            score += self.subprime_weights['Average Negative Balance Days per Month'] * 0.7
        elif neg_days <= 2:
            score += self.subprime_weights['Average Negative Balance Days per Month'] * 0.4
        elif neg_days <= 3:  # LOWERED max tolerance from 5
            score += self.subprime_weights['Average Negative Balance Days per Month'] * 0.2
        # More than 3 days gets 0 points (was 5)
        
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
    
    def _calculate_risk_factor_penalties(self, params: Dict[str, Any]) -> float:
        """NEW: Calculate penalties for risk factors (CCJs, defaults, etc.)"""
        
        total_penalty = 0
        applied_penalties = []
        
        # Check each risk factor and apply penalties

        if params.get('business_ccj', False):
            penalty = self.risk_factor_penalties['business_ccj']
            total_penalty += penalty
            applied_penalties.append(f"Business CCJ: -{penalty}")
        
        if params.get('director_ccj', False):
            penalty = self.risk_factor_penalties['director_ccj']
            total_penalty += penalty
            applied_penalties.append(f"Director CCJ: -{penalty}")
        

        if params.get('uses_generic_email', False):
            penalty = self.risk_factor_penalties['uses_generic_email']
            total_penalty += penalty
            applied_penalties.append(f"Generic Email: -{penalty}")

        if params.get('poor_or_no_online_presence', False):
            penalty = self.risk_factor_penalties['poor_or_no_online_presence']
            total_penalty += penalty
            applied_penalties.append(f"Poor/No Online Presence:  -{penalty}")

        # Cap maximum penalty to prevent "death by 1000 cuts"
        max_penalty_cap = 15
        if total_penalty > max_penalty_cap:
            applied_penalties.append(f"Penalty capped:  -{total_penalty} reduced to -{max_penalty_cap}")
            total_penalty = max_penalty_cap

        # Store applied penalties for breakdown
        params['_applied_risk_penalties'] = applied_penalties

        return total_penalty
    
    def _determine_risk_tier(self, score: float, metrics: Dict[str, Any], params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Determine risk tier and pricing guidance based on score and factors."""
        
        dscr = metrics.get('Debt Service Coverage Ratio', 0)
        growth = metrics.get('Revenue Growth Rate', 0)
        directors_score = params.get('directors_score', 0)

        has_major_risk_factors = (
                params.get('business_ccj', False) or
                params.get('director_ccj', False)
        )
        
        # Tier 1: Premium Subprime - TIGHTENED (82+ score with strong fundamentals)
        if (score >= 82 and dscr >= 2.5 and growth >= 0.15 and directors_score >= 75 
            and not has_major_risk_factors and metrics.get('Cash Flow Volatility', 1.0) <= 0.3):
            return "Tier 1", {
                "risk_level": "Premium Subprime",
                "suggested_rate": "1.5-1.6 factor rate",
                "max_loan_multiple": "4x monthly revenue",
                "term_range": "6-12 months",
                "monitoring": "Monthly reviews",
                "approval_probability": "Very High"
            }

        # Tier 2: Standard Subprime - TIGHTENED (70-82 score)
        elif (score >= 60 and dscr >= 1.5 and metrics.get('Cash Flow Volatility', 1.0) <= 0.60):
            rate_adjustment = "+0.1" if has_major_risk_factors else ""
            return "Tier 2", {
                "risk_level": "Standard Subprime", 
                "suggested_rate": f"1.7-1.85{rate_adjustment} factor rate",
                "max_loan_multiple": "3x monthly revenue",
                "term_range": "6-9 months",
                "monitoring": "Bi-weekly reviews" + (" + enhanced due diligence" if has_major_risk_factors else ""),
                "approval_probability": "High" if not has_major_risk_factors else "Moderate-High"
            }

        # Tier 3: High-Risk Subprime - TIGHTENED (55-70 score)
        elif (score >= 50 and dscr >= 1.3 and directors_score >= 50 and metrics.get('Cash Flow Volatility',
                                                                                    1.0) <= 0.70):
            rate_adjustment = "+0.15" if has_major_risk_factors else ""
            return "Tier 3", {
                "risk_level": "High-Risk Subprime",
                "suggested_rate": f"1.85-2.0{rate_adjustment} factor rate", 
                "max_loan_multiple": "2.    5x monthly revenue",
                "term_range": "4-6 months",
                "monitoring": "Weekly reviews" + (" + continuous risk monitoring" if has_major_risk_factors else ""),
                "approval_probability": "Moderate" if not has_major_risk_factors else "Low-Moderate"
            }

        # Tier 4: Enhanced Monitoring Required - TIGHTENED (40-55 score)
        elif (score >= 35 and dscr >= 1.1) or has_major_risk_factors:
            return "Tier 4", {
                "risk_level": "Enhanced Monitoring Required",
                "suggested_rate": "2.0-2.2+ factor rate",
                "max_loan_multiple": "2x monthly revenue", 
                "term_range": "3-6 months",
                "monitoring": "Weekly reviews + daily balance monitoring + personal guarantees REQUIRED",
                "approval_probability": "Low - Senior review required"
            }

        # Decline - More applications will fall here with tightened thresholds
        else:
            return "Decline", {
                "risk_level": "Decline",
                "suggested_rate": "N/A",
                "max_loan_multiple": "N/A",
                "term_range": "N/A", 
                "monitoring": "N/A",
                "approval_probability": "Decline - Risk too high for short-term subprime lending"
            }
    
    def _generate_scoring_breakdown(self, base_score, industry_score, growth_bonus, 
                                  stability_penalty, risk_factor_penalty, final_score, metrics, params) -> List[str]:
        """Generate detailed scoring breakdown including risk factor penalties."""
        
        breakdown = [
            f"Base Subprime Score: {base_score:.1f}/100",
            f"Industry Adjustment: {industry_score - base_score:+.1f} points",
            f"Growth Momentum Bonus: +{growth_bonus:.1f} points",
            f"Stability Penalty: -{stability_penalty:.1f} points",
            f"Risk Factor Penalties: -{risk_factor_penalty:.1f} points",  # NEW LINE
            f"Final Score: {final_score:.1f}/100",
            "",
            "Key Factors:",
            f"• DSCR: {metrics.get('Debt Service Coverage Ratio', 0):.2f}",
            f"• Revenue Growth: {metrics.get('Revenue Growth Rate', 0)*100:.1f}%",
            f"• Directors Score: {params.get('directors_score', 0)}/100", 
            f"• Cash Flow Volatility: {metrics.get('Cash Flow Volatility', 0):.3f}",
            f"• Operating Margin: {metrics.get('Operating Margin', 0)*100:.1f}%"
        ]
        
        # Add risk factor details if any were applied
        applied_penalties = params.get('_applied_risk_penalties', [])
        if applied_penalties:
            breakdown.append("")
            breakdown.append("Risk Factor Penalties Applied:")
            for penalty in applied_penalties:
                breakdown.append(f"• {penalty}")
        
        return breakdown
    
    def _generate_recommendation(self, risk_tier: str, metrics: Dict[str, Any], params: Dict[str, Any]) -> str:
        """Generate lending recommendation based on risk tier and risk factors."""
        
        has_major_risk_factors = (
            params.get('business_ccj', False) or
            params.get('director_ccj', False)
        )
        
        if risk_tier == "Tier 1":
            return "APPROVE - Excellent subprime candidate with strong fundamentals and minimal risk factors."
        elif risk_tier == "Tier 2": 
            if has_major_risk_factors:
                return "APPROVE - Good subprime candidate. Enhanced monitoring recommended due to risk factors."
            else:
                return "APPROVE - Good subprime candidate. Standard monitoring and pricing recommended."
        elif risk_tier == "Tier 3":
            if has_major_risk_factors:
                return "CONDITIONAL APPROVE - Acceptable with enhanced terms, close monitoring, and additional security due to risk factors."
            else:
                return "CONDITIONAL APPROVE - Acceptable with enhanced terms and close monitoring."
        elif risk_tier == "Tier 4":
            return "SENIOR REVIEW - High risk due to poor metrics or significant risk factors. Requires senior approval and strict conditions."
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
        if subprime_score >= 60:
            primary_rec = "Approve with appropriate subprime pricing"
        elif subprime_score >= 45:
            primary_rec = "Conditional approval with enhanced monitoring"
        elif subprime_score >= 30:
            primary_rec = "Enhanced monitoring required - senior review"    
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


# Example usage and testing with risk factors
def test_subprime_scoring_with_risk_factors():
    """Test the enhanced subprime scoring system with risk factors."""
    
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
    
    # Test different risk factor combinations
    test_scenarios = [
        {
            'name': 'No Risk Factors',
            'params': {
                'directors_score': 75,
                'company_age_months': 18,
                'industry': 'IT Services and Support Companies',
                'business_ccj': False,
                'director_ccj': False,
                'uses_generic_email': False,
                'poor_or_no_online_presence': False
            }
        },
        {
            'name': 'Minor Risk Factors',
            'params': {
                'directors_score': 75,
                'company_age_months': 18,
                'industry': 'IT Services and Support Companies',
                'business_ccj': False,
                'director_ccj': False,
                'uses_generic_email': True,
                'poor_or_no_online_presence': False
            }
        },
        {
            'name': 'Major Risk Factors',
            'params': {
                'directors_score': 75,
                'company_age_months': 18,
                'industry': 'IT Services and Support Companies',
                'business_ccj': True,
                'director_ccj': False,
                'uses_generic_email': False,
                'poor_or_no_online_presence': False
            }
        },
        {
            'name': 'All Risk Factors',
            'params': {
                'directors_score': 75,
                'company_age_months': 18,
                'industry': 'IT Services and Support Companies',
                'business_ccj': True,
                'director_ccj': True,
                'uses_generic_email': True,
                'poor_or_no_online_presence': True
            }
        }
    ]
    
    scorer = SubprimeScoring()
    
    print("=== ENHANCED SUBPRIME SCORING WITH RISK FACTORS ===\n")
    
    for scenario in test_scenarios:
        print(f"🧪 TEST SCENARIO: {scenario['name']}")
        print("=" * 50)
        
        result = scorer.calculate_subprime_score(test_metrics, scenario['params'])
        
        print(f"Subprime Score: {result['subprime_score']}/100")
        print(f"Risk Tier: {result['risk_tier']}")
        print(f"Recommendation: {result['recommendation']}")
        print("\nPricing Guidance:")
        for key, value in result['pricing_guidance'].items():
            print(f"  {key}: {value}")
        
        print(f"\nDetailed Breakdown:")
        for line in result['breakdown']:
            print(f"  {line}")
        
        print("\n" + "="*70 + "\n")
    
    return True

if __name__ == "__main__":
    test_subprime_scoring_with_risk_factors()