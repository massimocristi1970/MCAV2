# Enhanced subprime_scoring_system.py with risk factor penalties
"""
Subprime Business Finance Scoring System - MICRO ENTERPRISE FRIENDLY VERSION
Now includes risk factor penalties that match the V2 weighted system
Balanced for micro enterprises (£1-10k short-term lending)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from datetime import datetime

from sklearn import metrics


class SubprimeScoring:
    """Scoring system specifically designed for micro enterprise subprime lending with balanced risk assessment."""

    def __init__(self):
        # Subprime weights for £1-10k short-term lending (6-9 months)
        # Balanced for micro enterprise realities
        self.subprime_weights = {
            'Debt Service Coverage Ratio': 25,  # PRIMARY - Current ability to service debt
            'Average Month-End Balance': 18,  # Critical for short-term loans
            'Directors Score': 16,  # Personal reliability crucial in subprime
            'Cash Flow Volatility': 14,  # Stability important but realistic thresholds
            'Revenue Growth Rate': 10,  # Growth potential matters
            'Operating Margin': 6,  # Current losses acceptable for growth businesses
            'Net Income': 4,  # Growth more important than current profit
            'Average Negative Balance Days per Month': 5,  # Monitor but don't over-penalize
            'Company Age (Months)': 2,  # MINIMAL - Less relevant for growth businesses
        }

        # MICRO ENTERPRISE FRIENDLY risk tolerance thresholds
        self.subprime_thresholds = {
            'minimum_dscr': 0.8,  # Lowered - micro enterprises have variable cash flow
            'maximum_volatility': 1.0,  # Raised - micro enterprises have higher volatility
            'minimum_growth': -0.15,  # More tolerance for temporary decline
            'minimum_balance': 250,  # Lowered - realistic for micro enterprises
            'maximum_negative_days': 8,  # Raised - more tolerance for cash gaps
        }

        # Risk factor penalties - REDUCED for micro enterprise market
        self.risk_factor_penalties = {
            "business_ccj": 6,  # Reduced from 12 - still significant but not crushing
            "director_ccj": 4,  # Reduced from 8 - personal issues less penalized
            'poor_or_no_online_presence': 2,  # Reduced from 4 - many micro businesses lack online presence
            'uses_generic_email': 1  # Minimal penalty - very common for micro enterprises
        }

        # Industry risk adjustments - BALANCED for micro enterprise context
        self.industry_multipliers = {
            # Lower risk industries (modest bonus)
            'Medical Practices (GPs, Clinics, Dentists)': 1.05,
            'IT Services and Support Companies': 1.05,
            'Pharmacies (Independent or Small Chains)': 1.03,
            'Business Consultants': 1.03,
            'Education': 1.02,
            'Engineering': 1.02,
            'Telecommunications': 1.01,

            # Standard risk (no adjustment)
            'Manufacturing': 1.0,
            'Retail': 1.0,
            'Food Service': 0.98,
            'Tradesman': 0.98,
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
            'Fitness Centres and Gyms': 0.98,
            'Other': 0.97,

            # Higher risk but still acceptable - LESS HARSH penalties
            'Restaurants and Cafes': 0.95,
            'Construction Firms': 0.95,
            'Beauty Salons and Spas': 0.95,
            'Bars and Pubs': 0.93,
            'Event Planning and Management Firms': 0.92,
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

        # Apply risk factor penalties
        risk_factor_penalty = self._calculate_risk_factor_penalties(params)

        # Final score calculation - includes risk factor penalties
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
        """Calculate base score using micro enterprise friendly weights."""

        score = 0
        max_possible = sum(self.subprime_weights.values())

        # DEBT SERVICE COVERAGE RATIO (25 points) - Micro enterprise friendly
        dscr = metrics.get('Debt Service Coverage Ratio', 0)
        if dscr >= 1.8:
            score += self.subprime_weights['Debt Service Coverage Ratio']
        elif dscr >= 1.5:
            score += self.subprime_weights['Debt Service Coverage Ratio'] * 0.85
        elif dscr >= 1.2:
            score += self.subprime_weights['Debt Service Coverage Ratio'] * 0.70
        elif dscr >= 1.0:
            score += self.subprime_weights['Debt Service Coverage Ratio'] * 0.55
        elif dscr >= 0.8:
            score += self.subprime_weights['Debt Service Coverage Ratio'] * 0.35
        # Below 0.8 gets 0 points

        # REVENUE GROWTH RATE (10 points) - Realistic for micro enterprises
        growth = metrics.get('Revenue Growth Rate', 0)
        if growth >= 0.10:  # 10%+ growth - excellent
            score += self.subprime_weights['Revenue Growth Rate']
        elif growth >= 0.05:  # 5-10% growth - good
            score += self.subprime_weights['Revenue Growth Rate'] * 0.80
        elif growth >= 0.0:  # Flat to 5% - acceptable
            score += self.subprime_weights['Revenue Growth Rate'] * 0.60
        elif growth >= -0.05:  # Small decline - still acceptable
            score += self.subprime_weights['Revenue Growth Rate'] * 0.40
        elif growth >= -0.10:  # Moderate decline
            score += self.subprime_weights['Revenue Growth Rate'] * 0.20
        # Below -10% gets 0 points

        # DIRECTORS SCORE (16 points) - Balanced approach
        directors_score = params.get('directors_score', 50)
        if directors_score >= 75:
            score += self.subprime_weights['Directors Score']
        elif directors_score >= 55:
            score += self.subprime_weights['Directors Score'] * 0.80
        elif directors_score >= 45:
            score += self.subprime_weights['Directors Score'] * 0.60
        elif directors_score >= 35:
            score += self.subprime_weights['Directors Score'] * 0.40
        elif directors_score >= 25:
            score += self.subprime_weights['Directors Score'] * 0.20
        # Below 25 gets 0 points

        # AVERAGE MONTH-END BALANCE (18 points) - Micro enterprise friendly
        balance = metrics.get('Average Month-End Balance', 0)
        if balance >= 2000:
            score += self.subprime_weights['Average Month-End Balance']
        elif balance >= 1000:
            score += self.subprime_weights['Average Month-End Balance'] * 0.80
        elif balance >= 500:
            score += self.subprime_weights['Average Month-End Balance'] * 0.60
        elif balance >= 250:
            score += self.subprime_weights['Average Month-End Balance'] * 0.40
        elif balance >= 100:
            score += self.subprime_weights['Average Month-End Balance'] * 0.20
        # Below £100 gets 0 points

        # CASH FLOW VOLATILITY (14 points) - Micro enterprise friendly
        volatility = metrics.get('Cash Flow Volatility', 1.0)
        if volatility <= 0.35:
            score += self.subprime_weights['Cash Flow Volatility']
        elif volatility <= 0.50:
            score += self.subprime_weights['Cash Flow Volatility'] * 0.80
        elif volatility <= 0.65:
            score += self.subprime_weights['Cash Flow Volatility'] * 0.60
        elif volatility <= 0.80:
            score += self.subprime_weights['Cash Flow Volatility'] * 0.40
        elif volatility <= 1.0:
            score += self.subprime_weights['Cash Flow Volatility'] * 0.20
        # Above 1.0 gets 0 points

        # OPERATING MARGIN (6 points) - Tolerant of small losses
        margin = metrics.get('Operating Margin', 0)
        if margin >= 0.05:
            score += self.subprime_weights['Operating Margin']
        elif margin >= 0.03:
            score += self.subprime_weights['Operating Margin'] * 0.80
        elif margin >= 0.01:
            score += self.subprime_weights['Operating Margin'] * 0.60
        elif margin >= 0:
            score += self.subprime_weights['Operating Margin'] * 0.40
        elif margin >= -0.03:
            score += self.subprime_weights['Operating Margin'] * 0.20
        # Below -3% gets 0 points

        # NET INCOME (4 points) - Very tolerant for growth businesses
        net_income = metrics.get('Net Income', 0)
        if net_income >= 2000:
            score += self.subprime_weights['Net Income']
        elif net_income >= 0:
            score += self.subprime_weights['Net Income'] * 0.80
        elif net_income >= -5000:
            score += self.subprime_weights['Net Income'] * 0.60
        elif net_income >= -15000:
            score += self.subprime_weights['Net Income'] * 0.40
        elif net_income >= -25000:
            score += self.subprime_weights['Net Income'] * 0.20
        # Below -£25k gets 0 points

        # NEGATIVE BALANCE DAYS (5 points) - Micro enterprise friendly
        neg_days = metrics.get('Average Negative Balance Days per Month', 0)
        if neg_days <= 2:
            score += self.subprime_weights['Average Negative Balance Days per Month']
        elif neg_days <= 5:
            score += self.subprime_weights['Average Negative Balance Days per Month'] * 0.80
        elif neg_days <= 8:
            score += self.subprime_weights['Average Negative Balance Days per Month'] * 0.60
        elif neg_days <= 12:
            score += self.subprime_weights['Average Negative Balance Days per Month'] * 0.40
        elif neg_days <= 15:
            score += self.subprime_weights['Average Negative Balance Days per Month'] * 0.20
        # Above 15 days gets 0 points

        # COMPANY AGE (2 points) - Minimal weight
        age_months = params.get('company_age_months', 0)
        if age_months >= 18:
            score += self.subprime_weights['Company Age (Months)']
        elif age_months >= 12:
            score += self.subprime_weights['Company Age (Months)'] * 0.80
        elif age_months >= 9:
            score += self.subprime_weights['Company Age (Months)'] * 0.60
        elif age_months >= 6:
            score += self.subprime_weights['Company Age (Months)'] * 0.40
        elif age_months >= 3:
            score += self.subprime_weights['Company Age (Months)'] * 0.20
        # Below 3 months gets 0 points

        # Convert to percentage
        return (score / max_possible) * 100

    def _apply_industry_adjustment(self, base_score: float, industry: str) -> float:
        """Apply industry-specific risk adjustments - balanced approach."""
        multiplier = self.industry_multipliers.get(industry, 0.95)  # Default reasonable for unknown
        return base_score * multiplier

    def _calculate_growth_momentum_bonus(self, metrics: Dict[str, Any]) -> float:
        """Calculate bonus for strong growth momentum - micro enterprise friendly."""

        bonus = 0
        growth = metrics.get('Revenue Growth Rate', 0)
        dscr = metrics.get('Debt Service Coverage Ratio', 0)

        # Strong growth + adequate coverage = bonus (lowered thresholds)
        if growth >= 0.15 and dscr >= 1.5:
            bonus += 5  # Strong momentum bonus
        elif growth >= 0.10 and dscr >= 1.2:
            bonus += 3  # Good momentum bonus
        elif growth >= 0.05 and dscr >= 1.0:
            bonus += 2  # Modest momentum bonus
        elif growth >= 0 and dscr >= 0.8:
            bonus += 1  # Stability bonus

        return bonus

    def _calculate_stability_penalty(self, metrics: Dict[str, Any]) -> float:
        """Calculate penalty for instability - only for extreme cases."""

        penalty = 0
        volatility = metrics.get('Cash Flow Volatility', 0)
        operating_margin = metrics.get('Operating Margin', 0)
        neg_days = metrics.get('Average Negative Balance Days per Month', 0)
        dscr = metrics.get('Debt Service Coverage Ratio', 0)

        # Only penalize extreme volatility (above 1.0)
        if volatility > 1.0:
            penalty += (volatility - 1.0) * 10

        # Only penalize severe losses (below -10%)
        if operating_margin < -0.10:
            penalty += abs(operating_margin - (-0.10)) * 30

        # Only penalize excessive negative balance days (above 10)
        if neg_days > 10:
            penalty += (neg_days - 10) * 1.
            5

        # Only penalize very low DSCR (below 0.8)
        if dscr < 0.8:
            penalty += (0.8 - dscr) * 8

        return min(penalty, 15)  # Cap at 15 points max

    def _calculate_risk_factor_penalties(self, params: Dict[str, Any]) -> float:
        """Calculate penalties for risk factors (CCJs, defaults, etc.) - capped for fairness."""

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
            applied_penalties.append(f"Director CCJ:  -{penalty}")

        if params.get('uses_generic_email', False):
            penalty = self.risk_factor_penalties['uses_generic_email']
            total_penalty += penalty
            applied_penalties.append(f"Generic Email: -{penalty}")

        if params.get('poor_or_no_online_presence', False):
            penalty = self.risk_factor_penalties['poor_or_no_online_presence']
            total_penalty += penalty
            applied_penalties.append(f"Poor/No Online Presence: -{penalty}")

        # Cap maximum penalty to prevent "death by 1000 cuts"
        max_penalty_cap = 12
        if total_penalty > max_penalty_cap:
            applied_penalties.append(f"Penalty capped:  -{total_penalty} reduced to -{max_penalty_cap}")
            total_penalty = max_penalty_cap

        # Store applied penalties for breakdown
        params['_applied_risk_penalties'] = applied_penalties

        return total_penalty

    def _determine_risk_tier(self, score: float, metrics: Dict[str, Any], params: Dict[str, Any]) -> Tuple[
        str, Dict[str, Any]]:
        """Determine risk tier and pricing guidance - micro enterprise friendly thresholds."""

        dscr = metrics.get('Debt Service Coverage Ratio', 0)
        growth = metrics.get('Revenue Growth Rate', 0)
        directors_score = params.get('directors_score', 0)
        volatility = metrics.get('Cash Flow Volatility', 1.0)

        has_major_risk_factors = (
                params.get('business_ccj', False) or
                params.get('director_ccj', False)
        )

        # Tier 1: Premium - Score 65+, good fundamentals
        if (score >= 65 and dscr >= 1.5 and directors_score >= 55
                and not has_major_risk_factors and volatility <= 0.60):
            return "Tier 1", {
                "risk_level": "Premium Micro Enterprise",
                "suggested_rate": "1.5-1.6 factor rate",
                "max_loan_multiple": "4x monthly revenue",
                "term_range": "6-12 months",
                "monitoring": "Monthly reviews",
                "approval_probability": "Very High"
            }

        # Tier 2: Standard - Score 50-65, adequate fundamentals
        elif (score >= 50 and dscr >= 1.2 and volatility <= 0.80):
            rate_adjustment = "+0.1" if has_major_risk_factors else ""
            return "Tier 2", {
                "risk_level": "Standard Micro Enterprise",
                "suggested_rate": f"1.7-1.85{rate_adjustment} factor rate",
                "max_loan_multiple": "3x monthly revenue",
                "term_range": "6-9 months",
                "monitoring": "Bi-weekly reviews" + (" + enhanced due diligence" if has_major_risk_factors else ""),
                "approval_probability": "High" if not has_major_risk_factors else "Moderate-High"
            }

        # Tier 3: Higher Risk - Score 40-50, minimum viable fundamentals
        elif (score >= 40 and dscr >= 1.0 and directors_score >= 35 and volatility <= 1.0):
            rate_adjustment = "+0.15" if has_major_risk_factors else ""
            return "Tier 3", {
                "risk_level": "Higher Risk Micro Enterprise",
                "suggested_rate": f"1.85-2.0{rate_adjustment} factor rate",
                "max_loan_multiple": "2.5x monthly revenue",
                "term_range": "4-6 months",
                "monitoring": "Weekly reviews" + (" + continuous risk monitoring" if has_major_risk_factors else ""),
                "approval_probability": "Moderate" if not has_major_risk_factors else "Low-Moderate"
            }

        # Tier 4: Senior Review Required - Score 30-40
        elif (score >= 30 and dscr >= 0.8):
            return "Tier 4", {
                "risk_level": "Senior Review Required",
                "suggested_rate": "2.0-2.2+ factor rate",
                "max_loan_multiple": "2x monthly revenue",
                "term_range": "3-6 months",
                "monitoring": "Weekly reviews + daily balance monitoring + personal guarantees recommended",
                "approval_probability": "Low - Senior review required"
            }

        # Decline - Below minimum thresholds
        else:
            return "Decline", {
                "risk_level": "Decline",
                "suggested_rate": "N/A",
                "max_loan_multiple": "N/A",
                "term_range": "N/A",
                "monitoring": "N/A",
                "approval_probability": "Decline - Consider reapplying after 3-6 months of improved trading"
            }

    def _generate_scoring_breakdown(self, base_score, industry_score, growth_bonus,
                                    stability_penalty, risk_factor_penalty, final_score, metrics, params) -> List[str]:
        """Generate detailed scoring breakdown including risk factor penalties."""

        breakdown = [
            f"Base Score: {base_score:.1f}/100",
            f"Industry Adjustment: {industry_score - base_score: +.1f} points",
            f"Growth Momentum Bonus: +{growth_bonus:. 1f} points",
            f"Stability Penalty: -{stability_penalty:.1f} points",
            f"Risk Factor Penalties: -{risk_factor_penalty:.1f} points",
            f"Final Score: {final_score:.1f}/100",
            "",
            "Key Metrics (Micro Enterprise Thresholds):",
            f"• DSCR: {metrics.get('Debt Service Coverage Ratio', 0):.2f} (Need 1.2+ for good score)",
            f"• Revenue Growth: {metrics.get('Revenue Growth Rate', 0) * 100:.1f}% (Need 5%+ for good score)",
            f"• Directors Score: {params.get('directors_score', 0)}/100 (Need 55+ for good score)",
            f"• Cash Flow Volatility: {metrics.get('Cash Flow Volatility', 0):.3f} (Need <0.50 for good score)",
            f"• Operating Margin: {metrics.get('Operating Margin', 0) * 100:.1f}% (Need 3%+ for good score)",
            f"• Negative Balance Days:  {metrics.get('Average Negative Balance Days per Month', 0):. 0f} (Need <5 for good score)"
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
            return "APPROVE - Strong micro enterprise candidate with solid fundamentals."
        elif risk_tier == "Tier 2":
            if has_major_risk_factors:
                return "APPROVE - Good candidate with standard monitoring requirements.  Enhanced due diligence for risk factors."
            else:
                return "APPROVE - Good candidate with standard monitoring requirements."
        elif risk_tier == "Tier 3":
            if has_major_risk_factors:
                return "CONDITIONAL APPROVE - Viable candidate with enhanced terms and monitoring required."
            else:
                return "CONDITIONAL APPROVE - Viable candidate with enhanced terms and monitoring."
        elif risk_tier == "Tier 4":
            return "SENIOR REVIEW - Higher risk profile, requires additional review and guarantees."
        else:
            return "DECLINE - Consider reapplying after 3-6 months of improved trading performance."

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
            primary_rec = "Approve with appropriate pricing"
        elif subprime_score >= 40:
            primary_rec = "Conditional approval with enhanced monitoring"
        elif subprime_score >= 30:
            primary_rec = "Senior review required"
        else:
            primary_rec = "Decline - consider reapplying after improved trading"

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
            "most_relevant": "Subprime score most relevant for micro enterprise lending"
        }


# Example usage and testing with risk factors
def test_subprime_scoring_with_risk_factors():
    """Test the micro enterprise friendly subprime scoring system."""

    # Example micro enterprise metrics
    test_metrics = {
        'Revenue Growth Rate': 0.08,  # 8% growth
        'Operating Margin': 0.02,  # 2% margin
        'Number of Bounced Payments': 0,
        'Net Income': 1500,
        'Gross Burn Rate': 8000,
        'Debt Service Coverage Ratio': 1.3,
        'Cash Flow Volatility': 0.45,
        'Average Negative Balance Days per Month': 3,
        'Average Month-End Balance': 1200
    }

    # Test different risk factor combinations
    test_scenarios = [
        {
            'name': 'No Risk Factors',
            'params': {
                'directors_score': 60,
                'company_age_months': 14,
                'industry': 'Retail',
                'business_ccj': False,
                'director_ccj': False,
                'uses_generic_email': False,
                'poor_or_no_online_presence': False
            }
        },
        {
            'name': 'Minor Risk Factors',
            'params': {
                'directors_score': 60,
                'company_age_months': 14,
                'industry': 'Retail',
                'business_ccj': False,
                'director_ccj': False,
                'uses_generic_email': True,
                'poor_or_no_online_presence': True
            }
        },
        {
            'name': 'Major Risk Factors',
            'params': {
                'directors_score': 60,
                'company_age_months': 14,
                'industry': 'Retail',
                'business_ccj': True,
                'director_ccj': False,
                'uses_generic_email': False,
                'poor_or_no_online_presence': False
            }
        },
        {
            'name': 'All Risk Factors',
            'params': {
                'directors_score': 60,
                'company_age_months': 14,
                'industry': 'Retail',
                'business_ccj': True,
                'director_ccj': True,
                'uses_generic_email': True,
                'poor_or_no_online_presence': True
            }
        }
    ]

    scorer = SubprimeScoring()

    print("=== MICRO ENTERPRISE FRIENDLY SUBPRIME SCORING ===\n")

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

        print("\n" + "=" * 70 + "\n")

    return True


if __name__ == "__main__":
    test_subprime_scoring_with_risk_factors()