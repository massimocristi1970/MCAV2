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
        
        # Generate comprehensive diagnostics
        diagnostics = self._generate_score_diagnostics(metrics, params, final_score)

        return {
            'subprime_score': round(final_score, 1),
            'risk_tier': risk_tier,
            'pricing_guidance': pricing_guidance,
            'breakdown': breakdown,
            'recommendation': self._generate_recommendation(risk_tier, metrics, params),
            'diagnostics': diagnostics
        }

    def _calculate_base_subprime_score(self, metrics: Dict[str, Any], params: Dict[str, Any]) -> float:
        """Calculate base score using micro enterprise friendly weights."""

        score = 0
        max_possible = sum(self.subprime_weights.values())

        # DEBT SERVICE COVERAGE RATIO (25 points) - slightly tightened
        dscr = metrics.get('Debt Service Coverage Ratio', 0)
        if dscr >= 1.9:
            score += w
        elif dscr >= 1.6:
            score += w * 0.85
        elif dscr >= 1.3:
            score += w * 0.65  # was 0.70
        elif dscr >= 1.1:
            score += w * 0.45  # was 0.55
        elif dscr >= 0.9:
            score += w * 0.25  # was 0.35
        # Below 0.9 gets 0 points

        # REVENUE GROWTH RATE (10 points) - slightly tightened downside
        growth = metrics.get('Revenue Growth Rate', 0)
        if growth >= 0.12:
            score += w
        elif growth >= 0.06:
            score += w * 0.80
        elif growth >= 0.01:
            score += w * 0.55  # was 0.60 for >=0.0
        elif growth >= -0.03:
            score += w * 0.30  # was 0.40 for >=-0.05
        elif growth >= -0.07:
            score += w * 0.15  # was 0.20 for >=-0.10
        # Below -7% gets 0 points

        # DIRECTORS SCORE (16 points) - slightly tightened
        d = params.get('directors_score', 50)
        if d >= 78:
            score += w
        elif d >= 60:
            score += w * 0.80
        elif d >= 50:
            score += w * 0.55  # was 0.60 at 45
        elif d >= 40:
            score += w * 0.35  # was 0.40 at 35
        elif d >= 30:
            score += w * 0.15  # was 0.20 at 25
        # Below 30 gets 0 points

        # AVERAGE MONTH-END BALANCE (18 points) - slightly tightened
        bal = metrics.get('Average Month-End Balance', 0)
        if bal >= 2500:
            score += w
        elif bal >= 1500:
            score += w * 0.80
        elif bal >= 750:
            score += w * 0.55  # was 0.60 at 500
        elif bal >= 400:
            score += w * 0.35  # was 0.40 at 250
        elif bal >= 200:
            score += w * 0.15  # was 0.20 at 100
        # Below £200 gets 0 points

        # CASH FLOW VOLATILITY (14 points) - slightly tightened
        vol = metrics.get('Cash Flow Volatility', 1.0)
        if vol <= 0.30:
            score += w
        elif vol <= 0.45:
            score += w * 0.80
        elif vol <= 0.60:
            score += w * 0.55  # was 0.60 at 0.65
        elif vol <= 0.75:
            score += w * 0.30  # was 0.40 at 0.80
        elif vol <= 0.95:
            score += w * 0.10  # was 0.20 at 1.0
        # Above 0.95 gets 0 points

        # OPERATING MARGIN (6 points) - slightly tightened
        m = metrics.get('Operating Margin', 0)
        if m >= 0.06:
            score += w
        elif m >= 0.04:
            score += w * 0.80
        elif m >= 0.02:
            score += w * 0.60
        elif m >= 0.005:
            score += w * 0.35  # was 0.40 at 0
        elif m >= -0.02:
            score += w * 0.15  # was 0.20 at -0.03
        # Below -2% gets 0 points

        # NET INCOME (4 points) - slightly tightened
        ni = metrics.get('Net Income', 0)
        if ni >= 3000:
            score += w
        elif ni >= 500:
            score += w * 0.80
        elif ni >= -2500:
            score += w * 0.45  # was 0.60 to -5000
        elif ni >= -10000:
            score += w * 0.25  # was 0.40 to -15000
        elif ni >= -20000:
            score += w * 0.10  # was 0.20 to -25000
        # Below -£20k gets 0 points

        # NEGATIVE BALANCE DAYS (5 points) - slightly tightened
        nd = metrics.get('Average Negative Balance Days per Month', 0)
        if nd <= 1:
            score += w
        elif nd <= 4:
            score += w * 0.80
        elif nd <= 7:
            score += w * 0.55  # was 0.60 at 8
        elif nd <= 10:
            score += w * 0.30  # was 0.40 at 12
        elif nd <= 13:
            score += w * 0.10  # was 0.20 at 15
        # Above 13 gets 0 points

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
            f"Industry Adjustment: {industry_score - base_score:+.1f} points",
            f"Growth Momentum Bonus: +{growth_bonus:.1f} points",
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
            f"• Negative Balance Days: {metrics.get('Average Negative Balance Days per Month', 0):.0f} (Need <5 for good score)"
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

    def _generate_score_diagnostics(self, metrics: Dict[str, Any], params: Dict[str, Any], final_score: float) -> Dict[str, Any]:
        """
        Generate detailed diagnostics showing why the score is what it is.
        
        This provides comprehensive insights into:
        - How each metric performed
        - What's hurting the score most  
        - What's helping the score
        - Specific improvement suggestions
        """
        
        diagnostics = {
            'metric_breakdown': [],
            'top_negative_factors': [],
            'top_positive_factors': [],
            'threshold_failures': [],
            'improvement_suggestions': []
        }
        
        # Track metric performance with detailed breakdown
        metric_performance = []
        
        # DEBT SERVICE COVERAGE RATIO
        dscr = metrics.get('Debt Service Coverage Ratio', 0)
        dscr_threshold_full = 1.8
        dscr_threshold_min = 0.8
        dscr_max_points = self.subprime_weights['Debt Service Coverage Ratio']
        
        if dscr >= dscr_threshold_full:
            dscr_points = dscr_max_points
            dscr_percentage = 100.0
            dscr_status = 'PASS'
        elif dscr >= 1.5:
            dscr_points = dscr_max_points * 0.85
            dscr_percentage = 85.0
            dscr_status = 'PARTIAL'
        elif dscr >= 1.2:
            dscr_points = dscr_max_points * 0.70
            dscr_percentage = 70.0
            dscr_status = 'PARTIAL'
        elif dscr >= 1.0:
            dscr_points = dscr_max_points * 0.55
            dscr_percentage = 55.0
            dscr_status = 'PARTIAL'
        elif dscr >= dscr_threshold_min:
            dscr_points = dscr_max_points * 0.35
            dscr_percentage = 35.0
            dscr_status = 'PARTIAL'
        else:
            dscr_points = 0
            dscr_percentage = 0
            dscr_status = 'FAIL'
        
        metric_performance.append({
            'metric': 'Debt Service Coverage Ratio',
            'actual_value': dscr,
            'threshold_full_points': dscr_threshold_full,
            'threshold_min_points': dscr_threshold_min,
            'points_earned': round(dscr_points, 2),
            'points_possible': dscr_max_points,
            'percentage': round(dscr_percentage, 1),
            'status': dscr_status,
            'gap_to_full': max(0, dscr_threshold_full - dscr)
        })
        
        # CASH FLOW VOLATILITY
        volatility = metrics.get('Cash Flow Volatility', 1.0)
        vol_threshold_full = 0.35
        vol_threshold_max = 1.0
        vol_max_points = self.subprime_weights['Cash Flow Volatility']
        
        if volatility <= vol_threshold_full:
            vol_points = vol_max_points
            vol_percentage = 100.0
            vol_status = 'PASS'
        elif volatility <= 0.50:
            vol_points = vol_max_points * 0.80
            vol_percentage = 80.0
            vol_status = 'PARTIAL'
        elif volatility <= 0.65:
            vol_points = vol_max_points * 0.60
            vol_percentage = 60.0
            vol_status = 'PARTIAL'
        elif volatility <= 0.80:
            vol_points = vol_max_points * 0.40
            vol_percentage = 40.0
            vol_status = 'PARTIAL'
        elif volatility <= vol_threshold_max:
            vol_points = vol_max_points * 0.20
            vol_percentage = 20.0
            vol_status = 'PARTIAL'
        else:
            vol_points = 0
            vol_percentage = 0
            vol_status = 'FAIL'
        
        metric_performance.append({
            'metric': 'Cash Flow Volatility',
            'actual_value': volatility,
            'threshold_full_points': vol_threshold_full,
            'threshold_min_points': vol_threshold_max,
            'points_earned': round(vol_points, 2),
            'points_possible': vol_max_points,
            'percentage': round(vol_percentage, 1),
            'status': vol_status,
            'gap_to_full': max(0, volatility - vol_threshold_full)
        })
        
        # DIRECTORS SCORE
        directors = params.get('directors_score', 50)
        dir_threshold_full = 70
        dir_threshold_min = 25
        dir_max_points = self.subprime_weights['Directors Score']
        
        if directors >= dir_threshold_full:
            dir_points = dir_max_points
            dir_percentage = 100.0
            dir_status = 'PASS'
        elif directors >= 55:
            dir_points = dir_max_points * 0.80
            dir_percentage = 80.0
            dir_status = 'PARTIAL'
        elif directors >= 45:
            dir_points = dir_max_points * 0.60
            dir_percentage = 60.0
            dir_status = 'PARTIAL'
        elif directors >= 35:
            dir_points = dir_max_points * 0.40
            dir_percentage = 40.0
            dir_status = 'PARTIAL'
        elif directors >= dir_threshold_min:
            dir_points = dir_max_points * 0.20
            dir_percentage = 20.0
            dir_status = 'PARTIAL'
        else:
            dir_points = 0
            dir_percentage = 0
            dir_status = 'FAIL'
        
        metric_performance.append({
            'metric': 'Directors Score',
            'actual_value': directors,
            'threshold_full_points': dir_threshold_full,
            'threshold_min_points': dir_threshold_min,
            'points_earned': round(dir_points, 2),
            'points_possible': dir_max_points,
            'percentage': round(dir_percentage, 1),
            'status': dir_status,
            'gap_to_full': max(0, dir_threshold_full - directors)
        })
        
        # AVERAGE MONTH-END BALANCE
        balance = metrics.get('Average Month-End Balance', 0)
        bal_threshold_full = 2000
        bal_threshold_min = 100
        bal_max_points = self.subprime_weights['Average Month-End Balance']
        
        if balance >= bal_threshold_full:
            bal_points = bal_max_points
            bal_percentage = 100.0
            bal_status = 'PASS'
        elif balance >= 1000:
            bal_points = bal_max_points * 0.80
            bal_percentage = 80.0
            bal_status = 'PARTIAL'
        elif balance >= 500:
            bal_points = bal_max_points * 0.60
            bal_percentage = 60.0
            bal_status = 'PARTIAL'
        elif balance >= 250:
            bal_points = bal_max_points * 0.40
            bal_percentage = 40.0
            bal_status = 'PARTIAL'
        elif balance >= bal_threshold_min:
            bal_points = bal_max_points * 0.20
            bal_percentage = 20.0
            bal_status = 'PARTIAL'
        else:
            bal_points = 0
            bal_percentage = 0
            bal_status = 'FAIL'
        
        metric_performance.append({
            'metric': 'Average Month-End Balance',
            'actual_value': balance,
            'threshold_full_points': bal_threshold_full,
            'threshold_min_points': bal_threshold_min,
            'points_earned': round(bal_points, 2),
            'points_possible': bal_max_points,
            'percentage': round(bal_percentage, 1),
            'status': bal_status,
            'gap_to_full': max(0, bal_threshold_full - balance)
        })
        
        # REVENUE GROWTH RATE
        growth = metrics.get('Revenue Growth Rate', 0)
        growth_threshold_full = 0.10  # 10%
        growth_threshold_min = -0.15  # -15%
        growth_max_points = self.subprime_weights['Revenue Growth Rate']
        
        if growth >= growth_threshold_full:
            growth_points = growth_max_points
            growth_percentage = 100.0
            growth_status = 'PASS'
        elif growth >= 0.05:
            growth_points = growth_max_points * 0.80
            growth_percentage = 80.0
            growth_status = 'PARTIAL'
        elif growth >= 0:
            growth_points = growth_max_points * 0.60
            growth_percentage = 60.0
            growth_status = 'PARTIAL'
        elif growth >= -0.05:
            growth_points = growth_max_points * 0.40
            growth_percentage = 40.0
            growth_status = 'PARTIAL'
        elif growth >= growth_threshold_min:
            growth_points = growth_max_points * 0.20
            growth_percentage = 20.0
            growth_status = 'PARTIAL'
        else:
            growth_points = 0
            growth_percentage = 0
            growth_status = 'FAIL'
        
        metric_performance.append({
            'metric': 'Revenue Growth Rate',
            'actual_value': growth,
            'threshold_full_points': growth_threshold_full,
            'threshold_min_points': growth_threshold_min,
            'points_earned': round(growth_points, 2),
            'points_possible': growth_max_points,
            'percentage': round(growth_percentage, 1),
            'status': growth_status,
            'gap_to_full': max(0, growth_threshold_full - growth)
        })
        
        # OPERATING MARGIN
        margin = metrics.get('Operating Margin', 0)
        margin_threshold_full = 0.05  # 5%
        margin_threshold_min = -0.03  # -3%
        margin_max_points = self.subprime_weights['Operating Margin']
        
        if margin >= margin_threshold_full:
            margin_points = margin_max_points
            margin_percentage = 100.0
            margin_status = 'PASS'
        elif margin >= 0.03:
            margin_points = margin_max_points * 0.80
            margin_percentage = 80.0
            margin_status = 'PARTIAL'
        elif margin >= 0.01:
            margin_points = margin_max_points * 0.60
            margin_percentage = 60.0
            margin_status = 'PARTIAL'
        elif margin >= 0:
            margin_points = margin_max_points * 0.40
            margin_percentage = 40.0
            margin_status = 'PARTIAL'
        elif margin >= margin_threshold_min:
            margin_points = margin_max_points * 0.20
            margin_percentage = 20.0
            margin_status = 'PARTIAL'
        else:
            margin_points = 0
            margin_percentage = 0
            margin_status = 'FAIL'
        
        metric_performance.append({
            'metric': 'Operating Margin',
            'actual_value': margin,
            'threshold_full_points': margin_threshold_full,
            'threshold_min_points': margin_threshold_min,
            'points_earned': round(margin_points, 2),
            'points_possible': margin_max_points,
            'percentage': round(margin_percentage, 1),
            'status': margin_status,
            'gap_to_full': max(0, margin_threshold_full - margin)
        })
        
        # NEGATIVE BALANCE DAYS
        neg_days = metrics.get('Average Negative Balance Days per Month', 0)
        neg_threshold_full = 2
        neg_threshold_max = 15
        neg_max_points = self.subprime_weights['Average Negative Balance Days per Month']
        
        if neg_days <= neg_threshold_full:
            neg_points = neg_max_points
            neg_percentage = 100.0
            neg_status = 'PASS'
        elif neg_days <= 5:
            neg_points = neg_max_points * 0.80
            neg_percentage = 80.0
            neg_status = 'PARTIAL'
        elif neg_days <= 8:
            neg_points = neg_max_points * 0.60
            neg_percentage = 60.0
            neg_status = 'PARTIAL'
        elif neg_days <= 12:
            neg_points = neg_max_points * 0.40
            neg_percentage = 40.0
            neg_status = 'PARTIAL'
        elif neg_days <= neg_threshold_max:
            neg_points = neg_max_points * 0.20
            neg_percentage = 20.0
            neg_status = 'PARTIAL'
        else:
            neg_points = 0
            neg_percentage = 0
            neg_status = 'FAIL'
        
        metric_performance.append({
            'metric': 'Negative Balance Days',
            'actual_value': neg_days,
            'threshold_full_points': neg_threshold_full,
            'threshold_min_points': neg_threshold_max,
            'points_earned': round(neg_points, 2),
            'points_possible': neg_max_points,
            'percentage': round(neg_percentage, 1),
            'status': neg_status,
            'gap_to_full': max(0, neg_days - neg_threshold_full)
        })
        
        # COMPANY AGE
        age = params.get('company_age_months', 0)
        age_threshold_full = 18
        age_threshold_min = 3
        age_max_points = self.subprime_weights['Company Age (Months)']
        
        if age >= age_threshold_full:
            age_points = age_max_points
            age_percentage = 100.0
            age_status = 'PASS'
        elif age >= 12:
            age_points = age_max_points * 0.80
            age_percentage = 80.0
            age_status = 'PARTIAL'
        elif age >= 9:
            age_points = age_max_points * 0.60
            age_percentage = 60.0
            age_status = 'PARTIAL'
        elif age >= 6:
            age_points = age_max_points * 0.40
            age_percentage = 40.0
            age_status = 'PARTIAL'
        elif age >= age_threshold_min:
            age_points = age_max_points * 0.20
            age_percentage = 20.0
            age_status = 'PARTIAL'
        else:
            age_points = 0
            age_percentage = 0
            age_status = 'FAIL'
        
        metric_performance.append({
            'metric': 'Company Age',
            'actual_value': age,
            'threshold_full_points': age_threshold_full,
            'threshold_min_points': age_threshold_min,
            'points_earned': round(age_points, 2),
            'points_possible': age_max_points,
            'percentage': round(age_percentage, 1),
            'status': age_status,
            'gap_to_full': max(0, age_threshold_full - age)
        })
        
        # Add all metrics to breakdown
        diagnostics['metric_breakdown'] = metric_performance
        
        # Calculate TOP NEGATIVE FACTORS (biggest points lost)
        negative_factors = []
        for perf in metric_performance:
            points_lost = perf['points_possible'] - perf['points_earned']
            if points_lost > 0.5:  # Only include if significant loss
                suggestion = self._get_improvement_suggestion(perf)
                negative_factors.append({
                    'metric': perf['metric'],
                    'points_lost': round(points_lost, 2),
                    'suggestion': suggestion
                })
        
        # Sort by points lost and take top 3
        negative_factors.sort(key=lambda x: x['points_lost'], reverse=True)
        diagnostics['top_negative_factors'] = negative_factors[:3]
        
        # Calculate TOP POSITIVE FACTORS (best performers)
        positive_factors = []
        for perf in metric_performance:
            if perf['percentage'] >= 60:  # Only include if doing reasonably well
                status_desc = self._get_status_description(perf)
                positive_factors.append({
                    'metric': perf['metric'],
                    'points_earned': perf['points_earned'],
                    'status': status_desc
                })
        
        # Sort by points earned and take top 3
        positive_factors.sort(key=lambda x: x['points_earned'], reverse=True)
        diagnostics['top_positive_factors'] = positive_factors[:3]
        
        # THRESHOLD FAILURES
        threshold_failures = []
        for perf in metric_performance:
            if perf['status'] == 'FAIL':
                threshold_failures.append({
                    'metric': perf['metric'],
                    'actual': perf['actual_value'],
                    'required_minimum': perf['threshold_min_points'],
                    'impact': f"0 points (vs max {perf['points_possible']})"
                })
        
        diagnostics['threshold_failures'] = threshold_failures
        
        # IMPROVEMENT SUGGESTIONS
        suggestions = []
        
        # Focus on top 2-3 biggest gaps
        for neg_factor in diagnostics['top_negative_factors'][:2]:
            for perf in metric_performance:
                if perf['metric'] == neg_factor['metric']:
                    suggestion = self._create_specific_suggestion(perf, final_score)
                    if suggestion:
                        suggestions.append(suggestion)
        
        # Add tier movement suggestion
        current_tier = self._get_tier_from_score(final_score)
        next_tier_score = self._get_next_tier_threshold(final_score)
        if next_tier_score:
            points_needed = next_tier_score - final_score
            suggestions.append(
                f"You need {points_needed:.1f} more points to move from {current_tier} to the next tier"
            )
        
        diagnostics['improvement_suggestions'] = suggestions
        
        return diagnostics
    
    def _get_improvement_suggestion(self, perf: Dict[str, Any]) -> str:
        """Generate improvement suggestion for a metric"""
        metric = perf['metric']
        actual = perf['actual_value']
        target = perf['threshold_full_points']
        
        if metric == 'Debt Service Coverage Ratio':
            return f"Improve DSCR from {actual:.2f} to {target:.2f} for full points"
        elif metric == 'Cash Flow Volatility':
            return f"Reduce volatility from {actual:.3f} to below {target:.2f}"
        elif metric == 'Directors Score':
            return f"Director score of {int(target)}+ would help (currently {int(actual)})"
        elif metric == 'Average Month-End Balance':
            return f"Increase balance from £{actual:,.0f} to £{target:,.0f}"
        elif metric == 'Revenue Growth Rate':
            return f"Improve growth from {actual*100:.1f}% to {target*100:.1f}%+"
        elif metric == 'Operating Margin':
            return f"Improve margin from {actual*100:.1f}% to above {target*100:.1f}%"
        elif metric == 'Negative Balance Days':
            return f"Reduce negative days from {int(actual)} to {int(target)} or fewer"
        elif metric == 'Company Age':
            return f"Company age will naturally improve ({int(actual)} months currently)"
        else:
            return f"Improve {metric} to meet threshold"
    
    def _get_status_description(self, perf: Dict[str, Any]) -> str:
        """Get status description for a positive factor"""
        percentage = perf['percentage']
        points = perf['points_earned']
        max_points = perf['points_possible']
        
        if percentage >= 95:
            return f"Full points - excellent performance ({points}/{max_points})"
        elif percentage >= 80:
            return f"{percentage:.0f}% - strong performance ({points:.1f}/{max_points})"
        else:
            return f"{percentage:.0f}% - good performance ({points:.1f}/{max_points})"
    
    def _create_specific_suggestion(self, perf: Dict[str, Any], current_score: float) -> str:
        """Create specific improvement suggestion with point impact"""
        metric = perf['metric']
        actual = perf['actual_value']
        gap = perf['gap_to_full']
        points_lost = perf['points_possible'] - perf['points_earned']
        
        if points_lost < 1:
            return None
        
        # Calculate achievable improvement (50% of gap)
        if metric == 'Cash Flow Volatility' or metric == 'Negative Balance Days':
            achievable_improvement = gap * 0.5
            achievable_value = actual - achievable_improvement  # Need to REDUCE these metrics
        else:
            achievable_improvement = gap * 0.5
            achievable_value = actual + achievable_improvement  # Need to INCREASE these metrics
        
        # Estimate point gain (roughly 50% of points lost)
        estimated_gain = points_lost * 0.5
        
        if metric == 'Debt Service Coverage Ratio':
            return f"Improving DSCR from {actual:.2f} to {achievable_value:.2f} would add ~{estimated_gain:.1f} points"
        elif metric == 'Cash Flow Volatility':
            return f"Reducing volatility from {actual:.3f} to {achievable_value:.3f} would add ~{estimated_gain:.1f} points"
        elif metric == 'Directors Score':
            return f"Improving Directors Score from {int(actual)} to {int(achievable_value)} would add ~{estimated_gain:.1f} points"
        elif metric == 'Average Month-End Balance':
            return f"Increasing balance from £{actual:,.0f} to £{achievable_value:,.0f} would add ~{estimated_gain:.1f} points"
        elif metric == 'Revenue Growth Rate':
            return f"Improving growth from {actual*100:.1f}% to {achievable_value*100:.1f}% would add ~{estimated_gain:.1f} points"
        elif metric == 'Operating Margin':
            return f"Improving margin from {actual*100:.1f}% to {achievable_value*100:.1f}% would add ~{estimated_gain:.1f} points"
        else:
            return None
    
    def _get_tier_from_score(self, score: float) -> str:
        """Get tier name from score"""
        if score >= 75:
            return "Tier 1"
        elif score >= 60:
            return "Tier 2"
        elif score >= 45:
            return "Tier 3"
        elif score >= 30:
            return "Tier 4"
        else:
            return "Decline"
    
    def _get_next_tier_threshold(self, score: float) -> float:
        """Get score threshold for next tier"""
        if score < 30:
            return 30  # Tier 4
        elif score < 45:
            return 45  # Tier 3
        elif score < 60:
            return 60  # Tier 2
        elif score < 75:
            return 75  # Tier 1
        else:
            return None  # Already at top tier

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