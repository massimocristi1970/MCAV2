# Enhanced subprime_scoring_system.py with risk factor penalties
"""
Subprime Business Finance Scoring System - MICRO ENTERPRISE FRIENDLY VERSION
Now includes risk factor penalties that match the V2 weighted system
Balanced for micro enterprises (Â£1-10k short-term lending)

Uses centralized thresholds from app/config/scoring_thresholds.py
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from datetime import datetime

# Import centralized thresholds
try:
    from ..config.scoring_thresholds import get_thresholds, ScoringThresholds
    THRESHOLDS_AVAILABLE = True
except ImportError:
    THRESHOLDS_AVAILABLE = False
    get_thresholds = None


class SubprimeScoring:
    """Scoring system specifically designed for micro enterprise subprime lending with balanced risk assessment."""

    def __init__(self):
        # Load centralized thresholds if available
        self._thresholds = get_thresholds() if THRESHOLDS_AVAILABLE else None
        
        # Subprime weights for Â£1-10k short-term lending (6-9 months)
        # Balanced for micro enterprise realities
        # These are derived from centralized config when available
        if self._thresholds:
            self.subprime_weights = {
                'Debt Service Coverage Ratio': self._thresholds.DSCR.weight,
                'Average Month-End Balance': self._thresholds.BALANCE.weight,
                'Directors Score': self._thresholds.DIRECTORS.weight,
                'Cash Flow Volatility': self._thresholds.VOLATILITY.weight,
                'Revenue Growth Rate': self._thresholds.GROWTH.weight,
                'Operating Margin': self._thresholds.MARGIN.weight,
                'Net Income': self._thresholds.NET_INCOME.weight,
                'Average Negative Balance Days per Month': self._thresholds.NEGATIVE_DAYS.weight,
                'Company Age (Months)': self._thresholds.COMPANY_AGE.weight,
                'Number of Bounced Payments': self._thresholds.BOUNCED.weight,
            }
            self.risk_factor_penalties = self._thresholds.RISK_PENALTIES
            self.industry_multipliers = self._thresholds.INDUSTRY_MULTIPLIERS
            self._max_penalty_cap = self._thresholds.MAX_PENALTY_CAP
        else:
            # Fallback to hardcoded values if config not available
            self.subprime_weights = {
                'Debt Service Coverage Ratio': 25,
                'Average Month-End Balance': 18,
                'Directors Score': 16,
                'Cash Flow Volatility': 14,
                'Revenue Growth Rate': 10,
                'Operating Margin': 6,
                'Net Income': 4,
                'Average Negative Balance Days per Month': 5,
                'Company Age (Months)': 2,
                'Number of Bounced Payments': 3,
            }
            self.risk_factor_penalties = {
                "business_ccj": 6,
                "director_ccj": 4
            }
            self.industry_multipliers = {
                'Medical Practices (GPs, Clinics, Dentists)': 1.05,
                'IT Services and Support Companies': 1.05,
                'Pharmacies (Independent or Small Chains)': 1.03,
                'Business Consultants': 1.03,
                'Education': 1.02,
                'Engineering': 1.02,
                'Telecommunications': 1.01,
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
                'Restaurants and Cafes': 0.95,
                'Construction Firms': 0.95,
                'Beauty Salons and Spas': 0.95,
                'Bars and Pubs': 0.93,
                'Event Planning and Management Firms': 0.92,
            }
            self._max_penalty_cap = 12

        # MICRO ENTERPRISE FRIENDLY risk tolerance thresholds (reference targets for UI copy)
        self.subprime_thresholds = {
            'minimum_dscr': 1.0,
            'maximum_volatility': 0.95,
            'minimum_growth': -0.07,
            'minimum_balance': 400,
            'maximum_negative_days': 10,
        }

    def _score_metric_points(
        self,
        metric_name: str,
        value: float,
        metrics: Dict[str, Any] | None = None,
    ) -> Tuple[float, float, str]:
        """Score one metric using centralized thresholds."""
        if self._thresholds:
            points, percentage, status = self._thresholds.score_metric(metric_name, value)
        else:
            return 0.0, 0.0, "UNKNOWN"

        if (
            metric_name == "Average Month-End Balance"
            and (metrics or {}).get("Balance Source") == "estimated"
        ):
            factor = getattr(self._thresholds, "ESTIMATED_BALANCE_SCORE_FACTOR", 0.85)
            points *= factor
            percentage *= factor

        return points, percentage, status

    def _calculate_mca_flow_bonus(self, metrics: Dict[str, Any], params: Dict[str, Any]) -> float:
        """Small bounded bonus when MCA collection-pattern signals are strong."""
        if not self._thresholds:
            return 0.0

        signals = params.get("mca_rule_signals") or {}
        inflow_days = signals.get("inflow_days_30d")
        active_rate = metrics.get("OB Revenue Active Day Rate")

        bonus = 0.0
        try:
            if inflow_days is not None and float(inflow_days) >= 18:
                bonus = 3.0
            elif inflow_days is not None and float(inflow_days) >= 14:
                bonus = 2.0
            elif inflow_days is not None and float(inflow_days) >= 10:
                bonus = 1.0
        except (TypeError, ValueError):
            pass

        try:
            if active_rate is not None and float(active_rate) >= 0.55 and bonus < 2.0:
                bonus = max(bonus, 2.0)
            elif active_rate is not None and float(active_rate) >= 0.45 and bonus < 1.0:
                bonus = max(bonus, 1.0)
        except (TypeError, ValueError):
            pass

        return min(bonus, float(getattr(self._thresholds, "MCA_FLOW_BONUS_MAX", 5.0)))

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
        mca_flow_bonus = self._calculate_mca_flow_bonus(metrics, params)

        # Final score calculation - includes risk factor penalties and card processing overlay
        card_processing_adjustment = self._calculate_card_processing_adjustment(metrics, params)
        pre_penalty_score = industry_adjusted_score + growth_bonus + mca_flow_bonus - stability_penalty
        final_score = max(0, min(100, pre_penalty_score - risk_factor_penalty + card_processing_adjustment))

        # Determine risk tier and pricing
        risk_tier, pricing_guidance = self._determine_risk_tier(final_score, metrics, params)

        # Generate detailed breakdown
        breakdown = self._generate_scoring_breakdown(
            base_score, industry_adjusted_score, growth_bonus,
            stability_penalty, risk_factor_penalty, card_processing_adjustment, mca_flow_bonus, final_score, metrics, params
        )
        
        # Generate comprehensive diagnostics
        diagnostics = self._generate_score_diagnostics(metrics, params, final_score)

        return {
            'subprime_score': round(final_score, 1),
            'risk_tier': risk_tier,
            'pricing_guidance': pricing_guidance,
            'card_processing_score_adjustment': round(card_processing_adjustment, 1),
            'breakdown': breakdown,
            'recommendation': self._generate_recommendation(risk_tier, metrics, params),
            'diagnostics': diagnostics
        }

    def _calculate_base_subprime_score(self, metrics: Dict[str, Any], params: Dict[str, Any]) -> float:
        """Calculate base score using centralized metric thresholds."""
        score = 0.0
        max_possible = sum(self.subprime_weights.values())

        metric_values = {
            "Debt Service Coverage Ratio": float(metrics.get("Debt Service Coverage Ratio") or 0),
            "Revenue Growth Rate": float(metrics.get("Revenue Growth Rate") or 0),
            "Average Month-End Balance": float(metrics.get("Average Month-End Balance") or 0),
            "Cash Flow Volatility": float(metrics.get("Cash Flow Volatility") or 0),
            "Operating Margin": float(metrics.get("Operating Margin") or 0),
            "Net Income": float(metrics.get("Net Income") or 0),
            "Average Negative Balance Days per Month": float(
                metrics.get("Average Negative Balance Days per Month") or 0
            ),
            "Number of Bounced Payments": float(metrics.get("Number of Bounced Payments") or 0),
            "Directors Score": float(params.get("directors_score") or 0),
            "Company Age (Months)": float(params.get("company_age_months") or 0),
        }

        for metric_name, value in metric_values.items():
            if metric_name not in self.subprime_weights:
                continue
            points, _, _ = self._score_metric_points(metric_name, value, metrics)
            score += points

        return (score / max_possible) * 100 if max_possible else 0.0

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
            penalty += (neg_days - 10) * 1.5

        # Only penalize very low DSCR (below 0.8)
        if dscr < 0.8:
            penalty += (0.8 - dscr) * 8

        return min(penalty, 15)  # Cap at 15 points max

    def _calculate_card_processing_adjustment(self, metrics: Dict[str, Any], params: Dict[str, Any]) -> float:
        """Use uploaded card statements as a bounded, auditable score overlay."""
        if metrics.get("Card Processing Insight Layer") != "Available":
            return 0.0
        try:
            adjustment = float(metrics.get("Card Processing Score Adjustment", 0) or 0)
        except (TypeError, ValueError):
            adjustment = 0.0
        adjustment = max(-8.0, min(5.0, adjustment))
        if adjustment:
            params["_card_processing_score_adjustment"] = round(adjustment, 1)
            params["_card_processing_adjustment_reasons"] = metrics.get("Card Processing Score Adjustment Reasons") or []
        return adjustment

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
            penalty = self.risk_factor_penalties.get('director_ccj', 4)
            total_penalty += penalty
            applied_penalties.append(f"Director CCJ: -{penalty}")

        if params.get('personal_default_12m', False):
            penalty = self.risk_factor_penalties.get('personal_default_12m', 5)
            total_penalty += penalty
            applied_penalties.append(f"Personal default (12m): -{penalty}")

        if params.get('business_credit_score_suppressed', False):
            total_penalty += 2
            applied_penalties.append("Business bureau score suppressed: -2")

        credit_limit = params.get('business_credit_limit')
        max_credit = params.get('business_max_recommended_credit')
        if credit_limit == 0 and max_credit == 0:
            total_penalty += 2
            applied_penalties.append("Business bureau credit limit Â£0: -2")

        negative_impact_count = params.get('business_negative_impact_count') or 0
        try:
            negative_impact_count = int(negative_impact_count)
        except (TypeError, ValueError):
            negative_impact_count = 0
        if negative_impact_count >= 4:
            total_penalty += 2
            applied_penalties.append(f"Business bureau negative factors ({negative_impact_count}): -2")
        elif negative_impact_count >= 2:
            total_penalty += 1
            applied_penalties.append(f"Business bureau negative factors ({negative_impact_count}): -1")

        enquiries_3m = params.get('business_enquiries_3m') or 0
        try:
            enquiries_3m = int(enquiries_3m)
        except (TypeError, ValueError):
            enquiries_3m = 0
        if enquiries_3m >= 3:
            total_penalty += 1
            applied_penalties.append(f"Recent business bureau enquiries ({enquiries_3m}): -1")

        # Cap maximum penalty to prevent "death by 1000 cuts"
        if total_penalty > self._max_penalty_cap:
            applied_penalties.append(f"Penalty capped:  -{total_penalty} reduced to -{self._max_penalty_cap}")
            total_penalty = self._max_penalty_cap

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
            params.get("business_ccj", False)
            or params.get("director_ccj", False)
            or params.get("personal_default_12m", False)
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
                                    stability_penalty, risk_factor_penalty, card_processing_adjustment,
                                    mca_flow_bonus, final_score, metrics, params) -> List[str]:
        """Generate detailed scoring breakdown including risk factor penalties."""

        breakdown = [
            f"Base Score: {base_score:.1f}/100",
            f"Industry Adjustment: {industry_score - base_score:+.1f} points",
            f"Growth Momentum Bonus: +{growth_bonus:.1f} points",
            f"MCA Flow Bonus: +{mca_flow_bonus:.1f} points",
            f"Stability Penalty: -{stability_penalty:.1f} points",
            f"Risk Factor Penalties: -{risk_factor_penalty:.1f} points",
            f"Card Processing Overlay: {card_processing_adjustment:+.1f} points",
            f"Final Score: {final_score:.1f}/100",
            "",
            "Key Metrics (Micro Enterprise Thresholds):",
            f"â€¢ DSCR: {metrics.get('Debt Service Coverage Ratio', 0):.2f} (Need 1.2+ for good score)",
            f"â€¢ Revenue Growth: {metrics.get('Revenue Growth Rate', 0) * 100:.1f}% (Need 5%+ for good score)",
            f"â€¢ Directors Score: {params.get('directors_score', 0)}/100 (Need 55+ for good score)",
            f"â€¢ Cash Flow Volatility: {metrics.get('Cash Flow Volatility', 0):.3f} (Need <0.50 for good score)",
            f"â€¢ Operating Margin: {metrics.get('Operating Margin', 0) * 100:.1f}% (Need 3%+ for good score)",
            f"â€¢ Negative Balance Days: {metrics.get('Average Negative Balance Days per Month', 0):.0f} (Need <5 for good score)"
        ]

        # Add risk factor details if any were applied
        applied_penalties = params.get('_applied_risk_penalties', [])
        if applied_penalties:
            breakdown.append("")
            breakdown.append("Risk Factor Penalties Applied:")
            for penalty in applied_penalties:
                breakdown.append(f"â€¢ {penalty}")

        card_reasons = metrics.get("Card Processing Score Adjustment Reasons") or []
        if card_reasons:
            breakdown.append("")
            breakdown.append("Card Processing Overlay Applied:")
            for reason in card_reasons:
                breakdown.append(f"- {reason}")

        return breakdown

    def _generate_recommendation(self, risk_tier: str, metrics: Dict[str, Any], params: Dict[str, Any]) -> str:
        """Generate lending recommendation based on risk tier and risk factors."""

        has_major_risk_factors = (
            params.get("business_ccj", False)
            or params.get("director_ccj", False)
            or params.get("personal_default_12m", False)
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
        
        diagnostic_specs = [
            ('Debt Service Coverage Ratio', float(metrics.get('Debt Service Coverage Ratio') or 0)),
            ('Revenue Growth Rate', float(metrics.get('Revenue Growth Rate') or 0)),
            ('Directors Score', float(params.get('directors_score') or 0)),
            ('Average Month-End Balance', float(metrics.get('Average Month-End Balance') or 0)),
            ('Cash Flow Volatility', float(metrics.get('Cash Flow Volatility') or 0)),
            ('Operating Margin', float(metrics.get('Operating Margin') or 0)),
            ('Net Income', float(metrics.get('Net Income') or 0)),
            ('Average Negative Balance Days per Month', float(metrics.get('Average Negative Balance Days per Month') or 0)),
            ('Number of Bounced Payments', float(metrics.get('Number of Bounced Payments') or 0)),
            ('Company Age (Months)', float(params.get('company_age_months') or 0)),
        ]

        for metric_name, value in diagnostic_specs:
            points, percentage, status = self._score_metric_points(metric_name, value, metrics)
            max_points = self.subprime_weights.get(metric_name, 0)
            threshold = self._thresholds.get_metric_threshold(metric_name) if self._thresholds else None
            full_points = threshold.full_points if threshold else 0
            min_points = threshold.tiers[-1][0] if threshold and threshold.tiers else 0
            if threshold and threshold.lower_is_better:
                gap_to_full = max(0.0, value - full_points)
            else:
                gap_to_full = max(0.0, full_points - value)
            metric_performance.append({
                'metric': metric_name,
                'actual_value': value,
                'threshold_full_points': full_points,
                'threshold_min_points': min_points,
                'points_earned': round(points, 2),
                'points_possible': max_points,
                'percentage': round(percentage, 1),
                'status': status,
                'gap_to_full': round(gap_to_full, 4),
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
            return f"Increase balance from Â£{actual:,.0f} to Â£{target:,.0f}"
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
            return f"Increasing balance from Â£{actual:,.0f} to Â£{achievable_value:,.0f} would add ~{estimated_gain:.1f} points"
        elif metric == 'Revenue Growth Rate':
            return f"Improving growth from {actual*100:.1f}% to {achievable_value*100:.1f}% would add ~{estimated_gain:.1f} points"
        elif metric == 'Operating Margin':
            return f"Improving margin from {actual*100:.1f}% to {achievable_value*100:.1f}% would add ~{estimated_gain:.1f} points"
        else:
            return None
    
    def _get_tier_from_score(self, score: float) -> str:
        """Get tier name from score using centralized thresholds."""
        if self._thresholds:
            return self._thresholds.get_tier_from_score(score)
        
        # Fallback thresholds (aligned with _determine_risk_tier)
        if score >= 65:
            return "Tier 1"
        elif score >= 50:
            return "Tier 2"
        elif score >= 40:
            return "Tier 3"
        elif score >= 30:
            return "Tier 4"
        else:
            return "Decline"
    
    def _get_next_tier_threshold(self, score: float) -> float:
        """Get score threshold for next tier using centralized thresholds."""
        if self._thresholds:
            tier, points_needed = self._thresholds.get_next_tier_threshold(score)
            return score + points_needed if tier else None
        
        # Fallback thresholds (aligned with _determine_risk_tier)
        if score < 30:
            return 30  # Tier 4
        elif score < 40:
            return 40  # Tier 3
        elif score < 50:
            return 50  # Tier 2
        elif score < 65:
            return 65  # Tier 1
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
                'uses_generic_email': True,
                'poor_or_no_online_presence': True
            }
        }
    ]

    scorer = SubprimeScoring()

    print("=== MICRO ENTERPRISE FRIENDLY SUBPRIME SCORING ===\n")

    for scenario in test_scenarios:
        print(f"ðŸ§ª TEST SCENARIO: {scenario['name']}")
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
# ADD THIS LINE AT THE VERY END OF THE FILE:
SubprimeScoringSystem = SubprimeScoring

