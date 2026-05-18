# app/services/ensemble_scorer.py
"""
Unified Ensemble Scoring System

Combines multiple scoring approaches into a single, coherent recommendation.
This eliminates confusion from divergent scores and provides a clear decision.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class Decision(Enum):
    """Lending decision outcomes."""
    APPROVE = "APPROVE"
    CONDITIONAL_APPROVE = "CONDITIONAL_APPROVE"
    REFER = "REFER"
    SENIOR_REVIEW = "SENIOR_REVIEW"
    DECLINE = "DECLINE"


@dataclass
class ScoringResult:
    """Container for individual scoring system results."""
    score: float
    available: bool = True
    weight: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsembleResult:
    """Final ensemble scoring result."""
    combined_score: float
    decision: Decision
    confidence: float
    primary_reason: str
    contributing_scores: Dict[str, float]
    score_convergence: str
    risk_factors: List[str]
    positive_factors: List[str]
    recommendations: List[str]
    pricing_guidance: Dict[str, Any]
    detailed_breakdown: Dict[str, Any]


class EnsembleScorer:
    """
    Combines multiple scoring systems into unified recommendations.
    
    Decision scoring systems:
    1. MCA Rule Score (60%) - Transaction consistency, empirically validated
    2. Subprime Score (40%) - Comprehensive micro-enterprise assessment

    ML Score is retained as an informational signal only. It is displayed for
    context, but it does not affect the combined score, convergence penalty, or
    final decision.
    
    The MCA Rule Score is based on:
    - inflow_days_30d: Days with deposits in last 30 days
    - max_inflow_gap_days: Largest gap between deposits
    - inflow_cv: Coefficient of variation of inflows
    
    These transaction consistency metrics have been shown to be
    strong predictors of MCA repayment probability.
    
    Hard stops can override the ensemble, but ordinary MCA weakness is treated
    as a decision cap rather than an automatic decline.
    """
    
    DEFAULT_WEIGHTS = {
        'mca_score': 0.60,        # Transaction consistency - empirically validated
        'subprime_score': 0.40,   # Micro-enterprise assessment
        'ml_score': 0.00,         # Informational only
    }
    
    # Decision thresholds
    THRESHOLDS = {
        'approve': 75,
        'conditional_approve': 70,
        'refer': 65,
        'senior_review': 60
    }

    # MCA consistency can support an approval, but it cannot override a weak
    # comprehensive business/director profile.
    SUBPRIME_DECISION_GATES = {
        'approve_min': 65,
        'refer_min': 60,
    }
    
    # Hard stop conditions that override ensemble.
    # MCA transaction weakness is only a hard stop when the underlying signal is
    # severe. Softer MCA declines cap the maximum decision instead of killing the
    # case outright.
    HARD_STOP_CONDITIONS = {
        'mca_decline': "Transaction consistency too poor for MCA lending (inflow pattern analysis)",
        'dscr_critical': "Debt service coverage ratio critically low (<0.5)",
        'directors_critical': "Directors score critically low (<20)",
        'multiple_ccjs': "Multiple CCJs present - high default risk"
    }
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize ensemble scorer with optional custom weights.
        
        Args:
            weights: Custom weights for scoring systems (must sum to 1.0)
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self._validate_weights()
    
    def _validate_weights(self) -> None:
        """Ensure weights sum to approximately 1.0."""
        total = sum(self.weights.values())
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")
    
    def calculate_ensemble_score(
        self,
        scores: Dict[str, Any],
        metrics: Dict[str, Any],
        params: Dict[str, Any]
    ) -> EnsembleResult:
        """
        Calculate unified ensemble score and recommendation.
        
        Args:
            scores: Dictionary containing individual scoring results
            metrics: Financial metrics dictionary
            params: Business parameters dictionary
            
        Returns:
            EnsembleResult with combined score, decision, and detailed analysis
        """
        # Check for hard stops first
        hard_stop = self._check_hard_stops(scores, metrics, params)
        if hard_stop:
            return hard_stop
        
        # Collect available scores
        scoring_results = self._collect_scores(scores)
        
        # Calculate weighted ensemble score
        combined_score, contributing_scores = self._calculate_weighted_score(scoring_results)
        
        # Analyze score convergence between decision-driving systems only.
        convergence, convergence_penalty = self._analyze_convergence(scoring_results)
        
        # Apply convergence penalty (divergent scores reduce confidence)
        adjusted_score = combined_score - convergence_penalty
        
        # Determine decision
        decision = self._determine_decision(adjusted_score, scores, params)
        
        # Calculate confidence
        confidence = self._calculate_confidence(scoring_results, convergence)
        
        # Gather risk and positive factors
        risk_factors = self._identify_risk_factors(metrics, params, scores)
        positive_factors = self._identify_positive_factors(metrics, params, scores)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            decision, adjusted_score, risk_factors, positive_factors, params
        )
        
        # Generate pricing guidance
        pricing_guidance = self._generate_pricing_guidance(
            adjusted_score, decision, params
        )
        
        # Primary reason for decision
        primary_reason = self._get_primary_reason(
            decision,
            adjusted_score,
            combined_score,
            convergence_penalty,
            convergence,
            contributing_scores,
            scores,
            risk_factors,
            positive_factors,
        )
        
        return EnsembleResult(
            combined_score=round(adjusted_score, 1),
            decision=decision,
            confidence=round(confidence, 1),
            primary_reason=primary_reason,
            contributing_scores=contributing_scores,
            score_convergence=convergence,
            risk_factors=risk_factors,
            positive_factors=positive_factors,
            recommendations=recommendations,
            pricing_guidance=pricing_guidance,
            detailed_breakdown={
                'raw_combined_score': round(combined_score, 1),
                'convergence_penalty': round(convergence_penalty, 1),
                'scoring_systems_used': len([
                    s for s in scoring_results.values()
                    if s.available and s.weight > 0
                ]),
                'weights_applied': self.weights,
                'mca_decision_policy': (
                    "MCA severe failures hard-stop; soft MCA decline caps the decision; "
                    "MCA refer caps at referral; MCA approve supports but does not approve alone. "
                    "Subprime gates cap approvals when the comprehensive profile is weak."
                ),
                'subprime_decision_gates': self.SUBPRIME_DECISION_GATES,
                'informational_scores': {
                    name: round(result.score, 1)
                    for name, result in scoring_results.items()
                    if result.available and result.weight <= 0
                },
            }
        )
    
    def _check_hard_stops(
        self,
        scores: Dict[str, Any],
        metrics: Dict[str, Any],
        params: Dict[str, Any]
    ) -> Optional[EnsembleResult]:
        """Check for conditions that should immediately decline."""
        
        hard_stop_reasons = []
        
        # Severe MCA transaction consistency failure. A plain MCA DECLINE should
        # not always be a hard stop; the decision layer handles softer MCA
        # weakness as a cap after the combined score is calculated.
        mca_decision = scores.get('mca_decision') or params.get('mca_rule_decision')
        if self._is_severe_mca_failure(scores, params):
            hard_stop_reasons.append(self.HARD_STOP_CONDITIONS['mca_decline'])
        
        # Critical DSCR
        dscr = metrics.get('Debt Service Coverage Ratio', 1.0)
        if dscr < 0.5:
            hard_stop_reasons.append(self.HARD_STOP_CONDITIONS['dscr_critical'])
        
        # Critical directors score
        directors = params.get('directors_score', 50)
        if directors < 20:
            hard_stop_reasons.append(self.HARD_STOP_CONDITIONS['directors_critical'])
        
        # Only treat repeated CCJs as a hard stop. A single CCJ is still a
        # material risk factor, but it should not automatically trump strong
        # scores from the other models.
        business_ccj_count = params.get('business_ccj_count')
        if business_ccj_count is None:
            business_ccj_count = 1 if params.get('business_ccj', False) else 0

        if params.get('multiple_ccjs', False) or business_ccj_count >= 2:
            hard_stop_reasons.append(self.HARD_STOP_CONDITIONS['multiple_ccjs'])
        
        if hard_stop_reasons:
            return EnsembleResult(
                combined_score=0,
                decision=Decision.DECLINE,
                confidence=95.0,
                primary_reason=hard_stop_reasons[0],
                contributing_scores={},
                score_convergence="N/A - Hard Stop",
                risk_factors=hard_stop_reasons,
                positive_factors=[],
                recommendations=[
                    "Application does not meet minimum lending criteria",
                    "Consider reapplying after addressing critical issues",
                    "Minimum 3-6 months improved trading recommended"
                ],
                pricing_guidance={'decision': 'DECLINE', 'reason': 'Hard stop triggered'},
                detailed_breakdown={'hard_stop_reasons': hard_stop_reasons}
            )
        
        return None

    def _is_severe_mca_failure(self, scores: Dict[str, Any], params: Dict[str, Any]) -> bool:
        """Return True only for MCA failures that should hard-stop a case."""
        mca_decision = scores.get('mca_decision') or params.get('mca_rule_decision')
        if not mca_decision or str(mca_decision).upper() != 'DECLINE':
            return False

        mca_score = scores.get('mca_score', params.get('mca_rule_score'))
        try:
            if mca_score is not None and float(mca_score) <= 20:
                return True
        except (TypeError, ValueError):
            pass

        signals = params.get('mca_rule_signals') or scores.get('mca_rule_signals') or {}
        try:
            inflow_days = signals.get('inflow_days_30d')
            if inflow_days is not None and float(inflow_days) <= 8:
                return True
        except (TypeError, ValueError):
            pass

        try:
            max_gap = signals.get('max_inflow_gap_days')
            if max_gap is not None and float(max_gap) >= 21:
                return True
        except (TypeError, ValueError):
            pass

        try:
            inflow_cv = signals.get('inflow_cv')
            if inflow_cv is not None and float(inflow_cv) >= 1.3:
                return True
        except (TypeError, ValueError):
            pass

        reasons = scores.get('mca_reasons') or scores.get('mca_rule_reasons') or params.get('mca_rule_reasons') or []
        reason_text = " | ".join(str(reason).lower() for reason in reasons)
        severe_markers = [
            "inflow_days_30d<=",
            "max_inflow_gap_days>=",
            "inflow_cv>=",
            "all three mca",
            "severe",
        ]
        return any(marker in reason_text for marker in severe_markers)
    
    def _collect_scores(self, scores: Dict[str, Any]) -> Dict[str, ScoringResult]:
        """Collect and normalize scores from different systems."""
        
        results = {}
        
        # MCA Rule Score (MOST IMPORTANT - transaction consistency)
        # Score is 0-100 from decide_application()
        mca_score = scores.get('mca_score')
        mca_decision = scores.get('mca_decision', 'REFER')
        results['mca_score'] = ScoringResult(
            score=float(mca_score) if mca_score is not None else 50,
            available=mca_score is not None,
            weight=self.weights.get('mca_score', 0),
            details={
                'decision': mca_decision,
                'description': 'Transaction consistency (inflow days, gaps, variability)'
            }
        )
        
        # Subprime score (already 0-100)
        subprime = scores.get('subprime_score')
        results['subprime_score'] = ScoringResult(
            score=float(subprime) if subprime is not None else 0,
            available=subprime is not None,
            weight=self.weights.get('subprime_score', 0),
            details=scores.get('subprime_details', {})
        )
        
        # ML score (already 0-100 percentage). Informational only by default.
        ml_score = scores.get('ml_score')
        if ml_score is None:
            ml_score = scores.get('adjusted_ml_score')
        results['ml_score'] = ScoringResult(
            score=float(ml_score) if ml_score is not None else 0,
            available=ml_score is not None,
            weight=self.weights.get('ml_score', 0),
            details=scores.get('ml_details', {})
        )
        
        return results
    
    def _calculate_weighted_score(
        self,
        scoring_results: Dict[str, ScoringResult]
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate weighted average of available scores."""
        
        total_weight = 0
        weighted_sum = 0
        contributing_scores = {}
        
        for name, result in scoring_results.items():
            if result.available and result.weight > 0:
                weighted_sum += result.score * result.weight
                total_weight += result.weight
                contributing_scores[name] = round(result.score, 1)
        
        # Normalize if not all scores available
        if total_weight > 0 and total_weight < 1.0:
            combined_score = weighted_sum / total_weight
        elif total_weight > 0:
            combined_score = weighted_sum
        else:
            combined_score = 50  # Default neutral score
        
        return combined_score, contributing_scores
    
    def _analyze_convergence(
        self,
        scoring_results: Dict[str, ScoringResult]
    ) -> Tuple[str, float]:
        """
        Analyze how well all scoring systems converge.
        High divergence suggests uncertainty and warrants a penalty.
        """
        primary_scores = [
            r.score for r in scoring_results.values()
            if r.available and r.weight > 0
        ]
        
        if len(primary_scores) < 2:
            return "Insufficient Data", 5.0
        
        score_range = max(primary_scores) - min(primary_scores)
        
        if score_range <= 10:
            return "High Convergence", 0.0
        elif score_range <= 20:
            return "Good Convergence", 1.0
        elif score_range <= 30:
            return "Moderate Convergence", 3.0
        else:
            return "Low Convergence", 5.0
    
    def _determine_decision(
        self,
        score: float,
        scores: Dict[str, Any],
        params: Dict[str, Any]
    ) -> Decision:
        """Determine final decision based on score and context."""
        
        # Get MCA rule decision (important - based on transaction consistency)
        mca_decision = scores.get('mca_decision') or params.get('mca_rule_decision')
        mca_decision = str(mca_decision).upper() if mca_decision else None

        subprime_score = scores.get('subprime_score')
        try:
            subprime_score = float(subprime_score)
        except (TypeError, ValueError):
            subprime_score = None

        if subprime_score is not None:
            if subprime_score < self.SUBPRIME_DECISION_GATES['refer_min']:
                return Decision.DECLINE

            if subprime_score < self.SUBPRIME_DECISION_GATES['approve_min']:
                if score >= self.THRESHOLDS['senior_review']:
                    return Decision.REFER
                return Decision.DECLINE
        
        # MCA DECLINE is a hard stop only when _check_hard_stops has identified
        # severe transaction weakness. Softer MCA declines cap the maximum
        # decision at referral while still allowing nuance from the combined
        # score.
        if mca_decision == 'DECLINE':
            if score >= self.THRESHOLDS['senior_review']:
                return Decision.REFER
            else:
                return Decision.DECLINE

        # MCA REFER means transaction consistency is borderline. It caps the
        # maximum decision at referral but does not auto-decline.
        if mca_decision == 'REFER':
            if score >= self.THRESHOLDS['senior_review']:
                return Decision.REFER
            else:
                return Decision.DECLINE
        
        # MCA APPROVE supports the combined score, but it does not approve by
        # itself. The weighted score must still meet the relevant band.
        if mca_decision == 'APPROVE':
            if score >= self.THRESHOLDS['approve']:
                return Decision.APPROVE
            elif score >= self.THRESHOLDS['senior_review']:
                return Decision.REFER
            else:
                return Decision.DECLINE
        
        # Standard threshold-based decision (no MCA decision or unknown)
        if score >= self.THRESHOLDS['approve']:
            return Decision.APPROVE
        elif score >= self.THRESHOLDS['senior_review']:
            return Decision.REFER
        else:
            return Decision.DECLINE
    
    def _calculate_confidence(
        self,
        scoring_results: Dict[str, ScoringResult],
        convergence: str
    ) -> float:
        """Calculate confidence in the ensemble decision."""
        
        # Base confidence from number of available decision scores
        available_count = sum(1 for r in scoring_results.values() if r.available and r.weight > 0)
        base_confidence = min(90, 60 + available_count * 10)
        
        # Adjust for convergence
        convergence_adjustments = {
            "High Convergence": 10,
            "Good Convergence": 5,
            "Moderate Convergence": 0,
            "Low Convergence": -10,
            "Insufficient Data": -20
        }
        
        confidence = base_confidence + convergence_adjustments.get(convergence, 0)
        
        return max(20, min(95, confidence))
    
    def _identify_risk_factors(
        self,
        metrics: Dict[str, Any],
        params: Dict[str, Any],
        scores: Dict[str, Any]
    ) -> List[str]:
        """Identify key risk factors from metrics and params."""
        
        risk_factors = []
        
        # DSCR risks
        dscr = metrics.get('Debt Service Coverage Ratio', 1.0)
        if dscr < 1.0:
            risk_factors.append(f"Low DSCR ({dscr:.2f}) - debt servicing strain")
        
        # Cash flow volatility
        volatility = metrics.get('Cash Flow Volatility', 0)
        if volatility > 0.6:
            risk_factors.append(f"High cash flow volatility ({volatility:.2f})")
        
        # Negative balance days
        neg_days = metrics.get('Average Negative Balance Days per Month', 0)
        if neg_days > 5:
            risk_factors.append(f"Frequent negative balances ({neg_days:.0f} days/month)")
        
        # Directors score
        directors = params.get('directors_score', 50)
        if directors < 40:
            risk_factors.append(f"Low directors score ({directors})")
        
        # CCJs
        if params.get('business_ccj', False):
            risk_factors.append("Business CCJ on record")

        bureau_band = params.get('bureau_band') or params.get('business_bureau_band')
        if bureau_band and str(bureau_band).startswith(("C", "D")):
            risk_factors.append(f"Business bureau band: {bureau_band}")

        if params.get('business_credit_score_suppressed', False):
            risk_factors.append("Business bureau score suppressed")

        credit_limit = params.get('business_credit_limit')
        max_credit = params.get('business_max_recommended_credit')
        if credit_limit == 0 and max_credit == 0:
            risk_factors.append("Business bureau credit limit £0")

        negative_impact_count = params.get('business_negative_impact_count') or 0
        try:
            negative_impact_count = int(negative_impact_count)
        except (TypeError, ValueError):
            negative_impact_count = 0
        if negative_impact_count >= 3:
            risk_factors.append(f"Business bureau negative factors ({negative_impact_count})")

        # Company age
        age = params.get('company_age_months', 12)
        if age < 6:
            risk_factors.append(f"Young business ({age} months)")
        
        # Operating margin
        margin = metrics.get('Operating Margin', 0)
        if margin < 0:
            risk_factors.append(f"Negative operating margin ({margin*100:.1f}%)")
        
        # Bounced payments
        bounced = metrics.get('Number of Bounced Payments', 0)
        if bounced > 2:
            risk_factors.append(f"Multiple bounced payments ({bounced})")

        funding_reliance = metrics.get('Funding Reliance Ratio', 0)
        if funding_reliance >= 0.25:
            risk_factors.append(f"Material owner/funding inflows ({funding_reliance*100:.0f}% of core inflows)")

        transfer_ratio = metrics.get('Internal Transfer Activity Ratio', 0)
        if transfer_ratio >= 0.25:
            risk_factors.append(f"High internal transfer activity ({transfer_ratio*100:.0f}% of cash movement)")

        concentration_risk = metrics.get('Revenue Concentration Risk')
        top_source = metrics.get('Top Revenue Source Percentage', 0)
        if concentration_risk == 'High' or top_source >= 60:
            risk_factors.append(f"Revenue concentration risk (top source {top_source:.0f}%)")

        active_lenders = metrics.get('Active Lenders Detected', 0)
        if active_lenders >= 2:
            risk_factors.append(f"Multiple active lenders detected ({active_lenders})")

        bank_charge_count = metrics.get('Bank Charge Count', 0)
        if bank_charge_count >= 3:
            risk_factors.append(f"Frequent bank charges ({bank_charge_count})")

        recent_nsf = metrics.get('NSF Count 90D', 0)
        if recent_nsf > 0:
            risk_factors.append(f"Recent unpaid/NSF activity ({recent_nsf} in 90 days)")
        
        return risk_factors[:5]  # Top 5 risk factors
    
    def _identify_positive_factors(
        self,
        metrics: Dict[str, Any],
        params: Dict[str, Any],
        scores: Dict[str, Any]
    ) -> List[str]:
        """Identify key positive factors."""
        
        positive_factors = []
        
        # Strong DSCR
        dscr = metrics.get('Debt Service Coverage Ratio', 1.0)
        if dscr >= 1.5:
            positive_factors.append(f"Strong DSCR ({dscr:.2f})")
        
        # Good growth
        growth = metrics.get('Revenue Growth Rate', 0)
        if growth > 0.05:
            positive_factors.append(f"Positive revenue growth ({growth*100:.1f}%)")
        
        # Strong directors score
        directors = params.get('directors_score', 50)
        if directors >= 70:
            positive_factors.append(f"Strong directors score ({directors})")
        
        # Good balance
        balance = metrics.get('Average Month-End Balance', 0)
        if balance >= 2000:
            positive_factors.append(f"Healthy average balance (£{balance:,.0f})")
        
        # Low volatility
        volatility = metrics.get('Cash Flow Volatility', 1.0)
        if volatility <= 0.3:
            positive_factors.append(f"Stable cash flow (volatility: {volatility:.2f})")
        
        # Established business
        age = params.get('company_age_months', 0)
        if age >= 24:
            positive_factors.append(f"Established business ({age} months)")
        
        # No bounced payments
        bounced = metrics.get('Number of Bounced Payments', 0)
        if bounced == 0:
            positive_factors.append("No bounced payments")

        funding_reliance = metrics.get('Funding Reliance Ratio', 0)
        if funding_reliance <= 0.1:
            positive_factors.append("Low reliance on owner/funding inflows")

        concentration_risk = metrics.get('Revenue Concentration Risk')
        if concentration_risk == 'Low':
            positive_factors.append("Diversified revenue base")

        regularity = metrics.get('Revenue Regularity Score', 0)
        if regularity >= 70:
            positive_factors.append(f"Consistent revenue cadence ({regularity:.0f}/100)")

        recent_nsf = metrics.get('NSF Count 90D', 0)
        if recent_nsf == 0:
            positive_factors.append("No recent NSF activity")
        
        return positive_factors[:5]  # Top 5 positive factors
    
    def _generate_recommendations(
        self,
        decision: Decision,
        score: float,
        risk_factors: List[str],
        positive_factors: List[str],
        params: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations."""
        
        recommendations = []
        
        if decision == Decision.APPROVE:
            recommendations.extend([
                "Proceed with standard loan terms",
                "Monthly monitoring recommended",
                "Standard documentation package"
            ])
        
        elif decision == Decision.REFER:
            recommendations.extend([
                "Manual underwriter review required",
                "Request 3 months additional bank statements",
                "Verify revenue sources"
            ])
            if len(risk_factors) > 3:
                recommendations.append("Consider reduced loan amount")
        
        else:  # DECLINE
            recommendations.extend([
                "Application does not meet current criteria",
                "Suggest reapplying after 3-6 months improved trading",
                "Address identified risk factors before reapplication"
            ])
        
        return recommendations
    
    def _generate_pricing_guidance(
        self,
        score: float,
        decision: Decision,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate pricing guidance based on risk assessment."""
        
        if decision == Decision.DECLINE:
            return {
                'factor_rate': 'N/A',
                'max_term': 'N/A',
                'max_multiple': 'N/A',
                'collection_frequency': 'N/A'
            }
        
        has_ccj = params.get('business_ccj', False)
        
        if score >= 70:
            base_rate = "1.15-1.25"
            max_term = "12 months"
            max_multiple = "4x monthly revenue"
            frequency = "Monthly"
        elif score >= 55:
            base_rate = "1.25-1.35"
            max_term = "9 months"
            max_multiple = "3x monthly revenue"
            frequency = "Bi-weekly"
        elif score >= 45:
            base_rate = "1.35-1.45"
            max_term = "6 months"
            max_multiple = "2.5x monthly revenue"
            frequency = "Weekly"
        else:
            base_rate = "1.45-1.55"
            max_term = "4 months"
            max_multiple = "2x monthly revenue"
            frequency = "Daily"
        
        # CCJ adjustment
        if has_ccj:
            base_rate = base_rate.replace("-", "+0.05 to ")
        
        return {
            'factor_rate': base_rate,
            'max_term': max_term,
            'max_multiple': max_multiple,
            'collection_frequency': frequency,
            'ccj_adjustment_applied': has_ccj
        }
    
    def _get_primary_reason(
        self,
        decision: Decision,
        score: float,
        raw_score: float,
        convergence_penalty: float,
        convergence: str,
        contributing_scores: Dict[str, float],
        scores: Dict[str, Any],
        risk_factors: List[str],
        positive_factors: List[str]
    ) -> str:
        """Generate a decision-mechanics-first primary reason."""
        threshold_labels = {
            Decision.APPROVE: ("approval", self.THRESHOLDS['approve']),
            Decision.REFER: ("referral", self.THRESHOLDS['refer']),
            Decision.DECLINE: ("referral", self.THRESHOLDS['senior_review']),
        }

        mca = contributing_scores.get('mca_score')
        subprime = contributing_scores.get('subprime_score')
        score_context = f"MCA {mca:.0f} / Subprime {subprime:.0f}" if mca is not None and subprime is not None else "available decision scores"
        mca_decision = str(scores.get('mca_decision') or "").upper().strip()

        penalty_text = ""
        if convergence_penalty > 0:
            penalty_text = (
                f" Raw weighted score was {raw_score:.1f}, reduced by {convergence_penalty:.1f} "
                f"for {convergence.lower()} between MCA and Subprime."
            )

        if mca_decision == "DECLINE" and decision == Decision.REFER:
            return (
                f"Weighted MCA/Subprime score {score:.1f} was not an automatic decline, "
                f"but MCA transaction weakness caps the case at {decision.value.replace('_', ' ').title()} "
                f"({score_context}).{penalty_text}"
            )

        if mca_decision == "REFER" and decision == Decision.REFER:
            return (
                f"Weighted MCA/Subprime score {score:.1f} supports approval, but borderline MCA "
                f"transaction consistency caps the decision at referral ({score_context}).{penalty_text}"
            )

        if subprime is not None:
            if subprime < self.SUBPRIME_DECISION_GATES['refer_min']:
                return (
                    f"Subprime score {subprime:.1f} is below the minimum referral gate, so the case is declined "
                    f"even though the weighted score is {score:.1f} ({score_context}).{penalty_text}"
                )
            if subprime < self.SUBPRIME_DECISION_GATES['approve_min'] and decision == Decision.REFER:
                return (
                    f"Subprime score {subprime:.1f} caps the case at referral; MCA score alone is not enough "
                    f"to approve ({score_context}).{penalty_text}"
                )
        
        if decision == Decision.APPROVE:
            return f"Weighted MCA/Subprime score {score:.1f} meets approval threshold ({score_context})."
        
        elif decision == Decision.REFER:
            return f"Weighted MCA/Subprime score {score:.1f} falls in the referral band ({score_context}).{penalty_text}"
        
        else:  # DECLINE
            label, threshold = threshold_labels[Decision.DECLINE]
            return f"Weighted MCA/Subprime score {score:.1f} is below the {label} threshold of {threshold} ({score_context}).{penalty_text}"


def get_ensemble_recommendation(
    scores: Dict[str, Any],
    metrics: Dict[str, Any],
    params: Dict[str, Any],
    custom_weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Convenience function to get ensemble recommendation.
    
    Args:
        scores: Dictionary containing individual scoring results
        metrics: Financial metrics dictionary
        params: Business parameters dictionary
        custom_weights: Optional custom weights for scoring systems
        
    Returns:
        Dictionary with ensemble result
    """
    scorer = EnsembleScorer(weights=custom_weights)
    result = scorer.calculate_ensemble_score(scores, metrics, params)
    
    return {
        'combined_score': result.combined_score,
        'decision': result.decision.value,
        'confidence': result.confidence,
        'primary_reason': result.primary_reason,
        'contributing_scores': result.contributing_scores,
        'score_convergence': result.score_convergence,
        'risk_factors': result.risk_factors,
        'positive_factors': result.positive_factors,
        'recommendations': result.recommendations,
        'pricing_guidance': result.pricing_guidance,
        'detailed_breakdown': result.detailed_breakdown
    }
