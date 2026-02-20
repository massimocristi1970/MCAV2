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
    
    Scoring systems considered (weights based on predictive power):
    1. MCA Rule Score (40%) - Transaction consistency, empirically validated
    2. Subprime Score (40%) - Comprehensive micro-enterprise assessment
    3. ML Score (20%) - Data-driven probability (retrained model, 0.922 ROC-AUC)
    
    The MCA Rule Score is based on:
    - inflow_days_30d: Days with deposits in last 30 days
    - max_inflow_gap_days: Largest gap between deposits
    - inflow_cv: Coefficient of variation of inflows
    
    These transaction consistency metrics have been shown to be
    strong predictors of MCA repayment probability.
    
    Hard stops can override the ensemble (e.g., MCA DECLINE).
    """
    
    DEFAULT_WEIGHTS = {
        'mca_score': 0.40,        # Transaction consistency - empirically validated
        'subprime_score': 0.40,   # Micro-enterprise assessment
        'ml_score': 0.20,         # Data-driven probability (retrained, 0.922 ROC-AUC)
    }
    
    # Decision thresholds
    THRESHOLDS = {
        'approve': 65,
        'conditional_approve': 50,
        'refer': 40,
        'senior_review': 30
    }
    
    # Hard stop conditions that override ensemble
    # MCA DECLINE is a strong signal because it's based on empirically-validated
    # transaction consistency metrics
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
        
        # Analyze score convergence
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
            decision, adjusted_score, risk_factors, positive_factors
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
                'scoring_systems_used': len([s for s in scoring_results.values() if s.available]),
                'weights_applied': self.weights
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
        
        # MCA rule decline - THIS IS IMPORTANT
        # Based on empirically-validated transaction consistency metrics:
        # - inflow_days_30d <= 8: Very sparse deposits
        # - max_inflow_gap_days >= 21: Very long gaps between deposits
        # - inflow_cv >= 1.3: Very high variability
        mca_decision = scores.get('mca_decision') or params.get('mca_rule_decision')
        if mca_decision and str(mca_decision).upper() == 'DECLINE':
            hard_stop_reasons.append(self.HARD_STOP_CONDITIONS['mca_decline'])
        
        # Critical DSCR
        dscr = metrics.get('Debt Service Coverage Ratio', 1.0)
        if dscr < 0.5:
            hard_stop_reasons.append(self.HARD_STOP_CONDITIONS['dscr_critical'])
        
        # Critical directors score
        directors = params.get('directors_score', 50)
        if directors < 20:
            hard_stop_reasons.append(self.HARD_STOP_CONDITIONS['directors_critical'])
        
        # Multiple CCJs
        if params.get('business_ccj', False) and params.get('director_ccj', False):
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
    
    def _collect_scores(self, scores: Dict[str, Any]) -> Dict[str, ScoringResult]:
        """Collect and normalize scores from different systems."""
        
        results = {}
        
        # MCA Rule Score (MOST IMPORTANT - transaction consistency)
        # Score is 0-100 from decide_application()
        mca_score = scores.get('mca_score', 0)
        mca_decision = scores.get('mca_decision', 'REFER')
        results['mca_score'] = ScoringResult(
            score=float(mca_score) if mca_score else 50,
            available=mca_score is not None and mca_score > 0,
            weight=self.weights.get('mca_score', 0),
            details={
                'decision': mca_decision,
                'description': 'Transaction consistency (inflow days, gaps, variability)'
            }
        )
        
        # Subprime score (already 0-100)
        subprime = scores.get('subprime_score', 0)
        results['subprime_score'] = ScoringResult(
            score=float(subprime) if subprime else 0,
            available=subprime is not None and subprime > 0,
            weight=self.weights.get('subprime_score', 0),
            details=scores.get('subprime_details', {})
        )
        
        # ML score (already 0-100 percentage)
        ml_score = scores.get('ml_score') or scores.get('adjusted_ml_score', 0)
        results['ml_score'] = ScoringResult(
            score=float(ml_score) if ml_score else 0,
            available=ml_score is not None and ml_score > 0,
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
            if result.available and result.score > 0:
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
            if r.available and r.score > 0
        ]
        
        if len(primary_scores) < 2:
            return "Insufficient Data", 5.0
        
        score_range = max(primary_scores) - min(primary_scores)
        
        if score_range <= 10:
            return "High Convergence", 0.0
        elif score_range <= 20:
            return "Good Convergence", 2.0
        elif score_range <= 30:
            return "Moderate Convergence", 5.0
        else:
            return "Low Convergence", 8.0
    
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
        
        # MCA REFER means transaction consistency is borderline
        # Even with a good score, we should be cautious -- but never
        # override a DECLINE that the score thresholds demand
        if mca_decision == 'REFER':
            if score >= self.THRESHOLDS['approve']:
                return Decision.CONDITIONAL_APPROVE
            elif score >= self.THRESHOLDS['conditional_approve']:
                return Decision.REFER
            elif score >= self.THRESHOLDS['refer']:
                return Decision.SENIOR_REVIEW
            else:
                return Decision.DECLINE
        
        # MCA APPROVE gives confidence boost -- but still respects thresholds
        if mca_decision == 'APPROVE':
            if score >= self.THRESHOLDS['approve']:
                return Decision.APPROVE
            elif score >= self.THRESHOLDS['conditional_approve']:
                return Decision.CONDITIONAL_APPROVE
            elif score >= self.THRESHOLDS['refer']:
                return Decision.REFER
            elif score >= self.THRESHOLDS['senior_review']:
                return Decision.SENIOR_REVIEW
            else:
                return Decision.DECLINE
        
        # Standard threshold-based decision (no MCA decision or unknown)
        if score >= self.THRESHOLDS['approve']:
            return Decision.APPROVE
        elif score >= self.THRESHOLDS['conditional_approve']:
            return Decision.CONDITIONAL_APPROVE
        elif score >= self.THRESHOLDS['refer']:
            return Decision.REFER
        elif score >= self.THRESHOLDS['senior_review']:
            return Decision.SENIOR_REVIEW
        else:
            return Decision.DECLINE
    
    def _calculate_confidence(
        self,
        scoring_results: Dict[str, ScoringResult],
        convergence: str
    ) -> float:
        """Calculate confidence in the ensemble decision."""
        
        # Base confidence from number of available scores
        available_count = sum(1 for r in scoring_results.values() if r.available)
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
        if params.get('director_ccj', False):
            risk_factors.append("Director CCJ on record")
        
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
        
        elif decision == Decision.CONDITIONAL_APPROVE:
            recommendations.extend([
                "Approve with enhanced monitoring",
                "Consider bi-weekly payment collection",
                "Request additional documentation"
            ])
            if params.get('business_ccj') or params.get('director_ccj'):
                recommendations.append("Obtain personal guarantee")
        
        elif decision == Decision.REFER:
            recommendations.extend([
                "Manual underwriter review required",
                "Request 3 months additional bank statements",
                "Verify revenue sources"
            ])
            if len(risk_factors) > 3:
                recommendations.append("Consider reduced loan amount")
        
        elif decision == Decision.SENIOR_REVIEW:
            recommendations.extend([
                "Senior underwriter approval required",
                "Full due diligence package needed",
                "Consider alternative product structure"
            ])
        
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
        
        has_ccj = params.get('business_ccj', False) or params.get('director_ccj', False)
        
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
        risk_factors: List[str],
        positive_factors: List[str]
    ) -> str:
        """Generate primary reason for decision."""
        
        if decision == Decision.APPROVE:
            if positive_factors:
                return f"Strong overall profile: {positive_factors[0]}"
            return f"Combined score ({score:.0f}) exceeds approval threshold"
        
        elif decision == Decision.CONDITIONAL_APPROVE:
            return f"Good score ({score:.0f}) with manageable risk factors"
        
        elif decision == Decision.REFER:
            if risk_factors:
                return f"Manual review needed due to: {risk_factors[0]}"
            return f"Score ({score:.0f}) requires underwriter review"
        
        elif decision == Decision.SENIOR_REVIEW:
            if risk_factors:
                return f"Multiple concerns including: {risk_factors[0]}"
            return f"Low score ({score:.0f}) requires senior approval"
        
        else:  # DECLINE
            if risk_factors:
                return f"Does not meet criteria: {risk_factors[0]}"
            return f"Score ({score:.0f}) below minimum threshold"


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
