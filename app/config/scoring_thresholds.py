# app/config/scoring_thresholds.py
"""
Centralized Scoring Thresholds Configuration

All scoring thresholds are defined here to ensure consistency across:
- Subprime scoring system
- Adaptive scoring
- Traditional weighted scoring
- Score diagnostics

Changes to thresholds should only be made in this file.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any


@dataclass(frozen=True)
class MetricThreshold:
    """
    Threshold configuration for a single metric.
    
    Attributes:
        full_points: Value to achieve 100% of points
        tiers: List of (threshold, percentage) tuples for partial credit
        lower_is_better: True for metrics like volatility where lower values are better
        weight: Base weight for this metric in scoring
    """
    full_points: float
    tiers: Tuple[Tuple[float, float], ...]  # ((threshold, percentage), ...)
    lower_is_better: bool = False
    weight: int = 10


@dataclass
class ScoringThresholds:
    """
    Centralized threshold configuration for all scoring systems.
    
    This ensures consistency between:
    - Score calculation (how points are awarded)
    - Diagnostics (what thresholds are displayed)
    - Recommendations (what improvements are suggested)
    """
    
    # ===========================================
    # METRIC THRESHOLDS
    # ===========================================
    
    # Debt Service Coverage Ratio (higher is better)
    # Primary indicator of ability to service debt
    DSCR: MetricThreshold = field(default_factory=lambda: MetricThreshold(
        full_points=1.9,
        tiers=(
            (1.6, 0.85),
            (1.3, 0.65),
            (1.1, 0.45),
            (0.9, 0.25),
        ),
        lower_is_better=False,
        weight=25
    ))
    
    # Average Month-End Balance (higher is better)
    # Liquidity indicator
    BALANCE: MetricThreshold = field(default_factory=lambda: MetricThreshold(
        full_points=2500,
        tiers=(
            (1500, 0.80),
            (750, 0.55),
            (400, 0.35),
            (200, 0.15),
        ),
        lower_is_better=False,
        weight=18
    ))
    
    # Directors Score (higher is better)
    # Personal creditworthiness
    DIRECTORS: MetricThreshold = field(default_factory=lambda: MetricThreshold(
        full_points=78,
        tiers=(
            (60, 0.80),
            (50, 0.55),
            (40, 0.35),
            (30, 0.15),
        ),
        lower_is_better=False,
        weight=16
    ))
    
    # Cash Flow Volatility (lower is better)
    # Stability indicator
    VOLATILITY: MetricThreshold = field(default_factory=lambda: MetricThreshold(
        full_points=0.30,
        tiers=(
            (0.45, 0.80),
            (0.60, 0.55),
            (0.75, 0.30),
            (0.95, 0.10),
        ),
        lower_is_better=True,
        weight=14
    ))
    
    # Revenue Growth Rate (higher is better)
    # Growth indicator
    GROWTH: MetricThreshold = field(default_factory=lambda: MetricThreshold(
        full_points=0.12,
        tiers=(
            (0.06, 0.80),
            (0.01, 0.55),
            (-0.03, 0.30),
            (-0.07, 0.15),
        ),
        lower_is_better=False,
        weight=10
    ))
    
    # Operating Margin (higher is better)
    # Profitability indicator
    MARGIN: MetricThreshold = field(default_factory=lambda: MetricThreshold(
        full_points=0.06,
        tiers=(
            (0.04, 0.80),
            (0.02, 0.60),
            (0.005, 0.35),
            (-0.02, 0.15),
        ),
        lower_is_better=False,
        weight=6
    ))
    
    # Average Negative Balance Days per Month (lower is better)
    # Cash management indicator
    NEGATIVE_DAYS: MetricThreshold = field(default_factory=lambda: MetricThreshold(
        full_points=1,
        tiers=(
            (4, 0.80),
            (7, 0.55),
            (10, 0.30),
            (13, 0.10),
        ),
        lower_is_better=True,
        weight=5
    ))
    
    # Net Income (higher is better)
    # Absolute profitability
    NET_INCOME: MetricThreshold = field(default_factory=lambda: MetricThreshold(
        full_points=3000,
        tiers=(
            (500, 0.80),
            (-2500, 0.45),
            (-10000, 0.25),
            (-20000, 0.10),
        ),
        lower_is_better=False,
        weight=4
    ))
    
    # Company Age in Months (higher is better)
    # Track record indicator
    COMPANY_AGE: MetricThreshold = field(default_factory=lambda: MetricThreshold(
        full_points=18,
        tiers=(
            (12, 0.80),
            (9, 0.60),
            (6, 0.40),
            (3, 0.20),
        ),
        lower_is_better=False,
        weight=2
    ))
    
    # Number of Bounced Payments (lower is better)
    # Payment behavior indicator
    BOUNCED: MetricThreshold = field(default_factory=lambda: MetricThreshold(
        full_points=0,
        tiers=(
            (1, 0.70),
            (2, 0.40),
            (4, 0.15),
        ),
        lower_is_better=True,
        weight=3
    ))
    
    # ===========================================
    # RISK FACTOR PENALTIES
    # ===========================================
    
    RISK_PENALTIES: Dict[str, int] = field(default_factory=lambda: {
        "business_ccj": 6,
        "director_ccj": 4,
        "poor_or_no_online_presence": 2,
        "uses_generic_email": 1,
        "personal_default_12m": 5,
        "website_or_social_outdated": 2
    })
    
    # Maximum total penalty (prevents "death by 1000 cuts")
    MAX_PENALTY_CAP: int = 12
    
    # ===========================================
    # TIER THRESHOLDS
    # ===========================================
    
    TIER_THRESHOLDS: Dict[str, int] = field(default_factory=lambda: {
        "tier_1": 65,  # Premium
        "tier_2": 50,  # Standard
        "tier_3": 40,  # Higher Risk
        "tier_4": 30,  # Senior Review
        "decline": 0   # Below tier_4
    })
    
    # ===========================================
    # INDUSTRY MULTIPLIERS
    # ===========================================
    
    INDUSTRY_MULTIPLIERS: Dict[str, float] = field(default_factory=lambda: {
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
        
        # Higher risk but acceptable - less harsh penalties
        'Restaurants and Cafes': 0.95,
        'Construction Firms': 0.95,
        'Beauty Salons and Spas': 0.95,
        'Bars and Pubs': 0.93,
        'Event Planning and Management Firms': 0.92,
    })
    
    # Default multiplier for unknown industries
    DEFAULT_INDUSTRY_MULTIPLIER: float = 0.95
    
    # ===========================================
    # HELPER METHODS
    # ===========================================
    
    def get_metric_threshold(self, metric_name: str) -> MetricThreshold:
        """Get threshold configuration for a metric by name."""
        mapping = {
            'Debt Service Coverage Ratio': self.DSCR,
            'DSCR': self.DSCR,
            'Average Month-End Balance': self.BALANCE,
            'Balance': self.BALANCE,
            'Directors Score': self.DIRECTORS,
            'Directors': self.DIRECTORS,
            'Cash Flow Volatility': self.VOLATILITY,
            'Volatility': self.VOLATILITY,
            'Revenue Growth Rate': self.GROWTH,
            'Growth': self.GROWTH,
            'Operating Margin': self.MARGIN,
            'Margin': self.MARGIN,
            'Average Negative Balance Days per Month': self.NEGATIVE_DAYS,
            'Negative Days': self.NEGATIVE_DAYS,
            'Net Income': self.NET_INCOME,
            'Company Age (Months)': self.COMPANY_AGE,
            'Company Age': self.COMPANY_AGE,
            'Number of Bounced Payments': self.BOUNCED,
            'Bounced': self.BOUNCED,
        }
        return mapping.get(metric_name)
    
    def score_metric(
        self,
        metric_name: str,
        value: float,
        weight_override: int = None
    ) -> Tuple[float, float, str]:
        """
        Score a metric value using centralized thresholds.
        
        Args:
            metric_name: Name of the metric
            value: Actual value of the metric
            weight_override: Optional weight override
            
        Returns:
            Tuple of (points_earned, percentage, status)
        """
        threshold = self.get_metric_threshold(metric_name)
        if threshold is None:
            return 0, 0, 'UNKNOWN'
        
        weight = weight_override or threshold.weight
        
        # Check full points threshold
        if threshold.lower_is_better:
            if value <= threshold.full_points:
                return weight, 100.0, 'PASS'
        else:
            if value >= threshold.full_points:
                return weight, 100.0, 'PASS'
        
        # Check tier thresholds
        for tier_threshold, percentage in threshold.tiers:
            if threshold.lower_is_better:
                if value <= tier_threshold:
                    return weight * percentage, percentage * 100, 'PARTIAL'
            else:
                if value >= tier_threshold:
                    return weight * percentage, percentage * 100, 'PARTIAL'
        
        return 0, 0, 'FAIL'
    
    def get_industry_multiplier(self, industry: str) -> float:
        """Get industry risk multiplier."""
        return self.INDUSTRY_MULTIPLIERS.get(
            industry, 
            self.DEFAULT_INDUSTRY_MULTIPLIER
        )
    
    def calculate_risk_penalty(self, params: Dict[str, Any]) -> Tuple[float, List[str]]:
        """
        Calculate total risk penalty from parameters.
        
        Args:
            params: Dictionary containing risk flags
            
        Returns:
            Tuple of (total_penalty, list of applied penalties)
        """
        total_penalty = 0
        applied_penalties = []
        
        for risk_factor, penalty_value in self.RISK_PENALTIES.items():
            if params.get(risk_factor, False):
                total_penalty += penalty_value
                applied_penalties.append(f"{risk_factor}: -{penalty_value}")
        
        # Cap the penalty
        if total_penalty > self.MAX_PENALTY_CAP:
            applied_penalties.append(
                f"Penalty capped: -{total_penalty} reduced to -{self.MAX_PENALTY_CAP}"
            )
            total_penalty = self.MAX_PENALTY_CAP
        
        return total_penalty, applied_penalties
    
    def get_tier_from_score(self, score: float) -> str:
        """Determine tier from score."""
        if score >= self.TIER_THRESHOLDS['tier_1']:
            return "Tier 1"
        elif score >= self.TIER_THRESHOLDS['tier_2']:
            return "Tier 2"
        elif score >= self.TIER_THRESHOLDS['tier_3']:
            return "Tier 3"
        elif score >= self.TIER_THRESHOLDS['tier_4']:
            return "Tier 4"
        else:
            return "Decline"
    
    def get_next_tier_threshold(self, score: float) -> Tuple[str, float]:
        """
        Get the next tier up and points needed.
        
        Returns:
            Tuple of (next_tier_name, points_needed)
        """
        if score < self.TIER_THRESHOLDS['tier_4']:
            return "Tier 4", self.TIER_THRESHOLDS['tier_4'] - score
        elif score < self.TIER_THRESHOLDS['tier_3']:
            return "Tier 3", self.TIER_THRESHOLDS['tier_3'] - score
        elif score < self.TIER_THRESHOLDS['tier_2']:
            return "Tier 2", self.TIER_THRESHOLDS['tier_2'] - score
        elif score < self.TIER_THRESHOLDS['tier_1']:
            return "Tier 1", self.TIER_THRESHOLDS['tier_1'] - score
        else:
            return None, 0


# Global singleton instance
THRESHOLDS = ScoringThresholds()


def get_thresholds() -> ScoringThresholds:
    """Get the global thresholds instance."""
    return THRESHOLDS
