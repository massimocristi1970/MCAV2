"""
Scenario definitions for synthetic data generation.

Each scenario shifts distributions or sampling (e.g. lower Directors Score,
higher volatility) for stress testing and demos. Kept explicit and simple.

NOTE: Synthetic data is for testing/scenario/simulation only, not production.
"""

from typing import Any, Dict, Optional

# Scenario name -> dict of parameter shifts / sampling hints
# Shifts are applied when generating: e.g. mean_shift multiplies or adds to reference stats
SCENARIOS: Dict[str, Dict[str, Any]] = {
    "base_case": {
        "description": "Neutral; close to reference distribution",
        "mean_shift": {},  # no shift
        "sample_filter": None,
        "target_bad_rate_hint": None,
    },
    "growth_case": {
        "description": "Stronger revenue growth, better margins",
        "mean_shift": {
            "Revenue Growth Rate": 0.15,
            "Operating Margin": 0.05,
            "Directors Score": 5,
        },
        "sample_filter": None,
        "target_bad_rate_hint": 0.10,
    },
    "adverse_case": {
        "description": "Weaker credit: lower score, higher volatility, lower DSCR",
        "mean_shift": {
            "Directors Score": -12,
            "Cash Flow Volatility": 0.2,
            "Debt Service Coverage Ratio": -0.5,
            "Operating Margin": -0.05,
            "Average Month-End Balance": -500,
        },
        "sample_filter": None,
        "target_bad_rate_hint": 0.35,
    },
    "recession_case": {
        "description": "Stress: low growth, weak margins, high volatility",
        "mean_shift": {
            "Revenue Growth Rate": -0.10,
            "Operating Margin": -0.08,
            "Cash Flow Volatility": 0.25,
            "Directors Score": -8,
            "Debt Service Coverage Ratio": -0.3,
        },
        "sample_filter": None,
        "target_bad_rate_hint": 0.40,
    },
    "weak_credit_mix": {
        "description": "More weak credits: lower Directors Score, higher DTI",
        "mean_shift": {
            "Directors Score": -15,
            "Debt-to-Income Ratio": 0.5,
            "Number of Bounced Payments": 1,
        },
        "sample_filter": None,
        "target_bad_rate_hint": 0.30,
    },
    "strong_credit_mix": {
        "description": "More strong credits: higher score, lower DTI, better DSCR",
        "mean_shift": {
            "Directors Score": 10,
            "Debt-to-Income Ratio": -0.2,
            "Debt Service Coverage Ratio": 0.5,
            "Operating Margin": 0.03,
        },
        "sample_filter": None,
        "target_bad_rate_hint": 0.08,
    },
    "high_bounce_segment": {
        "description": "Segment with more bounced payments and negative balance days",
        "mean_shift": {
            "Number of Bounced Payments": 3,
            "Average Negative Balance Days per Month": 5,
            "Average Month-End Balance": -300,
        },
        "sample_filter": None,
        "target_bad_rate_hint": 0.28,
    },
    "young_business_segment": {
        "description": "Younger companies, more volatile",
        "mean_shift": {
            "Company Age (Months)": -24,
            "Cash Flow Volatility": 0.15,
        },
        "sample_filter": None,
        "target_bad_rate_hint": 0.25,
    },
}


def get_scenario(name: str) -> Dict[str, Any]:
    """Return scenario config for name. Raises KeyError if unknown."""
    if name not in SCENARIOS:
        raise KeyError(f"Unknown scenario: {name}. Known: {list(SCENARIOS.keys())}")
    return SCENARIOS[name].copy()


def get_scenario_mean_shifts(name: str) -> Dict[str, float]:
    """Return only the mean_shift dict for a scenario (for generator)."""
    s = get_scenario(name)
    return s.get("mean_shift", {})


def list_scenarios() -> list:
    """Return list of scenario names."""
    return list(SCENARIOS.keys())
