"""
Optional synthetic outcome generation for simulation only.

Rule-based synthetic PD from features, then probabilistic synthetic_outcome.
These outcomes must NEVER be used for production model calibration—only for
testing, scenario analysis, and dashboard/demo data.

NOTE: Synthetic outcomes are clearly labelled (synthetic_pd, synthetic_outcome,
synthetic_label_type) and must not be mixed with real outcomes in production training.
"""

from typing import Optional

import numpy as np
import pandas as pd

from .schema import ML_FEATURE_NAMES


# Weights for a simple linear risk score (higher = worse). Normalised to produce PD in [0,1].
# Transparent and auditable.
DEFAULT_PD_WEIGHTS = {
    "Directors Score": -0.02,  # higher score -> lower PD
    "Total Revenue": 0.0,     # scale; can use log later
    "Total Debt": 0.0,        # captured via DTI
    "Debt-to-Income Ratio": 0.15,
    "Operating Margin": -0.3,
    "Debt Service Coverage Ratio": -0.02,
    "Cash Flow Volatility": 0.25,
    "Revenue Growth Rate": -0.05,
    "Average Month-End Balance": -0.00001,  # higher balance -> lower PD (small scale)
    "Average Negative Balance Days per Month": 0.02,
    "Number of Bounced Payments": 0.04,
    "Company Age (Months)": -0.001,  # older -> slightly lower PD
    "Sector_Risk": 0.12,
}


def compute_rule_based_pd(
    row: pd.Series,
    weights: Optional[dict] = None,
    scale_bounds: tuple = (0.01, 0.99),
) -> float:
    """
    Compute a synthetic PD from features using a weighted linear score.
    Score is transformed to [0,1] via sigmoid-like scaling.
    """
    w = weights or DEFAULT_PD_WEIGHTS
    score = 0.0
    for col, weight in w.items():
        if col not in row.index:
            continue
        val = row[col]
        if pd.isna(val):
            val = 0.0
        try:
            score += weight * float(val)
        except (TypeError, ValueError):
            pass
    # Map score to PD: assume score roughly in [-5, 5] or similar; use sigmoid
    # mid = 0, steepness so that most PDs are in (0.05, 0.95)
    import math
    try:
        pd_val = 1.0 / (1.0 + math.exp(-score * 0.5))
    except (OverflowError, ValueError):
        pd_val = 0.5
    pd_val = max(scale_bounds[0], min(scale_bounds[1], pd_val))
    return round(float(pd_val), 4)


def generate_synthetic_outcomes(
    df: pd.DataFrame,
    feature_cols: list,
    random_seed: int = 42,
    target_bad_rate: Optional[float] = None,
    high_risk_sector_share: Optional[float] = None,
    weights: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Add synthetic_pd and synthetic_outcome to df. Modifies copy of df.
    - synthetic_pd: rule-based PD per row.
    - synthetic_outcome: 1 = good (repaid), 0 = bad (default); drawn from Bernoulli(1 - PD).
    If target_bad_rate is set, scale PDs so that mean(synthetic_outcome==0) ≈ target_bad_rate.
    high_risk_sector_share is not used here (handled in scenario/generator); can be used to
    subset or weight rows if needed later.
    """
    out = df.copy()
    rng = np.random.default_rng(random_seed)
    pds = []
    for i in out.index:
        row = out.loc[i]
        pd_val = compute_rule_based_pd(row, weights=weights)
        pds.append(pd_val)
    out["synthetic_pd"] = pds
    if target_bad_rate is not None and 0 < target_bad_rate < 1 and len(pds) > 0:
        current_bad_rate = np.mean([1 - (1 - p) for p in pds])  # expected bad = mean(PD)
        mean_pd = np.mean(pds)
        if mean_pd > 0:
            # Scale PDs so mean(PD) ≈ target_bad_rate: multiply by factor
            factor = target_bad_rate / mean_pd
            scaled = [min(0.99, max(0.01, p * factor)) for p in pds]
            out["synthetic_pd"] = scaled
            pds = scaled
    # Draw outcome: 1 with prob (1 - PD), 0 with prob PD
    out["synthetic_outcome"] = (rng.random(len(out)) > np.array(pds)).astype(int)
    out["synthetic_label_type"] = "rule_based"
    return out
