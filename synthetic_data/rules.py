"""
Business rules for synthetic feature coherence.

Ensures generated rows are plausible: e.g. Total Debt consistent with DTI and
Total Revenue; DSCR consistent with debt; high volatility often with weaker
DSCR/balance. Does not guarantee perfection—aim is to avoid obviously impossible
combinations.

NOTE: Synthetic data is for testing/simulation only, not production calibration.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .schema import ML_FEATURE_NAMES, FEATURE_SPEC, clip_value, get_bounds


def reconcile_debt_from_dti(row: pd.Series) -> pd.Series:
    """
    If Total Revenue and Debt-to-Income Ratio are set, set Total Debt = DTI * Revenue.
    Clips Total Debt to schema bounds.
    """
    out = row.copy()
    rev = out.get("Total Revenue")
    dti = out.get("Debt-to-Income Ratio")
    if pd.isna(rev) or pd.isna(dti) or rev <= 0:
        return out
    debt = rev * dti
    out["Total Debt"] = clip_value(debt, "Total Debt")
    return out


def reconcile_dti_from_debt_revenue(row: pd.Series) -> pd.Series:
    """If Total Debt and Total Revenue are set, set DTI = Debt / Revenue. Clip to bounds."""
    out = row.copy()
    rev = out.get("Total Revenue")
    debt = out.get("Total Debt")
    if pd.isna(rev) or pd.isna(debt) or rev <= 0:
        return out
    dti = debt / rev
    out["Debt-to-Income Ratio"] = clip_value(dti, "Debt-to-Income Ratio")
    return out


def reconcile_dscr_from_debt_net(row: pd.Series) -> pd.Series:
    """
    Approximate: if Total Revenue, Operating Margin, Total Debt are set,
    net income ≈ Revenue * Operating Margin; monthly debt service ≈ Debt/12;
    DSCR ≈ (net/12) / (debt/12) = net/debt (simplified). If debt is 0, set DSCR high.
    """
    out = row.copy()
    rev = out.get("Total Revenue")
    margin = out.get("Operating Margin")
    debt = out.get("Total Debt")
    if pd.isna(rev):
        return out
    net = rev * margin if not pd.isna(margin) else rev * 0.1
    if pd.isna(debt) or debt <= 0:
        out["Debt Service Coverage Ratio"] = clip_value(10.0, "Debt Service Coverage Ratio")
        return out
    # Annual net vs annual debt service (debt/12 * 12 = debt); DSCR = net / debt service
    # Simplified: debt service = debt/12 per month, annual = debt. DSCR = net / (debt/12) or similar.
    monthly_debt = debt / 12
    monthly_net = net / 12
    if monthly_debt <= 0:
        dscr = 10.0
    else:
        dscr = monthly_net / monthly_debt
    out["Debt Service Coverage Ratio"] = clip_value(max(0, dscr), "Debt Service Coverage Ratio")
    return out


def apply_coherence_rules(row: pd.Series, order: str = "debt_dti_dscr") -> pd.Series:
    """
    Apply a fixed order of reconciliation so dependent features are consistent.
    order: 'debt_dti_dscr' = set Debt from DTI and Revenue, then DSCR from Debt/Revenue/Margin.
    """
    out = row.copy()
    if order == "debt_dti_dscr":
        out = reconcile_debt_from_dti(out)
        out = reconcile_dscr_from_debt_net(out)
    elif order == "dti_debt_dscr":
        out = reconcile_dti_from_debt_revenue(out)
        out = reconcile_dscr_from_debt_net(out)
    else:
        out = reconcile_debt_from_dti(out)
        out = reconcile_dscr_from_debt_net(out)
    return out


def jitter_numeric(
    row: pd.Series,
    feature_cols: list,
    strength: float,
    rng: np.random.Generator,
    reference_std: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """
    Add Gaussian jitter to numeric features. strength scales the noise.
    If reference_std is provided, noise is strength * reference_std[col]; else 0.1 * value or 0.1.
    """
    out = row.copy()
    for col in feature_cols:
        if col not in out.index or col not in FEATURE_SPEC:
            continue
        dtype = FEATURE_SPEC[col].get("dtype", "float")
        if dtype == "binary":
            continue
        val = out[col]
        if pd.isna(val):
            continue
        try:
            x = float(val)
        except (TypeError, ValueError):
            continue
        if reference_std and col in reference_std:
            sigma = strength * reference_std[col]
        else:
            sigma = strength * (0.1 * abs(x) + 0.1)
        jitter = rng.normal(0, sigma)
        out[col] = clip_value(x + jitter, col)
    return out


def soft_constrain_volatility_balance(row: pd.Series) -> pd.Series:
    """
    Soft rule: very high Cash Flow Volatility often with lower Average Month-End Balance
    (or more negative). We don't force it—just nudge balance down slightly if volatility is high.
    """
    out = row.copy()
    vol = out.get("Cash Flow Volatility")
    bal = out.get("Average Month-End Balance")
    if pd.isna(vol) or pd.isna(bal):
        return out
    if vol > 1.0 and bal > 0:
        # Slight downward nudge for balance when volatility is high
        out["Average Month-End Balance"] = clip_value(bal * 0.9, "Average Month-End Balance")
    return out


def soft_constrain_bounced_balance(row: pd.Series) -> pd.Series:
    """
    High Number of Bounced Payments often with more Negative Balance Days or lower balance.
    Optional soft nudge.
    """
    out = row.copy()
    bounced = out.get("Number of Bounced Payments", 0)
    neg_days = out.get("Average Negative Balance Days per Month", 0)
    if bounced >= 5 and (pd.isna(neg_days) or float(neg_days) < 2):
        out["Average Negative Balance Days per Month"] = clip_value(
            max(2, float(neg_days) if not pd.isna(neg_days) else 0), "Average Negative Balance Days per Month"
        )
    return out


def apply_all_soft_rules(row: pd.Series) -> pd.Series:
    """Apply soft coherence rules (volatility/balance, bounced/balance)."""
    out = row.copy()
    out = soft_constrain_volatility_balance(out)
    out = soft_constrain_bounced_balance(out)
    return out
