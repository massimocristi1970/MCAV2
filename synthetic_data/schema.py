"""
Synthetic data schema: feature names, types, bounds, and validation.

Defines the 13 ML features used across the repo and synthetic metadata columns.
All bounds and clipping rules are explicit for auditability.

NOTE: Synthetic data produced by this package is for testing, scenario analysis,
stress testing, and simulation only. It must NOT be used for production model
calibration or mixed with real training data for production models.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ============================================================
# Feature names (must match repo: train_improved_model, reject_inference_pipeline)
# ============================================================

ML_FEATURE_NAMES: List[str] = [
    "Directors Score",
    "Total Revenue",
    "Total Debt",
    "Debt-to-Income Ratio",
    "Operating Margin",
    "Debt Service Coverage Ratio",
    "Cash Flow Volatility",
    "Revenue Growth Rate",
    "Average Month-End Balance",
    "Average Negative Balance Days per Month",
    "Number of Bounced Payments",
    "Company Age (Months)",
    "Sector_Risk",
]

# Synthetic metadata columns (always present in synthetic output)
SYNTHETIC_METADATA_COLS: List[str] = [
    "synthetic_id",
    "scenario_name",
    "generated_at",
    "data_source",
    "synthetic_label_type",
    "synthetic_pd",
    "synthetic_outcome",
]

# ============================================================
# Bounds and types
# ============================================================

# (low, high) for clipping; None means no bound on that side
# Types: "int", "float", "binary"
FEATURE_SPEC: Dict[str, Dict[str, Any]] = {
    "Directors Score": {"dtype": "int", "bounds": (0, 100), "description": "Credit score 0-100"},
    "Total Revenue": {"dtype": "float", "bounds": (0, None), "description": "Annual revenue, non-negative"},
    "Total Debt": {"dtype": "float", "bounds": (0, None), "description": "Total debt, non-negative"},
    "Debt-to-Income Ratio": {"dtype": "float", "bounds": (0, 20), "description": "Total debt / Total revenue, capped"},
    "Operating Margin": {"dtype": "float", "bounds": (-1.0, 1.0), "description": "Can be negative (loss-making)"},
    "Debt Service Coverage Ratio": {"dtype": "float", "bounds": (0, None), "description": "DSCR, non-negative"},
    "Cash Flow Volatility": {"dtype": "float", "bounds": (0, None), "description": "e.g. coefficient of variation"},
    "Revenue Growth Rate": {"dtype": "float", "bounds": (-1.0, 5.0), "description": "Growth rate, can be negative"},
    "Average Month-End Balance": {"dtype": "float", "bounds": (None, None), "description": "Can be negative (overdraft)"},
    "Average Negative Balance Days per Month": {"dtype": "float", "bounds": (0, 31), "description": "Days per month"},
    "Number of Bounced Payments": {"dtype": "int", "bounds": (0, None), "description": "Integer >= 0"},
    "Company Age (Months)": {"dtype": "int", "bounds": (0, 600), "description": "Months in business"},
    "Sector_Risk": {"dtype": "binary", "bounds": (0, 1), "description": "0 = lower risk, 1 = high risk"},
}


def get_bounds(feature: str) -> Tuple[Optional[float], Optional[float]]:
    """Return (low, high) bounds for a feature. None means no bound."""
    spec = FEATURE_SPEC.get(feature)
    if not spec:
        return (None, None)
    return spec.get("bounds", (None, None))


def clip_value(value: Any, feature: str) -> float:
    """Clip a single value to schema bounds. Returns float/int as per dtype."""
    spec = FEATURE_SPEC.get(feature)
    if not spec:
        return value
    lo, hi = spec.get("bounds", (None, None))
    dtype = spec.get("dtype", "float")
    try:
        if dtype == "int" or dtype == "binary":
            x = int(float(value))
        else:
            x = float(value)
    except (TypeError, ValueError):
        return 0.0 if dtype in ("int", "binary") else 0.0
    if lo is not None and x < lo:
        x = lo
    if hi is not None and x > hi:
        x = hi
    if dtype == "int" or dtype == "binary":
        return int(x)
    return float(x)


def clip_row(row: pd.Series, feature_cols: Optional[List[str]] = None) -> pd.Series:
    """Clip all feature values in a row to schema bounds. Returns new Series."""
    out = row.copy()
    cols = feature_cols or ML_FEATURE_NAMES
    for col in cols:
        if col not in out.index:
            continue
        if col not in FEATURE_SPEC:
            continue
        out[col] = clip_value(out[col], col)
    return out


def validate_row(row: pd.Series, feature_cols: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Check a row for impossible values (before clipping).
    Returns dict with keys: is_valid, out_of_bounds (list of (col, value, bound)), clipped_count.
    """
    cols = feature_cols or ML_FEATURE_NAMES
    out_of_bounds = []
    for col in cols:
        if col not in row.index or col not in FEATURE_SPEC:
            continue
        try:
            val = row[col]
            if pd.isna(val):
                continue
            lo, hi = get_bounds(col)
            if lo is not None and float(val) < lo:
                out_of_bounds.append((col, val, f"< {lo}"))
            if hi is not None and float(val) > hi:
                out_of_bounds.append((col, val, f"> {hi}"))
        except (TypeError, ValueError):
            out_of_bounds.append((col, val, "invalid type"))
    return {
        "is_valid": len(out_of_bounds) == 0,
        "out_of_bounds": out_of_bounds,
        "clipped_count": len(out_of_bounds),
    }


def clip_dataframe(df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, int]:
    """
    Clip all feature columns in df to schema bounds. Returns (clipped_df, total_clips).
    total_clips is the number of cells that were changed (for reporting).
    """
    cols = feature_cols or [c for c in ML_FEATURE_NAMES if c in df.columns]
    out = df.copy()
    clips = 0
    for col in cols:
        if col not in FEATURE_SPEC:
            continue
        lo, hi = FEATURE_SPEC[col].get("bounds", (None, None))
        dtype = FEATURE_SPEC[col].get("dtype", "float")
        before = out[col].copy()
        out[col] = out[col].apply(lambda x: clip_value(x, col))
        if dtype == "int" or dtype == "binary":
            out[col] = out[col].astype(int)
        clips += (before != out[col]).sum()
    return out, int(clips)


def get_required_columns() -> List[str]:
    """Return the list of feature columns required in a reference dataset."""
    return list(ML_FEATURE_NAMES)
