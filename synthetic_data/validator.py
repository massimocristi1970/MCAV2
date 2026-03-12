"""
Validation: compare synthetic vs reference dataset and produce summary CSVs.

Checks row counts, per-feature stats, quantiles, correlations, impossible values.
Outputs: synthetic_validation_summary.csv, synthetic_feature_comparison.csv,
synthetic_correlation_comparison.csv.

NOTE: Synthetic data is for testing/simulation only, not production.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import schema
from .schema import ML_FEATURE_NAMES, FEATURE_SPEC, get_bounds


def _safe_stats(ser: pd.Series) -> Dict[str, float]:
    """Mean, median, std, min, max, null_count for a series."""
    s = pd.to_numeric(ser, errors="coerce")
    return {
        "mean": s.mean(),
        "median": s.median(),
        "std": s.std(),
        "min": s.min(),
        "max": s.max(),
        "null_count": int(s.isna().sum()),
        "count": int(s.count()),
    }


def _impossible_count(ser: pd.Series, feature: str) -> int:
    """Count values outside schema bounds."""
    if feature not in FEATURE_SPEC:
        return 0
    lo, hi = get_bounds(feature)
    s = pd.to_numeric(ser, errors="coerce").dropna()
    count = 0
    for v in s:
        if lo is not None and v < lo:
            count += 1
        if hi is not None and v > hi:
            count += 1
    return count


def validate(
    synthetic: pd.DataFrame,
    reference: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compare synthetic vs reference. Returns dict with summary, feature_comparison, correlation_comparison.
    """
    cols = feature_cols or [c for c in ML_FEATURE_NAMES if c in reference.columns and c in synthetic.columns]
    summary = {
        "reference_rows": len(reference),
        "synthetic_rows": len(synthetic),
        "feature_count": len(cols),
    }
    if "synthetic_outcome" in synthetic.columns and synthetic["synthetic_outcome"].notna().any():
        summary["synthetic_bad_rate"] = (synthetic["synthetic_outcome"] == 0).mean()
        summary["synthetic_avg_pd"] = synthetic["synthetic_pd"].mean()
    else:
        summary["synthetic_bad_rate"] = None
        summary["synthetic_avg_pd"] = None

    # Per-feature comparison
    rows = []
    for col in cols:
        if col not in reference.columns or col not in synthetic.columns:
            continue
        ref_s = reference[col]
        syn_s = synthetic[col]
        ref_stats = _safe_stats(ref_s)
        syn_stats = _safe_stats(syn_s)
        imp_ref = _impossible_count(ref_s, col)
        imp_syn = _impossible_count(syn_s, col)
        rows.append({
            "feature": col,
            "ref_mean": ref_stats["mean"],
            "syn_mean": syn_stats["mean"],
            "ref_median": ref_stats["median"],
            "syn_median": syn_stats["median"],
            "ref_std": ref_stats["std"],
            "syn_std": syn_stats["std"],
            "ref_min": ref_stats["min"],
            "syn_min": syn_stats["min"],
            "ref_max": ref_stats["max"],
            "syn_max": syn_stats["max"],
            "ref_null_count": ref_stats["null_count"],
            "syn_null_count": syn_stats["null_count"],
            "ref_impossible_count": imp_ref,
            "syn_impossible_count": imp_syn,
        })
    feature_comparison = pd.DataFrame(rows)

    # Correlation comparison: numeric cols only
    ref_numeric = reference[cols].apply(pd.to_numeric, errors="coerce")
    syn_numeric = synthetic[cols].apply(pd.to_numeric, errors="coerce")
    ref_corr = ref_numeric.corr()
    syn_corr = syn_numeric.corr()
    diff = syn_corr - ref_corr
    correlation_comparison = []
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if i >= j:
                continue
            correlation_comparison.append({
                "feature_a": c1,
                "feature_b": c2,
                "ref_corr": ref_corr.loc[c1, c2] if c1 in ref_corr.index and c2 in ref_corr.columns else np.nan,
                "syn_corr": syn_corr.loc[c1, c2] if c1 in syn_corr.index and c2 in syn_corr.columns else np.nan,
                "corr_diff": diff.loc[c1, c2] if c1 in diff.index and c2 in diff.columns else np.nan,
            })
    correlation_df = pd.DataFrame(correlation_comparison)

    return {
        "summary": summary,
        "feature_comparison": feature_comparison,
        "correlation_comparison": correlation_df,
    }


def write_validation_outputs(
    result: Dict[str, Any],
    output_dir: str,
    scenario_name: str,
    mode: str,
    generate_outcomes: bool,
) -> None:
    """Write validation_summary, feature_comparison, correlation_comparison CSVs."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    summary = result["summary"]
    summary["scenario_name"] = scenario_name
    summary["mode"] = mode
    summary["generate_outcomes"] = generate_outcomes
    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(output_dir, "synthetic_validation_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    result["feature_comparison"].to_csv(
        os.path.join(output_dir, "synthetic_feature_comparison.csv"), index=False
    )
    result["correlation_comparison"].to_csv(
        os.path.join(output_dir, "synthetic_correlation_comparison.csv"), index=False
    )
