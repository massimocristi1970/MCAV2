"""
Synthetic data generator: bootstrap, independent sampling, hybrid_rules.

Learns from a real reference CSV (13 feature columns required) and generates
synthetic rows. Supports scenario shifts and optional synthetic outcomes.

NOTE: Synthetic data is for testing, scenario analysis, and simulation only.
It must NOT be used for production model calibration.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import schema
from . import rules
from . import scenarios
from . import outcomes
from .schema import ML_FEATURE_NAMES, clip_dataframe, get_required_columns


def load_reference(path: str) -> pd.DataFrame:
    """
    Load reference CSV or Excel and extract the 13 feature columns. Fail loudly if any missing.
    Supports: .csv (UTF-8 or latin-1), .xlsx, .xls.
    """
    path_lower = path.lower()
    if path_lower.endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1")
    required = get_required_columns()
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Reference dataset missing required columns: {missing}. "
            "Use a file that contains the 13 ML features (e.g. data/ml_training_dataset.csv, "
            "data/full_feature_dataset.csv, or data/augmented_training_dataset.csv)."
        )
    return df[required].copy()


def _reference_stats(ref: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute mean and std per feature for reference (numeric only)."""
    means = {}
    stds = {}
    for col in ML_FEATURE_NAMES:
        if col not in ref.columns:
            continue
        s = pd.to_numeric(ref[col], errors="coerce").dropna()
        means[col] = s.mean()
        stds[col] = s.std()
        if stds[col] == 0 or np.isnan(stds[col]):
            stds[col] = 1.0
    return means, stds


def _apply_scenario_shifts(
    row: pd.Series,
    scenario_name: str,
    means: Dict[str, float],
    stds: Dict[str, float],
) -> pd.Series:
    """Add scenario mean shifts to a row (e.g. adverse = lower Directors Score)."""
    out = row.copy()
    shifts = scenarios.get_scenario_mean_shifts(scenario_name)
    for col, delta in shifts.items():
        if col not in out.index:
            continue
        try:
            x = float(out[col])
            out[col] = x + delta
        except (TypeError, ValueError):
            pass
    return out


def bootstrap_rows(
    reference: pd.DataFrame,
    n_rows: int,
    scenario_name: str,
    perturbation_strength: float,
    random_seed: int,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, int]:
    """
    Sample whole rows with replacement, optionally apply scenario shift and jitter.
    Returns (synthetic_df, n_clipped).
    """
    cols = feature_cols or ML_FEATURE_NAMES
    ref = reference[[c for c in cols if c in reference.columns]].copy()
    rng = np.random.default_rng(random_seed)
    means, stds = _reference_stats(ref)
    indices = rng.integers(0, len(ref), size=n_rows)
    rows = []
    for i in indices:
        row = ref.iloc[i].copy()
        row = _apply_scenario_shifts(row, scenario_name, means, stds)
        if perturbation_strength > 0:
            row = rules.jitter_numeric(row, cols, perturbation_strength, rng, stds)
        row = rules.apply_coherence_rules(row)
        row = rules.apply_all_soft_rules(row)
        rows.append(row)
    out = pd.DataFrame(rows)
    out, n_clipped = clip_dataframe(out, cols)
    return out, n_clipped


def independent_feature_sampling(
    reference: pd.DataFrame,
    n_rows: int,
    scenario_name: str,
    random_seed: int,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, int]:
    """
    Sample each feature independently from empirical distribution (with replacement).
    Then apply scenario shift and coherence rules.
    """
    cols = feature_cols or ML_FEATURE_NAMES
    ref = reference[[c for c in cols if c in reference.columns]].copy()
    rng = np.random.default_rng(random_seed)
    means, stds = _reference_stats(ref)
    rows = []
    for _ in range(n_rows):
        row = pd.Series(dtype=float)
        for col in cols:
            if col not in ref.columns:
                continue
            sample = ref[col].dropna()
            if len(sample) == 0:
                row[col] = 0
            else:
                row[col] = rng.choice(sample)
        row = _apply_scenario_shifts(row, scenario_name, means, stds)
        row = rules.apply_coherence_rules(row)
        row = rules.apply_all_soft_rules(row)
        rows.append(row)
    out = pd.DataFrame(rows)
    out, n_clipped = clip_dataframe(out, cols)
    return out, n_clipped


def hybrid_rules(
    reference: pd.DataFrame,
    n_rows: int,
    scenario_name: str,
    perturbation_strength: float,
    random_seed: int,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, int]:
    """
    Default: sample anchor rows (bootstrap), apply scenario shift and jitter,
    then reconcile dependent features via rules. Best realism.
    """
    return bootstrap_rows(
        reference,
        n_rows,
        scenario_name,
        perturbation_strength,
        random_seed,
        feature_cols,
    )


def generate(
    reference_path: str,
    n_rows: int,
    mode: str = "hybrid_rules",
    scenario_name: str = "base_case",
    perturbation_strength: float = 0.2,
    random_seed: int = 42,
    generate_outcomes: bool = False,
    target_bad_rate: Optional[float] = None,
    high_risk_sector_share: Optional[float] = None,
) -> Tuple[pd.DataFrame, int, pd.DataFrame]:
    """
    Load reference, generate synthetic feature rows, optionally add synthetic outcomes.
    Returns (synthetic_df_with_metadata, n_clipped, reference_df).
    """
    ref = load_reference(reference_path)
    cols = [c for c in ML_FEATURE_NAMES if c in ref.columns]
    if mode == "bootstrap_rows":
        syn, n_clipped = bootstrap_rows(
            ref, n_rows, scenario_name, perturbation_strength, random_seed, cols
        )
    elif mode == "independent_feature_sampling":
        syn, n_clipped = independent_feature_sampling(
            ref, n_rows, scenario_name, random_seed, cols
        )
    else:
        syn, n_clipped = hybrid_rules(
            ref, n_rows, scenario_name, perturbation_strength, random_seed, cols
        )

    # High-risk sector share: if set, overwrite Sector_Risk for that fraction of rows
    if high_risk_sector_share is not None and 0 <= high_risk_sector_share <= 1 and "Sector_Risk" in syn.columns:
        rng = np.random.default_rng(random_seed + 1)
        n_high = int(len(syn) * high_risk_sector_share)
        idx = rng.permutation(len(syn))[:n_high]
        syn = syn.copy()
        syn["Sector_Risk"] = 0
        syn.iloc[idx, syn.columns.get_loc("Sector_Risk")] = 1

    # Metadata columns
    syn["synthetic_id"] = [f"syn_{i}" for i in range(len(syn))]
    syn["scenario_name"] = scenario_name
    syn["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    syn["data_source"] = "synthetic"
    syn["synthetic_label_type"] = ""
    syn["synthetic_pd"] = np.nan
    syn["synthetic_outcome"] = np.nan

    if generate_outcomes:
        syn = outcomes.generate_synthetic_outcomes(
            syn,
            cols,
            random_seed=random_seed + 2,
            target_bad_rate=target_bad_rate,
            high_risk_sector_share=high_risk_sector_share,
        )
    return syn, n_clipped, ref
