"""
Reject Inference Pipeline (Stages 2–5)

Loads full_feature_dataset.csv, trains a funded-only logistic regression model,
scores rejected applications, performs parceling-based reject inference, and
produces an augmented training dataset. Can run Stage 1 (build_full_feature_dataset)
if the full feature CSV is missing or --rebuild is set.

Outputs (all CSV):
  - full_feature_dataset.csv       (from Stage 1 if run)
  - funded_only_training.csv       (labelled rows used for training)
  - funded_only_holdout_scored.csv (optional, if holdout implemented and enough data)
  - rejected_scored.csv           (rejecteds with predicted_pd, pd_band)
  - parceling_band_summary.csv     (band-level summary)
  - rejected_inferred_labels.csv  (rejecteds with inferred_outcome)
  - augmented_training_dataset.csv (funded + inferred rejecteds, ready for training)

Usage:
  python reject_inference_pipeline.py
  python reject_inference_pipeline.py --rebuild   # Force rebuild full feature dataset first
"""

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIG
# ============================================================

_BASE_DIR = os.environ.get("TRAINING_DATA_DIR", os.getcwd())
_OUTPUT_DIR = os.environ.get("TRAINING_OUTPUT_DIR", os.path.join(_BASE_DIR, "data"))
FULL_FEATURE_CSV = os.path.join(_OUTPUT_DIR, "full_feature_dataset.csv")

# Pipeline outputs
FUNDED_ONLY_TRAINING_CSV = os.path.join(_OUTPUT_DIR, "funded_only_training.csv")
FUNDED_ONLY_HOLDOUT_CSV = os.path.join(_OUTPUT_DIR, "funded_only_holdout_scored.csv")
REJECTED_SCORED_CSV = os.path.join(_OUTPUT_DIR, "rejected_scored.csv")
PARCELING_BAND_SUMMARY_CSV = os.path.join(_OUTPUT_DIR, "parceling_band_summary.csv")
REJECTED_INFERRED_LABELS_CSV = os.path.join(_OUTPUT_DIR, "rejected_inferred_labels.csv")
AUGMENTED_TRAINING_CSV = os.path.join(_OUTPUT_DIR, "augmented_training_dataset.csv")

# Funded-only model artefacts (separate from app model.pkl / scaler.pkl)
FUNDED_ONLY_MODEL_PKL = os.path.join(_OUTPUT_DIR, "reject_inference_funded_only_model.pkl")
FUNDED_ONLY_SCALER_PKL = os.path.join(_OUTPUT_DIR, "reject_inference_funded_only_scaler.pkl")

# Parceling and training
N_BANDS = int(os.environ.get("REJECT_INFERENCE_N_BANDS", "5"))
UPLIFT = float(os.environ.get("REJECT_INFERENCE_UPLIFT", "1.0"))
RANDOM_SEED = int(os.environ.get("REJECT_INFERENCE_RANDOM_SEED", "42"))
MIN_SAMPLES_FOR_HOLDOUT = 30
HOLDOUT_FRACTION = 0.2

FEATURE_COLS = [
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
TARGET_COL = "Outcome"  # 1 = good/repaid, 0 = bad/default; blank = unlabelled


def _ensure_full_feature_dataset(rebuild: bool) -> str:
    """Run Stage 1 if full_feature_dataset.csv is missing or rebuild requested."""
    if rebuild or not os.path.isfile(FULL_FEATURE_CSV):
        from build_full_feature_dataset import main as build_full

        build_full()
    if not os.path.isfile(FULL_FEATURE_CSV):
        raise FileNotFoundError(
            f"Full feature dataset not found: {FULL_FEATURE_CSV}. "
            "Run build_full_feature_dataset.py or use --rebuild."
        )
    return FULL_FEATURE_CSV


def _load_and_validate_full(path: str) -> pd.DataFrame:
    """Load full feature CSV and fail loudly if required columns are missing."""
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLS + ["company_name", "Outcome", "is_labelled", "is_rejected_or_unfunded"] if c not in df.columns]
    if missing:
        raise ValueError(f"Full feature dataset missing required columns: {missing}")
    return df


def _prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and clean feature matrix (impute, no inf)."""
    X = df[FEATURE_COLS].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X = X.fillna(0)
    return X


# ---------- Stage 2: Train funded-only model ----------


def stage2_train_funded_only_model(
    df_full: pd.DataFrame,
    random_state: int = RANDOM_SEED,
):
    """
    Train a logistic regression model on labelled (funded) rows only.
    Optionally hold out a stratified fraction for evaluation; if sample size
    is too small, skip holdout and use all funded data for training.
    Saves funded_only_training.csv and optionally holdout_scored.csv and model artefacts.
    """
    funded = df_full[df_full["is_labelled"]].copy()
    if funded.empty:
        raise ValueError("No labelled (funded) rows in full feature dataset.")

    y = pd.to_numeric(funded[TARGET_COL], errors="coerce").astype("Int64")
    funded = funded.loc[y.notna()].copy()
    y = y.dropna().astype(int)
    if len(funded) != len(y):
        funded = funded.loc[y.index]
    if funded.empty:
        raise ValueError("No funded rows with valid Outcome (0 or 1).")

    X = _prepare_X(funded)
    y = y.astype(int)

    use_holdout = len(funded) >= MIN_SAMPLES_FOR_HOLDOUT
    if use_holdout:
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, funded.index,
            test_size=HOLDOUT_FRACTION,
            stratify=y,
            random_state=random_state,
        )
    else:
        X_train, y_train = X, y
        X_test, y_test = None, None
        idx_train, idx_test = funded.index, pd.Index([])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=random_state,
    )
    model.fit(X_train_scaled, y_train)

    # Score ALL funded rows so downstream stages have predicted_pd for every funded row
    X_all = _prepare_X(funded)
    prob_all = model.predict_proba(scaler.transform(X_all))[:, 1]
    funded["prob_repaid"] = prob_all
    funded["predicted_pd"] = 1 - prob_all
    funded["_train_set"] = funded.index.isin(idx_train).astype(int)

    training_out = funded.copy()
    training_out.to_csv(FUNDED_ONLY_TRAINING_CSV, index=False)
    print(f"  Saved {FUNDED_ONLY_TRAINING_CSV} ({len(training_out)} rows)")

    if use_holdout and X_test is not None and len(X_test) > 0:
        holdout = funded.loc[idx_test].copy()
        holdout.to_csv(FUNDED_ONLY_HOLDOUT_CSV, index=False)
        auc = roc_auc_score(y_test, model.predict_proba(scaler.transform(X_test))[:, 1])
        print(f"  Holdout AUC: {auc:.3f} -> {FUNDED_ONLY_HOLDOUT_CSV}")
    else:
        if len(funded) < MIN_SAMPLES_FOR_HOLDOUT:
            print(f"  Skipped holdout (n={len(funded)} < {MIN_SAMPLES_FOR_HOLDOUT})")
        auc = roc_auc_score(y, prob_all)
        print(f"  In-sample AUC: {auc:.3f}")

    # Save model and scaler for scoring (optional but useful)
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    joblib.dump(model, FUNDED_ONLY_MODEL_PKL)
    joblib.dump(scaler, FUNDED_ONLY_SCALER_PKL)
    print(f"  Saved model/scaler -> {FUNDED_ONLY_MODEL_PKL}, {FUNDED_ONLY_SCALER_PKL}")

    return model, scaler, funded


# ---------- Stage 3: Score rejected applications ----------


def stage3_score_rejected(
    df_full: pd.DataFrame,
    model,
    scaler,
    n_bands: int = N_BANDS,
):
    """
    Score rejected/unfunded rows with the funded-only model. Assign predicted_pd
    (probability of bad) and pd_band. Output rejected_scored.csv.
    """
    rejected = df_full[df_full["is_rejected_or_unfunded"]].copy()
    if rejected.empty:
        # Still write empty file with expected columns
        empty = pd.DataFrame(columns=FEATURE_COLS + ["company_name", "application_file", "predicted_pd", "pd_band"])
        empty.to_csv(REJECTED_SCORED_CSV, index=False)
        print(f"  No rejected rows -> {REJECTED_SCORED_CSV} (empty)")
        return rejected

    X = _prepare_X(rejected)
    prob_repaid = model.predict_proba(scaler.transform(X))[:, 1]
    rejected["predicted_pd"] = 1 - prob_repaid
    # Band by quantiles of predicted_pd (so band 1 = lowest PD, band n_bands = highest PD)
    rejected["pd_band"] = pd.qcut(
        rejected["predicted_pd"].rank(method="first"),
        q=n_bands,
        labels=range(1, n_bands + 1),
        duplicates="drop",
    ).astype(int)
    out_cols = FEATURE_COLS + ["company_name", "application_file", "Outcome", "predicted_pd", "pd_band"]
    out_cols = [c for c in out_cols if c in rejected.columns]
    rejected[out_cols].to_csv(REJECTED_SCORED_CSV, index=False)
    print(f"  Saved {REJECTED_SCORED_CSV} ({len(rejected)} rows)")
    return rejected


# ---------- Stage 4: Parceling / reject inference ----------


def stage4_parceling(
    df_full: pd.DataFrame,
    funded_scored: pd.DataFrame,
    rejected_scored: pd.DataFrame,
    n_bands: int = N_BANDS,
    uplift: float = UPLIFT,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Reject inference by parceling: use observed bad rate by band from funded data,
    apply (with optional uplift) to rejected counts in each band, and assign
    inferred labels deterministically. Uses combined (funded + rejected) predicted_pd
    quantiles so band definitions align. Returns DataFrame of rejected rows with
    inferred_outcome, label_source, etc.
    """
    rng = np.random.default_rng(random_state)

    # Single band definition from combined predicted_pd so funded and rejected align
    all_pd = pd.concat([
        funded_scored["predicted_pd"],
        rejected_scored["predicted_pd"] if not rejected_scored.empty else pd.Series(dtype=float),
    ])
    all_pd = all_pd.dropna()
    if len(all_pd) < n_bands:
        n_bands_actual = max(1, len(all_pd))
    else:
        n_bands_actual = n_bands
    try:
        _, bin_edges = pd.qcut(all_pd, q=n_bands_actual, retbins=True, duplicates="drop")
        bin_edges = np.unique(bin_edges)
        bin_edges[0] = 0
        bin_edges[-1] = 1.0
    except Exception:
        bin_edges = np.linspace(0, 1, n_bands_actual + 1)

    funded_scored = funded_scored.copy()
    rejected_scored = rejected_scored.copy()
    funded_scored["pd_band"] = pd.cut(
        funded_scored["predicted_pd"],
        bins=bin_edges,
        labels=range(1, len(bin_edges)),
        include_lowest=True,
    ).astype(int)
    if not rejected_scored.empty:
        rejected_scored["pd_band"] = pd.cut(
            rejected_scored["predicted_pd"],
            bins=bin_edges,
            labels=range(1, len(bin_edges)),
            include_lowest=True,
        ).astype(int)
        out_cols = [c for c in FEATURE_COLS + ["company_name", "application_file", "Outcome", "predicted_pd", "pd_band"] if c in rejected_scored.columns]
        rejected_scored[out_cols].to_csv(REJECTED_SCORED_CSV, index=False)

    # Observed bad rate by band (funded)
    funded_scored["_bad"] = (funded_scored[TARGET_COL].astype(int) == 0).astype(int)
    band_stats = funded_scored.groupby("pd_band").agg(
        funded_count=("_bad", "count"),
        funded_bad_count=("_bad", "sum"),
    ).reset_index()
    band_stats["funded_bad_rate"] = band_stats["funded_bad_count"] / band_stats["funded_count"].replace(0, np.nan)

    # All bands that appear in funded or rejected
    band_numbers = sorted(set(funded_scored["pd_band"].dropna().astype(int).tolist()) | set(
        rejected_scored["pd_band"].dropna().astype(int).tolist() if not rejected_scored.empty else set()
    ))
    if not band_numbers:
        band_numbers = list(range(1, n_bands + 1))
    if rejected_scored.empty:
        band_stats["rejected_count"] = 0
        band_stats["adjusted_rejected_bad_rate"] = np.nan
        band_stats["inferred_bad_count"] = 0
        band_stats.to_csv(PARCELING_BAND_SUMMARY_CSV, index=False)
        inferred = pd.DataFrame()
        inferred.to_csv(REJECTED_INFERRED_LABELS_CSV, index=False)
        print(f"  No rejected rows -> parceling summary and inferred labels written (empty)")
        return inferred

    # Ensure band_stats has a row for every band (add bands with no funded)
    for b in band_numbers:
        if b not in band_stats["pd_band"].values:
            band_stats = pd.concat([
                band_stats,
                pd.DataFrame([{"pd_band": b, "funded_count": 0, "funded_bad_count": 0, "funded_bad_rate": np.nan}]),
            ], ignore_index=True)
    overall_bad_rate = funded_scored["_bad"].mean() if len(funded_scored) else 0
    band_stats["funded_bad_rate"] = band_stats["funded_bad_rate"].fillna(overall_bad_rate)
    rej_by_band = rejected_scored.groupby("pd_band").size()
    band_stats["rejected_count"] = band_stats["pd_band"].map(lambda b: rej_by_band.get(b, 0)).fillna(0).astype(int)
    band_stats["adjusted_rejected_bad_rate"] = (band_stats["funded_bad_rate"] * uplift).clip(0, 1)
    band_stats["inferred_bad_count"] = (
        np.round(band_stats["rejected_count"] * band_stats["adjusted_rejected_bad_rate"]).fillna(0).astype(int)
    )
    band_stats.to_csv(PARCELING_BAND_SUMMARY_CSV, index=False)
    print(f"  Saved {PARCELING_BAND_SUMMARY_CSV}")

    # Assign inferred 0/1 to each rejected row deterministically by band
    inferred_list = []
    for band in band_numbers:
        rej_band = rejected_scored[rejected_scored["pd_band"] == band].copy()
        if rej_band.empty:
            continue
        row = band_stats[band_stats["pd_band"] == band]
        n_bad = int(row["inferred_bad_count"].item()) if len(row) else 0
        n_bad = int(min(n_bad, len(rej_band)))
        # Deterministic: sort by company_name then assign first n_bad as bad (0)
        rej_band = rej_band.sort_values("company_name")
        indices = rej_band.index.tolist()
        # Shuffle with fixed seed so assignment is reproducible but not trivially by name order only
        rng.shuffle(indices)
        bad_indices = set(indices[:n_bad])
        rej_band["inferred_outcome"] = rej_band.index.map(lambda i: 0 if i in bad_indices else 1)
        rej_band["label_source"] = "inferred"
        rej_band["original_outcome"] = pd.NA
        inferred_list.append(rej_band)

    if not inferred_list:
        inferred = pd.DataFrame()
    else:
        inferred = pd.concat(inferred_list, ignore_index=False)
    inferred["final_outcome_for_training"] = inferred["inferred_outcome"]

    out_cols = (
        FEATURE_COLS
        + ["company_name", "application_file", "predicted_pd", "pd_band", "original_outcome", "inferred_outcome", "label_source", "final_outcome_for_training"]
    )
    out_cols = [c for c in out_cols if c in inferred.columns]
    if not inferred.empty:
        inferred[out_cols].to_csv(REJECTED_INFERRED_LABELS_CSV, index=False)
    else:
        pd.DataFrame(columns=out_cols).to_csv(REJECTED_INFERRED_LABELS_CSV, index=False)
    print(f"  Saved {REJECTED_INFERRED_LABELS_CSV} ({len(inferred)} rows)")
    return inferred


# ---------- Stage 5: Augmented dataset ----------


def stage5_augmented_dataset(
    df_full: pd.DataFrame,
    funded_scored: pd.DataFrame,
    rejected_inferred: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build augmented training dataset: all funded rows (actual outcome) plus
    all rejected rows (inferred outcome). Ready for downstream model training.
    """
    funded = df_full[df_full["is_labelled"]].copy()
    funded = funded[funded[TARGET_COL].notna()].copy()
    funded["original_outcome"] = funded[TARGET_COL]
    funded["inferred_outcome"] = pd.NA
    funded["final_outcome_for_training"] = funded[TARGET_COL].astype(int)
    funded["label_source"] = "actual"
    if "predicted_pd" in funded_scored.columns and "company_name" in funded_scored.columns:
        merge_cols = funded_scored[["company_name", "predicted_pd", "pd_band"]].drop_duplicates("company_name")
        funded = funded.merge(merge_cols, on="company_name", how="left", suffixes=("", "_y"))
        if "pd_band_y" in funded.columns:
            funded["pd_band"] = funded["pd_band_y"]
            funded = funded.drop(columns=["pd_band_y"], errors="ignore")
    else:
        funded["predicted_pd"] = np.nan
        funded["pd_band"] = np.nan

    aug_cols = (
        FEATURE_COLS
        + [
            "company_name",
            "application_file",
            "Outcome",
            "original_outcome",
            "inferred_outcome",
            "final_outcome_for_training",
            "label_source",
            "predicted_pd",
            "pd_band",
        ]
    )
    funded_out = funded[[c for c in aug_cols if c in funded.columns]].copy()

    if rejected_inferred.empty:
        aug = funded_out
    else:
        rej_out = rejected_inferred[[c for c in aug_cols if c in rejected_inferred.columns]].copy()
        rej_out["Outcome"] = pd.NA
        aug = pd.concat([funded_out, rej_out], ignore_index=True)

    aug.to_csv(AUGMENTED_TRAINING_CSV, index=False)
    print(f"  Saved {AUGMENTED_TRAINING_CSV} ({len(aug)} rows)")
    return aug


# ---------- Run summary ----------


def print_run_summary(
    df_full: pd.DataFrame,
    funded_scored: pd.DataFrame,
    rejected_scored: pd.DataFrame,
    rejected_inferred: pd.DataFrame,
    aug: pd.DataFrame,
):
    """Print a concise console summary of the run."""
    n_matched = len(df_full)
    n_funded = df_full["is_labelled"].sum()
    n_rejected = df_full["is_rejected_or_unfunded"].sum()
    funded_bad_rate = (
        (funded_scored[TARGET_COL].astype(int) == 0).mean()
        if not funded_scored.empty and TARGET_COL in funded_scored.columns
        else float("nan")
    )
    rej_avg_pd = rejected_scored["predicted_pd"].mean() if not rejected_scored.empty and "predicted_pd" in rejected_scored.columns else float("nan")
    n_inferred_bad = (rejected_inferred["inferred_outcome"] == 0).sum() if not rejected_inferred.empty and "inferred_outcome" in rejected_inferred.columns else 0

    print("\n" + "=" * 60)
    print("REJECT INFERENCE PIPELINE — RUN SUMMARY")
    print("=" * 60)
    print(f"  Total matched applications:  {n_matched}")
    print(f"  Funded (labelled) count:     {n_funded}")
    print(f"  Rejected/unlabelled count:    {n_rejected}")
    print(f"  Funded bad rate:             {funded_bad_rate:.2%}" if not np.isnan(funded_bad_rate) else "  Funded bad rate:             N/A")
    print(f"  Rejected average PD:         {rej_avg_pd:.2%}" if not np.isnan(rej_avg_pd) else "  Rejected average PD:         N/A")
    print(f"  Inferred bad count:          {n_inferred_bad}")
    print(f"  Final augmented dataset size: {len(aug)}")
    print("=" * 60)


def main(rebuild: bool = False):
    """Run full pipeline: ensure Stage 1, then Stages 2–5."""
    _ensure_full_feature_dataset(rebuild)
    df_full = _load_and_validate_full(FULL_FEATURE_CSV)

    print("\n--- Stage 2: Train funded-only model ---")
    model, scaler, funded_scored = stage2_train_funded_only_model(df_full)

    print("\n--- Stage 3: Score rejected applications ---")
    rejected_scored = stage3_score_rejected(df_full, model, scaler)

    print("\n--- Stage 4: Parceling / reject inference ---")
    rejected_inferred = stage4_parceling(df_full, funded_scored, rejected_scored)

    print("\n--- Stage 5: Augmented dataset ---")
    aug = stage5_augmented_dataset(df_full, funded_scored, rejected_inferred)

    print_run_summary(df_full, funded_scored, rejected_scored, rejected_inferred, aug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reject inference pipeline (Stages 2–5, optional Stage 1). "
        "Outputs: full_feature_dataset.csv, funded_only_training.csv, rejected_scored.csv, "
        "parceling_band_summary.csv, rejected_inferred_labels.csv, augmented_training_dataset.csv. "
        "Config: REJECT_INFERENCE_N_BANDS, REJECT_INFERENCE_UPLIFT, REJECT_INFERENCE_RANDOM_SEED.",
    )
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild full feature dataset before running")
    args = parser.parse_args()
    main(rebuild=args.rebuild)
