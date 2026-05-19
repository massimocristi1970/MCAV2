"""
Train an ML challenger model from enriched batch scorecard exports.

This script intentionally does not replace production model.pkl/scaler.pkl.
It builds a labelled dataset from saved batch runs, trains only when there is
enough paid/not-paid evidence, and writes comparison/readiness reports.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET_COL = "outcome"
LABEL_COL = "outcome_label"

FEATURE_COLS = [
    # Existing ML feature family
    "directors_score",
    "Total Revenue",
    "Monthly Average Revenue",
    "Total Debt",
    "Debt-to-Income Ratio",
    "Operating Margin",
    "Debt Service Coverage Ratio",
    "Cash Flow Volatility",
    "Revenue Growth Rate",
    "Average Month-End Balance",
    "Average Negative Balance Days per Month",
    "Number of Bounced Payments",
    "company_age_months",
    # Current scorecard outputs
    "subprime_score",
    "mca_rule_score",
    # Bureau signals
    "business_ccj",
    "business_credit_score",
    "business_credit_score_min",
    "business_credit_score_max",
    "business_credit_score_suppressed",
    "business_credit_limit",
    "business_max_recommended_credit",
    "business_negative_impact_count",
    "business_enquiries_3m",
    "business_company_searches_12m",
    "business_bureau_needs_attention",
    # Open banking insight layer
    "OB Transaction Count",
    "OB History Months",
    "OB True Revenue",
    "OB Non-Revenue Inflow Ratio",
    "OB Revenue Active Day Rate",
    "OB Top Revenue Source Percentage",
    "OB Card Processor Revenue Share",
    "OB Weakest Month Revenue",
    "OB Debt Repayment Burden",
    "OB Recent Loan Credits 30D",
    "OB Low Balance Days <1000",
    "OB Recent Failed Payments 30D",
    "OB Requested Loan To Monthly Revenue",
    "OB Requested Loan To Weakest Month Revenue",
    # Card processor insight layer; may be absent until new runs are produced.
    "Card Processor Statements Parsed",
    "Card Processor Months Present",
    "Card Sales Total",
    "Card Sales Monthly Average",
    "Card Weakest Month Sales",
    "Card Strongest Month Sales",
    "Card Latest Month Sales",
    "Card Sales Volatility",
    "Card Latest Month Drop Pct",
    "Card Refund Ratio",
    "Card Chargeback Ratio",
    "Card Fee Ratio",
    "Card Average Transaction Value",
    "Card Transaction Count",
    "Card vs OB Revenue Ratio",
    "Card Unmatched Sales Shortfall",
    "Card Unmatched Sales Shortfall Pct",
]


def _read_csv_from_zip(zip_path: Path, suffix: str) -> pd.DataFrame | None:
    with zipfile.ZipFile(zip_path) as zf:
        matches = [n for n in zf.namelist() if n.endswith(suffix)]
        if not matches:
            return None
        # Prefer root scorecard_features over calibration report paths.
        name = sorted(matches, key=lambda n: n.count("/"))[0]
        with zf.open(name) as f:
            return pd.read_csv(f)


def _load_saved_run(path: Path) -> pd.DataFrame | None:
    if path.is_file() and path.suffix.lower() == ".zip":
        scorecard = _read_csv_from_zip(path, "scorecard_features.csv")
        if scorecard is not None:
            return scorecard
        return _read_csv_from_zip(path, "case_scores_with_outcomes.csv")
    if path.is_dir():
        scorecard = path / "scorecard_features.csv"
        case_scores = path / "case_scores_with_outcomes.csv"
        if scorecard.exists():
            return pd.read_csv(scorecard)
        if case_scores.exists():
            return pd.read_csv(case_scores)
    if path.is_file() and path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return None


def discover_inputs(paths: Iterable[Path]) -> list[Path]:
    found: list[Path] = []
    for path in paths:
        if path.is_file():
            found.append(path)
        elif path.is_dir():
            found.extend(path.glob("*/scorecard_features.csv"))
            found.extend(path.glob("*.zip"))
    return sorted(set(found))


def load_labelled_dataset(inputs: list[Path]) -> pd.DataFrame:
    frames = []
    for path in inputs:
        df = _load_saved_run(path if path.name != "scorecard_features.csv" else path.parent)
        if df is None or df.empty or LABEL_COL not in df.columns:
            continue
        df = df.copy()
        df["source_run"] = str(path)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    labelled = combined[combined[LABEL_COL].isin(["paid", "not_paid"])].copy()
    if labelled.empty:
        return labelled

    labelled[TARGET_COL] = (labelled[LABEL_COL] == "paid").astype(int)

    dedupe_cols = [c for c in ["application_id", "company_name", "original_filename"] if c in labelled.columns]
    if dedupe_cols:
        labelled["_dedupe_key"] = labelled[dedupe_cols].fillna("").astype(str).agg("|".join, axis=1)
        labelled = labelled.sort_values("source_run").drop_duplicates("_dedupe_key", keep="last")
        labelled.drop(columns=["_dedupe_key"], inplace=True)

    return labelled.reset_index(drop=True)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    X = df.reindex(columns=FEATURE_COLS)
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    available_features = [c for c in X.columns if X[c].notna().any()]
    X = X[available_features]
    y = df[TARGET_COL].astype(int)
    return X, y, available_features


def _decision_bad_score(series: pd.Series) -> pd.Series:
    mapping = {"APPROVE": 0.15, "CONDITIONAL_APPROVE": 0.35, "REFER": 0.50, "SENIOR_REVIEW": 0.65, "DECLINE": 0.85}
    return series.fillna("").astype(str).str.upper().map(mapping)


def compare_current_engine(df: pd.DataFrame) -> dict:
    y_bad = (df[LABEL_COL] == "not_paid").astype(int)
    out: dict[str, object] = {}
    for score_col in ["subprime_score", "mca_rule_score"]:
        if score_col in df.columns and pd.to_numeric(df[score_col], errors="coerce").notna().nunique() > 1:
            good_score = pd.to_numeric(df[score_col], errors="coerce")
            bad_score = 100 - good_score
            out[f"{score_col}_bad_auc"] = float(roc_auc_score(y_bad, bad_score))
    if "final_decision" in df.columns:
        bad_score = _decision_bad_score(df["final_decision"])
        mask = bad_score.notna()
        if mask.sum() and bad_score[mask].nunique() > 1:
            out["final_decision_bad_auc"] = float(roc_auc_score(y_bad[mask], bad_score[mask]))
    return out


def train_challenger(df: pd.DataFrame, output_dir: Path, min_samples: int, min_class: int) -> dict:
    class_counts = df[TARGET_COL].value_counts().to_dict()
    paid = int(class_counts.get(1, 0))
    not_paid = int(class_counts.get(0, 0))
    readiness = {
        "labelled_rows": int(len(df)),
        "paid_rows": paid,
        "not_paid_rows": not_paid,
        "min_samples_required": min_samples,
        "min_class_required": min_class,
        "trained": False,
    }

    X, y, available_features = prepare_features(df)
    readiness["available_features"] = available_features
    readiness["missing_features"] = [c for c in FEATURE_COLS if c not in available_features]
    readiness["current_engine_comparison"] = compare_current_engine(df)

    if len(df) < min_samples or paid < min_class or not_paid < min_class:
        readiness["reason"] = "Insufficient reliable labelled outcomes for challenger training."
        return readiness

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                CalibratedClassifierCV(
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=6,
                        min_samples_leaf=4,
                        min_samples_split=8,
                        max_features="sqrt",
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                    method="sigmoid",
                    cv=min(3, paid, not_paid),
                ),
            ),
        ]
    )

    folds = min(5, paid, not_paid)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    prob_paid = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    pred = (prob_paid >= 0.5).astype(int)

    output_dir.mkdir(parents=True, exist_ok=True)
    scored = df.copy()
    scored["challenger_prob_paid_cv"] = prob_paid
    scored["challenger_pd_cv"] = 1 - prob_paid
    scored["challenger_pred_label_cv"] = np.where(pred == 1, "paid", "not_paid")
    scored.to_csv(output_dir / "challenger_scored_cases.csv", index=False)

    model.fit(X, y)
    joblib.dump(model, output_dir / "challenger_model.pkl")
    (output_dir / "challenger_features.json").write_text(json.dumps(available_features, indent=2), encoding="utf-8")

    readiness.update(
        {
            "trained": True,
            "cv_folds": folds,
            "challenger_paid_auc": float(roc_auc_score(y, prob_paid)),
            "challenger_brier": float(brier_score_loss(y, prob_paid)),
            "challenger_accuracy": float(accuracy_score(y, pred)),
            "classification_report": classification_report(y, pred, output_dict=True),
            "model_path": str(output_dir / "challenger_model.pkl"),
            "scored_cases_path": str(output_dir / "challenger_scored_cases.csv"),
        }
    )
    return readiness


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/readiness-check enriched ML challenger model.")
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        default=[],
        help="Saved run directory, saved run zip, scorecard CSV, or folder containing saved runs. Can be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/ml_challenger"))
    parser.add_argument("--min-samples", type=int, default=80)
    parser.add_argument("--min-class", type=int, default=25)
    args = parser.parse_args()

    inputs = discover_inputs(args.input or [Path("MCAV2_BatchProcessor/saved_runs")])
    dataset = load_labelled_dataset(inputs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.output_dir / "challenger_training_dataset.csv"
    dataset.to_csv(dataset_path, index=False)

    report = train_challenger(dataset, args.output_dir, args.min_samples, args.min_class)
    report.update(
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "inputs": [str(p) for p in inputs],
            "training_dataset_path": str(dataset_path),
            "production_artefacts_replaced": False,
            "ml_decision_policy": "Info only; challenger model is not wired into final decision.",
        }
    )
    (args.output_dir / "challenger_readiness_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
