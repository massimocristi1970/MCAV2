"""
Improved ML Model Training Script

Addresses issues found in the current model.pkl (RandomForestClassifier):
  - Small dataset (~237 samples) → need regularisation to prevent overfitting
  - min_samples_leaf=1 → single-sample leaves memorise noise
  - min_samples_split=2 → too aggressive
  - max_depth=20 → far too deep for 237 samples
  - class_weight=None → 1.83:1 class imbalance unhandled
  - No probability calibration → RF probabilities are poorly calibrated
  - No OOB / cross-validation evaluation

Usage:
    python train_improved_model.py --data path/to/training_data.csv

    The CSV must contain these columns (same as the existing model):
        Directors Score, Total Revenue, Total Debt, Debt-to-Income Ratio,
        Operating Margin, Debt Service Coverage Ratio, Cash Flow Volatility,
        Revenue Growth Rate, Average Month-End Balance,
        Average Negative Balance Days per Month, Number of Bounced Payments,
        Company Age (Months), Sector_Risk, outcome

    Outputs:
        app/models/model_artifacts/model.pkl   (calibrated RF)
        app/models/model_artifacts/scaler.pkl  (StandardScaler)
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    brier_score_loss,
    log_loss,
)

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

TARGET_COL = "outcome"

OUTPUT_DIR = Path("app/models/model_artifacts")


def load_data(path: str) -> pd.DataFrame:
    if path.endswith(".xlsx") or path.endswith(".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in training data: {missing}")

    df = df[FEATURE_COLS + [TARGET_COL]].copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=[TARGET_COL], inplace=True)
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    return df


def train(df: pd.DataFrame, output_dir: Path = OUTPUT_DIR) -> None:
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    n_samples = len(df)
    class_counts = y.value_counts()
    print(f"\nDataset: {n_samples} samples")
    print(f"Class distribution:\n{class_counts.to_string()}")
    print(f"Imbalance ratio: {class_counts.max() / class_counts.min():.2f}:1\n")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Base estimator with regularisation tuned for small datasets ---
    # Compared to the old model (max_depth=20, min_samples_leaf=1):
    #   - max_depth=8 caps tree complexity  (was 20)
    #   - min_samples_leaf=5 prevents memorising single examples  (was 1)
    #   - min_samples_split=10 requires meaningful support for a split  (was 2)
    #   - class_weight='balanced' adjusts for 1.83:1 imbalance  (was None)
    #   - oob_score=True gives a free validation metric  (was False)
    base_rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        min_samples_split=10,
        max_features="sqrt",
        class_weight="balanced",
        oob_score=True,
        random_state=42,
        n_jobs=-1,
    )

    # --- Cross-validated evaluation of the base RF ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(base_rf, X_scaled, y, cv=cv, scoring="roc_auc")
    cv_acc = cross_val_score(base_rf, X_scaled, y, cv=cv, scoring="accuracy")
    print(f"5-fold CV  ROC-AUC:  {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")
    print(f"5-fold CV  Accuracy: {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")

    # Also evaluate a Gradient Boosting model for comparison
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=10,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    gb_auc = cross_val_score(gb, X_scaled, y, cv=cv, scoring="roc_auc")
    print(f"\nGradientBoosting CV ROC-AUC: {gb_auc.mean():.3f} ± {gb_auc.std():.3f}")

    # Pick best base model
    if gb_auc.mean() > cv_auc.mean() + 0.01:
        print("\n→ GradientBoosting outperforms RF; using GB as base model.")
        best_base = gb
    else:
        print("\n→ Using RandomForest as base model.")
        best_base = base_rf

    # --- Probability calibration via Platt scaling (sigmoid) ---
    # Uses inner 3-fold CV so we don't need a held-out calibration set.
    calibrated_model = CalibratedClassifierCV(
        best_base,
        method="sigmoid",
        cv=3,
    )
    calibrated_model.fit(X_scaled, y)

    # --- Final evaluation on full data (with calibrated probabilities) ---
    y_prob = calibrated_model.predict_proba(X_scaled)[:, 1]
    y_pred = calibrated_model.predict(X_scaled)

    print(f"\nFull-data metrics (calibrated model):")
    print(f"  ROC-AUC:     {roc_auc_score(y, y_prob):.3f}")
    print(f"  Brier Score: {brier_score_loss(y, y_prob):.4f} (lower is better)")
    print(f"  Log Loss:    {log_loss(y, y_prob):.4f}")
    print(f"\n{classification_report(y, y_pred)}")

    # --- OOB score from base RF (if we used it) ---
    if isinstance(best_base, RandomForestClassifier):
        best_base.fit(X_scaled, y)
        print(f"RF OOB Score:  {best_base.oob_score_:.3f}")
        print("\nFeature importances (from RF):")
        for name, imp in sorted(
            zip(FEATURE_COLS, best_base.feature_importances_), key=lambda x: -x[1]
        ):
            print(f"  {name:45s} {imp:.4f}")

    # --- Save artefacts ---
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pkl"
    scaler_path = output_dir / "scaler.pkl"

    joblib.dump(calibrated_model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"\nSaved model  → {model_path}")
    print(f"Saved scaler → {scaler_path}")
    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description="Train improved ML scoring model")
    parser.add_argument(
        "--data",
        required=True,
        help="Path to training CSV/XLSX with feature columns and 'outcome' target",
    )
    args = parser.parse_args()

    df = load_data(args.data)
    train(df)


if __name__ == "__main__":
    main()
