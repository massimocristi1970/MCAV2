"""
Build synthetic MCA/business lending dataset from a reference CSV.

Generates synthetic application rows using the same 13 ML features as the repo,
with optional scenario controls and optional synthetic outcomes. Outputs are
clearly labelled as synthetic and saved to a dedicated folder.

IMPORTANT: Synthetic data is for testing, scenario analysis, stress testing,
dashboard/demo data, and pipeline validation ONLY. It must NOT be used for
production model calibration or mixed with real training data for production models.

Usage:
  python build_synthetic_dataset.py --input data/ml_training_dataset.csv
  python build_synthetic_dataset.py --input data/augmented_training_dataset.csv --rows 5000 --scenario adverse_case --generate-outcomes --seed 42
"""

import argparse
import json
import os
from datetime import datetime, timezone

from synthetic_data import generate, validate, write_validation_outputs
from synthetic_data.schema import ML_FEATURE_NAMES, SYNTHETIC_METADATA_COLS, get_required_columns


# ============================================================
# Defaults
# ============================================================

_DEFAULT_INPUT = "data/ml_training_dataset.csv"
_DEFAULT_OUTPUT_DIR = "data/synthetic"
_DEFAULT_ROWS = 1000
_DEFAULT_MODE = "hybrid_rules"
_DEFAULT_SCENARIO = "base_case"
_DEFAULT_SEED = 42
_DEFAULT_PERTURBATION = 0.2


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic MCA application dataset for testing/scenario use only. Not for production calibration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", default=_DEFAULT_INPUT, help="Path to reference CSV (must contain 13 ML feature columns)")
    parser.add_argument("--output-dir", "-o", default=_DEFAULT_OUTPUT_DIR, help="Output directory for synthetic and validation files")
    parser.add_argument("--rows", "-n", type=int, default=_DEFAULT_ROWS, help="Number of synthetic rows to generate")
    parser.add_argument("--mode", "-m", choices=["bootstrap_rows", "independent_feature_sampling", "hybrid_rules"], default=_DEFAULT_MODE, help="Generation mode")
    parser.add_argument("--scenario", "-s", default=_DEFAULT_SCENARIO, help="Scenario name (e.g. base_case, adverse_case)")
    parser.add_argument("--generate-outcomes", action="store_true", help="Generate synthetic_pd and synthetic_outcome")
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED, help="Random seed for reproducibility")
    parser.add_argument("--perturbation-strength", type=float, default=_DEFAULT_PERTURBATION, help="Jitter strength (0 = none)")
    parser.add_argument("--target-bad-rate", type=float, default=None, help="Target synthetic bad rate (if --generate-outcomes)")
    parser.add_argument("--high-risk-sector-share", type=float, default=None, help="Fraction of rows with Sector_Risk=1 (0-1)")
    parser.add_argument("--metadata-json", action="store_true", help="Write synthetic_run_metadata.json")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Reference input not found: {args.input}")

    # Generate
    syn, n_clipped, ref = generate(
        reference_path=args.input,
        n_rows=args.rows,
        mode=args.mode,
        scenario_name=args.scenario,
        perturbation_strength=args.perturbation_strength,
        random_seed=args.seed,
        generate_outcomes=args.generate_outcomes,
        target_bad_rate=args.target_bad_rate,
        high_risk_sector_share=args.high_risk_sector_share,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    dataset_path = os.path.join(args.output_dir, "synthetic_dataset.csv")
    # Column order: features then metadata
    feature_cols = [c for c in ML_FEATURE_NAMES if c in syn.columns]
    meta_cols = [c for c in SYNTHETIC_METADATA_COLS if c in syn.columns]
    other = [c for c in syn.columns if c not in feature_cols and c not in meta_cols]
    syn = syn[feature_cols + meta_cols + other]
    syn.to_csv(dataset_path, index=False)

    # Validation
    val_result = validate(syn, ref)
    write_validation_outputs(
        val_result,
        args.output_dir,
        scenario_name=args.scenario,
        mode=args.mode,
        generate_outcomes=args.generate_outcomes,
    )

    if args.metadata_json:
        meta = {
            "run_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "input_path": args.input,
            "output_dir": args.output_dir,
            "reference_rows": len(ref),
            "synthetic_rows": len(syn),
            "scenario": args.scenario,
            "mode": args.mode,
            "generate_outcomes": args.generate_outcomes,
            "seed": args.seed,
            "perturbation_strength": args.perturbation_strength,
            "target_bad_rate": args.target_bad_rate,
            "high_risk_sector_share": args.high_risk_sector_share,
            "n_clipped": n_clipped,
        }
        if args.generate_outcomes:
            meta["synthetic_avg_pd"] = float(syn["synthetic_pd"].mean())
            meta["synthetic_bad_rate"] = float((syn["synthetic_outcome"] == 0).mean())
        with open(os.path.join(args.output_dir, "synthetic_run_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

    # Console summary
    print("\n" + "=" * 60)
    print("SYNTHETIC DATA ENGINE — RUN SUMMARY")
    print("=" * 60)
    print(f"  Input reference:        {args.input}")
    print(f"  Reference rows:        {len(ref)}")
    print(f"  Synthetic rows:        {len(syn)}")
    print(f"  Scenario:              {args.scenario}")
    print(f"  Generation mode:      {args.mode}")
    print(f"  Synthetic outcomes:    {'Yes' if args.generate_outcomes else 'No'}")
    if args.generate_outcomes:
        print(f"  Synthetic average PD:   {syn['synthetic_pd'].mean():.4f}")
        print(f"  Synthetic bad rate:    {(syn['synthetic_outcome'] == 0).mean():.2%}")
    print(f"  Rows clipped/fixed:    {n_clipped}")
    print(f"  Output folder:         {os.path.abspath(args.output_dir)}")
    print("=" * 60)
    print(f"  Outputs:")
    print(f"    {dataset_path}")
    print(f"    {os.path.join(args.output_dir, 'synthetic_validation_summary.csv')}")
    print(f"    {os.path.join(args.output_dir, 'synthetic_feature_comparison.csv')}")
    print(f"    {os.path.join(args.output_dir, 'synthetic_correlation_comparison.csv')}")
    if args.metadata_json:
        print(f"    {os.path.join(args.output_dir, 'synthetic_run_metadata.json')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
