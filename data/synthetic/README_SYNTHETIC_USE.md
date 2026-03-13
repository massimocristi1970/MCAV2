# Review: Using Synthetic Data to Improve the Model (Safely)

## What’s in `data\synthetic`

- **synthetic_dataset.csv** – 1,000 rows (base_case), same 13 ML features as the repo, plus metadata (`synthetic_id`, `scenario_name`, `data_source=synthetic`). No synthetic outcomes in this run (synthetic_pd/synthetic_outcome empty).
- **synthetic_validation_summary.csv** – Reference 268 rows vs 1,000 synthetic; 13 features compared.
- **synthetic_feature_comparison.csv** – Per-feature mean/median/std/min/max. Synthetic is in the same ballpark as reference for Directors Score, Operating Margin, Cash Flow Volatility, Average Month-End Balance, etc. Some differences: reference has Total Debt mostly 0; synthetic has more spread (from bootstrap + rules). Revenue Growth Rate is tamer in synthetic (no extreme 183x). No impossible values in synthetic (0 syn_impossible_count).
- **synthetic_correlation_comparison.csv** – Pairwise correlations ref vs synthetic. Most pairs are similar; a few differ where reference had no variance (e.g. Total Debt).
- **synthetic_run_metadata.json** – Another run (150 rows, adverse_case, with synthetic outcomes) for reference.

**Conclusion:** The synthetic data is plausible and aligned with the reference. It’s suitable for testing and analysis, not for replacing real data in production model training.

---

## Rule: Do Not Use Synthetic Data for Production Model Calibration

Synthetic data is for:

- Testing and pipeline validation  
- Scenario and stress testing  
- Dashboard/demo data  
- Sensitivity analysis  

It must **not** be used to:

- Train or calibrate the **production** credit model (`model.pkl` / `scaler.pkl` used by the app)  
- Replace or mix with real outcomes for live decision-making  

So we do **not** “improve the current model” by training it on synthetic data and deploying that model.

---

## Safe Ways to Use Synthetic Data to “Improve” Things

### 1. **Stress-test the current model (no retraining)**

- Use synthetic datasets (e.g. adverse_case, high_bounce_segment) with many rows.
- **Score** them with the **existing** `model.pkl` and `scaler.pkl`.
- Review score distribution, approval rates, and behaviour at the tails.
- This improves **confidence** in the current model and reveals edge cases without changing the production model.

### 2. **Pipeline validation**

- Confirm that the training pipeline runs on different sample sizes and feature distributions.
- Prepare a synthetic file in training format (13 features + `outcome`) and run `train_improved_model.py` with an **alternate output directory** (e.g. `app/models/model_artifacts_synthetic_test`), so production artefacts are never overwritten.
- Improves **reliability** of the pipeline, not the production model itself.

### 3. **Sensitivity analysis**

- Use synthetic rows to see how predictions change when you vary one feature (e.g. Directors Score, DSCR) while holding others fixed.
- Improves **understanding** of the model and documentation for stakeholders.

### 4. **More realistic demos and training**

- Use synthetic data in demos and training so the app and reports look realistic without exposing real applications.
- Improves **safety and usability** of demos.

---

## Script: Prepare Synthetic for Training-Format and Optional Validation Train

**prepare_synthetic_for_training.py** does the following:

1. Reads `data/synthetic/synthetic_dataset.csv` (or a path you pass).
2. If it has `synthetic_outcome`, builds a file with the 13 ML features + `outcome` (for `train_improved_model.py`).
3. Saves to `data/synthetic/synthetic_training_format.csv`.
4. Optionally runs `train_improved_model.py` with **--output-dir app/models/model_artifacts_synthetic_test** so production `model.pkl` / `scaler.pkl` are **never** overwritten.

**Note:** Your current `synthetic_dataset.csv` has no synthetic outcomes. To use this script, generate synthetic again with outcomes, for example:

```bash
python build_synthetic_dataset.py --input data/ml_training_dataset.csv --output-dir data/synthetic --rows 1000 --generate-outcomes --target-bad-rate 0.2
```

Then:

```bash
python prepare_synthetic_for_training.py --run-validation
```

That way you use synthetic only for **pipeline validation**, not for production model calibration.

---

## Summary

| Use | Improves | Production model changed? |
|-----|----------|---------------------------|
| Stress-test scoring with synthetic | Confidence, edge-case checks | No |
| Train pipeline on synthetic to a **test** folder | Pipeline validation | No (different artefacts) |
| Sensitivity analysis on synthetic | Interpretability, docs | No |
| Demos/training on synthetic | Safety, usability | No |
| Train on synthetic and deploy as production model | — | **Not allowed** |

So: you **can** use the output in `data\synthetic` to improve the **process**, **validation**, and **understanding** around the current model, but the **current model** itself should continue to be trained and calibrated only on **real** data.
