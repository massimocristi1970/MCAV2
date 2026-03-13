# From JSON banking data to synthetic data

The synthetic data engine does **not** read raw JSON files. It needs a **reference CSV** that already has the 13 ML features. That CSV is built from your JSON banking files + a spreadsheet by the existing training pipeline.

## Step 1: Put your files in place

1. **JSON banking files**  
   Put all your banking/transaction JSON files in:
   ```
   data/JsonExport/
   ```
   (Or set the environment variable `TRAINING_JSON_ROOT` to another folder.)

2. **Application spreadsheet**  
   Put (or create) an Excel file at:
   ```
   data/training_dataset.xlsx
   ```
   It must have at least:
   - **company_name** – matches the JSON filename (e.g. `"ABC Ltd"` for `ABC Ltd.json`)
   - **Outcome** – `1` = repaid, `0` = defaulted, *blank* = unfunded/rejected (those rows are skipped for the standard training CSV)

   Optional columns (used when building features): Total Revenue, Net Income, Operating Margin, Debt Service Coverage Ratio, requested_loan, directors_score, company_age_months, industry, total_debt.

   Company names in the sheet should match the JSON filenames (without `.json`), e.g. same spelling or the script’s matching will find them (it tries filename, stem, lowercase).

## Step 2: Build the reference CSV from JSON + spreadsheet

From the repo root run:

```bat
python build_training_dataset.py
```

Or double‑click:

```bat
build_training_dataset.bat
```

This reads `data/JsonExport/*.json` and `data/training_dataset.xlsx`, matches by company name, builds transaction-based features, and writes:

- `data/ml_training_dataset.csv`  ← **this is the reference the synthetic engine needs**
- `data/mca_training_dataset.csv`

Only rows with a non-blank **Outcome** are included in `ml_training_dataset.csv`. If you want to include rejecteds as well, use the reject-inference pipeline to get `full_feature_dataset.csv` or `augmented_training_dataset.csv`, then use one of those as the synthetic reference instead.

## Step 3: Run the synthetic data engine on that CSV

**Option A – Launcher (pick file and folder):**

```bat
build_synthetic_dataset.bat
```

When the dialogs open:

1. Select **`data/ml_training_dataset.csv`** (or `full_feature_dataset.csv` / `augmented_training_dataset.csv` if you have them).
2. Select the **output folder** (e.g. `data/synthetic`).

**Option B – Command line:**

```bat
python build_synthetic_dataset.py --input data/ml_training_dataset.csv --output-dir data/synthetic --rows 1000
```

With synthetic outcomes:

```bat
python build_synthetic_dataset.py --input data/ml_training_dataset.csv --output-dir data/synthetic --rows 1000 --generate-outcomes --target-bad-rate 0.2
```

## Summary

| Step | What you do | Result |
|------|-------------|--------|
| 1 | Put JSONs in `data/JsonExport/`, spreadsheet in `data/training_dataset.xlsx` | Files in place |
| 2 | Run `build_training_dataset.py` | `data/ml_training_dataset.csv` (reference with 13 ML features) |
| 3 | Run `build_synthetic_dataset.bat` or `build_synthetic_dataset.py --input data/ml_training_dataset.csv` | Synthetic dataset in `data/synthetic/` |

So you “upload” the JSON banking data by placing the files in `data/JsonExport/` (and the spreadsheet in `data/`), then run the training builder once to get the reference CSV, then run the synthetic model on that CSV.
