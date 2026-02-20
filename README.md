# MCA v2 - Business Finance Scorecard

A risk assessment platform for **short-term subprime business lending** (typically £1,000-£10,000 loans, 6-9 month terms). The application analyses bank transaction data, calculates financial metrics, and produces a unified lending recommendation using an ensemble of three scoring systems.

---

## Scoring System

The application combines three scoring methodologies into a single weighted recommendation:

```
Combined Score = (MCA Rule x 40%) + (Subprime x 40%) + (ML Score x 20%)

Decision: APPROVE | CONDITIONAL | REFER | SENIOR_REVIEW | DECLINE
```

### MCA Rule Score (40%)

Assesses transaction consistency -- the most important predictor of MCA repayment.

| Signal | What It Measures |
|--------|------------------|
| Inflow Days (30d) | Days with positive inflows in the last 30 days |
| Maximum Gap | Longest period without any inflows |
| Inflow Volatility | Stability of inflow amounts (coefficient of variation) |

Decision logic:
- Insufficient data -> DECLINE
- Very sparse deposits AND long gaps -> DECLINE
- Extremely erratic cash flow -> DECLINE
- Score >= 70 with no hard stops -> APPROVE
- Score >= 50 -> REFER
- Below 50 -> DECLINE

The MCA Rule engine is implemented in `mca_scorecard_rules.py`.

### Subprime Score (40%)

A comprehensive risk-tier assessment designed specifically for micro-enterprise lending.

| Component | Weight | What It Measures |
|-----------|--------|------------------|
| Debt Service Coverage Ratio | 28% | Ability to service payments |
| Average Month-End Balance | 18% | Liquidity buffer |
| Directors Score | 16% | Personal creditworthiness |
| Cash Flow Volatility | 12% | Stability for repayment |
| Revenue Growth Rate | 10% | Business trajectory |
| Operating Margin | 6% | Current profitability |
| Net Income | 4% | Overall profitability |
| Negative Balance Days | 4% | Cash management |
| Company Age | 2% | Business maturity |

Risk tier classification:

| Tier | Score | Decision | Factor Rate |
|------|-------|----------|-------------|
| Tier 1 (Premium) | 65+ | APPROVE | 1.5-1.6x |
| Tier 2 (Standard) | 50-65 | APPROVE | 1.7-1.85x |
| Tier 3 (Conditional) | 40-50 | CONDITIONAL | 1.85-2.0x |
| Tier 4 (High Risk) | 30-40 | SENIOR_REVIEW | 2.0-2.2x |
| Decline | <30 | DECLINE | N/A |

Penalties are applied for business CCJs (-12), personal defaults (-8), director CCJs (-8), no online presence (-4), outdated web presence (-3), and generic email (-2).

### ML Score (20%)

A Random Forest classifier trained on 272 historical loan applications, wrapped in probability calibration (CalibratedClassifierCV). Achieves 0.922 ROC-AUC on 5-fold cross-validation.

Top features by importance:

| Feature | Importance |
|---------|-----------|
| Cash Flow Volatility | 45% |
| Revenue Growth Rate | 19% |
| Total Revenue | 7% |
| Company Age | 6% |
| Operating Margin | 5% |
| DSCR | 5% |
| Directors Score | 4% |

The model uses 13 features in total. It is regularised with `max_depth=8`, `min_samples_leaf=5`, and `class_weight='balanced'` to handle the 3.5:1 class imbalance and prevent overfitting on the small dataset.

### Decision Process

```
1. HARD STOPS (immediate decline):
   - MCA Rule = DECLINE
   - DSCR < 0.5
   - Directors Score < 20
   - Both business and director CCJs present

2. SCORE-BASED DECISION:
   Combined Score >= 65  ->  APPROVE
   Combined Score >= 50  ->  CONDITIONAL_APPROVE
   Combined Score >= 40  ->  REFER
   Combined Score >= 30  ->  SENIOR_REVIEW
   Combined Score <  30  ->  DECLINE

3. CONVERGENCE CHECK:
   All three scores compared. If they disagree significantly
   (range > 30 points), confidence is reduced and the combined
   score is penalised by up to 8 points.
```

### Score Convergence

| Gap Between Scores | Label | Penalty |
|--------------------|-------|---------|
| 0-10 | High Convergence | 0 |
| 11-20 | Good Convergence | -2 |
| 21-30 | Moderate Convergence | -5 |
| 31+ | Low Convergence | -8 |

---

## Installation

### Prerequisites

- Python 3.10+
- pip

### Setup

```bash
git clone <repository-url>
cd MCAV2
pip install -r requirements.txt
```

### Running the Main Application

```bash
streamlit run app/main.py
```

Opens a browser at `http://localhost:8501`.

### Docker

```bash
docker-compose up -d
```

Or use the Makefile:

```bash
make install    # Install dependencies
make build      # Build Docker image
make run        # Run in production mode
make dev        # Run in development mode
make stop       # Stop all services
```

---

## Usage

### Single Application Processing

1. Configure business parameters in the sidebar:
   - Company name, industry, company age
   - Directors credit score (0-100)
   - Requested loan amount
   - Risk factors (CCJs, defaults, web presence)

2. Upload transaction data (JSON format, Plaid export supported)

3. Review the unified recommendation at the top of the dashboard, then drill into detailed scoring, charts, and metrics below.

4. Export reports as HTML or CSV.

### Dashboard Layout

The dashboard is organised top-down by importance:

1. **Unified Recommendation** -- Decision (APPROVE/DECLINE/REFER), combined score, confidence percentage, contributing scores for MCA Rule (40%), Subprime (40%), ML Score (20%), convergence indicator, and expandable pricing guidance / risk factors / recommendations.
2. **Revenue Insights** -- Unique revenue sources, daily metrics, revenue active days.
3. **Charts and Analysis** -- Score comparison, financial metrics, monthly trends, threshold comparisons.
4. **Monthly Breakdown** -- Transaction counts and amounts by category, monthly tables.
5. **Transaction Analysis** -- Category distribution, transaction type breakdown, pattern identification.
6. **Loans and Debt Repayments** -- Existing debt analysis, repayment behaviour, debt stacking risk.
7. **Detailed Financial Metrics** -- All calculated metrics vs thresholds with pass/fail status.
8. **Detailed Scoring Analysis** -- Expandable section with subprime breakdown, metric performance, improvement suggestions, ML reliability assessment.

---

## Batch Processor

The batch processor (`MCAV2_BatchProcessor/`) is a standalone Streamlit application that processes multiple loan applications in one go.

### Running the Batch Processor

```bash
streamlit run MCAV2_BatchProcessor/batch_processor_standalone.py --server.port 8502
```

Opens at `http://localhost:8502` (separate from the main app on 8501).

### How It Works

1. **Upload a CSV** with application parameters. Required columns: `company_name`, `industry`, `directors_score`, `company_age_months`, `requested_loan`. Optional risk flag columns: `business_ccj`, `director_ccj`, `personal_default_12m`, etc.

2. **Upload a ZIP** containing one JSON transaction file per application. Filenames must match the `company_name` column in the CSV (fuzzy matching is used via `rapidfuzz`).

3. The batch processor then, for each application:
   - Extracts and categorises transactions from the JSON file
   - Calculates all financial metrics
   - Runs the Subprime scoring system
   - Runs the ML model (with feature clipping)
   - Evaluates MCA Rule (transparent transaction consistency checks)
   - Determines a final decision using MCA Rule overrides on top of the Subprime recommendation

4. **Results dashboard** shows:
   - Summary statistics (total processed, average scores, average revenue)
   - Score distributions (Subprime histogram, MCA Rule Score histogram)
   - Risk tier analysis (pie chart and cross-tabulation)
   - Detailed results table with all scores and key metrics
   - CSV export of all results

### Batch Processor Scoring

The batch processor uses the same scoring systems as the main app:
- Subprime scoring (tightened thresholds for batch use)
- ML scoring (same calibrated model, with feature clipping)
- MCA Rule evaluation (same `mca_scorecard_rules.py` engine)

The final decision follows this override logic:
- Base decision from Subprime recommendation
- MCA Rule DECLINE overrides everything (hard stop)
- MCA Rule REFER downgrades non-DECLINE decisions to REFER
- MCA Rule APPROVE does not override (no uplift)

### Batch Processor Files

| File | Purpose |
|------|---------|
| `batch_processor_standalone.py` | Main application (self-contained) |
| `company_matcher.py` | Company name fuzzy matching utility |
| `compare_with_files.py` | Debugging tool to compare CSV rows with JSON files |
| `debug_csv.py` | CSV inspection utility |
| `requirements.txt` | Dependencies (same as main app) |

---

## ML Model Training

### Overview

The ML model can be retrained when new outcome data becomes available. The process has two steps: building the training dataset, then training the model.

### Step 1: Build the Training Dataset

Prepare an Excel spreadsheet (`data/training_dataset.xlsx`) with these columns:

| Column | Required | Description |
|--------|----------|-------------|
| `company_name` | Yes | Must match the JSON filename in `data/JsonExport/` |
| `Outcome` | Yes | 1 = repaid, 0 = defaulted (leave blank for unfunded) |
| `Total Revenue` | Yes | Total revenue from bank statements |
| `Net Income` | Yes | Net income |
| `Operating Margin` | Yes | Operating margin ratio |
| `Debt Service Coverage Ratio` | Yes | DSCR |
| `requested_loan` | Yes | Loan amount requested |
| `directors_score` | No | Director credit score, defaults to 50 |
| `company_age_months` | No | Business age in months, defaults to 12 |
| `industry` | No | Industry name, defaults to Other |
| `total_debt` | No | Known debt amount, defaults to 0 |

Place the JSON transaction files in `data/JsonExport/`. Then run:

```bash
python build_training_dataset.py
```

This produces two output files:
- `data/mca_training_dataset.csv` -- MCA transaction features
- `data/ml_training_dataset.csv` -- 13 ML model features + outcome, ready for training

The script derives additional features from the transaction data: Cash Flow Volatility, Revenue Growth Rate, Average Month-End Balance, Negative Balance Days, and Bounced Payments. Where the spreadsheet provides a value (e.g., Total Revenue, DSCR), it uses that; where the value is blank or zero, it falls back to the transaction-derived estimate.

### Step 2: Train the Model

```bash
python train_improved_model.py --data "data/ml_training_dataset.csv"
```

This will:
1. Load and validate the training data
2. Train a Random Forest with regularised hyperparameters
3. Also train a Gradient Boosting model and compare them
4. Select the better model
5. Wrap it in `CalibratedClassifierCV` for well-calibrated probabilities
6. Report 5-fold stratified cross-validation metrics (ROC-AUC, accuracy)
7. Save `model.pkl` and `scaler.pkl` to `app/models/model_artifacts/`

The new model replaces the old one. Commit and push the updated `.pkl` files so all machines get the retrained model.

### Model Improvements Over Default

The training script addresses issues with naive Random Forest defaults:

| Parameter | Default RF | This Script | Why |
|-----------|-----------|-------------|-----|
| max_depth | 20 | 8 | Caps tree complexity for small datasets |
| min_samples_leaf | 1 | 5 | Prevents memorising single examples |
| min_samples_split | 2 | 10 | Requires meaningful support for splits |
| class_weight | None | balanced | Handles class imbalance |
| oob_score | False | True | Free validation metric |
| Calibration | None | CalibratedClassifierCV | Well-calibrated probabilities |

---

## Project Structure

```
MCAV2/
├── app/
│   ├── main.py                              # Main Streamlit application
│   ├── adaptive_score_calculation.py        # Adaptive scoring functions
│   ├── services/
│   │   ├── ensemble_scorer.py               # Unified ensemble scoring (40/40/20 weights)
│   │   ├── subprime_scoring_system.py       # Subprime scoring system
│   │   ├── advanced_metrics.py              # Advanced risk metrics
│   │   ├── data_processor.py                # Transaction processing
│   │   └── financial_analyzer.py            # Financial metrics calculation
│   ├── config/
│   │   ├── scoring_thresholds.py            # Centralized scoring thresholds
│   │   ├── settings.py                      # Application settings
│   │   └── industry_config.py               # Industry-specific configurations
│   ├── pages/
│   │   ├── scoring.py                       # Scoring calculation functions
│   │   ├── transactions.py                  # Transaction processing
│   │   ├── charts.py                        # Chart generation
│   │   └── reports.py                       # Report generation and export
│   ├── models/
│   │   ├── ml_predictor.py                  # ML prediction service
│   │   └── model_artifacts/
│   │       ├── model.pkl                    # Trained ML model (CalibratedClassifierCV)
│   │       └── scaler.pkl                   # Feature scaler (StandardScaler)
│   ├── utils/
│   │   ├── feature_alignment.py             # Feature mapping between systems
│   │   ├── weight_calibration.py            # ML weight extraction
│   │   └── chart_utils.py                   # Chart utilities
│   ├── core/
│   │   ├── exceptions.py                    # Custom exceptions
│   │   ├── validators.py                    # Input validation
│   │   ├── logger.py                        # Logging configuration
│   │   └── cache.py                         # Caching layer
│   └── components/
│       └── alerts.py                        # Alert components
├── MCAV2_BatchProcessor/
│   ├── batch_processor_standalone.py        # Batch processing application
│   ├── company_matcher.py                   # Fuzzy company name matching
│   ├── compare_with_files.py                # CSV/JSON comparison utility
│   ├── debug_csv.py                         # CSV debugging tool
│   └── requirements.txt                     # Batch processor dependencies
├── mca_scorecard_rules.py                   # MCA Rule engine
├── build_training_dataset.py                # Training data builder
├── train_improved_model.py                  # ML model training script
├── score_all_apps.py                        # Batch MCA rule scorer
├── requirements.txt                         # Python dependencies
├── docker-compose.yml                       # Docker production config
├── docker-compose.dev.yml                   # Docker development config
├── Makefile                                 # Build and run commands
└── README.md
```

### Key Files

| File | Purpose |
|------|---------|
| `app/main.py` | Main Streamlit application -- all dashboard UI, scoring orchestration, and export logic |
| `app/services/ensemble_scorer.py` | Combines MCA Rule (40%), Subprime (40%), and ML (20%) into a unified recommendation |
| `app/services/subprime_scoring_system.py` | Subprime risk-tier scoring with industry adjustments and penalties |
| `app/models/ml_predictor.py` | ML prediction service with feature clipping, confidence intervals, and explainability |
| `mca_scorecard_rules.py` | Transaction consistency rule engine (inflow days, gaps, volatility) |
| `app/config/scoring_thresholds.py` | Centralised threshold definitions for all scoring systems |
| `build_training_dataset.py` | Builds ML training data from transaction JSONs + application spreadsheet |
| `train_improved_model.py` | Trains and calibrates the ML model with regularised hyperparameters |

---

## Configuration

### Scoring Weights

Edit `app/services/ensemble_scorer.py`:

```python
DEFAULT_WEIGHTS = {
    'mca_score': 0.40,
    'subprime_score': 0.40,
    'ml_score': 0.20,
}
```

### Scoring Thresholds

Edit `app/config/scoring_thresholds.py` for metric thresholds, risk penalties, tier boundaries, and industry multipliers.

### Industry Thresholds

Industry-specific benchmarks for 25+ industries are defined in `app/config/industry_config.py` and in the batch processor's `INDUSTRY_THRESHOLDS` dictionary.

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `APP_NAME` | Business Finance Scorecard | Application title |
| `DEBUG` | false | Enable debug logging |
| `MODEL_PATH` | app/models/model_artifacts/model.pkl | ML model location |
| `SCALER_PATH` | app/models/model_artifacts/scaler.pkl | Scaler location |
| `LOG_LEVEL` | INFO | Logging level |
| `CACHE_TTL` | 3600 | Cache time-to-live in seconds |
| `PLAID_CLIENT_ID` | None | Plaid API client ID (optional) |
| `PLAID_SECRET` | None | Plaid API secret (optional) |
| `TRAINING_OUTCOMES_XLSX` | data/training_dataset.xlsx | Training spreadsheet path |
| `TRAINING_JSON_ROOT` | data/JsonExport | Training JSON directory |

---

## Transaction Categorisation

Transactions are automatically classified into:

| Category | Examples |
|----------|----------|
| Income | Stripe, SumUp, Square, PayPal, direct sales |
| Expenses | Utilities, payroll, supplies, rent |
| Loans | YouLend, Iwoca, Funding Circle, capital advances |
| Debt Repayments | Loan payments, financing charges |
| Special Inflows | Grants, refunds, investments |
| Special Outflows | Owner withdrawals, transfers |
| Failed Payments | Bounced, NSF, insufficient funds |

---

## Version History

### v2.3.0 (February 2026)
- Removed Weighted (5%) scoring system from ensemble
- New scoring weights: MCA Rule 40%, Subprime 40%, ML Score 20%
- Retrained ML model on 272 applications (0.922 ROC-AUC, up from no validation)
- Added probability calibration (CalibratedClassifierCV) to ML model
- Added feature clipping to prevent extrapolation on extreme values
- Updated convergence check to include all three scoring systems
- Updated build_training_dataset.py to produce ML-ready training data
- Added train_improved_model.py for reproducible model retraining
- Aligned batch processor with all scoring changes
- Comprehensive README update

### v2.2.0 (January 2026)
- Unified Ensemble Scoring combining 4 methods
- Dashboard reorganisation with decision at top
- Centralized thresholds configuration
- Modular refactoring (pages/, services/, utils/)
- Continuous scoring (partial credit instead of binary pass/fail)
- Improved ML confidence estimation

### v2.1.0 (December 2024)
- Tightened scoring thresholds for £1-10k lending
- Increased balance weight, reduced growth weight
- Target approval rate 15-25%

### v2.0.0
- Enhanced subprime scoring system
- Risk factor penalty integration
- Interactive dashboard with exports
- Docker deployment

### v1.0.0
- Basic financial analysis
- Simple threshold scoring

---

## License

This project is licensed under the MIT License.
