# Business Finance Scorecard v2.2

An advanced business finance analysis and risk assessment platform designed for **short-term subprime business lending** (£1,000-£10,000 loans, 6-9 month terms). This application provides comprehensive financial analysis, **unified ensemble scoring**, and risk-tiered lending recommendations.

## 🚀 Key Features

### Core Capabilities

- **Unified Ensemble Scoring**: Combines 4 scoring methodologies into a single, weighted recommendation
- **Multiple Scoring Systems**: MCA Rules, Subprime-optimized, V2 Weighted, and ML-based predictions
- **Advanced Transaction Categorization**: Intelligent classification with 95%+ accuracy
- **Interactive Dashboard**: Real-time metrics visualization with Plotly charts
- **Industry Benchmarking**: 25+ industry-specific thresholds
- **Risk Assessment**: Comprehensive evaluation including CCJs, defaults, and operational factors
- **Export Functionality**: HTML reports, JSON exports, and CSV downloads
- **Batch Processing**: Process multiple loan applications simultaneously

### Recent Enhancements (v2.2)

- **🎯 Unified Recommendation System**: Single decision combining all scoring methods
- **📊 Streamlined Dashboard**: Clean hierarchy with decision at top
- **⚙️ Centralized Thresholds**: All scoring parameters in one configuration
- **🔄 Modular Architecture**: Refactored codebase for maintainability
- **📈 Continuous Scoring**: Partial credit instead of binary pass/fail
- **🤖 Improved ML Confidence**: Better uncertainty estimation

---

## 📊 Scoring & Decisioning Process

### Overview

The application uses an **Ensemble Scoring** approach that combines four different scoring methodologies, each with specific weights based on their predictive power for MCA lending outcomes:

```
┌─────────────────────────────────────────────────────────────────┐
│                    UNIFIED RECOMMENDATION                        │
│                                                                  │
│   Combined Score = (MCA × 35%) + (Subprime × 35%)               │
│                  + (ML × 25%) + (Weighted × 5%)                  │
│                                                                  │
│   Decision: APPROVE | CONDITIONAL | REFER | SENIOR_REVIEW | DECLINE
└─────────────────────────────────────────────────────────────────┘
         ▲              ▲              ▲              ▲
         │              │              │              │
    ┌────┴────┐   ┌────┴────┐   ┌────┴────┐   ┌────┴────┐
    │MCA Rule │   │Subprime │   │   ML    │   │Weighted │
    │  (35%)  │   │  (35%)  │   │  (25%)  │   │  (5%)   │
    │         │   │         │   │         │   │         │
    │Cashflow │   │  Risk   │   │ Pattern │   │ Binary  │
    │Consist. │   │  Tiers  │   │ Predict │   │Threshold│
    └─────────┘   └─────────┘   └─────────┘   └─────────┘
```

### 1. MCA Rule Score (35% Weight)

**Purpose**: Assesses transaction consistency required to sustainably support a Merchant Cash Advance.

**Why 35% Weight**: Analysis of historic MCA outcomes showed that **revenue consistency** (not just total revenue) is one of the strongest differentiators between businesses that repay smoothly vs. those that default.

**Core Signals Evaluated**:

| Signal | What It Measures | Why It Matters |
|--------|------------------|----------------|
| **Inflow Days (30d)** | Days with positive inflows | Trading regularity |
| **Maximum Gap** | Longest period without inflows | Business continuity risk |
| **Inflow Volatility** | Stability of inflow amounts (CV) | Predictability of cashflow |

**Decision Logic**:
```
IF insufficient_data → DECLINE (data quality gate)
IF inflow_days < 8 AND max_gap > 14 → DECLINE (severe consistency failure)
IF volatility > 2.0 → DECLINE (extremely erratic)
IF score >= 70 AND no_hard_stops → APPROVE
IF score >= 50 → REFER
ELSE → DECLINE
```

### 2. Subprime Score (35% Weight)

**Purpose**: Primary lending decision tool optimized for short-term subprime market (£1-10k, 6-9 months).

**Why 35% Weight**: Designed specifically for micro-enterprise assessment with industry-specific adjustments and risk penalties.

**Component Weights**:

| Component | Weight | Rationale |
|-----------|--------|-----------|
| Debt Service Coverage Ratio | 28% | Primary ability to service payments |
| Average Month-End Balance | 18% | Critical liquidity buffer |
| Directors Score | 16% | Personal creditworthiness |
| Cash Flow Volatility | 12% | Stability for repayment |
| Revenue Growth Rate | 10% | Business trajectory |
| Operating Margin | 6% | Current profitability |
| Net Income | 4% | Overall profitability |
| Negative Balance Days | 4% | Cash management |
| Company Age | 2% | Business maturity |

**Risk Tier Classification**:

| Tier | Score Range | Decision | Factor Rate |
|------|-------------|----------|-------------|
| Tier 1 (Premium) | 65+ | APPROVE | 1.5-1.6x |
| Tier 2 (Standard) | 50-65 | APPROVE | 1.7-1.85x |
| Tier 3 (Conditional) | 40-50 | CONDITIONAL | 1.85-2.0x |
| Tier 4 (High Risk) | 30-40 | SENIOR_REVIEW | 2.0-2.2x |
| Decline | <30 | DECLINE | N/A |

**Risk Factor Penalties**:

| Risk Factor | Penalty | Impact |
|-------------|---------|--------|
| Business CCJ | -12 pts | Severe litigation risk |
| Personal Default (12m) | -8 pts | Personal credit crucial |
| Director CCJ | -8 pts | Director financial issues |
| No Online Presence | -4 pts | Business viability concern |
| Outdated Web Presence | -3 pts | Operational concerns |
| Generic Email | -2 pts | Professionalism indicator |

### 3. ML Score (25% Weight)

**Purpose**: Machine learning probability prediction based on historical loan outcomes.

**Why 25% Weight**: Data-driven predictions, but requires larger dataset for full reliability. Currently supports decision-making but shouldn't override rule-based systems.

**Features Used**:
- Directors Score
- Total Revenue & Debt
- Debt-to-Income Ratio
- Operating Margin
- DSCR
- Cash Flow Volatility
- Revenue Growth Rate
- Average Month-End Balance
- Negative Balance Days
- Bounced Payments
- Company Age
- Sector Risk

**Confidence Calculation**:
- Uses probability margin (distance from 50%)
- Adjusts for prediction extremity
- Reports confidence intervals

### 4. V2 Weighted Score (5% Weight)

**Purpose**: Simple binary threshold validation system.

**Why 5% Weight**: Analysis showed binary pass/fail approach is less predictive than continuous scoring, but provides useful validation.

**Method**: Each metric evaluated against industry benchmark:
- **Pass**: Full points if value exceeds threshold
- **Partial Credit**: Proportional points based on how close to threshold
- **Fail**: Zero points if significantly below threshold

---

## 🎯 Unified Decision Process

### Decision Hierarchy

```
1. HARD STOPS (immediate decline):
   - MCA Rule = DECLINE
   - Subprime Tier = "Decline"
   - Combined Score < 25
   - Business CCJ present

2. SCORE-BASED DECISION:
   Combined Score 70+ → APPROVE
   Combined Score 55-70 → CONDITIONAL_APPROVE
   Combined Score 40-55 → REFER
   Combined Score 30-40 → SENIOR_REVIEW
   Combined Score <30 → DECLINE

3. CONVERGENCE CHECK:
   - High convergence (all methods agree): Increases confidence
   - Low convergence (methods disagree): Triggers REFER/SENIOR_REVIEW
```

### Score Convergence

The system evaluates how well the four scoring methods agree:

| Convergence | Std Dev | Meaning |
|-------------|---------|---------|
| High | ≤10 | All methods strongly agree |
| Good | ≤15 | Methods mostly align |
| Moderate | ≤20 | Some disagreement |
| Low | >20 | Significant disagreement - review required |

### Pricing Guidance

Based on combined score and tier:

| Score Range | Factor Rate | Max Term | Max Multiple | Collection |
|-------------|-------------|----------|--------------|------------|
| 70+ | 1.35-1.45 | 12 months | 4x monthly rev | Weekly |
| 55-70 | 1.45-1.55 | 9 months | 3x monthly rev | Weekly |
| 40-55 | 1.55-1.65 | 6 months | 2x monthly rev | Daily |
| 30-40 | Case-by-case | 3-6 months | 1.5x monthly rev | Daily |

---

## 📱 Dashboard Layout

The dashboard presents information in a clear hierarchy:

### 1. Unified Recommendation (Top)
- **Primary Decision**: APPROVE/DECLINE/REFER with confidence %
- **Combined Score**: 0-100 with contributing breakdown
- **4 Score Metrics**: MCA (35%), Subprime (35%), ML (25%), Weighted (5%)
- **Convergence Indicator**: How well methods agree
- **Expandable**: Pricing guidance, risk factors, recommendations

### 2. Revenue Insights
- Unique revenue sources
- Average transactions per day
- Daily revenue metrics
- Revenue active days

### 3. Charts & Analysis
- Score comparison charts
- Financial metrics visualization
- Monthly trend analysis
- Threshold comparison charts

### 4. Monthly Breakdown
- Transaction counts by category
- Transaction amounts by category
- Detailed monthly tables

### 5. Transaction Analysis
- Category distribution
- Transaction type breakdown
- Pattern identification

### 6. Loans & Debt Repayments
- Existing debt analysis
- Repayment behavior tracking
- Debt stacking risk assessment

### 7. Detailed Financial Metrics
- All calculated metrics vs thresholds
- Pass/fail status for each

### 8. Detailed Scoring Analysis (Expandable)
- Subprime score breakdown
- V2 Weighted component analysis
- Metric performance table
- Top risk/strength factors
- Improvement suggestions
- ML reliability assessment

---

## 🏗️ Architecture

```
MCAV2/
├── app/
│   ├── main.py                              # Main Streamlit application
│   ├── services/
│   │   ├── ensemble_scorer.py               # 🆕 Unified scoring system
│   │   ├── subprime_scoring_system.py       # Subprime scoring (v2.2)
│   │   ├── advanced_metrics.py              # 🆕 Advanced risk metrics
│   │   ├── data_processor.py                # Transaction processing
│   │   └── financial_analyzer.py            # Financial metrics
│   ├── config/
│   │   ├── scoring_thresholds.py            # 🆕 Centralized thresholds
│   │   ├── settings.py                      # Application settings
│   │   └── industry_config.py               # Industry configurations
│   ├── pages/
│   │   ├── scoring.py                       # 🆕 Modular scoring functions
│   │   ├── transactions.py                  # 🆕 Transaction processing
│   │   ├── charts.py                        # 🆕 Chart generation
│   │   └── reports.py                       # 🆕 Report generation
│   ├── utils/
│   │   ├── feature_alignment.py             # 🆕 Feature mapping
│   │   ├── weight_calibration.py            # 🆕 ML weight extraction
│   │   └── chart_utils.py                   # Chart utilities
│   └── core/
│       ├── exceptions.py                    # Custom exceptions
│       ├── validators.py                    # Input validation
│       ├── logger.py                        # Logging
│       └── cache.py                         # Caching
├── MCAV2_BatchProcessor/
│   └── batch_processor_standalone.py        # Batch processing (standalone)
├── mca_scorecard_rules.py                   # MCA rule engine
├── build_training_dataset.py                # Training data builder
├── score_all_apps.py                        # Batch MCA scorer
├── model.pkl                                # ML model
├── scaler.pkl                               # Feature scaler
├── requirements.txt                         # Dependencies
├── Dockerfile                               # Container config
├── docker-compose.yml                       # Multi-service deployment
└── README.md                                # This file
```

### Key New Components

#### `ensemble_scorer.py`
Combines all scoring methods with configurable weights:
```python
SCORING_WEIGHTS = {
    'mca_rule': 0.35,      # Transaction consistency
    'subprime': 0.35,      # Risk-tier assessment
    'ml': 0.25,            # Pattern prediction
    'adaptive': 0.05       # Threshold validation
}
```

#### `scoring_thresholds.py`
Centralized threshold configuration:
```python
THRESHOLDS = ScoringThresholds(
    dscr=MetricThreshold(full_points=1.5, tiers=[(1.3, 0.8), (1.0, 0.5)]),
    operating_margin=MetricThreshold(full_points=0.15, tiers=[(0.05, 0.7), (0.0, 0.4)]),
    # ... all metrics defined here
)
```

#### `advanced_metrics.py`
Calculates additional risk signals:
- Deposit frequency score
- Revenue concentration (HHI)
- Seasonality coefficient
- Days since last NSF
- Balance trend
- Debt stacking risk

---

## 🔧 Installation

### Prerequisites
- Python 3.9+
- pip package manager
- 4GB+ RAM recommended

### Quick Start

```bash
# Clone repository
git clone <repository-url>
cd MCAV2

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app/main.py
```

### Docker Deployment

```bash
# Build and run
docker-compose up -d

# Or use make commands
make build
make run
```

---

## 📋 Usage Guide

### Single Application Processing

1. **Configure Business Parameters**
   - Company name, industry, age
   - Directors credit score (0-100)
   - Requested loan amount
   - Risk factors (CCJs, defaults, web presence)

2. **Upload Transaction Data**
   - JSON format (Plaid export supported)
   - Automatic categorization
   - Period filtering available

3. **Review Analysis**
   - Unified recommendation at top
   - Drill into detailed scoring
   - Export reports

### Batch Processing

```bash
# Run batch processor on separate port
streamlit run MCAV2_BatchProcessor/batch_processor_standalone.py --server.port 8502
```

### MCA Batch Scoring

```bash
# Score all applications in JsonExport folder
python score_all_apps.py
```

---

## 🔍 Transaction Categorization

Automatic classification into:

| Category | Examples |
|----------|----------|
| **Income** | Stripe, SumUp, Square, PayPal, direct sales |
| **Expenses** | Utilities, payroll, supplies, rent |
| **Loans** | YouLend, Iwoca, Funding Circle, capital |
| **Debt Repayments** | Loan payments, financing charges |
| **Special Inflows** | Grants, refunds, investments |
| **Special Outflows** | Owner withdrawals, transfers |
| **Failed Payments** | Bounced, NSF, insufficient funds |

---

## 📊 Key Metrics

### Financial Performance
- Total Revenue, Monthly Average, Growth Rate
- Operating Margin, Net Income
- DSCR, Debt-to-Income Ratio
- Cash Flow Volatility, Negative Balance Days

### Consistency Signals (MCA-specific)
- Inflow frequency (days with deposits)
- Maximum gap between inflows
- Inflow amount volatility (CV)

### Risk Indicators
- Bounced payments count
- Failed transaction rate
- Seasonal patterns
- Revenue concentration

---

## 📈 Version History

### v2.2.0 (Current - January 2026)
- **🎯 Unified Ensemble Scoring**: Combined recommendation from 4 methods
- **📊 Dashboard Reorganization**: Decision at top, consolidated details
- **⚙️ Centralized Thresholds**: Single source of truth for all scoring
- **🔄 Modular Refactoring**: Extracted pages/, services/, utils/ modules
- **📈 Continuous Scoring**: Partial credit instead of binary pass/fail
- **🤖 Improved ML Confidence**: Better uncertainty estimation
- **🐛 Bug Fixes**: Subprime penalty calculation, tier alignment

### v2.1.0 (December 2024)
- Tightened scoring thresholds for £1-10k lending
- Increased balance weight (12% → 18%)
- Reduced growth weight (20% → 10%)
- Target approval rate ~15-25%

### v2.0.0
- Enhanced subprime scoring system
- Risk factor penalty integration
- Interactive dashboard with exports
- Docker deployment support

### v1.0.0
- Basic financial analysis
- Simple threshold scoring

---

## 🛠️ Development

### Modifying Scoring Weights

Edit `app/services/ensemble_scorer.py`:
```python
SCORING_WEIGHTS = {
    'mca_rule': 0.35,
    'subprime': 0.35,
    'ml': 0.25,
    'adaptive': 0.05
}
```

### Modifying Thresholds

Edit `app/config/scoring_thresholds.py`:
```python
THRESHOLDS = ScoringThresholds(
    dscr=MetricThreshold(full_points=1.5, ...),
    # modify values here
)
```

### Running Tests

```bash
python -m pytest tests/ -v --cov=app
```

### Code Quality

```bash
flake8 app/
mypy app/
black app/
```

---

## 📄 License

This project is licensed under the MIT License.

---

**Built for short-term subprime business lending analysis**

*Unified ensemble scoring for £1-10k loans with 6-9 month terms*

*Combines MCA consistency rules, subprime risk tiers, ML predictions, and threshold validation*
