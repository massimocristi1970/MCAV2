# Business Finance Scorecard v2.1

An advanced business finance analysis and risk assessment platform designed for **short-term subprime business lending** (£1,000-£10,000 loans, 6-9 month terms). This application provides comprehensive financial analysis, multiple scoring methodologies, and risk-tiered lending recommendations.

## 🚀 Features

### Core Capabilities

- **Multiple Scoring Systems**: Subprime-optimized scoring, weighted threshold scoring, and ML-based predictions
- **Advanced Transaction Categorization**: Intelligent classification of financial transactions with 95%+ accuracy
- **Interactive Dashboard**: Real-time financial metrics visualization with Plotly charts
- **Industry Benchmarking**: Compare against 25+ industry-specific thresholds
- **Risk Assessment**: Comprehensive evaluation including CCJs, defaults, and operational factors
- **Export Functionality**: Generate HTML reports, JSON data exports, and CSV downloads
- **Batch Processing**: Process multiple loan applications simultaneously

### Specialized Features

- **Short-Term Subprime Focus**: Optimized for £1-10k loans with 6-9 month repayment terms
- **Cash Flow Stability Emphasis**: Prioritizes liquidity and payment ability over growth
- **Loans & Debt Analysis**: Detailed tracking of borrowing patterns and repayment behavior
- **Revenue Insights**: Source diversification and transaction pattern analysis
- **Tightened Risk Thresholds**: Calibrated for ~20% approval rate

## 📊 Scoring Methodologies

### 1. 🎯 Subprime Score (Primary)

- **Purpose**: Primary lending decision tool for short-term subprime market
- **Focus**: Debt service coverage (28%), cash balance (18%), and stability (12%)
- **Target Approval Rate**: ~15-25% (tightened thresholds)
- **Risk Factors**: Includes penalties for CCJs, defaults, and operational concerns

### 2. 🏛️ V2 Weighted Score (Secondary)

- **Purpose**: Binary threshold validation system
- **Method**: Pass/fail evaluation against industry benchmarks
- **Transparency**: Simple, explainable scoring methodology
- **Use Case**: Secondary validation and regulatory compliance

### 3. 🤖 ML Score (Future Use)

- **Purpose**: Machine learning predictions (currently for monitoring only)
- **Status**: Requires larger dataset for production use
- **Enhancement**: Includes growth business adjustments
- **Recommendation**: Not for lending decisions until sufficient training data

## 💰 Lending Parameters

This scoring system is calibrated for:

| Parameter                | Value            |
| ------------------------ | ---------------- |
| **Loan Range**           | £1,000 - £10,000 |
| **Typical Test Amount**  | £2,000           |
| **Factor Rate**          | 1.8x             |
| **Term Length**          | 6-9 months       |
| **Target Approval Rate** | ~15-25%          |

## 📈 Scoring Breakdown (v2.1 - Tightened)

### Subprime Score Components

| Component                       | Weight | Rationale                                        |
| ------------------------------- | ------ | ------------------------------------------------ |
| **Debt Service Coverage Ratio** | 28%    | Primary ability to service debt payments         |
| **Average Month-End Balance**   | 18%    | Critical liquidity buffer for short-term loans   |
| **Directors Score**             | 16%    | Personal reliability and creditworthiness        |
| **Cash Flow Volatility**        | 12%    | Stability crucial for 6-9 month repayment        |
| **Revenue Growth Rate**         | 10%    | Reduced weight - less relevant for short terms   |
| **Operating Margin**            | 6%     | Current profitability (tolerant of small losses) |
| **Net Income**                  | 4%     | Overall profitability indicator                  |
| **Negative Balance Days**       | 4%     | Cash flow management capability                  |
| **Company Age**                 | 2%     | Business maturity (minimal weight)               |

### Tightened Thresholds (v2.1)

| Metric                         | Old Threshold | New Threshold | Impact                            |
| ------------------------------ | ------------- | ------------- | --------------------------------- |
| **Minimum Balance**            | £500          | £1,500        | Must cover ~3 months payments     |
| **Maximum Volatility**         | 1.0 (100%)    | 0.6 (60%)     | Requires stable cash flow         |
| **Minimum DSCR for points**    | 1.0           | 1.3           | Must have margin for loan payment |
| **Tier 1 Score**               | 75+           | 82+           | Premium tier harder to reach      |
| **Tier 2 Score (Approve)**     | 60+           | 70+           | Standard approval tightened       |
| **Tier 3 Score (Conditional)** | 45+           | 55+           | Conditional approval tightened    |
| **Tier 4 Score (Review)**      | 30+           | 40+           | More applications decline         |

### Risk Factor Penalties

| Risk Factor                | Penalty    | Notes                                 |
| -------------------------- | ---------- | ------------------------------------- |
| **Business CCJ**           | -12 points | Severe - business litigation risk     |
| **Personal Default (12m)** | -8 points  | High - personal credit crucial        |
| **Director CCJ**           | -8 points  | High - director financial issues      |
| **No Online Presence**     | -4 points  | Moderate - business viability concern |
| **Outdated Web Presence**  | -3 points  | Minor - operational concerns          |
| **Generic Email**          | -2 points  | Minor - professionalism indicator     |

## 🎯 Risk Tiers & Pricing (v2.1)

### Tier 1: Premium Subprime (82+ score)

- **Requirements**: Score 82+, DSCR ≥2.5, Growth ≥15%, Directors ≥75, Volatility ≤0.3
- **Rate**: 1.5-1.6 factor rate
- **Loan Multiple**: 4x monthly revenue (max £10k)
- **Term**: 6-12 months
- **Monitoring**: Monthly reviews

### Tier 2: Standard Subprime (70-82 score)

- **Requirements**: Score 70+, DSCR ≥2.0, Volatility ≤0.45
- **Rate**: 1.7-1.85 factor rate
- **Loan Multiple**: 3x monthly revenue (max £8k)
- **Term**: 6-9 months
- **Monitoring**: Bi-weekly reviews

### Tier 3: High-Risk Subprime (55-70 score)

- **Requirements**: Score 55+, DSCR ≥1.5, Directors ≥55, Volatility ≤0.55
- **Rate**: 1. 85-2.0 factor rate
- **Loan Multiple**: 2. 5x monthly revenue (max £5k)
- **Term**: 4-6 months
- **Monitoring**: Weekly reviews

### Tier 4: Enhanced Monitoring (40-55 score)

- **Requirements**: Score 40+, DSCR ≥1. 3
- **Rate**: 2.0-2.2+ factor rate
- **Loan Multiple**: 2x monthly revenue (max £3k)
- **Term**: 3-6 months
- **Monitoring**: Weekly reviews + daily balance + **Personal Guarantees REQUIRED**

### Decline (<40 score)

- Applications below 40 score are declined
- Risk profile exceeds acceptable parameters for short-term subprime lending

## 🏗️ Architecture

```
MCAV2/
├── app/
│   ├── main.py                          # Main Streamlit application (single app)
│   ├── services/
│   │   ├── subprime_scoring_system.py   # Tightened subprime scoring (v2.1)
│   │   ├── data_processor.py            # Transaction processing
│   │   └── financial_analyzer.py        # Financial metrics calculation
│   ├── core/
│   │   ├── exceptions.py                # Custom exception handling
│   │   ├── validators.py                # Input validation
│   │   ├── logger.py                    # Logging configuration
│   │   └── cache.py                     # Caching utilities
│   ├── config/
│   │   ├── settings.py                  # Application settings
│   │   └── industry_config.py           # Industry thresholds
│   └── utils/
│       └── chart_utils.py               # Interactive chart generation
├── MCAV2_BatchProcessor/
│   └── batch_processor_standalone.py    # Batch processing application (v2.1)
├── model. pkl                            # ML model (scikit-learn)
├── scaler.pkl                           # Feature scaler
├── requirements.txt                     # Python dependencies
├── Dockerfile                           # Container configuration
├── docker-compose.yml                   # Multi-service deployment
└── README.md                            # This file
```

## 🔧 Installation

### Prerequisites

- Python 3.9+
- pip package manager
- 4GB+ RAM recommended

### Quick Start

1. **Clone the repository**

```bash
git clone <repository-url>
cd MCAV2
```

2.  **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the single application processor**

```bash
streamlit run app/main.py
```

4. **Run the batch processor**

```bash
streamlit run MCAV2_BatchProcessor/batch_processor_standalone. py --server.port 8502
```

The applications will be available at:

- Single app: `http://localhost:8501`
- Batch processor: `http://localhost:8502`

### Windows Quick Start

```batch
# Run the provided batch file
run_app.bat
```

## 📋 Usage Guide

### Single Application Processing (main.py)

1. **Business Parameters Setup**

   - Company Information: Name, industry, age, requested loan amount
   - Director Details: Credit score (0-100)
   - Risk Factors: CCJs, defaults, web presence, etc.

2. **Data Upload**

   - Upload JSON transaction data (Plaid format supported)
   - Automatic transaction categorization
   - Period filtering (3, 6, 9, 12 months or all data)

3. **Analysis Results**
   - Primary Scores: Subprime (primary), V2 Weighted, ML predictions
   - Financial Metrics: 15+ key performance indicators
   - Risk Assessment: Comprehensive risk evaluation
   - Recommendations: Tier-based lending guidance

### Batch Processing (batch_processor_standalone.py)

1. **Upload CSV Parameter File**

   - Contains company-specific parameters (industry, directors score, loan amount, etc.)
   - Fuzzy matching links CSV rows to JSON transaction files

2. **Upload Transaction Files**

   - Multiple JSON files or ZIP archive
   - Automatic company name extraction and matching

3. **Process & Analyze**
   - Batch scoring with tightened thresholds
   - Comprehensive debug information
   - Export results to CSV

### CSV Parameter File Format

```csv
company_name,industry,directors_score,requested_loan,company_age_months,personal_default_12m,business_ccj,director_ccj
ABC Manufacturing Ltd,Manufacturing,78,5000,24,FALSE,FALSE,FALSE
Smith's Restaurant,Restaurants and Cafes,65,3000,18,FALSE,FALSE,FALSE
Tech Solutions UK,IT Services and Support Companies,82,8000,36,FALSE,FALSE,FALSE
```

## 🎯 Industry Support

The application supports 25+ industries with specific risk profiles:

**Low Risk Sectors (multiplier: 1.05-1.1x):**

- Medical Practices (GPs, Clinics, Dentists)
- IT Services and Support Companies
- Business Consultants
- Education
- Engineering

**Standard Risk Sectors (multiplier: 1.0x):**

- Manufacturing, Retail, Food Service
- Professional Services
- E-commerce and Technology

**Higher Risk Sectors (multiplier: 0.8-0.9x):**

- Restaurants and Cafes
- Construction Firms
- Bars and Pubs
- Beauty Salons and Spas
- Event Planning and Management Firms

## 📊 Key Metrics Calculated

### Financial Performance

- **Revenue Metrics**: Total revenue, monthly average, growth rate
- **Profitability**: Operating margin, net income, gross burn rate
- **Debt Management**: DSCR, debt-to-income ratio, repayment tracking
- **Cash Flow**: Volatility, negative balance days, average balances

### Risk Indicators

- **Payment Reliability**: Bounced payments, failed transactions
- **Stability Metrics**: Cash flow consistency, seasonal patterns
- **Growth Analysis**: Revenue trajectory (reduced weight in v2.1)

## 🔍 Transaction Categorization

The system automatically categorizes transactions into:

- **Income**: Payment processors (Stripe, SumUp, Square, etc.), direct revenue, sales
- **Expenses**: Operational costs, payroll, utilities
- **Loans**: Business financing, advances, capital injections (YouLend, Iwoca, etc.)
- **Debt Repayments**: Loan payments, debt service
- **Special Inflows**: Grants, refunds, investments
- **Special Outflows**: Withdrawals, transfers, investments
- **Failed Payments**: Bounced transactions, insufficient funds

## ⚠️ Known Issues

### scikit-learn Version Warning

You may see this warning when running the application:

```
Trying to unpickle estimator StandardScaler from version 1.6.1 when using version 1.7.0
```

**Resolution**: This is a warning, not an error. The ML score is secondary to the subprime score. To fix permanently:

```python
import joblib
model = joblib.load('model. pkl')
scaler = joblib.load('scaler.pkl')
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
```

## 📦 Dependencies

### Core Framework

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing

### Visualization

- **Plotly**: Interactive charts and graphs

### Machine Learning

- **Scikit-learn**: ML algorithms and utilities (v1.6.1+ recommended)
- **Joblib**: Model serialization

### Additional Features

- **RapidFuzz**: Fuzzy text matching for company name matching
- **Python-dotenv**: Environment configuration

## 📊 Version History

### v2.1.0 (Current - December 2024)

- **TIGHTENED scoring thresholds** for £1-10k short-term lending
- Increased balance weight (12% → 18%)
- Reduced growth weight (20% → 10%)
- Increased volatility weight (8% → 12%)
- Raised minimum balance threshold (£500 → £1,500)
- Lowered maximum volatility tolerance (1.0 → 0.6)
- Raised tier score thresholds (Tier 1: 75→82, Tier 2: 60→70, etc.)
- Target approval rate reduced from ~80% to ~15-25%
- Synchronized scoring between main app and batch processor

### v2.0.0

- Enhanced subprime scoring system
- Risk factor penalty integration
- Advanced transaction categorization
- Interactive dashboard with export functionality
- Docker deployment support
- Batch processing capability

### v1.0.0

- Basic financial analysis
- Simple threshold scoring
- Manual transaction categorization

## 🛠️ Development

### Running Tests

```bash
python -m pytest tests/ -v --cov=app
```

### Code Quality

```bash
# Linting
flake8 app/

# Type checking
mypy app/

# Code formatting
black app/
isort app/
```

### Modifying Scoring Thresholds

To adjust scoring thresholds, edit both files to maintain consistency:

1. `app/services/subprime_scoring_system.py` - Single application scoring
2. `MCAV2_BatchProcessor/batch_processor_standalone.py` - Batch processing scoring

Key sections to modify:

- `__init__`: Weight allocations and threshold definitions
- `_calculate_base_subprime_score` / `_calculate_tightened_base_score`: Score calculations
- `_determine_risk_tier` / `_determine_tightened_risk_tier`: Tier boundaries

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Built for short-term subprime business lending analysis**

_Calibrated for £1-10k loans with 1.8x factor rate and 6-9 month terms_
