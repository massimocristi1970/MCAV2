# Business Finance Scorecard v2.0

An advanced business finance analysis and risk assessment platform designed for subprime business lending decisions. This application provides comprehensive financial analysis, multiple scoring methodologies, and detailed risk assessment capabilities.

## 🚀 Features

### Core Capabilities
- **Multiple Scoring Systems**: Subprime-optimized scoring, weighted threshold scoring, and ML-based predictions
- **Advanced Transaction Categorization**: Intelligent classification of financial transactions with 95%+ accuracy
- **Interactive Dashboard**: Real-time financial metrics visualization with Plotly charts
- **Industry Benchmarking**: Compare against 25+ industry-specific thresholds
- **Risk Assessment**: Comprehensive evaluation including CCJs, defaults, and operational factors
- **Export Functionality**: Generate HTML reports, JSON data exports, and CSV downloads

### Specialized Features
- **Subprime Lending Focus**: Optimized for growth businesses with temporary losses
- **Loans & Debt Analysis**: Detailed tracking of borrowing patterns and repayment behavior
- **Revenue Insights**: Source diversification and transaction pattern analysis
- **Seasonal Analysis**: Monthly and quarterly performance trending
- **Growth Business Adjustments**: Enhanced scoring for high-growth, low-margin businesses

## 📊 Scoring Methodologies

### 1. 🎯 Subprime Score (Primary)
- **Purpose**: Primary lending decision tool for subprime market
- **Focus**: Debt service coverage ratio (28% weight) and revenue growth (20% weight)
- **Risk Tolerance**: Accepts temporary losses if growth trajectory is strong
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

## 🏗️ Architecture

```
business_finance_app/
├── app/
│   ├── main.py                     # Main Streamlit application
│   ├── services/
│   │   ├── subprime_scoring_system.py    # Enhanced subprime scoring
│   │   ├── data_processor.py             # Transaction processing
│   │   └── financial_analyzer.py         # Financial metrics calculation
│   ├── core/
│   │   ├── exceptions.py           # Custom exception handling
│   │   ├── validators.py           # Input validation
│   │   ├── logger.py              # Logging configuration
│   │   └── cache.py               # Caching utilities
│   ├── config/
│   │   ├── settings.py            # Application settings
│   │   └── industry_config.py     # Industry thresholds
│   └── utils/
│       └── chart_utils.py         # Interactive chart generation
├── requirements.txt               # Python dependencies
├── Dockerfile                    # Container configuration
├── docker-compose.yml           # Multi-service deployment
└── README.md                    # This file
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
cd business-finance-scorecard
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. **Run the application**
```bash
streamlit run app/main.py
```

The application will be available at `http://localhost:8501`

### Docker Deployment

1. **Build and run with Docker Compose**
```bash
docker-compose up --build
```

2. **For development with hot reload**
```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

### Windows Quick Start
```batch
# Run the provided batch file
run_app.bat
```

## 📋 Usage Guide

### 1. Business Parameters Setup
- **Company Information**: Name, industry, age, requested loan amount
- **Director Details**: Credit score (0-100)
- **Risk Factors**: CCJs, defaults, web presence, etc.

### 2. Data Upload
- Upload JSON transaction data (Plaid format supported)
- Automatic transaction categorization
- Period filtering (3, 6, 9, 12 months or all data)

### 3. Analysis Results
- **Primary Scores**: Subprime, V2 Weighted, ML predictions
- **Financial Metrics**: 15+ key performance indicators
- **Risk Assessment**: Comprehensive risk evaluation
- **Recommendations**: Tier-based lending guidance

### 4. Export Options
- **HTML Report**: Professional lending report
- **JSON Data**: Complete analysis dataset
- **CSV Files**: Transaction data and financial metrics

## 🎯 Industry Support

The application supports 25+ industries with specific risk profiles:

**Low Risk Sectors:**
- Medical Practices (GPs, Clinics, Dentists)
- IT Services and Support Companies
- Business Consultants
- Education
- Engineering

**Standard Risk Sectors:**
- Retail, Manufacturing, Food Service
- Professional Services
- E-commerce and Technology

**Higher Risk Sectors:**
- Restaurants and Cafes
- Construction Firms
- Bars and Pubs
- Beauty Salons and Spas

## 📊 Key Metrics Calculated

### Financial Performance
- **Revenue Metrics**: Total revenue, monthly average, growth rate
- **Profitability**: Operating margin, net income, gross burn rate
- **Debt Management**: DSCR, debt-to-income ratio, repayment tracking
- **Cash Flow**: Volatility, negative balance days, average balances

### Risk Indicators
- **Payment Reliability**: Bounced payments, failed transactions
- **Stability Metrics**: Cash flow consistency, seasonal patterns
- **Growth Analysis**: Revenue trajectory, customer concentration

## ⚙️ Configuration

### Environment Variables
```env
# Application Configuration
APP_NAME=Business Finance Scorecard
APP_VERSION=2.0.0
DEBUG=False
LOG_LEVEL=INFO

# Optional: Plaid Integration
PLAID_CLIENT_ID=your_plaid_client_id
PLAID_SECRET=your_plaid_secret
PLAID_ENV=production

# Security
SECRET_KEY=your_secret_key

# Cache Configuration
CACHE_TTL=3600
```

### Industry Threshold Customization
Edit `app/config/industry_config.py` to modify industry-specific thresholds:
```python
INDUSTRY_THRESHOLDS = {
    'Your Industry': {
        'Debt Service Coverage Ratio': 1.50,
        'Operating Margin': 0.08,
        'Revenue Growth Rate': 0.05,
        # ... other thresholds
    }
}
```

## 🔍 Transaction Categorization

The system automatically categorizes transactions into:

- **Income**: Payment processors, direct revenue, sales
- **Expenses**: Operational costs, payroll, utilities
- **Loans**: Business financing, advances, capital injections
- **Debt Repayments**: Loan payments, debt service
- **Special Inflows**: Grants, refunds, investments
- **Special Outflows**: Withdrawals, transfers, investments
- **Failed Payments**: Bounced transactions, insufficient funds

## 📈 Scoring Breakdown

### Subprime Score Components
- **Debt Service Coverage Ratio** (28%): Primary ability to service debt
- **Revenue Growth Rate** (20%): Business trajectory and momentum
- **Directors Score** (16%): Personal reliability assessment
- **Average Month-End Balance** (12%): Liquidity buffer
- **Cash Flow Volatility** (8%): Financial stability measure
- **Operating Margin** (6%): Current profitability (tolerant of losses)
- **Net Income** (4%): Overall profitability
- **Negative Balance Days** (4%): Cash flow management
- **Company Age** (2%): Business maturity (minimal weight)

### Risk Factor Penalties
- **Business CCJ**: -12 points
- **Personal Default (12m)**: -8 points
- **Director CCJ**: -8 points
- **No Online Presence**: -4 points
- **Outdated Web Presence**: -3 points
- **Generic Email**: -2 points

## 🎯 Risk Tiers & Pricing

### Tier 1: Premium Subprime (75+ score)
- **Rate**: 1.4-1.5 factor rate
- **Loan Multiple**: 6x monthly revenue
- **Term**: 12-24 months
- **Monitoring**: Quarterly reviews

### Tier 2: Standard Subprime (60-75 score)
- **Rate**: 1.5-1.6 factor rate
- **Loan Multiple**: 4x monthly revenue
- **Term**: 6-18 months
- **Monitoring**: Monthly reviews

### Tier 3: High-Risk Subprime (45-60 score)
- **Rate**: 1.6-1.75 factor rate
- **Loan Multiple**: 3x monthly revenue
- **Term**: 6-12 months
- **Monitoring**: Bi-weekly reviews

### Tier 4: Enhanced Monitoring (30-45 score)
- **Rate**: 1.75-2.0+ factor rate
- **Loan Multiple**: 2x monthly revenue
- **Term**: 3-9 months
- **Monitoring**: Weekly reviews + guarantees

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

### Adding New Industries
1. Add industry to `INDUSTRY_THRESHOLDS` in `industry_config.py`
2. Set appropriate risk multiplier in `subprime_scoring_system.py`
3. Test with representative data

## 📦 Dependencies

### Core Framework
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing

### Visualization
- **Plotly**: Interactive charts and graphs
- **Matplotlib**: Additional plotting capabilities

### Machine Learning
- **Scikit-learn**: ML algorithms and utilities
- **Joblib**: Model serialization

### Additional Features
- **RapidFuzz**: Text matching for categorization
- **Python-dotenv**: Environment configuration
- **Cryptography**: Security utilities

## 🚀 Deployment

### Production Deployment
```bash
# Build production image
docker build -t business-finance-scorecard .

# Run with production settings
docker run -d \
  --name finance-app \
  -p 8501:8501 \
  --env-file .env \
  business-finance-scorecard
```

### Scaling Considerations
- **Memory**: 4GB+ recommended for large datasets
- **CPU**: Multi-core beneficial for ML processing
- **Storage**: SSD recommended for caching performance

## 📞 Support & Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure Python path is correct
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Memory Issues**
- Reduce analysis period (use 6-12 months vs full data)
- Clear cache: `st.cache_data.clear()` in app
- Increase system memory allocation

**Performance Optimization**
- Enable caching for frequently accessed data
- Use period filtering for large datasets
- Consider database storage for enterprise use

### Logging
Application logs are stored in the `logs/` directory:
- `app.log`: General application logs
- `errors.log`: Error-specific logs
- `audit.log`: Sensitive operation audit trail


### Development Guidelines
- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation for API changes
- Ensure backwards compatibility

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Additional Resources

- **API Documentation**: See `docs/api_reference.md`
- **User Guide**: See `docs/user_guide.md`
- **Developer Guide**: See `docs/developer_guide.md`
- **Deployment Guide**: See `docs/deployment.md`

## 📊 Version History

### v2.0.0 (Current)
- Enhanced subprime scoring system
- Risk factor penalty integration
- Advanced transaction categorization
- Interactive dashboard with export functionality
- Docker deployment support

### v1.0.0
- Basic financial analysis
- Simple threshold scoring
- Manual transaction categorization

---

**Built for modern business finance analysis**