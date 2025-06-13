# .env.example - Copy this to .env and fill in your values
# Plaid Configuration
PLAID_CLIENT_ID=your_plaid_client_id_here
PLAID_SECRET=your_plaid_secret_here
PLAID_ENV=production

# Application Configuration
APP_NAME=Business Finance Scorecard
APP_VERSION=2.0.0
DEBUG=False
LOG_LEVEL=INFO

# Database Configuration (for future use)
DATABASE_URL=sqlite:///./business_finance.db

# Security
SECRET_KEY=your_secret_key_here

# Cache Configuration
CACHE_TTL=3600

# Model Configuration
MODEL_PATH=app/models/model_artifacts/model.pkl
SCALER_PATH=app/models/model_artifacts/scaler.pkl

# Company Access Tokens (encrypted or reference to secure storage)
COMPANY_TOKENS_CONFIG=app/config/company_tokens.json