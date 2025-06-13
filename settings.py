# app/config/settings.py
import os
from typing import Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Application settings and configuration."""
    
    # Application Info
    APP_NAME: str = os.getenv("APP_NAME", "Business Finance Scorecard")
    APP_VERSION: str = os.getenv("APP_VERSION", "2.0.0")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Plaid Configuration
    PLAID_CLIENT_ID: str = os.getenv("PLAID_CLIENT_ID")
    PLAID_SECRET: str = os.getenv("PLAID_SECRET")
    PLAID_ENV: str = os.getenv("PLAID_ENV", "production")
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    MODEL_PATH: str = os.getenv("MODEL_PATH", "app/models/model_artifacts/model.pkl")
    SCALER_PATH: str = os.getenv("SCALER_PATH", "app/models/model_artifacts/scaler.pkl")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Cache
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))
    
    # Business Rules
    DEFAULT_COMPANY_AGE: int = 24
    DEFAULT_DIRECTOR_SCORE: int = 75
    MONTHS_THRESHOLD: int = 6
    MAX_LOAN_AMOUNT: float = 10_000_000.0
    
    # Plaid Hosts
    PLAID_HOSTS: Dict[str, str] = {
        "sandbox": "https://sandbox.plaid.com",
        "development": "https://development.plaid.com",
        "production": "https://production.plaid.com"
    }
    
    @property
    def plaid_host(self) -> str:
        """Get the appropriate Plaid host URL."""
        return self.PLAID_HOSTS.get(self.PLAID_ENV, self.PLAID_HOSTS["production"])
    
    def validate_required_settings(self) -> None:
        """Validate that all required settings are present."""
        required_settings = [
            ("PLAID_CLIENT_ID", self.PLAID_CLIENT_ID),
            ("PLAID_SECRET", self.PLAID_SECRET),
        ]
        
        missing_settings = [
            setting_name for setting_name, setting_value in required_settings 
            if not setting_value
        ]
        
        if missing_settings:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_settings)}"
            )

# Global settings instance
settings = Settings()

# Validate settings on import
try:
    settings.validate_required_settings()
except ValueError as e:
    if not settings.DEBUG:
        raise e
    print(f"Warning: {e}")

# Company access tokens (should be encrypted in production)
COMPANY_ACCESS_TOKENS = {
    "Bound Studios": os.getenv("BOUND_STUDIOS_TOKEN"),
    "Moving Ewe": os.getenv("MOVING_EWE_TOKEN"),
    "Sanitaire Ltd": os.getenv("SANITAIRE_TOKEN"),
    "Ellevate limited": os.getenv("ELLEVATE_TOKEN"),
    "Boiler Solution Cover UK": os.getenv("BOILER_SOLUTION_TOKEN")
}

# Filter out None values
COMPANY_ACCESS_TOKENS = {
    k: v for k, v in COMPANY_ACCESS_TOKENS.items() if v is not None
}