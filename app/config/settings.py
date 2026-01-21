# app/config/settings.py
import os
import warnings
from typing import Dict, Any, List, Optional
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
    
    # Plaid Configuration (optional - only needed for Plaid API features)
    PLAID_CLIENT_ID: Optional[str] = os.getenv("PLAID_CLIENT_ID")
    PLAID_SECRET: Optional[str] = os.getenv("PLAID_SECRET")
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
    
    @property
    def plaid_configured(self) -> bool:
        """Check if Plaid API credentials are configured."""
        return bool(self.PLAID_CLIENT_ID and self.PLAID_SECRET)
    
    def validate_plaid_settings(self) -> None:
        """
        Validate Plaid settings - call this only when Plaid features are needed.
        Raises ValueError if Plaid credentials are not configured.
        """
        if not self.plaid_configured:
            raise ValueError(
                "Plaid API credentials not configured. "
                "Set PLAID_CLIENT_ID and PLAID_SECRET environment variables "
                "to use Plaid API features."
            )
    
    def get_missing_optional_settings(self) -> List[str]:
        """Get list of missing optional settings for informational purposes."""
        optional_settings = [
            ("PLAID_CLIENT_ID", self.PLAID_CLIENT_ID),
            ("PLAID_SECRET", self.PLAID_SECRET),
        ]
        return [name for name, value in optional_settings if not value]

# Global settings instance
settings = Settings()

# Log warnings about missing optional settings (don't raise errors)
_missing = settings.get_missing_optional_settings()
if _missing and settings.DEBUG:
    warnings.warn(
        f"Optional settings not configured: {', '.join(_missing)}. "
        "Plaid API features will be disabled. File upload will still work.",
        UserWarning
    )

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