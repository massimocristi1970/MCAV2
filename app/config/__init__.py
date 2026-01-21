# app/config/__init__.py
"""
Configuration module containing settings and thresholds.

Available configurations:
- settings: Application settings and environment configuration
- industry_config: Industry-specific configurations
- scoring_thresholds: Centralized scoring thresholds
"""

# Import settings (may depend on environment)
try:
    from .settings import settings, Settings
except ImportError:
    settings = None
    Settings = None

# Import industry config
try:
    from .industry_config import (
        INDUSTRY_THRESHOLDS,
        LOW_RISK_SECTORS,
        HIGH_RISK_SECTORS,
        MODERATE_RISK_SECTORS
    )
except (ImportError, SyntaxError):
    INDUSTRY_THRESHOLDS = {}
    LOW_RISK_SECTORS = []
    HIGH_RISK_SECTORS = []
    MODERATE_RISK_SECTORS = []

# Import scoring thresholds
from .scoring_thresholds import (
    ScoringThresholds,
    MetricThreshold,
    get_thresholds,
    THRESHOLDS
)

__all__ = [
    'settings',
    'Settings',
    'ScoringThresholds',
    'MetricThreshold',
    'get_thresholds',
    'THRESHOLDS',
]
