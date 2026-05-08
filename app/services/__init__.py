# app/services/__init__.py
"""
Services module containing business logic and scoring systems.

Available services:
- SubprimeScoring: Micro-enterprise lending scoring
- EnsembleScorer: Unified multi-model scoring
- AdvancedMetricsCalculator: Transaction pattern analysis
- DataProcessor: Enhanced data processing
- FinancialAnalyzer: Financial metrics calculation
"""

from .subprime_scoring_system import SubprimeScoring, SubprimeScoringSystem
from .ensemble_scorer import (
    EnsembleScorer,
    get_ensemble_recommendation,
    Decision,
    EnsembleResult,
    ScoringResult
)
from .advanced_metrics import (
    AdvancedMetricsCalculator,
    calculate_advanced_metrics
)

# Optional imports (may depend on external services or file system)
try:
    from .data_processor import DataProcessor, TransactionCategorizer
except (ImportError, FileNotFoundError, OSError):
    DataProcessor = None
    TransactionCategorizer = None

try:
    from .financial_analyzer import FinancialAnalyzer
except (ImportError, FileNotFoundError, OSError):
    FinancialAnalyzer = None

try:
    from .card_terminal_ingestion import CardTerminalIngestionService
except (ImportError, FileNotFoundError, OSError):
    CardTerminalIngestionService = None

try:
    from .payment_provider_registry import ProviderSpec, PROVIDER_SPECS, provider_catalog, detect_providers_in_text
except (ImportError, FileNotFoundError, OSError):
    ProviderSpec = None
    PROVIDER_SPECS = []
    provider_catalog = None
    detect_providers_in_text = None

try:
    from .provider_parser_profiles import ProviderParserProfile, PROVIDER_PARSER_PROFILES, providers_with_native_profiles
except (ImportError, FileNotFoundError, OSError):
    ProviderParserProfile = None
    PROVIDER_PARSER_PROFILES = {}
    providers_with_native_profiles = None

__all__ = [
    # Scoring systems
    'SubprimeScoring',
    'SubprimeScoringSystem',
    'EnsembleScorer',
    'get_ensemble_recommendation',
    'Decision',
    'EnsembleResult',
    'ScoringResult',
    # Metrics
    'AdvancedMetricsCalculator',
    'calculate_advanced_metrics',
    # Data processing
    'DataProcessor',
    'TransactionCategorizer',
    'FinancialAnalyzer',
    'CardTerminalIngestionService',
    'ProviderSpec',
    'PROVIDER_SPECS',
    'provider_catalog',
    'detect_providers_in_text',
    'ProviderParserProfile',
    'PROVIDER_PARSER_PROFILES',
    'providers_with_native_profiles',
]
