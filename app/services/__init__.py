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
]
