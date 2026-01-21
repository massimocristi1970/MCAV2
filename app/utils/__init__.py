# app/utils/__init__.py
"""
Utilities module containing helper functions and tools.

Available utilities:
- WeightCalibrator: Calibrate scoring weights from ML model
- FeatureAligner: Align features between different scoring systems
- chart_utils: Chart and visualization utilities
"""

from .weight_calibration import (
    WeightCalibrator,
    calibrate_weights_from_model,
    get_weight_comparison_report
)
from .feature_alignment import (
    FeatureAligner,
    align_features_for_scoring,
    validate_scoring_input
)

# Optional chart utilities
try:
    from .chart_utils import *
except ImportError:
    pass

__all__ = [
    # Weight calibration
    'WeightCalibrator',
    'calibrate_weights_from_model',
    'get_weight_comparison_report',
    # Feature alignment
    'FeatureAligner',
    'align_features_for_scoring',
    'validate_scoring_input',
]
