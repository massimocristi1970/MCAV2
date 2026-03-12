"""
Synthetic data engine for MCA/business lending application datasets.

This package generates synthetic application data for:
  - Testing and pipeline validation
  - Scenario and stress testing
  - Dashboard and demo data
  - Sensitivity analysis

IMPORTANT: Synthetic data produced here is for simulation and testing ONLY.
It must NOT be used for production model calibration or mixed with real
training data for production credit models. All outputs are clearly
labelled as synthetic (data_source='synthetic').
"""

from . import schema
from . import generator
from . import validator
from . import scenarios
from . import rules
from . import outcomes

from .schema import (
    ML_FEATURE_NAMES,
    SYNTHETIC_METADATA_COLS,
    get_required_columns,
    get_bounds,
    clip_row,
    clip_dataframe,
)
from .generator import load_reference, generate
from .validator import validate, write_validation_outputs
from .scenarios import get_scenario, list_scenarios

__all__ = [
    "schema",
    "generator",
    "validator",
    "scenarios",
    "rules",
    "outcomes",
    "ML_FEATURE_NAMES",
    "SYNTHETIC_METADATA_COLS",
    "get_required_columns",
    "get_bounds",
    "clip_row",
    "clip_dataframe",
    "load_reference",
    "generate",
    "validate",
    "write_validation_outputs",
    "get_scenario",
    "list_scenarios",
]
