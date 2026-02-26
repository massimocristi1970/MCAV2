"""
TU XML Scorecard - Bureau-only credit scoring for TransUnion XML data.

This package provides tools for:
- Extracting features from TransUnion XML bureau files
- Scoring applications using rule-based criteria
- Identifying recovery candidates among declined applications
"""

from .feature_extractor import extract_features_from_xml_bytes, TUFeatures, XMLParseError
from .scorecard_rules import score_tu_features, ScoreResult
from .xml_utils import find_first_by_local, find_all_by_local, get_text, to_int, to_float

__all__ = [
    # Feature extraction
    "extract_features_from_xml_bytes",
    "TUFeatures",
    "XMLParseError",
    # Scoring
    "score_tu_features",
    "ScoreResult",
    # XML utilities
    "find_first_by_local",
    "find_all_by_local",
    "get_text",
    "to_int",
    "to_float",
]
