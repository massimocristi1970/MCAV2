"""Map TransUnion director XML features into scorecard params and metrics."""

from __future__ import annotations

from typing import Any


def tu_features_to_scoring_fields(features: dict[str, Any] | None) -> dict[str, Any]:
    """Extract underwriting fields used by Subprime and ensemble overlays."""
    feats = features or {}
    defaults_36m = int(feats.get("defaults_36m_total", 0) or 0)
    defaults_12m = int(feats.get("defaults_12m_total", 0) or 0)
    ccj_active = int(feats.get("ccj_active_total", 0) or feats.get("active_judgments_total", 0) or 0)

    return {
        "director_defaults_36m": defaults_36m,
        "personal_default_12m": defaults_12m > 0,
        "director_ccj": ccj_active > 0,
    }


def apply_tu_features_to_params_and_metrics(
    params: dict[str, Any],
    metrics: dict[str, Any],
    features: dict[str, Any] | None,
) -> None:
    """Merge TU-derived director risk fields into params and metrics."""
    fields = tu_features_to_scoring_fields(features)
    params.update(fields)
    metrics["Director Defaults 36m"] = fields["director_defaults_36m"]


def director_tu_is_ready(
    *,
    tu_parse_status: str,
    tu_director_score: int | float | None,
    params: dict[str, Any] | None = None,
) -> bool:
    """Return True when a parsed TU director score is available for scoring."""
    if tu_parse_status == "parsed" and tu_director_score is not None:
        return True
    saved = params or {}
    return saved.get("tu_director_score") is not None and bool(saved.get("tu_director_decision"))
