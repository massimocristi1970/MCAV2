"""Tests for TU director feature mapping and readiness checks."""

from __future__ import annotations

from app.services.tu_director_params import director_tu_is_ready, tu_features_to_scoring_fields


def test_tu_features_to_scoring_fields_maps_defaults_and_ccj():
    fields = tu_features_to_scoring_fields(
        {
            "defaults_36m_total": 4,
            "defaults_12m_total": 1,
            "ccj_active_total": 2,
        }
    )
    assert fields["director_defaults_36m"] == 4
    assert fields["personal_default_12m"] is True
    assert fields["director_ccj"] is True


def test_director_tu_is_ready_requires_parsed_score():
    assert director_tu_is_ready(tu_parse_status="parsed", tu_director_score=72) is True
    assert director_tu_is_ready(tu_parse_status="missing", tu_director_score=None) is False
    assert director_tu_is_ready(
        tu_parse_status="missing",
        tu_director_score=None,
        params={"tu_director_score": 65, "tu_director_decision": "APPROVE"},
    ) is True
