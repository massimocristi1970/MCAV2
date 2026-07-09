"""Tests for business bureau CCJ counting."""

from __future__ import annotations

from app.services.business_bureau_pdf import count_business_ccjs, explicit_ccj_present


def test_count_business_ccjs_returns_zero_when_report_says_none():
    text = "Legal notices\nNo county court judgments recorded on this business."
    assert count_business_ccjs(text) == 0
    assert explicit_ccj_present(text) is False


def test_count_business_ccjs_detects_numeric_count():
    text = "Legal notices\n2 county court judgments registered against the company."
    assert count_business_ccjs(text) == 2


def test_count_business_ccjs_falls_back_to_one_for_single_positive_signal():
    text = "County court judgment registered on 12 Jan 2024 for £1,250"
    assert count_business_ccjs(text) == 1
