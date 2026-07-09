"""Tests for MCA rule period alignment."""

from __future__ import annotations

from app.services.mca_rule_runner import filter_transactions_by_period, run_mca_rule_scoring


def test_filter_transactions_by_period_limits_to_recent_months():
    txns = [
        {"date": "2024-01-01", "amount": -100, "name": "Old revenue"},
        {"date": "2024-11-01", "amount": -200, "name": "Recent revenue"},
        {"date": "2024-12-15", "amount": -300, "name": "Latest revenue"},
    ]
    scoped = filter_transactions_by_period(txns, 3)
    assert len(scoped) == 2
    assert all("2024-11" in t["date"] or "2024-12" in t["date"] for t in scoped)


def test_run_mca_rule_scoring_returns_decision_fields():
    txns = [
        {"date": "2024-10-01", "amount": -500, "name": "Stripe"},
        {"date": "2024-10-15", "amount": -600, "name": "Square"},
        {"date": "2024-11-01", "amount": 100, "name": "Rent"},
    ]
    result = run_mca_rule_scoring(txns, "All")
    assert "mca_rule_decision" in result
    assert "mca_rule_score" in result
    assert "mca_rule_reasons" in result
    assert "mca_rule_signals" in result
