"""Align metrics shared between MCA rule and Subprime scoring."""

from __future__ import annotations

from typing import Any


def align_scoring_metrics(metrics: dict[str, Any], params: dict[str, Any]) -> None:
    """
    Unify cross-model signals so Subprime volatility reflects MCA inflow CV,
    and expose MCA flow metrics on the metrics dict for diagnostics/UI.
    """
    signals = params.get("mca_rule_signals") or {}
    if not signals:
        return

    inflow_cv = signals.get("inflow_cv")
    if inflow_cv is not None:
        try:
            mca_cv = float(inflow_cv)
            bank_vol = float(metrics.get("Cash Flow Volatility") or 0)
            unified = max(bank_vol, mca_cv)
            if unified != bank_vol:
                metrics["Cash Flow Volatility Before MCA Align"] = bank_vol
                metrics["Cash Flow Volatility"] = round(unified, 3)
                metrics["Cash Flow Volatility Unified"] = True
            metrics["MCA Inflow CV"] = round(mca_cv, 3)
        except (TypeError, ValueError):
            pass

    for key in ("inflow_days_30d", "max_inflow_gap_days", "txn_count_avg_month", "months_covered"):
        if signals.get(key) is not None:
            metrics[f"MCA {key.replace('_', ' ').title()}"] = signals.get(key)
