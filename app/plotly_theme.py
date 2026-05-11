"""MCA Plotly theme: slate chart surfaces matching app UI (avoids pure-black plot boxes)."""

from __future__ import annotations

import copy
from typing import Any

import plotly.io as pio

_REGISTERED = False

MCA_TEMPLATE_NAME = "mca_dark"

# Align with app/ui_theme.py dark surfaces
MCA_PAPER = "#1e293b"
MCA_PLOT = "#172032"

# Horizontal legend under plot area (avoids collision with title — Streamlit default legend was y≈1)
LEGEND_BELOW = dict(
    orientation="h",
    yanchor="top",
    y=-0.22,
    xanchor="center",
    x=0.5,
    bgcolor="rgba(15, 23, 42, 0.92)",
    bordercolor="rgba(148, 163, 184, 0.35)",
    borderwidth=1,
)
# High-contrast on dark plot area (replaces black markers/lines)
THRESHOLD_LINE = "#fbbf24"
THRESHOLD_MARKER = "#fbbf24"


def register_mca_plotly_template() -> None:
    """Extend plotly_dark with slate backgrounds, spacing, and readable axes (idempotent)."""
    global _REGISTERED
    if _REGISTERED:
        return

    base = copy.deepcopy(pio.templates["plotly_dark"])
    lo = base.layout
    lo.paper_bgcolor = MCA_PAPER
    lo.plot_bgcolor = MCA_PLOT
    lo.font = dict(
        family="DM Sans, ui-sans-serif, system-ui, -apple-system, sans-serif",
        color="#e2e8f0",
        size=12,
    )
    lo.title = dict(
        font=dict(size=15, color="#f8fafc"),
        x=0.02,
        xanchor="left",
        pad=dict(t=10, b=6),
    )
    lo.margin = dict(l=56, r=36, t=62, b=72)
    lo.colorway = [
        "#34d399",
        "#38bdf8",
        "#fbbf24",
        "#f472b6",
        "#a78bfa",
        "#fb7185",
        "#2dd4bf",
        "#60a5fa",
    ]
    lo.hoverlabel = dict(
        bgcolor="#0f172a",
        bordercolor="#475569",
        font=dict(size=12, color="#f8fafc"),
    )
    lo.legend = dict(
        bgcolor="rgba(15, 23, 42, 0.55)",
        bordercolor="rgba(148, 163, 184, 0.35)",
        borderwidth=1,
        font=dict(size=11, color="#e2e8f0"),
    )

    axis_common = dict(
        gridcolor="rgba(148, 163, 184, 0.18)",
        zerolinecolor="rgba(148, 163, 184, 0.35)",
        linecolor="rgba(148, 163, 184, 0.45)",
        tickfont=dict(color="#cbd5e1", size=11),
        title=dict(font=dict(size=12, color="#94a3b8")),
        automargin=True,
        showline=True,
        mirror=False,
    )
    lo.update(xaxis=axis_common, yaxis=axis_common)

    pio.templates[MCA_TEMPLATE_NAME] = base
    pio.templates.default = MCA_TEMPLATE_NAME
    _REGISTERED = True


def finalize_mca_figure(fig: Any) -> None:
    """Merge slate template + explicit bg colors (survives Streamlit figure JSON round-trip)."""
    register_mca_plotly_template()
    fig.update_layout(
        template=MCA_TEMPLATE_NAME,
        paper_bgcolor=MCA_PAPER,
        plot_bgcolor=MCA_PLOT,
    )


def show_mca_plotly(fig: Any, *, key: str | None = None, **kwargs: Any):
    """
    Display a Plotly figure without Streamlit's chart theme overlay.

    Streamlit defaults st.plotly_chart(theme=\"streamlit\"), which forces placeholder
    colors on the frontend (pure black plot areas). theme=None uses our template.
    """
    import streamlit as st

    finalize_mca_figure(fig)
    opts: dict[str, Any] = {"theme": None, "use_container_width": True}
    opts.update(kwargs)
    if key is not None:
        opts["key"] = key
    return st.plotly_chart(fig, **opts)
