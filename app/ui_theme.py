"""
Shared Streamlit UI styling for MCA Scorecard apps.

Visual-only: CSS injection and optional Plotly defaults. Safe to call once per session.
"""

from __future__ import annotations

import html

import streamlit as st


def apply_ui_theme() -> None:
    """Inject global styles and Plotly dark template (call once per page run)."""
    try:
        from app.plotly_theme import register_mca_plotly_template

        register_mca_plotly_template()
    except Exception:
        try:
            import plotly.io as pio

            pio.templates.default = "plotly_dark"
        except Exception:
            pass

    # Use st.html (not st.markdown): markdown sanitization strips <style>, which leaves raw CSS on screen.
    st.html(
        """
<style>
@import url("https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&display=swap");

  :root {
    --mca-accent: #0e7490;
    --mca-accent-deep: #0f766e;
    --mca-accent-soft: rgba(14, 116, 144, 0.12);
    --mca-ink: #0f172a;
    /* Darker than slate-500 so body / captions stay readable on white */
    --mca-muted: #475569;
    --mca-line: #e2e8f0;
    --mca-surface: #f8fafc;
    --mca-card: #ffffff;
    --mca-radius: 12px;
    --mca-radius-sm: 8px;
    --mca-shadow: 0 1px 2px rgba(15, 23, 42, 0.06), 0 4px 12px rgba(15, 23, 42, 0.04);
    --mca-shadow-lg: 0 4px 6px rgba(15, 23, 42, 0.04), 0 12px 28px rgba(15, 23, 42, 0.07);
    --mca-sticky-h2-bg: linear-gradient(to bottom, rgba(248, 250, 252, 0.98), rgba(248, 250, 252, 0.9));
    --mca-sticky-h2-shadow: 0 10px 28px rgba(15, 23, 42, 0.07);
  }

  html, body, [class*="css"] {
    font-family: "DM Sans", ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    -webkit-font-smoothing: antialiased;
  }

  .stApp {
    background: linear-gradient(165deg, var(--mca-surface) 0%, #eef2f7 52%, #f1f5f9 100%);
    /* Streamlit / OS dark theme can leave light label colors — force readable ink on light UI */
    color: var(--mca-ink);
  }

  header[data-testid="stHeader"] {
    background: rgba(248, 250, 252, 0.92) !important;
    backdrop-filter: blur(8px);
    border-bottom: 1px solid rgba(226, 232, 240, 0.85);
  }

  footer {
    visibility: hidden;
  }

  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
    border-right: 1px solid var(--mca-line);
    box-shadow: 4px 0 24px rgba(15, 23, 42, 0.04);
  }

  section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
  }

  section[data-testid="stSidebar"] [data-baseweb="input"] input,
  section[data-testid="stSidebar"] textarea,
  section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    border-radius: var(--mca-radius-sm) !important;
    background-color: #ffffff !important;
    color: var(--mca-ink) !important;
    border: 1px solid #cbd5e1 !important;
    box-shadow: none !important;
  }

  section[data-testid="stSidebar"] [data-baseweb="select"] svg {
    fill: var(--mca-ink) !important;
  }

  /* Multipage nav (main / charts / …) — Streamlit uses theme bodyText; can become white-on-white */
  section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"],
  section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] * {
    color: var(--mca-ink) !important;
  }

  section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] svg {
    fill: var(--mca-ink) !important;
    color: var(--mca-ink) !important;
  }

  /* Sidebar markdown / st.write (e.g. bureau lines) */
  section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: var(--mca-ink) !important;
  }

  /* +/- steppers on number inputs (avoid black chrome on light theme) */
  section[data-testid="stSidebar"] [data-baseweb="input"] button {
    background-color: #f1f5f9 !important;
    color: var(--mca-ink) !important;
    border: 1px solid #cbd5e1 !important;
  }

  section[data-testid="stSidebar"] div[data-testid="stExpander"] summary,
  section[data-testid="stSidebar"] div[data-testid="stExpander"] summary * {
    color: var(--mca-ink) !important;
  }

  /* Widget labels: Streamlit nests label text in spans — target descendants */
  section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
  section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
  .main [data-testid="stWidgetLabel"],
  .main [data-testid="stWidgetLabel"] * {
    color: var(--mca-ink) !important;
  }

  section[data-testid="stSidebar"] [data-testid="stCaption"],
  .main [data-testid="stCaption"] {
    color: var(--mca-muted) !important;
  }

  div[data-testid="stFileUploader"] [data-testid="stWidgetLabel"],
  div[data-testid="stFileUploader"] [data-testid="stWidgetLabel"] * {
    color: var(--mca-ink) !important;
  }

  div[data-testid="stFileUploader"] [data-testid="stCaption"] {
    color: var(--mca-muted) !important;
  }

  .mca-hero {
    background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
    border: 1px solid var(--mca-line);
    border-radius: var(--mca-radius);
    padding: 1.35rem 1.5rem 1.25rem 1.5rem;
    margin: 0 0 1rem 0;
    box-shadow: var(--mca-shadow-lg);
    position: relative;
    overflow: hidden;
  }

  .mca-hero-eyebrow {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--mca-accent);
    margin: 0 0 0.5rem 0;
  }

  .mca-hero::before {
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 4px;
    background: linear-gradient(180deg, var(--mca-accent) 0%, #155e75 100%);
    border-radius: 4px 0 0 4px;
  }

  .mca-hero h1 {
    font-size: 1.85rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: var(--mca-ink);
    margin: 0 0 0.35rem 0;
    line-height: 1.2;
  }

  .mca-hero .mca-hero-tagline {
    margin: 0;
    font-size: 1rem;
    color: var(--mca-muted);
    font-weight: 400;
    line-height: 1.5;
    max-width: 52rem;
  }

  .mca-page-bar {
    margin: 0 0 1.25rem 0;
    padding: 0 0 0.85rem 0;
    border-bottom: 1px solid var(--mca-line);
  }

  .mca-page-bar h2 {
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--mca-ink);
    margin: 0 0 0.25rem 0;
  }

  .mca-page-bar .mca-page-caption {
    margin: 0;
    font-size: 0.92rem;
    color: var(--mca-muted);
    line-height: 1.45;
  }

  .mca-page-bar .mca-page-eyebrow {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--mca-accent);
    margin: 0 0 0.35rem 0;
  }

  .mca-workflow {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.35rem 0.5rem;
    padding: 0.65rem 1rem;
    margin: 0 0 1.35rem 0;
    background: rgba(255, 255, 255, 0.82);
    border: 1px solid var(--mca-line);
    border-radius: var(--mca-radius);
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
  }

  .mca-workflow-step {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--mca-muted);
  }

  .mca-workflow-step span:first-child {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.35rem;
    height: 1.35rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    background: var(--mca-accent-soft);
    color: var(--mca-accent-deep);
  }

  .mca-workflow-arrow {
    color: #cbd5e1;
    font-size: 0.75rem;
    user-select: none;
  }

  .mca-intake-panel {
    margin: 0 0 1.25rem 0;
    padding: 1rem 1.15rem 1.05rem 1.15rem;
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid var(--mca-line);
    border-radius: var(--mca-radius);
    border-left: 4px solid var(--mca-accent);
    box-shadow: var(--mca-shadow);
  }

  .mca-intake-kicker {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--mca-accent);
    margin: 0 0 0.35rem 0;
  }

  .mca-intake-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--mca-ink);
    margin: 0 0 0.35rem 0;
    letter-spacing: -0.02em;
  }

  .mca-intake-desc {
    margin: 0;
    font-size: 0.9rem;
    color: var(--mca-muted);
    line-height: 1.5;
    max-width: 44rem;
  }

  .mca-empty {
    text-align: center;
    padding: 2.25rem 1.5rem;
    margin: 1rem 0 0 0;
    background: rgba(255, 255, 255, 0.75);
    border: 1px dashed #cbd5e1;
    border-radius: var(--mca-radius);
  }

  .mca-empty-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--mca-ink);
    margin: 0 0 0.5rem 0;
  }

  .mca-empty-body {
    margin: 0 auto;
    max-width: 26rem;
    font-size: 0.92rem;
    color: var(--mca-muted);
    line-height: 1.55;
  }

  .mca-sidebar-footnote {
    font-size: 0.72rem;
    color: var(--mca-muted);
    line-height: 1.45;
    margin-top: 1.25rem;
    padding-top: 1rem;
    border-top: 1px solid var(--mca-line);
  }

  .mca-sb-section {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--mca-muted);
    margin: 1.25rem 0 0.6rem 0;
    padding-bottom: 0.25rem;
    border-bottom: 1px solid var(--mca-line);
  }

  .mca-sb-section:first-child {
    margin-top: 0;
  }

  .mca-sb-sub {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--mca-ink);
    margin: 0.85rem 0 0.45rem 0;
  }

  .main .block-container {
    padding-top: 1.35rem;
    padding-bottom: 3.5rem;
    max-width: min(1180px, 100%);
    color: var(--mca-ink);
  }

  section[data-testid="stSidebar"] .block-container {
    color: var(--mca-ink);
  }

  div[data-testid="stVerticalBlockBorderWrapper"] {
    gap: 0.35rem;
  }

  [data-testid="stDataFrame"],
  [data-testid="stDataFrame"] > div {
    border-radius: var(--mca-radius-sm) !important;
  }

  div[data-testid="stMetric"] {
    background: var(--mca-card);
    border: 1px solid var(--mca-line);
    border-radius: var(--mca-radius);
    padding: 0.65rem 0.85rem;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
  }

  div[data-testid="stMetric"] label[data-testid="stMetricLabel"] {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
    color: var(--mca-muted) !important;
  }

  div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-weight: 700 !important;
    color: var(--mca-ink) !important;
  }

  .main h2 {
    font-size: 1.35rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
    color: var(--mca-ink) !important;
    margin-top: 1.25rem !important;
    padding-top: 0.45rem !important;
    padding-bottom: 0.4rem !important;
    padding-left: 0.45rem !important;
    padding-right: 0.45rem !important;
    margin-left: -0.25rem !important;
    margin-right: -0.25rem !important;
    border-bottom: 2px solid var(--mca-accent-soft) !important;
    position: sticky !important;
    top: 3.35rem !important;
    z-index: 28 !important;
    background: var(--mca-sticky-h2-bg) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    box-shadow: var(--mca-sticky-h2-shadow) !important;
  }

  .main h3 {
    font-size: 1.08rem !important;
    font-weight: 600 !important;
    color: var(--mca-ink) !important;
  }

  hr {
    border: none;
    border-top: 1px solid var(--mca-line);
    margin: 1.25rem 0;
  }

  div[data-testid="stExpander"] {
    border: 1px solid var(--mca-line) !important;
    border-radius: var(--mca-radius) !important;
    background: rgba(255, 255, 255, 0.65) !important;
    overflow: hidden;
  }

  .stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background-color: transparent;
    border-bottom: 1px solid var(--mca-line);
  }

  .stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    font-weight: 600;
  }

  div[data-testid="stFileUploader"] section {
    border-radius: var(--mca-radius);
    border-style: dashed !important;
    border-color: #cbd5e1 !important;
    background: rgba(255, 255, 255, 0.85);
  }

  .stDownloadButton button,
  button[kind="primary"] {
    border-radius: 10px !important;
    font-weight: 600 !important;
    background: linear-gradient(180deg, var(--mca-accent) 0%, var(--mca-accent-deep) 100%) !important;
    border: none !important;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.12) !important;
  }

  button[kind="primary"]:hover {
    filter: brightness(1.06);
  }

  section[data-testid="stSidebar"] button[kind="primary"] {
    border-radius: 10px !important;
    font-weight: 600 !important;
  }

  .stAlert {
    border-radius: var(--mca-radius) !important;
    border: 1px solid var(--mca-line) !important;
  }

  iframe[title="streamlit_download_link"],
  [data-testid="stCaption"] {
    font-size: 0.82rem;
  }
</style>
        """
    )

    st.html(
        """
<style>
  .stApp {
    background: linear-gradient(165deg, #0f172a 0%, #1e293b 48%, #0f172a 100%) !important;
    color: #e2e8f0 !important;
  }

  header[data-testid="stHeader"] {
    background: rgba(15, 23, 42, 0.94) !important;
    border-bottom: 1px solid #334155 !important;
  }

  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
    border-right: 1px solid #334155 !important;
    box-shadow: 4px 0 28px rgba(0, 0, 0, 0.35) !important;
  }

  section[data-testid="stSidebar"] [data-baseweb="input"] input,
  section[data-testid="stSidebar"] textarea,
  section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #1e293b !important;
    color: #f1f5f9 !important;
    border: 1px solid #64748b !important;
    box-shadow: none !important;
  }

  section[data-testid="stSidebar"] [data-baseweb="select"] svg {
    fill: #e2e8f0 !important;
  }

  .main .block-container {
    color: #e2e8f0 !important;
  }

  .main .block-container label,
  .main .block-container p,
  .main .block-container span[data-testid="stMarkdownContainer"] {
    color: inherit !important;
  }

  section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
  section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
  .main [data-testid="stWidgetLabel"],
  .main [data-testid="stWidgetLabel"] * {
    color: #e2e8f0 !important;
  }

  section[data-testid="stSidebar"] [data-testid="stCaption"],
  .main [data-testid="stCaption"] {
    color: #cbd5e1 !important;
  }

  div[data-testid="stFileUploader"] [data-testid="stWidgetLabel"],
  div[data-testid="stFileUploader"] [data-testid="stWidgetLabel"] * {
    color: #e2e8f0 !important;
  }

  div[data-testid="stFileUploader"] [data-testid="stCaption"] {
    color: #cbd5e1 !important;
  }

  section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: #e2e8f0 !important;
  }

  section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"],
  section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] * {
    color: #e2e8f0 !important;
  }

  section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] svg {
    fill: #e2e8f0 !important;
    color: #e2e8f0 !important;
  }

  section[data-testid="stSidebar"] [data-baseweb="input"] button {
    background-color: #334155 !important;
    color: #f8fafc !important;
    border: 1px solid #64748b !important;
  }

  section[data-testid="stSidebar"] div[data-testid="stExpander"] summary,
  section[data-testid="stSidebar"] div[data-testid="stExpander"] summary * {
    color: #e2e8f0 !important;
  }

  /* Custom section headings from sidebar_section / sidebar_subsection */
  section[data-testid="stSidebar"] .mca-sb-section {
    color: #94a3b8 !important;
    border-bottom-color: #475569 !important;
  }

  section[data-testid="stSidebar"] .mca-sb-sub {
    color: #e2e8f0 !important;
  }

  .mca-intake-title {
    color: #f8fafc !important;
  }

  .mca-intake-desc {
    color: #cbd5e1 !important;
  }

  .mca-workflow-step {
    color: #e2e8f0 !important;
  }

  .mca-workflow-arrow {
    color: #94a3b8 !important;
  }

  div[data-testid="stMetric"] {
    background: #1e293b !important;
    border-color: #334155 !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25) !important;
  }

  div[data-testid="stMetric"] label[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
  }

  div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #f8fafc !important;
  }

  .mca-hero {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
    border-color: #334155 !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35) !important;
  }

  .mca-hero h1 {
    color: #f8fafc !important;
  }

  .mca-hero .mca-hero-tagline {
    color: #cbd5e1 !important;
  }

  .mca-workflow {
    background: rgba(30, 41, 59, 0.85) !important;
    border-color: #334155 !important;
  }

  .mca-intake-panel {
    background: rgba(30, 41, 59, 0.92) !important;
    border-color: #334155 !important;
  }

  .mca-empty {
    background: rgba(30, 41, 59, 0.65) !important;
    border-color: #475569 !important;
  }

  .mca-page-bar {
    border-bottom-color: #334155 !important;
  }

  .mca-page-bar h2 {
    color: #f8fafc !important;
  }

  .mca-page-bar .mca-page-caption {
    color: #94a3b8 !important;
  }

  div[data-testid="stExpander"] {
    background: rgba(30, 41, 59, 0.75) !important;
    border-color: #334155 !important;
  }

  .stAlert {
    background-color: rgba(30, 41, 59, 0.9) !important;
    border-color: #475569 !important;
    color: #e2e8f0 !important;
  }

  div[data-testid="stFileUploader"] section {
    border-color: #475569 !important;
    background: rgba(15, 23, 42, 0.65) !important;
  }

  .stTabs [data-baseweb="tab-list"] {
    border-bottom-color: #334155 !important;
  }

  hr {
    border-top-color: #334155 !important;
  }

  :root {
    --mca-sticky-h2-bg: linear-gradient(to bottom, rgba(15, 23, 42, 0.97), rgba(30, 41, 59, 0.88)) !important;
    --mca-sticky-h2-shadow: 0 12px 28px rgba(0, 0, 0, 0.35) !important;
  }

  .main h2 {
    border-bottom-color: rgba(45, 212, 191, 0.25) !important;
  }

  .main h3 {
    color: #f1f5f9 !important;
  }

  .mca-empty-title {
    color: #f8fafc !important;
  }

  .mca-empty-body {
    color: #cbd5e1 !important;
  }

  .mca-empty-body strong {
    color: #f8fafc !important;
  }

  .mca-sidebar-footnote {
    color: #94a3b8 !important;
    border-top-color: #475569 !important;
  }

  [data-testid="stDataFrame"] {
    filter: none;
  }
</style>
        """
    )


def render_compact_page_title(
    title: str,
    caption: str | None = None,
    eyebrow: str | None = None,
) -> None:
    """Secondary-page header (lighter than the main hero)."""
    safe_title = html.escape(title)
    if eyebrow:
        eb_html = f'<p class="mca-page-eyebrow">{html.escape(eyebrow)}</p>'
    else:
        eb_html = ""
    if caption:
        safe_cap = html.escape(caption)
        cap_html = f'<p class="mca-page-caption">{safe_cap}</p>'
    else:
        cap_html = ""
    st.markdown(
        f"""
<div class="mca-page-bar">
  {eb_html}
  <h2>{safe_title}</h2>
  {cap_html}
</div>
        """,
        unsafe_allow_html=True,
    )


def render_main_hero(title: str, tagline: str, eyebrow: str | None = None) -> None:
    """Polished header row for the primary dashboard page."""
    safe_title = html.escape(title)
    safe_tag = html.escape(tagline)
    if eyebrow:
        eb = f'<p class="mca-hero-eyebrow">{html.escape(eyebrow)}</p>'
    else:
        eb = ""
    st.markdown(
        f"""
<div class="mca-hero">
  {eb}
  <h1>{safe_title}</h1>
  <p class="mca-hero-tagline">{safe_tag}</p>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_workflow_rail() -> None:
    """Compact step indicator for the main scoring flow."""
    st.markdown(
        """
<div class="mca-workflow" aria-label="Workflow">
  <div class="mca-workflow-step"><span>1</span><span>Configure case</span></div>
  <span class="mca-workflow-arrow">→</span>
  <div class="mca-workflow-step"><span>2</span><span>Upload bank JSON</span></div>
  <span class="mca-workflow-arrow">→</span>
  <div class="mca-workflow-step"><span>3</span><span>Review dashboard</span></div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_intake_panel_intro(
    title: str = "Transaction data",
    description: str = (
        "Upload your bank transaction export (JSON). Optional card-terminal statements "
        "improve reconciliation against bank inflows."
    ),
) -> None:
    """Section intro above the main file upload controls."""
    st.markdown(
        f"""
<div class="mca-intake-panel">
  <p class="mca-intake-kicker">Data intake</p>
  <p class="mca-intake-title">{html.escape(title)}</p>
  <p class="mca-intake-desc">{html.escape(description)}</p>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state_main() -> None:
    """Placeholder when no JSON has been loaded yet."""
    st.markdown(
        """
<div class="mca-empty">
  <p class="mca-empty-title">Ready when you are</p>
  <p class="mca-empty-body">
    Configure the company in the sidebar, then upload a transaction JSON file above.
    After scoring, use <strong>Scoring</strong>, <strong>Charts</strong>, and <strong>Reports</strong>
    in the sidebar—your analysis stays available when you return here.
  </p>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state_no_run(page_name: str, action_hint: str) -> None:
    """Standard empty state for multipage tools that need a Main-page run."""
    st.markdown(
        f"""
<div class="mca-empty">
  <p class="mca-empty-title">No analysis to show yet</p>
  <p class="mca-empty-body">
    Run a score on the <strong>Main</strong> page first, then open <strong>{html.escape(page_name)}</strong> again.
    {html.escape(action_hint)}
  </p>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_help_footer() -> None:
    """Small sidebar footer hint (call once at end of sidebar content)."""
    st.sidebar.markdown(
        """
<p class="mca-sidebar-footnote">
  Tip: Use the sidebar pages for focused views. Your last completed run stays on Main until you upload new JSON.
</p>
        """,
        unsafe_allow_html=True,
    )


def sidebar_section(title: str) -> None:
    """Sidebar major section label (replaces st.sidebar.header)."""
    st.sidebar.markdown(
        f'<p class="mca-sb-section">{html.escape(title)}</p>',
        unsafe_allow_html=True,
    )


def sidebar_subsection(title: str) -> None:
    """Sidebar subsection label (replaces st.sidebar.subheader)."""
    st.sidebar.markdown(
        f'<p class="mca-sb-sub">{html.escape(title)}</p>',
        unsafe_allow_html=True,
    )
