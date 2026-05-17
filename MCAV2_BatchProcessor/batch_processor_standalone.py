"""
MCA v2 Batch Processing Application - COMPLETE FIXED VERSION
Run this as: streamlit run batch_processor_standalone.py --server.port 8502

MAJOR FIXES:
1. FIXED fuzzy matching logic and company name extraction
2. FIXED debug information display and storage
3. FIXED CSV parameter mapping with proper column handling
4. FIXED error handling and comprehensive logging
5. ADDED detailed debugging throughout the process
"""


from typing import Dict, Any, Tuple, List
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import zipfile
import tempfile
import hashlib
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from pathlib import Path
import joblib
import sys
from io import BytesIO

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

REPO_ROOT = Path(__file__).resolve().parents[1]
BATCH_PROCESSOR_DIR = Path(__file__).resolve().parent
MODEL_PATH = BATCH_PROCESSOR_DIR / "model.pkl"
SCALER_PATH = BATCH_PROCESSOR_DIR / "scaler.pkl"

if load_dotenv:
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(BATCH_PROCESSOR_DIR / ".env")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from app.services.ensemble_scorer import get_ensemble_recommendation
    from app.services.open_banking_insights import derive_open_banking_insights
    ENSEMBLE_SCORER_AVAILABLE = True
except ImportError as e:
    get_ensemble_recommendation = None
    derive_open_banking_insights = None
    ENSEMBLE_SCORER_AVAILABLE = False
    print(f"Ensemble scorer not available: {e}")

try:
    from app.plotly_theme import show_mca_plotly
    from app.ui_theme import (
        apply_ui_theme,
        render_intake_panel_intro,
        render_main_hero,
        sidebar_section,
        sidebar_subsection,
    )
except ImportError as e:
    show_mca_plotly = None
    apply_ui_theme = None
    render_intake_panel_intro = None
    render_main_hero = None
    sidebar_section = None
    sidebar_subsection = None
    print(f"Shared UI theme not available: {e}")

# Try to import rapidfuzz, fallback if not available
try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
    print("RapidFuzz imported successfully")
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    print("RapidFuzz not available, using fallback matching")

st.set_page_config(
    page_title="MCA v2 Batch Processor",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": None,
    },
)

if apply_ui_theme:
    apply_ui_theme()


def render_batch_workflow_rail() -> None:
    """Compact batch workflow indicator."""
    st.markdown(
        """
<div class="mca-workflow" aria-label="Batch workflow">
  <div class="mca-workflow-step"><span>1</span><span>Upload data and mapping</span></div>
  <div class="mca-workflow-step"><span>2</span><span>Label paid and not-paid JSONs</span></div>
  <div class="mca-workflow-step"><span>3</span><span>Process and export scorecard data</span></div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_batch_empty_state() -> None:
    """Initial empty state for the batch processor."""
    st.markdown(
        """
<div class="mca-empty">
  <p class="mca-empty-title">Ready for a batch run</p>
  <p class="mca-empty-body">
    Upload the data file and mapping, add the full JSON pool, then add paid and not-paid
    JSONs to label the outcomes automatically.
  </p>
</div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title: str, caption: str | None = None) -> None:
    """Consistent compact section heading."""
    st.markdown(f"### {title}")
    if caption:
        st.caption(caption)


def plot_mca_chart(fig, key: str | None = None) -> None:
    """Display Plotly charts using the main app theme when available."""
    if show_mca_plotly:
        show_mca_plotly(fig, key=key)
    else:
        st.plotly_chart(fig, use_container_width=True)


def normalize_upload_name(name: str) -> str:
    """Normalize uploaded file names for exact-ish set matching."""
    base = Path(str(name)).name.lower().strip()
    base = re.sub(r"\.(json|pdf)$", "", base)
    base = re.sub(r"^\d+[_\s-]+", "", base)
    base = base.replace("&", " and ")
    base = re.sub(r"\bapp\s+(\d+)\b", r"app\1", base)
    base = re.sub(r"\s+", " ", base.replace("_", " "))
    base = re.sub(r"\s*\(\d+\)$", "", base)
    base = re.sub(r"[^a-z0-9]+", " ", base)
    base = re.sub(r"\b(app)\s+(\d+)\b", r"\1\2", base)
    base = re.sub(r"\s+", " ", base)
    return base.strip()


def stable_json_signature(json_data: Any) -> str:
    """Return a stable content signature for matching JSONs with changed names."""
    try:
        payload = json.dumps(json_data, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:
        payload = str(json_data)
    return hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()


def read_tabular_upload(uploaded_file) -> pd.DataFrame:
    """Read CSV/XLSX uploads without caring about the uploaded filename."""
    suffix = Path(uploaded_file.name).suffix.lower()
    uploaded_file.seek(0)
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(uploaded_file)
    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    raise ValueError("Upload must be CSV, XLS, or XLSX")


def canonical_mapping_name(raw_column: str, mapped_name: str) -> str | None:
    """Map source/mapping labels into the internal batch parameter names."""
    raw = str(raw_column or "").strip().lower()
    mapped = str(mapped_name or "").strip().lower()
    text = f"{raw} {mapped}"

    if mapped in ["ignore", "nan", "none", ""]:
        return None
    if raw == "appid" or mapped == "application_id":
        return "application_id"
    if raw == "customername" or mapped == "company_name":
        return "company_name"
    if raw == "requestedamount" or "loan amount" in mapped or "requested" in mapped:
        return "requested_loan"
    if raw == "starttraiding" or "started trading" in mapped or "trading" in text:
        return "trading_start_date"
    if raw == "annualturnover" or "annual turnover" in mapped:
        return "annual_turnover"
    if raw == "score" or "director credit" in mapped:
        return "directors_score"
    if raw == "dft12" or "defaults in the last 12" in mapped:
        return "director_defaults_12m"
    if raw == "dft36" or "defaults in the last 36" in mapped:
        return "director_defaults_36m"
    if raw == "ccjnum" or "number of ccjs" in mapped:
        return "director_ccj_count"
    if raw == "ccjvalue" or "value of ccjs" in mapped:
        return "director_ccj_value"
    if raw == "loanpurpose":
        return "loan_purpose"
    if raw == "declinereason":
        return "decline_reason"
    if raw == "residentalstatus" or "residential status" in mapped:
        return "director_residential_status"
    if raw == "dateofbirth":
        return "director_date_of_birth"
    if raw in ["industry", "sic", "sector"]:
        return "industry"

    return re.sub(r"\W+", "_", raw).strip("_") or None


def read_mapping_upload(mapping_file) -> dict:
    """Read the two-column Power BI mapping workbook into source->canonical map."""
    suffix = Path(mapping_file.name).suffix.lower()
    mapping_file.seek(0)
    if suffix in [".xlsx", ".xls"]:
        mapping_df = pd.read_excel(mapping_file, header=None)
    elif suffix == ".csv":
        mapping_df = pd.read_csv(mapping_file, header=None)
    else:
        raise ValueError("Mapping upload must be CSV, XLS, or XLSX")

    if mapping_df.shape[1] < 2:
        raise ValueError("Mapping file must have at least two columns: source column and mapped meaning")

    mapping = {}
    for _, row in mapping_df.iterrows():
        source = row.iloc[0]
        mapped = row.iloc[1]
        if pd.isna(source):
            continue
        canonical = canonical_mapping_name(source, mapped)
        if canonical:
            mapping[str(source).strip()] = canonical
    return mapping


def coerce_bool_like(value) -> bool:
    if pd.isna(value) or value is None:
        return False
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ["", "{nd}", "nd", "none", "nan", "no", "false", "0", "n"]:
            return False
        if text in ["yes", "true", "1", "y"]:
            return True
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return bool(value)


def coerce_numeric(value, default=None):
    if pd.isna(value) or value in ["", "{ND}", "{nd}"]:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def calculate_company_age_months(start_value, as_of=None):
    if pd.isna(start_value) or start_value is None:
        return None
    as_of = as_of or datetime.now()
    start = pd.to_datetime(start_value, errors="coerce")
    if pd.isna(start):
        return None
    months = (as_of.year - start.year) * 12 + (as_of.month - start.month)
    if as_of.day < start.day:
        months -= 1
    return max(int(months), 0)


def prepare_application_metadata(data_file, mapping_file, fallback_industry: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build deduped application metadata from the uploaded data + mapping files."""
    raw_df = read_tabular_upload(data_file)
    mapping = read_mapping_upload(mapping_file)
    rename_map = {source: target for source, target in mapping.items() if source in raw_df.columns}
    metadata = raw_df.rename(columns=rename_map).copy()

    audit = {
        "raw_rows": len(raw_df),
        "mapped_columns": rename_map,
        "unmapped_source_columns": [col for col in raw_df.columns if col not in rename_map],
    }

    if "application_id" not in metadata.columns:
        raise ValueError("Mapping must identify an application_id column")
    if "company_name" not in metadata.columns:
        raise ValueError("Mapping must identify a company_name column")

    metadata["application_id"] = metadata["application_id"].astype(str).str.strip()
    metadata["company_name"] = metadata["company_name"].astype(str).str.strip()
    metadata = metadata[(metadata["application_id"] != "") & (metadata["company_name"] != "")]

    if "requested_loan" in metadata.columns:
        metadata["requested_loan"] = metadata["requested_loan"].apply(coerce_numeric)
    if "directors_score" in metadata.columns:
        metadata["directors_score"] = metadata["directors_score"].apply(coerce_numeric)
    if "trading_start_date" in metadata.columns:
        metadata["company_age_months"] = metadata["trading_start_date"].apply(calculate_company_age_months)

    metadata["industry"] = metadata.get("industry", fallback_industry)
    metadata["industry"] = metadata["industry"].fillna(fallback_industry).replace("", fallback_industry)
    metadata["business_ccj"] = False
    metadata["poor_or_no_online_presence"] = False
    metadata["uses_generic_email"] = False
    metadata["director_ccj"] = False

    if "director_defaults_12m" in metadata.columns:
        metadata["director_defaults_12m"] = metadata["director_defaults_12m"].apply(coerce_numeric).fillna(0)
        metadata["personal_default_12m"] = metadata["director_defaults_12m"] > 0
    if "director_defaults_36m" in metadata.columns:
        metadata["director_defaults_36m"] = metadata["director_defaults_36m"].apply(coerce_numeric).fillna(0)
    if "director_ccj_count" in metadata.columns:
        metadata["director_ccj_count"] = metadata["director_ccj_count"].apply(coerce_numeric).fillna(0)
    if "director_ccj_value" in metadata.columns:
        metadata["director_ccj_value"] = metadata["director_ccj_value"].apply(coerce_numeric).fillna(0)

    duplicate_mask = metadata.duplicated("application_id", keep=False)
    duplicates_df = metadata.loc[duplicate_mask].copy()
    if not metadata.empty:
        score_cols = [col for col in ["requested_loan", "directors_score", "company_age_months"] if col in metadata.columns]
        metadata["_completeness_score"] = metadata.notna().sum(axis=1)
        if score_cols:
            metadata["_critical_score"] = metadata[score_cols].notna().sum(axis=1)
        else:
            metadata["_critical_score"] = 0
        metadata = (
            metadata.sort_values(["application_id", "_critical_score", "_completeness_score"], ascending=[True, False, False])
            .drop_duplicates("application_id", keep="first")
            .drop(columns=["_critical_score", "_completeness_score"], errors="ignore")
        )

    audit["deduped_rows"] = len(metadata)
    audit["duplicate_rows_removed"] = int(len(raw_df) - len(metadata))
    return metadata.reset_index(drop=True), duplicates_df.reset_index(drop=True), audit


def build_metadata_mapping(metadata_df: pd.DataFrame) -> dict:
    """Create company-keyed parameter mapping for the existing processor."""
    mapping = {}
    for _, row in metadata_df.iterrows():
        company = str(row.get("company_name", "")).strip()
        if not company:
            continue
        mapping[company] = {k: v for k, v in row.to_dict().items() if not k.startswith("_")}
    return mapping

# COMPLETE INDUSTRY THRESHOLDS
INDUSTRY_THRESHOLDS = dict(sorted({
    'Medical Practices (GPs, Clinics, Dentists)': {
        'Debt Service Coverage Ratio': 1.60, 'Net Income': 1000, 'Operating Margin': 0.10,
        'Revenue Growth Rate':  0.06, 'Cash Flow Volatility': 0.08, 'Gross Burn Rate': 16000,
        'Directors Score':  75, 'Sector Risk': 0, 'Average Month-End Balance': 900,
        'Average Negative Balance Days per Month': 3, 'Number of Bounced Payments': 1,
    },
    'Pharmacies (Independent or Small Chains)': {
        'Debt Service Coverage Ratio': 1.60, 'Net Income': 500, 'Operating Margin': 0.08,
        'Revenue Growth Rate':  0.05, 'Cash Flow Volatility': 0.08, 'Gross Burn Rate': 15000,
        'Directors Score':  75, 'Sector Risk': 0, 'Average Month-End Balance': 750,
        'Average Negative Balance Days per Month': 3, 'Number of Bounced Payments': 1,
    },
    'Business Consultants': {
        'Debt Service Coverage Ratio': 1.60, 'Net Income': 1000, 'Operating Margin': 0.09,
        'Revenue Growth Rate':  0.05, 'Cash Flow Volatility': 0.08, 'Gross Burn Rate': 14000,
        'Directors Score':  75, 'Sector Risk': 0, 'Average Month-End Balance': 750,
        'Average Negative Balance Days per Month': 3, 'Number of Bounced Payments': 1,
    },
    'IT Services and Support Companies': {
        'Debt Service Coverage Ratio': 1.55, 'Net Income': 500, 'Operating Margin': 0.12,
        'Revenue Growth Rate':  0.07, 'Cash Flow Volatility': 0.10, 'Gross Burn Rate': 13000,
        'Directors Score':  75, 'Sector Risk': 0, 'Average Month-End Balance': 700,
        'Average Negative Balance Days per Month': 3, 'Number of Bounced Payments': 1,
    },
    'Courier Services (Independent and Regional Operators)': {
        'Debt Service Coverage Ratio': 1.50, 'Net Income': 500, 'Operating Margin': 0.08,
        'Revenue Growth Rate':  0.06, 'Cash Flow Volatility': 0.10, 'Gross Burn Rate': 12000,
        'Directors Score': 75, 'Sector Risk': 0, 'Average Month-End Balance':  600,
        'Average Negative Balance Days per Month': 3, 'Number of Bounced Payments': 1,
    },
    'Grocery Stores and Mini-Markets': {
        'Debt Service Coverage Ratio': 1.50, 'Net Income':  500, 'Operating Margin': 0.07,
        'Revenue Growth Rate':  0.06, 'Cash Flow Volatility': 0.12, 'Gross Burn Rate': 10000,
        'Directors Score':  75, 'Sector Risk': 0, 'Average Month-End Balance': 600,
        'Average Negative Balance Days per Month': 4, 'Number of Bounced Payments': 1,
    },
    'Education': {
        'Debt Service Coverage Ratio': 1.45, 'Net Income': 1500, 'Operating Margin': 0.10,
        'Revenue Growth Rate':  0.05, 'Cash Flow Volatility': 0.09, 'Gross Burn Rate': 11500,
        'Directors Score': 75, 'Sector Risk': 0, 'Average Month-End Balance': 700,
        'Average Negative Balance Days per Month': 4, 'Number of Bounced Payments': 1,
    },
    'Engineering': {
        'Debt Service Coverage Ratio': 1.55, 'Net Income': 7000, 'Operating Margin':  0.07,
        'Revenue Growth Rate':  0.04, 'Cash Flow Volatility': 0.10, 'Gross Burn Rate': 13000,
        'Directors Score': 75, 'Sector Risk': 0, 'Average Month-End Balance':  650,
        'Average Negative Balance Days per Month': 4, 'Number of Bounced Payments': 1,
    },
    'Estate Agent': {
        'Debt Service Coverage Ratio':  1.50, 'Net Income': 4500, 'Operating Margin':  0.09,
        'Revenue Growth Rate': 0.04, 'Cash Flow Volatility': 0.10, 'Gross Burn Rate': 13000,
        'Directors Score':  75, 'Sector Risk': 0, 'Average Month-End Balance': 650,
        'Average Negative Balance Days per Month': 3, 'Number of Bounced Payments': 1,
    },
    'Food Service': {
        'Debt Service Coverage Ratio': 1.55, 'Net Income': 2500, 'Operating Margin':  0.06,
        'Revenue Growth Rate':  0.06, 'Cash Flow Volatility': 0.12, 'Gross Burn Rate': 11000,
        'Directors Score':  75, 'Sector Risk': 0, 'Average Month-End Balance': 700,
        'Average Negative Balance Days per Month': 4, 'Number of Bounced Payments': 1,
    },
    'Import / Export':  {
        'Debt Service Coverage Ratio': 1.55, 'Net Income': 3000, 'Operating Margin':  0.07,
        'Revenue Growth Rate': 0.06, 'Cash Flow Volatility': 0.15, 'Gross Burn Rate': 11000,
        'Directors Score': 75, 'Sector Risk':  0, 'Average Month-End Balance': 700,
        'Average Negative Balance Days per Month': 4, 'Number of Bounced Payments': 1,
    },
    'Manufacturing': {
        'Debt Service Coverage Ratio': 1.60, 'Net Income': 1500, 'Operating Margin': 0.08,
        'Revenue Growth Rate': 0.05, 'Cash Flow Volatility':  0.11, 'Gross Burn Rate': 13500,
        'Directors Score': 75, 'Sector Risk':  0, 'Average Month-End Balance': 750,
        'Average Negative Balance Days per Month': 4, 'Number of Bounced Payments': 1,
    },
    'Marketing / Advertising / Design': {
        'Debt Service Coverage Ratio': 1.55, 'Net Income': 5000, 'Operating Margin':  0.11,
        'Revenue Growth Rate':  0.06, 'Cash Flow Volatility': 0.11, 'Gross Burn Rate': 13500,
        'Directors Score':  75, 'Sector Risk': 0, 'Average Month-End Balance': 750,
        'Average Negative Balance Days per Month': 4, 'Number of Bounced Payments': 1,
    },
    'Off-Licence Business': {
        'Debt Service Coverage Ratio': 1.55, 'Net Income': 4500, 'Operating Margin':  0.08,
        'Revenue Growth Rate':  0.06, 'Cash Flow Volatility': 0.11, 'Gross Burn Rate': 14000,
        'Directors Score':  75, 'Sector Risk': 0, 'Average Month-End Balance': 650,
        'Average Negative Balance Days per Month': 4, 'Number of Bounced Payments': 1,
    },
    'Telecommunications': {
        'Debt Service Coverage Ratio': 1.55, 'Net Income': 5000, 'Operating Margin':  0.11,
        'Revenue Growth Rate': 0.05, 'Cash Flow Volatility': 0.12, 'Gross Burn Rate': 13000,
        'Directors Score':  75, 'Sector Risk': 0, 'Average Month-End Balance': 700,
        'Average Negative Balance Days per Month': 4, 'Number of Bounced Payments': 1,
    },
    'Tradesman':  {
        'Debt Service Coverage Ratio': 1.40, 'Net Income': 4000, 'Operating Margin': 0.08,
        'Revenue Growth Rate':  0.05, 'Cash Flow Volatility': 0.15, 'Gross Burn Rate': 11500,
        'Directors Score': 75, 'Sector Risk':  0, 'Average Month-End Balance': 700,
        'Average Negative Balance Days per Month': 5, 'Number of Bounced Payments': 1,
    },
    'Wholesaler / Distributor': {
        'Debt Service Coverage Ratio': 1.50, 'Net Income': 3500, 'Operating Margin': 0.10,
        'Revenue Growth Rate':  0.06, 'Cash Flow Volatility': 0.13, 'Gross Burn Rate': 13000,
        'Directors Score':  75, 'Sector Risk': 0, 'Average Month-End Balance': 700,
        'Average Negative Balance Days per Month': 4, 'Number of Bounced Payments': 1,
    },
    'Other': {
        'Debt Service Coverage Ratio': 1.50, 'Net Income': 3000, 'Operating Margin':  0.08,
        'Revenue Growth Rate':  0.05, 'Cash Flow Volatility': 0.13, 'Gross Burn Rate': 11000,
        'Directors Score': 75, 'Sector Risk': 1, 'Average Month-End Balance': 700,
        'Average Negative Balance Days per Month': 4, 'Number of Bounced Payments': 1,
    },
    'Personal Services': {
        'Debt Service Coverage Ratio': 1.50, 'Net Income': 2000, 'Operating Margin':  0.09,
        'Revenue Growth Rate': 0.05, 'Cash Flow Volatility': 0.12, 'Gross Burn Rate': 12000,
        'Directors Score':  75, 'Sector Risk': 1, 'Average Month-End Balance': 700,
        'Average Negative Balance Days per Month': 4, 'Number of Bounced Payments': 1,
    },
    'Restaurants and Cafes': {
        'Debt Service Coverage Ratio': 1.30, 'Net Income': 0, 'Operating Margin': 0.05,
        'Revenue Growth Rate':  0.04, 'Cash Flow Volatility': 0.16, 'Gross Burn Rate': 11000,
        'Directors Score': 75, 'Sector Risk':  1, 'Average Month-End Balance': 600,
        'Average Negative Balance Days per Month': 5, 'Number of Bounced Payments': 1,
    },
    'Bars and Pubs': {
        'Debt Service Coverage Ratio':  1.25, 'Net Income': 0, 'Operating Margin': 0.04,
        'Revenue Growth Rate': 0.03, 'Cash Flow Volatility':  0.18, 'Gross Burn Rate': 10000,
        'Directors Score': 75, 'Sector Risk': 1, 'Average Month-End Balance': 600,
        'Average Negative Balance Days per Month': 5, 'Number of Bounced Payments': 1,
    },
    'Beauty Salons and Spas': {
        'Debt Service Coverage Ratio': 1.30, 'Net Income': 500, 'Operating Margin': 0.06,
        'Revenue Growth Rate':  0.04, 'Cash Flow Volatility': 0.14, 'Gross Burn Rate': 9500,
        'Directors Score': 75, 'Sector Risk':  1, 'Average Month-End Balance': 550,
        'Average Negative Balance Days per Month': 5, 'Number of Bounced Payments': 1,
    },
    'E-Commerce Retailers': {
        'Debt Service Coverage Ratio': 1.35, 'Net Income': 1000, 'Operating Margin':  0.07,
        'Revenue Growth Rate':  0.05, 'Cash Flow Volatility': 0.14, 'Gross Burn Rate': 10000,
        'Directors Score': 75, 'Sector Risk': 1, 'Average Month-End Balance': 600,
        'Average Negative Balance Days per Month': 4, 'Number of Bounced Payments': 1,
    },
    'Event Planning and Management Firms': {
        'Debt Service Coverage Ratio': 1.30, 'Net Income': 500, 'Operating Margin': 0.05,
        'Revenue Growth Rate':  0.03, 'Cash Flow Volatility': 0.16, 'Gross Burn Rate': 10000,
        'Directors Score':  75, 'Sector Risk': 1, 'Average Month-End Balance': 600,
        'Average Negative Balance Days per Month': 4, 'Number of Bounced Payments': 1,
    },
    'Auto Repair Shops': {
        'Debt Service Coverage Ratio': 1.40, 'Net Income': 1000, 'Operating Margin':  0.08,
        'Revenue Growth Rate': 0.05, 'Cash Flow Volatility': 0.12, 'Gross Burn Rate': 9500,
        'Directors Score': 75, 'Sector Risk':  1, 'Average Month-End Balance': 650,
        'Average Negative Balance Days per Month': 4, 'Number of Bounced Payments': 1,
    },
    'Fitness Centres and Gyms': {
        'Debt Service Coverage Ratio': 1.35, 'Net Income': 500, 'Operating Margin': 0.06,
        'Revenue Growth Rate': 0.04, 'Cash Flow Volatility':  0.18, 'Gross Burn Rate': 10000,
        'Directors Score':  75, 'Sector Risk': 1, 'Average Month-End Balance': 600,
        'Average Negative Balance Days per Month': 4, 'Number of Bounced Payments': 1,
    },
    'Construction Firms': {
        'Debt Service Coverage Ratio': 1.30, 'Net Income': 1000, 'Operating Margin':  0.08,
        'Revenue Growth Rate':  0.04, 'Cash Flow Volatility': 0.16, 'Gross Burn Rate': 12500,
        'Directors Score': 75, 'Sector Risk':  1, 'Average Month-End Balance': 700,
        'Average Negative Balance Days per Month': 5, 'Number of Bounced Payments': 1,
    },
    'Printing / Publishing': {
        'Debt Service Coverage Ratio': 1.50, 'Net Income': 2500, 'Operating Margin': 0.08,
        'Revenue Growth Rate':  0.05, 'Cash Flow Volatility': 0.14, 'Gross Burn Rate': 11000,
        'Directors Score': 75, 'Sector Risk': 1, 'Average Month-End Balance': 650,
        'Average Negative Balance Days per Month': 4, 'Number of Bounced Payments': 1,
    },
    'Recruitment':  {
        'Debt Service Coverage Ratio': 1.50, 'Net Income': 2000, 'Operating Margin': 0.09,
        'Revenue Growth Rate': 0.05, 'Cash Flow Volatility':  0.10, 'Gross Burn Rate': 13000,
        'Directors Score':  75, 'Sector Risk': 1, 'Average Month-End Balance': 600,
        'Average Negative Balance Days per Month': 4, 'Number of Bounced Payments': 1,
    },
    'Retail': {
        'Debt Service Coverage Ratio': 1.40, 'Net Income': 2500, 'Operating Margin': 0.09,
        'Revenue Growth Rate':  0.04, 'Cash Flow Volatility': 0.14, 'Gross Burn Rate': 11500,
        'Directors Score': 75, 'Sector Risk': 1, 'Average Month-End Balance': 620,
        'Average Negative Balance Days per Month': 4, 'Number of Bounced Payments': 1,
    },
}. items()))

# Scoring weights
WEIGHTS = {
    'Debt Service Coverage Ratio': 19, 'Net Income': 13, 'Operating Margin': 9,
    'Revenue Growth Rate': 5, 'Cash Flow Volatility': 12, 'Gross Burn Rate': 3,
    'Company Age (Months)': 4, 'Directors Score': 18, 'Sector Risk': 3,
    'Average Month-End Balance': 5, 'Average Negative Balance Days per Month': 6,
    'Number of Bounced Payments': 3,
}

PENALTIES = {
    "business_ccj": 12,
    "director_ccj": 8,
    'poor_or_no_online_presence': 4,
    'uses_generic_email':  2
}

# -----------------------------
# MCA Transparent Rule Engine
# -----------------------------
MCA_RULES = {
    # These are your 3 core signals — keep them easy to tune
    "cash_flow_volatility_max": 0.35,                 # lower is better
    "avg_negative_balance_days_max": 6,               # lower is better
    "bounced_payments_max": 2,                        # lower is better

    # Scoring model: start at 100 and subtract for breaches
    "penalty_volatility": 40,
    "penalty_negative_days": 30,
    "penalty_bounced": 30,

    # Decision bands (transparent + adjustable)
    "approve_min_score": 75,
    "refer_min_score": 50,
}

def evaluate_mca_rule(metrics: dict, params: dict) -> dict:
    """
    Returns:
      {
        "mca_rule_score": int,
        "mca_rule_decision": "APPROVE"|"REFER"|"DECLINE",
        "mca_rule_reasons": [str, ...]
      }
    """
    score = 100
    reasons = []

    vol = float(metrics.get("Cash Flow Volatility", 0) or 0)
    neg_days = float(metrics.get("Average Negative Balance Days per Month", 0) or 0)
    bounced = float(metrics.get("Number of Bounced Payments", 0) or 0)

    if vol > MCA_RULES["cash_flow_volatility_max"]:
        score -= MCA_RULES["penalty_volatility"]
        reasons.append(
            f"Cash Flow Volatility {vol:.2f} > {MCA_RULES['cash_flow_volatility_max']:.2f}"
        )

    if neg_days > MCA_RULES["avg_negative_balance_days_max"]:
        score -= MCA_RULES["penalty_negative_days"]
        reasons.append(
            f"Avg Negative Balance Days {neg_days:.0f} > {MCA_RULES['avg_negative_balance_days_max']}"
        )

    if bounced > MCA_RULES["bounced_payments_max"]:
        score -= MCA_RULES["penalty_bounced"]
        reasons.append(
            f"Bounced Payments {bounced:.0f} > {MCA_RULES['bounced_payments_max']}"
        )

    score = max(0, min(100, int(round(score))))

    if score >= MCA_RULES["approve_min_score"]:
        decision = "APPROVE"
    elif score >= MCA_RULES["refer_min_score"]:
        decision = "REFER"
    else:
        decision = "DECLINE"

    if not reasons:
        reasons.append("All MCA core signals within threshold.")

    return {
        "mca_rule_score": score,
        "mca_rule_decision": decision,
        "mca_rule_reasons": reasons
    }


def map_transaction_category(transaction: Dict[str, Any]) -> str:
    """
    Enhanced transaction categorization matching original version.

    This is the canonical categorization function for the main app.
    Includes comprehensive patterns for payment processors, loan providers,
    and business expenses.

    IMPORTANT: The order of checks matters! Reversal/failed payment detection
    must happen BEFORE income/loan pattern matching to prevent misclassification.

    Args:
        transaction: Dictionary containing transaction data

    Returns:
        Category string
    """
    name = transaction.get("name", "")
    if isinstance(name, list):
        name = " ".join(map(str, name))
    else:
        name = str(name)
    name = name.lower()

    description = transaction.get("merchant_name", "")
    if isinstance(description, list):
        description = " ".join(map(str, description))
    else:
        description = str(description)
    description = description.lower()

    category = transaction.get("personal_finance_category.detailed", "")
    if isinstance(category, list):
        category = " ".join(map(str, category))
    else:
        category = str(category)
    category = category.lower().strip().replace(" ", "_")

    amount = transaction.get("amount", 0)
    amount_original = transaction.get("amount_original", amount)
    transaction_type = str(transaction.get("transaction_type", "")).lower()
    transaction_name = str(transaction.get("transaction_name", "")).lower()
    combined_text = f"{name} {transaction_name} {description}"
    normalized_text = combined_text.replace("_", " ")

    is_credit = amount_original < 0  # Money coming in (negative in Plaid)
    is_debit = amount_original > 0  # Money going out (positive in Plaid)

    if transaction_type in ("credit", "deposit", "refund"):
        is_credit = True
        is_debit = False
    elif transaction_type in ("debit", "withdrawal", "payment"):
        is_debit = True
        is_credit = False

    if not is_credit and category.startswith("transfer_in_"):
        is_credit = True
        is_debit = False

    # STEP 1 (CRITICAL): Failed payment and reversal patterns - MUST CHECK FIRST!
    # These patterns should take precedence over income/loan patterns to prevent
    # misclassifying reversals like "STRIPE REVERSAL" as income
    failed_payment_patterns = (
        r"(unpaid|returned|bounced|insufficient\s+funds|nsf|declined|failed|"
        r"reversed|reversal|chargeback|refund\s+fee|dispute|unp\b|"
        r"rejected|cancelled\s+payment|payment\s+returned)"
    )
    if re.search(failed_payment_patterns, combined_text, re.IGNORECASE):
        return "Failed Payment"

    # Also check Plaid categories for failed payments first
    if category in ("bank_fees_insufficient_funds", "bank_fees_late_payment",
                    "bank_fees_overdraft", "bank_fees_returned_payment"):
        return "Failed Payment"

    # STEP 2: Handle refunds/credits that look like expenses
    # If it's a credit with expense-like Plaid category, it's likely a refund
    refund_indicators = r"(refund|rebate|credit\s+adj|adjustment|cashback|reimburs)"
    if is_credit and re.search(refund_indicators, combined_text, re.IGNORECASE):
        return "Special Inflow"

    # STEP 3: Custom keyword overrides - Payment processors and income sources
    # Only apply if this is a credit (money coming in) and NOT a reversal
    if is_credit and re.search(
        r"(?i)\b("
        r"stripe|sumup|zettle|square|take\s*payments|shopify|card\s+settlement|daily\s+takings|payout"
        r"|paypal|go\s*cardless|klarna|worldpay|izettle|ubereats|just\s*eat|deliveroo|uber|bolt"
        r"|fresha|treatwell|taskrabbit|terminal|pos\s+deposit|revolut"
        r"|capital\s+one|evo\s*payments?|tink|teya(\s+solutions)?|talech"
        r"|barclaycard|elavon|adyen|payzone|verifone|ingenico"
        r"|nmi|trust\s+payments?|global\s+payments?|checkout\.com|epdq|santander|handepay"
        r"|dojo|valitor|paypoint|mypos|moneris|paymentsense"
        r"|merchant\s+services|payment\s+sense"
        r"|bcard\d*\s*bcard|bcard\d+|bcard\s+\d+"
        r")\b",
        combined_text
    ):
        return "Income"

    # STEP 3.25: Disbursement credits should be treated as loans
    if re.search(r"disbursement", normalized_text, re.IGNORECASE):
        if is_credit or category.startswith("transfer_in_") or transaction_type in ("credit", "deposit", "refund"):
            return "Loans"

    # STEP 3.5: YouLend special handling (before general loan patterns)
    if is_credit and re.search(r"(you\s?lend|yl\s?ii|yl\s?ltd|yl\s?limited|yl\s?a\s?limited|\byl\b)", combined_text, flags=re.IGNORECASE):
        # Check if it contains funding indicators (including within reference numbers)
        if re.search(r"(fnd|fund|funding)", combined_text):
            return "Loans"
        else:
            return "Income"

    # STEP 4: Loan providers (credits = loans received)
    if is_credit and re.search(
            r'\biwoca\b|\bcapify\b|\bfundbox\b|\bgot[\s\-]?capital\b|\bfunding[\s\-]?circle\b|'
        r'\bfleximize\b|\bmarketfinance\b|\bliberis\b|\besme[\s\-]?loans\b|\bthincats\b|'
        r'\bwhite[\s\-]?oak\b|\bgrowth[\s\-]?street\b|\bnucleus[\s\-]?commercial[\s\-]?finance\b|'
        r'\bultimate[\s\-]?finance\b|\bjust[\s\-]?cash[\s\-]?flow\b|\bboost[\s\-]?capital\b|'
        r'\bmerchant[\s\-]?money\b|\bcapital[\s\-]?on[\s\-]?tap\b|\bkriya\b|\buncapped\b|'
        r'\blendingcrowd\b|\bfolk2folk\b|\bfunding[\s\-]?tree\b|\bstart[\s\-]?up[\s\-]?loans\b|'
        r'\bbcrs[\s\-]?business[\s\-]?loans\b|\bbusiness[\s\-]?enterprise[\s\-]?fund\b|'
        r'\bswig[\s\-]?finance\b|\benterprise[\s\-]?answers\b|\blet\'s[\s\-]?do[\s\-]?business[\s\-]?finance\b|'
        r'\bfinance[\s\-]?for[\s\-]?enterprise\b|\bdsl[\s\-]?business[\s\-]?finance\b|'
        r'\bbizcap[\s\-]?uk\b|\bsigma[\s\-]?lending\b|\bbizlend[\s\-]?ltd\b|\bcubefunder\b|'
        r'\bbarclays\b|\bnatwest\b|\bhsbc\b|\blloyds\b|\bsantander\b|'
        r'\bmetro[\s\-]?bank\b|\broyal[\s\-]?bank[\s\-]?of[\s\-]?scotland\b|\brbs\b|'
        r'\bstarling\b|\bzempler\b|\boak[\s\-]?north\b|\ballica\b|\bmonzo\b|\brevolut\b|'
        r'\bfunding[\s\-]?agent\b|\bnationwide[\s\-]?finance\b|\bspotcap\b|'
        r'\btime[\s\-]?finance\b|\btogether\b|\bcorporate[\s\-]?asset[\s\-]?solutions\b|'
        r'\bcreative[\s\-]?capital\b|\bcredit4\b|\bcrowd2fund\b|\bfgi[\s\-]?finance\b|'
        r'\bhampshire[\s\-]?trust[\s\-]?bank\b|\bhodge[\s\-]?bank\b|'
        r'\bigf[\s\-]?invoice[\s\-]?finance\b|\binvestec\b|\blendinvest\b|'
        r'\bmaslow[\s\-]?capital\b|\bmycashline\b|\boctane[\s\-]?capital\b|'
        r'\bsecure[\s\-]?trust[\s\-]?bank\b|\bsme[\s\-]?capital\b|\bswishfund\b|'
        r'\bgrowth[\s\-]?guarantee[\s\-]?scheme\b|\bbritish[\s\-]?business[\s\-]?bank\b|'
        r'\bcommunity[\s\-]?development[\s\-]?finance\b|\bcdfi\b|'
        r'\beveryday[\s\-]?people[\s\-]?financ(?:e)?\b|'
        r'\bloans?\b|\bdisbursement\b|\byou\s?lend\b|\byl\b',
        combined_text,
        flags=re.IGNORECASE
    ):
        return "Loans"

    # STEP 5: Loan repayments (debits to loan providers)
    if is_debit and re.search(
            r"\biwoca\b|\bcapify\b|\bfundbox\b|\bgot[\s\-]?capital\b|\bfunding[\s\-]?circle\b|\bfleximize\b|\bmarketfinance\b|\bliberis\b|"
        r"\besme[\s\-]?loans\b|\bthincats\b|\bwhite[\s\-]?oak\b|\bgrowth[\s\-]?street\b|\bnucleus[\s\-]?commercial[\s\-]?finance\b|"
        r"\bultimate[\s\-]?finance\b|\bjust[\s\-]?cash[\s\-]?flow\b|\bboost[\s\-]?capital\b|\bmerchant[\s\-]?money\b|"
        r"\bcapital[\s\-]?on[\s\-]?tap\b|\bkriya\b|\buncapped\b|\blendingcrowd\b|\bfolk2folk\b|\bfunding[\s\-]?tree\b|"
        r"\bstart[\s\-]?up[\s\-]?loans\b|\bbcrs[\s\-]?business[\s\-]?loans\b|\bbusiness[\s\-]?enterprise[\s\-]?fund\b|"
        r"\bswig[\s\-]?finance\b|\benterprise[\s\-]?answers\b|\blet's[\s\-]?do[\s\-]?business[\s\-]?finance\b|"
        r"\bfinance[\s\-]?for[\s\-]?enterprise\b|\bdsl[\s\-]?business[\s\-]?finance\b|\bbizcap[\s\-]?uk\b|"
        r"\bsigma[\s\-]?lending\b|\bbizlend[\s\-]?ltd\b|\bcubefunder\b|"
        r"\bbarclays\b|\bnatwest\b|\bhsbc\b|\blloyds\b|\bsantander\b|"
        r"\bmetro[\s\-]?bank\b|\broyal[\s\-]?bank[\s\-]?of[\s\-]?scotland\b|\brbs\b|"
        r"\bstarling\b|\bzempler\b|\boak[\s\-]?north\b|\ballica\b|\bmonzo\b|\brevolut\b|"
        r"\bfunding[\s\-]?agent\b|\bnationwide[\s\-]?finance\b|\bspotcap\b|"
        r"\btime[\s\-]?finance\b|\btogether\b|\bcorporate[\s\-]?asset[\s\-]?solutions\b|"
        r"\bcreative[\s\-]?capital\b|\bcredit4\b|\bcrowd2fund\b|\bfgi[\s\-]?finance\b|"
        r"\bhampshire[\s\-]?trust[\s\-]?bank\b|\bhodge[\s\-]?bank\b|"
        r"\bigf[\s\-]?invoice[\s\-]?finance\b|\binvestec\b|\blendinvest\b|"
        r"\bmaslow[\s\-]?capital\b|\bmycashline\b|\boctane[\s\-]?capital\b|"
        r"\bsecure[\s\-]?trust[\s\-]?bank\b|\bsme[\s\-]?capital\b|\bswishfund\b|"
        r"\bgrowth[\s\-]?guarantee[\s\-]?scheme\b|\bbritish[\s\-]?business[\s\-]?bank\b|"
        r"\bcommunity[\s\-]?development[\s\-]?finance\b|\bcdfi\b|"
        r'\beveryday[\s\-]?people[\s\-]?financ(?:e)?\b|'
        r"\bloan[\s\-]?repayment\b|\bdebt[\s\-]?repayment\b|\binstal?ments?\b|\bpay[\s\-]+back\b|\brepay(?:ing|ment|ed)?\b|"
        r"\byou\s?lend\b|\byl\b",
        combined_text,
        flags=re.IGNORECASE
    ):
        return "Debt Repayments"

    # STEP 6: Business expense override (before Plaid fallback)
    # Only apply to debits to prevent expense refunds being marked as expenses
    if is_debit and re.search(
            r"(facebook|facebk|fb\.me|outlook|office365|microsoft|google\s+ads|linkedin|twitter|adobe|zoom|slack|shopify|wix|squarespace|mailchimp|hubspot|hmrc\s*vat|hmrc|hm\s*revenue|hm\s*customs)",
            combined_text, re.IGNORECASE):
        return "Expenses"

    # STEP 7: Plaid category fallback with validation
    plaid_map = {
        "income_wages": "Income",
        "income_other_income": "Income",
        "income_dividends": "Special Inflow",
        "income_interest_earned": "Special Inflow",
        "income_retirement_pension": "Special Inflow",
        "income_unemployment": "Special Inflow",

        "transfer_in_cash_advances_and_loans": "Loans",
        "transfer_in_investment_and_retirement_funds": "Special Inflow",
        "transfer_in_savings": "Special Inflow",
        "transfer_in_account_transfer": "Special Inflow",
        "transfer_in_other_transfer_in": "Special Inflow",
        "transfer_in_deposit": "Special Inflow",

        "transfer_out_investment_and_retirement_funds": "Special Outflow",
        "transfer_out_savings": "Special Outflow",
        "transfer_out_other_transfer_out": "Special Outflow",
        "transfer_out_withdrawal": "Special Outflow",
        "transfer_out_account_transfer": "Special Outflow",

        "bank_fees_insufficient_funds": "Failed Payment",
        "bank_fees_late_payment": "Failed Payment",

        "loan_disbursements_auto": "Loans",
        "loan_disbursements_bnpl": "Loans",
        "loan_disbursements_cash_advances": "Loans",
        "loan_disbursements_ewa": "Loans",
        "loan_disbursements_mortgage": "Loans",
        "loan_disbursements_personal": "Loans",
        "loan_disbursements_student": "Loans",
        "loan_disbursements_other_disbursement": "Loans",

        "loan_payments_car_payment": "Debt Repayments",
        "loan_payments_cash_advances": "Debt Repayments",
        "loan_payments_credit_card_payment": "Debt Repayments",
        "loan_payments_ewa": "Debt Repayments",
        "loan_payments_mortgage_payment": "Debt Repayments",
        "loan_payments_personal_loan_payment": "Debt Repayments",
        "loan_payments_student_loan_payment": "Debt Repayments",
        "loan_payments_other_payment": "Debt Repayments",
    }

    # Handle loan payment categories with validation
    if category.startswith("loan_payments_"):
        # Only trust Plaid if transaction contains actual loan/debt keywords
        if re.search(r"(loan|debt|repay|finance|lending|credit|iwoca|capify|fundbox|you\s?lend|\byl\b)", combined_text,
                     re.IGNORECASE):
            return "Debt Repayments"
        # Otherwise, don't trust Plaid and continue to other checks

    # Match exact key
    if category in plaid_map:
        return plaid_map[category]

    # STEP 8: Fallback for Plaid broad categories
    # CRITICAL FIX: Only apply expense categories to DEBIT transactions
    # Credits with expense-like categories are likely refunds
    expense_category_prefixes = [
        "bank_fees_", "entertainment_", "food_and_drink_", "general_merchandise_",
        "general_services_", "government_and_non_profit_", "home_improvement_",
        "medical_", "personal_care_", "rent_and_utilities_", "transportation_", "travel_"
    ]

    if any(category.startswith(p) for p in expense_category_prefixes):
        if is_debit:
            return "Expenses"
        else:
            # Credit with expense-like category = refund
            return "Special Inflow"

    # STEP 9: Default fallback based on amount direction
    # Debit transactions become Expenses, credit transactions stay Uncategorised
    if is_debit:
        return "Expenses"
    else:
        return "Uncategorised"


def categorize_transactions(data):
    """Apply categorization"""
    if data.empty:
        return data

    data = data.copy()
    data['subcategory'] = data.apply(map_transaction_category, axis=1)
    data['is_revenue'] = data['subcategory'].isin(['Income', 'Special Inflow'])
    data['is_expense'] = data['subcategory'].isin(['Expenses', 'Special Outflow'])
    data['is_debt_repayment'] = data['subcategory'].isin(['Debt Repayments'])
    data['is_debt'] = data['subcategory'].isin(['Loans'])
    data['is_failed_payment'] = data['subcategory'].isin(['Failed Payment'])
    data['is_transfer_in'] = data['subcategory'].isin(['Transfer In'])
    data['is_transfer_out'] = data['subcategory'].isin(['Transfer Out'])
    data['is_funding_injection'] = data['subcategory'].isin(['Funding Inflow'])
    data['is_bank_charge'] = data['subcategory'].isin(['Bank Charge'])
    data['is_special_inflow'] = data['subcategory'].isin(['Special Inflow'])

    return data

# FINANCIAL METRICS CALCULATION
def calculate_financial_metrics(data, company_age_months):
    """Calculate comprehensive financial metrics - ENHANCED VERSION"""
    if data.empty:
        return {}

    try:
        data = categorize_transactions(data)

        # FIXED: Use absolute values for all amounts
        total_revenue = abs(data.loc[data['is_revenue'], 'amount'].sum()) if data['is_revenue'].any() else 0
        total_expenses = abs(data.loc[data['is_expense'], 'amount'].sum()) if data['is_expense'].any() else 0
        net_income = total_revenue - total_expenses
        total_debt_repayments = abs(data.loc[data['is_debt_repayment'], 'amount'].sum()) if data[
            'is_debt_repayment'].any() else 0
        total_debt = abs(data.loc[data['is_debt'], 'amount'].sum()) if data['is_debt'].any() else 0

        # Ensure minimum values to prevent division by zero
        total_revenue = max(total_revenue, 1)  # Minimum £1 to prevent division by zero

        # Time-based calculations
        data['date'] = pd.to_datetime(data['date'])
        data['year_month'] = data['date'].dt.to_period('M')
        unique_months = data['year_month'].nunique()
        months_count = max(unique_months, 1)

        monthly_avg_revenue = total_revenue / months_count

        # Financial ratios - ENHANCED
        debt_to_income_ratio = min(total_debt / total_revenue, 10) if total_revenue > 0 else 0  # Cap at 10x
        expense_to_revenue_ratio = total_expenses / total_revenue if total_revenue > 0 else 1
        operating_margin = max(-1, min(1,
                                       net_income / total_revenue)) if total_revenue > 0 else -1  # Cap between -100% and 100%

        # FIXED: Debt Service Coverage Ratio calculation
        if total_debt_repayments > 0:
            debt_service_coverage_ratio = total_revenue / total_debt_repayments
        elif total_debt > 0:
            # Estimate minimum required payments (10% of debt as annual payment)
            estimated_annual_payment = total_debt * 0.1
            debt_service_coverage_ratio = total_revenue / estimated_annual_payment if estimated_annual_payment > 0 else 0
        else:
            debt_service_coverage_ratio = 10  # No debt = excellent coverage

        # Cap DSCR at reasonable maximum
        debt_service_coverage_ratio = min(debt_service_coverage_ratio, 50)

        # Monthly analysis - ENHANCED
        monthly_summary = data.groupby('year_month').agg({
            'amount': [
                lambda x: abs(x[data.loc[x.index, 'is_revenue']].sum()) if data.loc[x.index, 'is_revenue'].any() else 0,
                lambda x: abs(x[data.loc[x.index, 'is_expense']].sum()) if data.loc[x.index, 'is_expense'].any() else 0
            ]
        }).round(2)

        monthly_summary.columns = ['monthly_revenue', 'monthly_expenses']

        # Volatility metrics - ENHANCED
        if len(monthly_summary) > 1:
            revenue_values = monthly_summary['monthly_revenue']
            revenue_mean = revenue_values.mean()

            if revenue_mean > 0:
                cash_flow_volatility = min(revenue_values.std() / revenue_mean, 2.0)  # Cap at 200%
            else:
                cash_flow_volatility = 0.5  # Default moderate volatility

            # Revenue growth calculation - FIXED
            revenue_growth_changes = revenue_values.pct_change().dropna()
            if len(revenue_growth_changes) > 0:
                # Don't multiply by 100 - store as decimal (0.245 = 24.5%)
                revenue_growth_rate = revenue_growth_changes.median()
                revenue_growth_rate = max(-0.5, min(0.5, revenue_growth_rate))  # Cap between -50% and +50%

                # Debug output
                print(f"  Revenue Growth Rate Calculation:")
                print(f"    Monthly changes: {revenue_growth_changes.tolist()}")

                # Safe formatting with None check
                if revenue_growth_rate is not None:
                    print(f"    Median change: {revenue_growth_rate:.3f} ({revenue_growth_rate * 100:.1f}%)")
                else:
                    print(f"    Median change: None (using 0.0%)")
                    revenue_growth_rate = 0
            else:
                revenue_growth_rate = 0
                print(f"    No growth data available, using 0.0%")

            gross_burn_rate = monthly_summary['monthly_expenses'].mean()
        else:
            cash_flow_volatility = 0.1  # Low volatility for single month
            revenue_growth_rate = 0
            gross_burn_rate = total_expenses / months_count

        # Balance metrics - IMPROVED with realistic estimates
        if 'balances.available' in data.columns and not data['balances.available'].isna().all():
            avg_month_end_balance = data['balances.available'].mean()
        else:
            # Estimate based on revenue and expenses
            monthly_net = (total_revenue - total_expenses) / months_count
            avg_month_end_balance = max(1000, monthly_net * 0.5)  # Conservative estimate

        # Negative balance days - estimated
        if cash_flow_volatility > 0.3:
            avg_negative_days = min(10, int(cash_flow_volatility * 10))
        elif operating_margin < 0:
            avg_negative_days = 3
        else:
            avg_negative_days = 0

        # Bounced payments - scan transaction names
        bounced_payments = 0
        if 'name_y' in data.columns:
            failed_payment_keywords = ['unpaid', 'returned', 'bounced', 'insufficient', 'failed', 'declined', 'nsf',
                                       'unp']
            for keyword in failed_payment_keywords:
                bounced_payments += data['name_y'].str.contains(keyword, case=False, na=False).sum()

        # DEBUGGING: Print key values
        print(f"\nDEBUG - Financial Metrics:")
        print(f"  Total Revenue: £{total_revenue:,.2f}" if total_revenue is not None else "  Total Revenue: N/A")
        print(f"  Total Expenses: £{total_expenses:,.2f}" if total_expenses is not None else "  Total Expenses: N/A")
        print(f"  Net Income: £{net_income:,.2f}" if net_income is not None else "  Net Income: N/A")
        print(
            f"  DSCR: {debt_service_coverage_ratio:.2f}" if debt_service_coverage_ratio is not None else "  DSCR: N/A")
        print(
            f"  Operating Margin: {operating_margin:.3f} ({operating_margin * 100:.1f}%)" if operating_margin is not None else "  Operating Margin: N/A")
        print(
            f"  Cash Flow Volatility: {cash_flow_volatility:.3f}" if cash_flow_volatility is not None else "  Cash Flow Volatility: N/A")
        print(
            f"  Revenue Growth Rate: {revenue_growth_rate:.2f}%" if revenue_growth_rate is not None else "  Revenue Growth Rate: N/A")
        print(
            f"  Avg Month-End Balance: £{avg_month_end_balance:,.2f}" if avg_month_end_balance is not None else "  Avg Month-End Balance: N/A")
        print(
            f"  Avg Negative Days: {avg_negative_days}" if avg_negative_days is not None else "  Avg Negative Days: N/A")
        print(f"  Bounced Payments: {bounced_payments}" if bounced_payments is not None else "  Bounced Payments: N/A")

        metrics = {
            "Total Revenue": round(total_revenue, 2),
            "Monthly Average Revenue": round(monthly_avg_revenue, 2),
            "Total Expenses": round(total_expenses, 2),
            "Net Income": round(net_income, 2),
            "Total Debt Repayments": round(total_debt_repayments, 2),
            "Total Debt": round(total_debt, 2),
            "Debt-to-Income Ratio": round(debt_to_income_ratio, 3),
            "Expense-to-Revenue Ratio": round(expense_to_revenue_ratio, 3),
            "Operating Margin": round(operating_margin, 3),
            "Debt Service Coverage Ratio": round(debt_service_coverage_ratio, 2),
            "Gross Burn Rate": round(gross_burn_rate, 2),
            "Cash Flow Volatility": round(cash_flow_volatility, 3),
            "Revenue Growth Rate": round(revenue_growth_rate, 2),
            "Average Month-End Balance": round(avg_month_end_balance, 2),
            "Average Negative Balance Days per Month": avg_negative_days,
            "Number of Bounced Payments": bounced_payments,
            "monthly_summary": monthly_summary
        }
        if derive_open_banking_insights is not None:
            metrics.update(derive_open_banking_insights(data))
        else:
            metrics["Open Banking Insights Used In Score"] = "No - analysis/export only"
        return metrics

    except Exception as e:
        st.error(f"Error calculating metrics: {e}")
        import traceback
        print(f"Full error traceback: {traceback.format_exc()}")
        return {}

# SCORING CALCULATION FUNCTIONS
def calculate_weighted_scores(metrics, params, industry_thresholds):
    """Calculate weighted score"""
    weighted_score = 0
    for metric, weight in WEIGHTS.items():
        if metric == 'Company Age (Months)':
            if params['company_age_months'] >= 6:
                weighted_score += weight
        elif metric == 'Directors Score':
            if params['directors_score'] >= industry_thresholds['Directors Score']:
                weighted_score += weight
        elif metric == 'Sector Risk':
            sector_risk = industry_thresholds['Sector Risk']
            if sector_risk <= industry_thresholds['Sector Risk']:
                weighted_score += weight
        elif metric in metrics:
            threshold = industry_thresholds.get(metric, 0)
            if metric in ['Cash Flow Volatility', 'Average Negative Balance Days per Month', 'Number of Bounced Payments']:
                if metrics[metric] <= threshold:
                    weighted_score += weight
            else:
                if metrics[metric] >= threshold:
                    weighted_score += weight
    
    # Apply penalties
    penalties = 0
    for flag, penalty in PENALTIES.items():
        if params.get(flag, False):
            penalties += penalty
    
    weighted_score = max(0, weighted_score - penalties)
    return weighted_score

def load_models():
    """Load ML models"""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception as e:
        print(f"ML model artifacts unavailable: {e}")
        return None, None

# TIGHTENED SUBPRIME SCORING CLASS - More Realistic Risk Assessment
class TightenedSubprimeScoring:
    """Enhanced scoring system with tightened thresholds for realistic subprime business lending."""
    
    def __init__(self):
        # Subprime weights - matched to subprime_scoring_system. py
        # For £1-10k short-term lending (6-9 months)
        self.subprime_weights = {
            'Debt Service Coverage Ratio': 25,  # PRIMARY - ability to repay
            'Average Month-End Balance': 18,  # Critical for short terms
            'Directors Score': 16,  # Personal reliability
            'Cash Flow Volatility': 14,  # Stability crucial
            'Revenue Growth Rate': 10,  # Less relevant short term
            'Operating Margin': 6,  # Profitability indicator
            'Average Negative Balance Days per Month': 6,  # Monitor cash gaps
            'Company Age (Months)': 5,  # Business maturity
            'Net Income': 0,
        }

        # Industry multipliers - micro enterprise friendly
        self.industry_multipliers = {
            'Medical Practices (GPs, Clinics, Dentists)': 1.05,
            'IT Services and Support Companies': 1.05,
            'Pharmacies (Independent or Small Chains)': 1.03,
            'Business Consultants': 1.03,
            'Education': 1.02,
            'Engineering': 1.02,
            'Telecommunications': 1.01,
            'Manufacturing': 1.0,
            'Retail': 1.0,
            'Food Service': 0.98,
            'Tradesman': 0.98,
            'Other': 0.97,
            'Restaurants and Cafes': 0.95,
            'Construction Firms': 0.95,
            'Beauty Salons and Spas': 0.95,
            'Bars and Pubs': 0.93,
            'Event Planning and Management Firms': 0.92,
        }

        # Penalty system - balanced for micro enterprises
        self.enhanced_penalties = {
            'business_ccj': 6,
            'director_ccj': 4,
            'poor_or_no_online_presence': 2,
            'uses_generic_email': 1,
        }
    
    def calculate_subprime_score(self, metrics, params):
        """Calculate tightened subprime business score."""
        
        # Calculate base weighted score with TIGHTENED thresholds
        base_score = self._calculate_tightened_base_score(metrics, params)
        
        # Apply industry adjustment (more conservative)
        industry_adjusted_score = self._apply_industry_adjustment(base_score, params.get('industry'))
        
        # Calculate growth momentum bonus (reduced impact)
        growth_bonus = self._calculate_conservative_growth_bonus(metrics)
        
        # Calculate ENHANCED stability penalty 
        stability_penalty = self._calculate_enhanced_stability_penalty(metrics, params)
        
        # Apply ENHANCED risk factor penalties
        risk_penalty = self._calculate_enhanced_risk_penalties(params)
        
        # Final score calculation - much more conservative
        final_score = max(0, min(100, industry_adjusted_score + growth_bonus - stability_penalty - risk_penalty))
        
        # Determine risk tier with TIGHTENED criteria
        risk_tier, pricing_guidance = self._determine_tightened_risk_tier(final_score, metrics, params)
        
        # Generate detailed breakdown
        breakdown = self._generate_tightened_breakdown(
            base_score, industry_adjusted_score, growth_bonus, 
            stability_penalty, risk_penalty, final_score, metrics, params
        )
        
        # Generate comprehensive diagnostics (reuse the same method from SubprimeScoring)
        diagnostics = self._generate_score_diagnostics(metrics, params, final_score)
        
        return {
            'subprime_score': round(final_score, 1),
            'risk_tier': risk_tier,
            'pricing_guidance': pricing_guidance,
            'breakdown': breakdown,
            'recommendation': self._generate_tightened_recommendation(risk_tier, metrics, params),
            'diagnostics': diagnostics
        }

    def _calculate_tightened_base_score(self, metrics, params):
        """Calculate base score - balanced for micro enterprises."""

        score = 0
        max_possible = sum(self.subprime_weights.values())

        # DEBT SERVICE COVERAGE RATIO (25 points) - slightly tightened
        dscr = metrics.get('Debt Service Coverage Ratio', 0)
        w = self.subprime_weights['Debt Service Coverage Ratio']
        if dscr >= 1.9:
            score += w
        elif dscr >= 1.6:
            score += w * 0.85
        elif dscr >= 1.3:
            score += w * 0.65
        elif dscr >= 1.1:
            score += w * 0.45
        elif dscr >= 0.9:
            score += w * 0.25

        # REVENUE GROWTH RATE (10 points) - slightly tightened downside
        growth = metrics.get('Revenue Growth Rate', 0)
        w = self.subprime_weights['Revenue Growth Rate']
        if growth >= 0.12:
            score += w
        elif growth >= 0.06:
            score += w * 0.80
        elif growth >= 0.01:
            score += w * 0.55
        elif growth >= -0.03:
            score += w * 0.30
        elif growth >= -0.07:
            score += w * 0.15

        # DIRECTORS SCORE (16 points) - slightly tightened
        d = params.get('directors_score', 50)
        w = self.subprime_weights['Directors Score']
        if d >= 78:
            score += w
        elif d >= 60:
            score += w * 0.80
        elif d >= 50:
            score += w * 0.55
        elif d >= 40:
            score += w * 0.35
        elif d >= 30:
            score += w * 0.15

        # AVERAGE MONTH-END BALANCE (18 points) - slightly tightened
        bal = metrics.get('Average Month-End Balance', 0)
        w = self.subprime_weights['Average Month-End Balance']
        if bal >= 2500:
            score += w
        elif bal >= 1500:
            score += w * 0.80
        elif bal >= 750:
            score += w * 0.55
        elif bal >= 400:
            score += w * 0.35
        elif bal >= 200:
            score += w * 0.15

        # CASH FLOW VOLATILITY (14 points) - slightly tightened
        vol = metrics.get('Cash Flow Volatility', 1.0)
        w = self.subprime_weights['Cash Flow Volatility']
        if vol <= 0.30:
            score += w
        elif vol <= 0.45:
            score += w * 0.80
        elif vol <= 0.60:
            score += w * 0.55
        elif vol <= 0.75:
            score += w * 0.30
        elif vol <= 0.95:
            score += w * 0.10

        # OPERATING MARGIN (6 points) - slightly tightened
        m = metrics.get('Operating Margin', 0)
        w = self.subprime_weights['Operating Margin']
        if m >= 0.06:
            score += w
        elif m >= 0.04:
            score += w * 0.80
        elif m >= 0.02:
            score += w * 0.60
        elif m >= 0.005:
            score += w * 0.35
        elif m >= -0.02:
            score += w * 0.15

        # NET INCOME (4 points) - slightly tightened
        ni = metrics.get('Net Income', 0)
        w = self.subprime_weights['Net Income']
        if ni >= 3000:
            score += w
        elif ni >= 500:
            score += w * 0.80
        elif ni >= -2500:
            score += w * 0.45
        elif ni >= -10000:
            score += w * 0.25
        elif ni >= -20000:
            score += w * 0.10

        # NEGATIVE BALANCE DAYS (5 points) - slightly tightened
        nd = metrics.get('Average Negative Balance Days per Month', 0)
        w = self.subprime_weights['Average Negative Balance Days per Month']
        if nd <= 1:
            score += w
        elif nd <= 4:
            score += w * 0.80
        elif nd <= 7:
            score += w * 0.55
        elif nd <= 10:
            score += w * 0.30
        elif nd <= 13:
            score += w * 0.10

        # COMPANY AGE (2 points) - Minimal weight
        age_months = params.get('company_age_months', 0)
        # Uses self.subprime_weights directly.
        if age_months >= 18:
            score += self.subprime_weights['Company Age (Months)']
        elif age_months >= 12:
            score += self.subprime_weights['Company Age (Months)'] * 0.80
        elif age_months >= 9:
            score += self.subprime_weights['Company Age (Months)'] * 0.60
        elif age_months >= 6:
            score += self.subprime_weights['Company Age (Months)'] * 0.40
        elif age_months >= 3:
            score += self.subprime_weights['Company Age (Months)'] * 0.20

        # Convert to percentage
        return (score / max_possible) * 100

    def _apply_industry_adjustment(self, base_score, industry):
        """Apply industry-specific risk adjustments - balanced approach."""
        multiplier = self.industry_multipliers.get(industry, 0.95)  # Default reasonable
        return base_score * multiplier

    def _calculate_conservative_growth_bonus(self, metrics):
        """Calculate bonus for growth momentum - micro enterprise friendly."""
        bonus = 0
        growth = metrics.get('Revenue Growth Rate', 0)
        dscr = metrics.get('Debt Service Coverage Ratio', 0)
        volatility = metrics.get('Cash Flow Volatility', 1.0)

        # Achievable criteria for micro enterprises
        if growth >= 0.15 and dscr >= 1.8 and volatility <= 0.35:
            bonus += 5
        elif growth >= 0.10 and dscr >= 1.5 and volatility <= 0.50:
            bonus += 3
        elif growth >= 0.05 and dscr >= 1.2 and volatility <= 0.65:
            bonus += 2
        elif growth >= 0.0 and dscr >= 1.0 and volatility <= 0.80:
            bonus += 1

        return bonus

    def _calculate_enhanced_stability_penalty(self, metrics, params):
        """Calculate penalty for instability - balanced for micro enterprises."""
        penalty = 0
        volatility = metrics.get('Cash Flow Volatility', 0)
        operating_margin = metrics.get('Operating Margin', 0)
        neg_days = metrics.get('Average Negative Balance Days per Month', 0)
        dscr = metrics.get('Debt Service Coverage Ratio', 0)

        # VOLATILITY penalties - only for extreme cases
        if volatility > 1.0:
            penalty += (volatility - 1.0) * 10

        # OPERATING MARGIN penalties - only for significant losses
        if operating_margin < -0.10:
            penalty += abs(operating_margin - (-0.10)) * 30

        # NEGATIVE BALANCE penalties - more forgiving
        if neg_days > 10:
            penalty += (neg_days - 10) * 1.5

        # DSCR penalties - only for very low DSCR
        if dscr < 0.8:
            penalty += (0.8 - dscr) * 8

        return min(penalty, 15)  # Cap at 15 points max

    def _calculate_enhanced_risk_penalties(self, params):
        """Calculate penalties for risk factors with cap to prevent 'death by 1000 cuts'."""
        penalty = 0

        for factor, penalty_points in self.enhanced_penalties.items():
            if params.get(factor, False):
                penalty += penalty_points

        if params.get('business_credit_score_suppressed', False):
            penalty += 2

        if params.get('business_credit_limit') == 0 and params.get('business_max_recommended_credit') == 0:
            penalty += 2

        negative_impact_count = params.get('business_negative_impact_count') or 0
        try:
            negative_impact_count = int(negative_impact_count)
        except (TypeError, ValueError):
            negative_impact_count = 0
        if negative_impact_count >= 4:
            penalty += 2
        elif negative_impact_count >= 2:
            penalty += 1

        enquiries_3m = params.get('business_enquiries_3m') or 0
        try:
            enquiries_3m = int(enquiries_3m)
        except (TypeError, ValueError):
            enquiries_3m = 0
        if enquiries_3m >= 3:
            penalty += 1

        # Cap maximum penalty to prevent excessive stacking - micro enterprise friendly
        max_penalty_cap = 12
        if penalty > max_penalty_cap:
            penalty = max_penalty_cap

        return penalty
    
    def _determine_tightened_risk_tier(self, score, metrics, params):
        """Determine risk tier - MATCHED to main subprime_scoring_system.py"""
        dscr = metrics. get('Debt Service Coverage Ratio', 0)
        growth = metrics.get('Revenue Growth Rate', 0)
        directors_score = params.get('directors_score', 0)
        volatility = metrics.get('Cash Flow Volatility', 1.0)

        has_major_risk_factors = (
                params.get('business_ccj', False) or
                params.get('director_ccj', False)
        )

        # TIER 1 - Premium Subprime
        if (score >= 75 and dscr >= 2.0 and growth >= 0.10 and directors_score >= 70
                and not has_major_risk_factors and volatility <= 0.45):
            return "Tier 1", {
                "risk_level": "Premium Subprime",
                "suggested_rate": "1.5-1.6 factor rate",
                "max_loan_multiple": "4x monthly revenue",
                "term_range": "6-12 months",
                "monitoring": "Monthly reviews",
                "approval_probability": "Very High"
            }

        # TIER 2 - Standard Subprime
        elif (score >= 55 and dscr >= 1.3 and volatility <= 0.70):
            rate_adjustment = "+0.1" if has_major_risk_factors else ""
            return "Tier 2", {
                "risk_level": "Standard Subprime",
                "suggested_rate": f"1.7-1.85{rate_adjustment} factor rate",
                "max_loan_multiple": "3x monthly revenue",
                "term_range": "6-9 months",
                "monitoring": "Bi-weekly reviews" + (" + enhanced due diligence" if has_major_risk_factors else ""),
                "approval_probability": "High" if not has_major_risk_factors else "Moderate-High"
            }

        # TIER 3 - High-Risk Subprime
        elif (score >= 42 and dscr >= 1.0 and directors_score >= 45 and volatility <= 0.85):
            rate_adjustment = "+0.15" if has_major_risk_factors else ""
            return "Tier 3", {
                "risk_level": "High-Risk Subprime",
                "suggested_rate": f"1.85-2.0{rate_adjustment} factor rate",
                "max_loan_multiple": "2.5x monthly revenue",
                "term_range": "4-6 months",
                "monitoring": "Weekly reviews" + (" + continuous risk monitoring" if has_major_risk_factors else ""),
                "approval_probability": "Moderate" if not has_major_risk_factors else "Low-Moderate"
            }

        # TIER 4 - Enhanced Monitoring
        elif (score >= 30 and dscr >= 0.8) or has_major_risk_factors:
            return "Tier 4", {
                "risk_level": "Enhanced Monitoring Required",
                "suggested_rate": "2.0-2.2+ factor rate",
                "max_loan_multiple": "2x monthly revenue",
                "term_range": "3-6 months",
                "monitoring": "Weekly reviews + daily balance monitoring + personal guarantees REQUIRED",
                "approval_probability": "Low - Senior review required"
            }

        # TIER 5 - DECLINE (only for very poor applications)
        else:
            return "Decline", {
                "risk_level": "Below Minimum Standards",
                "suggested_rate": "N/A",
                "max_loan_multiple": "N/A",
                "term_range": "N/A",
                "monitoring": "Application declined - consider reapplying after 3-6 months of improved trading",
                "approval_probability": "None - Declined",
                "decline_reasons": self._get_decline_reasons(score, metrics, params)
            }

    def _generate_tightened_breakdown(self, base_score, industry_score, growth_bonus,
                                      stability_penalty, risk_penalty, final_score, metrics, params):
        """Generate detailed scoring breakdown - micro enterprise focused."""
        breakdown = [
            f"Base Score: {base_score:.1f}/100",
            f"Industry Adjustment: {industry_score - base_score:+.1f} points",
            f"Growth Momentum Bonus: +{growth_bonus:.1f} points",
            f"Stability Penalty: -{stability_penalty:.1f} points",
            f"Risk Factor Penalty: -{risk_penalty:.1f} points",
            f"Final Score: {final_score:.1f}/100",
            "",
            "Key Metrics (Micro Enterprise Thresholds):",
            f"- DSCR: {metrics.get('Debt Service Coverage Ratio', 0):.2f} (Need 1.2+ for good score)",
            f"- Revenue Growth: {metrics.get('Revenue Growth Rate', 0) * 100:.1f}% (Need 5%+ for good score)",
            f"- Directors Score: {params.get('directors_score', 0)}/100 (Need 55+ for good score)",
            f"- Cash Flow Volatility: {metrics.get('Cash Flow Volatility', 0):.3f} (Need <0.50 for good score)",
            f"- Operating Margin: {metrics.get('Operating Margin', 0) * 100:.1f}% (Need 3%+ for good score)",
            f"- Negative Balance Days:  {metrics.get('Average Negative Balance Days per Month', 0):.0f} (Need <5 for good score)"
        ]
        return breakdown

    def _generate_tightened_recommendation(self, risk_tier, metrics, params):
        """Generate lending recommendation - balanced for micro enterprises."""
        if risk_tier == "Tier 1":
            return "APPROVE - Strong micro enterprise candidate with solid fundamentals."
        elif risk_tier == "Tier 2":
            return "APPROVE - Good candidate with standard monitoring requirements."
        elif risk_tier == "Tier 3":
            return "CONDITIONAL APPROVE - Viable candidate with enhanced terms and monitoring."
        elif risk_tier == "Tier 4":
            return "SENIOR REVIEW - Higher risk profile, requires additional review and guarantees."
        else:
            return "DECLINE - Consider reapplying after 3-6 months of improved trading performance."

    def _generate_score_diagnostics(self, metrics: Dict[str, Any], params: Dict[str, Any], final_score: float) -> Dict[str, Any]:
        """
        Generate detailed diagnostics showing why the score is what it is.
        
        This provides comprehensive insights into:
        - How each metric performed
        - What's hurting the score most  
        - What's helping the score
        - Specific improvement suggestions
        """
        
        diagnostics = {
            'metric_breakdown': [],
            'top_negative_factors': [],
            'top_positive_factors': [],
            'threshold_failures': [],
            'improvement_suggestions': []
        }
        
        # Track metric performance with detailed breakdown
        metric_performance = []
        
        # DEBT SERVICE COVERAGE RATIO
        dscr = metrics.get('Debt Service Coverage Ratio', 0)
        dscr_threshold_full = 1.8
        dscr_threshold_min = 0.8
        dscr_max_points = self.subprime_weights['Debt Service Coverage Ratio']
        
        if dscr >= dscr_threshold_full:
            dscr_points = dscr_max_points
            dscr_percentage = 100.0
            dscr_status = 'PASS'
        elif dscr >= 1.5:
            dscr_points = dscr_max_points * 0.85
            dscr_percentage = 85.0
            dscr_status = 'PARTIAL'
        elif dscr >= 1.2:
            dscr_points = dscr_max_points * 0.70
            dscr_percentage = 70.0
            dscr_status = 'PARTIAL'
        elif dscr >= 1.0:
            dscr_points = dscr_max_points * 0.55
            dscr_percentage = 55.0
            dscr_status = 'PARTIAL'
        elif dscr >= dscr_threshold_min:
            dscr_points = dscr_max_points * 0.35
            dscr_percentage = 35.0
            dscr_status = 'PARTIAL'
        else:
            dscr_points = 0
            dscr_percentage = 0
            dscr_status = 'FAIL'
        
        metric_performance.append({
            'metric': 'Debt Service Coverage Ratio',
            'actual_value': dscr,
            'threshold_full_points': dscr_threshold_full,
            'threshold_min_points': dscr_threshold_min,
            'points_earned': round(dscr_points, 2),
            'points_possible': dscr_max_points,
            'percentage': round(dscr_percentage, 1),
            'status': dscr_status,
            'gap_to_full': max(0, dscr_threshold_full - dscr)
        })
        
        # CASH FLOW VOLATILITY
        volatility = metrics.get('Cash Flow Volatility', 1.0)
        vol_threshold_full = 0.35
        vol_threshold_max = 1.0
        vol_max_points = self.subprime_weights['Cash Flow Volatility']
        
        if volatility <= vol_threshold_full:
            vol_points = vol_max_points
            vol_percentage = 100.0
            vol_status = 'PASS'
        elif volatility <= 0.50:
            vol_points = vol_max_points * 0.80
            vol_percentage = 80.0
            vol_status = 'PARTIAL'
        elif volatility <= 0.65:
            vol_points = vol_max_points * 0.60
            vol_percentage = 60.0
            vol_status = 'PARTIAL'
        elif volatility <= 0.80:
            vol_points = vol_max_points * 0.40
            vol_percentage = 40.0
            vol_status = 'PARTIAL'
        elif volatility <= vol_threshold_max:
            vol_points = vol_max_points * 0.20
            vol_percentage = 20.0
            vol_status = 'PARTIAL'
        else:
            vol_points = 0
            vol_percentage = 0
            vol_status = 'FAIL'
        
        metric_performance.append({
            'metric': 'Cash Flow Volatility',
            'actual_value': volatility,
            'threshold_full_points': vol_threshold_full,
            'threshold_min_points': vol_threshold_max,
            'points_earned': round(vol_points, 2),
            'points_possible': vol_max_points,
            'percentage': round(vol_percentage, 1),
            'status': vol_status,
            'gap_to_full': max(0, volatility - vol_threshold_full)
        })
        
        # DIRECTORS SCORE
        directors = params.get('directors_score', 50)
        dir_threshold_full = 70
        dir_threshold_min = 25
        dir_max_points = self.subprime_weights['Directors Score']
        
        if directors >= dir_threshold_full:
            dir_points = dir_max_points
            dir_percentage = 100.0
            dir_status = 'PASS'
        elif directors >= 55:
            dir_points = dir_max_points * 0.80
            dir_percentage = 80.0
            dir_status = 'PARTIAL'
        elif directors >= 45:
            dir_points = dir_max_points * 0.60
            dir_percentage = 60.0
            dir_status = 'PARTIAL'
        elif directors >= 35:
            dir_points = dir_max_points * 0.40
            dir_percentage = 40.0
            dir_status = 'PARTIAL'
        elif directors >= dir_threshold_min:
            dir_points = dir_max_points * 0.20
            dir_percentage = 20.0
            dir_status = 'PARTIAL'
        else:
            dir_points = 0
            dir_percentage = 0
            dir_status = 'FAIL'
        
        metric_performance.append({
            'metric': 'Directors Score',
            'actual_value': directors,
            'threshold_full_points': dir_threshold_full,
            'threshold_min_points': dir_threshold_min,
            'points_earned': round(dir_points, 2),
            'points_possible': dir_max_points,
            'percentage': round(dir_percentage, 1),
            'status': dir_status,
            'gap_to_full': max(0, dir_threshold_full - directors)
        })
        
        # AVERAGE MONTH-END BALANCE
        balance = metrics.get('Average Month-End Balance', 0)
        bal_threshold_full = 2000
        bal_threshold_min = 100
        bal_max_points = self.subprime_weights['Average Month-End Balance']
        
        if balance >= bal_threshold_full:
            bal_points = bal_max_points
            bal_percentage = 100.0
            bal_status = 'PASS'
        elif balance >= 1000:
            bal_points = bal_max_points * 0.80
            bal_percentage = 80.0
            bal_status = 'PARTIAL'
        elif balance >= 500:
            bal_points = bal_max_points * 0.60
            bal_percentage = 60.0
            bal_status = 'PARTIAL'
        elif balance >= 250:
            bal_points = bal_max_points * 0.40
            bal_percentage = 40.0
            bal_status = 'PARTIAL'
        elif balance >= bal_threshold_min:
            bal_points = bal_max_points * 0.20
            bal_percentage = 20.0
            bal_status = 'PARTIAL'
        else:
            bal_points = 0
            bal_percentage = 0
            bal_status = 'FAIL'
        
        metric_performance.append({
            'metric': 'Average Month-End Balance',
            'actual_value': balance,
            'threshold_full_points': bal_threshold_full,
            'threshold_min_points': bal_threshold_min,
            'points_earned': round(bal_points, 2),
            'points_possible': bal_max_points,
            'percentage': round(bal_percentage, 1),
            'status': bal_status,
            'gap_to_full': max(0, bal_threshold_full - balance)
        })
        
        # REVENUE GROWTH RATE
        growth = metrics.get('Revenue Growth Rate', 0)
        growth_threshold_full = 0.10  # 10%
        growth_threshold_min = -0.15  # -15%
        growth_max_points = self.subprime_weights['Revenue Growth Rate']
        
        if growth >= growth_threshold_full:
            growth_points = growth_max_points
            growth_percentage = 100.0
            growth_status = 'PASS'
        elif growth >= 0.05:
            growth_points = growth_max_points * 0.80
            growth_percentage = 80.0
            growth_status = 'PARTIAL'
        elif growth >= 0:
            growth_points = growth_max_points * 0.60
            growth_percentage = 60.0
            growth_status = 'PARTIAL'
        elif growth >= -0.05:
            growth_points = growth_max_points * 0.40
            growth_percentage = 40.0
            growth_status = 'PARTIAL'
        elif growth >= growth_threshold_min:
            growth_points = growth_max_points * 0.20
            growth_percentage = 20.0
            growth_status = 'PARTIAL'
        else:
            growth_points = 0
            growth_percentage = 0
            growth_status = 'FAIL'
        
        metric_performance.append({
            'metric': 'Revenue Growth Rate',
            'actual_value': growth,
            'threshold_full_points': growth_threshold_full,
            'threshold_min_points': growth_threshold_min,
            'points_earned': round(growth_points, 2),
            'points_possible': growth_max_points,
            'percentage': round(growth_percentage, 1),
            'status': growth_status,
            'gap_to_full': max(0, growth_threshold_full - growth)
        })
        
        # OPERATING MARGIN
        margin = metrics.get('Operating Margin', 0)
        margin_threshold_full = 0.05  # 5%
        margin_threshold_min = -0.03  # -3%
        margin_max_points = self.subprime_weights['Operating Margin']
        
        if margin >= margin_threshold_full:
            margin_points = margin_max_points
            margin_percentage = 100.0
            margin_status = 'PASS'
        elif margin >= 0.03:
            margin_points = margin_max_points * 0.80
            margin_percentage = 80.0
            margin_status = 'PARTIAL'
        elif margin >= 0.01:
            margin_points = margin_max_points * 0.60
            margin_percentage = 60.0
            margin_status = 'PARTIAL'
        elif margin >= 0:
            margin_points = margin_max_points * 0.40
            margin_percentage = 40.0
            margin_status = 'PARTIAL'
        elif margin >= margin_threshold_min:
            margin_points = margin_max_points * 0.20
            margin_percentage = 20.0
            margin_status = 'PARTIAL'
        else:
            margin_points = 0
            margin_percentage = 0
            margin_status = 'FAIL'
        
        metric_performance.append({
            'metric': 'Operating Margin',
            'actual_value': margin,
            'threshold_full_points': margin_threshold_full,
            'threshold_min_points': margin_threshold_min,
            'points_earned': round(margin_points, 2),
            'points_possible': margin_max_points,
            'percentage': round(margin_percentage, 1),
            'status': margin_status,
            'gap_to_full': max(0, margin_threshold_full - margin)
        })
        
        # NEGATIVE BALANCE DAYS
        neg_days = metrics.get('Average Negative Balance Days per Month', 0)
        neg_threshold_full = 2
        neg_threshold_max = 15
        neg_max_points = self.subprime_weights['Average Negative Balance Days per Month']
        
        if neg_days <= neg_threshold_full:
            neg_points = neg_max_points
            neg_percentage = 100.0
            neg_status = 'PASS'
        elif neg_days <= 5:
            neg_points = neg_max_points * 0.80
            neg_percentage = 80.0
            neg_status = 'PARTIAL'
        elif neg_days <= 8:
            neg_points = neg_max_points * 0.60
            neg_percentage = 60.0
            neg_status = 'PARTIAL'
        elif neg_days <= 12:
            neg_points = neg_max_points * 0.40
            neg_percentage = 40.0
            neg_status = 'PARTIAL'
        elif neg_days <= neg_threshold_max:
            neg_points = neg_max_points * 0.20
            neg_percentage = 20.0
            neg_status = 'PARTIAL'
        else:
            neg_points = 0
            neg_percentage = 0
            neg_status = 'FAIL'
        
        metric_performance.append({
            'metric': 'Negative Balance Days',
            'actual_value': neg_days,
            'threshold_full_points': neg_threshold_full,
            'threshold_min_points': neg_threshold_max,
            'points_earned': round(neg_points, 2),
            'points_possible': neg_max_points,
            'percentage': round(neg_percentage, 1),
            'status': neg_status,
            'gap_to_full': max(0, neg_days - neg_threshold_full)
        })
        
        # COMPANY AGE
        age = params.get('company_age_months', 0)
        age_threshold_full = 18
        age_threshold_min = 3
        age_max_points = self.subprime_weights['Company Age (Months)']
        
        if age >= age_threshold_full:
            age_points = age_max_points
            age_percentage = 100.0
            age_status = 'PASS'
        elif age >= 12:
            age_points = age_max_points * 0.80
            age_percentage = 80.0
            age_status = 'PARTIAL'
        elif age >= 9:
            age_points = age_max_points * 0.60
            age_percentage = 60.0
            age_status = 'PARTIAL'
        elif age >= 6:
            age_points = age_max_points * 0.40
            age_percentage = 40.0
            age_status = 'PARTIAL'
        elif age >= age_threshold_min:
            age_points = age_max_points * 0.20
            age_percentage = 20.0
            age_status = 'PARTIAL'
        else:
            age_points = 0
            age_percentage = 0
            age_status = 'FAIL'
        
        metric_performance.append({
            'metric': 'Company Age',
            'actual_value': age,
            'threshold_full_points': age_threshold_full,
            'threshold_min_points': age_threshold_min,
            'points_earned': round(age_points, 2),
            'points_possible': age_max_points,
            'percentage': round(age_percentage, 1),
            'status': age_status,
            'gap_to_full': max(0, age_threshold_full - age)
        })
        
        # Add all metrics to breakdown
        diagnostics['metric_breakdown'] = metric_performance
        
        # Calculate TOP NEGATIVE FACTORS (biggest points lost)
        negative_factors = []
        for perf in metric_performance:
            points_lost = perf['points_possible'] - perf['points_earned']
            if points_lost > 0.5:  # Only include if significant loss
                suggestion = self._get_improvement_suggestion(perf)
                negative_factors.append({
                    'metric': perf['metric'],
                    'points_lost': round(points_lost, 2),
                    'suggestion': suggestion
                })
        
        # Sort by points lost and take top 3
        negative_factors.sort(key=lambda x: x['points_lost'], reverse=True)
        diagnostics['top_negative_factors'] = negative_factors[:3]
        
        # Calculate TOP POSITIVE FACTORS (best performers)
        positive_factors = []
        for perf in metric_performance:
            if perf['percentage'] >= 60:  # Only include if doing reasonably well
                status_desc = self._get_status_description(perf)
                positive_factors.append({
                    'metric': perf['metric'],
                    'points_earned': perf['points_earned'],
                    'status': status_desc
                })
        
        # Sort by points earned and take top 3
        positive_factors.sort(key=lambda x: x['points_earned'], reverse=True)
        diagnostics['top_positive_factors'] = positive_factors[:3]
        
        # THRESHOLD FAILURES
        threshold_failures = []
        for perf in metric_performance:
            if perf['status'] == 'FAIL':
                threshold_failures.append({
                    'metric': perf['metric'],
                    'actual': perf['actual_value'],
                    'required_minimum': perf['threshold_min_points'],
                    'impact': f"0 points (vs max {perf['points_possible']})"
                })
        
        diagnostics['threshold_failures'] = threshold_failures
        
        # IMPROVEMENT SUGGESTIONS
        suggestions = []
        
        # Focus on top 2-3 biggest gaps
        for neg_factor in diagnostics['top_negative_factors'][:2]:
            for perf in metric_performance:
                if perf['metric'] == neg_factor['metric']:
                    suggestion = self._create_specific_suggestion(perf, final_score)
                    if suggestion:
                        suggestions.append(suggestion)
        
        # Add tier movement suggestion
        current_tier = self._get_tier_from_score(final_score)
        next_tier_score = self._get_next_tier_threshold(final_score)
        if next_tier_score:
            points_needed = next_tier_score - final_score
            suggestions.append(
                f"You need {points_needed:.1f} more points to move from {current_tier} to the next tier"
            )
        
        diagnostics['improvement_suggestions'] = suggestions
        
        return diagnostics
    
    def _get_improvement_suggestion(self, perf: Dict[str, Any]) -> str:
        """Generate improvement suggestion for a metric"""
        metric = perf['metric']
        actual = perf['actual_value']
        target = perf['threshold_full_points']
        
        if metric == 'Debt Service Coverage Ratio':
            return f"Improve DSCR from {actual:.2f} to {target:.2f} for full points"
        elif metric == 'Cash Flow Volatility':
            return f"Reduce volatility from {actual:.3f} to below {target:.2f}"
        elif metric == 'Directors Score':
            return f"Director score of {int(target)}+ would help (currently {int(actual)})"
        elif metric == 'Average Month-End Balance':
            return f"Increase balance from £{actual:,.0f} to £{target:,.0f}"
        elif metric == 'Revenue Growth Rate':
            return f"Improve growth from {actual*100:.1f}% to {target*100:.1f}%+"
        elif metric == 'Operating Margin':
            return f"Improve margin from {actual*100:.1f}% to above {target*100:.1f}%"
        elif metric == 'Negative Balance Days':
            return f"Reduce negative days from {int(actual)} to {int(target)} or fewer"
        elif metric == 'Company Age':
            return f"Company age will naturally improve ({int(actual)} months currently)"
        else:
            return f"Improve {metric} to meet threshold"
    
    def _get_status_description(self, perf: Dict[str, Any]) -> str:
        """Get status description for a positive factor"""
        percentage = perf['percentage']
        points = perf['points_earned']
        max_points = perf['points_possible']
        
        if percentage >= 95:
            return f"Full points - excellent performance ({points}/{max_points})"
        elif percentage >= 80:
            return f"{percentage:.0f}% - strong performance ({points:.1f}/{max_points})"
        else:
            return f"{percentage:.0f}% - good performance ({points:.1f}/{max_points})"
    
    def _create_specific_suggestion(self, perf: Dict[str, Any], current_score: float) -> str:
        """Create specific improvement suggestion with point impact"""
        metric = perf['metric']
        actual = perf['actual_value']
        gap = perf['gap_to_full']
        points_lost = perf['points_possible'] - perf['points_earned']
        
        if points_lost < 1:
            return None
        
        # Calculate achievable improvement (50% of gap)
        if metric == 'Cash Flow Volatility' or metric == 'Negative Balance Days':
            achievable_improvement = gap * 0.5
            achievable_value = actual - achievable_improvement  # Need to REDUCE these metrics
        else:
            achievable_improvement = gap * 0.5
            achievable_value = actual + achievable_improvement  # Need to INCREASE these metrics
        
        # Estimate point gain (roughly 50% of points lost)
        estimated_gain = points_lost * 0.5
        
        if metric == 'Debt Service Coverage Ratio':
            return f"Improving DSCR from {actual:.2f} to {achievable_value:.2f} would add ~{estimated_gain:.1f} points"
        elif metric == 'Cash Flow Volatility':
            return f"Reducing volatility from {actual:.3f} to {achievable_value:.3f} would add ~{estimated_gain:.1f} points"
        elif metric == 'Directors Score':
            return f"Improving Directors Score from {int(actual)} to {int(achievable_value)} would add ~{estimated_gain:.1f} points"
        elif metric == 'Average Month-End Balance':
            return f"Increasing balance from £{actual:,.0f} to £{achievable_value:,.0f} would add ~{estimated_gain:.1f} points"
        elif metric == 'Revenue Growth Rate':
            return f"Improving growth from {actual*100:.1f}% to {achievable_value*100:.1f}% would add ~{estimated_gain:.1f} points"
        elif metric == 'Operating Margin':
            return f"Improving margin from {actual*100:.1f}% to {achievable_value*100:.1f}% would add ~{estimated_gain:.1f} points"
        else:
            return None
    
    def _get_tier_from_score(self, score: float) -> str:
        """Get tier name from score"""
        if score >= 75:
            return "Tier 1"
        elif score >= 60:
            return "Tier 2"
        elif score >= 45:
            return "Tier 3"
        elif score >= 30:
            return "Tier 4"
        else:
            return "Decline"
    
    def _get_next_tier_threshold(self, score: float) -> float:
        """Get score threshold for next tier"""
        if score < 30:
            return 30  # Tier 4
        elif score < 45:
            return 45  # Tier 3
        elif score < 60:
            return 60  # Tier 2
        elif score < 75:
            return 75  # Tier 1
        else:
            return None  # Already at top tier

    def compare_scoring_methods(self, mca_rule_score: float,
                                ml_score: float, subprime_score: float) -> Dict[str, Any]:
        """Compare MCA/Subprime decision scores and show ML for context."""

        # Convergence based on the two trusted systems only (MCA Rule + Subprime)
        primary_scores = [s for s in [mca_rule_score, subprime_score] if s and s > 0]
        score_range = max(primary_scores) - min(primary_scores) if len(primary_scores) >= 2 else 0

        if score_range <= 15:
            convergence = "High"
        elif score_range <= 30:
            convergence = "Moderate"
        else:
            convergence = "Low"

        # Primary recommendation based on subprime score (most relevant)
        if subprime_score >= 50:
            primary_rec = "Approve with appropriate pricing"
        elif subprime_score >= 40:
            primary_rec = "Conditional approval with enhanced monitoring"
        elif subprime_score >= 30:
            primary_rec = "Senior review required"
        else:
            primary_rec = "Decline - consider reapplying after improved trading"


# UPDATED CALCULATION FUNCTION to use tightened scoring
def calculate_all_scores_tightened(metrics, params):
    """Enhanced scoring calculation with TIGHTENED subprime scoring"""
    industry_thresholds = INDUSTRY_THRESHOLDS[params['industry']]
    sector_risk = industry_thresholds['Sector Risk']
    
    # TIGHTENED subprime scoring
    try:
        tightened_scorer = TightenedSubprimeScoring()
        subprime_result = tightened_scorer.calculate_subprime_score(metrics, params)
    except Exception as e:
        subprime_result = {
            'subprime_score': 0,
            'risk_tier': 'Error',
            'pricing_guidance': {'suggested_rate': 'N/A'},
            'recommendation': f'Subprime scoring failed: {str(e)}',
            'breakdown': [f'Error: {str(e)}']
        }
    
    # Industry Score (simplified for batch processing)
    industry_score = 0
    score_breakdown = {}
    
    # Check each threshold
    for metric, threshold in industry_thresholds.items():
        if metric in ['Directors Score', 'Sector Risk']:
            continue
        
        if metric in metrics:
            actual_value = metrics[metric]
            if metric in ['Cash Flow Volatility', 'Average Negative Balance Days per Month', 'Number of Bounced Payments']:
                meets_threshold = actual_value <= threshold
            else:
                meets_threshold = actual_value >= threshold
            
            if meets_threshold:
                industry_score += 1
            
            score_breakdown[metric] = {
                'actual': actual_value,
                'threshold': threshold,
                'meets': meets_threshold,
                'direction': 'lower' if metric in ['Cash Flow Volatility', 'Average Negative Balance Days per Month', 'Number of Bounced Payments'] else 'higher'
            }
    
    # Add non-metric scores
    if params['company_age_months'] >= 6:
        industry_score += 1
    if params['directors_score'] >= industry_thresholds['Directors Score']:
        industry_score += 1
    if sector_risk <= industry_thresholds['Sector Risk']:
        industry_score += 1
    
    # ML Score - load models if available
    model, scaler = load_models()
    ml_score = None

    _ML_FEATURE_BOUNDS = {
        'Directors Score': (0, 100),
        'Total Revenue': (0, 5_000_000),
        'Total Debt': (0, 2_000_000),
        'Debt-to-Income Ratio': (0, 10),
        'Operating Margin': (-1.0, 1.0),
        'Debt Service Coverage Ratio': (0, 500_000),
        'Cash Flow Volatility': (0, 100),
        'Revenue Growth Rate': (-500, 500),
        'Average Month-End Balance': (-500_000, 500_000),
        'Average Negative Balance Days per Month': (0, 31),
        'Number of Bounced Payments': (0, 100),
        'Company Age (Months)': (0, 600),
        'Sector_Risk': (0, 1),
    }

    if model and scaler:
        try:
            features = {
                'Directors Score': params['directors_score'],
                'Total Revenue': metrics["Total Revenue"],
                'Total Debt': metrics["Total Debt"],
                'Debt-to-Income Ratio': metrics["Debt-to-Income Ratio"],
                'Operating Margin': metrics["Operating Margin"],
                'Debt Service Coverage Ratio': metrics["Debt Service Coverage Ratio"],
                'Cash Flow Volatility': metrics["Cash Flow Volatility"],
                'Revenue Growth Rate': metrics["Revenue Growth Rate"],
                'Average Month-End Balance': metrics["Average Month-End Balance"],
                'Average Negative Balance Days per Month': metrics["Average Negative Balance Days per Month"],
                'Number of Bounced Payments': metrics["Number of Bounced Payments"],
                'Company Age (Months)': params['company_age_months'],
                'Sector_Risk': sector_risk
            }
            
            features_df = pd.DataFrame([features])
            features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            features_df.fillna(0, inplace=True)
            
            for col, (lo, hi) in _ML_FEATURE_BOUNDS.items():
                if col in features_df.columns:
                    features_df[col] = features_df[col].clip(lo, hi)
            
            features_scaled = scaler.transform(features_df)
            # The retrained model is already wrapped in CalibratedClassifierCV
            probability = model.predict_proba(features_scaled)[:, 1][0]
            ml_score = round(probability * 100, 2)
            
        except Exception as e:
            ml_score = None
    
    # Loan Risk (unchanged)
    monthly_revenue = metrics.get('Monthly Average Revenue', 0)
    if monthly_revenue > 0:
        ratio = params['requested_loan'] / monthly_revenue
        if ratio <= 0.7:
            loan_risk = "Low Risk"
        elif ratio <= 1.0:
            loan_risk = "Moderate Low Risk"
        elif ratio <= 1.2:
            loan_risk = "Medium Risk"
        elif ratio <= 1.5:
            loan_risk = "Moderate High Risk"
        else:
            loan_risk = "High Risk"
    else:
        loan_risk = "High Risk"
    
    # -----------------------------
    # MCA transparent rule layer
    # -----------------------------
    mca_rule = evaluate_mca_rule(metrics, params)
    mca_rule_decision = (mca_rule.get("mca_rule_decision") or "").upper().strip()

    # -----------------------------
    # FINAL decision rules (batch)
    # -----------------------------
    # Keep batch aligned with the main app: MCA Rule 60%, Subprime 40%,
    # ML displayed as information only.
    ensemble_result = None
    final_decision = "REFER"
    final_reasons = []

    if ENSEMBLE_SCORER_AVAILABLE and get_ensemble_recommendation:
        ensemble_result = get_ensemble_recommendation(
            scores={
                "mca_score": mca_rule.get("mca_rule_score"),
                "mca_decision": mca_rule.get("mca_rule_decision"),
                "subprime_score": subprime_result["subprime_score"],
                "ml_score": ml_score,
            },
            metrics=metrics,
            params=params,
        )
        final_decision = ensemble_result["decision"]
        final_reasons = [
            f"Ensemble decision: {final_decision}",
            ensemble_result.get("primary_reason", "No primary reason returned"),
        ]
    else:
        def _base_decision_from_subprime(recommendation_text: str) -> str:
            s = (recommendation_text or "").upper()
            if "APPROVE" in s:
                return "APPROVE"
            if "CONDITIONAL" in s or "SENIOR REVIEW" in s or "REVIEW" in s:
                return "REFER"
            return "DECLINE"

        base_decision = _base_decision_from_subprime(subprime_result.get("recommendation", ""))
        final_decision = base_decision
        final_reasons = [f"Base decision from Subprime: {base_decision}"]

        if mca_rule_decision == "DECLINE":
            mca_score = float(mca_rule.get("mca_rule_score") or 0)
            if mca_score <= 20:
                final_decision = "DECLINE"
                final_reasons.append("MCA Rule severe failure: DECLINE hard stop")
            elif base_decision == "APPROVE":
                final_decision = "REFER"
                final_reasons.append("MCA Rule soft decline: capped APPROVE to REFER")
            elif base_decision != "DECLINE":
                final_decision = "REFER"
                final_reasons.append("MCA Rule soft decline: manual review cap")
        elif mca_rule_decision == "REFER" and base_decision != "DECLINE":
            final_decision = "REFER"
            final_reasons.append("MCA Rule cap: REFER (manual review)")
        elif mca_rule_decision == "APPROVE":
            final_reasons.append("MCA Rule supports combined decision but does not approve by itself")

    return {
        'industry_score': industry_score,
        'ml_score': ml_score,
        'ensemble': ensemble_result,
        'combined_score': ensemble_result.get('combined_score') if ensemble_result else None,
        'score_convergence': ensemble_result.get('score_convergence') if ensemble_result else None,
        'loan_risk': loan_risk,
        'score_breakdown': score_breakdown,

        'subprime_score': subprime_result['subprime_score'],
        'subprime_tier': subprime_result['risk_tier'],
        'subprime_pricing': subprime_result['pricing_guidance'],
        'subprime_recommendation': subprime_result['recommendation'],
        'subprime_breakdown': subprime_result['breakdown'],

        # MCA rule outputs
        'mca_rule_score': mca_rule.get('mca_rule_score'),
        'mca_rule_decision': mca_rule.get('mca_rule_decision'),
        'mca_rule_reasons': mca_rule.get('mca_rule_reasons'),

        # Final operating decision
        'final_decision': final_decision,
        'final_decision_reasons': final_reasons,
    }


# FIXED BATCH PROCESSOR CLASS
class BatchProcessor:
    """FIXED: Process multiple loan applications with working fuzzy matching and debug"""
    
    def __init__(self):
        self.results = []
        self.processed_count = 0
        self.error_count = 0
        self.error_log = []
        self.debug_log = []

    def normalize_industry(self, industry):
        """Return an industry label that exists in INDUSTRY_THRESHOLDS."""
        if pd.isna(industry) or industry is None:
            return "Other"

        industry_text = str(industry).strip()
        if not industry_text:
            return "Other"

        if industry_text in INDUSTRY_THRESHOLDS:
            return industry_text

        industry_lookup = {key.lower(): key for key in INDUSTRY_THRESHOLDS}
        exact_case_match = industry_lookup.get(industry_text.lower())
        if exact_case_match:
            return exact_case_match

        if RAPIDFUZZ_AVAILABLE:
            match = process.extractOne(
                industry_text,
                list(INDUSTRY_THRESHOLDS.keys()),
                scorer=fuzz.token_sort_ratio,
                score_cutoff=85,
            )
            if match:
                return match[0]

        return industry_text
    
    def clean_company_name(self, name):
        """Simple cleaning that preserves matching"""
        if pd.isna(name) or not name:
            return ""
    
        # Convert to string and lowercase
        clean_name = str(name).lower().strip()
    
        # Remove ONLY obvious filename junk
        clean_name = re.sub(r'\.(json|csv|xlsx?|txt)$', '', clean_name, flags=re.IGNORECASE)
        clean_name = re.sub(r'[_\s-]*\d+,\d+$', '', clean_name)
        clean_name = re.sub(r'_\d+,\d+\.json$', '', clean_name)
        clean_name = re.sub(r'_\d+,\d+$', '', clean_name) 
        clean_name = re.sub(r'\s+transaction\s+reports?\b.*$', '', clean_name)

        # Normalize punctuation and common legal suffixes so CSV and filenames match.
        clean_name = re.sub(r'[^\w\s&]', ' ', clean_name)
        clean_name = re.sub(r'\blimited\b', 'ltd', clean_name)
        clean_name = re.sub(r'\bincorporated\b', 'inc', clean_name)
        clean_name = re.sub(r'\bcorporation\b', 'corp', clean_name)
        clean_name = re.sub(r'\bcompany\b', 'co', clean_name)
    
        # Normalize whitespace
        clean_name = re.sub(r'\s+', ' ', clean_name).strip()
    
        return clean_name
    
    def create_name_variations(self, name):
        """Create variations of a company name for better matching"""
        if not name:
            return []
        
        base_name = self.clean_company_name(name)
        variations = [base_name]
        
        # Create variations without business suffixes
        no_suffix = re.sub(r'\b(ltd|limited|llc|inc|corp|corporation|co|company|plc|llp|private|pvt)\b', '', base_name, flags=re.IGNORECASE).strip()
        if no_suffix and no_suffix != base_name:
            variations.append(no_suffix)
        
        # Create variation with "ltd" if it doesn't have it
        if 'ltd' not in base_name.lower() and 'limited' not in base_name.lower():
            variations.append(f"{base_name} ltd")
        
        # Create variation without numbers
        no_numbers = re.sub(r'\d+', '', base_name).strip()
        if no_numbers and no_numbers != base_name:
            variations.append(no_numbers)
        
        # Handle specific known variations
        specific_variations = {
            'hans lec design': ['hans-lec design'],
            'e gener8': ['e-gener8'],
            'weird inc': ['weird.inc'],
            'south coast auto': ['south coast'],
            'neu hair 4 men': ['neu hair'],
            'xtc clothing': ['xtcclothing'],
            'sanitaire limited': ['santaire'],
            'smithstrades': ['smiths trades', 'smith trades'],
            'hursley childcare hub': ['hursley childcare'],
            'marvellous glow': ['marvellous glow ltd'],
            'valtchev group': ['valtchev'],
        }
        
        for key, var_list in specific_variations.items():
            if key in base_name.lower():
                variations.extend(var_list)
        
        # Remove duplicates and empty strings
        return list(set([v.strip() for v in variations if v.strip()]))
    
    def extract_company_name_from_json(self, json_data, filename):
        """Extract company name with debug info"""
        company_name = None
        extraction_method = "none"
    
        # Debug: Show what fields are available
        print(f"DEBUG: Processing {filename}")
        if isinstance(json_data, dict):
            print(f"DEBUG: Available root keys: {list(json_data.keys())}")
        
            # Check for account_owner
            if 'account_owner' in json_data:
                print(f"DEBUG: account_owner = '{json_data['account_owner']}'")
                if json_data['account_owner']:
                    company_name = str(json_data['account_owner']).strip()
                    extraction_method = "root.account_owner"
    
        # Fallback to filename if no account owner found
        if not company_name:
            print(f"DEBUG: No account_owner found, using filename")
            # Clean filename approach (same as before)
            clean_filename = filename
            clean_filename = re.sub(r'\.(json|csv|xlsx?|txt)$', '', clean_filename, flags=re.IGNORECASE)
            clean_filename = clean_filename.replace('_', ' ').replace('*', ' ').replace('-', ' ')
        
            patterns_to_remove = [
                r'\s*transaction\s*reports?\s*.*$',
                r'\s*app\s*\d+.*$',
                r'\s*\d+,\d+.*$',
                r'\s*\d+\s*$',
                r'\s*v\d+.*$',
            ]
        
            for pattern in patterns_to_remove:
                clean_filename = re.sub(pattern, '', clean_filename, flags=re.IGNORECASE)
        
            clean_filename = ' '.join(clean_filename.split())
        
            if clean_filename and len(clean_filename.strip()) > 2:
                company_name = clean_filename.strip().title()
                extraction_method = "filename_enhanced"
    
        print(f"DEBUG: Final extracted name: '{company_name}' via {extraction_method}")
        return company_name, extraction_method
        
    def simple_match_test(self, search_name, csv_companies, debug_info):
        """Simple matching for testing"""
        if not search_name or not csv_companies:
            print(f"DEBUG: Empty inputs - search_name: {search_name}, csv_companies count: {len(csv_companies) if csv_companies else 0}")
            return None, 0, "none", False
    
        # Clean the search name
        search_clean = search_name.lower()
        search_clean = search_clean.replace('limited', '').replace('ltd', '').replace('inc', '').replace('corp', '')
        search_clean = ' '.join(search_clean.split())  # normalize spaces
    
        print(f"DEBUG: Searching for '{search_name}' -> cleaned: '{search_clean}'")
    
        for csv_company in csv_companies:
            # Clean the CSV company name
            csv_clean = csv_company.lower()
            csv_clean = csv_clean.replace('limited', '').replace('ltd', '').replace('inc', '').replace('corp', '')
            csv_clean = ' '.join(csv_clean.split())
        
            # Check for exact match
            if search_clean == csv_clean:
                print(f"DEBUG: EXACT MATCH FOUND! '{search_name}' -> '{csv_company}'")
                return csv_company, 100, "simple_exact", True
        
            # Check if one contains the other
            if search_clean in csv_clean or csv_clean in search_clean:
                print(f"DEBUG: CONTAINS MATCH FOUND! '{search_name}' -> '{csv_company}'")
                return csv_company, 90, "simple_contains", True
    
        print(f"DEBUG: NO MATCH for '{search_name}' (cleaned: '{search_clean}')")
        return None, 0, "none", False
    
    def fuzzy_match_company(self, search_name, csv_companies, debug_info):
        """FIXED: Fuzzy match company name with detailed debugging"""
        
        print(f"DEBUG START: fuzzy_match_company called with search_name='{search_name}', csv_companies count={len(csv_companies) if csv_companies else 0}")
        
        if not search_name or not csv_companies:
            debug_info['fuzzy_match_debug'] = "No search name or CSV companies provided"
            return None, 0, "none", False
        
        clean_search = self.clean_company_name(search_name)
        debug_info['cleaned_search_name'] = clean_search
        
        # Clean CSV company names for matching
        clean_csv_companies = {}
        for company in csv_companies:
            clean_company = self.clean_company_name(company)
            if clean_company and clean_company not in clean_csv_companies:
                clean_csv_companies[clean_company] = company
        
        debug_info['csv_companies_count'] = len(csv_companies)
        debug_info['first_few_csv'] = list(csv_companies)[:3]
        debug_info['cleaned_csv_sample'] = list(clean_csv_companies.keys())[:3]
        
        best_match = None
        best_score = 0
        best_strategy = "none"
        
        if RAPIDFUZZ_AVAILABLE:
            debug_info['fuzzy_match_debug'] = "Using RapidFuzz for matching"
            print(f"DEBUG FUZZY: Trying to match '{clean_search}' against {len(clean_csv_companies)} companies")
            
            # Try exact/contains matches first to avoid fuzzy false positives.
            if clean_search in clean_csv_companies:
                original_company = clean_csv_companies[clean_search]
                best_match = (original_company, 100)
                best_score = 100
                best_strategy = "exact_clean"
            else:
                for clean_company, original_company in clean_csv_companies.items():
                    if clean_search and clean_company and (clean_search in clean_company or clean_company in clean_search):
                        best_match = (original_company, 95)
                        best_score = 95
                        best_strategy = "contains_clean"
                        break

            strategies = [
                ('token_sort_ratio', fuzz.token_sort_ratio, 60),
                ('token_set_ratio', fuzz.token_set_ratio, 55),
                ('partial_ratio', fuzz.partial_ratio, 65),
                ('ratio', fuzz.ratio, 70)
            ]
            
            for strategy_name, strategy_func, min_score in strategies:
                try:
                    # Use clean names for matching
                    match_result = process.extractOne(
                        clean_search, 
                        list(clean_csv_companies.keys()), 
                        scorer=strategy_func,
                        score_cutoff=min_score
                    )
                    
                    if match_result and match_result[1] > best_score:
                        # Get original company name
                        matched_clean = match_result[0]
                        original_company = clean_csv_companies[matched_clean]
                        best_match = (original_company, match_result[1])
                        best_score = match_result[1]
                        best_strategy = strategy_name
                        
                        debug_info[f'{strategy_name}_score'] = match_result[1]
                        debug_info[f'{strategy_name}_match'] = original_company
                        
                except Exception as e:
                    debug_info[f'{strategy_name}_error'] = str(e)
        else:
            # Fallback exact matching
            debug_info['fuzzy_match_debug'] = "Using fallback exact matching"
            
            if clean_search in clean_csv_companies:
                original_company = clean_csv_companies[clean_search]
                best_match = (original_company, 100)
                best_score = 100
                best_strategy = "exact_fallback"
            else:
                for clean_company, original_company in clean_csv_companies.items():
                    if clean_search and clean_company and (clean_search in clean_company or clean_company in clean_search):
                        best_match = (original_company, 95)
                        best_score = 95
                        best_strategy = "contains_fallback"
                        break
        
        if best_match:
            debug_info['best_match_company'] = best_match[0]
            debug_info['best_match_score'] = best_score
            debug_info['best_strategy'] = best_strategy
            return best_match[0], best_score, best_strategy, True
        else:
            debug_info['fuzzy_match_debug'] = f"No matches found for '{clean_search}'"
            return None, 0, "none", False

    def process_single_application(self, json_data, filename, default_params=None, parameter_mapping=None):
        """FIXED: Process a single application with working fuzzy matching and comprehensive debugging"""

        # Initialize debug info with comprehensive tracking
        debug_info = {
            'filename': filename,
            'processed_at': datetime.now().isoformat(),
            'debug_step': 'Starting processing',
            'parameter_mapping_available': bool(parameter_mapping),
            'parameter_mapping_count': len(parameter_mapping) if parameter_mapping else 0,
            'rapidfuzz_available': RAPIDFUZZ_AVAILABLE
        }

        try:
            # Step 1: Validate transaction data AND preserve original json_data for metadata extraction
            original_json_data = json_data  # Keep original for metadata extraction

            # HANDLE BOTH LIST AND DICT formats properly
            if isinstance(json_data, list):
                # Direct list format - USE ALL TRANSACTIONS
                transactions = json_data
                json_data_dict = {}  # No metadata available from list format
                print(f"DEBUG: Direct list format detected:  {len(transactions)} transactions")

            elif isinstance(json_data, dict):
                json_data_dict = json_data  # Keep dict for metadata extraction
                # Dictionary format - check for 'transactions' key
                if 'transactions' in json_data:
                    transactions = json_data.get('transactions', [])
                    print(f"DEBUG: Dictionary with 'transactions' key: {len(transactions)} transactions")
                else:
                    # This file IS a single transaction - wrap it in a list
                    if 'transaction_id' in json_data or 'amount' in json_data or 'date' in json_data:
                        transactions = [json_data]
                        print(f"DEBUG: Single transaction file detected: {filename}")
                    else:
                        transactions = []
                        print(f"DEBUG: Empty or invalid JSON structure in {filename}")
            else:
                raise ValueError("Unexpected JSON format - expected list or dictionary")

            if not transactions:
                raise ValueError("No transactions found in JSON")

            debug_info['transaction_count'] = len(transactions)
            debug_info['debug_step'] = f'Found {len(transactions)} transactions'
            print(f"Loaded {len(transactions)} transactions from {filename}")

            # Convert to DataFrame
            df = pd.json_normalize(transactions)
            debug_info['dataframe_shape'] = df.shape
            debug_info['dataframe_columns'] = list(df.columns)

            # Validate required columns
            required_columns = ['date', 'amount', 'name']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                debug_info['missing_columns'] = missing_columns
                raise ValueError(f"Missing required columns: {missing_columns}")

            debug_info['debug_step'] = 'Required columns validated'

            # Clean data
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
            original_count = len(df)
            df = df.dropna(subset=['date', 'amount'])

            debug_info['original_transaction_count'] = original_count
            debug_info['cleaned_transaction_count'] = len(df)
            debug_info['dropped_transactions'] = original_count - len(df)

            if df.empty:
                raise ValueError("No valid transactions after data cleaning")

            debug_info['debug_step'] = f'Data cleaned:  {len(df)} valid transactions'

            # Step 2: FIXED company name extraction
            params = default_params.copy() if default_params else {}

            # Extract company name with comprehensive tracking
            # Use json_data_dict for metadata extraction (safe for both list and dict)
            company_name, extraction_method = self.extract_company_name_from_json(
                json_data_dict if isinstance(original_json_data, dict) else {}, filename)
            debug_info['extracted_company_name'] = company_name
            debug_info['extraction_method'] = extraction_method
            debug_info['debug_step'] = f'Company name extracted: "{company_name}" via {extraction_method}'

            params['company_name'] = company_name
            params['original_filename'] = filename

            # Initialize fuzzy match results with comprehensive tracking
            debug_info.update({
                'fuzzy_match_company': 'No match attempted',
                'fuzzy_match_score': 0,
                'fuzzy_match_strategy': 'none',
                'fuzzy_match_success': False,
                'fuzzy_match_debug': 'No matching attempted yet',
                'cleaned_search_name': '',
                'csv_companies_count': 0,
                'parameters_applied_from_csv': 0
            })

            # Step 3:  FIXED fuzzy matching with comprehensive CSV data handling
            if parameter_mapping and company_name:
                debug_info['debug_step'] = 'Starting fuzzy matching process'
                debug_info['fuzzy_match_debug'] = 'Fuzzy matching initiated'

                csv_companies = list(parameter_mapping.keys())
                debug_info['csv_companies_available'] = csv_companies

                matched_company, score, strategy, success = self.fuzzy_match_company(
                    company_name, csv_companies, debug_info
                )

                if success and matched_company:
                    debug_info['fuzzy_match_company'] = matched_company
                    debug_info['fuzzy_match_score'] = score
                    debug_info['fuzzy_match_strategy'] = strategy
                    debug_info['fuzzy_match_success'] = True
                    debug_info[
                        'debug_step'] = f'FUZZY MATCH SUCCESS: "{company_name}" -> "{matched_company}" ({score}% confidence via {strategy})'

                    # Apply CSV parameters with tracking
                    csv_params = parameter_mapping[matched_company]
                    debug_info['csv_parameters_available'] = csv_params

                    params_applied = 0
                    applied_params = {}

                    # Apply each parameter with validation and tracking
                    for param_key, param_value in csv_params.items():
                        if param_key == 'company_name' or param_key == 'filename':
                            continue  # Skip these meta fields

                        if pd.notna(param_value) and param_value != '' and param_value is not None:
                            # Convert boolean strings to actual booleans
                            if isinstance(param_value, str):
                                if param_value.lower() in ['true', '1', 'yes', 'y']:
                                    param_value = True
                                elif param_value.lower() in ['false', '0', 'no', 'n']:
                                    param_value = False

                            # Apply parameter
                            params[param_key] = param_value
                            applied_params[param_key] = param_value
                            params_applied += 1

                    debug_info['parameters_applied_from_csv'] = params_applied
                    debug_info['applied_csv_parameters'] = applied_params
                    debug_info[
                        'debug_step'] += f', Applied {params_applied} CSV parameters:  {list(applied_params.keys())}'

                else:
                    debug_info['fuzzy_match_success'] = False
                    debug_info[
                        'debug_step'] = f'FUZZY MATCH FAILED:  No match found for "{company_name}" in {len(csv_companies)} CSV companies'
                    debug_info['fuzzy_match_debug'] = f'Failed to find match for "{company_name}" against CSV companies'

            elif not parameter_mapping:
                debug_info['debug_step'] = 'No CSV parameter mapping provided'
                debug_info['fuzzy_match_debug'] = 'No CSV parameter mapping available'
            elif not company_name:
                debug_info['debug_step'] = 'No company name extracted for matching'
                debug_info['fuzzy_match_debug'] = 'No company name available for matching'

            # Step 4: Try to extract application-specific data from JSON (ONLY if dict format)
            debug_info['debug_step'] = 'Checking for JSON metadata'

            if isinstance(original_json_data, dict):
                # Check for application metadata in JSON
                metadata_found = {}
                app_data = json_data_dict.get('application_data', {})
                if app_data:
                    debug_info['json_application_data_found'] = True
                    for key in ['industry', 'directors_score', 'requested_loan', 'company_age_months']:
                        if key in app_data and app_data[key] is not None:
                            params[key] = app_data[key]
                            metadata_found[key] = app_data[key]

                # Check for metadata in root JSON
                root_metadata = {}
                metadata_fields = {
                    'industry': json_data_dict.get('industry'),
                    'directors_score': json_data_dict.get('directors_score'),
                    'requested_loan': json_data_dict.get('requested_loan'),
                    'company_age_months': json_data_dict.get('company_age_months')
                }

                for key, value in metadata_fields.items():
                    if value is not None and pd.notna(value):
                        params[key] = value
                        root_metadata[key] = value

                debug_info['json_metadata_found'] = {**metadata_found, **root_metadata}
                debug_info[
                    'debug_step'] = f'JSON metadata extraction complete, found: {list(debug_info["json_metadata_found"].keys())}'
            else:
                debug_info['json_metadata_found'] = {}
                debug_info['debug_step'] = 'No JSON metadata available (list format)'

            # Step 5: Validate and set required parameters
            debug_info['debug_step'] = 'Validating and setting required parameters'

            # List of absolutely required parameters for processing
            critical_params = ['directors_score', 'requested_loan', 'industry', 'company_age_months']
            missing_critical = []

            for param in critical_params:
                if param not in params or params[param] is None or pd.isna(params[param]):
                    missing_critical.append(param)

            debug_info['missing_critical_parameters'] = missing_critical
            debug_info['using_defaults_for'] = missing_critical
            debug_info['using_defaults'] = len(missing_critical) > 0

            # Set company name if still missing
            if 'company_name' not in params or not params['company_name']:
                params['company_name'] = filename.replace('.json', '').replace('_', ' ').title()
                debug_info['company_name_set_from_filename'] = True

            # Step 6: Data type validation and conversion
            debug_info['debug_step'] = 'Converting parameter data types'

            try:
                # Convert numeric parameters
                if 'directors_score' in params:
                    params['directors_score'] = int(float(params['directors_score']))
                if 'requested_loan' in params:
                    params['requested_loan'] = float(params['requested_loan'])
                if 'company_age_months' in params:
                    params['company_age_months'] = int(float(params['company_age_months']))
                if 'industry' in params:
                    params['industry'] = self.normalize_industry(params['industry'])

                # Convert boolean risk factors
                risk_factors = ['business_ccj', 'director_ccj',
                                'poor_or_no_online_presence', 'uses_generic_email']

                for factor in risk_factors:
                    if factor in params:
                        value = params[factor]
                        if isinstance(value, str):
                            params[factor] = value.lower() in ['true', '1', 'yes', 'y']
                        elif isinstance(value, (int, float)):
                            params[factor] = bool(value)
                        else:
                            params[factor] = bool(value)

                debug_info['parameter_conversion_successful'] = True

            except (ValueError, TypeError) as e:
                debug_info['parameter_conversion_error'] = str(e)
                raise ValueError(f"Invalid parameter data types: {e}")

            # Step 7: Parameter validation
            debug_info['debug_step'] = 'Validating parameter ranges'

            # Validate parameter ranges
            if not (0 <= params['directors_score'] <= 100):
                raise ValueError(f"Directors score must be 0-100, got {params['directors_score']}")

            if params['requested_loan'] <= 0:
                raise ValueError(f"Requested loan must be positive, got {params['requested_loan']}")

            if params['company_age_months'] < 0:
                raise ValueError(f"Company age must be non-negative, got {params['company_age_months']}")

            if params['industry'] not in INDUSTRY_THRESHOLDS:
                raise ValueError(f"Unknown industry: {params['industry']}")

            debug_info['parameter_validation_successful'] = True
            debug_info['final_parameters'] = {k: v for k, v in params.items() if k not in ['company_name']}

            # Step 8: Calculate financial metrics
            debug_info['debug_step'] = 'Calculating financial metrics'

            metrics = calculate_financial_metrics(df, params['company_age_months'])

            if not metrics:
                raise ValueError("Could not calculate financial metrics from transaction data")

            if derive_open_banking_insights is not None:
                metrics.update(derive_open_banking_insights(categorize_transactions(df), params.get("requested_loan")))

            debug_info['metrics_calculation_successful'] = True
            debug_info['key_metrics'] = {
                'Total Revenue': metrics.get('Total Revenue', 0),
                'Net Income': metrics.get('Net Income', 0),
                'Debt Service Coverage Ratio': metrics.get('Debt Service Coverage Ratio', 0),
                'Operating Margin': metrics.get('Operating Margin', 0)
            }

            # Step 9: Calculate all scores
            debug_info['debug_step'] = 'Calculating scoring algorithms'

            scores = calculate_all_scores_tightened(metrics, params)

            debug_info['scoring_calculation_successful'] = True
            debug_info['calculated_scores'] = {
                'subprime_score': scores.get('subprime_score', 0),
                'mca_rule_score': scores.get('mca_rule_score', 0),
                'ml_score': scores.get('ml_score', None),
                'subprime_tier': scores.get('subprime_tier', 'Unknown')
            }

            # Step 10: Combine all results
            debug_info['debug_step'] = 'Combining final results'

            # Create comprehensive result with all debug information
            result = {
                # Debug and tracking information
                **debug_info,

                # Application parameters
                **params,

                # Financial metrics
                **metrics,

                # Scores
                **scores,

                # Additional metadata
                'transaction_count': len(df),
                'date_range_start': df['date'].min().isoformat(),
                'date_range_end': df['date'].max().isoformat(),
                'total_amount': abs(df['amount']).sum(),
                'processing_successful': True
            }

            debug_info['debug_step'] = 'Processing completed successfully'
            result['debug_step'] = 'Processing completed successfully'

            self.processed_count += 1
            self.debug_log.append({
                'filename': filename,
                'status': 'SUCCESS',
                'company_name': company_name,
                'fuzzy_match_success': debug_info.get('fuzzy_match_success', False),
                'fuzzy_match_score': debug_info.get('fuzzy_match_score', 0),
                'parameters_from_csv': debug_info.get('parameters_applied_from_csv', 0),
                'using_defaults': debug_info.get('using_defaults', False)
            })

            return result

        except Exception as e:
            self.error_count += 1

            # Create detailed error information
            error_info = {
                'filename': filename,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'debug_step': debug_info.get('debug_step', 'Unknown step'),
                'extraction_method': debug_info.get('extraction_method', 'None'),
                'extracted_company_name': debug_info.get('extracted_company_name', 'None'),
                'fuzzy_match_attempted': debug_info.get('parameter_mapping_available', False),
                'fuzzy_match_success': debug_info.get('fuzzy_match_success', False),
                'transaction_count': debug_info.get('transaction_count', 0),
                'processing_stage': debug_info.get('debug_step', 'Unknown')
            }

            self.error_log.append(error_info)
            self.debug_log.append({
                'filename': filename,
                'status': 'ERROR',
                'error': str(e),
                'debug_step': debug_info.get('debug_step', 'Unknown')
            })

            return None
    
    def process_batch(self, files_data, default_params, parameter_mapping=None, progress_bar=None, outcome_mapping=None):
        """Process multiple applications with comprehensive tracking"""
        self.results = []
        self.processed_count = 0
        self.error_count = 0
        self.error_log = []
        self.debug_log = []
        
        total_files = len(files_data)
        
        print(f"Starting batch processing of {total_files} files...")
        print(f"Parameter mapping available: {bool(parameter_mapping)}")
        if parameter_mapping:
            print(f"CSV companies available: {len(parameter_mapping)}")
            print(f"First few CSV companies: {list(parameter_mapping.keys())[:3]}")
        
        for i, (filename, json_data) in enumerate(files_data):
            if progress_bar:
                progress_bar.progress((i + 1) / total_files, text=f"Processing {filename}...")
            
            print(f"\nProcessing {i+1}/{total_files}: {filename}")

            params_for_file = default_params.copy() if default_params else {}
            if outcome_mapping and filename in outcome_mapping:
                params_for_file.update(outcome_mapping[filename])

            result = self.process_single_application(json_data, filename, params_for_file, parameter_mapping)
            
            if result:
                self.results.append(result)
                print(f"SUCCESS: {filename}")
                if result.get('fuzzy_match_success'):
                    print(f"   Matched to: {result.get('fuzzy_match_company')} ({result.get('fuzzy_match_score')}%)")
                if result.get('using_defaults'):
                    print(f"   Using defaults for: {result.get('using_defaults_for', [])}")
            else:
                print(f"FAILED: {filename}")
        
        print(f"\nBatch processing complete!")
        print(f"Successful: {self.processed_count}")
        print(f"Failed: {self.error_count}")
        
        return pd.DataFrame(self.results) if self.results else pd.DataFrame()

def load_json_files(uploaded_files):
    """Load JSON files from uploaded files"""
    files_data = []
    
    for uploaded_file in uploaded_files:
        try:
            uploaded_name = uploaded_file.name
            uploaded_name_lower = uploaded_name.lower()

            if uploaded_name_lower.endswith('.zip'):
                # Handle ZIP files
                with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                    for file_info in zip_ref.filelist:
                        if file_info.filename.lower().endswith('.json') and not file_info.is_dir():
                            with zip_ref.open(file_info.filename) as json_file:
                                json_data = json.load(json_file)
                                files_data.append((file_info.filename, json_data))
            
            elif uploaded_name_lower.endswith('.json'):
                # Handle individual JSON files
                json_data = json.load(uploaded_file)
                files_data.append((uploaded_name, json_data))
                
        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {e}")
    
    return files_data


def uploaded_json_match_sets(uploaded_files) -> tuple[set[str], set[str], dict[str, set[str]], dict[str, set[str]]]:
    """Return normalized names and content signatures from direct JSON/ZIP uploads."""
    names = set()
    signatures = set()
    name_to_signatures: dict[str, set[str]] = {}
    signature_to_names: dict[str, set[str]] = {}
    for uploaded_file in uploaded_files or []:
        uploaded_name = uploaded_file.name
        if uploaded_name.lower().endswith(".zip"):
            try:
                uploaded_file.seek(0)
                with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
                    for file_info in zip_ref.filelist:
                        if file_info.filename.lower().endswith(".json") and not file_info.is_dir():
                            key = normalize_upload_name(file_info.filename)
                            names.add(key)
                            try:
                                with zip_ref.open(file_info.filename) as json_file:
                                    signature = stable_json_signature(json.load(json_file))
                                    signatures.add(signature)
                                    name_to_signatures.setdefault(key, set()).add(signature)
                                    signature_to_names.setdefault(signature, set()).add(Path(file_info.filename).name)
                            except Exception:
                                pass
            except Exception:
                names.add(normalize_upload_name(uploaded_name))
        elif uploaded_name.lower().endswith(".json"):
            key = normalize_upload_name(uploaded_name)
            names.add(key)
            try:
                uploaded_file.seek(0)
                signature = stable_json_signature(json.load(uploaded_file))
                signatures.add(signature)
                name_to_signatures.setdefault(key, set()).add(signature)
                signature_to_names.setdefault(signature, set()).add(uploaded_name)
            except Exception:
                pass
    return names, signatures, name_to_signatures, signature_to_names


def assign_outcomes(files_data, paid_files, not_paid_files) -> tuple[dict, pd.DataFrame]:
    """Build per-file outcome labels from paid and not-paid JSON uploads."""
    paid_names, paid_signatures, paid_name_to_signatures, paid_signature_to_names = uploaded_json_match_sets(paid_files)
    not_paid_names, not_paid_signatures, not_paid_name_to_signatures, not_paid_signature_to_names = uploaded_json_match_sets(not_paid_files)
    outcomes = {}
    audit_rows = []
    all_names = set()
    all_signatures = set()
    matched_names = set()
    matched_signatures = set()

    for filename, json_data in files_data:
        key = normalize_upload_name(filename)
        signature = stable_json_signature(json_data)
        all_names.add(key)
        all_signatures.add(signature)
        paid_name_match = key in paid_names
        paid_content_match = signature in paid_signatures
        not_paid_name_match = key in not_paid_names
        not_paid_content_match = signature in not_paid_signatures
        in_paid = paid_name_match or paid_content_match
        in_not_paid = not_paid_name_match or not_paid_content_match
        if paid_name_match:
            matched_names.add(key)
            matched_signatures.update(paid_name_to_signatures.get(key, set()))
        if not_paid_name_match:
            matched_names.add(key)
            matched_signatures.update(not_paid_name_to_signatures.get(key, set()))
        if paid_content_match:
            matched_signatures.add(signature)
        if not_paid_content_match:
            matched_signatures.add(signature)
        if in_paid and in_not_paid:
            outcome = "conflict"
        elif in_paid:
            outcome = "paid"
        elif in_not_paid:
            outcome = "not_paid"
        else:
            outcome = "unlabelled"

        outcomes[filename] = {
            "outcome_label": outcome,
            "paid_flag": outcome == "paid",
            "not_paid_flag": outcome == "not_paid",
            "defaulted_flag": outcome == "not_paid",
            "outcome_conflict": outcome == "conflict",
        }
        audit_rows.append(
            {
                "json_file": filename,
                "outcome_label": outcome,
                "in_paid_upload": in_paid,
                "in_not_paid_upload": in_not_paid,
                "match_method": (
                    "name_and_content" if (paid_name_match or not_paid_name_match) and (paid_content_match or not_paid_content_match)
                    else "content" if paid_content_match or not_paid_content_match
                    else "name" if paid_name_match or not_paid_name_match
                    else "none"
                ),
            }
        )

    for signature in sorted((paid_signatures | not_paid_signatures) - matched_signatures):
        upload_names = sorted(paid_signature_to_names.get(signature, set()) | not_paid_signature_to_names.get(signature, set()))
        audit_rows.append(
            {
                "json_file": ", ".join(upload_names) if upload_names else signature[:12],
                "outcome_label": "content_not_in_all_json_upload",
                "in_paid_upload": signature in paid_signatures,
                "in_not_paid_upload": signature in not_paid_signatures,
                "match_method": "content_missing_from_all_upload",
            }
        )

    return outcomes, pd.DataFrame(audit_rows)


def _extract_text_from_pdf_bytes(pdf_bytes: bytes) -> tuple[str, str, str]:
    """Extract text from a company credit report PDF using available backends."""
    if not pdf_bytes or not pdf_bytes[:4] == b"%PDF":
        return "", "none", "Not a valid PDF header"

    errors = []
    try:
        import pdfplumber

        parts = []
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                parts.append(page.extract_text() or "")
        text = "\n".join(parts).strip()
        if text:
            return text, "pdfplumber", ""
        errors.append("pdfplumber extracted empty text")
    except Exception as e:
        errors.append(f"pdfplumber: {repr(e)}")

    try:
        import fitz

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n".join((page.get_text("text") or "") for page in doc).strip()
        if text:
            return text, "pymupdf", ""
        errors.append("pymupdf extracted empty text")
    except Exception as e:
        errors.append(f"pymupdf: {repr(e)}")

    try:
        import PyPDF2

        reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
        text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
        if text:
            return text, "pypdf2", ""
        errors.append("pypdf2 extracted empty text")
    except Exception as e:
        errors.append(f"pypdf2: {repr(e)}")

    return "", "none", " | ".join(errors)


def _norm_pdf_text(t: str) -> str:
    if not t:
        return ""
    return (
        t.replace("\ufb01", "fi")
        .replace("\ufb02", "fl")
        .replace("�", "")
        .replace("\x00", "fi")
        .replace("Ł", "£")
    )


def _re_first(pattern: str, text: str, flags=re.IGNORECASE):
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else None


def _money_to_int(s: str | None) -> int | None:
    if not s:
        return None
    digits = re.sub(r"[^\d]", "", str(s))
    return int(digits) if digits else None


def _parse_business_bureau_signals(full_text: str) -> dict[str, object]:
    """Extract modelling-safe Capital report signals for batch scorecard outputs."""
    t = _norm_pdf_text(full_text or "")
    tl = t.lower()

    credit_score = _re_first(r"\bcredit score\b\s*[\:\-]?\s*(\d{1,3})\b", t)
    credit_limit = _re_first(r"\bcredit limit\b[\s\S]{0,80}?(£\s*\d[\d,]*)", t)
    max_credit = _re_first(r"\bmax\.?\s*recommended\s*credit\b[\s\S]{0,80}?(£\s*\d[\d,]*)", t)
    searches_12m = _re_first(r"\bin\s+the\s+last\s+12\s+months\s*\n?\s*(\d+)\b", t)
    enquiries_3m = _re_first(r"\b(\d+)\s+enquiries\s+in\s+last\s+3\s+months\b", t)
    negative_impact = _re_first(r"\bnegative impact:\s*(\d+)\b", t)
    neutral_impact = _re_first(r"\bneutral impact:\s*(\d+)\b", t)
    total_factors = _re_first(r"\btotal factors\s*\n?\s*(\d+)\b", t)

    return {
        "business_credit_score": int(credit_score) if credit_score else None,
        "business_credit_score_suppressed": "risk score suppressed" in tl or bool(re.search(r"\bcredit score\s*-\s*risk score suppressed\b", tl)),
        "business_credit_limit": _money_to_int(credit_limit),
        "business_max_recommended_credit": _money_to_int(max_credit),
        "business_company_searches_12m": int(searches_12m) if searches_12m else None,
        "business_enquiries_3m": int(enquiries_3m) if enquiries_3m else None,
        "business_negative_impact_count": int(negative_impact) if negative_impact else 0,
        "business_neutral_impact_count": int(neutral_impact) if neutral_impact else None,
        "business_total_factor_count": int(total_factors) if total_factors else None,
        "business_bureau_needs_attention": "needs attention" in tl,
        "business_no_registered_charges": "no registered mortgages or charges" in tl or "no mortgages or charges" in tl,
    }


def _explicit_ccj_present(pdf_text: str) -> bool:
    """Detect explicit business CCJ presence using the same policy as the main app."""
    t = (pdf_text or "").lower()
    negative_patterns = [
        r"\bno\s+county\s+court\s+judg(e)?ment(s)?\b",
        r"\bno\s+ccj\b",
        r"\bccj\s*:\s*(none|no|0)\b",
        r"\bnone\s+recorded\b.*\bccj\b",
    ]
    for pattern in negative_patterns:
        if re.search(pattern, t, re.IGNORECASE):
            return False

    positive_patterns = [
        r"county\s+court\s+judg(e)?ment\s+registered",
        r"county\s+court\s+judg(e)?ments\s+registered",
        r"county\s+court\s+judg(e)?ment\s+has\s+been\s+registered",
        r"\bccj\s+registered\b",
        r"\bat\s+least\s+one\s+county\s+court\s+judg(e)?ment\b",
        r"\blegal\s+notices\b[\s\S]{0,300}county\s+court\s+judg(e)?ment",
    ]
    for pattern in positive_patterns:
        if re.search(pattern, t, re.IGNORECASE):
            return True

    if re.search(r"county\s+court\s+judg(e)?ments?\b", t):
        if re.search(r"£\s*\d", t) and re.search(r"\b\d{1,2}\s+[a-z]{3}\s+\d{4}\b", t):
            return True
    return False


def build_pdf_risk_mapping(pdf_files, metadata_df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Match uploaded bureau PDFs to companies and derive business CCJ flags."""
    if not pdf_files:
        return {}, pd.DataFrame()

    processor = BatchProcessor()
    companies = metadata_df["company_name"].dropna().astype(str).tolist() if "company_name" in metadata_df else []
    risk_mapping = {}
    audit_rows = []

    for pdf_file in pdf_files:
        try:
            pdf_file.seek(0)
            text, backend, error = _extract_text_from_pdf_bytes(pdf_file.getvalue())
            business_ccj = _explicit_ccj_present(text) if text else False
            bureau_signals = _parse_business_bureau_signals(text) if text else {}
            search_name = Path(pdf_file.name).stem
            debug_info = {}
            matched_company, score, strategy, success = processor.fuzzy_match_company(search_name, companies, debug_info)

            if success and matched_company:
                existing = risk_mapping.get(matched_company, {})
                existing_score = float(existing.get("bureau_pdf_match_score", -1) or -1)
                if existing and score < existing_score:
                    audit_rows.append(
                        {
                            "pdf_file": pdf_file.name,
                            "matched_company": matched_company,
                            "match_score": score,
                            "match_strategy": f"{strategy}_duplicate_lower_confidence",
                            "business_ccj": business_ccj,
                            **bureau_signals,
                            "text_backend": backend,
                            "error": "Matched company already has a higher-confidence PDF; not applied",
                        }
                    )
                    continue
                risk_mapping[matched_company] = {
                    **existing,
                    **bureau_signals,
                    "business_ccj": bool(existing.get("business_ccj", False) or business_ccj),
                    "bureau_pdf_file": pdf_file.name,
                    "bureau_pdf_match_score": score,
                    "bureau_pdf_match_strategy": strategy,
                    "bureau_pdf_backend": backend,
                }

            audit_rows.append(
                {
                    "pdf_file": pdf_file.name,
                    "matched_company": matched_company if success else None,
                    "match_score": score,
                    "match_strategy": strategy,
                    "business_ccj": business_ccj,
                    **bureau_signals,
                    "text_backend": backend,
                    "error": error,
                }
            )
        except Exception as e:
            audit_rows.append(
                {
                    "pdf_file": getattr(pdf_file, "name", "unknown"),
                    "matched_company": None,
                    "match_score": 0,
                    "match_strategy": "error",
                    "business_ccj": False,
                    "text_backend": "none",
                    "error": str(e),
                }
            )

    return risk_mapping, pd.DataFrame(audit_rows)


def add_pdf_match_review_flags(pdf_audit_df: pd.DataFrame) -> pd.DataFrame:
    """Add user-facing confidence flags to the PDF matching audit."""
    if pdf_audit_df is None or pdf_audit_df.empty:
        return pd.DataFrame()

    review_df = pdf_audit_df.copy()
    review_df["match_score"] = pd.to_numeric(review_df.get("match_score", 0), errors="coerce").fillna(0)
    review_df["pdf_match_status"] = "Strong"
    review_df.loc[review_df["matched_company"].isna() | (review_df["matched_company"].astype(str).str.strip() == ""), "pdf_match_status"] = "No match"
    review_df.loc[(review_df["match_score"] > 0) & (review_df["match_score"] < 75), "pdf_match_status"] = "Review"
    review_df.loc[(review_df["match_score"] >= 75) & (review_df["match_score"] < 88), "pdf_match_status"] = "Check"
    review_df.loc[review_df["text_backend"].fillna("none").eq("none"), "pdf_match_status"] = "Read failed"

    signal_cols = [
        "business_ccj",
        "business_credit_score_suppressed",
        "business_credit_limit",
        "business_max_recommended_credit",
        "business_negative_impact_count",
        "business_enquiries_3m",
        "business_bureau_needs_attention",
    ]
    available = [col for col in signal_cols if col in review_df.columns]
    review_df["signals_found"] = 0
    for col in available:
        value = review_df[col]
        if col in ["business_credit_limit", "business_max_recommended_credit"]:
            review_df["signals_found"] += value.notna().astype(int)
        else:
            review_df["signals_found"] += value.fillna(False).infer_objects(copy=False).astype(bool).astype(int)

    return review_df


def render_pdf_match_review(pdf_audit_df: pd.DataFrame) -> None:
    """Render a concise review table for company credit report PDF matching."""
    review_df = add_pdf_match_review_flags(pdf_audit_df)
    if review_df.empty:
        return

    section_title(
        "Business PDF Match Review",
        "Check that each uploaded business credit report matched the right company before trusting bureau calibration.",
    )

    status_counts = review_df["pdf_match_status"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("PDFs Uploaded", len(review_df))
    with c2:
        st.metric("Strong Matches", int(status_counts.get("Strong", 0)))
    with c3:
        review_count = int(status_counts.get("Check", 0) + status_counts.get("Review", 0))
        st.metric("Needs Check", review_count)
    with c4:
        st.metric("No/Failed Match", int(status_counts.get("No match", 0) + status_counts.get("Read failed", 0)))

    weak = review_df[review_df["pdf_match_status"].isin(["Check", "Review", "No match", "Read failed"])]
    if not weak.empty:
        st.warning("Some PDF matches need review. The batch can still run, but treat those bureau signals cautiously.")

    display_cols = [
        "pdf_match_status",
        "pdf_file",
        "matched_company",
        "match_score",
        "match_strategy",
        "business_ccj",
        "business_credit_score_suppressed",
        "business_credit_limit",
        "business_max_recommended_credit",
        "business_negative_impact_count",
        "business_enquiries_3m",
        "signals_found",
        "text_backend",
        "error",
    ]
    display_cols = [col for col in display_cols if col in review_df.columns]
    st.dataframe(
        review_df[display_cols].sort_values(["pdf_match_status", "match_score"], ascending=[True, False]),
        use_container_width=True,
        hide_index=True,
    )


def apply_pdf_risk_to_mapping(parameter_mapping: dict, pdf_risk_mapping: dict) -> dict:
    """Overlay PDF-derived risk factors onto the company parameter mapping."""
    merged = {key: value.copy() for key, value in parameter_mapping.items()}
    for company, risk_values in pdf_risk_mapping.items():
        if company in merged:
            merged[company].update(risk_values)
    return merged


def build_scorecard_features(results_df: pd.DataFrame) -> pd.DataFrame:
    """Create a modelling-friendly feature export from full processing results."""
    preferred_columns = [
        "application_id",
        "company_name",
        "original_filename",
        "outcome_label",
        "paid_flag",
        "not_paid_flag",
        "defaulted_flag",
        "industry",
        "requested_loan",
        "company_age_months",
        "directors_score",
        "director_defaults_12m",
        "director_defaults_36m",
        "director_ccj_count",
        "director_ccj_value",
        "business_ccj",
        "business_credit_score",
        "business_credit_score_suppressed",
        "business_credit_limit",
        "business_max_recommended_credit",
        "business_negative_impact_count",
        "business_enquiries_3m",
        "business_company_searches_12m",
        "business_bureau_needs_attention",
        "Total Revenue",
        "Monthly Average Revenue",
        "Total Expenses",
        "Net Income",
        "Total Debt",
        "Debt-to-Income Ratio",
        "Operating Margin",
        "Debt Service Coverage Ratio",
        "Gross Burn Rate",
        "Cash Flow Volatility",
        "Revenue Growth Rate",
        "Average Month-End Balance",
        "Average Negative Balance Days per Month",
        "Number of Bounced Payments",
        "Open Banking Insights Used In Score",
        "OB Transaction Count",
        "OB History Months",
        "OB True Revenue",
        "OB Non-Revenue Inflow Ratio",
        "OB Revenue Active Day Rate",
        "OB Top Revenue Source Percentage",
        "OB Card Processor Revenue Share",
        "OB Weakest Month Revenue",
        "OB Debt Repayment Burden",
        "OB Recent Loan Credits 30D",
        "OB Low Balance Days <1000",
        "OB Recent Failed Payments 30D",
        "mca_rule_score",
        "mca_rule_decision",
        "subprime_score",
        "subprime_tier",
        "final_decision",
    ]
    return results_df[[col for col in preferred_columns if col in results_df.columns]].copy()


def _labelled_outcome_frame(results_df: pd.DataFrame) -> pd.DataFrame:
    """Return only paid/not-paid cases with a numeric bad outcome flag."""
    if "outcome_label" not in results_df.columns:
        return pd.DataFrame()
    labelled = results_df[results_df["outcome_label"].isin(["paid", "not_paid"])].copy()
    if labelled.empty:
        return labelled
    labelled["bad_outcome"] = (labelled["outcome_label"] == "not_paid").astype(int)
    return labelled


def calibration_confidence(labelled_n: int, bad_n: int, separation: float) -> str:
    """Keep confidence conservative because the outcome sample is limited."""
    if labelled_n < 30 or bad_n < 8:
        return "Weak"
    if labelled_n >= 100 and bad_n >= 20 and separation >= 0.25:
        return "Strong"
    if labelled_n >= 50 and bad_n >= 12 and separation >= 0.15:
        return "Moderate"
    return "Weak"


def build_score_band_report(results_df: pd.DataFrame) -> pd.DataFrame:
    """Show paid/not-paid performance by broad score bands."""
    labelled = _labelled_outcome_frame(results_df)
    if labelled.empty:
        return pd.DataFrame()

    rows = []
    band_edges = [0, 30, 40, 50, 60, 70, 80, 101]
    band_labels = ["0-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
    for score_col in ["subprime_score", "mca_rule_score"]:
        if score_col not in labelled.columns:
            continue
        working = labelled[[score_col, "bad_outcome", "outcome_label"]].copy()
        working[score_col] = pd.to_numeric(working[score_col], errors="coerce")
        working = working.dropna(subset=[score_col])
        if working.empty:
            continue
        working["score_band"] = pd.cut(
            working[score_col].clip(lower=0, upper=100),
            bins=band_edges,
            labels=band_labels,
            right=False,
        )
        grouped = working.groupby("score_band", observed=False)
        for band, group in grouped:
            total = len(group)
            if total == 0:
                continue
            not_paid = int(group["bad_outcome"].sum())
            paid = total - not_paid
            rows.append(
                {
                    "score": score_col,
                    "score_band": str(band),
                    "applications": total,
                    "paid": paid,
                    "not_paid": not_paid,
                    "not_paid_rate": not_paid / total,
                    "avg_score": group[score_col].mean(),
                }
            )
    return pd.DataFrame(rows)


def _numeric_calibration_features(results_df: pd.DataFrame) -> list[str]:
    candidates = [
        "subprime_score",
        "mca_rule_score",
        "directors_score",
        "requested_loan",
        "company_age_months",
        "director_defaults_12m",
        "director_defaults_36m",
        "director_ccj_count",
        "director_ccj_value",
        "business_credit_score",
        "business_credit_limit",
        "business_max_recommended_credit",
        "business_negative_impact_count",
        "business_enquiries_3m",
        "business_company_searches_12m",
        "Total Revenue",
        "Monthly Average Revenue",
        "Net Income",
        "Total Debt",
        "Debt-to-Income Ratio",
        "Operating Margin",
        "Debt Service Coverage Ratio",
        "Gross Burn Rate",
        "Cash Flow Volatility",
        "Revenue Growth Rate",
        "Average Month-End Balance",
        "Average Negative Balance Days per Month",
        "Number of Bounced Payments",
        "OB Non-Revenue Inflow Ratio",
        "OB Revenue Active Day Rate",
        "OB Top Revenue Source Percentage",
        "OB Card Processor Revenue Share",
        "OB Weakest Month Revenue",
        "OB Debt Repayment Burden",
        "OB Recent Loan Credits 30D",
        "OB Low Balance Days <1000",
        "OB Recent Failed Payments 30D",
        "OB Requested Loan To Monthly Revenue",
        "OB Requested Loan To Weakest Month Revenue",
    ]
    return [col for col in candidates if col in results_df.columns]


def _feature_direction(frame: pd.DataFrame, feature: str) -> tuple[str, float, float]:
    paid_median = frame.loc[frame["bad_outcome"] == 0, feature].median()
    bad_median = frame.loc[frame["bad_outcome"] == 1, feature].median()
    if pd.isna(paid_median) or pd.isna(bad_median):
        return "unknown", paid_median, bad_median
    if bad_median <= paid_median:
        return "low_is_risk", paid_median, bad_median
    return "high_is_risk", paid_median, bad_median


def classify_recommendation_quality(row: dict) -> tuple[str, str]:
    """Classify threshold suggestions so weak/noisy signals do not look equally actionable."""
    feature = str(row.get("feature", ""))
    direction = row.get("direction")
    flagged_cases = int(row.get("flagged_cases", 0) or 0)
    flagged_paid = int(row.get("flagged_paid", 0) or 0)
    bad_capture = float(row.get("bad_capture_rate", 0) or 0)
    paid_capture = float(row.get("paid_capture_rate", 0) or 0)
    lift = float(row.get("lift_vs_base", 0) or 0)
    paid_median = row.get("paid_median")
    not_paid_median = row.get("not_paid_median")

    expected_low_risk = {
        "subprime_score",
        "mca_rule_score",
        "directors_score",
        "company_age_months",
        "business_credit_score",
        "business_credit_limit",
        "business_max_recommended_credit",
        "Total Revenue",
        "Monthly Average Revenue",
        "Net Income",
        "Operating Margin",
        "Debt Service Coverage Ratio",
        "Average Month-End Balance",
        "Revenue Growth Rate",
        "OB Revenue Active Day Rate",
        "OB Card Processor Revenue Share",
        "OB Weakest Month Revenue",
    }
    expected_high_risk = {
        "requested_loan",
        "director_defaults_12m",
        "director_defaults_36m",
        "director_ccj_count",
        "director_ccj_value",
        "business_negative_impact_count",
        "business_enquiries_3m",
        "business_company_searches_12m",
        "Total Debt",
        "Debt-to-Income Ratio",
        "Gross Burn Rate",
        "Cash Flow Volatility",
        "Average Negative Balance Days per Month",
        "Number of Bounced Payments",
        "OB Non-Revenue Inflow Ratio",
        "OB Top Revenue Source Percentage",
        "OB Debt Repayment Burden",
        "OB Recent Loan Credits 30D",
        "OB Low Balance Days <1000",
        "OB Recent Failed Payments 30D",
        "OB Requested Loan To Monthly Revenue",
        "OB Requested Loan To Weakest Month Revenue",
    }

    if feature in expected_low_risk and direction != "low_is_risk":
        return "Ignore for now", "Counterintuitive direction for this metric"
    if feature in expected_high_risk and direction != "high_is_risk":
        return "Ignore for now", "Counterintuitive direction for this metric"
    if pd.notna(paid_median) and pd.notna(not_paid_median) and float(paid_median) == float(not_paid_median):
        return "Ignore for now", "Paid and not-paid medians are the same"
    if flagged_cases < 2 or lift < 1.5:
        return "Ignore for now", "Weak lift or too few flagged cases"

    clean_signal = flagged_paid == 0 and bad_capture >= 0.4 and lift >= 2.0
    useful_signal = bad_capture >= 0.4 and paid_capture <= 0.35 and lift >= 1.8
    if clean_signal:
        return "Use as candidate rule", "Clean separation in this sample; validate before changing scorecard"
    if useful_signal:
        return "Review manually", "Useful separation, but it also catches some paid cases"
    return "Review manually", "Exploratory signal; keep monitoring with more outcomes"


def build_threshold_recommendations(results_df: pd.DataFrame) -> pd.DataFrame:
    """Test simple one-feature thresholds and rank practical scorecard candidates."""
    labelled = _labelled_outcome_frame(results_df)
    if labelled.empty:
        return pd.DataFrame()

    total_labelled = len(labelled)
    total_bad = int(labelled["bad_outcome"].sum())
    total_paid = total_labelled - total_bad
    base_bad_rate = total_bad / total_labelled if total_labelled else 0
    rows = []

    for feature in _numeric_calibration_features(labelled):
        working = labelled[[feature, "bad_outcome"]].copy()
        working[feature] = pd.to_numeric(working[feature], errors="coerce")
        working = working.dropna(subset=[feature])
        if len(working) < 6 or working[feature].nunique() < 3:
            continue

        direction, paid_median, bad_median = _feature_direction(working, feature)
        if direction == "unknown":
            continue

        quantiles = working[feature].quantile([0.1, 0.2, 0.25, 0.33, 0.4, 0.5, 0.6, 0.67, 0.75, 0.8, 0.9])
        thresholds = sorted(set(float(v) for v in quantiles.dropna().tolist()))
        min_bucket = max(2, int(round(len(working) * 0.08)))
        best = None

        for threshold in thresholds:
            if direction == "low_is_risk":
                flagged = working[feature] <= threshold
                operator = "<="
            else:
                flagged = working[feature] >= threshold
                operator = ">="

            flagged_n = int(flagged.sum())
            unflagged_n = int((~flagged).sum())
            if flagged_n < min_bucket or unflagged_n < min_bucket:
                continue

            flagged_bad = int(working.loc[flagged, "bad_outcome"].sum())
            flagged_paid = flagged_n - flagged_bad
            bad_capture = flagged_bad / total_bad if total_bad else 0
            paid_capture = flagged_paid / total_paid if total_paid else 0
            flagged_bad_rate = flagged_bad / flagged_n if flagged_n else 0
            lift = flagged_bad_rate / base_bad_rate if base_bad_rate else 0
            separation = bad_capture - paid_capture
            score = (separation * 0.55) + ((lift - 1) * 0.20) + (bad_capture * 0.25)

            candidate = {
                "feature": feature,
                "direction": direction,
                "operator": operator,
                "suggested_threshold": threshold,
                "labelled_cases": total_labelled,
                "paid_cases": total_paid,
                "not_paid_cases": total_bad,
                "flagged_cases": flagged_n,
                "flagged_not_paid": flagged_bad,
                "flagged_paid": flagged_paid,
                "flagged_not_paid_rate": flagged_bad_rate,
                "bad_capture_rate": bad_capture,
                "paid_capture_rate": paid_capture,
                "lift_vs_base": lift,
                "separation": separation,
                "paid_median": paid_median,
                "not_paid_median": bad_median,
                "recommendation_score": score,
            }
            if best is None or candidate["recommendation_score"] > best["recommendation_score"]:
                best = candidate

        if best:
            confidence = calibration_confidence(total_labelled, total_bad, best["separation"])
            if best["feature"] in ["subprime_score", "mca_rule_score"]:
                action = "Consider recalibrating the overall decision band"
            else:
                action = "Consider testing this individual threshold in the scorecard"
            if total_labelled < 30 or total_bad < 8:
                action = f"Exploratory only; based on {total_labelled} labelled cases and {total_bad} not-paid cases"
            elif best["separation"] < 0.08:
                action = "Weak separation; monitor before changing the scorecard"

            best["confidence"] = confidence
            quality, quality_reason = classify_recommendation_quality(best)
            best["recommendation_quality"] = quality
            best["quality_reason"] = quality_reason
            best["suggested_action"] = action
            rows.append(best)

    if not rows:
        return pd.DataFrame()
    recs = pd.DataFrame(rows)
    quality_order = {"Use as candidate rule": 0, "Review manually": 1, "Ignore for now": 2}
    confidence_order = {"Strong": 0, "Moderate": 1, "Weak": 2}
    recs["_quality_order"] = recs["recommendation_quality"].map(quality_order).fillna(3)
    recs["_confidence_order"] = recs["confidence"].map(confidence_order).fillna(3)
    recs = recs.sort_values(
        ["_quality_order", "_confidence_order", "recommendation_score"],
        ascending=[True, True, False],
    )
    return recs.drop(columns=["_quality_order", "_confidence_order"]).reset_index(drop=True)


def build_rule_signal_report(results_df: pd.DataFrame) -> pd.DataFrame:
    """Summarise text rule reasons by outcome to reveal repeated scorecard drivers."""
    labelled = _labelled_outcome_frame(results_df)
    if labelled.empty:
        return pd.DataFrame()

    reason_cols = [col for col in ["mca_rule_reasons", "final_decision_reasons"] if col in labelled.columns]
    if not reason_cols:
        return pd.DataFrame()

    rows = []
    for _, row in labelled.iterrows():
        outcome = row["outcome_label"]
        for col in reason_cols:
            reasons = row.get(col)
            if isinstance(reasons, str):
                parts = [part.strip() for part in re.split(r"\s*\|\s*|\n", reasons) if part.strip()]
            elif isinstance(reasons, list):
                parts = [str(part).strip() for part in reasons if str(part).strip()]
            else:
                parts = []
            for reason in parts:
                rows.append({"reason_source": col, "reason": reason, "outcome_label": outcome})

    if not rows:
        return pd.DataFrame()

    reason_df = pd.DataFrame(rows)
    pivot = (
        reason_df.assign(count=1)
        .pivot_table(index=["reason_source", "reason"], columns="outcome_label", values="count", aggfunc="sum", fill_value=0)
        .reset_index()
    )
    for col in ["paid", "not_paid"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot["total_mentions"] = pivot["paid"] + pivot["not_paid"]
    pivot["not_paid_share"] = pivot["not_paid"] / pivot["total_mentions"].replace(0, np.nan)
    return pivot.sort_values(["not_paid_share", "total_mentions"], ascending=[False, False]).reset_index(drop=True)


def build_bureau_signal_report(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compare business credit PDF signals across paid and not-paid outcomes."""
    labelled = _labelled_outcome_frame(results_df)
    if labelled.empty:
        return pd.DataFrame()

    signal_cols = [
        "business_ccj",
        "business_credit_score_suppressed",
        "business_bureau_needs_attention",
        "business_no_registered_charges",
    ]
    numeric_cols = [
        "business_credit_score",
        "business_credit_limit",
        "business_max_recommended_credit",
        "business_negative_impact_count",
        "business_enquiries_3m",
        "business_company_searches_12m",
    ]

    rows = []
    for col in signal_cols:
        if col not in labelled.columns:
            continue
        flag = labelled[col].fillna(False).astype(bool)
        for flagged_value, group in labelled.groupby(flag):
            total = len(group)
            if total == 0:
                continue
            not_paid = int(group["bad_outcome"].sum())
            rows.append(
                {
                    "signal": col,
                    "signal_type": "boolean",
                    "bucket": "Yes" if flagged_value else "No",
                    "applications": total,
                    "paid": total - not_paid,
                    "not_paid": not_paid,
                    "not_paid_rate": not_paid / total,
                    "paid_median": None,
                    "not_paid_median": None,
                }
            )

    for col in numeric_cols:
        if col not in labelled.columns:
            continue
        working = labelled[[col, "bad_outcome"]].copy()
        working[col] = pd.to_numeric(working[col], errors="coerce")
        working = working.dropna(subset=[col])
        if working.empty:
            continue
        paid_median = working.loc[working["bad_outcome"] == 0, col].median()
        not_paid_median = working.loc[working["bad_outcome"] == 1, col].median()
        rows.append(
            {
                "signal": col,
                "signal_type": "numeric",
                "bucket": "median comparison",
                "applications": len(working),
                "paid": int((working["bad_outcome"] == 0).sum()),
                "not_paid": int((working["bad_outcome"] == 1).sum()),
                "not_paid_rate": working["bad_outcome"].mean(),
                "paid_median": paid_median,
                "not_paid_median": not_paid_median,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["signal", "bucket"]).reset_index(drop=True)


def build_evidence_quality_report(results_df: pd.DataFrame) -> pd.DataFrame:
    """Summarise which evidence sources were available in the processed batch."""
    if results_df.empty:
        return pd.DataFrame()

    checks = [
        ("Bank JSON processed", "processing_successful"),
        ("Application metadata matched", "parameters_applied_from_csv"),
        ("Business PDF matched", "bureau_pdf_file"),
        ("Business CCJ extracted", "business_ccj"),
        ("Business bureau score suppressed", "business_credit_score_suppressed"),
        ("Business bureau negative factors", "business_negative_impact_count"),
        ("Outcome label present", "outcome_label"),
    ]
    rows = []
    total = len(results_df)
    for label, col in checks:
        if col not in results_df.columns:
            rows.append({"evidence": label, "available": 0, "missing": total, "coverage": 0.0, "note": "Column not present"})
            continue
        series = results_df[col]
        if col == "outcome_label":
            available_mask = series.isin(["paid", "not_paid"])
        elif series.dtype == bool:
            available_mask = series.notna()
        else:
            available_mask = series.notna() & (series.astype(str).str.strip() != "")
        available = int(available_mask.sum())
        rows.append(
            {
                "evidence": label,
                "available": available,
                "missing": total - available,
                "coverage": available / total if total else 0,
                "note": "Used for calibration" if label == "Outcome label present" else "Used for score confidence",
            }
        )
    return pd.DataFrame(rows)


def build_paid_lookalike_report(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compare every application with the paid-case profile using robust numeric distance."""
    if "outcome_label" not in results_df.columns:
        return pd.DataFrame()

    paid_df = results_df[results_df["outcome_label"] == "paid"].copy()
    if len(paid_df) < 2:
        return pd.DataFrame()

    feature_cols = []
    for feature in _numeric_calibration_features(results_df):
        numeric = pd.to_numeric(results_df[feature], errors="coerce")
        paid_numeric = pd.to_numeric(paid_df[feature], errors="coerce")
        if numeric.notna().sum() >= 5 and paid_numeric.notna().sum() >= 2 and numeric.nunique(dropna=True) >= 2:
            feature_cols.append(feature)

    if not feature_cols:
        return pd.DataFrame()

    numeric_all = results_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    paid_numeric = paid_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    paid_profile = paid_numeric.median()
    paid_iqr = paid_numeric.quantile(0.75) - paid_numeric.quantile(0.25)
    overall_iqr = numeric_all.quantile(0.75) - numeric_all.quantile(0.25)
    scale = paid_iqr.where(paid_iqr > 0, overall_iqr).replace(0, np.nan).fillna(numeric_all.std()).replace(0, 1).fillna(1)

    rows = []
    paid_reference = paid_df.reset_index(drop=True)
    paid_matrix = paid_reference[feature_cols].apply(pd.to_numeric, errors="coerce")

    for idx, row in results_df.iterrows():
        values = pd.to_numeric(row[feature_cols], errors="coerce")
        available = values.notna() & paid_profile.notna()
        if int(available.sum()) < 3:
            continue

        z_distance = ((values[available] - paid_profile[available]).abs() / scale[available]).clip(upper=5)
        avg_distance = float(z_distance.mean())
        similarity = max(0, min(100, 100 / (1 + avg_distance)))

        nearest_paid_name = None
        nearest_paid_distance = None
        nearest_paid_similarity = None
        if not paid_matrix.empty:
            distances = []
            for paid_idx, paid_row in paid_matrix.iterrows():
                common = values.notna() & paid_row.notna()
                if int(common.sum()) < 3:
                    continue
                dist = float((((values[common] - paid_row[common]).abs() / scale[common]).clip(upper=5)).mean())
                distances.append((paid_idx, dist))
            if distances:
                nearest_idx, nearest_paid_distance = min(distances, key=lambda item: item[1])
                nearest_paid_name = paid_reference.loc[nearest_idx].get("company_name", paid_reference.loc[nearest_idx].get("original_filename"))
                nearest_paid_similarity = max(0, min(100, 100 / (1 + nearest_paid_distance)))

        if similarity >= 75:
            band = "Strong paid lookalike"
        elif similarity >= 60:
            band = "Moderate paid lookalike"
        elif similarity >= 45:
            band = "Weak paid lookalike"
        else:
            band = "Not close to paid profile"

        identifier = row.get("company_name") or row.get("original_filename") or idx
        rows.append(
            {
                "application_id": row.get("application_id"),
                "company_name": identifier,
                "original_filename": row.get("original_filename"),
                "outcome_label": row.get("outcome_label"),
                "paid_profile_similarity": similarity,
                "paid_lookalike_band": band,
                "nearest_paid_case": nearest_paid_name,
                "nearest_paid_similarity": nearest_paid_similarity,
                "features_used": int(available.sum()),
                "subprime_score": row.get("subprime_score"),
                "mca_rule_score": row.get("mca_rule_score"),
                "final_decision": row.get("final_decision"),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("paid_profile_similarity", ascending=False).reset_index(drop=True)


def build_calibration_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """High-level calibration health check for the processed batch."""
    labelled = _labelled_outcome_frame(results_df)
    total_processed = len(results_df)
    labelled_n = len(labelled)
    not_paid = int(labelled["bad_outcome"].sum()) if not labelled.empty else 0
    paid = labelled_n - not_paid
    unlabelled = int((results_df.get("outcome_label", pd.Series(dtype=str)) == "unlabelled").sum()) if "outcome_label" in results_df else total_processed
    conflict = int((results_df.get("outcome_label", pd.Series(dtype=str)) == "conflict").sum()) if "outcome_label" in results_df else 0
    confidence = calibration_confidence(labelled_n, not_paid, 0)

    rows = [
        {"metric": "Processed applications", "value": total_processed, "note": "All JSONs processed successfully"},
        {"metric": "Labelled paid/not-paid cases", "value": labelled_n, "note": "Used for calibration analysis"},
        {"metric": "Paid cases", "value": paid, "note": "Good outcome sample"},
        {"metric": "Not-paid cases", "value": not_paid, "note": "Bad outcome sample"},
        {"metric": "Unlabelled cases", "value": unlabelled, "note": "Useful for score distribution, not outcome calibration"},
        {"metric": "Outcome conflicts", "value": conflict, "note": "Must be resolved before trusting exports"},
        {"metric": "Calibration confidence", "value": confidence, "note": "Conservative because outcome data is limited"},
    ]
    return pd.DataFrame(rows)


def build_calibration_reports(results_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build all calibration artefacts in one place for UI and exports."""
    return {
        "summary": build_calibration_summary(results_df),
        "score_bands": build_score_band_report(results_df),
        "threshold_recommendations": build_threshold_recommendations(results_df),
        "rule_signals": build_rule_signal_report(results_df),
        "bureau_signals": build_bureau_signal_report(results_df),
        "evidence_quality": build_evidence_quality_report(results_df),
        "paid_lookalikes": build_paid_lookalike_report(results_df),
    }


SHARED_SAVED_RUNS_RELATIVE_PATH = Path(
    "Merchant Cash Advance (MCA)",
    "Scorecard",
    "Scorecard Development",
    "Saved_batch_processor_runs",
)
SAVVY_ONEDRIVE_FOLDER_NAME = "OneDrive - Savvy Loan Products Ltd"


def _saved_run_candidates() -> list[Path]:
    """Return likely shared saved-run folders for different OneDrive layouts."""
    candidates: list[Path] = []

    for env_name in ("OneDriveCommercial", "OneDrive", "ONEDRIVE", "OneDriveConsumer"):
        env_path = os.getenv(env_name)
        if env_path:
            candidates.append(Path(env_path).expanduser() / SHARED_SAVED_RUNS_RELATIVE_PATH)

    user_profile = os.getenv("USERPROFILE")
    if user_profile:
        candidates.append(Path(user_profile) / SAVVY_ONEDRIVE_FOLDER_NAME / SHARED_SAVED_RUNS_RELATIVE_PATH)

    for drive in ("C:", "D:"):
        users_root = Path(drive) / "Users"
        if users_root.exists():
            try:
                for user_dir in users_root.iterdir():
                    candidates.append(user_dir / SAVVY_ONEDRIVE_FOLDER_NAME / SHARED_SAVED_RUNS_RELATIVE_PATH)
            except OSError:
                continue

    seen: set[str] = set()
    unique_candidates = []
    for candidate in candidates:
        key = str(candidate).lower()
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)
    return unique_candidates


def resolve_saved_runs_dir() -> Path:
    """Resolve the saved-runs folder, allowing a shared path across machines."""
    configured_path = os.getenv("MCAV2_BATCH_SAVED_RUNS_DIR", "").strip().strip('"')
    if configured_path:
        return Path(configured_path).expanduser()
    for candidate in _saved_run_candidates():
        if candidate.exists():
            return candidate
    return BATCH_PROCESSOR_DIR / "saved_runs"


SAVED_RUNS_DIR = resolve_saved_runs_dir()


def slugify_run_name(name: str) -> str:
    """Create a stable folder-safe run id from the user-facing run name."""
    cleaned = re.sub(r"[^a-zA-Z0-9._ -]+", "", str(name or "").strip())
    cleaned = re.sub(r"\s+", "_", cleaned).strip("._-")
    return cleaned[:80] or datetime.now().strftime("batch_run_%Y%m%d_%H%M%S")


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    if df is None or df.empty:
        return b""
    export_df = df.copy()
    for col in export_df.columns:
        if export_df[col].apply(lambda value: isinstance(value, (list, dict))).any():
            export_df[col] = export_df[col].apply(
                lambda value: json.dumps(value, default=str) if isinstance(value, (list, dict)) else value
            )
    return export_df.to_csv(index=False).encode("utf-8")


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value) if not isinstance(value, (dict, list, tuple, np.ndarray)) else False:
        return None
    return value


def save_uploaded_sources(run_dir: Path, uploads: dict[str, Any]) -> None:
    """Save original uploaded files where practical so a run can be reproduced."""
    input_dir = run_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    for group, files in uploads.items():
        if not files:
            continue
        group_dir = input_dir / group
        group_dir.mkdir(parents=True, exist_ok=True)
        file_list = files if isinstance(files, list) else [files]
        for index, uploaded_file in enumerate(file_list, start=1):
            try:
                uploaded_file.seek(0)
                safe_name = re.sub(r"[^a-zA-Z0-9._ -]+", "_", uploaded_file.name)
                target = group_dir / f"{index:03d}_{safe_name}"
                target.write_bytes(uploaded_file.getvalue())
                uploaded_file.seek(0)
            except Exception:
                continue


def save_named_run(
    run_name: str,
    results_df: pd.DataFrame,
    scorecard_features_df: pd.DataFrame,
    outcome_audit_df: pd.DataFrame,
    pdf_audit_df: pd.DataFrame,
    calibration_reports: dict[str, pd.DataFrame],
    metadata: dict[str, Any],
    uploads: dict[str, Any],
) -> Path:
    """Persist a processed batch run under the configured saved-runs folder."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{slugify_run_name(run_name)}"
    run_dir = SAVED_RUNS_DIR / run_id
    suffix = 2
    while run_dir.exists():
        run_dir = SAVED_RUNS_DIR / f"{run_id}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)

    (run_dir / "case_scores_with_outcomes.csv").write_bytes(dataframe_to_csv_bytes(results_df))
    (run_dir / "scorecard_features.csv").write_bytes(dataframe_to_csv_bytes(scorecard_features_df))
    (run_dir / "matching_audit.csv").write_bytes(dataframe_to_csv_bytes(outcome_audit_df))
    if pdf_audit_df is not None and not pdf_audit_df.empty:
        (run_dir / "bureau_pdf_audit.csv").write_bytes(dataframe_to_csv_bytes(pdf_audit_df))

    reports_dir = run_dir / "calibration_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    for report_name, report_df in calibration_reports.items():
        (reports_dir / f"{report_name}.csv").write_bytes(dataframe_to_csv_bytes(report_df))

    manifest = {
        "run_name": run_name,
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "metadata": json_safe(metadata),
        "files": {
            "case_scores": "case_scores_with_outcomes.csv",
            "scorecard_features": "scorecard_features.csv",
            "matching_audit": "matching_audit.csv",
            "pdf_audit": "bureau_pdf_audit.csv" if pdf_audit_df is not None and not pdf_audit_df.empty else None,
            "calibration_reports": "calibration_reports",
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    save_uploaded_sources(run_dir, uploads)
    return run_dir


def build_saved_run_package(run_dir: Path) -> bytes:
    """Create a portable zip package for a saved run folder."""
    run_dir = Path(run_dir)
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in run_dir.rglob("*"):
            if path.is_file():
                archive.write(path, arcname=str(Path(run_dir.name) / path.relative_to(run_dir)))
    buffer.seek(0)
    return buffer.getvalue()


def import_saved_run_package(uploaded_file: Any) -> dict[str, Any]:
    """Import a portable saved-run zip package into the configured saved-runs folder."""
    SAVED_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    package_bytes = uploaded_file.getvalue()
    with zipfile.ZipFile(BytesIO(package_bytes), "r") as archive:
        manifest_entries = [
            name for name in archive.namelist()
            if Path(name).name == "manifest.json" and not name.endswith("/")
        ]
        if not manifest_entries:
            raise ValueError("No manifest.json was found in the saved run package.")

        manifest_entry = manifest_entries[0]
        package_root = str(Path(manifest_entry).parent).replace("\\", "/")
        manifest = json.loads(archive.read(manifest_entry).decode("utf-8"))
        base_run_id = slugify_run_name(manifest.get("run_id") or manifest.get("run_name") or Path(package_root).name)
        target_dir = SAVED_RUNS_DIR / base_run_id
        suffix = 2
        while target_dir.exists():
            target_dir = SAVED_RUNS_DIR / f"{base_run_id}_{suffix}"
            suffix += 1
        target_dir.mkdir(parents=True, exist_ok=False)

        for member in archive.infolist():
            if member.is_dir():
                continue
            member_name = member.filename.replace("\\", "/")
            if package_root and not member_name.startswith(f"{package_root}/"):
                continue
            relative_name = member_name[len(package_root):].lstrip("/") if package_root else member_name
            relative_path = Path(relative_name)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                continue
            target_path = target_dir / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(archive.read(member))

    imported_manifest_path = target_dir / "manifest.json"
    imported_manifest = json.loads(imported_manifest_path.read_text(encoding="utf-8"))
    imported_manifest["path"] = str(target_dir)
    return imported_manifest


def list_saved_runs() -> list[dict[str, Any]]:
    """Return saved run manifests, newest first."""
    if not SAVED_RUNS_DIR.exists():
        return []
    runs = []
    for manifest_path in SAVED_RUNS_DIR.glob("*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["path"] = str(manifest_path.parent)
            runs.append(manifest)
        except Exception:
            continue
    return sorted(runs, key=lambda item: item.get("created_at", ""), reverse=True)


def resolve_saved_run_dir(manifest: dict[str, Any]) -> Path:
    """Resolve a saved run against the current machine, ignoring stale absolute paths."""
    run_id = str(manifest.get("run_id", "")).strip()
    candidates = []
    if run_id:
        candidates.append(SAVED_RUNS_DIR / run_id)
    manifest_path = manifest.get("path")
    if manifest_path:
        candidates.append(Path(str(manifest_path)))

    for candidate in candidates:
        if candidate.exists() and (candidate / "manifest.json").exists():
            return candidate

    if run_id and SAVED_RUNS_DIR.exists():
        matches = list(SAVED_RUNS_DIR.glob(f"*{run_id}*"))
        for match in matches:
            if match.is_dir() and (match / "manifest.json").exists():
                return match

    return candidates[0] if candidates else SAVED_RUNS_DIR


def _read_saved_csv(run_dir: Path, relative_path: str | None) -> pd.DataFrame:
    if not relative_path:
        return pd.DataFrame()
    path = run_dir / relative_path
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def load_saved_run(manifest: dict[str, Any]) -> dict[str, Any]:
    """Load a saved run's output files back into memory for display."""
    run_dir = resolve_saved_run_dir(manifest)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["path"] = str(run_dir)
    files = manifest.get("files", {})
    reports_dir = run_dir / str(files.get("calibration_reports", "calibration_reports"))
    reports = {}
    for report_name in [
        "summary",
        "score_bands",
        "threshold_recommendations",
        "rule_signals",
        "bureau_signals",
        "evidence_quality",
        "paid_lookalikes",
    ]:
        reports[report_name] = _read_saved_csv(reports_dir, f"{report_name}.csv")

    return {
        "manifest": manifest,
        "run_dir": run_dir,
        "results_df": _read_saved_csv(run_dir, files.get("case_scores")),
        "scorecard_features_df": _read_saved_csv(run_dir, files.get("scorecard_features")),
        "outcome_audit_df": _read_saved_csv(run_dir, files.get("matching_audit")),
        "pdf_audit_df": _read_saved_csv(run_dir, files.get("pdf_audit")),
        "calibration_reports": reports,
    }


def render_loaded_saved_run(saved_run: dict[str, Any]) -> None:
    """Render a saved run without reprocessing the original inputs."""
    manifest = saved_run["manifest"]
    run_dir = saved_run["run_dir"]
    results_df = saved_run["results_df"]
    scorecard_features_df = saved_run["scorecard_features_df"]
    outcome_audit_df = saved_run["outcome_audit_df"]
    pdf_audit_df = saved_run["pdf_audit_df"]
    saved_reports = saved_run["calibration_reports"]

    section_title("Loaded Saved Run", "Viewing saved outputs without reprocessing the uploaded files.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Run", manifest.get("run_name", manifest.get("run_id", "Saved run")))
    with c2:
        st.metric("Created", manifest.get("created_at", "Unknown"))
    with c3:
        st.metric("Cases", len(results_df))
    st.caption(str(run_dir))
    with st.expander("Portable saved run package", expanded=False):
        st.caption("Only needed if you want to move this run without the shared OneDrive folder.")
        if st.button("Prepare Portable Saved Run", key=f"prepare_saved_run_{manifest.get('run_id', 'run')}"):
            try:
                st.session_state["prepared_saved_run_package"] = {
                    "run_id": manifest.get("run_id", ""),
                    "data": build_saved_run_package(run_dir),
                    "name": f"{slugify_run_name(manifest.get('run_name', 'saved_run'))}_saved_run.zip",
                }
            except OSError as e:
                st.error(f"Could not prepare portable package from {run_dir}: {e}")
        prepared_package = st.session_state.get("prepared_saved_run_package")
        if prepared_package and prepared_package.get("run_id") == manifest.get("run_id", ""):
            st.download_button(
                "Download Portable Saved Run",
                data=prepared_package["data"],
                file_name=prepared_package["name"],
                mime="application/zip",
            )

    if results_df.empty:
        st.warning("This saved run does not contain case score output.")
        return

    if "outcome_label" in results_df.columns:
        section_title("Outcome Summary")
        st.dataframe(results_df["outcome_label"].value_counts().reset_index(), use_container_width=True, hide_index=True)

    if not outcome_audit_df.empty:
        with st.expander("Matching Audit", expanded=False):
            st.dataframe(outcome_audit_df, use_container_width=True, hide_index=True)

    if not pdf_audit_df.empty:
        with st.expander("Company Credit Report PDF Audit", expanded=False):
            st.dataframe(pdf_audit_df, use_container_width=True, hide_index=True)

    section_title("Saved Run Downloads")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.download_button(
            "Download Case Scores",
            data=dataframe_to_csv_bytes(results_df),
            file_name=f"{slugify_run_name(manifest.get('run_name', 'saved_run'))}_case_scores.csv",
            mime="text/csv",
            type="primary",
        )
    with d2:
        st.download_button(
            "Download Scorecard Features",
            data=dataframe_to_csv_bytes(scorecard_features_df),
            file_name=f"{slugify_run_name(manifest.get('run_name', 'saved_run'))}_scorecard_features.csv",
            mime="text/csv",
        )
    with d3:
        st.download_button(
            "Download Matching Audit",
            data=dataframe_to_csv_bytes(outcome_audit_df),
            file_name=f"{slugify_run_name(manifest.get('run_name', 'saved_run'))}_matching_audit.csv",
            mime="text/csv",
        )
    with d4:
        if not pdf_audit_df.empty:
            st.download_button(
                "Download PDF Audit",
                data=dataframe_to_csv_bytes(pdf_audit_df),
                file_name=f"{slugify_run_name(manifest.get('run_name', 'saved_run'))}_pdf_audit.csv",
                mime="text/csv",
            )

    calibration_reports = render_scorecard_calibration(results_df)
    section_title("Calibration Report Exports")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.download_button(
            "Download Calibration Summary",
            data=dataframe_to_csv_bytes(saved_reports.get("summary", calibration_reports["summary"])),
            file_name=f"{slugify_run_name(manifest.get('run_name', 'saved_run'))}_calibration_summary.csv",
            mime="text/csv",
        )
    with c2:
        st.download_button(
            "Download Threshold Recommendations",
            data=dataframe_to_csv_bytes(saved_reports.get("threshold_recommendations", calibration_reports["threshold_recommendations"])),
            file_name=f"{slugify_run_name(manifest.get('run_name', 'saved_run'))}_threshold_recommendations.csv",
            mime="text/csv",
        )
    with c3:
        st.download_button(
            "Download Score Band Report",
            data=dataframe_to_csv_bytes(saved_reports.get("score_bands", calibration_reports["score_bands"])),
            file_name=f"{slugify_run_name(manifest.get('run_name', 'saved_run'))}_score_bands.csv",
            mime="text/csv",
        )
    with c4:
        st.download_button(
            "Download Rule Signal Report",
            data=dataframe_to_csv_bytes(saved_reports.get("rule_signals", calibration_reports["rule_signals"])),
            file_name=f"{slugify_run_name(manifest.get('run_name', 'saved_run'))}_rule_signals.csv",
            mime="text/csv",
        )
    st.download_button(
        "Download Paid Lookalike Report",
        data=dataframe_to_csv_bytes(saved_reports.get("paid_lookalikes", calibration_reports["paid_lookalikes"])),
        file_name=f"{slugify_run_name(manifest.get('run_name', 'saved_run'))}_paid_lookalikes.csv",
        mime="text/csv",
    )
    c5, c6 = st.columns(2)
    with c5:
        st.download_button(
            "Download Bureau Signal Report",
            data=dataframe_to_csv_bytes(saved_reports.get("bureau_signals", calibration_reports["bureau_signals"])),
            file_name=f"{slugify_run_name(manifest.get('run_name', 'saved_run'))}_bureau_signals.csv",
            mime="text/csv",
        )
    with c6:
        st.download_button(
            "Download Evidence Quality Report",
            data=dataframe_to_csv_bytes(saved_reports.get("evidence_quality", calibration_reports["evidence_quality"])),
            file_name=f"{slugify_run_name(manifest.get('run_name', 'saved_run'))}_evidence_quality.csv",
            mime="text/csv",
        )

    create_results_dashboard(results_df)
    st.session_state["batch_results"] = results_df


def render_scorecard_calibration(results_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Render scorecard calibration workbench and return exportable reports."""
    reports = build_calibration_reports(results_df)
    labelled = _labelled_outcome_frame(results_df)

    section_title(
        "Scorecard Calibration",
        "Use labelled paid/not-paid outcomes to test score bands, individual thresholds, and recurring rule reasons.",
    )

    summary_df = reports["summary"]
    if not summary_df.empty:
        metric_lookup = dict(zip(summary_df["metric"], summary_df["value"]))
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Labelled Cases", metric_lookup.get("Labelled paid/not-paid cases", 0))
        with c2:
            st.metric("Paid", metric_lookup.get("Paid cases", 0))
        with c3:
            st.metric("Not Paid", metric_lookup.get("Not-paid cases", 0))
        with c4:
            st.metric("Confidence", metric_lookup.get("Calibration confidence", "Weak"))

    if labelled.empty or labelled["bad_outcome"].sum() == 0:
        st.warning("Calibration needs at least some paid and not-paid outcomes. Exports are still available, but recommendations will be limited.")
        return reports

    tabs = st.tabs([
        "Recommendations",
        "Paid Lookalikes",
        "Score Bands",
        "Bureau Signals",
        "Rule Signals",
        "Evidence Quality",
        "Calibration Data",
    ])

    with tabs[0]:
        recs = reports["threshold_recommendations"]
        if recs.empty:
            st.info("No threshold candidates were found in this run. Paid lookalikes may still identify applications that resemble good funded cases.")
        else:
            display_cols = [
                "feature",
                "recommendation_quality",
                "quality_reason",
                "operator",
                "suggested_threshold",
                "confidence",
                "suggested_action",
                "flagged_cases",
                "flagged_not_paid_rate",
                "bad_capture_rate",
                "paid_capture_rate",
                "lift_vs_base",
                "paid_median",
                "not_paid_median",
            ]
            st.dataframe(recs[[col for col in display_cols if col in recs.columns]].head(20), use_container_width=True, hide_index=True)

            top = recs.iloc[0]
            st.info(
                f"Top candidate: {top['feature']} {top['operator']} {top['suggested_threshold']:.2f}. "
                f"{top.get('recommendation_quality', 'Review manually')} with {top['flagged_cases']} flagged cases. "
                f"Confidence is {top['confidence'].lower()}."
            )

    with tabs[1]:
        lookalikes = reports["paid_lookalikes"]
        if lookalikes.empty:
            st.info("Paid lookalike analysis needs at least two paid cases and enough numeric features.")
        else:
            focus = lookalikes[lookalikes["outcome_label"].isin(["unlabelled", "not_paid", "conflict"])].copy()
            if focus.empty:
                focus = lookalikes.copy()
            display_cols = [
                "application_id",
                "company_name",
                "outcome_label",
                "paid_profile_similarity",
                "paid_lookalike_band",
                "nearest_paid_case",
                "nearest_paid_similarity",
                "features_used",
                "subprime_score",
                "final_decision",
            ]
            st.dataframe(focus[[col for col in display_cols if col in focus.columns]].head(50), use_container_width=True, hide_index=True)

            chart_df = focus.head(30).copy()
            chart_df["label"] = chart_df["company_name"].astype(str).str.slice(0, 42)
            fig = px.bar(
                chart_df.sort_values("paid_profile_similarity", ascending=True),
                x="paid_profile_similarity",
                y="label",
                color="outcome_label",
                orientation="h",
                title="Applications closest to the paid-case profile",
                labels={"paid_profile_similarity": "Paid Profile Similarity", "label": "Application"},
            )
            fig.update_xaxes(range=[0, 100])
            plot_mca_chart(fig, key="paid_lookalike_similarity")

    with tabs[2]:
        bands = reports["score_bands"]
        if bands.empty:
            st.info("Score band analysis is not available for this run.")
        else:
            st.dataframe(bands, use_container_width=True, hide_index=True)
            for score_col in bands["score"].dropna().unique():
                chart_df = bands[bands["score"] == score_col]
                fig = px.bar(
                    chart_df,
                    x="score_band",
                    y="not_paid_rate",
                    text=chart_df["applications"],
                    title=f"{score_col} not-paid rate by band",
                    labels={"score_band": "Score Band", "not_paid_rate": "Not-paid Rate"},
                )
                fig.update_yaxes(tickformat=".0%")
                plot_mca_chart(fig, key=f"calibration_{score_col}_bands")

    with tabs[3]:
        bureau = reports["bureau_signals"]
        if bureau.empty:
            st.info("No business bureau PDF signals were available for calibration in this run.")
        else:
            st.dataframe(bureau, use_container_width=True, hide_index=True)
            bool_rows = bureau[bureau["signal_type"] == "boolean"].copy()
            if not bool_rows.empty:
                fig = px.bar(
                    bool_rows,
                    x="signal",
                    y="not_paid_rate",
                    color="bucket",
                    barmode="group",
                    text="applications",
                    title="Business bureau signal not-paid rates",
                    labels={"not_paid_rate": "Not-paid Rate", "signal": "Bureau Signal"},
                )
                fig.update_yaxes(tickformat=".0%")
                plot_mca_chart(fig, key="bureau_signal_not_paid_rates")

    with tabs[4]:
        rules = reports["rule_signals"]
        if rules.empty:
            st.info("No rule reason signals were available in the processed results.")
        else:
            display_cols = ["reason_source", "reason", "paid", "not_paid", "total_mentions", "not_paid_share"]
            st.dataframe(rules[display_cols].head(30), use_container_width=True, hide_index=True)

    with tabs[5]:
        evidence = reports["evidence_quality"]
        if evidence.empty:
            st.info("Evidence quality report is not available for this run.")
        else:
            st.dataframe(evidence, use_container_width=True, hide_index=True)
            fig = px.bar(
                evidence,
                x="evidence",
                y="coverage",
                text="available",
                title="Evidence coverage across processed applications",
                labels={"coverage": "Coverage", "evidence": "Evidence Source"},
            )
            fig.update_yaxes(tickformat=".0%", range=[0, 1])
            plot_mca_chart(fig, key="evidence_quality_coverage")

    with tabs[6]:
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    return reports


def create_results_dashboard(results_df):
    """Create comprehensive dashboard for batch results"""
    
    if results_df.empty:
        st.warning("No results to display")
        return
    
    # Summary Statistics
    section_title("Batch Processing Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Applications Processed", len(results_df))
    
    with col2:
        avg_subprime = results_df['subprime_score'].mean() if 'subprime_score' in results_df.columns else 0
        st.metric("Avg Subprime Score", f"{avg_subprime:.1f}")
    
    with col3:
        avg_mca = results_df['mca_rule_score'].mean() if 'mca_rule_score' in results_df.columns else 0
        st.metric("Avg MCA Rule Score", f"{avg_mca:.1f}")
    
    with col4:
        if 'ml_score' in results_df.columns and results_df['ml_score'].notna().any():
            avg_ml = results_df['ml_score'].mean()
            st.metric("Avg ML Score (Info Only)", f"{avg_ml:.1f}%")
        else:
            st.metric("Avg ML Score (Info Only)", "N/A")
    
    with col5:
        avg_revenue = results_df['Total Revenue'].mean() if 'Total Revenue' in results_df.columns else 0
        st.metric("Avg Revenue", f"£{avg_revenue:,.0f}")
    
    # Score Distribution Charts
    section_title("Score Distributions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'subprime_score' in results_df.columns:
            fig_subprime = px.histogram(
                results_df, 
                x='subprime_score', 
                title="Subprime Score Distribution",
                labels={'subprime_score': 'Subprime Score', 'count': 'Number of Applications'}
            )
            fig_subprime.add_vline(x=60, line_dash="dash", line_color="#fbbf24",
                                  annotation_text="Configured Threshold (60)")
            plot_mca_chart(fig_subprime, key="batch_subprime_distribution")
    
    with col2:
        if 'mca_rule_score' in results_df.columns:
            fig_mca = px.histogram(
                results_df, 
                x='mca_rule_score', 
                title="MCA Rule Score Distribution",
                labels={'mca_rule_score': 'MCA Rule Score', 'count': 'Number of Applications'}
            )
            fig_mca.add_vline(x=70, line_dash="dash", line_color="#fbbf24",
                              annotation_text="Typical Threshold (70)")
            plot_mca_chart(fig_mca, key="batch_mca_distribution")
    
    # Risk Tier Analysis
    if 'subprime_tier' in results_df.columns:
        section_title("Risk Tier Analysis")
        
        tier_counts = results_df['subprime_tier'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_tier = px.pie(
                values=tier_counts.values,
                names=tier_counts.index,
                title="Applications by Risk Tier",
                color_discrete_map={
                    'Tier 1': '#22c55e',
                    'Tier 2': '#3b82f6', 
                    'Tier 3': '#f59e0b',
                    'Tier 4': '#ef4444',
                    'Decline': '#6b7280'
                }
            )
            plot_mca_chart(fig_tier, key="batch_tier_distribution")
        
        with col2:
            st.write("**Risk Tier Breakdown:**")
            for tier, count in tier_counts.items():
                percentage = (count / len(results_df)) * 100
                st.write(f"- **{tier}**: {count} applications ({percentage:.1f}%)")
            
            # Approval rate calculation
            approval_tiers = ['Tier 1', 'Tier 2', 'Tier 3']
            approved = tier_counts[tier_counts.index.isin(approval_tiers)].sum()
            approval_rate = (approved / len(results_df)) * 100
            
            st.metric("Potential Approval Rate", f"{approval_rate:.1f}%")

    # ==========================
    # BP-5: MI Summary (On-screen)
    # ==========================
    section_title("MI Summary")

    c1, c2 = st.columns(2)

    with c1:
        if "final_decision" in results_df.columns:
            st.write("**Final Decision counts**")
            final_counts = results_df["final_decision"].fillna("UNKNOWN").value_counts()

            final_counts_df = final_counts.reset_index()
            final_counts_df.columns = ["final_decision", "count"]

            st.dataframe(final_counts_df, use_container_width=True)
        else:
            st.info("final_decision not found in results.")

    with c2:
        if "mca_rule_decision" in results_df.columns:
            st.write("**MCA Rule Decision counts**")
            mca_counts = results_df["mca_rule_decision"].fillna("UNKNOWN").value_counts()

            mca_counts_df = mca_counts.reset_index()
            mca_counts_df.columns = ["mca_rule_decision", "count"]

            st.dataframe(mca_counts_df, use_container_width=True)
        else:
            st.info("mca_rule_decision not found in results.")

    # Cross-tab: MCA Rule vs Final Decision
    if "mca_rule_decision" in results_df.columns and "final_decision" in results_df.columns:
        st.write("**MCA Rule to Final Decision cross-tab**")
        ctab = pd.crosstab(
            results_df["mca_rule_decision"].fillna("UNKNOWN"),
            results_df["final_decision"].fillna("UNKNOWN"),
            margins=True
        )
        st.dataframe(ctab, use_container_width=True)

    ob_cols = [
        "company_name",
        "outcome_label",
        "Open Banking Insights Used In Score",
        "OB History Months",
        "OB Transaction Count",
        "OB True Revenue",
        "OB Non-Revenue Inflow Ratio",
        "OB Revenue Active Day Rate",
        "OB Top Revenue Source Percentage",
        "OB Card Processor Revenue Share",
        "OB Weakest Month Revenue",
        "OB Debt Repayment Burden",
        "OB Recent Loan Credits 30D",
        "OB Low Balance Days <1000",
        "OB Recent Failed Payments 30D",
        "final_decision",
    ]
    available_ob_cols = [col for col in ob_cols if col in results_df.columns]
    if len(available_ob_cols) > 4:
        section_title("Open Banking Derived Insights")
        st.caption(
            "These extra transaction-derived fields are displayed and exported for review/calibration. "
            "They do not change scoring unless you later choose to add them to the scorecard."
        )
        st.dataframe(results_df[available_ob_cols].head(100), use_container_width=True, hide_index=True)

    # Detailed Results Table
    section_title("Detailed Results")

    # Prefer a curated set, but ALWAYS fall back to something visible
    preferred_columns = [
        # identifiers
        'original_filename', 'company_name', 'extracted_company_name', 'industry',
        # new stack outputs
        'mca_rule_decision', 'mca_rule_score', 'final_decision',
        # scores
        'subprime_score', 'subprime_tier',
        # bureau PDF signals
        'business_ccj', 'business_credit_score_suppressed', 'business_credit_limit',
        'business_max_recommended_credit', 'business_negative_impact_count', 'business_enquiries_3m',
        # key metrics
        'Total Revenue', 'Net Income', 'Operating Margin', 'Debt Service Coverage Ratio',
        'OB Non-Revenue Inflow Ratio', 'OB Debt Repayment Burden', 'OB Low Balance Days <1000',
        # params
        'requested_loan'
    ]

    available_columns = [c for c in preferred_columns if c in results_df.columns]

    if not available_columns:
        # Nothing matched (column naming mismatch) — show *something* so the table isn't empty
        st.warning("No preferred columns matched results; showing first 30 columns for debugging.")
        display_df = results_df.copy()
        display_df = display_df.iloc[:, :30]  # avoid a massive table
    else:
        display_df = results_df[available_columns].copy()

    # Format numeric columns (only if they exist in the current display_df)
    def _safe_numeric_format(x, fmt=".0f"):
        try:
            return f"{float(x):{fmt}}"
        except (TypeError, ValueError):
            return ""

    numeric_format_cols = {
        'Total Revenue': lambda x: f"£{_safe_numeric_format(x, ',.0f')}" if _safe_numeric_format(x, ',.0f') else "",
        'Net Income': lambda x: f"£{_safe_numeric_format(x, ',.0f')}" if _safe_numeric_format(x, ',.0f') else "",
        'requested_loan': lambda x: f"£{_safe_numeric_format(x, ',.0f')}" if _safe_numeric_format(x, ',.0f') else "",
        'Operating Margin': lambda x: _safe_numeric_format(float(x) * 100, '.1f') + "%" if _safe_numeric_format(x) else "",
        'subprime_score': lambda x: _safe_numeric_format(x, '.1f'),
        'mca_rule_score': lambda x: _safe_numeric_format(x, '.0f'),
        'Debt Service Coverage Ratio': lambda x: _safe_numeric_format(x, '.2f'),
    }

    for col, formatter in numeric_format_cols.items():
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(formatter)

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Download button (export-friendly)
    export_df = results_df.copy()

    for col in ["mca_rule_reasons", "final_decision_reasons"]:
        if col in export_df.columns:
            export_df[col] = export_df[col].apply(
                lambda x: " | ".join(x) if isinstance(x, list) else ("" if x is None else str(x))
            )

    csv_data = export_df.to_csv(index=False)
    st.download_button(
        label="Download Full Results (CSV)",
        data=csv_data,
        file_name=f"mcav2_batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        type="primary"
    )


def legacy_main():
    """Main application"""

    if render_main_hero:
        render_main_hero(
            "MCA v2 Batch Processor",
            "Score multiple applications, audit CSV-to-JSON matching, and export scorecard-ready results.",
            eyebrow="Batch scoring workspace",
        )
    else:
        st.title("MCA v2 Batch Processor")
        st.markdown("Score multiple applications, audit CSV-to-JSON matching, and export scorecard-ready results.")
    render_batch_workflow_rail()

    # Sidebar for parameters
    if sidebar_section:
        sidebar_section("Parameter Sources")
    else:
        st.sidebar.header("Parameter Sources")
    st.sidebar.markdown("""
    **Parameter Priority:**
    1. **Individual JSON files** (if they contain application data)
    2. **CSV mapping file** (upload a CSV with company-specific parameters)
    3. **Fallback defaults** (only used when data is missing)
    """)
    st.sidebar.caption(
        "Matching mode: RapidFuzz" if RAPIDFUZZ_AVAILABLE else "Matching mode: exact fallback"
    )
    
    # Option to upload parameter mapping
    if sidebar_subsection:
        sidebar_subsection("Upload Parameter Mapping")
    else:
        st.sidebar.subheader("Upload Parameter Mapping")
    parameter_file = st.sidebar.file_uploader(
        "Upload CSV with application parameters",
        type=['csv'],
        help="CSV should include columns: company_name, industry, directors_score, requested_loan, etc."
    )
    
    # Load parameter mapping if provided
    parameter_mapping = {}
    if parameter_file:
        try:
            param_df = pd.read_csv(parameter_file)
            st.sidebar.success(f"Loaded parameters for {len(param_df)} companies")
            
            # Show available columns
            available_cols = list(param_df.columns)
            expected_cols = ['industry', 'directors_score', 'requested_loan', 'company_age_months',
                             'business_ccj', 'director_ccj', 'poor_or_no_online_presence', 'uses_generic_email']
            
            missing_cols = [col for col in expected_cols if col not in available_cols]
            if missing_cols:
                st.sidebar.warning(f"Missing columns: {', '.join(missing_cols)}")
            
            identifier_candidates = [
                'company_name',
                'application_id',
                'Company Name',
                'company',
                'business_name',
                'Business Name',
            ]
            identifier_column = next((col for col in identifier_candidates if col in param_df.columns), None)

            # Convert to dictionary using the best available company/application identifier.
            if identifier_column:
                for _, row in param_df.iterrows():
                    company_name = row[identifier_column]
                    if pd.isna(company_name) or not str(company_name).strip():
                        continue

                    row_dict = row.to_dict()
                    row_dict['company_name'] = str(company_name).strip()
                    parameter_mapping[str(company_name).strip()] = row_dict
                
                st.sidebar.caption(f"Using `{identifier_column}` as the matching column")
                st.sidebar.info(f"CSV company examples: {list(parameter_mapping.keys())[:3]}")
            else:
                st.sidebar.error("CSV must have a company identifier column such as 'company_name' or 'application_id'")
            
            # Show preview
            with st.sidebar.expander("Preview Parameter CSV"):
                st.dataframe(param_df.head(), use_container_width=True)
                
            # Add diagnostic button here
            st.sidebar.markdown("---")
            if sidebar_subsection:
                sidebar_subsection("Diagnostic Tools")
            else:
                st.sidebar.subheader("Diagnostic Tools")
            if st.sidebar.button("Diagnose Specific Cases"):
                st.session_state['show_diagnostic'] = True
                
        except Exception as e:
            st.sidebar.error(f"Error reading parameter file: {e}")
            
    # Fallback defaults
    if sidebar_subsection:
        sidebar_subsection("Fallback Defaults")
    else:
        st.sidebar.subheader("Fallback Defaults")
    st.sidebar.markdown("*Only used when data is missing from JSON or CSV*")
    
    default_industry = st.sidebar.selectbox(
        "Fallback Industry", 
        list(INDUSTRY_THRESHOLDS.keys()),
        index=list(INDUSTRY_THRESHOLDS.keys()).index('Other')
    )
    
    default_loan = st.sidebar.number_input("Fallback Requested Loan (£)", min_value=0.0, value=5000.0, step=1000.0)
    default_directors_score = st.sidebar.slider("Fallback Director Credit Score", 0, 100, 75)
    default_company_age = st.sidebar.number_input("Fallback Company Age (Months)", min_value=0, value=12, step=1)

    # Risk factors
    if sidebar_subsection:
        sidebar_subsection("Default Risk Factors")
    else:
        st.sidebar.subheader("Default Risk Factors")
    default_business_ccj = st.sidebar.checkbox("Business CCJs", False)
    default_director_ccj = st.sidebar.checkbox("Director CCJs", False)
    default_poor_or_no_online = st.sidebar.checkbox("Poor/No Online Presence", False)
    default_generic_email = st.sidebar.checkbox("Generic Email", False)

    default_params = {
        'industry': default_industry,
        'requested_loan': default_loan,
        'directors_score': default_directors_score,
        'company_age_months': default_company_age,
        'business_ccj': default_business_ccj,
        'director_ccj': default_director_ccj,
        'poor_or_no_online_presence': default_poor_or_no_online,
        'uses_generic_email': default_generic_email
    }
    
    # NEW: Show diagnostic section here (after parameter_mapping is loaded)
    if st.session_state.get('show_diagnostic', False) and parameter_mapping:
        section_title("Direct Matching Diagnostic", "Inspect how a JSON filename is normalized against the CSV identifiers.")
    
        # Test the exact cases that are failing
        failing_cases = [
            ("22LUSH LTD", "22Lush Limited Transaction Report_0,0.json"),
        ]
    
        processor = BatchProcessor()
        csv_companies = list(parameter_mapping.keys())
    
        for csv_name, json_filename in failing_cases:
            st.write(f"**CSV:** `{csv_name}` vs **JSON:** `{json_filename}`")
            extracted_name, method = processor.extract_company_name_from_json({}, json_filename)
            st.write(f"**Extracted:** `{extracted_name}`")
    
            # Simple comparison
            csv_lower = csv_name.lower().replace('ltd', '').strip()
            extracted_lower = extracted_name.lower().replace('limited', '').strip()
    
            st.write(f"**CSV cleaned:** `{csv_lower}`")
            st.write(f"**Extracted cleaned:** `{extracted_lower}`")
    
            if csv_lower == extracted_lower:
                st.success("Match after simple cleaning")
            else:
                st.error(f"No match: '{csv_lower}' vs '{extracted_lower}'")
    
            break  # Just show first case for now
    
        if st.button("Close Diagnostic"):
            st.session_state['show_diagnostic'] = False
            st.rerun()
    
        st.markdown("---")
    
    # File upload section
    if render_intake_panel_intro:
        render_intake_panel_intro(
            title="Application batch",
            description="Upload individual JSON files or a ZIP archive containing multiple JSON files. The optional CSV mapping controls company-specific parameters.",
        )
    else:
        section_title("Application Batch", "Upload individual JSON files or a ZIP archive containing multiple JSON files.")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['json', 'zip'],
        accept_multiple_files=True,
        help="Upload JSON transaction files or ZIP archives. Each JSON should contain transaction data."
    )
    
    if uploaded_files:
        # Load files
        with st.spinner("Loading files..."):
            files_data = load_json_files(uploaded_files)
        
        if files_data:
            loaded_col, csv_col, ready_col = st.columns(3)
            with loaded_col:
                st.metric("JSON Files Loaded", len(files_data))
            with csv_col:
                st.metric("CSV Companies", len(parameter_mapping) if parameter_mapping else 0)
            with ready_col:
                st.metric("Ready to Process", "Yes")
            
            # Show file list
            with st.expander("Loaded Files", expanded=False):
                for filename, _ in files_data:
                    st.write(f"- {filename}")
            
            # Enhanced file matching analysis
            if parameter_mapping:
                section_title("File Matching Analysis", "Preview how uploaded JSON filenames align with the CSV mapping before processing.")
                
                # Get list of uploaded JSON filenames (without .json extension)
                uploaded_filenames = [filename.replace('.json', '') for filename, _ in files_data]
                
                # Get list of companies from CSV
                csv_companies = list(parameter_mapping.keys())
                
                # Perform fuzzy matching analysis
                potential_matches = []
                missing_jsons = []
                extra_jsons = []
                
                if RAPIDFUZZ_AVAILABLE:
                    # Use fuzzy matching for analysis
                    for csv_company in csv_companies:
                        match_result = process.extractOne(
                            csv_company,
                            uploaded_filenames,
                            scorer=fuzz.token_sort_ratio,
                            score_cutoff=60
                        )
                        if match_result:
                            potential_matches.append({
                                'csv_company': csv_company,
                                'json_file': match_result[0],
                                'score': match_result[1]
                            })
                        else:
                            missing_jsons.append(csv_company)
                    
                    # Find extra JSON files
                    matched_json_files = [m['json_file'] for m in potential_matches]
                    extra_jsons = [f for f in uploaded_filenames if f not in matched_json_files]
                
                else:
                    # Fallback exact matching
                    for csv_company in csv_companies:
                        if csv_company in uploaded_filenames:
                            potential_matches.append({
                                'csv_company': csv_company,
                                'json_file': csv_company,
                                'score': 100
                            })
                        else:
                            missing_jsons.append(csv_company)
                    
                    matched_json_files = [m['json_file'] for m in potential_matches]
                    extra_jsons = [f for f in uploaded_filenames if f not in matched_json_files]
                
                # Display matching results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Companies in CSV", len(csv_companies))
                with col2:
                    st.metric("JSON Files Uploaded", len(uploaded_filenames))
                with col3:
                    st.metric("Expected Matches", len(potential_matches))
                
                # Show potential matches
                if potential_matches:
                    st.success(f"{len(potential_matches)} potential matches found")
                    
                    with st.expander(f"Potential Matches ({len(potential_matches)})", expanded=True):
                        match_df = pd.DataFrame(potential_matches)
                        match_df['score'] = match_df['score'].apply(lambda x: f"{x:.1f}%")
                        match_df.columns = ['CSV Company', 'JSON File', 'Match Score']
                        st.dataframe(match_df, use_container_width=True, hide_index=True)
                
                # Show missing JSON files
                if missing_jsons:
                    st.warning(f"{len(missing_jsons)} companies from CSV do not have matching JSON files")
                    
                    with st.expander(f"Missing JSON Files ({len(missing_jsons)})", expanded=False):
                        missing_df = pd.DataFrame({
                            'Company Name (from CSV)': missing_jsons,
                            'Status': ['No matching JSON file found'] * len(missing_jsons)
                        })
                        st.dataframe(missing_df, use_container_width=True, hide_index=True)
                
                # Show extra JSON files
                if extra_jsons:
                    with st.expander(f"Extra JSON Files ({len(extra_jsons)})", expanded=False):
                        st.markdown("**JSON files uploaded but not in CSV (will use default parameters):**")
                        for filename in extra_jsons:
                            st.write(f"- {filename}.json")
            
            else:
                st.info("Upload a CSV file to see file matching analysis.")
            
            # Process button
            if st.button("Process All Applications", type="primary"):
                
                # Initialize processor
                processor = BatchProcessor()
                
                # Progress tracking
                progress_bar = st.progress(0, text="Starting batch processing...")
                
                # Process batch
                with st.spinner("Processing applications..."):
                    results_df = processor.process_batch(files_data, default_params, parameter_mapping, progress_bar)
                
                # Clear progress bar
                progress_bar.empty()
                
                # Show processing summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Processed Successfully", processor.processed_count)
                with col2:
                    st.metric("Processing Errors", processor.error_count)
                with col3:
                    success_rate = (processor.processed_count / len(files_data)) * 100
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                
                # ENHANCED: Show fuzzy matching results
                if not results_df.empty and 'fuzzy_match_success' in results_df.columns:
                    section_title("Fuzzy Matching Results")
                    
                    # Calculate fuzzy matching stats
                    total_processed = len(results_df)
                    successful_matches = results_df['fuzzy_match_success'].sum()
                    csv_params_applied = results_df['parameters_applied_from_csv'].sum()
                    using_defaults = results_df['using_defaults'].sum()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Applications Processed", total_processed)
                    with col2:
                        match_rate = (successful_matches / total_processed) * 100 if total_processed > 0 else 0
                        st.metric("Fuzzy Match Success", f"{match_rate:.1f}%")
                    with col3:
                        st.metric("CSV Parameters Applied", int(csv_params_applied))
                    with col4:
                        default_rate = (using_defaults / total_processed) * 100 if total_processed > 0 else 0
                        st.metric("Using Defaults", f"{default_rate:.1f}%")
                    
                    # Show detailed matching results
                    if successful_matches > 0:
                        with st.expander("Detailed Fuzzy Matching Results", expanded=False):
                            
                            # Successful matches
                            successful_df = results_df[results_df['fuzzy_match_success'] == True]
                            if not successful_df.empty:
                                st.write("**Successful Matches:**")
                                match_details = successful_df[['original_filename', 'extracted_company_name', 'fuzzy_match_company', 'fuzzy_match_score', 'fuzzy_match_strategy', 'parameters_applied_from_csv']].copy()
                                match_details['fuzzy_match_score'] = match_details['fuzzy_match_score'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
                                match_details.columns = ['JSON File', 'Extracted Name', 'CSV Matched Company', 'Match Score', 'Strategy', 'Params Applied']
                                st.dataframe(match_details, use_container_width=True, hide_index=True)
                            
                            # Failed matches
                            failed_df = results_df[results_df['fuzzy_match_success'] == False]
                            if not failed_df.empty:
                                st.write("**Failed Matches:**")
                                failed_details = failed_df[['original_filename', 'extracted_company_name', 'fuzzy_match_debug']].copy()
                                failed_details.columns = ['JSON File', 'Extracted Company Name', 'Debug Info']
                                st.dataframe(failed_details, use_container_width=True, hide_index=True)
                    
                    # Show parameter source breakdown
                    section_title("Parameter Source Analysis")
                    
                    param_source_data = []
                    for _, row in results_df.iterrows():
                        if row.get('fuzzy_match_success', False):
                            source = f"CSV Match ({row.get('fuzzy_match_score', 0):.1f}%)"
                        elif row.get('using_defaults', False):
                            source = "Fallback Defaults"
                        else:
                            source = "JSON Metadata"
                        
                        param_source_data.append({
                            'Company': row.get('company_name', 'Unknown'),
                            'Parameter Source': source,
                            'Industry': row.get('industry', 'Unknown'),
                            'Directors Score': row.get('directors_score', 'Unknown'),
                            'Requested Loan': f"£{row.get('requested_loan', 0):,.0f}"
                        })
                    
                    if param_source_data:
                        param_df = pd.DataFrame(param_source_data)
                        st.dataframe(param_df, use_container_width=True, hide_index=True)
                
                # Show errors if any
                if processor.error_log:
                    section_title("Processing Errors Analysis")
                    
                    # Categorize errors
                    error_categories = {}
                    for error in processor.error_log:
                        error_msg = error['error']
                        if 'No transactions found' in error_msg:
                            category = 'No Transactions'
                        elif 'Missing required columns' in error_msg:
                            category = 'Missing Required Columns'
                        elif 'No valid transactions after cleaning' in error_msg:
                            category = 'Invalid Transaction Data'
                        elif 'Could not calculate financial metrics' in error_msg:
                            category = 'Financial Calculation Error'
                        elif 'Invalid parameter data types' in error_msg:
                            category = 'Parameter Type Error'
                        elif 'Unknown industry' in error_msg:
                            category = 'Unknown Industry'
                        else:
                            category = 'Other Error'
                        
                        if category not in error_categories:
                            error_categories[category] = []
                        error_categories[category].append(error)
                    
                    # Show error summary
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Error Categories:**")
                        for category, errors in error_categories.items():
                            st.write(f"- **{category}**: {len(errors)} files")
                    
                    with col2:
                        st.write("**Most Common Errors:**")
                        sorted_categories = sorted(error_categories.items(), key=lambda x: len(x[1]), reverse=True)
                        for category, errors in sorted_categories[:3]:
                            percentage = (len(errors) / len(processor.error_log)) * 100
                            st.write(f"- {category}: {percentage:.1f}%")
                    
                    # Detailed error breakdown
                    with st.expander(f"Detailed Error Breakdown ({len(processor.error_log)} errors)", expanded=False):
                        error_df = pd.DataFrame(processor.error_log)
                        st.dataframe(error_df, use_container_width=True, hide_index=True)
                        
                        # Download error log
                        error_csv = error_df.to_csv(index=False)
                        st.download_button(
                            label="Download Error Log (CSV)",
                            data=error_csv,
                            file_name=f"processing_errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                
                # Display results dashboard
                if not results_df.empty:
                    st.markdown("---")
                    
                    # COMPREHENSIVE DEBUG SECTION - Show all debug information
                    section_title("Comprehensive Debug Information")
                    
                    # Processing summary with debug info
                    debug_summary = []
                    for _, row in results_df.iterrows():
                        debug_summary.append({
                            'File': row.get('original_filename', 'Unknown'),
                            'Company Extracted': row.get('extracted_company_name', 'None'),
                            'Extraction Method': row.get('extraction_method', 'None'),
                            'CSV Available': row.get('parameter_mapping_available', False),
                            'Fuzzy Match Success': row.get('fuzzy_match_success', False),
                            'Match Score': f"{row.get('fuzzy_match_score', 0):.1f}%",
                            'Match Strategy': row.get('fuzzy_match_strategy', 'None'),
                            'CSV Params Applied': row.get('parameters_applied_from_csv', 0),
                            'Using Defaults': row.get('using_defaults', False),
                            'Final Score': f"{row.get('subprime_score', 0):.1f}",
                            'Processing Status': 'SUCCESS' if row.get('processing_successful', False) else 'UNKNOWN'
                        })
                    
                    if debug_summary:
                        debug_df = pd.DataFrame(debug_summary)
                        st.dataframe(debug_df, use_container_width=True, hide_index=True)
                        
                        # Debug download
                        debug_csv = debug_df.to_csv(index=False)
                        st.download_button(
                            label="Download Debug Log (CSV)",
                            data=debug_csv,
                            file_name=f"debug_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    # Quick debug stats
                    if debug_summary:
                        st.write("**Quick Debug Stats:**")
                        total_apps = len(debug_summary)
                        csv_available_count = sum(1 for d in debug_summary if d['CSV Available'])
                        fuzzy_success_count = sum(1 for d in debug_summary if d['Fuzzy Match Success'])
                        using_defaults_count = sum(1 for d in debug_summary if d['Using Defaults'])
                        
                        debug_col1, debug_col2, debug_col3, debug_col4 = st.columns(4)
                        with debug_col1:
                            st.metric("Total Applications", total_apps)
                        with debug_col2:
                            csv_rate = (csv_available_count / total_apps) * 100 if total_apps > 0 else 0
                            st.metric("CSV Available", f"{csv_rate:.1f}%")
                        with debug_col3:
                            fuzzy_rate = (fuzzy_success_count / total_apps) * 100 if total_apps > 0 else 0
                            st.metric("Fuzzy Match Success", f"{fuzzy_rate:.1f}%")
                        with debug_col4:
                            defaults_rate = (using_defaults_count / total_apps) * 100 if total_apps > 0 else 0
                            st.metric("Using Defaults", f"{defaults_rate:.1f}%")
                        
                        # Extraction method breakdown
                        extraction_methods = {}
                        for d in debug_summary:
                            method = d['Extraction Method']
                            extraction_methods[method] = extraction_methods.get(method, 0) + 1
                        
                        st.write("**Company Name Extraction Methods:**")
                        for method, count in extraction_methods.items():
                            percentage = (count / total_apps) * 100
                            st.write(f"- **{method}**: {count} files ({percentage:.1f}%)")
                    
                    st.markdown("---")
                    
                    create_results_dashboard(results_df)
                    
                    # Store results in session state for potential re-use
                    st.session_state['batch_results'] = results_df
                
                else:
                    st.error("No applications were processed successfully")
        
        else:
            st.error("No valid JSON files found in uploaded files")
    
    else:
        render_batch_empty_state()
        

def main():
    """Simplified batch dataset builder."""

    if render_main_hero:
        render_main_hero(
            "MCA v2 Batch Processor",
            "Build paid/not-paid scorecard datasets from application data, transaction JSONs, and optional bureau PDFs.",
            eyebrow="Batch scoring workspace",
        )
    else:
        st.title("MCA v2 Batch Processor")
        st.markdown("Build paid/not-paid scorecard datasets from application data, transaction JSONs, and optional bureau PDFs.")
    render_batch_workflow_rail()

    if sidebar_section:
        sidebar_section("Fallbacks")
    else:
        st.sidebar.header("Fallbacks")
    st.sidebar.caption("Only used where the uploaded data and mapping cannot supply a value.")
    st.sidebar.caption("Matching mode: RapidFuzz" if RAPIDFUZZ_AVAILABLE else "Matching mode: exact fallback")

    default_industry = st.sidebar.selectbox(
        "Fallback Industry",
        list(INDUSTRY_THRESHOLDS.keys()),
        index=list(INDUSTRY_THRESHOLDS.keys()).index("Other"),
    )
    default_loan = st.sidebar.number_input("Fallback Requested Loan (£)", min_value=0.0, value=5000.0, step=1000.0)
    default_directors_score = st.sidebar.slider("Fallback Director Credit Score", 0, 100, 75)
    default_company_age = st.sidebar.number_input("Fallback Company Age (Months)", min_value=0, value=12, step=1)

    default_params = {
        "industry": default_industry,
        "requested_loan": default_loan,
        "directors_score": default_directors_score,
        "company_age_months": default_company_age,
        "business_ccj": False,
        "director_ccj": False,
        "poor_or_no_online_presence": False,
        "uses_generic_email": False,
    }

    if render_intake_panel_intro:
        render_intake_panel_intro(
            title="Batch inputs",
            description=(
                "Upload the application data file, its mapping file, the full JSON pool, "
                "then paid and not-paid JSON subsets for automatic outcome labels."
            ),
        )
    else:
        section_title("Batch Inputs", "Upload the data, mapping, JSON pool, and outcome JSON subsets.")

    data_col, mapping_col = st.columns(2)
    with data_col:
        data_file = st.file_uploader(
            "Application data file",
            type=["csv", "xlsx", "xls"],
            help="Example: data (6).xlsx. Filename does not matter.",
            key="batch_data_file",
        )
    with mapping_col:
        mapping_file = st.file_uploader(
            "CSV mapping file",
            type=["csv", "xlsx", "xls"],
            help="Example: PBI_CSV_Mapping.xlsx.",
            key="batch_mapping_file",
        )

    json_col, paid_col, not_paid_col = st.columns(3)
    with json_col:
        uploaded_files = st.file_uploader(
            "All application JSONs",
            type=["json", "zip"],
            accept_multiple_files=True,
            help="Full pool of transaction JSON files.",
            key="batch_all_jsons",
        )
    with paid_col:
        paid_files = st.file_uploader(
            "Paid JSONs",
            type=["json", "zip"],
            accept_multiple_files=True,
            help="Files uploaded here are labelled paid.",
            key="batch_paid_jsons",
        )
    with not_paid_col:
        not_paid_files = st.file_uploader(
            "Not-paid JSONs",
            type=["json", "zip"],
            accept_multiple_files=True,
            help="Files uploaded here are labelled not_paid/defaulted.",
            key="batch_not_paid_jsons",
        )

    bureau_pdf_files = st.file_uploader(
        "Company credit report PDFs (optional)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Optional bureau PDFs used to derive business CCJ flags.",
        key="batch_bureau_pdfs",
    )

    metadata_df = pd.DataFrame()
    duplicates_df = pd.DataFrame()
    metadata_audit = {}
    parameter_mapping = {}

    if data_file and mapping_file:
        try:
            metadata_df, duplicates_df, metadata_audit = prepare_application_metadata(
                data_file,
                mapping_file,
                default_industry,
            )
            parameter_mapping = build_metadata_mapping(metadata_df)

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Raw Data Rows", metadata_audit.get("raw_rows", 0))
            with m2:
                st.metric("Deduped Applications", len(metadata_df))
            with m3:
                st.metric("Duplicate Rows Removed", metadata_audit.get("duplicate_rows_removed", 0))
            with m4:
                st.metric("Mapped Companies", len(parameter_mapping))

            with st.expander("Mapped Data Preview", expanded=False):
                preview_cols = [
                    c for c in [
                        "application_id",
                        "company_name",
                        "requested_loan",
                        "company_age_months",
                        "directors_score",
                        "director_defaults_12m",
                        "director_ccj_count",
                    ] if c in metadata_df.columns
                ]
                st.dataframe(metadata_df[preview_cols].head(20), use_container_width=True, hide_index=True)

            if not duplicates_df.empty:
                with st.expander(f"Duplicate AppID Rows Found ({len(duplicates_df)})", expanded=False):
                    st.dataframe(duplicates_df.head(100), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Could not read data/mapping files: {e}")

    if not uploaded_files:
        render_batch_empty_state()
        return

    with st.spinner("Loading JSON files..."):
        files_data = load_json_files(uploaded_files)

    if not files_data:
        st.error("No valid JSON files found in uploaded files")
        return

    outcome_mapping, outcome_audit_df = assign_outcomes(files_data, paid_files, not_paid_files)

    pdf_risk_mapping = {}
    pdf_audit_df = pd.DataFrame()
    if bureau_pdf_files and not metadata_df.empty:
        with st.spinner("Reading company credit report PDFs..."):
            pdf_risk_mapping, pdf_audit_df = build_pdf_risk_mapping(bureau_pdf_files, metadata_df)
            parameter_mapping = apply_pdf_risk_to_mapping(parameter_mapping, pdf_risk_mapping)
        render_pdf_match_review(pdf_audit_df)

    loaded_col, csv_col, ready_col = st.columns(3)
    with loaded_col:
        st.metric("JSON Files Loaded", len(files_data))
    with csv_col:
        st.metric("CSV Companies", len(parameter_mapping))
    with ready_col:
        st.metric("Ready to Process", "Yes" if parameter_mapping else "Needs data/mapping")

    if not outcome_audit_df.empty:
        outcome_counts = outcome_audit_df["outcome_label"].value_counts().reset_index()
        outcome_counts.columns = ["Outcome", "Files"]
        st.dataframe(outcome_counts, use_container_width=True, hide_index=True)
        if (outcome_audit_df["outcome_label"] == "conflict").any():
            st.error("Some JSON files are present in both paid and not-paid uploads. Resolve these before using the outputs.")

    with st.expander("Loaded Files", expanded=False):
        for filename, _ in files_data:
            st.write(f"- {filename}")

    if parameter_mapping:
        section_title("File Matching Analysis", "Preview how uploaded JSON filenames align with the mapped company data.")
        uploaded_filenames = [filename.replace(".json", "") for filename, _ in files_data]
        csv_companies = list(parameter_mapping.keys())
        potential_matches = []
        missing_jsons = []
        extra_jsons = []

        if RAPIDFUZZ_AVAILABLE:
            for csv_company in csv_companies:
                match_result = process.extractOne(
                    csv_company,
                    uploaded_filenames,
                    scorer=fuzz.token_sort_ratio,
                    score_cutoff=60,
                )
                if match_result:
                    potential_matches.append(
                        {"csv_company": csv_company, "json_file": match_result[0], "score": match_result[1]}
                    )
                else:
                    missing_jsons.append(csv_company)
            matched_json_files = [m["json_file"] for m in potential_matches]
            extra_jsons = [f for f in uploaded_filenames if f not in matched_json_files]
        else:
            for csv_company in csv_companies:
                if csv_company in uploaded_filenames:
                    potential_matches.append({"csv_company": csv_company, "json_file": csv_company, "score": 100})
                else:
                    missing_jsons.append(csv_company)
            matched_json_files = [m["json_file"] for m in potential_matches]
            extra_jsons = [f for f in uploaded_filenames if f not in matched_json_files]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Companies in Data", len(csv_companies))
        with col2:
            st.metric("JSON Files Uploaded", len(uploaded_filenames))
        with col3:
            st.metric("Expected Matches", len(potential_matches))

        if potential_matches:
            with st.expander(f"Potential Matches ({len(potential_matches)})", expanded=False):
                match_df = pd.DataFrame(potential_matches)
                match_df["score"] = match_df["score"].apply(lambda x: f"{x:.1f}%")
                match_df.columns = ["CSV Company", "JSON File", "Match Score"]
                st.dataframe(match_df, use_container_width=True, hide_index=True)

        if missing_jsons:
            with st.expander(f"Missing JSON Files ({len(missing_jsons)})", expanded=False):
                st.dataframe(
                    pd.DataFrame({"Company Name (from data)": missing_jsons}),
                    use_container_width=True,
                    hide_index=True,
                )

        if extra_jsons:
            with st.expander(f"Extra JSON Files ({len(extra_jsons)})", expanded=False):
                st.dataframe(pd.DataFrame({"JSON File": extra_jsons}), use_container_width=True, hide_index=True)
    else:
        st.info("Upload the data and mapping files to enable metadata matching.")

    if not pdf_audit_df.empty:
        with st.expander("Company Credit Report PDF Audit", expanded=False):
            st.dataframe(pdf_audit_df, use_container_width=True, hide_index=True)

    if not st.button("Process All Applications", type="primary"):
        return

    processor = BatchProcessor()
    progress_bar = st.progress(0, text="Starting batch processing...")
    with st.spinner("Processing applications..."):
        results_df = processor.process_batch(
            files_data,
            default_params,
            parameter_mapping,
            progress_bar,
            outcome_mapping=outcome_mapping,
        )
    progress_bar.empty()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Processed Successfully", processor.processed_count)
    with col2:
        st.metric("Processing Errors", processor.error_count)
    with col3:
        success_rate = (processor.processed_count / len(files_data)) * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")

    if processor.error_log:
        section_title("Processing Errors Analysis")
        error_df = pd.DataFrame(processor.error_log)
        st.dataframe(error_df, use_container_width=True, hide_index=True)
        st.download_button(
            label="Download Error Log (CSV)",
            data=error_df.to_csv(index=False),
            file_name=f"processing_errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

    if results_df.empty:
        st.error("No applications were processed successfully")
        return

    scorecard_features_df = build_scorecard_features(results_df)
    section_title("Scorecard Dataset Exports")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.download_button(
            "Download Case Scores",
            data=results_df.to_csv(index=False),
            file_name=f"case_scores_with_outcomes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            type="primary",
        )
    with d2:
        st.download_button(
            "Download Scorecard Features",
            data=scorecard_features_df.to_csv(index=False),
            file_name=f"scorecard_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    with d3:
        st.download_button(
            "Download Matching Audit",
            data=outcome_audit_df.to_csv(index=False),
            file_name=f"matching_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    with d4:
        if not pdf_audit_df.empty:
            st.download_button(
                "Download PDF Audit",
                data=pdf_audit_df.to_csv(index=False),
                file_name=f"bureau_pdf_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

    if "outcome_label" in results_df.columns:
        section_title("Outcome Summary")
        st.dataframe(results_df["outcome_label"].value_counts().reset_index(), use_container_width=True, hide_index=True)

    calibration_reports = render_scorecard_calibration(results_df)
    section_title("Calibration Report Exports")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.download_button(
            "Download Calibration Summary",
            data=calibration_reports["summary"].to_csv(index=False),
            file_name=f"calibration_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    with c2:
        st.download_button(
            "Download Threshold Recommendations",
            data=calibration_reports["threshold_recommendations"].to_csv(index=False),
            file_name=f"threshold_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    with c3:
        st.download_button(
            "Download Score Band Report",
            data=calibration_reports["score_bands"].to_csv(index=False),
            file_name=f"score_band_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    with c4:
        st.download_button(
            "Download Rule Signal Report",
            data=calibration_reports["rule_signals"].to_csv(index=False),
            file_name=f"rule_signal_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    st.download_button(
        "Download Paid Lookalike Report",
        data=calibration_reports["paid_lookalikes"].to_csv(index=False),
        file_name=f"paid_lookalike_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

    create_results_dashboard(results_df)
    st.session_state["batch_results"] = results_df


def main():
    """Simplified batch dataset builder with submit-only processing and saved runs."""

    if render_main_hero:
        render_main_hero(
            "MCA v2 Batch Processor",
            "Build paid/not-paid scorecard datasets from application data, transaction JSONs, and optional bureau PDFs.",
            eyebrow="Batch scoring workspace",
        )
    else:
        st.title("MCA v2 Batch Processor")
        st.markdown("Build paid/not-paid scorecard datasets from application data, transaction JSONs, and optional bureau PDFs.")
    render_batch_workflow_rail()

    if sidebar_section:
        sidebar_section("Fallbacks")
    else:
        st.sidebar.header("Fallbacks")
    st.sidebar.caption("Only used where the uploaded data and mapping cannot supply a value.")
    st.sidebar.caption("Matching mode: RapidFuzz" if RAPIDFUZZ_AVAILABLE else "Matching mode: exact fallback")

    default_industry = st.sidebar.selectbox(
        "Fallback Industry",
        list(INDUSTRY_THRESHOLDS.keys()),
        index=list(INDUSTRY_THRESHOLDS.keys()).index("Other"),
    )
    default_loan = st.sidebar.number_input("Fallback Requested Loan (£)", min_value=0.0, value=5000.0, step=1000.0)
    default_directors_score = st.sidebar.slider("Fallback Director Credit Score", 0, 100, 75)
    default_company_age = st.sidebar.number_input("Fallback Company Age (Months)", min_value=0, value=12, step=1)

    if sidebar_section:
        sidebar_section("Saved Runs")
    else:
        st.sidebar.header("Saved Runs")
    st.sidebar.caption(f"Folder: {SAVED_RUNS_DIR}")
    if os.getenv("MCAV2_BATCH_SAVED_RUNS_DIR"):
        st.sidebar.caption("Shared saved-runs folder is enabled. Runs saved on another synced machine will appear here.")
    elif SAVED_RUNS_DIR.name == "Saved_batch_processor_runs":
        st.sidebar.caption("Using the shared Savvy OneDrive saved-runs folder. Runs synced from another machine will appear here.")
    else:
        st.sidebar.caption(
            "Using the local saved-runs folder. Set MCAV2_BATCH_SAVED_RUNS_DIR to a OneDrive/shared folder "
            "to see the same runs on every machine."
        )
        with st.sidebar.expander("Shared folder help", expanded=True):
            st.code(
                "MCAV2_BATCH_SAVED_RUNS_DIR="
                r"C:\Users\Massimo Cristi\OneDrive - Savvy Loan Products Ltd"
                r"\Merchant Cash Advance (MCA)\Scorecard\Scorecard Development"
                r"\Saved_batch_processor_runs",
                language="text",
            )
            st.caption("Add this line to the repo .env file on that machine if auto-detect cannot find OneDrive.")

    imported_run_package = st.sidebar.file_uploader(
        "Import saved run package",
        type=["zip"],
        key="saved_run_import_package",
        help="Upload a saved-run zip exported from another machine.",
    )
    if imported_run_package is not None:
        import_key = f"{imported_run_package.name}:{getattr(imported_run_package, 'size', 0)}"
        if st.session_state.get("last_imported_saved_run_package") != import_key:
            try:
                imported_manifest = import_saved_run_package(imported_run_package)
                st.session_state["last_imported_saved_run_package"] = import_key
                st.session_state["loaded_saved_run"] = imported_manifest
                st.sidebar.success("Saved run imported.")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Could not import saved run: {e}")

    saved_runs = list_saved_runs()
    if saved_runs:
        selected_run = st.sidebar.selectbox(
            "Recent saved runs",
            saved_runs,
            format_func=lambda run: f"{run.get('run_name', run.get('run_id', 'Run'))} - {run.get('created_at', '')}",
        )
        if selected_run:
            st.sidebar.caption(selected_run.get("path", ""))
            if st.sidebar.button("Load Selected Run", type="primary"):
                st.session_state["loaded_saved_run"] = selected_run
                st.rerun()
    else:
        selected_run = None
        st.sidebar.info("No saved runs on this machine yet. Import a saved-run package or process and save a new run.")

    if st.session_state.get("loaded_saved_run"):
        if st.button("Close Saved Run View"):
            st.session_state.pop("loaded_saved_run", None)
            st.rerun()
        try:
            render_loaded_saved_run(load_saved_run(st.session_state["loaded_saved_run"]))
        except Exception as e:
            st.error(f"Could not load saved run: {e}")
        return

    default_params = {
        "industry": default_industry,
        "requested_loan": default_loan,
        "directors_score": default_directors_score,
        "company_age_months": default_company_age,
        "business_ccj": False,
        "director_ccj": False,
        "poor_or_no_online_presence": False,
        "uses_generic_email": False,
    }

    if render_intake_panel_intro:
        render_intake_panel_intro(
            title="Batch run setup",
            description=(
                "Name the run, upload the application data and mapping, add the full JSON pool, "
                "then add paid and not-paid JSON subsets. Processing starts only when you press the button."
            ),
        )
    else:
        section_title("Batch Run Setup", "Processing starts only when you press the button.")

    with st.form("batch_run_form", clear_on_submit=False):
        run_name = st.text_input(
            "Run name",
            value=datetime.now().strftime("Scorecard run %Y-%m-%d %H%M"),
            help="Used as the saved run folder name and manifest label.",
        )
        save_run = st.checkbox("Save this run", value=True)

        data_col, mapping_col = st.columns(2)
        with data_col:
            data_file = st.file_uploader(
                "Application data file",
                type=["csv", "xlsx", "xls"],
                help="Example: data (6).xlsx. Filename does not matter.",
                key="batch_data_file_form",
            )
        with mapping_col:
            mapping_file = st.file_uploader(
                "CSV mapping file",
                type=["csv", "xlsx", "xls"],
                help="Example: PBI_CSV_Mapping.xlsx.",
                key="batch_mapping_file_form",
            )

        json_col, paid_col, not_paid_col = st.columns(3)
        with json_col:
            uploaded_files = st.file_uploader(
                "All application JSONs",
                type=["json", "zip"],
                accept_multiple_files=True,
                help="Full pool of transaction JSON files.",
                key="batch_all_jsons_form",
            )
        with paid_col:
            paid_files = st.file_uploader(
                "Paid JSONs",
                type=["json", "zip"],
                accept_multiple_files=True,
                help="Files uploaded here are labelled paid.",
                key="batch_paid_jsons_form",
            )
        with not_paid_col:
            not_paid_files = st.file_uploader(
                "Not-paid JSONs",
                type=["json", "zip"],
                accept_multiple_files=True,
                help="Files uploaded here are labelled not_paid/defaulted.",
                key="batch_not_paid_jsons_form",
            )

        bureau_pdf_files = st.file_uploader(
            "Company credit report PDFs (optional)",
            type=["pdf"],
            accept_multiple_files=True,
            help="Optional bureau PDFs used to derive business CCJ flags.",
            key="batch_bureau_pdfs_form",
        )

        submitted = st.form_submit_button("Process All Applications", type="primary")

    if not submitted:
        if selected_run:
            st.caption("Saved run selected in the sidebar. Its files are available in the saved run folder.")
        render_batch_empty_state()
        return

    if not run_name.strip():
        st.error("Enter a run name before processing.")
        return
    if not data_file or not mapping_file:
        st.error("Upload both the application data file and mapping file.")
        return
    if not uploaded_files:
        st.error("Upload the full JSON pool before processing.")
        return

    metadata_df = pd.DataFrame()
    duplicates_df = pd.DataFrame()
    metadata_audit = {}
    parameter_mapping = {}

    with st.spinner("Reading data and mapping files..."):
        try:
            metadata_df, duplicates_df, metadata_audit = prepare_application_metadata(
                data_file,
                mapping_file,
                default_industry,
            )
            parameter_mapping = build_metadata_mapping(metadata_df)
        except Exception as e:
            st.error(f"Could not read data/mapping files: {e}")
            return

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Raw Data Rows", metadata_audit.get("raw_rows", 0))
    with m2:
        st.metric("Deduped Applications", len(metadata_df))
    with m3:
        st.metric("Duplicate Rows Removed", metadata_audit.get("duplicate_rows_removed", 0))
    with m4:
        st.metric("Mapped Companies", len(parameter_mapping))

    with st.expander("Mapped Data Preview", expanded=False):
        preview_cols = [
            c for c in [
                "application_id",
                "company_name",
                "requested_loan",
                "company_age_months",
                "directors_score",
                "director_defaults_12m",
                "director_ccj_count",
            ] if c in metadata_df.columns
        ]
        st.dataframe(metadata_df[preview_cols].head(20), use_container_width=True, hide_index=True)

    if not duplicates_df.empty:
        with st.expander(f"Duplicate AppID Rows Found ({len(duplicates_df)})", expanded=False):
            st.dataframe(duplicates_df.head(100), use_container_width=True, hide_index=True)

    with st.spinner("Loading JSON files..."):
        files_data = load_json_files(uploaded_files)

    if not files_data:
        st.error("No valid JSON files found in uploaded files")
        return

    outcome_mapping, outcome_audit_df = assign_outcomes(files_data, paid_files, not_paid_files)

    pdf_risk_mapping = {}
    pdf_audit_df = pd.DataFrame()
    if bureau_pdf_files and not metadata_df.empty:
        with st.spinner("Reading company credit report PDFs..."):
            pdf_risk_mapping, pdf_audit_df = build_pdf_risk_mapping(bureau_pdf_files, metadata_df)
            parameter_mapping = apply_pdf_risk_to_mapping(parameter_mapping, pdf_risk_mapping)

    loaded_col, csv_col, ready_col = st.columns(3)
    with loaded_col:
        st.metric("JSON Files Loaded", len(files_data))
    with csv_col:
        st.metric("CSV Companies", len(parameter_mapping))
    with ready_col:
        st.metric("Ready to Process", "Yes" if parameter_mapping else "Needs data/mapping")

    if not outcome_audit_df.empty:
        outcome_counts = outcome_audit_df["outcome_label"].value_counts().reset_index()
        outcome_counts.columns = ["Outcome", "Files"]
        st.dataframe(outcome_counts, use_container_width=True, hide_index=True)
        if (outcome_audit_df["outcome_label"] == "conflict").any():
            st.error("Some JSON files are present in both paid and not-paid uploads. Resolve these before using the outputs.")

    with st.expander("Loaded Files", expanded=False):
        for filename, _ in files_data:
            st.write(f"- {filename}")

    if parameter_mapping:
        section_title("File Matching Analysis", "Preview how uploaded JSON filenames align with the mapped company data.")
        uploaded_filenames = [filename.replace(".json", "") for filename, _ in files_data]
        csv_companies = list(parameter_mapping.keys())
        potential_matches = []
        missing_jsons = []
        extra_jsons = []

        if RAPIDFUZZ_AVAILABLE:
            for csv_company in csv_companies:
                match_result = process.extractOne(
                    csv_company,
                    uploaded_filenames,
                    scorer=fuzz.token_sort_ratio,
                    score_cutoff=60,
                )
                if match_result:
                    potential_matches.append(
                        {"csv_company": csv_company, "json_file": match_result[0], "score": match_result[1]}
                    )
                else:
                    missing_jsons.append(csv_company)
            matched_json_files = [m["json_file"] for m in potential_matches]
            extra_jsons = [f for f in uploaded_filenames if f not in matched_json_files]
        else:
            for csv_company in csv_companies:
                if csv_company in uploaded_filenames:
                    potential_matches.append({"csv_company": csv_company, "json_file": csv_company, "score": 100})
                else:
                    missing_jsons.append(csv_company)
            matched_json_files = [m["json_file"] for m in potential_matches]
            extra_jsons = [f for f in uploaded_filenames if f not in matched_json_files]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Companies in Data", len(csv_companies))
        with col2:
            st.metric("JSON Files Uploaded", len(uploaded_filenames))
        with col3:
            st.metric("Expected Matches", len(potential_matches))

        if potential_matches:
            with st.expander(f"Potential Matches ({len(potential_matches)})", expanded=False):
                match_df = pd.DataFrame(potential_matches)
                match_df["score"] = match_df["score"].apply(lambda x: f"{x:.1f}%")
                match_df.columns = ["CSV Company", "JSON File", "Match Score"]
                st.dataframe(match_df, use_container_width=True, hide_index=True)

        if missing_jsons:
            with st.expander(f"Missing JSON Files ({len(missing_jsons)})", expanded=False):
                st.dataframe(
                    pd.DataFrame({"Company Name (from data)": missing_jsons}),
                    use_container_width=True,
                    hide_index=True,
                )

        if extra_jsons:
            with st.expander(f"Extra JSON Files ({len(extra_jsons)})", expanded=False):
                st.dataframe(pd.DataFrame({"JSON File": extra_jsons}), use_container_width=True, hide_index=True)

    if not pdf_audit_df.empty:
        with st.expander("Company Credit Report PDF Audit", expanded=False):
            st.dataframe(pdf_audit_df, use_container_width=True, hide_index=True)

    processor = BatchProcessor()
    progress_bar = st.progress(0, text="Starting batch processing...")
    with st.spinner("Processing applications..."):
        results_df = processor.process_batch(
            files_data,
            default_params,
            parameter_mapping,
            progress_bar,
            outcome_mapping=outcome_mapping,
        )
    progress_bar.empty()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Processed Successfully", processor.processed_count)
    with col2:
        st.metric("Processing Errors", processor.error_count)
    with col3:
        success_rate = (processor.processed_count / len(files_data)) * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")

    error_df = pd.DataFrame(processor.error_log)
    if processor.error_log:
        section_title("Processing Errors Analysis")
        st.dataframe(error_df, use_container_width=True, hide_index=True)
        st.download_button(
            label="Download Error Log (CSV)",
            data=dataframe_to_csv_bytes(error_df),
            file_name=f"processing_errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

    if results_df.empty:
        st.error("No applications were processed successfully")
        return

    scorecard_features_df = build_scorecard_features(results_df)
    calibration_reports = build_calibration_reports(results_df)

    run_dir = None
    if save_run:
        try:
            run_dir = save_named_run(
                run_name=run_name,
                results_df=results_df,
                scorecard_features_df=scorecard_features_df,
                outcome_audit_df=outcome_audit_df,
                pdf_audit_df=pdf_audit_df,
                calibration_reports=calibration_reports,
                metadata={
                    "default_params": default_params,
                    "metadata_audit": metadata_audit,
                    "processed_successfully": processor.processed_count,
                    "processing_errors": processor.error_count,
                    "json_files_loaded": len(files_data),
                    "mapped_companies": len(parameter_mapping),
                },
                uploads={
                    "data": data_file,
                    "mapping": mapping_file,
                    "all_jsons": uploaded_files,
                    "paid_jsons": paid_files,
                    "not_paid_jsons": not_paid_files,
                    "bureau_pdfs": bureau_pdf_files,
                },
            )
            st.success(f"Saved run: {run_dir}")
            st.download_button(
                "Download Portable Saved Run",
                data=build_saved_run_package(run_dir),
                file_name=f"{slugify_run_name(run_name)}_saved_run.zip",
                mime="application/zip",
                help="Use this zip to load the saved run on another machine without reprocessing.",
            )
        except Exception as e:
            st.warning(f"Run processed, but saving failed: {e}")

    section_title("Scorecard Dataset Exports")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.download_button(
            "Download Case Scores",
            data=dataframe_to_csv_bytes(results_df),
            file_name=f"case_scores_with_outcomes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            type="primary",
        )
    with d2:
        st.download_button(
            "Download Scorecard Features",
            data=dataframe_to_csv_bytes(scorecard_features_df),
            file_name=f"scorecard_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    with d3:
        st.download_button(
            "Download Matching Audit",
            data=dataframe_to_csv_bytes(outcome_audit_df),
            file_name=f"matching_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    with d4:
        if not pdf_audit_df.empty:
            st.download_button(
                "Download PDF Audit",
                data=dataframe_to_csv_bytes(pdf_audit_df),
                file_name=f"bureau_pdf_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

    if "outcome_label" in results_df.columns:
        section_title("Outcome Summary")
        st.dataframe(results_df["outcome_label"].value_counts().reset_index(), use_container_width=True, hide_index=True)

    calibration_reports = render_scorecard_calibration(results_df)
    section_title("Calibration Report Exports")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.download_button(
            "Download Calibration Summary",
            data=dataframe_to_csv_bytes(calibration_reports["summary"]),
            file_name=f"calibration_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    with c2:
        st.download_button(
            "Download Threshold Recommendations",
            data=dataframe_to_csv_bytes(calibration_reports["threshold_recommendations"]),
            file_name=f"threshold_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    with c3:
        st.download_button(
            "Download Score Band Report",
            data=dataframe_to_csv_bytes(calibration_reports["score_bands"]),
            file_name=f"score_band_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    with c4:
        st.download_button(
            "Download Rule Signal Report",
            data=dataframe_to_csv_bytes(calibration_reports["rule_signals"]),
            file_name=f"rule_signal_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    st.download_button(
        "Download Paid Lookalike Report",
        data=dataframe_to_csv_bytes(calibration_reports["paid_lookalikes"]),
        file_name=f"paid_lookalike_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )
    c5, c6 = st.columns(2)
    with c5:
        st.download_button(
            "Download Bureau Signal Report",
            data=dataframe_to_csv_bytes(calibration_reports["bureau_signals"]),
            file_name=f"bureau_signal_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    with c6:
        st.download_button(
            "Download Evidence Quality Report",
            data=dataframe_to_csv_bytes(calibration_reports["evidence_quality"]),
            file_name=f"evidence_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

    create_results_dashboard(results_df)
    st.session_state["batch_results"] = results_df
    if run_dir:
        st.session_state["last_saved_run"] = str(run_dir)


if __name__ == "__main__":
    main()
