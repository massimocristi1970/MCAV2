import sys
import builtins
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]   # app/main.py -> repo_root
SRC_PATH = REPO_ROOT / "src"

# root-level modules (e.g. mca_scorecard_rules.py)
sys.path.insert(0, str(REPO_ROOT))

# src package (e.g. src/tu_scorecard)
sys.path.insert(0, str(SRC_PATH))

import streamlit as st

from app.plotly_theme import LEGEND_BELOW, THRESHOLD_LINE, THRESHOLD_MARKER, show_mca_plotly
from app.ui_theme import (
    apply_ui_theme,
    render_empty_state_main,
    render_intake_panel_intro,
    render_main_hero,
    render_sidebar_help_footer,
    render_workflow_rail,
    sidebar_section,
    sidebar_subsection,
)

import pandas as pd
import joblib
import json
import re
from datetime import datetime, timedelta
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import base64
from io import BytesIO
from dataclasses import asdict
import gc

from app.workflows.application_analysis import AnalysisCallbacks, analyse_open_banking_application
from app.config.industry_config import (
    DIRECTOR_SCORE_PASS_THRESHOLD,
    INDUSTRY_THRESHOLDS as CANONICAL_INDUSTRY_THRESHOLDS,
    get_industry_thresholds,
    get_sector_risk,
)

# --- TU XML Scorecard (Director) ---
from src.tu_scorecard.feature_extractor import extract_features_from_xml_bytes
from src.tu_scorecard.scorecard_rules import score_tu_features



# IMPORTANT:
# Do NOT import from app.pages inside app/main.py.
# app/pages is Streamlit's multipage folder; importing it here can execute page scripts
# at import-time and override the Main page UI.
MODULAR_IMPORTS_AVAILABLE = False

modular_weighted_scores = None
modular_load_models = None
modular_subprime_score = None
modular_ml_adjustment = None

modular_score_charts = None
modular_financial_charts = None
modular_loans_charts = None

_map_transaction_category = None  # force fallback map_transaction_category() below
modular_categorize = None
modular_filter_period = None
modular_calc_metrics = None
modular_revenue_insights = None
modular_create_csv = None
modular_analyze_loans = None

ModularDashboardExporter = None

# Import ensemble scorer for unified recommendations
try:
    from app.services.ensemble_scorer import get_ensemble_recommendation, Decision
    ENSEMBLE_SCORER_AVAILABLE = True
except ImportError as e:
    ENSEMBLE_SCORER_AVAILABLE = False
    get_ensemble_recommendation = None
    Decision = None
    print(f"Note: Ensemble scorer not available ({e}).")

try:
    from app.services.business_risk_signals import (
        calculate_business_metrics as modular_business_metrics,
        categorize_business_transactions as modular_business_categorize,
    )
    BUSINESS_SIGNAL_SERVICES_AVAILABLE = True
except ImportError as e:
    BUSINESS_SIGNAL_SERVICES_AVAILABLE = False
    modular_business_metrics = None
    modular_business_categorize = None
    print(f"Note: business risk signal services not available ({e}).")

try:
    from app.services.card_terminal_ingestion import CardTerminalIngestionService
    CARD_TERMINAL_SERVICE_AVAILABLE = True
except ImportError as e:
    CARD_TERMINAL_SERVICE_AVAILABLE = False
    CardTerminalIngestionService = None
    print(f"Note: card terminal ingestion service not available ({e}).")

from app.services.business_bureau_pdf import parse_business_bureau_pdf

STREAMLIT_CACHE_TTL_SECONDS = int(os.getenv("STREAMLIT_CACHE_TTL_SECONDS", "1800"))
STREAMLIT_CACHE_MAX_ENTRIES = int(os.getenv("STREAMLIT_CACHE_MAX_ENTRIES", "8"))


@st.cache_data(
    show_spinner=False,
    ttl=STREAMLIT_CACHE_TTL_SECONDS,
    max_entries=STREAMLIT_CACHE_MAX_ENTRIES,
)
def _score_tu_xml_bytes(xml_bytes: bytes, app_id: str) -> dict:
    feats_obj = extract_features_from_xml_bytes(xml_bytes=xml_bytes, app_id=app_id)
    sr = score_tu_features(feats_obj.features)
    return {
        "score": sr.score,
        "decision": sr.decision,
        "reasons": sr.reasons,
        "flags": sr.recovery_flags,
    }

# Debug mode - only enabled when DEBUG environment variable is set to 'true'
DEBUG_MODE = os.environ.get('DEBUG', 'false').lower() == 'true'


@st.cache_resource(show_spinner=False)
def _load_ml_insights_scaler():
    return joblib.load('scaler.pkl')


def debug_file_structure():
    """Debug helper to understand the file structure - only runs in DEBUG mode"""
    if not DEBUG_MODE:
        return
        
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"🔍 DEBUG - File Structure:")
    print(f"  Current file location: {__file__}")
    print(f"  Current directory: {current_dir}")
    
    # Check for services directory
    services_dir = os.path.join(current_dir, 'services')
    print(f"  Services directory exists: {os.path.exists(services_dir)}")
    
    if os.path.exists(services_dir):
        services_files = os.listdir(services_dir)
        print(f"  Files in services/: {services_files}")
        
        subprime_file = os.path.join(services_dir, 'subprime_scoring_system.py')
        print(f"  subprime_scoring_system.py exists: {os.path.exists(subprime_file)}")
    
    # Check parent directory structure  
    parent_dir = os.path.dirname(current_dir)
    print(f"  Parent directory: {parent_dir}")
    
    app_services_dir = os.path.join(parent_dir, 'app', 'services')
    print(f"  app/services/ directory exists: {os.path.exists(app_services_dir)}")
    
    if os.path.exists(app_services_dir):
        app_services_files = os.listdir(app_services_dir)
        print(f"  Files in app/services/: {app_services_files}")

# Only run debug output if DEBUG mode is enabled
debug_file_structure()
    
class MLScalerInsights:
    """ML validation specifically calibrated for your training data patterns"""
    
    def __init__(self):
        try:
            self.scaler = _load_ml_insights_scaler()
            self.has_scaler = True
            self.training_means = self.scaler.mean_
            self.training_stds = self.scaler.scale_
            
            # Your actual training data statistics
            self.known_stats = {
                'Directors Score': {'mean': 71.116, 'std': 22.948},
                'Total Revenue': {'mean': 310756.109, 'std': 2052232.580},
                'Total Debt': {'mean': 23646.318, 'std': 136326.828},
                'Debt-to-Income Ratio': {'mean': 0.460, 'std': 2.801},
                'Operating Margin': {'mean': 0.185, 'std': 1.288},
                'Debt Service Coverage Ratio': {'mean': 4691.638, 'std': 66589.036},
                'Cash Flow Volatility': {'mean': 1.189, 'std': 27.430},
                'Revenue Growth Rate': {'mean': 30.395, 'std': 233.195},
                'Average Month-End Balance': {'mean': 5406.797, 'std': 62023.093},
                'Average Negative Balance Days per Month': {'mean': 6.993, 'std': 6.482},
                'Number of Bounced Payments': {'mean': 3.455, 'std': 11.778},
                'Company Age (Months)': {'mean': 8.099, 'std': 5.206},
                'Sector_Risk': {'mean': 0.566, 'std': 0.496}
            }
            
            if DEBUG_MODE:
                print("ML validation available (calibrated for your training data)")
            
        except Exception as e:
            self.has_scaler = False
            if DEBUG_MODE:
                print(f"ML validation not available: {e}")
    
    def validate_business_data(self, metrics, params):
        """Validate business data with awareness of your training data's extreme variability"""
        
        if not self.has_scaler:
            return {'available': False}
        
        try:
            # Prepare features exactly like your ML model
            features = {
                'Directors Score': params.get('directors_score', 0),
                'Total Revenue': metrics.get("Total Revenue", 0),
                'Total Debt': metrics.get("Total Debt", 0),
                'Debt-to-Income Ratio': metrics.get("Debt-to-Income Ratio", 0),
                'Operating Margin': metrics.get("Operating Margin", 0),
                'Debt Service Coverage Ratio': metrics.get("Debt Service Coverage Ratio", 0),
                'Cash Flow Volatility': metrics.get("Cash Flow Volatility", 0),
                'Revenue Growth Rate': metrics.get("Revenue Growth Rate", 0),
                'Average Month-End Balance': metrics.get("Average Month-End Balance", 0),
                'Average Negative Balance Days per Month': metrics.get("Average Negative Balance Days per Month", 0),
                'Number of Bounced Payments': metrics.get("Number of Bounced Payments", 0),
                'Company Age (Months)': params.get('company_age_months', 0),
                'Sector_Risk': get_sector_risk(params.get('industry', 'Other'))
            }
            
            # Calculate z-scores and identify issues
            z_scores = []
            outliers = []
            
            for feature_name, value in features.items():
                if feature_name in self.known_stats:
                    mean = self.known_stats[feature_name]['mean']
                    std = self.known_stats[feature_name]['std']
                    
                    z_score = (value - mean) / std if std > 0 else 0
                    z_scores.append(abs(z_score))
                    
                    # Flag outliers with business context
                    if abs(z_score) > 2:
                        severity = 'High' if abs(z_score) > 3 else 'Moderate'
                        
                        # Special handling for your data's known issues
                        business_interpretation = self._interpret_outlier(feature_name, value, z_score, mean)
                        
                        outliers.append({
                            'feature': feature_name,
                            'value': value,
                            'z_score': z_score,
                            'training_mean': mean,
                            'severity': severity,
                            'interpretation': business_interpretation
                        })
            
            # Overall assessment with your data's context
            avg_z = np.mean(z_scores) if z_scores else 0
            
            # Adjusted quality scoring for your highly variable training data
            data_quality = max(0, 100 - (avg_z * 10))
            
            # Confidence assessment tailored to your data
            confidence, desc = self._assess_confidence_for_your_data(avg_z, outliers, features)
            
            return {
                'available': True,
                'data_quality_score': round(data_quality, 1),
                'ml_confidence': confidence,
                'ml_confidence_desc': desc,
                'outlier_count': len(outliers),
                'outliers': outliers,
                'avg_z_score': round(avg_z, 2),
                'recommendations': self._generate_tailored_recommendations(outliers, avg_z, features)
            }
            
        except Exception as e:
            return {'available': False, 'error': str(e)}
    
    def _interpret_outlier(self, feature_name, value, z_score, training_mean):
        """Provide business interpretation of outliers given your training data issues"""
        
        interpretations = {
            'Debt Service Coverage Ratio': {
                'high': f"DSCR of {value:.2f} is reasonable (training mean of {training_mean:.0f} was likely erroneous)",
                'low': f"DSCR of {value:.2f} is low but more realistic than extreme training data"
            },
            'Revenue Growth Rate': {
                'high': f"Growth of {value*100:.1f}% is high but reasonable (training averaged {training_mean:.0f}% - likely incorrect)",
                'low': f"Growth of {value*100:.1f}% is more typical than extreme training data"
            },
            'Cash Flow Volatility': {
                'high': f"Volatility of {value:.3f} is high but training data was extremely variable",
                'low': f"Volatility of {value:.3f} is more stable than most training businesses"
            },
            'Total Revenue': {
                'high': f"Revenue of £{value:,.0f} is substantial but reasonable",
                'low': f"Revenue of £{value:,.0f} is lower than training average but viable"
            },
            'Directors Score': {
                'high': f"Directors score of {value} is excellent",
                'low': f"Directors score of {value} needs attention"
            }
        }
        
        direction = 'high' if z_score > 0 else 'low'
        return interpretations.get(feature_name, {}).get(direction, f"Value differs from training pattern")
    
    def _assess_confidence_for_your_data(self, avg_z, outliers, features):
        """Assess confidence with awareness of your training data's quality issues"""
        
        # Check for specific red flags vs. training data artifacts
        business_red_flags = 0
        training_artifacts = 0
        
        for outlier in outliers:
            feature = outlier['feature']
            z_score = outlier['z_score']
            
            # Features where being an "outlier" might actually be GOOD
            if feature in ['Debt Service Coverage Ratio', 'Cash Flow Volatility', 'Revenue Growth Rate']:
                if abs(z_score) > 3:
                    training_artifacts += 1
                else:
                    business_red_flags += 0.5
            else:
                business_red_flags += 1
        
        # Confidence assessment factoring in training data quality
        if avg_z <= 1.5 or training_artifacts > business_red_flags:
            confidence = "High"
            desc = "Business profile reasonable despite differences from chaotic training data"
        elif avg_z <= 2.5 and business_red_flags <= 2:
            confidence = "Moderate"
            desc = "Some deviation but likely more normal than much of training data"
        elif avg_z <= 4.0:
            confidence = "Low"
            desc = "Significant differences - verify data accuracy"
        else:
            confidence = "Very Low"
            desc = "Extreme differences - manual review required"
        
        return confidence, desc
    
    def _generate_tailored_recommendations(self, outliers, avg_z, features):
        """Generate recommendations specific to your training data context"""
        recommendations = []
        
        # Check for reasonable vs unreasonable outliers
        concerning_outliers = []
        good_outliers = []
        
        for outlier in outliers:
            feature = outlier['feature']
            value = outlier['value']
            
            # Features where being different from training data is often GOOD
            if feature == 'Debt Service Coverage Ratio':
                if 1.0 <= value <= 10.0:
                    good_outliers.append(f"{feature} ({value:.2f}) is more realistic")
                else:
                    concerning_outliers.append(f"{feature} ({value:.2f}) - verify calculation")
            
            elif feature == 'Revenue Growth Rate':
                if -0.5 <= value <= 1.0:
                    good_outliers.append(f"{feature} ({value*100:.1f}%) is more reasonable")
                else:
                    concerning_outliers.append(f"{feature} ({value*100:.1f}%) - verify calculation")
            
            elif feature == 'Cash Flow Volatility':
                if value <= 2.0:
                    good_outliers.append(f"{feature} ({value:.3f}) is more stable")
                else:
                    concerning_outliers.append(f"{feature} ({value:.3f}) - high volatility")
            
            else:
                concerning_outliers.append(f"{feature} - review value")
        
        # Generate appropriate recommendations
        if good_outliers:
            recommendations.append(
                "Some differences indicate healthier metrics than training data"
            )

        if concerning_outliers and len(concerning_outliers) <= 2:
            recommendations.append("A few metrics need verification")
        elif len(concerning_outliers) > 2:
            recommendations.append("Multiple metrics need review")

        if avg_z <= 2.0:
            recommendations.append("Business profile more stable than much of training data")
        elif avg_z > 4.0:
            recommendations.append("Business very different — prioritize rule-based scores")

        # ML usage guidance
        if avg_z <= 2.0 and len(concerning_outliers) <= 1:
            recommendations.append("ML score likely reliable despite training data issues")
        else:
            recommendations.append("Prioritize subprime and weighted scores over ML score")
        
        return recommendations

       

# Improved import with multiple fallback strategies
def import_subprime_scoring():
    """Import SubprimeScoring with comprehensive fallback strategies"""
    
    # Strategy 1: Direct import from services (if main.py is in app/)
    try:
        from services.subprime_scoring_system import SubprimeScoring
        print("OK Imported SubprimeScoring from services.subprime_scoring_system")
        return SubprimeScoring, True
    except ImportError as e:
        print(f"WARN Failed import from services: {e}")
    
    # Strategy 2: Import from app.services (if main.py is in root/)
    try:
        from app.services.subprime_scoring_system import SubprimeScoring
        print("OK Imported SubprimeScoring from app.services.subprime_scoring_system")
        return SubprimeScoring, True
    except ImportError as e:
        print(f"WARN Failed import from app.services: {e}")
    
    # Strategy 3: Add path and import
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        services_dir = os.path.join(current_dir, 'services')
        
        if os.path.exists(services_dir) and services_dir not in sys.path:
            sys.path.insert(0, services_dir)
            print(f"📁 Added to path: {services_dir}")
        
        from subprime_scoring_system import SubprimeScoring
        print("OK Imported SubprimeScoring after adding services to path")
        return SubprimeScoring, True
    except ImportError as e:
        print(f"WARN Failed import after path addition: {e}")
    
    # Strategy 4: Check parent app/services directory
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        app_services_dir = os.path.join(parent_dir, 'app', 'services')
        
        if os.path.exists(app_services_dir) and app_services_dir not in sys.path:
            sys.path.insert(0, app_services_dir)
            print(f"📁 Added to path: {app_services_dir}")
        
        from subprime_scoring_system import SubprimeScoring
        print("OK Imported SubprimeScoring from parent app/services")
        return SubprimeScoring, True
    except ImportError as e:
        print(f"WARN Failed import from parent app/services: {e}")
    
    # Strategy 5: Absolute import with file loading
    try:
        import importlib.util
        
        # Find the file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(current_dir, 'services', 'subprime_scoring_system.py'),
            os.path.join(current_dir, 'subprime_scoring_system.py'),
            os.path.join(os.path.dirname(current_dir), 'app', 'services', 'subprime_scoring_system.py')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"📄 Found subprime_scoring_system.py at: {path}")
                spec = importlib.util.spec_from_file_location("subprime_scoring_system", path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                SubprimeScoring = module.SubprimeScoring
                print("OK Imported SubprimeScoring using importlib")
                return SubprimeScoring, True
        
        print("WARN subprime_scoring_system.py not found in any expected location")
        
    except Exception as e:
        print(f"WARN Failed absolute import: {e}")
    
    # If all strategies fail, return None
    print("🚨 All import strategies failed!")
    return None, False

# Use the import function
SubprimeScoring, SUBPRIME_SCORING_AVAILABLE = import_subprime_scoring()

# Create fallback class if import failed
if not SUBPRIME_SCORING_AVAILABLE:
    print("WARN Creating fallback SubprimeScoring class")
    class SubprimeScoring:
        def calculate_subprime_score(self, metrics, params):
            return {
                'subprime_score': 0,
                'risk_tier': 'Import Failed',
                'pricing_guidance': {'suggested_rate': 'N/A'},
                'recommendation': 'Subprime scoring import failed - check file structure',
                'breakdown': ['Import failed - check console for details']
            }

st.set_page_config(
    page_title="Business Finance Scorecard",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": None,
    },
)

# Complete Industry thresholds with all sectors
INDUSTRY_THRESHOLDS = dict(sorted({
    'Medical Practices (GPs, Clinics, Dentists)': {
        'Debt Service Coverage Ratio': 1.60, 'Net Income': 1000, 'Operating Margin': 0.10,
        'Revenue Growth Rate': 0.06, 'Cash Flow Volatility': 0.08, 'Gross Burn Rate': 16000,
        'Directors Score': 68, 'Sector Risk': 0, 'Average Month-End Balance': 900,
        'Average Negative Balance Days per Month': 0, 'Number of Bounced Payments': 0,
    },
    'Pharmacies (Independent or Small Chains)': {
        'Debt Service Coverage Ratio': 1.60, 'Net Income': 500, 'Operating Margin': 0.08,
        'Revenue Growth Rate': 0.05, 'Cash Flow Volatility': 0.08, 'Gross Burn Rate': 15000,
        'Directors Score': 68, 'Sector Risk': 0, 'Average Month-End Balance': 750,
        'Average Negative Balance Days per Month': 0, 'Number of Bounced Payments': 0,
    },
    'Business Consultants': {
        'Debt Service Coverage Ratio': 1.60, 'Net Income': 1000, 'Operating Margin': 0.09,
        'Revenue Growth Rate': 0.05, 'Cash Flow Volatility': 0.08, 'Gross Burn Rate': 14000,
        'Directors Score': 68, 'Sector Risk': 0, 'Average Month-End Balance': 750,
        'Average Negative Balance Days per Month': 0, 'Number of Bounced Payments': 0,
    },
    'IT Services and Support Companies': {
        'Debt Service Coverage Ratio': 1.55, 'Net Income': 500, 'Operating Margin': 0.12,
        'Revenue Growth Rate': 0.07, 'Cash Flow Volatility': 0.10, 'Gross Burn Rate': 13000,
        'Directors Score': 68, 'Sector Risk': 0, 'Average Month-End Balance': 700,
        'Average Negative Balance Days per Month': 0, 'Number of Bounced Payments': 0,
    },
    'Courier Services (Independent and Regional Operators)': {
        'Debt Service Coverage Ratio': 1.50, 'Net Income': 500, 'Operating Margin': 0.08,
        'Revenue Growth Rate': 0.06, 'Cash Flow Volatility': 0.10, 'Gross Burn Rate': 12000,
        'Directors Score': 68, 'Sector Risk': 0, 'Average Month-End Balance': 600,
        'Average Negative Balance Days per Month': 0, 'Number of Bounced Payments': 0,
    },
    'Grocery Stores and Mini-Markets': {
        'Debt Service Coverage Ratio': 1.50, 'Net Income': 500, 'Operating Margin': 0.07,
        'Revenue Growth Rate': 0.06, 'Cash Flow Volatility': 0.12, 'Gross Burn Rate': 10000,
        'Directors Score': 68, 'Sector Risk': 0, 'Average Month-End Balance': 600,
        'Average Negative Balance Days per Month': 1, 'Number of Bounced Payments': 0,
    },
    'Education': {
        'Debt Service Coverage Ratio': 1.45, 'Net Income': 1500, 'Operating Margin': 0.10,
        'Revenue Growth Rate': 0.05, 'Cash Flow Volatility': 0.09, 'Gross Burn Rate': 11500,
        'Directors Score': 68, 'Sector Risk': 0, 'Average Month-End Balance': 700,
        'Average Negative Balance Days per Month': 1, 'Number of Bounced Payments': 0,
    },
    'Engineering': {
        'Debt Service Coverage Ratio': 1.55, 'Net Income': 7000, 'Operating Margin': 0.07,
        'Revenue Growth Rate': 0.04, 'Cash Flow Volatility': 0.10, 'Gross Burn Rate': 13000,
        'Directors Score': 68, 'Sector Risk': 0, 'Average Month-End Balance': 650,
        'Average Negative Balance Days per Month': 1, 'Number of Bounced Payments': 0,
    },
    'Estate Agent': {
        'Debt Service Coverage Ratio': 1.50, 'Net Income': 4500, 'Operating Margin': 0.09,
        'Revenue Growth Rate': 0.04, 'Cash Flow Volatility': 0.10, 'Gross Burn Rate': 13000,
        'Directors Score': 68, 'Sector Risk': 0, 'Average Month-End Balance': 650,
        'Average Negative Balance Days per Month': 0, 'Number of Bounced Payments': 0,
    },
    'Food Service': {
        'Debt Service Coverage Ratio': 1.55, 'Net Income': 2500, 'Operating Margin': 0.06,
        'Revenue Growth Rate': 0.06, 'Cash Flow Volatility': 0.12, 'Gross Burn Rate': 11000,
        'Directors Score': 68, 'Sector Risk': 0, 'Average Month-End Balance': 700,
        'Average Negative Balance Days per Month': 1, 'Number of Bounced Payments': 0,
    },
    'Import / Export': {
        'Debt Service Coverage Ratio': 1.55, 'Net Income': 3000, 'Operating Margin': 0.07,
        'Revenue Growth Rate': 0.06, 'Cash Flow Volatility': 0.15, 'Gross Burn Rate': 11000,
        'Directors Score': 68, 'Sector Risk': 0, 'Average Month-End Balance': 700,
        'Average Negative Balance Days per Month': 1, 'Number of Bounced Payments': 0,
    },
    'Manufacturing': {
        'Debt Service Coverage Ratio': 1.60, 'Net Income': 1500, 'Operating Margin': 0.08,
        'Revenue Growth Rate': 0.05, 'Cash Flow Volatility': 0.11, 'Gross Burn Rate': 13500,
        'Directors Score': 68, 'Sector Risk': 0, 'Average Month-End Balance': 750,
        'Average Negative Balance Days per Month': 1, 'Number of Bounced Payments': 0,
    },
    'Marketing / Advertising / Design': {
        'Debt Service Coverage Ratio': 1.55, 'Net Income': 5000, 'Operating Margin': 0.11,
        'Revenue Growth Rate': 0.06, 'Cash Flow Volatility': 0.11, 'Gross Burn Rate': 13500,
        'Directors Score': 68, 'Sector Risk': 0, 'Average Month-End Balance': 750,
        'Average Negative Balance Days per Month': 1, 'Number of Bounced Payments': 0,
    },
    'Off-Licence Business': {
        'Debt Service Coverage Ratio': 1.55, 'Net Income': 4500, 'Operating Margin': 0.08,
        'Revenue Growth Rate': 0.06, 'Cash Flow Volatility': 0.11, 'Gross Burn Rate': 14000,
        'Directors Score': 68, 'Sector Risk': 0, 'Average Month-End Balance': 650,
        'Average Negative Balance Days per Month': 1, 'Number of Bounced Payments': 0,
    },
    'Telecommunications': {
        'Debt Service Coverage Ratio': 1.55, 'Net Income': 5000, 'Operating Margin': 0.11,
        'Revenue Growth Rate': 0.05, 'Cash Flow Volatility': 0.12, 'Gross Burn Rate': 13000,
        'Directors Score': 68, 'Sector Risk': 0, 'Average Month-End Balance': 700,
        'Average Negative Balance Days per Month': 1, 'Number of Bounced Payments': 0,
    },
    'Tradesman': {
        'Debt Service Coverage Ratio': 1.40, 'Net Income': 4000, 'Operating Margin': 0.08,
        'Revenue Growth Rate': 0.05, 'Cash Flow Volatility': 0.15, 'Gross Burn Rate': 11500,
        'Directors Score': 68, 'Sector Risk': 0, 'Average Month-End Balance': 700,
        'Average Negative Balance Days per Month': 2, 'Number of Bounced Payments': 0,
    },
    'Wholesaler / Distributor': {
        'Debt Service Coverage Ratio': 1.50, 'Net Income': 3500, 'Operating Margin': 0.10,
        'Revenue Growth Rate': 0.06, 'Cash Flow Volatility': 0.13, 'Gross Burn Rate': 13000,
        'Directors Score': 68, 'Sector Risk': 0, 'Average Month-End Balance': 700,
        'Average Negative Balance Days per Month': 1, 'Number of Bounced Payments': 0,
    },
    'Other': {
        'Debt Service Coverage Ratio': 1.50, 'Net Income': 3000, 'Operating Margin': 0.08,
        'Revenue Growth Rate': 0.05, 'Cash Flow Volatility': 0.13, 'Gross Burn Rate': 11000,
        'Directors Score': 68, 'Sector Risk': 1, 'Average Month-End Balance': 700,
        'Average Negative Balance Days per Month': 1, 'Number of Bounced Payments': 0,
    },
    'Personal Services': {
        'Debt Service Coverage Ratio': 1.50, 'Net Income': 2000, 'Operating Margin': 0.09,
        'Revenue Growth Rate': 0.05, 'Cash Flow Volatility': 0.12, 'Gross Burn Rate': 12000,
        'Directors Score': 68, 'Sector Risk': 1, 'Average Month-End Balance': 700,
        'Average Negative Balance Days per Month': 1, 'Number of Bounced Payments': 0,
    },
    'Restaurants and Cafes': {
        'Debt Service Coverage Ratio': 1.30, 'Net Income': 0, 'Operating Margin': 0.05,
        'Revenue Growth Rate': 0.04, 'Cash Flow Volatility': 0.16, 'Gross Burn Rate': 11000,
        'Directors Score': 68, 'Sector Risk': 1, 'Average Month-End Balance': 600,
        'Average Negative Balance Days per Month': 2, 'Number of Bounced Payments': 0,
    },
    'Bars and Pubs': {
        'Debt Service Coverage Ratio': 1.25, 'Net Income': 0, 'Operating Margin': 0.04,
        'Revenue Growth Rate': 0.03, 'Cash Flow Volatility': 0.18, 'Gross Burn Rate': 10000,
        'Directors Score': 68, 'Sector Risk': 1, 'Average Month-End Balance': 600,
        'Average Negative Balance Days per Month': 2, 'Number of Bounced Payments': 0,
    },
    'Beauty Salons and Spas': {
        'Debt Service Coverage Ratio': 1.30, 'Net Income': 500, 'Operating Margin': 0.06,
        'Revenue Growth Rate': 0.04, 'Cash Flow Volatility': 0.14, 'Gross Burn Rate': 9500,
        'Directors Score': 68, 'Sector Risk': 1, 'Average Month-End Balance': 550,
        'Average Negative Balance Days per Month': 2, 'Number of Bounced Payments': 0,
    },
    'E-Commerce Retailers': {
        'Debt Service Coverage Ratio': 1.35, 'Net Income': 1000, 'Operating Margin': 0.07,
        'Revenue Growth Rate': 0.05, 'Cash Flow Volatility': 0.14, 'Gross Burn Rate': 10000,
        'Directors Score': 68, 'Sector Risk': 1, 'Average Month-End Balance': 600,
        'Average Negative Balance Days per Month': 1, 'Number of Bounced Payments': 0,
    },
    'Event Planning and Management Firms': {
        'Debt Service Coverage Ratio': 1.30, 'Net Income': 500, 'Operating Margin': 0.05,
        'Revenue Growth Rate': 0.03, 'Cash Flow Volatility': 0.16, 'Gross Burn Rate': 10000,
        'Directors Score': 68, 'Sector Risk': 1, 'Average Month-End Balance': 600,
        'Average Negative Balance Days per Month': 1, 'Number of Bounced Payments': 0,
    },
    'Auto Repair Shops': {
        'Debt Service Coverage Ratio': 1.40, 'Net Income': 1000, 'Operating Margin': 0.08,
        'Revenue Growth Rate': 0.05, 'Cash Flow Volatility': 0.12, 'Gross Burn Rate': 9500,
        'Directors Score': 68, 'Sector Risk': 1, 'Average Month-End Balance': 650,
        'Average Negative Balance Days per Month': 1, 'Number of Bounced Payments': 0,
    },
    'Fitness Centres and Gyms': {
        'Debt Service Coverage Ratio': 1.35, 'Net Income': 500, 'Operating Margin': 0.06,
        'Revenue Growth Rate': 0.04, 'Cash Flow Volatility': 0.18, 'Gross Burn Rate': 10000,
        'Directors Score': 68, 'Sector Risk': 1, 'Average Month-End Balance': 600,
        'Average Negative Balance Days per Month': 1, 'Number of Bounced Payments': 0,
    },
    'Construction Firms': {
        'Debt Service Coverage Ratio': 1.30, 'Net Income': 1000, 'Operating Margin': 0.08,
        'Revenue Growth Rate': 0.04, 'Cash Flow Volatility': 0.16, 'Gross Burn Rate': 12500,
        'Directors Score': 68, 'Sector Risk': 1, 'Average Month-End Balance': 700,
        'Average Negative Balance Days per Month': 2, 'Number of Bounced Payments': 0,
    },
    'Printing / Publishing': {
        'Debt Service Coverage Ratio': 1.50, 'Net Income': 2500, 'Operating Margin': 0.08,
        'Revenue Growth Rate': 0.05, 'Cash Flow Volatility': 0.14, 'Gross Burn Rate': 11000,
        'Directors Score': 68, 'Sector Risk': 1, 'Average Month-End Balance': 650,
        'Average Negative Balance Days per Month': 1, 'Number of Bounced Payments': 0,
    },
    'Recruitment': {
        'Debt Service Coverage Ratio': 1.50, 'Net Income': 2000, 'Operating Margin': 0.09,
        'Revenue Growth Rate': 0.05, 'Cash Flow Volatility': 0.10, 'Gross Burn Rate': 13000,
        'Directors Score': 68, 'Sector Risk': 1, 'Average Month-End Balance': 600,
        'Average Negative Balance Days per Month': 1, 'Number of Bounced Payments': 0,
    },
    'Retail': {
        'Debt Service Coverage Ratio': 1.40, 'Net Income': 2500, 'Operating Margin': 0.09,
        'Revenue Growth Rate': 0.04, 'Cash Flow Volatility': 0.14, 'Gross Burn Rate': 11500,
        'Directors Score': 68, 'Sector Risk': 1, 'Average Month-End Balance': 620,
        'Average Negative Balance Days per Month': 1, 'Number of Bounced Payments': 0,
    },
}.items()))

INDUSTRY_THRESHOLDS = CANONICAL_INDUSTRY_THRESHOLDS

# Scoring weights
WEIGHTS = {
    'Debt Service Coverage Ratio': 19, 'Net Income': 13, 'Operating Margin': 9,
    'Revenue Growth Rate': 5, 'Cash Flow Volatility': 12, 'Gross Burn Rate': 3,
    'Company Age (Months)': 4, 'Directors Score': 18, 'Sector Risk': 3,
    'Average Month-End Balance': 5, 'Average Negative Balance Days per Month': 6,
    'Number of Bounced Payments': 3,
}

PENALTIES = {
    "business_ccj": 5
}

def calculate_weighted_scores(metrics, params, industry_thresholds):
    """Calculate weighted score only"""
    
    # weighted score (keep existing logic)
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

@st.cache_resource(show_spinner=False)
def load_models():
    """Load ML models"""
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        return None, None

def map_transaction_category(transaction):
    """
    Enhanced transaction categorization matching original version.
    
    Delegates to the canonical implementation in app.pages.transactions
    when available, otherwise falls back to basic categorization.
    """
    # Use modular implementation when available (canonical source)
    if MODULAR_IMPORTS_AVAILABLE and _map_transaction_category is not None:
        return _map_transaction_category(transaction)
    
    # Fallback: basic categorization if modular imports unavailable
    # This should rarely be used in production
    amount = transaction.get("amount", 0)
    is_credit = amount < 0
    is_debit = amount > 0
    
    if is_credit:
        return "Income"
    elif is_debit:
        return "Expenses"
    else:
        return "Uncategorised"


def categorize_transactions(data):
    """
    Apply categorization to transaction DataFrame.
    
    Delegates to the canonical implementation in app.pages.transactions
    when available.
    """
    # Use modular implementation when available (canonical source)
    if MODULAR_IMPORTS_AVAILABLE and modular_categorize is not None:
        return modular_categorize(data)

    if BUSINESS_SIGNAL_SERVICES_AVAILABLE and modular_business_categorize is not None:
        return modular_business_categorize(data)
    
    # Fallback implementation
    if data.empty:
        return data
        
    data = data.copy()
    data['subcategory'] = data.apply(map_transaction_category, axis=1)
    data['is_revenue'] = data['subcategory'].isin(['Income'])
    data['is_expense'] = data['subcategory'].isin(['Expenses', 'Special Outflow', 'Bank Charge'])
    data['is_debt_repayment'] = data['subcategory'].isin(['Debt Repayments'])
    data['is_debt'] = data['subcategory'].isin(['Loans'])
    data['is_failed_payment'] = data['subcategory'].isin(['Failed Payment'])
    data['is_transfer_in'] = data['subcategory'].isin(['Transfer In'])
    data['is_transfer_out'] = data['subcategory'].isin(['Transfer Out'])
    data['is_internal_transfer'] = data['is_transfer_in'] | data['is_transfer_out']
    data['is_funding_injection'] = data['subcategory'].isin(['Funding Inflow'])
    data['is_bank_charge'] = data['subcategory'].isin(['Bank Charge'])
    data['is_special_inflow'] = data['subcategory'].isin(['Special Inflow'])
    
    return data

def filter_data_by_period(df, period_months):
    """Filter data by time period"""
    if df.empty or period_months == 'All':
        return df
    
    df['date'] = pd.to_datetime(df['date'])
    latest_date = df['date'].max()
    start_date = latest_date - pd.DateOffset(months=int(period_months))
    
    return df[df['date'] >= start_date]

def calculate_financial_metrics(data, company_age_months):
    """Calculate comprehensive financial metrics - ENHANCED VERSION"""
    if data.empty:
        return {}
    
    try:
        if BUSINESS_SIGNAL_SERVICES_AVAILABLE and modular_business_metrics is not None:
            metrics = modular_business_metrics(data, company_age_months)
        else:
            data = categorize_transactions(data)
            metrics = {}

        # DEBUGGING: Print key values
        print(f"\n🔍 DEBUG - Financial Metrics:")
        print(
            f"  Total Revenue: £{metrics.get('Total Revenue', 0):,.2f}"
            if metrics.get("Total Revenue") is not None else "  Total Revenue: N/A"
        )
        print(
            f"  Total Expenses: £{metrics.get('Total Expenses', 0):,.2f}"
            if metrics.get("Total Expenses") is not None else "  Total Expenses: N/A"
        )
        print(
            f"  Net Income: £{metrics.get('Net Income', 0):,.2f}"
            if metrics.get("Net Income") is not None else "  Net Income: N/A"
        )
        print(
            f"  DSCR: {metrics.get('Debt Service Coverage Ratio', 0):.2f}"
            if metrics.get("Debt Service Coverage Ratio") is not None else "  DSCR: N/A"
        )
        print(
            f"  Operating Margin: {metrics.get('Operating Margin', 0):.3f} ({metrics.get('Operating Margin', 0)*100:.1f}%)"
            if metrics.get("Operating Margin") is not None else "  Operating Margin: N/A"
        )
        print(
            f"  Cash Flow Volatility: {metrics.get('Cash Flow Volatility', 0):.3f}"
            if metrics.get("Cash Flow Volatility") is not None else "  Cash Flow Volatility: N/A"
        )
        print(
            f"  Revenue Growth Rate: {metrics.get('Revenue Growth Rate', 0)*100:.1f}%"
            if metrics.get("Revenue Growth Rate") is not None else "  Revenue Growth Rate: N/A"
        )
        print(
            f"  Avg Month-End Balance: £{metrics.get('Average Month-End Balance', 0):,.2f}"
            if metrics.get("Average Month-End Balance") is not None else "  Avg Month-End Balance: N/A"
        )
        print(
            f"  Avg Negative Days: {metrics.get('Average Negative Balance Days per Month', 0)}"
            if metrics.get("Average Negative Balance Days per Month") is not None else "  Avg Negative Days: N/A"
        )
        print(
            f"  Bounced Payments: {metrics.get('Number of Bounced Payments', 0)}"
            if metrics.get("Number of Bounced Payments") is not None else "  Bounced Payments: N/A"
        )
        print(
            f"  Funding Reliance: {metrics.get('Funding Reliance Ratio', 0)*100:.1f}%"
            if metrics.get("Funding Reliance Ratio") is not None else "  Funding Reliance: N/A"
        )
        print(
            f"  Transfer Activity: {metrics.get('Internal Transfer Activity Ratio', 0)*100:.1f}%"
            if metrics.get("Internal Transfer Activity Ratio") is not None else "  Transfer Activity: N/A"
        )
        print(
            f"  Revenue Concentration: {metrics.get('Revenue Concentration Risk', 'Unknown')}"
        )
        print(
            f"  Active Lenders: {metrics.get('Active Lenders Detected', 0)}"
            if metrics.get("Active Lenders Detected") is not None else "  Active Lenders: N/A"
        )
        return metrics
        
    except Exception as e:
        st.error(f"Error calculating metrics: {e}")
        import traceback
        print(f"Full error traceback: {traceback.format_exc()}")
        return {}

def calculate_all_scores_enhanced(metrics, params):
    """Enhanced scoring calculation with better debugging and subprime scoring"""
    print = builtins.print if DEBUG_MODE else (lambda *args, **kwargs: None)
    industry_thresholds = get_industry_thresholds(params['industry'])
    sector_risk = get_sector_risk(params['industry'])
    
    print(f"\n🎯 DEBUG - Scoring Calculation:")
    print(f"  Industry: {params['industry']}")
    print(f"  Sector Risk: {sector_risk}")
    print(f"  Directors Score: {params['directors_score']}")
    print(f"  Company Age: {params['company_age_months']} months")
    
    # ADD THIS BLOCK to your calculate_all_scores_enhanced function
    # Right after the debug prints about industry/sector/directors/age

    # NEW: Safe ML validation
    try:
        ml_validator = MLScalerInsights()
        ml_validation = ml_validator.validate_business_data(metrics, params)
    
        if ml_validation.get('available', False):
            print(f"\n🤖 ML Validation:")
            print(f"  Data Quality: {ml_validation['data_quality_score']}/100")
            print(f"  ML Confidence: {ml_validation['ml_confidence']}")
            print(f"  Different Metrics: {ml_validation['outlier_count']}")
        
            if ml_validation.get('outliers'):
                print(f"  Notable differences:")
                for outlier in ml_validation['outliers'][:2]:
                    print(f"    • {outlier['feature']}: {outlier.get('interpretation', 'differs from training')}")
        else:
            ml_validation = {'available': False}
            print(f"\n🤖 ML Validation: Not available")
        
    except Exception as e:
        ml_validation = {'available': False}
        print(f"\n🤖 ML Validation: Error - {e}")


    # DEBUG: Check subprime scoring availability
    print(f"\n🔍 DEBUG - Subprime Scoring Check:")
    print(f"  SUBPRIME_SCORING_AVAILABLE: {globals().get('SUBPRIME_SCORING_AVAILABLE', 'Not defined')}")
    
    try:
        print(f"  SubprimeScoring class available: {SubprimeScoring}")
        scorer_test = SubprimeScoring()
        print(f"  SubprimeScoring instance created successfully: {type(scorer_test)}")
    except Exception as e:
        print(f"  ERROR creating SubprimeScoring: {e}")
        print(f"  This is why subprime score is not working!")
    
    # Calculate weighted scores 
    weighted_score = calculate_weighted_scores(metrics, params, industry_thresholds)
          
    print(f"  Weighted Score: {weighted_score}/100")
        
    # NEW: Subprime scoring - WITH ERROR HANDLING
    print(f"\n🎯 DEBUG - Attempting Subprime Scoring:")
    try:
        subprime_scorer = SubprimeScoring()
        print(f"  Subprime scorer created: {type(subprime_scorer)}")
        
        print(f"  Calling calculate_subprime_score with:")
        print(f"    Metrics keys: {list(metrics.keys())}")
        print(f"    Params keys: {list(params.keys())}")
        print(f"DEBUG Directors Score going into Subprime: {params.get('directors_score')}")
        
        subprime_result = subprime_scorer.calculate_subprime_score(metrics, params)
        print(f"  Subprime calculation successful!")
        print(f"  Subprime Score: {subprime_result['subprime_score']:.1f}/100" if subprime_result.get('subprime_score') is not None else "  Subprime Score: N/A/100")
        print(f"  Subprime Tier: {subprime_result['risk_tier']}")
        
    except Exception as e:
        print(f"  ERROR in subprime scoring: {e}")
        import traceback
        print(f"  Full traceback: {traceback.format_exc()}")
        
        # Fallback result
        subprime_result = {
            'subprime_score': 0,
            'risk_tier': 'Error',
            'pricing_guidance': {'suggested_rate': 'N/A'},
            'recommendation': f'Subprime scoring failed: {str(e)}',
            'breakdown': [f'Error: {str(e)}']
        }
    
    # Industry Score (binary) - ENHANCED with debugging
    industry_score = 0
    score_breakdown = {}
    
    # Check each threshold
    threshold_checks = []
    
    for metric, threshold in industry_thresholds.items():
        if metric in ['Directors Score', 'Sector Risk']:
            continue
        
        if metric in metrics:
            actual_value = metrics[metric]
            if metric in ['Cash Flow Volatility', 'Average Negative Balance Days per Month', 'Number of Bounced Payments']:
                meets_threshold = actual_value <= threshold
                direction = "≤"
            else:
                meets_threshold = actual_value >= threshold
                direction = "≥"
            
            if meets_threshold:
                industry_score += 1
            
            threshold_checks.append(f"  {metric}: {actual_value:.3f} {direction} {threshold} = {'✅' if meets_threshold else '❌'}")
            
            score_breakdown[metric] = {
                'actual': actual_value,
                'threshold': threshold,
                'meets': meets_threshold,
                'direction': 'lower' if metric in ['Cash Flow Volatility', 'Average Negative Balance Days per Month', 'Number of Bounced Payments'] else 'higher'
            }
    
    # Add non-metric scores
    if params['company_age_months'] >= 6:
        industry_score += 1
        threshold_checks.append(f"  Company Age: {params['company_age_months']} ≥ 6 months = ✅")
    else:
        threshold_checks.append(f"  Company Age: {params['company_age_months']} ≥ 6 months = ❌")
        
    if params['directors_score'] >= industry_thresholds['Directors Score']:
        industry_score += 1
        threshold_checks.append(f"  Directors Score: {params['directors_score']} ≥ {industry_thresholds['Directors Score']} = ✅")
    else:
        threshold_checks.append(f"  Directors Score: {params['directors_score']} ≥ {industry_thresholds['Directors Score']} = ❌")
        
    if sector_risk <= industry_thresholds['Sector Risk']:
        industry_score += 1
        threshold_checks.append(f"  Sector Risk: {sector_risk} ≤ {industry_thresholds['Sector Risk']} = ✅")
    else:
        threshold_checks.append(f"  Sector Risk: {sector_risk} ≤ {industry_thresholds['Sector Risk']} = ❌")
    
    print(f"\n📊 Industry Score Breakdown ({industry_score}/12):")
    for check in threshold_checks:
        print(check)
    
    # ML Score (if available) - ENHANCED with growth business adjustment
    model, scaler = load_models()
    ml_score = None
    adjusted_ml_score = None

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
            
            features_scaled = scaler.transform(features_df)
            probability = model.predict_proba(features_scaled)[:, 1] * 100
            ml_score = round(probability[0], 2)
            
            # NEW: Apply growth business adjustment
            adjusted_ml_score = adjust_ml_score_for_growth_business(ml_score, metrics, params)

            if ml_score is not None:
                print(f"  Raw ML Score: {ml_score:.1f}%")
            else:
                print(f"  Raw ML Score: N/A")

            if adjusted_ml_score is not None:
                print(f"  Adjusted ML Score: {adjusted_ml_score:.1f}%")
            else:
                print(f"  Adjusted ML Score: N/A")

        except Exception as e:
            print(f"  ML Score: Error - {e}")
    
    # Loan Risk
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
    
    print(f"  Loan Risk: {loan_risk}")
    
    # NEW: Ensemble scoring for unified recommendation
    # Now properly integrating MCA Rule Decision which is based on 
    # empirically-validated transaction consistency metrics
    ensemble_result = None
    if ENSEMBLE_SCORER_AVAILABLE and get_ensemble_recommendation:
        try:
            # Get MCA rule decision from params (calculated earlier in the flow)
            mca_rule_decision = params.get('mca_rule_decision', 'REFER')
            mca_rule_score = params.get('mca_rule_score', 50)
            
            # Prepare scores for ensemble
            # Decision weights:
            # - MCA Rule (transaction consistency): 60%
            # - Subprime Score: 40%
            # - ML Score: informational only
            ensemble_scores = {
                'subprime_score': subprime_result['subprime_score'],
                'ml_score': adjusted_ml_score or ml_score,
                'mca_score': mca_rule_score,
                'mca_decision': mca_rule_decision,
                'mca_rule_signals': params.get('mca_rule_signals'),
                'mca_rule_reasons': params.get('mca_rule_reasons'),
            }
            
            ensemble_result = get_ensemble_recommendation(
                scores=ensemble_scores,
                metrics=metrics,
                params=params
            )
            
            print(f"\n🎯 ENSEMBLE RECOMMENDATION:")
            print(f"  Combined Score: {ensemble_result['combined_score']:.1f}/100")
            print(f"  Decision: {ensemble_result['decision']}")
            print(f"  Confidence: {ensemble_result['confidence']}%")
            print(f"  Convergence: {ensemble_result['score_convergence']}")
            print(f"  Primary Reason: {ensemble_result['primary_reason']}")
            print(f"  MCA Rule Input: {mca_rule_decision} (score: {mca_rule_score})")
            
        except Exception as e:
            print(f"  Ensemble scoring error: {e}")
            import traceback
            print(f"  {traceback.format_exc()}")
            ensemble_result = None

    # -----------------------------
    # 4B) Apply TU × MCA Decision Logic
    # -----------------------------
    tu_decision = params.get("tu_director_decision")
    mca_decision = ensemble_result.get("decision") if isinstance(ensemble_result, dict) else params.get(
        "mca_rule_decision", "REFER")

    final_decision = combine_mca_and_tu_decisions(mca_decision, tu_decision)

    params["mca_main_decision"] = mca_decision
    params["final_decision"] = final_decision
    
    print(f"="*50)
    
    return {
        'weighted_score': weighted_score,
        'industry_score': industry_score,
        'ml_score': ml_score,
        'adjusted_ml_score': adjusted_ml_score,
        'loan_risk': loan_risk,
        'score_breakdown': score_breakdown,
        'mca_rule_score': params.get('mca_rule_score', 0),
        'mca_rule_decision': params.get('mca_rule_decision', 'REFER'),
        # Subprime scoring results
        'subprime_score': subprime_result['subprime_score'],
        'subprime_tier': subprime_result['risk_tier'],
        'subprime_pricing': subprime_result['pricing_guidance'],
        'subprime_recommendation': subprime_result['recommendation'],
        'subprime_breakdown': subprime_result['breakdown'],
        'ml_validation': ml_validation,
        # Ensemble scoring results
        'ensemble': ensemble_result,
        "final_decision": final_decision,
    }

def adjust_ml_score_for_growth_business(raw_ml_score, metrics, params):
    """
    Adjust ML score for growth businesses that the traditional model undervalues.
    """
    print = builtins.print if DEBUG_MODE else (lambda *args, **kwargs: None)
    
    if raw_ml_score is None:
        return None
    
    print(f"\n🔧 ML Score Adjustment Analysis:")
    if raw_ml_score is not None:
        print(f"  Raw ML Score: {raw_ml_score:.1f}%")
    else:
        print(f"  Raw ML Score: N/A")
    
    # Initialize adjustment
    adjustment = 0
    adjustment_reasons = []
    
    # Factor 1: Strong DSCR despite losses
    dscr = metrics.get('Debt Service Coverage Ratio', 0)
    operating_margin = metrics.get('Operating Margin', 0)
    
    if dscr >= 3.0 and operating_margin < 0:
        adjustment += 4
        adjustment_reasons.append(f"Strong DSCR ({dscr:.1f}) despite losses (+15)")
    elif dscr >= 2.0 and operating_margin < 0:
        adjustment += 3
        adjustment_reasons.append(f"Good DSCR ({dscr:.1f}) despite losses (+12)")
    elif dscr >= 1.5 and operating_margin < 0:
        adjustment += 2
        adjustment_reasons.append(f"Adequate DSCR ({dscr:.1f}) despite losses (+8)")
    
    # Factor 2: High growth trajectory
    growth = metrics.get('Revenue Growth Rate', 0)
    if growth > 1:  # Convert percentage if needed
        growth = growth / 100
    
    if growth >= 0.2:  # 20%+ growth
        adjustment += 3
        adjustment_reasons.append(f"High growth ({growth*100:.1f}%) (+12)")
    elif growth >= 0.15:  # 15%+ growth
        adjustment += 2
        adjustment_reasons.append(f"Strong growth ({growth*100:.1f}%) (+8)")
    elif growth >= 0.1:  # 10%+ growth
        adjustment += 1
        adjustment_reasons.append(f"Good growth ({growth*100:.1f}%) (+5)")
    
    # Factor 3: Growth + DSCR combination (compounding effect)
    if growth >= 0.15 and dscr >= 2.0:
        adjustment += 3
        adjustment_reasons.append("Growth + Strong DSCR combination (+10)")
    elif growth >= 0.1 and dscr >= 1.5:
        adjustment += 1
        adjustment_reasons.append("Growth + Good DSCR combination (+5)")
    
    # Factor 4: Director reliability
    directors_score = params.get('directors_score', 0)
    if directors_score >= 80:
        adjustment += 1
        adjustment_reasons.append(f"Excellent director score ({directors_score}) (+3)")
    elif directors_score >= 70:
        adjustment += 1
        adjustment_reasons.append(f"Good director score ({directors_score}) (+2)")
    
    # Factor 5: Young company bonus (growth trajectory more important)
    company_age = params.get('company_age_months', 0)
    if company_age <= 36 and growth >= 0.15:  # Young + high growth
        adjustment += 2
        adjustment_reasons.append(f"Young high-growth company ({company_age}m, {growth*100:.1f}%) (+5)")
    
    # Factor 6: Revenue scale adjustment
    revenue = metrics.get('Total Revenue', 0)
    monthly_revenue = metrics.get('Monthly Average Revenue', 0)
    
    if monthly_revenue >= 10000:  # £10k+ monthly revenue
        adjustment += 1
        adjustment_reasons.append(f"Strong revenue scale (£{monthly_revenue:,.0f}/month) (+3)")
    elif monthly_revenue >= 5000:  # £5k+ monthly revenue
        adjustment += 1
        adjustment_reasons.append(f"Good revenue scale (£{monthly_revenue:,.0f}/month) (+2)")
    
    # Apply adjustment with cap. The adjustment is intended to uplift
    # undervalued growth businesses, not compress already-strong ML scores.
    adjusted_score = min(85, raw_ml_score + adjustment)
    
    print(f"  Adjustment Factors:")
    for reason in adjustment_reasons:
        print(f"    • {reason}")
    
    print(f"  Total Adjustment: +{adjustment:.1f} points")
    print(f"  Adjusted ML Score: {adjusted_score:.1f}%")
    print(f"  Improvement: {adjusted_score - raw_ml_score:+.1f} points")
    
    return adjusted_score

def get_ml_score_interpretation(adjusted_score, raw_score):
    """Provide interpretation of the adjusted ML score."""
    
    if adjusted_score is None:
        return "ML model not available"
    
    improvement = adjusted_score - raw_score
    
    if adjusted_score >= 80:
        risk_level = "Low Risk"
    elif adjusted_score >= 70:
        risk_level = "Moderate Risk"
    elif adjusted_score >= 60:
        risk_level = "Higher Risk"
    else:
        risk_level = "High Risk"

    interpretation = f"**{risk_level}** (Adjusted: {adjusted_score:.1f}%, Raw: {raw_score:.1f}%)"

    if improvement >= 15:
        interpretation += f"\n  **Significant upward adjustment** (+{improvement:.1f}) for growth business profile"
    elif improvement >= 8:
        interpretation += f"\n  **Notable upward adjustment** (+{improvement:.1f}) for growth characteristics"
    elif improvement >= 3:
        interpretation += f"\n  **Minor upward adjustment** (+{improvement:.1f}) for positive factors"
    else:
        interpretation += f"\n  **Minimal adjustment** (+{improvement:.1f}) — standard risk profile"
    
    return interpretation

def calculate_revenue_insights(df):
    """Calculate revenue-specific insights"""
    if df.empty:
        return {}
    
    # Apply categorization first
    categorized_data = categorize_transactions(df.copy())
    
    # Filter for revenue transactions only
    revenue_data = categorized_data[categorized_data['is_revenue']].copy()
    
    if revenue_data.empty:
        return {
            'unique_revenue_sources': 0,
            'avg_revenue_transactions_per_day': 0,
            'avg_daily_revenue_amount': 0,
            'total_revenue_days': 0
        }
    
    # Ensure we have the name column for revenue sources
    name_column = 'name' if 'name' in revenue_data.columns else 'name_y' if 'name_y' in revenue_data.columns else None
    
    if name_column is None:
        unique_revenue_sources = 0
    else:
        # Count unique revenue sources (unique company/merchant names)
        unique_revenue_sources = revenue_data[name_column].nunique()
    
    # Calculate daily metrics
    revenue_data['date'] = pd.to_datetime(revenue_data['date'])
    revenue_data['date_only'] = revenue_data['date'].dt.date
    
    # Group by date to get daily totals
    daily_revenue = revenue_data.groupby('date_only').agg({
        'amount': ['count', lambda x: abs(x).sum()]
    }).round(2)
    
    daily_revenue.columns = ['daily_transaction_count', 'daily_revenue_amount']
    
    # Calculate averages
    total_revenue_days = len(daily_revenue)
    avg_revenue_transactions_per_day = daily_revenue['daily_transaction_count'].mean()
    avg_daily_revenue_amount = daily_revenue['daily_revenue_amount'].mean()
    
    return {
        'unique_revenue_sources': unique_revenue_sources,
        'avg_revenue_transactions_per_day': round(avg_revenue_transactions_per_day, 1),
        'avg_daily_revenue_amount': round(avg_daily_revenue_amount, 2),
        'total_revenue_days': total_revenue_days,
        'daily_revenue_data': daily_revenue
    }


def assess_primary_account_signal(df):
    """
    UW-only heuristic signal to indicate whether uploaded data may not represent
    the primary operating account.
    This is informational only and does not affect scoring decisions.
    """
    default_result = {
        "status": "unable_to_determine",
        "is_potential_non_primary": False,
        "note": "Unable to determine primary account reliability from uploaded data.",
        "active_accounts": 0,
        "top_account_share_non_transfer_credits": 0.0,
        "top_account_share_activity": 0.0,
        "internal_transfer_ratio": 0.0,
    }

    if df is None or df.empty or "account_id" not in df.columns:
        return default_result

    work = df.copy()
    work["amount"] = pd.to_numeric(work.get("amount"), errors="coerce").fillna(0.0)
    work = work[work["account_id"].notna()]
    if work.empty:
        return default_result

    category_col = "personal_finance_category.detailed"
    name_col = "name"
    if name_col not in work.columns:
        name_col = "name_y" if "name_y" in work.columns else None
    if category_col not in work.columns:
        work[category_col] = ""
    if name_col is None:
        work["_txn_name"] = ""
    else:
        work["_txn_name"] = work[name_col].fillna("").astype(str).str.lower()

    work["_pf_category"] = work[category_col].fillna("").astype(str).str.lower()
    transfer_name_regex = r"\b(transfer|trf|faster payment|own account|between accounts|sweep)\b"
    work["_is_transfer_like"] = (
        work["_pf_category"].str.startswith("transfer_")
        | work["_txn_name"].str.contains(transfer_name_regex, regex=True)
    )

    active_threshold = 5
    account_activity = work.groupby("account_id").size().rename("txn_count")
    active_accounts = int((account_activity >= active_threshold).sum())
    if active_accounts == 0:
        active_accounts = int((account_activity > 0).sum())

    non_transfer = work[~work["_is_transfer_like"]].copy()
    if non_transfer.empty:
        return {
            **default_result,
            "status": "unable_to_determine",
            "note": "Most transactions appear transfer-like; unable to infer a clear primary operating account.",
            "active_accounts": active_accounts,
            "internal_transfer_ratio": 1.0,
        }

    credits = non_transfer[non_transfer["amount"] < 0].copy()
    if credits.empty:
        return {
            **default_result,
            "status": "unable_to_determine",
            "note": "No non-transfer credit inflows found; unable to infer a primary operating account.",
            "active_accounts": active_accounts,
        }

    credit_by_account = credits.groupby("account_id")["amount"].apply(lambda s: abs(float(s.sum())))
    total_credits = float(credit_by_account.sum())
    top_credit_share = float((credit_by_account.max() / total_credits) if total_credits > 0 else 0.0)

    total_txn = float(len(work))
    top_activity_share = float((account_activity.max() / total_txn) if total_txn > 0 else 0.0)

    transfer_ratio = float(work["_is_transfer_like"].mean()) if len(work) > 0 else 0.0

    is_potential_non_primary = (
        (active_accounts >= 2 and top_credit_share < 0.60 and top_activity_share < 0.60)
        or (active_accounts >= 2 and top_credit_share < 0.50)
        or (transfer_ratio >= 0.45)
    )

    if is_potential_non_primary:
        note = (
            "Potential non-primary account data: activity appears spread across accounts "
            "or transfer-heavy. Underwriter review recommended."
        )
        status = "potential_non_primary"
    else:
        note = "Primary account likely represented in uploaded data (informational check)."
        status = "likely_primary"

    return {
        "status": status,
        "is_potential_non_primary": bool(is_potential_non_primary),
        "note": note,
        "active_accounts": active_accounts,
        "top_account_share_non_transfer_credits": round(top_credit_share, 3),
        "top_account_share_activity": round(top_activity_share, 3),
        "internal_transfer_ratio": round(transfer_ratio, 3),
    }


def render_card_terminal_reconciliation(bank_df, card_files, card_processing_payload: dict | None = None):
    """Render card terminal reconciliation from already uploaded files."""
    st.markdown("---")
    st.subheader("Card terminal statements (multi-company)")
    st.caption(
        "Upload one or more terminal statements (PDF/CSV/Excel) from any provider. "
        "The app normalizes totals and compares monthly card sales to bank revenue inflows."
    )

    payload = card_processing_payload
    parsed_payload_df = payload.get("parsed_df") if payload else None
    has_payload = bool(
        payload
        and (
            payload.get("parse_output")
            or (isinstance(parsed_payload_df, pd.DataFrame) and not parsed_payload_df.empty)
            or payload.get("error")
        )
    )
    if not card_files and not has_payload:
        st.info("No card terminal statements uploaded yet.")
        return

    payload = payload or derive_card_processing_payload(bank_df, card_files)
    if payload.get("error"):
        st.error(f"Failed to parse card terminal statements: {payload.get('error')}")
        return
    parse_output = payload.get("parse_output", {})
    parsed_df = payload.get("parsed_df", pd.DataFrame())
    parse_errors = parse_output.get("errors", [])

    top1, top2, top3 = st.columns(3)
    with top1:
        st.metric("Statements Parsed", int(parse_output.get("parsed_count", 0)))
    with top2:
        st.metric("Parse Errors", int(parse_output.get("error_count", 0)))
    with top3:
        providers = parsed_df["provider"].nunique() if not parsed_df.empty and "provider" in parsed_df.columns else 0
        st.metric("Providers Detected", int(providers))

    if parse_errors:
        with st.expander("Card Statement Parse Errors", expanded=False):
            st.json(parse_errors)

    if parsed_df.empty:
        st.warning("No card terminal statements were parsed successfully.")
        return

    st.markdown("**Parsed Statement Totals**")
    parsed_view = parsed_df.copy()
    if "extraction_diagnostics" in parsed_view.columns:
        def _diag_status(diag):
            if not isinstance(diag, dict):
                return "Unknown"
            fields = (diag.get("fields") or {})
            ok = sum(1 for _, v in fields.items() if bool(v))
            total = len(fields)
            if total == 0:
                return "Unknown"
            ratio = ok / total
            if ratio >= 0.85:
                return "High"
            if ratio >= 0.6:
                return "Medium"
            return "Low"

        parsed_view["parser_reliability"] = parsed_view["extraction_diagnostics"].apply(_diag_status)
        parsed_view["profile_used"] = parsed_view["extraction_diagnostics"].apply(
            lambda d: (d.get("profile") if isinstance(d, dict) else "Unknown")
        )
    else:
        parsed_view["parser_reliability"] = "Unknown"
        parsed_view["profile_used"] = "Unknown"

    rel_counts = parsed_view["parser_reliability"].value_counts().to_dict()
    r1, r2, r3 = st.columns(3)
    with r1:
        st.metric("High Reliability Files", int(rel_counts.get("High", 0)))
    with r2:
        st.metric("Medium Reliability Files", int(rel_counts.get("Medium", 0)))
    with r3:
        st.metric("Low Reliability Files", int(rel_counts.get("Low", 0)))

    st.dataframe(
        parsed_view[
            [
                "filename",
                "provider",
                "profile_used",
                "parser",
                "merchant_id",
                "statement_start",
                "statement_end",
                "gross_card_sales",
                "fees_total",
                "transaction_count",
                "confidence",
                "parser_reliability",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Per-file Extraction Diagnostics", expanded=False):
        st.dataframe(
            parsed_view[["filename", "provider", "parser", "parser_reliability", "extraction_diagnostics", "warnings"]],
            use_container_width=True,
            hide_index=True,
        )

    monthly_terminal = payload.get("monthly_terminal", pd.DataFrame())
    if monthly_terminal.empty:
        st.info("Unable to build monthly card-sales summary from uploaded statements.")
        return

    comparison_payload = payload.get("comparison_payload", {"comparison": pd.DataFrame(), "summary": {}})
    comp_df = comparison_payload.get("comparison", pd.DataFrame())
    provider_bank_df = comparison_payload.get("provider_bank_monthly", pd.DataFrame())
    summary = comparison_payload.get("summary", {})
    uploaded_providers = sorted(
        p for p in parsed_df.get("provider", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()
        if p and p != "Unknown"
    )
    detected_providers = summary.get("providers_detected_in_bank_narration", [])
    provider_overlap = sorted(set(uploaded_providers).intersection(set(detected_providers)))
    provider_detection_coverage_pct = float(summary.get("provider_detection_coverage_pct", 0.0) or 0.0)

    if provider_overlap and provider_detection_coverage_pct >= 40:
        narration_confidence = "High"
    elif provider_overlap or provider_detection_coverage_pct >= 20:
        narration_confidence = "Medium"
    else:
        narration_confidence = "Low"

    st.markdown("**Monthly Card Sales vs Bank Revenue Inflows**")
    if not comp_df.empty:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Months Compared", int(summary.get("months_compared", 0)))
        with c2:
            st.metric("Avg Abs Variance", f"{summary.get('average_abs_variance_pct', 0):.1f}%")
        with c3:
            st.metric("Reconciliation Quality", summary.get("reconciliation_quality", "N/A"))

        c4, c5, c6 = st.columns(3)
        with c4:
            st.metric("Uploaded Statement Providers", ", ".join(uploaded_providers) if uploaded_providers else "Unknown")
        with c5:
            st.metric("Bank Narration Match Coverage", f"{provider_detection_coverage_pct:.1f}%")
        with c6:
            st.metric("Narration Signal Confidence", narration_confidence)

        if detected_providers:
            st.info("Bank narration detected provider-like terms: " + ", ".join(detected_providers))
        else:
            st.info("No known provider narration detected in bank inflows for this period.")

        if provider_overlap:
            st.success(
                "Verified overlap between uploaded providers and bank narration: "
                + ", ".join(provider_overlap)
            )
        else:
            st.warning(
                "No direct overlap between uploaded statement providers and bank narration signals. "
                "Treat narration-based providers as unverified leads."
            )

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=comp_df["year_month"],
                y=comp_df["gross_card_sales"],
                name="Terminal Gross Card Sales",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=comp_df["year_month"],
                y=comp_df["bank_revenue_inflows"],
                mode="lines+markers",
                name="Bank Revenue Inflows",
            )
        )
        fig.update_layout(
            title=dict(text="Terminal card sales vs bank revenue inflows", x=0.02, xanchor="left"),
            height=420,
            xaxis_title="Month",
            yaxis_title="Amount (£)",
            legend_title_text="Series",
            legend=LEGEND_BELOW,
            margin=dict(t=56, b=108, l=56, r=36),
        )
        show_mca_plotly(fig, key="card_vs_bank_monthly")

        display_df = comp_df.copy()
        display_df["difference_amount"] = display_df["difference_amount"].round(2)
        display_df["difference_pct_vs_terminal"] = display_df["difference_pct_vs_terminal"].round(2)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        if provider_bank_df is not None and not provider_bank_df.empty:
            st.markdown("**Bank Narration Provider Signals (Unverified unless matched to uploaded provider)**")
            provider_bank_display = provider_bank_df.copy()
            provider_bank_display["verified_against_uploaded"] = provider_bank_display["provider"].isin(uploaded_providers)
            provider_bank_display["signal_confidence"] = provider_bank_display["verified_against_uploaded"].map(
                lambda v: "High" if v else ("Medium" if narration_confidence == "High" else "Low")
            )
            st.dataframe(provider_bank_display, use_container_width=True, hide_index=True)
    else:
        st.info("No comparable months found between uploaded card statements and bank transaction period.")

    with st.expander("Supported Provider Reference (from current rules + docs links)", expanded=False):
        catalog = summary.get("provider_catalog", [])
        if catalog:
            catalog_df = pd.DataFrame(catalog)
            st.dataframe(catalog_df, use_container_width=True, hide_index=True)
        else:
            st.info("Provider reference catalog not available.")

def create_score_charts(scores, metrics):
    """Create clean bar charts for scores - Updated for 3 scoring methods"""

    def _score_value(raw, fallback=None):
        if raw is not None:
            try:
                return float(raw)
            except (TypeError, ValueError):
                pass
        if fallback is not None:
            try:
                return float(fallback)
            except (TypeError, ValueError):
                pass
        return 0.0

    def _score_label(key, raw, fallback=None):
        display_raw = raw if raw is not None else fallback
        if display_raw is None:
            return "N/A"
        try:
            val = float(display_raw)
        except (TypeError, ValueError):
            return "N/A"
        suffix = "%" if key == "Adjusted ML" else "/100"
        return f"{val:.1f}{suffix}"

    adjusted_raw = scores.get("adjusted_ml_score")
    if adjusted_raw is None:
        adjusted_raw = scores.get("ml_score")

    score_data = {
        "Subprime Score": _score_value(scores.get("subprime_score")),
        "MCA Rule": _score_value(scores.get("mca_rule_score")),
        "Adjusted ML": _score_value(adjusted_raw),
    }
    score_labels = {
        "Subprime Score": _score_label("Subprime Score", scores.get("subprime_score")),
        "MCA Rule": _score_label("MCA Rule", scores.get("mca_rule_score")),
        "Adjusted ML": _score_label("Adjusted ML", scores.get("adjusted_ml_score"), scores.get("ml_score")),
    }

    # Score comparison chart
    fig_scores = go.Figure()

    colors = ['#2ca02c', '#1f77b4', '#ff7f0e']  # Green, Blue, Orange

    fig_scores.add_trace(go.Bar(
        x=list(score_data.keys()),
        y=list(score_data.values()),
        marker_color=colors,
        text=[score_labels[k] for k in score_data.keys()],
        textposition='outside'
    ))
    
    fig_scores.update_layout(
        title=dict(text="Primary scoring methods", x=0.02, xanchor="left"),
        yaxis_title="Score",
        showlegend=False,
        height=420,
        yaxis=dict(range=[0, 100]),
    )
    
    return fig_scores

def create_financial_charts(metrics):
    """Create financial performance charts"""
    
    # Financial metrics bar chart
    key_metrics = {
        'Total Revenue': metrics.get('Total Revenue', 0),
        'Total Expenses': metrics.get('Total Expenses', 0),
        'Net Income': metrics.get('Net Income', 0),
        'Total Debt': metrics.get('Total Debt', 0),
        'Debt Repayments': metrics.get('Total Debt Repayments', 0)
    }
    
    fig_financial = go.Figure()
    
    colors = ['green' if v >= 0 else 'red' for v in key_metrics.values()]
    
    fig_financial.add_trace(go.Bar(
        x=list(key_metrics.keys()),
        y=list(key_metrics.values()),
        marker_color=colors,
        text=[f"£{v:,.0f}" for v in key_metrics.values()],
        textposition='outside'
    ))
    
    fig_financial.update_layout(
        title=dict(text="Financial overview", x=0.02, xanchor="left"),
        yaxis_title="Amount (£)",
        showlegend=False,
        height=420,
    )
    
    # Monthly trend if available
    fig_trend = None
    if 'monthly_summary' in metrics and not metrics['monthly_summary'].empty:
        monthly_data = metrics['monthly_summary'].reset_index()
        monthly_data['date'] = monthly_data['year_month'].astype(str)
        
        fig_trend = go.Figure()
        
        fig_trend.add_trace(go.Scatter(
            x=monthly_data['date'],
            y=monthly_data['monthly_revenue'],
            mode='lines+markers',
            name='Revenue',
            line=dict(color='#34d399', width=2.5),
            marker=dict(size=7, line=dict(width=1, color='#064e3b')),
        ))
        
        fig_trend.add_trace(go.Scatter(
            x=monthly_data['date'],
            y=monthly_data['monthly_expenses'],
            mode='lines+markers',
            name='Expenses',
            line=dict(color='#fb7185', width=2.5),
            marker=dict(size=7, line=dict(width=1, color='#881337')),
        ))
        
        fig_trend.update_layout(
            title=dict(text="Monthly revenue vs expenses", x=0.02, xanchor="left"),
            xaxis_title="Month",
            yaxis_title="Amount (£)",
            height=440,
            legend=LEGEND_BELOW,
            margin=dict(t=56, b=108, l=56, r=36),
        )
    
    return fig_financial, fig_trend

def create_threshold_chart(score_breakdown):
    """Create threshold comparison chart"""
    
    metrics = []
    actual_values = []
    threshold_values = []
    colors = []
    
    for metric, data in score_breakdown.items():
        metrics.append(metric.replace('_', ' ').title())
        actual_values.append(data['actual'])
        threshold_values.append(data['threshold'])
        colors.append('green' if data['meets'] else 'red')
    
    fig = go.Figure()
    
    # Actual values
    fig.add_trace(go.Bar(
        name='Actual',
        x=metrics,
        y=actual_values,
        marker_color=colors,
        opacity=0.8
    ))
    
    # Threshold lines
    fig.add_trace(go.Scatter(
        name='Threshold',
        x=metrics,
        y=threshold_values,
        mode='lines+markers',
        line=dict(color=THRESHOLD_LINE, width=2, dash='dash'),
        marker=dict(
            size=9,
            symbol='diamond',
            color=THRESHOLD_MARKER,
            line=dict(color='#f8fafc', width=1),
        ),
        connectgaps=True,
    ))
    
    fig.update_layout(
        title=dict(text="Actual vs industry thresholds", x=0.02, xanchor="left"),
        xaxis_title="Metric",
        yaxis_title="Value",
        height=540,
        legend=LEGEND_BELOW,
        margin=dict(t=56, b=120, l=56, r=36),
    )
    fig.update_xaxes(tickangle=-28, tickfont=dict(size=10), automargin=True)

    return fig

def create_monthly_breakdown(df):
    """Create monthly breakdown by subcategory"""
    if df.empty:
        return None, None
    
    # Apply categorization
    categorized_data = categorize_transactions(df.copy())
    
    # Create monthly summary
    categorized_data['date'] = pd.to_datetime(categorized_data['date'])
    categorized_data['year_month'] = categorized_data['date'].dt.to_period('M')
    
    # Group by month and subcategory
    monthly_breakdown = categorized_data.groupby(['year_month', 'subcategory']).agg({
        'amount': ['count', lambda x: abs(x).sum()]
    }).round(2)
    
    monthly_breakdown.columns = ['Transaction_Count', 'Total_Amount']
    monthly_breakdown = monthly_breakdown.reset_index()
    
    # Pivot to get subcategories as columns
    pivot_counts = monthly_breakdown.pivot(index='year_month', columns='subcategory', values='Transaction_Count').fillna(0)
    pivot_amounts = monthly_breakdown.pivot(index='year_month', columns='subcategory', values='Total_Amount').fillna(0)
    
    return pivot_counts, pivot_amounts

def create_monthly_charts(pivot_counts, pivot_amounts):
    """Create monthly breakdown charts"""
    
    # Transaction count chart
    months = [str(month) for month in pivot_counts.index]
    
    fig_counts = go.Figure()
    colors = ['#22c55e', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4']
    
    for i, category in enumerate(pivot_counts.columns):
        fig_counts.add_trace(go.Bar(
            name=category,
            x=months,
            y=pivot_counts[category],
            marker_color=colors[i % len(colors)]
        ))
    
    fig_counts.update_layout(
        title=dict(text="Monthly transaction counts by category", x=0.02, xanchor="left"),
        xaxis_title="Month",
        yaxis_title="Number of transactions",
        barmode='stack',
        height=460,
        legend=LEGEND_BELOW,
        margin=dict(t=56, b=130, l=56, r=28),
    )
    
    # Amount chart
    fig_amounts = go.Figure()
    
    for i, category in enumerate(pivot_amounts.columns):
        fig_amounts.add_trace(go.Bar(
            name=category,
            x=months,
            y=pivot_amounts[category],
            marker_color=colors[i % len(colors)]
        ))
    
    fig_amounts.update_layout(
        title=dict(text="Monthly transaction amounts by category", x=0.02, xanchor="left"),
        xaxis_title="Month",
        yaxis_title="Amount (£)",
        barmode='stack',
        height=460,
        legend=LEGEND_BELOW,
        margin=dict(t=56, b=130, l=56, r=28),
    )
    
    return fig_counts, fig_amounts

def create_categorized_csv(df):
    """Create CSV with categorization"""
    if df.empty:
        return None
    
    # Apply categorization
    categorized_df = categorize_transactions(df.copy())
    
    # Select and order columns for export
    export_columns = [
        'date', 'name', 'amount', 'subcategory', 
        'is_revenue', 'is_expense', 'is_debt_repayment', 'is_debt'
    ]
    
    # Add any additional columns that exist
    additional_cols = ['merchant_name', 'category', 'personal_finance_category.detailed']
    for col in additional_cols:
        if col in categorized_df.columns:
            export_columns.append(col)
    
    # Filter to existing columns
    available_columns = [col for col in export_columns if col in categorized_df.columns]
    export_df = categorized_df[available_columns].copy()
    
    # Sort by date (newest first)
    export_df = export_df.sort_values('date', ascending=False)
    
    return export_df.to_csv(index=False)

def analyze_loans_and_repayments(df):
    """Comprehensive analysis of loans received and debt repayments"""
    if df.empty:
        return {}
    
    # Apply categorization to ensure we have subcategories
    categorized_data = categorize_transactions(df.copy())
    
    # Extract loans and repayments
    loans_data = categorized_data[categorized_data['subcategory'] == 'Loans'].copy()
    repayments_data = categorized_data[categorized_data['subcategory'] == 'Debt Repayments'].copy()
    
    # Prepare date columns
    for data in [loans_data, repayments_data]:
        if not data.empty:
            data['date'] = pd.to_datetime(data['date'])
            data['month'] = data['date'].dt.to_period('M')
            data['amount_abs'] = abs(data['amount'])
    
    analysis = {}
    
    # === LOANS ANALYSIS ===
    if not loans_data.empty:
        analysis['total_loans_received'] = loans_data['amount_abs'].sum()
        analysis['loan_count'] = len(loans_data)
        analysis['avg_loan_amount'] = loans_data['amount_abs'].mean()
        analysis['largest_loan'] = loans_data['amount_abs'].max()
        analysis['smallest_loan'] = loans_data['amount_abs'].min()
        
        # Monthly loans
        analysis['loans_by_month'] = loans_data.groupby('month')['amount_abs'].agg(['count', 'sum']).reset_index()
        analysis['loans_by_month']['month_str'] = analysis['loans_by_month']['month'].astype(str)
        
        # Lender analysis
        loans_data['lender_clean'] = loans_data['name'].str.lower().str.strip()
        analysis['loans_by_lender'] = loans_data.groupby('lender_clean')['amount_abs'].agg(['count', 'sum']).reset_index()
        analysis['loans_by_lender'] = analysis['loans_by_lender'].sort_values('sum', ascending=False)
    else:
        analysis.update({
            'total_loans_received': 0, 'loan_count': 0, 'avg_loan_amount': 0,
            'largest_loan': 0, 'smallest_loan': 0, 'loans_by_month': pd.DataFrame(),
            'loans_by_lender': pd.DataFrame()
        })
    
    # === REPAYMENTS ANALYSIS ===
    if not repayments_data.empty:
        analysis['total_repayments_made'] = repayments_data['amount_abs'].sum()
        analysis['repayment_count'] = len(repayments_data)
        analysis['avg_repayment_amount'] = repayments_data['amount_abs'].mean()
        analysis['largest_repayment'] = repayments_data['amount_abs'].max()
        analysis['smallest_repayment'] = repayments_data['amount_abs'].min()
        
        # Monthly repayments
        analysis['repayments_by_month'] = repayments_data.groupby('month')['amount_abs'].agg(['count', 'sum']).reset_index()
        analysis['repayments_by_month']['month_str'] = analysis['repayments_by_month']['month'].astype(str)
        
        # Recipient analysis
        repayments_data['recipient_clean'] = repayments_data['name'].str.lower().str.strip()
        analysis['repayments_by_recipient'] = repayments_data.groupby('recipient_clean')['amount_abs'].agg(['count', 'sum']).reset_index()
        analysis['repayments_by_recipient'] = analysis['repayments_by_recipient'].sort_values('sum', ascending=False)
    else:
        analysis.update({
            'total_repayments_made': 0, 'repayment_count': 0, 'avg_repayment_amount': 0,
            'largest_repayment': 0, 'smallest_repayment': 0, 'repayments_by_month': pd.DataFrame(),
            'repayments_by_recipient': pd.DataFrame()
        })
    
    # === COMBINED ANALYSIS ===
    analysis['net_borrowing'] = analysis['total_loans_received'] - analysis['total_repayments_made']
    analysis['repayment_ratio'] = (analysis['total_repayments_made'] / analysis['total_loans_received']) if analysis['total_loans_received'] > 0 else None
    analysis['repayments_without_visible_loan'] = bool(analysis['loan_count'] == 0 and analysis['repayment_count'] > 0)

    if not analysis['repayments_by_recipient'].empty:
        possible_lenders = analysis['repayments_by_recipient'].copy()
        possible_lenders['possible_lender'] = possible_lenders['recipient_clean'].str.title()
        visible_lenders = set()
        if not analysis['loans_by_lender'].empty:
            visible_lenders = set(analysis['loans_by_lender']['lender_clean'].astype(str).str.lower().str.strip())
        possible_lenders['loan_credit_seen'] = possible_lenders['recipient_clean'].isin(visible_lenders)
        possible_lenders['review_reason'] = possible_lenders['loan_credit_seen'].map({
            True: "Repayment recipient with visible loan credit",
            False: "Repayments found but no matching loan credit in selected bank data",
        })
        possible_lenders = possible_lenders.rename(
            columns={'count': 'repayment_count', 'sum': 'total_repaid_in_period'}
        )
        analysis['possible_lenders_from_repayments'] = possible_lenders[
            [
                'possible_lender',
                'recipient_clean',
                'repayment_count',
                'total_repaid_in_period',
                'loan_credit_seen',
                'review_reason',
            ]
        ]
    else:
        analysis['possible_lenders_from_repayments'] = pd.DataFrame()
    
    # Monthly net borrowing trend
    if not loans_data.empty or not repayments_data.empty:
        all_months = set()
        if not loans_data.empty:
            all_months.update(loans_data['month'].unique())
        if not repayments_data.empty:
            all_months.update(repayments_data['month'].unique())
        
        monthly_net = []
        for month in sorted(all_months):
            month_loans = loans_data[loans_data['month'] == month]['amount_abs'].sum() if not loans_data.empty else 0
            month_repayments = repayments_data[repayments_data['month'] == month]['amount_abs'].sum() if not repayments_data.empty else 0
            monthly_net.append({
                'month': month,
                'month_str': str(month),
                'loans': month_loans,
                'repayments': month_repayments,
                'net_borrowing': month_loans - month_repayments
            })
        
        analysis['monthly_net_borrowing'] = pd.DataFrame(monthly_net)
    else:
        analysis['monthly_net_borrowing'] = pd.DataFrame()
    
    return analysis


def get_manual_outstanding_debt_total():
    """Return underwriter-entered outstanding debt balances from session state."""
    manual_balances = st.session_state.get("manual_outstanding_debt_balances", {}) or {}
    total = 0.0
    for value in manual_balances.values():
        try:
            total += max(float(value or 0), 0.0)
        except (TypeError, ValueError):
            continue
    return round(total, 2)


def get_manual_outstanding_debt_count():
    """Return count of underwriter-entered outstanding debt balances."""
    manual_balances = st.session_state.get("manual_outstanding_debt_balances", {}) or {}
    count = 0
    for value in manual_balances.values():
        try:
            if float(value or 0) > 0:
                count += 1
        except (TypeError, ValueError):
            continue
    return count


def apply_manual_outstanding_debt(metrics):
    """Apply underwriter-entered known balances to debt metrics before scoring."""
    manual_total = get_manual_outstanding_debt_total()
    metrics["Manual Outstanding Debt"] = manual_total
    if manual_total <= 0:
        return metrics

    existing_total_debt = float(metrics.get("Total Debt", 0) or 0)
    adjusted_total_debt = existing_total_debt + manual_total
    revenue_for_ratios = max(float(metrics.get("Total Revenue", 0) or 0), 1.0)

    metrics["Open Banking Total Debt"] = round(existing_total_debt, 2)
    metrics["Total Debt"] = round(adjusted_total_debt, 2)
    metrics["Debt-to-Income Ratio"] = round(min(adjusted_total_debt / revenue_for_ratios, 10.0), 3)
    metrics["Manual Debt Applied In Score"] = "Yes"
    return metrics


def apply_manual_outstanding_debt_to_loans_analysis(analysis):
    """Reflect underwriter-entered balances in loan/repayment display metrics and charts."""
    manual_total = get_manual_outstanding_debt_total()
    analysis["manual_outstanding_debt"] = manual_total
    if manual_total <= 0:
        analysis["total_known_borrowing"] = analysis.get("total_loans_received", 0)
        analysis["known_outstanding_balance"] = max(float(analysis.get("net_borrowing", 0) or 0), 0.0)
        return analysis

    visible_loans = float(analysis.get("total_loans_received", 0) or 0)
    repayments = float(analysis.get("total_repayments_made", 0) or 0)

    if visible_loans > 0:
        total_known_borrowing = visible_loans + manual_total
    else:
        total_known_borrowing = repayments + manual_total

    analysis["total_known_borrowing"] = round(total_known_borrowing, 2)
    analysis["known_outstanding_balance"] = manual_total
    analysis["net_borrowing"] = manual_total
    analysis["repayment_ratio"] = repayments / total_known_borrowing if total_known_borrowing > 0 else None

    monthly_net = analysis.get("monthly_net_borrowing", pd.DataFrame())
    if monthly_net.empty:
        today_month = pd.Timestamp.today().to_period("M")
        monthly_net = pd.DataFrame(
            [
                {
                    "month": today_month,
                    "month_str": str(today_month),
                    "loans": 0.0,
                    "repayments": 0.0,
                    "manual_balance_adjustment": manual_total,
                    "net_borrowing": manual_total,
                }
            ]
        )
    else:
        monthly_net = monthly_net.copy()
        monthly_net["manual_balance_adjustment"] = 0.0
        current_final_position = float(monthly_net["net_borrowing"].sum())
        balance_adjustment = manual_total - current_final_position
        monthly_net.loc[monthly_net.index[0], "manual_balance_adjustment"] = balance_adjustment
        monthly_net["net_borrowing"] = (
            monthly_net["loans"] + monthly_net["manual_balance_adjustment"] - monthly_net["repayments"]
        )

    analysis["monthly_net_borrowing"] = monthly_net
    return analysis


def rerun_last_analysis_with_manual_debt():
    """Recalculate the current session run after underwriter-entered balances change."""
    run = st.session_state.get("last_run")
    if not run:
        return False

    df = run["df"]
    analysis_period = run["analysis_period"]
    params = run["params"]
    card_terminal_files = run.get("card_terminal_files")

    filtered_df = filter_data_by_period(df, analysis_period)
    primary_account_assessment = assess_primary_account_signal(filtered_df)
    params["primary_account_assessment"] = primary_account_assessment

    metrics = calculate_financial_metrics(filtered_df, params["company_age_months"])
    metrics = apply_manual_outstanding_debt(metrics)
    card_processing_payload = derive_card_processing_payload(filtered_df, card_terminal_files)
    metrics.update(card_processing_payload.get("insights") or {})
    scores = calculate_all_scores_enhanced(metrics, params)

    scores["mca_rule_decision"] = params.get("mca_rule_decision")
    scores["mca_rule_score"] = params.get("mca_rule_score")
    scores["mca_rule_reasons"] = params.get("mca_rule_reasons", [])
    scores["primary_account_assessment"] = primary_account_assessment

    def _base_decision_from_subprime(recommendation_text: str) -> str:
        s = (recommendation_text or "").upper()
        if "CONDITIONAL" in s or "SENIOR REVIEW" in s or "REVIEW" in s:
            return "REFER"
        if "APPROVE" in s:
            return "APPROVE"
        return "DECLINE"

    ensemble = scores.get("ensemble") or {}
    ensemble_decision = ensemble.get("decision")
    if ensemble_decision is not None:
        ensemble_decision = str(ensemble_decision).upper().strip()

    if ensemble_decision in ("DECLINE", "REFER", "APPROVE", "SENIOR_REVIEW", "CONDITIONAL_APPROVE"):
        final_decision = combine_mca_and_tu_decisions(
            ensemble_decision,
            params.get("tu_director_decision"),
        )
        final_reasons = [
            f"Decision from weighted MCA/Subprime engine: {ensemble_decision}",
            f"Final TU overlay: {ensemble_decision} -> {final_decision}",
            f"Reason: {ensemble.get('primary_reason', 'n/a')}",
        ]
    else:
        base_decision = _base_decision_from_subprime(scores.get("subprime_recommendation", ""))
        final_decision = combine_mca_and_tu_decisions(
            base_decision,
            params.get("tu_director_decision"),
        )
        final_reasons = [
            f"Fallback decision from Subprime: {base_decision}",
            f"Final TU overlay: {base_decision} -> {final_decision}",
        ]

    scores["final_decision"] = final_decision
    scores["final_decision_reasons"] = final_reasons
    params["final_decision"] = final_decision
    params["final_decision_reasons"] = final_reasons

    revenue_insights = calculate_revenue_insights(filtered_df)

    run.update(
        {
            "filtered_df": filtered_df,
            "params": params,
            "metrics": metrics,
            "scores": scores,
            "revenue_insights": revenue_insights,
            "card_processing_payload": card_processing_payload,
        }
    )
    st.session_state["last_run"] = run
    return True


def render_manual_outstanding_debt_form(possible_lenders, key):
    """Render a batch-save editor for possible lender outstanding balances."""
    if possible_lenders.empty:
        return

    display_possible_lenders = possible_lenders.copy()
    display_possible_lenders["Possible lender"] = display_possible_lenders["possible_lender"]
    display_possible_lenders["Repayments"] = display_possible_lenders["repayment_count"]
    display_possible_lenders["Repaid in period (£)"] = display_possible_lenders["total_repaid_in_period"].round(2)
    display_possible_lenders["Outstanding balance (£)"] = display_possible_lenders["Possible lender"].str.lower().str.strip().map(
        st.session_state.get("manual_outstanding_debt_balances", {}) or {}
    ).fillna(0.0)

    with st.form(f"{key}_form", clear_on_submit=False):
        edited_lenders = st.data_editor(
            display_possible_lenders[
                ["Possible lender", "Repayments", "Repaid in period (£)", "Outstanding balance (£)"]
            ],
            hide_index=True,
            use_container_width=True,
            disabled=["Possible lender", "Repayments", "Repaid in period (£)"],
            key=f"{key}_editor",
        )
        submitted = st.form_submit_button("Save balances and reprocess", type="primary", use_container_width=True)

    manual_total_before = get_manual_outstanding_debt_total()
    if manual_total_before > 0:
        st.caption(f"Currently saved outstanding balances: £{manual_total_before:,.2f}")

    if submitted:
        manual_balances = st.session_state.get("manual_outstanding_debt_balances", {}) or {}
        for _, row in edited_lenders.iterrows():
            lender_key = str(row["Possible lender"]).lower().strip()
            manual_balances[lender_key] = float(row.get("Outstanding balance (£)", 0) or 0)
        st.session_state["manual_outstanding_debt_balances"] = manual_balances
        if rerun_last_analysis_with_manual_debt():
            st.success("Outstanding balances saved and the application was reprocessed.")
            st.rerun()
        else:
            st.success("Outstanding balances saved. Process the application to apply them to scoring.")

def create_loans_repayments_charts(analysis):
    """Create charts for loans and repayments analysis"""
    charts = {}
    
    # 1. Monthly Loans vs Repayments
    if not analysis['monthly_net_borrowing'].empty:
        monthly_data = analysis['monthly_net_borrowing']
        if 'manual_balance_adjustment' not in monthly_data.columns:
            monthly_data = monthly_data.copy()
            monthly_data['manual_balance_adjustment'] = 0.0
        
        fig_monthly = go.Figure()
        
        fig_monthly.add_trace(go.Bar(
            name='Loans Received',
            x=monthly_data['month_str'],
            y=monthly_data['loans'],
            marker_color='lightcoral',
            opacity=0.8
        ))

        if monthly_data['manual_balance_adjustment'].abs().sum() > 0:
            fig_monthly.add_trace(go.Bar(
                name='Entered Outstanding Balances',
                x=monthly_data['month_str'],
                y=monthly_data['manual_balance_adjustment'],
                marker_color='#f59e0b',
                opacity=0.85
            ))
        
        fig_monthly.add_trace(go.Bar(
            name='Debt Repayments',
            x=monthly_data['month_str'],
            y=-monthly_data['repayments'],  # Negative for visual distinction
            marker_color='lightblue',
            opacity=0.8
        ))
        
        fig_monthly.add_trace(go.Scatter(
            name='Net Borrowing',
            x=monthly_data['month_str'],
            y=monthly_data['net_borrowing'],
            mode='lines+markers',
            line=dict(color='darkgreen', width=3),
            marker=dict(size=8)
        ))
        
        fig_monthly.update_layout(
            title=dict(text="Monthly loans vs repayments", x=0.02, xanchor="left"),
            xaxis_title="Month",
            yaxis_title="Amount (£)",
            barmode='relative',
            height=440,
            legend=LEGEND_BELOW,
            margin=dict(t=56, b=108, l=56, r=36),
        )
        
        charts['monthly_comparison'] = fig_monthly
    
    # 2. Loans by Lender (Top 10)
    if not analysis['loans_by_lender'].empty:
        top_lenders = analysis['loans_by_lender'].head(10)
        
        fig_lenders = go.Figure(data=[
            go.Bar(
                x=top_lenders['sum'],
                y=[name.title()[:42] + '…' if len(name) > 42 else name.title() for name in top_lenders['lender_clean']],
                orientation='h',
                marker_color='lightcoral',
                text=[f"£{amount:,.0f}" for amount in top_lenders['sum']],
                textposition='auto'
            )
        ])
        
        fig_lenders.update_layout(
            title=dict(text="Loans by lender (top 10)", x=0.02, xanchor="left"),
            xaxis_title="Total Amount (£)",
            yaxis_title="Lender",
            height=440,
            yaxis=dict(autorange="reversed"),
            margin=dict(t=56, b=56, l=56, r=36),
        )
        fig_lenders.update_yaxes(automargin=True, tickfont=dict(size=11))

        charts['loans_by_lender'] = fig_lenders
    
    # 3. Repayments by Recipient (Top 10)
    if not analysis['repayments_by_recipient'].empty:
        top_recipients = analysis['repayments_by_recipient'].head(10)
        
        fig_recipients = go.Figure(data=[
            go.Bar(
                x=top_recipients['sum'],
                y=[name.title()[:42] + '…' if len(name) > 42 else name.title() for name in top_recipients['recipient_clean']],
                orientation='h',
                marker_color='lightblue',
                text=[f"£{amount:,.0f}" for amount in top_recipients['sum']],
                textposition='auto'
            )
        ])
        
        fig_recipients.update_layout(
            title=dict(text="Repayments by recipient (top 10)", x=0.02, xanchor="left"),
            xaxis_title="Total Amount (£)",
            yaxis_title="Recipient",
            height=440,
            yaxis=dict(autorange="reversed"),
            margin=dict(t=56, b=56, l=56, r=36),
        )
        fig_recipients.update_yaxes(automargin=True, tickfont=dict(size=11))

        charts['repayments_by_recipient'] = fig_recipients
    
    # 4. Cumulative Borrowing Position
    if not analysis['monthly_net_borrowing'].empty:
        monthly_data = analysis['monthly_net_borrowing'].copy()
        monthly_data['cumulative_borrowing'] = monthly_data['net_borrowing'].cumsum()
        
        fig_cumulative = go.Figure()
        
        fig_cumulative.add_trace(go.Scatter(
            x=monthly_data['month_str'],
            y=monthly_data['cumulative_borrowing'],
            mode='lines+markers',
            fill='tonexty',
            name='Cumulative Net Borrowing',
            line=dict(color='purple', width=3),
            marker=dict(size=8)
        ))
        
        fig_cumulative.add_hline(y=0, line_dash="dash", line_color="#64748b", opacity=0.7)
        
        fig_cumulative.update_layout(
            title=dict(text="Cumulative net borrowing position", x=0.02, xanchor="left"),
            xaxis_title="Month",
            yaxis_title="Cumulative Amount (£)",
            height=440,
            margin=dict(t=56, b=56, l=56, r=36),
        )
        
        charts['cumulative_borrowing'] = fig_cumulative
    
    return charts

def display_loans_repayments_section(df, analysis_period):
    """Display the complete loans and repayments analysis section"""
    st.markdown("---")
    st.subheader("Loans and debt repayments analysis")
    
    # Filter data by period if needed
    filtered_df = filter_data_by_period(df, analysis_period)
    
    # Perform analysis
    analysis = analyze_loans_and_repayments(filtered_df)
    manual_debt_total = get_manual_outstanding_debt_total()
    manual_debt_count = get_manual_outstanding_debt_count()
    analysis = apply_manual_outstanding_debt_to_loans_analysis(analysis)
    total_known_borrowing = analysis.get('total_known_borrowing', analysis['total_loans_received'])
    
    # Key Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Loans / Known Balances" if manual_debt_total > 0 else "Total Loans Received",
            f"£{total_known_borrowing:,.0f}",
            delta=f"£{manual_debt_total:,.0f} entered balances" if manual_debt_total > 0 else None,
            help=(
                f"£{analysis['total_loans_received']:,.0f} visible loan credits, "
                f"£{analysis['total_repayments_made']:,.0f} repayments already seen, and "
                f"£{manual_debt_total:,.0f} underwriter-entered outstanding balances"
                if manual_debt_total > 0
                else f"From {analysis['loan_count']} loan transactions"
            )
        )
    
    with col2:
        st.metric(
            "Total Repayments Made", 
            f"£{analysis['total_repayments_made']:,.0f}",
            help=f"From {analysis['repayment_count']} repayment transactions"
        )
    
    with col3:
        net_borrowing = analysis['net_borrowing']
        display_net_borrowing = analysis.get('known_outstanding_balance', manual_debt_total) if manual_debt_total > 0 else abs(net_borrowing)
        st.metric(
            "Known Outstanding Balance" if manual_debt_total > 0 else "Net Borrowing Position",
            f"£{display_net_borrowing:,.0f}",
            delta="Confirmed outstanding" if manual_debt_total > 0 else "Outstanding" if net_borrowing > 0 else "Net Repaid" if net_borrowing < 0 else "Balanced"
        )
    
    with col4:
        repayment_ratio_raw = analysis.get('repayment_ratio', 0)
        if repayment_ratio_raw is not None:
            repayment_ratio = repayment_ratio_raw * 100
            st.metric(
                "Repayment Ratio", 
                f"{repayment_ratio:.1f}%",
                help="Percentage of loans that have been repaid"
            )
        else:
            st.metric(
                "Repayment Ratio", 
                "N/A",
                help="Percentage of loans that have been repaid"
            )
    
    with col5:
        avg_loan = analysis['avg_loan_amount']
        if manual_debt_total > 0 and analysis['loan_count'] == 0:
            avg_loan = manual_debt_total / max(manual_debt_count, 1)
        st.metric(
            "Average Known Balance" if manual_debt_total > 0 and analysis['loan_count'] == 0 else "Average Loan Amount",
            f"£{avg_loan:,.0f}" if avg_loan > 0 else "N/A"
        )
    
    # Risk Assessment Row
    st.markdown("### Risk assessment")
    risk_col1, risk_col2, risk_col3 = st.columns(3)
    
    with risk_col1:
        if manual_debt_total > 0 and analysis['loan_count'] == 0 and analysis['repayment_count'] > 0:
            st.success("**Existing borrowing confirmed** — repayments plus entered balances are included in assessment")
        elif analysis['loan_count'] == 0 and analysis['repayment_count'] > 0:
            st.warning("**Possible existing borrowing** — repayments found but no loan credit appears in this bank data")
        elif analysis['loan_count'] == 0:
            st.info("**No visible external debt** — no loan credits or repayments found in this bank data")
        elif analysis['repayment_ratio'] >= 0.8:
            st.success("**Good repayment behavior** — consistently repays debt obligations")
        elif analysis['repayment_ratio'] >= 0.5:
            st.warning("**Moderate debt management** — some outstanding obligations")
        else:
            st.error("**High debt risk** — low repayment ratio indicates potential issues")
    
    with risk_col2:
        if analysis['loan_count'] == 0 and analysis['repayment_count'] > 0:
            st.warning("**Borrowing not fully visible** — loan may pre-date this period or sit in another account")
        elif analysis['loan_count'] == 0:
            st.info("**No borrowing history** — cannot assess borrowing patterns")
        elif analysis['loan_count'] <= 3:
            st.success("**Conservative borrowing** — infrequent use of external financing")
        elif analysis['loan_count'] <= 10:
            st.warning("**Moderate borrowing** — regular use of external financing")
        else:
            st.error("**High borrowing frequency** — heavy reliance on external financing")
    
    with risk_col3:
        if manual_debt_total > 0:
            st.success("**Outstanding balance included** — saved balances are included in ratio, charts, and scoring")
        elif analysis['loan_count'] == 0 and analysis['repayment_count'] > 0:
            st.warning("**Outstanding balance required** — enter known balances below, then rerun the application")
        elif analysis['net_borrowing'] <= 0:
            st.success("**Positive net position** — more repaid than borrowed")
        elif analysis['net_borrowing'] <= analysis['total_loans_received'] * 0.3:
            st.info("**Manageable outstanding** — reasonable debt burden")
        else:
            st.warning("**High outstanding debt** — significant borrowing position")
    
    # Charts Section
    if analysis['loan_count'] > 0 or analysis['repayment_count'] > 0:
        st.markdown("### Borrowing and repayment patterns")
        
        charts = create_loans_repayments_charts(analysis)
        
        # Row 1: Monthly comparison and cumulative position
        if 'monthly_comparison' in charts and 'cumulative_borrowing' in charts:
            col1, col2 = st.columns(2)
            with col1:
                show_mca_plotly(charts['monthly_comparison'], key="loans_monthly_comparison")
            with col2:
                show_mca_plotly(charts['cumulative_borrowing'], key="loans_cumulative")
        
        # Row 2: Lender and recipient analysis
        chart_row2_col1, chart_row2_col2 = st.columns(2)
        
        if 'loans_by_lender' in charts and charts['loans_by_lender'] is not None:
            with chart_row2_col1:
                show_mca_plotly(charts['loans_by_lender'], key="loans_by_lender")
        else:
            with chart_row2_col1:
                possible_lenders = analysis.get('possible_lenders_from_repayments', pd.DataFrame())
                if possible_lenders.empty:
                    st.info("No loan data available for lender analysis")
                else:
                    st.warning("No loan credits found. Repayments below may indicate existing borrowing outside the visible bank data.")
                    render_manual_outstanding_debt_form(possible_lenders, "possible_lenders_manual_debt")
        
        if 'repayments_by_recipient' in charts and charts['repayments_by_recipient'] is not None:
            with chart_row2_col2:
                show_mca_plotly(charts['repayments_by_recipient'], key="repayments_by_recipient")
        else:
            with chart_row2_col2:
                st.info("No repayment data available for recipient analysis")

        possible_lenders = analysis.get('possible_lenders_from_repayments', pd.DataFrame())
        if 'loans_by_lender' in charts and charts['loans_by_lender'] is not None and not possible_lenders.empty:
            unmatched_possible_lenders = possible_lenders[possible_lenders['loan_credit_seen'] == False].copy()
            if not unmatched_possible_lenders.empty:
                st.warning(
                    "Additional possible lenders found from repayments with no matching loan credit. "
                    "Enter confirmed outstanding balances, then rerun the application."
                )
                render_manual_outstanding_debt_form(unmatched_possible_lenders, "possible_additional_lenders_manual_debt")
    
    # Detailed Breakdown Tables
    with st.expander("Detailed loan and repayment breakdown", expanded=False):
        tab1, tab2, tab3 = st.tabs(["Loans Received", "Repayments Made", "Monthly Summary"])
        
        with tab1:
            if not analysis['loans_by_lender'].empty:
                st.write("**Loans by Lender:**")
                display_loans = analysis['loans_by_lender'].copy()
                display_loans['lender_clean'] = display_loans['lender_clean'].str.title()
                display_loans.columns = ['Lender', 'Number of Loans', 'Total Amount (£)']
                display_loans['Total Amount (£)'] = display_loans['Total Amount (£)'].apply(lambda x: f"£{x:,.2f}")
                st.dataframe(display_loans, use_container_width=True, hide_index=True)
            else:
                st.info("No loan transactions found in the selected period.")
                possible_lenders = analysis.get('possible_lenders_from_repayments', pd.DataFrame())
                if not possible_lenders.empty:
                    st.write("**Possible lenders from repayments:**")
                    display_possible = possible_lenders.copy()
                    display_possible['possible_lender'] = display_possible['possible_lender'].str.title()
                    display_possible = display_possible[
                        ['possible_lender', 'repayment_count', 'total_repaid_in_period', 'review_reason']
                    ]
                    display_possible.columns = ['Possible Lender', 'Repayments', 'Repaid in Period (£)', 'Review Reason']
                    display_possible['Repaid in Period (£)'] = display_possible['Repaid in Period (£)'].apply(lambda x: f"£{x:,.2f}")
                    st.dataframe(display_possible, use_container_width=True, hide_index=True)
        
        with tab2:
            if not analysis['repayments_by_recipient'].empty:
                st.write("**Repayments by Recipient:**")
                display_repayments = analysis['repayments_by_recipient'].copy()
                display_repayments['recipient_clean'] = display_repayments['recipient_clean'].str.title()
                display_repayments.columns = ['Recipient', 'Number of Repayments', 'Total Amount (£)']
                display_repayments['Total Amount (£)'] = display_repayments['Total Amount (£)'].apply(lambda x: f"£{x:,.2f}")
                st.dataframe(display_repayments, use_container_width=True, hide_index=True)
            else:
                st.info("No repayment transactions found in the selected period.")
        
        with tab3:
            if not analysis['monthly_net_borrowing'].empty:
                st.write("**Monthly Borrowing Summary:**")
                display_monthly = analysis['monthly_net_borrowing'].copy()
                display_monthly['loans'] = display_monthly['loans'].apply(lambda x: f"£{x:,.2f}")
                display_monthly['repayments'] = display_monthly['repayments'].apply(lambda x: f"£{x:,.2f}")
                display_monthly['net_borrowing'] = display_monthly['net_borrowing'].apply(
                    lambda x: f"£{x:,.2f}" if x >= 0 else f"-£{abs(x):,.2f}"
                )
                display_monthly = display_monthly[['month_str', 'loans', 'repayments', 'net_borrowing']]
                display_monthly.columns = ['Month', 'Loans Received', 'Repayments Made', 'Net Borrowing']
                st.dataframe(display_monthly, use_container_width=True, hide_index=True)
            else:
                st.info("No loan or repayment data found for monthly analysis.")
    
    return analysis
class DashboardExporter:
    """Safe dashboard export system integrated directly into main.py"""
    
    def __init__(self):
        self.export_timestamp = datetime.now()
    
    def export_dashboard_data(
        self, 
        company_name: str,
        params: dict,
        metrics: dict, 
        scores: dict,
        analysis_period: str,
        revenue_insights: dict,
        loans_analysis: dict = None
    ) -> dict:
        """Prepare all dashboard data for export."""
        
        export_data = {
            'export_info': {
                'company_name': company_name,
                'export_timestamp': self.export_timestamp.isoformat(),
                'analysis_period': analysis_period,
                'generated_by': 'Business Finance Scorecard v2.0'
            },
            'business_parameters': {
                'industry': params.get('industry'),
                'requested_loan': params.get('requested_loan'),
                'directors_score': params.get('directors_score'),
                'company_age_months': params.get('company_age_months'),

                # TU director outputs
                'tu_director_score': params.get('tu_director_score'),
                'tu_director_decision': params.get('tu_director_decision'),
                'tu_director_flags': params.get('tu_director_flags', []),
                'tu_director_reasons': params.get('tu_director_reasons', []),

                # Optional: overall decision
                'overall_decision': params.get('overall_decision'),
                'mca_main_decision': params.get('mca_main_decision'),
                'primary_account_assessment': params.get('primary_account_assessment'),

            },
            'financial_metrics': metrics,
            'scoring_results': {
                'subprime_score': scores.get('subprime_score'),
                'subprime_tier': scores.get('subprime_tier'),
                'subprime_recommendation': scores.get('subprime_recommendation'),
                'mca_rule_score': scores.get('mca_rule_score', params.get('mca_rule_score', 0)),
                'ml_score': scores.get('ml_score'),
                'adjusted_ml_score': scores.get('adjusted_ml_score'),
                'industry_score': scores.get('industry_score'),
                'loan_risk': scores.get('loan_risk'),
                'primary_account_assessment': scores.get('primary_account_assessment', params.get('primary_account_assessment'))
            },
            'revenue_insights': revenue_insights,
            'loans_analysis': loans_analysis or {}
        }
        
        return export_data
    
    def generate_html_report(self, export_data: dict) -> str:
        """Generate comprehensive HTML report."""

        sr = export_data.get("scoring_results", {}) or {}

        def get_score_class(score):
            if score is None:
                return "low"
            try:
                score = float(score)
            except (TypeError, ValueError):
                return "low"
            if score >= 70:
                return "high"
            if score >= 40:
                return "medium"
            return "low"

        def format_score(score, suffix="/100", precision=1):
            if score is None:
                return "N/A"
            try:
                val = float(score)
            except (TypeError, ValueError):
                return "N/A"
            if precision == 0:
                return f"{val:.0f}{suffix}"
            return f"{val:.{precision}f}{suffix}"

        subprime_raw = sr.get("subprime_score")
        mca_raw = sr.get("mca_rule_score")
        adjusted_raw = sr.get("adjusted_ml_score")
        if adjusted_raw is None:
            adjusted_raw = sr.get("ml_score")
        ml_display_raw = sr.get("adjusted_ml_score")
        if ml_display_raw is None:
            ml_display_raw = sr.get("ml_score")

        subprime_display = format_score(subprime_raw)
        mca_display = format_score(mca_raw, precision=0)
        adjusted_display = format_score(adjusted_raw, suffix="%")
        ml_table_display = (
            format_score(ml_display_raw, suffix="%")
            if ml_display_raw is not None
            else "N/A"
        )
        # Generate loans section HTML if data exists
        loans_section = ""
        if export_data['loans_analysis'] and export_data['loans_analysis'].get('loan_count', 0) > 0:
            loans_section = f"""
            <div class="section">
                <h2>Loans &amp; debt analysis</h2>
                <div class="metric-grid">
                    <div class="metric-card">
                        <h4>Total Loans Received</h4>
                        <div>£{export_data['loans_analysis'].get('total_loans_received', 0):,.0f}</div>
                        <p>{export_data['loans_analysis'].get('loan_count', 0)} transactions</p>
                    </div>
                    <div class="metric-card">
                        <h4>Total Repayments</h4>
                        <div>£{export_data['loans_analysis'].get('total_repayments_made', 0):,.0f}</div>
                        <p>{export_data['loans_analysis'].get('repayment_count', 0)} transactions</p>
                    </div>
                    <div class="metric-card">
                        <h4>Net Borrowing</h4>
                        <div>£{export_data['loans_analysis'].get('net_borrowing', 0):,.0f}</div>
                    </div>
                    <div class="metric-card">
                        <h4>Repayment Ratio</h4>
                        <div>{(export_data['loans_analysis'].get('repayment_ratio') or 0)*100:.1f}%</div>
                    </div>
                </div>
            </div>
            """

        # ---- Safe access (prevents KeyError if risk_factors missing) ----
        bp = export_data.get("business_parameters", {}) or {}
        rf = bp.get("risk_factors", {}) or {}

        business_ccj = bool(rf.get("business_ccj", False))
        poor_online = bool(rf.get("poor_or_no_online_presence", False))
        generic_email = bool(rf.get("uses_generic_email", False))

        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Business Finance Scorecard Report - {export_data['export_info']['company_name']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
                .section {{ margin-bottom: 30px; padding: 15px; border-left: 4px solid #007bff; }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
                .metric-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }}
                .score-high {{ color: #28a745; font-weight: bold; }}
                .score-medium {{ color: #ffc107; font-weight: bold; }}
                .score-low {{ color: #dc3545; font-weight: bold; }}
                .table-responsive {{ overflow-x: auto; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f8f9fa; font-weight: bold; }}
                .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #6c757d; }}
            </style>
        </head>
        <body>
            <!-- Header Section -->
            <div class="header">
                <h1>Business Finance Scorecard report</h1>
                <h2>{export_data['export_info']['company_name']}</h2>
                <p><strong>Generated:</strong> {datetime.fromisoformat(export_data['export_info']['export_timestamp']).strftime('%B %d, %Y at %I:%M %p')}</p>
                <p><strong>Analysis Period:</strong> {export_data['export_info']['analysis_period']}</p>
                <p><strong>Industry:</strong> {export_data['business_parameters']['industry']}</p>
            </div>
            
            <!-- Executive Summary -->
            <div class="section">
                <h2>Executive summary</h2>
                <div class="metric-grid">
                    <div class="metric-card">
                        <h3>Subprime score</h3>
                        <div class="score-{get_score_class(subprime_raw)}">{subprime_display}</div>
                        <p>{export_data['scoring_results']['subprime_tier']}</p>
                    </div>
                    <div class="metric-card">
                        <h3>MCA rule (60%)</h3>
                        <div class="score-{get_score_class(mca_raw)}">{mca_display}</div>
                    </div>
                    <div class="metric-card">
                        <h3>ML score (informational)</h3>
                        <div class="score-{get_score_class(adjusted_raw)}">{adjusted_display}</div>
                    </div>
                    <div class="metric-card">
                        <h3>Requested loan</h3>
                        <div>£{export_data['business_parameters']['requested_loan']:,.0f}</div>
                        <p>{export_data['scoring_results']['loan_risk']}</p>
                    </div>
                    <div class="metric-card">
                        <h3>MCA rule</h3>
                        <div>{export_data['scoring_results'].get('mca_rule_decision', 'N/A')}</div>
                        <p>Score: {export_data['scoring_results'].get('mca_rule_score', 'N/A')}</p>
                    </div>
                </div>
                
                <h3>Primary recommendation</h3>
                <p><strong>{export_data['scoring_results']['subprime_recommendation']}</strong></p>

                <h3>MCA rule decision (transparent)</h3>
                <p><strong>{export_data['scoring_results'].get('mca_rule_decision', 'N/A')}</strong>
                &nbsp;&nbsp;|&nbsp;&nbsp; Score: <strong>{export_data['scoring_results'].get('mca_rule_score', 'N/A')}</strong></p>

                <p><strong>Reasons:</strong></p>
                <ul>
                {''.join([f"<li>{r}</li>" for r in export_data['scoring_results'].get('mca_rule_reasons', [])])}
                </ul>

                <h3>Decision stack summary</h3>
                <table>
                    <tr><th>Layer</th><th>Result</th></tr>
                    <tr>
                        <td><strong>FINAL Decision</strong></td>
                        <td><strong>{export_data['scoring_results'].get('final_decision', 'N/A')}</strong></td>
                    </tr>
                    <tr>
                        <td>MCA Rule</td>
                        <td><strong>{export_data['scoring_results'].get('mca_rule_decision', 'N/A')}</strong>
                        &nbsp;&nbsp;|&nbsp;&nbsp; Score: {export_data['scoring_results'].get('mca_rule_score', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>Subprime (Existing)</td>
                        <td><strong>{export_data['scoring_results'].get('subprime_recommendation', 'N/A')}</strong>
                        &nbsp;&nbsp;|&nbsp;&nbsp; Tier: {export_data['scoring_results'].get('subprime_tier', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>MCA Rule (60%)</td>
                        <td>{export_data['scoring_results'].get('mca_rule_score', 'N/A')}/100</td>
                    </tr>
                    <tr>
                        <td>ML Score (Info Only)</td>
                        <td>{ml_table_display}</td>
                    </tr>
                    <tr>
                        <td>Requested Loan</td>
                        <td>£{export_data['business_parameters'].get('requested_loan', 0):,.0f}</td>
                    </tr>
                </table>

                </div>

            
            <!-- Financial Metrics -->
            <div class="section">
                <h2>Financial performance</h2>
                <div class="table-responsive">
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
                        <tr><td>Total Revenue</td><td>£{export_data['financial_metrics'].get('Total Revenue', 0):,.2f}</td></tr>
                        <tr><td>Monthly Average Revenue</td><td>£{export_data['financial_metrics'].get('Monthly Average Revenue', 0):,.2f}</td></tr>
                        <tr><td>Net Income</td><td>£{export_data['financial_metrics'].get('Net Income', 0):,.2f}</td></tr>
                        <tr><td>Operating Margin</td><td>{export_data['financial_metrics'].get('Operating Margin', 0)*100:.1f}%</td></tr>
                        <tr><td>Revenue Growth Rate</td><td>{export_data['financial_metrics'].get('Revenue Growth Rate', 0)*100:.1f}%</td></tr>
                        <tr><td>Debt Service Coverage Ratio</td><td>{export_data['financial_metrics'].get('Debt Service Coverage Ratio', 0):.2f}</td></tr>
                        <tr><td>Cash Flow Volatility</td><td>{export_data['financial_metrics'].get('Cash Flow Volatility', 0):.3f}</td></tr>
                        <tr><td>Average Month-End Balance</td><td>£{export_data['financial_metrics'].get('Average Month-End Balance', 0):,.2f}</td></tr>
                    </table>
                </div>
            </div>
            
            <!-- Revenue Analysis -->
            <div class="section">
                <h2>Revenue insights</h2>
                <div class="metric-grid">
                    <div class="metric-card">
                        <h4>Revenue Sources</h4>
                        <div>{export_data['revenue_insights'].get('unique_revenue_sources', 0)}</div>
                    </div>
                    <div class="metric-card">
                        <h4>Avg Daily Revenue</h4>
                        <div>£{export_data['revenue_insights'].get('avg_daily_revenue_amount', 0):,.2f}</div>
                    </div>
                    <div class="metric-card">
                        <h4>Revenue Active Days</h4>
                        <div>{export_data['revenue_insights'].get('total_revenue_days', 0)}</div>
                    </div>
                    <div class="metric-card">
                        <h4>Transactions/Day</h4>
                        <div>{export_data['revenue_insights'].get('avg_revenue_transactions_per_day', 0):.1f}</div>
                    </div>
                </div>
            </div>
            
            <!-- Loans Analysis -->
            {loans_section}
            
            <!-- Risk Factors -->
            <div class="section">
                <h2>Risk factors assessment</h2>
                <div class="table-responsive">
                    <table>
                        <tr><th>Risk Factor</th><th>Status</th></tr>
                        <tr><td>Business CCJs</td><td>{'Yes' if business_ccj else 'No'}</td></tr>
                        <tr><td>Poor/No Online Presence</td><td>{'Yes' if poor_online else 'No'}</td></tr>
                        <tr><td>Generic Email</td><td>{'Yes' if generic_email else 'No'}</td></tr>
                    </table>
                </div>
            </div>
            
            <!-- Business Parameters -->
            <div class="section">
                <h2>Business information</h2>
                <div class="table-responsive">
                    <table>
                        <tr><th>Parameter</th><th>Value</th></tr>
                        <tr><td>Company Name</td><td>{export_data['export_info']['company_name']}</td></tr>
                        <tr><td>Industry</td><td>{export_data['business_parameters']['industry']}</td></tr>
                        <tr><td>Company Age</td><td>{export_data['business_parameters']['company_age_months']} months</td></tr>
                        <tr><td>Directors Score</td><td>{export_data['business_parameters']['directors_score']}/100</td></tr>
                        <tr><td>Requested Loan Amount</td><td>£{export_data['business_parameters']['requested_loan']:,.0f}</td></tr>
                    </table>
                </div>
            </div>
            
            <!-- Footer -->
            <div class="footer">
                <p><strong>Report Generated by:</strong> {export_data['export_info']['generated_by']}</p>
                <p><strong>Disclaimer:</strong> This report is for informational purposes only and should not be considered as financial advice. 
                All lending decisions should involve comprehensive due diligence and risk assessment.</p>
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def create_export_buttons(
        self,
        company_name: str,
        params: dict,
        metrics: dict, 
        scores: dict,
        analysis_period: str,
        revenue_insights: dict,
        loans_analysis: dict = None
    ) -> None:
        """Create export buttons in Streamlit interface."""
        
        st.markdown("---")
        st.subheader("Export dashboard report")
        
        # Prepare export data
        export_data = self.export_dashboard_data(
            company_name, params, metrics, scores, 
            analysis_period, revenue_insights, loans_analysis
        )
        
        # Create columns for export buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # HTML Export
            html_report = self.generate_html_report(export_data)
            st.download_button(
                label="Export HTML report",
                data=html_report,
                file_name=f"{company_name.replace(' ', '_')}_financial_report_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
                mime="text/html",
                help="Download comprehensive HTML report (opens in any browser)",
                type="primary"
            )
        
        with col2:
            # JSON Export
            json_data = json.dumps(export_data, indent=2, default=str)
            st.download_button(
                label="Export JSON data",
                data=json_data,
                file_name=f"{company_name.replace(' ', '_')}_data_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                help="Download all data in JSON format for further analysis"
            )
        
        with col3:
            # CSV Export (Financial Metrics)
            metrics_df = pd.DataFrame([
                {'Metric': k, 'Value': v} for k, v in metrics.items() 
                if k != 'monthly_summary' and isinstance(v, (int, float))
            ])
            csv_data = metrics_df.to_csv(index=False)
            st.download_button(
                label="Export CSV metrics",
                data=csv_data,
                file_name=f"{company_name.replace(' ', '_')}_metrics_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                help="Download financial metrics as CSV"
            )
        
        # Export information
        st.info("""
        **Export options**
        - **HTML report** — full dashboard in a web page you can open in any browser
        - **JSON data** — all fields for analysis or integration
        - **CSV metrics** — key numbers for spreadsheet work

        **Includes:** scoring results, financial metrics, revenue insights, risk factors, loans analysis, and business parameters.
        """)

def _normalise_mca_decision_for_overlay(d: str | None) -> str:
    d = (d or "REFER").upper()
    if d in {"CONDITIONAL_APPROVE"}:
        return "APPROVE"
    if d in {"SENIOR_REVIEW"}:
        return "REFER"
    if d not in {"APPROVE", "REFER", "DECLINE"}:
        return "REFER"
    return d

# -----------------------------
# 4A) Decision combiner (TU × MCA)
# -----------------------------
def combine_mca_and_tu_decisions(mca_decision: str | None, tu_decision: str | None) -> str:
    """
    Overlay TU on top of MCA ensemble decision.

    Policy:
      - If MCA is DECLINE -> final DECLINE
      - If MCA is REFER/SENIOR_REVIEW -> final REFER
      - If MCA is APPROVE/CONDITIONAL_APPROVE:
            TU DECLINE -> final REFER
            TU APPROVE -> keep MCA decision (APPROVE or CONDITIONAL_APPROVE)
            TU missing/REFER -> REFER
    """
    raw_mca = (mca_decision or "REFER").upper()
    mca_norm = _normalise_mca_decision_for_overlay(raw_mca)
    tu = (tu_decision or "").upper()

    if mca_norm == "DECLINE":
        return "DECLINE"
    if mca_norm == "REFER":
        return "REFER"

    # MCA is APPROVE (including CONDITIONAL_APPROVE normalised)
    if tu == "DECLINE":
        return "REFER"
    if tu == "APPROVE":
        # keep conditional if that’s what ensemble said
        return "CONDITIONAL_APPROVE" if raw_mca == "CONDITIONAL_APPROVE" else "APPROVE"

    return "REFER"


def build_evidence_quality(params: dict, scores: dict, df: pd.DataFrame | None = None) -> list[dict[str, str]]:
    """Summarise whether the main evidence sources were present and usable."""
    row_count = len(df) if df is not None else 0
    tu_score = params.get("tu_director_score")
    tu_status = params.get("tu_parse_status", "parsed" if tu_score is not None else "missing")
    bureau_band = params.get("bureau_band")
    bureau_status = params.get("business_bureau_parse_status", "parsed" if bureau_band else "missing")
    evidence = [
        {
            "evidence": "Bank transactions",
            "status": "Present" if row_count else "Missing",
            "detail": f"{row_count:,} transaction rows" if row_count else "No transaction JSON processed",
        },
        {
            "evidence": "MCA rule signals",
            "status": "Present" if params.get("mca_rule_score") is not None else "Missing",
            "detail": f"Score {params.get('mca_rule_score')} / {params.get('mca_rule_decision')}" if params.get("mca_rule_score") is not None else "No MCA rule output",
        },
        {
            "evidence": "Director TU XML",
            "status": "Present" if tu_status == "parsed" else ("Failed" if tu_status == "failed" else "Missing"),
            "detail": (
                f"Score {tu_score} / {params.get('tu_director_decision')}"
                if tu_status == "parsed"
                else (params.get("tu_parse_error") or "No director TU XML parsed")
            ),
        },
        {
            "evidence": "Business credit PDF",
            "status": "Present" if bureau_status == "parsed" else ("Failed" if bureau_status == "failed" else "Missing"),
            "detail": (
                f"{bureau_band}"
                if bureau_status == "parsed" and bureau_band
                else (params.get("business_bureau_parse_error") or "No business credit PDF parsed")
            ),
        },
        {
            "evidence": "Business bureau score",
            "status": "Suppressed" if params.get("business_credit_score_suppressed") else ("Present" if params.get("business_credit_score_range") or params.get("business_credit_score") is not None else "Missing"),
            "detail": (
                "Score suppressed by bureau"
                if params.get("business_credit_score_suppressed")
                else (
                    f"Score {params.get('business_credit_score_range')}"
                    if params.get("business_credit_score_range")
                    else (f"Score {params.get('business_credit_score')}" if params.get("business_credit_score") is not None else "No usable bureau score")
                )
            ),
        },
    ]

    if scores.get("ensemble"):
        evidence.append(
            {
                "evidence": "Decision engine",
                "status": "Present",
                "detail": f"Confidence {scores['ensemble'].get('confidence', 0):.0f}%",
            }
        )
    return evidence


def build_score_impact_rows(params: dict, metrics: dict, scores: dict) -> list[dict[str, object]]:
    """Create a plain-English score impact table for the underwriting UI."""
    ensemble = scores.get("ensemble") or {}
    detailed = ensemble.get("detailed_breakdown", {}) or {}
    contributing = ensemble.get("contributing_scores", {}) or {}
    rows = []

    raw_combined = detailed.get("raw_combined_score")
    combined = ensemble.get("combined_score")
    if contributing:
        rows.append(
            {
                "component": "MCA rule",
                "value": contributing.get("mca_score", params.get("mca_rule_score")),
                "impact": "60% weighted input",
                "decision_effect": params.get("mca_rule_decision") or "N/A",
            }
        )
        rows.append(
            {
                "component": "Subprime score",
                "value": contributing.get("subprime_score", scores.get("subprime_score")),
                "impact": "40% weighted input",
                "decision_effect": scores.get("subprime_tier") or "N/A",
            }
        )
    if raw_combined is not None:
        rows.append(
            {
                "component": "Raw weighted score",
                "value": round(float(raw_combined), 1),
                "impact": "MCA/Subprime before disagreement penalty",
                "decision_effect": ensemble.get("score_convergence", "N/A"),
            }
        )
    if raw_combined is not None and combined is not None:
        penalty = round(float(raw_combined) - float(combined), 1)
        rows.append(
            {
                "component": "Convergence adjustment",
                "value": -penalty,
                "impact": "Penalty for score disagreement" if penalty else "No penalty",
                "decision_effect": ensemble.get("score_convergence", "N/A"),
            }
        )

    for penalty in params.get("_applied_risk_penalties", []) or []:
        rows.append(
            {
                "component": "Risk penalty",
                "value": "",
                "impact": penalty,
                "decision_effect": "Included in Subprime score",
            }
        )

    tu_decision = params.get("tu_director_decision")
    if tu_decision:
        rows.append(
            {
                "component": "Director TU overlay",
                "value": params.get("tu_director_score"),
                "impact": f"TU decision {tu_decision}",
                "decision_effect": "May cap approve to refer",
            }
        )

    bureau_band = params.get("bureau_band")
    if bureau_band:
        rows.append(
            {
                "component": "Business credit PDF",
                "value": bureau_band,
                "impact": "; ".join((params.get("bureau_band_reasons") or [])[:3]),
                "decision_effect": "Feeds bureau risk penalties",
            }
        )

    rows.append(
        {
            "component": "Final decision",
            "value": scores.get("final_decision") or ensemble.get("decision"),
            "impact": "After MCA/Subprime decision and TU overlay",
            "decision_effect": "Current displayed recommendation",
        }
    )
    return rows


def build_open_banking_insight_rows(metrics: dict, params: dict) -> list[dict[str, object]]:
    """Create a compact table of derived open-banking signals for review."""
    requested = float(params.get("requested_loan") or 0)
    monthly_revenue = float(metrics.get("Monthly Average Revenue") or 0)
    weakest_revenue = float(metrics.get("OB Weakest Month Revenue") or 0)
    rows = [
        ("Scoring impact", metrics.get("Open Banking Insights Used In Score", "No - analysis/export only"), "New derived fields are displayed/exported only"),
        ("History", f"{metrics.get('OB History Months', 0)} months / {metrics.get('OB Transaction Count', 0)} transactions", "Coverage and file depth"),
        ("True revenue", f"£{float(metrics.get('OB True Revenue', metrics.get('Total Revenue', 0)) or 0):,.0f}", "Revenue after categorisation"),
        ("Non-revenue inflows", f"{float(metrics.get('OB Non-Revenue Inflow Ratio', 0) or 0) * 100:.1f}%", "Transfers, funding injections, loans and other non-trading inflows"),
        ("Revenue concentration", f"{float(metrics.get('OB Top Revenue Source Percentage', 0) or 0):.1f}%", "Largest payer or processor share of revenue"),
        ("Card processor share", f"{float(metrics.get('OB Card Processor Revenue Share', 0) or 0) * 100:.1f}%", "Revenue from recognised card/payment processors"),
        ("Weakest month revenue", f"£{weakest_revenue:,.0f}", "Lowest observed trading revenue month"),
        ("Debt repayment burden", f"{float(metrics.get('OB Debt Repayment Burden', 0) or 0) * 100:.1f}%", "Debt repayments as share of trading revenue"),
        ("Recent loan credits", f"£{float(metrics.get('OB Recent Loan Credits 30D', 0) or 0):,.0f}", "Funding credits in the latest 30 days"),
        ("Low balance days", f"{int(metrics.get('OB Low Balance Days <1000', 0) or 0)} below £1k", "Daily balance pressure"),
        ("Failed payments 30D", int(metrics.get("OB Recent Failed Payments 30D", 0) or 0), "Recent returned or failed payment markers"),
    ]
    if requested > 0:
        rows.append(
            (
                "Requested amount cover",
                f"{requested / monthly_revenue:.2f}x monthly / {requested / weakest_revenue:.2f}x weakest"
                if monthly_revenue and weakest_revenue
                else "N/A",
                "Requested loan versus normal and weakest-month revenue",
            )
        )
    return [{"signal": name, "value": value, "meaning": meaning} for name, value, meaning in rows]


def build_card_processing_insight_rows(metrics: dict) -> list[dict[str, object]]:
    """Create a compact table of card processor statement signals for review."""
    if metrics.get("Card Processing Insight Layer") != "Available":
        return [
            {
                "signal": "Card processing insight layer",
                "value": metrics.get("Card Processing Insight Layer", "Not available"),
                "meaning": "Upload card terminal statements to derive these signals",
            }
        ]

    rows = [
        ("Scoring impact", metrics.get("Card Processing Insights Used In Score", "No - analysis/export only"), "Displayed/exported for underwriting and calibration"),
        ("Statements", f"{int(metrics.get('Card Processor Statements Parsed', 0) or 0)} files / {int(metrics.get('Card Processor Months Present', 0) or 0)} months", "Coverage from uploaded processor statements"),
        ("Card sales total", f"£{float(metrics.get('Card Sales Total', 0) or 0):,.0f}", "Gross card sales from statements"),
        ("Average card sales", f"£{float(metrics.get('Card Sales Monthly Average', 0) or 0):,.0f}", "Monthly card sales average"),
        ("Weakest card month", f"£{float(metrics.get('Card Weakest Month Sales', 0) or 0):,.0f}", "Lowest observed card sales month"),
        ("Card sales volatility", f"{float(metrics.get('Card Sales Volatility', 0) or 0) * 100:.1f}%", "Month-to-month card sales stability"),
        ("Latest month drop", f"{float(metrics.get('Card Latest Month Drop Pct', 0) or 0) * 100:.1f}%", "Latest card month versus prior average"),
        ("Refund ratio", f"{float(metrics.get('Card Refund Ratio', 0) or 0) * 100:.1f}%", "Refunds as share of gross card sales"),
        ("Chargeback ratio", f"{float(metrics.get('Card Chargeback Ratio', 0) or 0) * 100:.1f}%", "Chargebacks as share of gross card sales"),
        ("Fee ratio", f"{float(metrics.get('Card Fee Ratio', 0) or 0) * 100:.1f}%", "Processor fees as share of gross card sales"),
        ("Average transaction value", f"£{float(metrics.get('Card Average Transaction Value', 0) or 0):,.2f}", "Gross card sales divided by transaction count"),
        ("Card vs OB revenue", f"{float(metrics.get('Card vs OB Revenue Ratio', 0) or 0) * 100:.1f}%", "Card sales as share of open banking revenue evidence"),
        ("Unmatched card shortfall", f"£{float(metrics.get('Card Unmatched Sales Shortfall', 0) or 0):,.0f} / {float(metrics.get('Card Unmatched Sales Shortfall Pct', 0) or 0) * 100:.1f}%", "Card sales not covered by bank revenue in matching months"),
        ("Reconciliation quality", metrics.get("Card Reconciliation Quality", "N/A"), "Monthly card sales versus bank revenue"),
        ("MCA suitability", metrics.get("Card MCA Suitability", "N/A"), "Initial card-led underwriting view"),
    ]
    concerns = metrics.get("Card Processing Concerns") or []
    positives = metrics.get("Card Processing Positive Signals") or []
    if positives:
        rows.append(("Positive signals", "; ".join(positives[:4]), "Supportive card processor evidence"))
    if concerns:
        rows.append(("Concerns", "; ".join(concerns[:4]), "Items for underwriting review"))
    return [{"signal": name, "value": value, "meaning": meaning} for name, value, meaning in rows]


def derive_card_processing_payload(bank_df, card_files) -> dict:
    """Parse uploaded card statements once and return dataframes plus insights."""
    empty = {
        "parse_output": {},
        "parsed_df": pd.DataFrame(),
        "monthly_terminal": pd.DataFrame(),
        "comparison_payload": {"comparison": pd.DataFrame(), "summary": {}},
        "insights": {
            "Card Processing Insight Layer": "Not available",
            "Card Processing Insights Used In Score": "No - analysis/export only",
        },
        "error": None,
    }
    if not card_files:
        return empty
    if not CARD_TERMINAL_SERVICE_AVAILABLE or CardTerminalIngestionService is None:
        empty["error"] = "Card terminal ingestion service is currently unavailable."
        return empty

    try:
        service = CardTerminalIngestionService()
        parse_output = service.parse_uploaded_files(card_files)
        parsed_df = parse_output.get("dataframe", pd.DataFrame())
        monthly_terminal = service.summarize_by_month(parsed_df)
        comparison_payload = service.compare_with_banking_data(bank_df, monthly_terminal)
        insights = service.derive_card_processing_insights(parsed_df, monthly_terminal, comparison_payload)
        return {
            "parse_output": parse_output,
            "parsed_df": parsed_df,
            "monthly_terminal": monthly_terminal,
            "comparison_payload": comparison_payload,
            "insights": insights,
            "error": None,
        }
    except Exception as exc:
        empty["error"] = str(exc)
        return empty


def render_full_financial_dashboard(
    company_name: str,
    analysis_period: str,
    df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    params: dict,
    metrics: dict,
    scores: dict,
    revenue_insights: dict,
    card_terminal_files,
    card_processing_payload: dict | None = None,
) -> None:
    """Render the full main-page dashboard (shared by fresh upload and session restore)."""
    # ENHANCED DASHBOARD RENDERING
    period_label = f"Last {analysis_period} Months" if analysis_period != 'All' else "Full Period"
    st.header(f"Financial Dashboard: {company_name} ({period_label})")

    # ============================================
    # UNIFIED RECOMMENDATION (TOP OF DASHBOARD)
    # ============================================
    ensemble = scores.get('ensemble')
    if ensemble:
        decision = scores.get('final_decision') or ensemble.get('decision', 'REFER')
        ensemble_decision = str(ensemble.get('decision', '')).upper().strip()
        combined_score = ensemble.get('combined_score', 0)
        confidence = ensemble.get('confidence', 0)
        final_reasons = scores.get('final_decision_reasons', []) or []
        ensemble_reason = ensemble.get('primary_reason', '')

        # Keep reason text consistent with displayed decision.
        reason_text = ensemble_reason
        if ensemble_decision and decision != ensemble_decision:
            override_reason = final_reasons[-1] if final_reasons else (
                f"Final decision overridden from {ensemble_decision} to {decision} by policy overlay."
            )
            reason_text = f"{override_reason}\nEnsemble context: {ensemble_reason}"
        elif final_reasons:
            reason_text = final_reasons[-1]

        # Main recommendation display with prominent styling
        if decision == 'APPROVE':
            st.success(f"""
            ## Recommendation: APPROVE
            **Combined Score:** {combined_score:.1f}/100  |  **Confidence:** {confidence:.0f}%

            *{reason_text}*
            """)
        elif decision == 'CONDITIONAL_APPROVE':
            st.info(f"""
            ## Recommendation: CONDITIONAL APPROVE
            **Combined Score:** {combined_score:.1f}/100  |  **Confidence:** {confidence:.0f}%

            *{reason_text}*
            """)
        elif decision == 'REFER':
            st.warning(f"""
            ## Recommendation: REFER FOR REVIEW
            **Combined Score:** {combined_score:.1f}/100  |  **Confidence:** {confidence:.0f}%

            *{reason_text}*
            """)
        elif decision == 'SENIOR_REVIEW':
            st.warning(f"""
            ## Recommendation: SENIOR REVIEW REQUIRED
            **Combined Score:** {combined_score:.1f}/100  |  **Confidence:** {confidence:.0f}%

            *{reason_text}*
            """)
        else:  # DECLINE
            st.error(f"""
            ## Recommendation: DECLINE
            **Combined Score:** {combined_score:.1f}/100  |  **Confidence:** {confidence:.0f}%

            *{reason_text}*
            """)

        # Contributing scores in compact row
        contributing = ensemble.get('contributing_scores', {})
        score_cols = st.columns(3)

        with score_cols[0]:
            mca_s = contributing.get('mca_score', params.get('mca_rule_score', 50))
            st.metric("MCA Rule (60%)", f"{mca_s:.0f}")

        with score_cols[1]:
            subprime_s = contributing.get('subprime_score', scores.get('subprime_score', 0))
            st.metric("Subprime (40%)", f"{subprime_s:.1f}")

        with score_cols[2]:
            ml_info = ensemble.get('detailed_breakdown', {}).get('informational_scores', {})
            ml_s = ml_info.get('ml_score', scores.get('adjusted_ml_score') or scores.get('ml_score') or 0)
            st.metric("ML Score (Info Only)", f"{ml_s:.1f}%" if ml_s else "N/A")

        # Score convergence indicator
        convergence = ensemble.get('score_convergence', 'Unknown')
        if 'High' in convergence:
            st.success(f"**Score convergence:** {convergence} — MCA and Subprime agree")
        elif 'Good' in convergence:
            st.info(f"**Score convergence:** {convergence}")
        elif 'Moderate' in convergence:
            st.warning(
                f"**Score convergence:** {convergence} — some disagreement between MCA and Subprime"
            )
        else:
            st.error(f"**Score convergence:** {convergence} — significant disagreement")

        impact_rows = build_score_impact_rows(params, metrics, scores)
        if impact_rows:
            with st.expander("Score impact", expanded=True):
                st.dataframe(pd.DataFrame(impact_rows), use_container_width=True, hide_index=True)
                final_reasons = scores.get("final_decision_reasons", []) or []
                if final_reasons:
                    st.markdown("**Decision path:**")
                    for reason in final_reasons:
                        st.write(f"• {reason}")

        evidence_rows = build_evidence_quality(params, scores, filtered_df)
        if evidence_rows:
            with st.expander("Evidence quality", expanded=False):
                evidence_df = pd.DataFrame(evidence_rows)
                st.dataframe(evidence_df, use_container_width=True, hide_index=True)

        ob_rows = build_open_banking_insight_rows(metrics, params)
        with st.expander("Open banking derived insights", expanded=True):
            st.caption("These additional transaction-derived signals are shown for underwriting review and exports. They do not change the current score unless explicitly added to the scorecard later.")
            st.dataframe(pd.DataFrame(ob_rows), use_container_width=True, hide_index=True)

        card_rows = build_card_processing_insight_rows(metrics)
        with st.expander("Card processor derived insights", expanded=True):
            st.caption("These card-statement signals are shown for underwriting review and exports. They do not change the current score unless explicitly added to the scorecard later.")
            st.dataframe(pd.DataFrame(card_rows), use_container_width=True, hide_index=True)

        # Pricing and details in expander
        with st.expander("Pricing guidance & risk analysis", expanded=False):
            pricing = ensemble.get('pricing_guidance', {})
            if pricing and pricing.get('factor_rate') != 'N/A':
                pricing_cols = st.columns(4)
                with pricing_cols[0]:
                    st.write(f"**Factor Rate:** {pricing.get('factor_rate', 'N/A')}")
                with pricing_cols[1]:
                    st.write(f"**Max Term:** {pricing.get('max_term', 'N/A')}")
                with pricing_cols[2]:
                    st.write(f"**Max Amount:** {pricing.get('max_multiple', 'N/A')}")
                with pricing_cols[3]:
                    st.write(f"**Collection:** {pricing.get('collection_frequency', 'N/A')}")

            st.markdown("---")
            risk_col, positive_col = st.columns(2)

            with risk_col:
                st.markdown("**Risk factors:**")
                risk_factors = ensemble.get('risk_factors', [])
                if risk_factors:
                    for rf in risk_factors:
                        st.write(f"• {rf}")
                else:
                    st.write("• No significant risk factors identified")

            with positive_col:
                st.markdown("**Positive factors:**")
                positive_factors = ensemble.get('positive_factors', [])
                if positive_factors:
                    for pf in positive_factors:
                        st.write(f"• {pf}")
                else:
                    st.write("• No notable positive factors")

            st.markdown("---")
            st.markdown("**Recommendations:**")
            recommendations = ensemble.get('recommendations', [])
            for rec in recommendations:
                st.write(f"• {rec}")

            # MCA Rule signals (moved from standalone section)
            if "mca_rule_decision" in params:
                st.markdown("---")
                st.markdown("**MCA Rule Analysis:**")
                mca_r = params.get("mca_rule_reasons", [])
                for r in mca_r:
                    st.write(f"• {r}")

                # Show MCA signals in a compact format (no nested expander)
                mca_signals = params.get("mca_rule_signals", {})
                if mca_signals:
                    st.markdown("**MCA Rule Signals:**")
                    st.code(str(mca_signals), language="json")
    else:
        # Fallback if ensemble not available
        st.info("Unified ensemble scoring not available. Showing individual scores below.")

    primary_signal = params.get("primary_account_assessment", {})
    primary_note = primary_signal.get("note")
    if primary_note:
        if primary_signal.get("is_potential_non_primary", False):
            st.warning(f"**Primary account check (UW note):** {primary_note}")
        elif primary_signal.get("status") == "unable_to_determine":
            st.info(f"**Primary account check (UW note):** {primary_note}")
        else:
            st.success(f"**Primary account check (UW note):** {primary_note}")

        with st.expander("Primary Account Check Details", expanded=False):
            st.json(primary_signal)

    render_card_terminal_reconciliation(filtered_df, card_terminal_files, card_processing_payload)

    # Revenue Insights
    st.markdown("---")
    st.subheader("Revenue insights")

    rev_col1, rev_col2, rev_col3, rev_col4 = st.columns(4)
    with rev_col1:
        sources_count = revenue_insights.get('unique_revenue_sources', 0)
        st.metric("Unique Revenue Sources", f"{sources_count}")
        if sources_count == 1:
            st.warning("Single revenue source — consider diversification")
        elif sources_count <= 3:
            st.info("Limited revenue sources — moderate concentration risk")
        else:
            st.success("Good revenue diversification")
    with rev_col2:
        avg_txns = revenue_insights.get('avg_revenue_transactions_per_day', 0)
        st.metric("Avg Revenue Transactions/Day", f"{avg_txns:.1f}")
    with rev_col3:
        avg_daily_rev = revenue_insights.get('avg_daily_revenue_amount', 0)
        st.metric("Avg Daily Revenue", f"£{avg_daily_rev:,.2f}")
    with rev_col4:
        total_days = revenue_insights.get('total_revenue_days', 0)
        st.metric("Revenue Active Days", f"{total_days}")

    # Charts Section
    st.markdown("---")
    st.subheader("Charts & Analysis")

    # Row 1: Enhanced Score and Financial Charts
    col1, col2 = st.columns(2)
    with col1:
        fig_scores = create_score_charts(scores, metrics)
        show_mca_plotly(fig_scores, key="enhanced_scores_chart")
    with col2:
        fig_financial, fig_trend = create_financial_charts(metrics)
        show_mca_plotly(fig_financial, key="main_financial_chart")

    # Row 2: Trend and Threshold Charts
    col1, col2 = st.columns(2)
    with col1:
        if fig_trend:
            show_mca_plotly(fig_trend, key="main_trend_chart")
        else:
            st.info("Monthly trend requires multiple months of data")
    with col2:
        fig_threshold = create_threshold_chart(scores['score_breakdown'])
        show_mca_plotly(fig_threshold, key="main_threshold_chart")

    # Monthly Breakdown Section
    st.markdown("---")
    st.subheader("Monthly Breakdown by Category")

    pivot_counts, pivot_amounts = create_monthly_breakdown(filtered_df)

    if pivot_counts is not None and not pivot_counts.empty:
        # Create monthly breakdown charts
        fig_monthly_counts, fig_monthly_amounts = create_monthly_charts(pivot_counts, pivot_amounts)

        # Display charts
        col1, col2 = st.columns(2)
        with col1:
            show_mca_plotly(fig_monthly_counts, key="main_monthly_counts")
        with col2:
            show_mca_plotly(fig_monthly_amounts, key="main_monthly_amounts")

        # Monthly summary table
        with st.expander("Detailed Monthly Breakdown", expanded=False):
            tab1, tab2 = st.tabs(["Transaction Counts", "Transaction Amounts (£)"])

            with tab1:
                counts_display = pivot_counts.copy()
                counts_display.index = counts_display.index.astype(str)
                counts_display = counts_display.astype(int)
                st.dataframe(counts_display, use_container_width=True)

                # Add totals
                totals_counts = counts_display.sum()
                st.write("**Totals:**")
                total_cols = st.columns(len(totals_counts))
                for i, (cat, total) in enumerate(totals_counts.items()):
                    with total_cols[i]:
                        st.metric(cat, f"{total:,.0f}")

            with tab2:
                amounts_display = pivot_amounts.copy()
                amounts_display.index = amounts_display.index.astype(str)
                amounts_display = amounts_display.round(2)
                st.dataframe(amounts_display, use_container_width=True)

                # Add totals
                totals_amounts = amounts_display.sum()
                st.write("**Totals:**")
                total_cols = st.columns(len(totals_amounts))
                for i, (cat, total) in enumerate(totals_amounts.items()):
                    with total_cols[i]:
                        st.metric(cat, f"£{total:,.2f}")
    else:
        st.info("Monthly breakdown requires multiple months of data")

    # Transaction Category Analysis
    st.markdown("---")
    st.subheader("Transaction Analysis")

    categorized_data = categorize_transactions(filtered_df)
    category_summary = categorized_data['subcategory'].value_counts()

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Transaction Categories:**")
        for category, count in category_summary.items():
            category_amount = abs(categorized_data[categorized_data['subcategory'] == category]['amount'].sum())
            percentage = (count / len(categorized_data)) * 100
            st.write(f"• **{category}**: {count} transactions (£{category_amount:,.2f}) - {percentage:.1f}%")

    with col2:
        # Category pie chart
        fig_pie = go.Figure(data=[go.Pie(
            labels=category_summary.index,
            values=category_summary.values,
            hole=0.3,
            marker_colors=['#22c55e', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4']
        )])

        fig_pie.update_layout(
            title=dict(text="Transaction distribution", x=0.02, xanchor="left"),
            height=340,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(15, 23, 42, 0.85)",
                bordercolor="rgba(148, 163, 184, 0.35)",
                borderwidth=1,
            ),
            margin=dict(t=52, b=48, l=48, r=180),
        )
        fig_pie.update_traces(
            textposition="inside",
            textinfo="percent",
            insidetextfont=dict(color="#f8fafc", size=11),
            hovertemplate="<b>%{label}</b><br>%{percent}<br>£%{value:,.0f}<extra></extra>",
        )

        show_mca_plotly(fig_pie, key="main_category_pie")

    # NEW: Loans and Debt Repayments Analysis
    display_loans_repayments_section(filtered_df, analysis_period)

    # Detailed Metrics Table
    st.markdown("---")
    st.subheader("Detailed Financial Metrics")

    # Create metrics table
    metrics_data = []
    industry_thresholds = get_industry_thresholds(params['industry'])

    for metric, value in metrics.items():
        if metric in industry_thresholds and metric != 'monthly_summary':
            threshold = industry_thresholds[metric]
            if metric in ['Cash Flow Volatility', 'Average Negative Balance Days per Month', 'Number of Bounced Payments']:
                meets_threshold = value <= threshold
                comparison = "≤"
            else:
                meets_threshold = value >= threshold
                comparison = "≥"

            # Format values appropriately
            if isinstance(value, float):
                if metric in ['Operating Margin', 'Debt-to-Income Ratio', 'Expense-to-Revenue Ratio']:
                    formatted_value = f"{value:.3f} ({value*100:.1f}%)"
                elif metric in ['Revenue Growth Rate']:
                    formatted_value = f"{value:.3f} ({value:.1f}%)"
                else:
                    formatted_value = f"{value:.2f}"
            else:
                formatted_value = f"£{value:,.2f}" if 'Income' in metric or 'Revenue' in metric or 'Debt' in metric or 'Balance' in metric or 'Rate' in metric else str(value)

            metrics_data.append({
                'Metric': metric,
                'Actual Value': formatted_value,
                'Threshold': f"{comparison} {threshold}",
                'Status': 'Pass' if meets_threshold else 'Fail'
            })

    if metrics_data:
        df_metrics = pd.DataFrame(metrics_data)
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)

    # Period Comparison (if applicable)
    if analysis_period != 'All':
        st.markdown("---")
        with st.expander(f"Compare with Full Period Analysis", expanded=False):
            full_metrics = calculate_financial_metrics(df, params['company_age_months'])
            full_metrics = apply_manual_outstanding_debt(full_metrics)
            full_scores = calculate_all_scores_enhanced(full_metrics, params)

            st.write("**Full Period vs Selected Period Comparison:**")
            comp_col1, comp_col2, comp_col3, = st.columns(3)

            with comp_col1:
                delta_subprime = scores.get('subprime_score', 0) - full_scores.get('subprime_score', 0)
                st.metric("Full Period Subprime Score", f"{full_scores.get('subprime_score', 0):.1f}/100", 
                        delta=f"{delta_subprime:+.1f} difference")

            with comp_col2:
                if full_scores['ml_score'] and scores['ml_score']:
                    delta_ml = scores['ml_score'] - full_scores['ml_score']
                    st.metric("Full Period ML Probability", f"{full_scores['ml_score']:.1f}%",
                            delta=f"{delta_ml:+.1f}% difference")
                else:
                    st.metric("Full Period ML Probability", "N/A")

            with comp_col3:
                delta_revenue = metrics.get('Monthly Average Revenue', 0) - full_metrics.get('Monthly Average Revenue', 0)
                st.metric("Full Period Monthly Revenue", f"£{full_metrics.get('Monthly Average Revenue', 0):,.0f}",
                        delta=f"£{delta_revenue:+,.0f} difference")

    # ============================================
    # DETAILED SCORING ANALYSIS (Consolidated)
    # ============================================
    st.markdown("---")
    with st.expander("Detailed scoring analysis", expanded=False):
        # Subprime scoring overview
        subprime_col1, subprime_col2, subprime_col3 = st.columns(3)

        with subprime_col1:
            score = scores['subprime_score']
            if score >= 75:
                st.success(f"**Excellent candidate**\nScore: {score:.1f}/100")
            elif score >= 60:
                st.info(f"**Good candidate**\nScore: {score:.1f}/100")
            elif score >= 45:
                st.warning(f"**Conditional**\nScore: {score:.1f}/100")
            elif score >= 30:
                st.warning(f"**High monitoring**\nScore: {score:.1f}/100")
            else:
                st.error(f"**High risk**\nScore: {score:.1f}/100")

        with subprime_col2:
            st.write("**Pricing Guidance:**")
            pricing = scores['subprime_pricing']
            for key, value in pricing.items():
                if key in ['suggested_rate', 'max_loan_multiple', 'term_range']:
                    st.write(f"• **{key.replace('_', ' ').title()}**: {value}")

        with subprime_col3:
            st.write("**Monitoring:**")
            monitoring = pricing.get('monitoring', 'Standard reviews')
            approval_prob = pricing.get('approval_probability', 'Unknown')
            st.write(f"• {monitoring}")
            st.write(f"• Approval: {approval_prob}")

        # Subprime recommendation
        recommendation = scores['subprime_recommendation']
        if "APPROVE" in recommendation:
            st.success(f"**Subprime Recommendation**: {recommendation}")
        elif "CONDITIONAL" in recommendation:
            st.warning(f"**Subprime Recommendation**: {recommendation}")
        elif "SENIOR REVIEW" in recommendation:
            st.info(f"**Subprime Recommendation**: {recommendation}")
        else:
            st.error(f"**Subprime Recommendation**: {recommendation}")

        # Metric Scoring Breakdown
        st.markdown("---")
        st.markdown("**Metric Scoring Breakdown:**")
        if scores.get('score_breakdown'):
            breakdown_data = []
            for metric, data in scores['score_breakdown'].items():
                status = 'Pass' if data['meets'] else 'Fail'
                breakdown_data.append({
                    'Metric': metric,
                    'Actual': f"{data['actual']:.3f}",
                    'Status': status
                })
            if breakdown_data:
                breakdown_df = pd.DataFrame(breakdown_data)
                st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

        # Subprime breakdown
        st.markdown("---")
        st.markdown("**Subprime Scoring Components:**")
        for line in scores['subprime_breakdown']:
            st.write(f"• {line}")

        # Score Diagnostics
        if scores.get('diagnostics'):
            st.markdown("---")
            st.markdown("**Metric performance:**")
            diagnostics = scores['diagnostics']

            if diagnostics.get('metric_breakdown'):
                metric_data = []
                for metric in diagnostics['metric_breakdown']:
                    actual = metric['actual_value']
                    if 'Ratio' in metric['metric'] or 'DSCR' in metric['metric']:
                        actual_str = f"{actual:.2f}"
                    elif 'Volatility' in metric['metric']:
                        actual_str = f"{actual:.3f}"
                    elif 'Balance' in metric['metric']:
                        actual_str = f"£{actual:,.0f}"
                    elif 'Growth' in metric['metric'] or 'Margin' in metric['metric']:
                        actual_str = f"{actual*100:.1f}%"
                    elif 'Days' in metric['metric']:
                        actual_str = f"{int(actual)}"
                    elif 'Age' in metric['metric'] or 'Score' in metric['metric']:
                        actual_str = f"{int(actual)}"
                    else:
                        actual_str = f"{actual:.2f}"

                    threshold = metric['threshold_full_points']
                    if 'Ratio' in metric['metric'] or 'DSCR' in metric['metric']:
                        threshold_str = f"{threshold:.2f}"
                    elif 'Volatility' in metric['metric']:
                        threshold_str = f"≤{threshold:.2f}"
                    elif 'Balance' in metric['metric']:
                        threshold_str = f"£{threshold:,.0f}+"
                    elif 'Growth' in metric['metric'] or 'Margin' in metric['metric']:
                        threshold_str = f"{threshold*100:.1f}%+"
                    elif 'Days' in metric['metric']:
                        threshold_str = f"≤{int(threshold)}"
                    elif 'Age' in metric['metric'] or 'Score' in metric['metric']:
                        threshold_str = f"{int(threshold)}+"
                    else:
                        threshold_str = f"{threshold:.2f}"

                    status_label = {'PASS': 'Pass', 'PARTIAL': 'Partial', 'FAIL': 'Fail'}

                    metric_data.append({
                        'Metric': metric['metric'],
                        'Actual': actual_str,
                        'Target': threshold_str,
                        'Points': f"{metric['points_earned']:.1f}/{metric['points_possible']}",
                        'Status': status_label.get(metric['status'], metric['status']),
                    })

                if metric_data:
                    df_metrics = pd.DataFrame(metric_data)
                    st.dataframe(df_metrics, use_container_width=True, hide_index=True)

            # Key factors in columns
            if diagnostics.get('top_negative_factors') or diagnostics.get('top_positive_factors'):
                neg_col, pos_col = st.columns(2)

                with neg_col:
                    if diagnostics.get('top_negative_factors'):
                        st.markdown("**Top risk factors:**")
                        for factor in diagnostics['top_negative_factors'][:3]:
                            st.write(f"• {factor['metric']}: -{factor['points_lost']:.1f} pts")

                with pos_col:
                    if diagnostics.get('top_positive_factors'):
                        st.markdown("**Top strengths:**")
                        for factor in diagnostics['top_positive_factors'][:3]:
                            st.write(f"• {factor['metric']}: +{factor['points_earned']:.1f} pts")

            # Improvement suggestions
            if diagnostics.get('improvement_suggestions'):
                st.markdown("**Improvements:**")
                for suggestion in diagnostics['improvement_suggestions'][:3]:
                    st.info(f"• {suggestion}")

        # ML Validation (if available)
        ml_validation = scores.get('ml_validation', {})
        if ml_validation.get('available', False):
            st.markdown("---")
            st.markdown("**ML score reliability:**")
            ml_col1, ml_col2, ml_col3 = st.columns(3)

            with ml_col1:
                quality = ml_validation.get('data_quality_score', 0)
                st.metric("Data Quality", f"{quality}/100")

            with ml_col2:
                confidence = ml_validation.get('ml_confidence', 'Unknown')
                st.metric("ML Confidence", confidence)

            with ml_col3:
                outliers = ml_validation.get('outlier_count', 0)
                st.metric("Unusual Metrics", outliers)

    # Dashboard Export Section
    try:
        exporter = DashboardExporter()
        exporter.create_export_buttons(
            company_name=company_name,
            params=params,
            metrics=metrics,
            scores=scores,
            analysis_period=analysis_period,
            revenue_insights=revenue_insights,
            loans_analysis=None,  # Will be added when loans analysis is available

        )
        st.success("Enhanced Dashboard complete with Export Functionality")

    except Exception as e:
        st.error(f"Export functionality error: {str(e)}")
        st.success("Enhanced Dashboard complete (export disabled due to error)")


def render_saved_run_loader():
    """Load a saved scorecard run into session state without reprocessing."""
    from app.services.run_persistence import (
        choose_directory,
        list_reloadable_scorecard_runs,
        reloadable_runs_default_root,
        load_reloadable_scorecard_run,
    )

    st.markdown("##### Saved runs")
    with st.expander("Load a saved run", expanded=False):
        load_dir = st.session_state.get("app_saved_runs_load_dir") or str(reloadable_runs_default_root())
        cols = st.columns([1, 2])
        with cols[0]:
            if st.button("Choose saved run folder", use_container_width=True, key="choose_saved_run_folder"):
                selected = choose_directory(title="Choose saved run folder", initial_dir=load_dir)
                if selected:
                    st.session_state["app_saved_runs_load_dir"] = selected
                    st.rerun()
                else:
                    st.warning("No folder was selected.")
        with cols[1]:
            st.caption(f"Selected: `{load_dir}`")

        saved_runs = list_reloadable_scorecard_runs(load_dir)
        if not saved_runs:
            st.info("No saved runs found in the selected folder.")
            return

        def _label(run):
            saved_at = str(run.get("saved_at_utc", ""))[:19].replace("T", " ")
            company = run.get("company_name") or "Unknown company"
            name = run.get("run_name") or company
            return f"{saved_at} | {name} | {company}"

        selected_run = st.selectbox(
            "Saved run",
            saved_runs,
            format_func=_label,
            key="saved_run_selector_main",
        )
        if st.button("Load selected run", type="primary", use_container_width=True, key="load_saved_run_main"):
            try:
                st.session_state["last_run"] = load_reloadable_scorecard_run(selected_run)
                st.success("Saved run loaded. Rendering the dashboard now.")
                st.rerun()
            except Exception as e:
                st.error(f"Could not load saved run: {e}")


def render_saved_run_saver():
    """Save the current session run to a user-selected folder."""
    from app.services.run_persistence import (
        choose_directory,
        reloadable_runs_default_root,
        save_reloadable_scorecard_run,
    )

    run = st.session_state.get("last_run")
    if not run:
        return

    with st.expander("Save this run", expanded=False):
        save_dir = st.session_state.get("app_saved_runs_save_dir") or str(reloadable_runs_default_root())
        cols = st.columns([1, 2])
        with cols[0]:
            if st.button("Choose save folder", use_container_width=True, key="choose_save_run_folder"):
                selected = choose_directory(title="Choose folder to save this run", initial_dir=save_dir)
                if selected:
                    st.session_state["app_saved_runs_save_dir"] = selected
                    st.rerun()
                else:
                    st.warning("No folder was selected.")
        with cols[1]:
            st.caption(f"Selected: `{save_dir}`")

        default_name = run.get("company_name") or run.get("params", {}).get("company_name") or "Scorecard run"
        run_name = st.text_input("Run name", value=default_name, key="save_run_name_main")
        if st.button("Save run to selected folder", type="primary", use_container_width=True, key="save_run_main"):
            try:
                run_dir = save_reloadable_scorecard_run(
                    run=run,
                    save_root=save_dir,
                    run_name=run_name,
                )
                st.session_state["last_saved_reloadable_run"] = str(run_dir)
                st.success(f"Saved run: `{run_dir}`")
            except FileExistsError:
                st.error("A saved run folder with that generated name already exists. Try again in a moment.")
            except Exception as e:
                st.error(f"Could not save run: {e}")


def main():
    """Main application"""
    try:
        apply_ui_theme()

        render_main_hero(
            "Business Finance Scorecard",
            "Bank intelligence, bureau signals, and director credit—in one underwriting workspace.",
            eyebrow="Merchant cash advance · underwriting",
        )
        render_workflow_rail()

        # Sidebar inputs
        sidebar_section("Business parameters")
        
        company_name = st.sidebar.text_input("Company Name", "Sample Business Ltd")
        industry = st.sidebar.selectbox("Industry", list(INDUSTRY_THRESHOLDS.keys()))
        requested_loan = st.sidebar.number_input("Requested Loan (£)", min_value=0.0, value=5000.0, step=1000.0)
        company_age_months = st.sidebar.number_input("Company Age (Months)", min_value=0, value=12, step=1)

        from pathlib import Path

        # -----------------------------
        # Director TU XML Upload
        # -----------------------------
        sidebar_subsection("Director TU credit file (TransUnion XML)")
        tu_xml_file = st.sidebar.file_uploader("Upload TU XML", type=["xml"], key="tu_xml_upload")

        tu_result = None
        tu_present = tu_xml_file is not None
        tu_parse_status = "missing"
        tu_parse_error = ""
        directors_score = 50  # Neutral legacy fallback; real TU score is stored separately.
        tu_director_score = None
        revenue_insights = {}

        if tu_xml_file is not None:
            try:
                xml_bytes = tu_xml_file.getvalue()
                app_id = Path(tu_xml_file.name).stem

                tu_result = _score_tu_xml_bytes(xml_bytes, app_id)
                tu_director_score = tu_result["score"]
                directors_score = tu_director_score
                tu_parse_status = "parsed"

                st.sidebar.success(f"Director TU Score: {tu_result['score']}/100")
                st.sidebar.write(f"Director TU Decision: **{tu_result['decision']}**")

                if tu_result["flags"]:
                    st.sidebar.caption("Recovery Flags")
                    st.sidebar.write(tu_result["flags"])

                if tu_result["reasons"]:
                    st.sidebar.caption("Reasons")
                    st.sidebar.write(tu_result["reasons"])

            except Exception as e:
                tu_parse_status = "failed"
                tu_parse_error = str(e)
                st.sidebar.error(f"TU XML scoring failed: {e}")
                directors_score = 50
                tu_director_score = None
                tu_result = None

        # -----------------------------
        # Business Bureau PDF Upload (NEW)
        # -----------------------------
        sidebar_subsection("Business credit report (PDF)")
        bureau_pdf = st.sidebar.file_uploader(
            "Upload Business Credit Report (PDF)",
            type=["pdf"],
            key="bureau_pdf_upload"
        )

        bureau_text = ""
        bureau_backend = "none"
        bureau_err = ""
        bureau_parse_status = "missing"
        business_ccj = None
        bureau_band = None
        bureau_band_reasons = []
        bureau_signals = {}
        report_info = {}

        if bureau_pdf is not None:
            bureau_result = parse_business_bureau_pdf(bureau_pdf.getvalue())
            bureau_text = bureau_result.text
            bureau_backend = bureau_result.backend
            bureau_err = bureau_result.error
            bureau_parse_status = bureau_result.parse_status
            business_ccj = bureau_result.business_ccj
            bureau_band = bureau_result.bureau_band
            bureau_band_reasons = bureau_result.bureau_band_reasons
            bureau_signals = bureau_result.signals
            report_info = bureau_result.report_information

            if bureau_parse_status == "failed":
                st.sidebar.error(
                    "Business credit PDF text extraction failed. "
                    f"No bureau band or CCJ status was inferred. {bureau_err}"
                )
            elif bureau_parse_status == "parsed":
                st.sidebar.success(f"Business credit PDF parsed via {bureau_backend}")
        with st.sidebar.expander("Report Information", expanded=False):
            for section_name in ["Credit information", "Payment performance", "Legal notices", "Public Record",
                                 "Charges"]:
                st.markdown(f"**{section_name}**")
                bullets = report_info.get(section_name, []) or []

                if not bullets:
                    st.write("Not found in report")
                else:
                    for b in bullets:
                        st.write(f"• {b}")

                st.markdown("---")

        # -----------------------------
        # Risk Factors (UPDATED)
        # -----------------------------
        sidebar_subsection("Risk factors")

        # Show CCJ as derived (no manual tick box)
        ccj_label = "Unknown" if business_ccj is None else ("Yes" if business_ccj else "No")
        st.sidebar.write(f"**Business CCJ:** {ccj_label}")

        # Removed: Poor/No Online Presence + Generic Email (no longer used / no penalties)
        poor_or_no_online_presence = False
        uses_generic_email = False

        # Show banding (informational)
        if bureau_band:
            st.sidebar.markdown("---")
            st.sidebar.write(f"**bureau_band = {bureau_band}**")
            with st.sidebar.expander("Why this band?", expanded=False):
                if bureau_band_reasons:
                    for r in bureau_band_reasons[:10]:
                        st.sidebar.write(f"• {r}")
                else:
                    st.sidebar.write("• No specific indicators extracted (text-only fallback).")

        # Time period filter
        sidebar_subsection("Analysis period")
        analysis_period = st.sidebar.selectbox(
            "Select Time Period",
            ["All", "3", "6", "9", "12"],
            help="Choose how many months of data to analyze",
        )

        render_sidebar_help_footer()

        params = {
            "company_name": company_name,
            "industry": industry,
            "requested_loan": requested_loan,

            # IMPORTANT: keep the same param name the rest of the app already uses
            "directors_score": directors_score,

            # Persist TU outputs for display/export
            "tu_present": tu_present,
            "tu_parse_status": tu_parse_status,
            "tu_parse_error": tu_parse_error,
            "tu_director_score": tu_director_score,
            "tu_director_decision": (tu_result or {}).get("decision"),
            "tu_director_flags": (tu_result or {}).get("flags") or [],
            "tu_director_reasons": (tu_result or {}).get("reasons") or [],

            "company_age_months": company_age_months,
            "business_ccj": business_ccj,  # True/False when parsed, None when missing or failed
            "business_credit_score": bureau_signals.get("credit_score"),
            "business_credit_score_min": bureau_signals.get("credit_score_min"),
            "business_credit_score_max": bureau_signals.get("credit_score_max"),
            "business_credit_score_range": bureau_signals.get("credit_score_range"),
            "business_credit_score_suppressed": bool(bureau_signals.get("credit_score_suppressed", False)),
            "business_credit_limit": bureau_signals.get("credit_limit"),
            "business_max_recommended_credit": bureau_signals.get("max_recommended_credit"),
            "business_negative_impact_count": bureau_signals.get("negative_impact_count", 0),
            "business_enquiries_3m": bureau_signals.get("enquiries_3m"),
            "business_company_searches_12m": bureau_signals.get("company_searches_12m"),
            "business_bureau_needs_attention": bool(bureau_signals.get("needs_attention", False)),
            "business_no_registered_charges": bool(bureau_signals.get("no_registered_charges", False)),
            "business_bureau_parse_status": bureau_parse_status,
            "business_bureau_backend": bureau_backend,
            "business_bureau_parse_error": bureau_err,
            "poor_or_no_online_presence": False,
            "uses_generic_email": False,

            # Optional: persist bureau band for display/export (NOT used in scoring)
            "bureau_band": bureau_band,
            "bureau_band_reasons": bureau_band_reasons,
        }


        # File upload
        render_intake_panel_intro()
        render_saved_run_loader()
        uploaded_file = st.file_uploader(
            "Transaction file (JSON)",
            type=["json"],
            help="Plaid-style export or list of transactions with date, amount, and name.",
        )
        st.markdown("##### Card terminal statements")
        st.caption(
            "PDF, CSV, Excel, or a ZIP of those files from your card provider—used to reconcile card sales against bank inflows."
        )
        card_terminal_files = st.file_uploader(
            "Upload card terminal statements",
            type=["pdf", "csv", "xls", "xlsx", "zip"],
            accept_multiple_files=True,
            key="card_terminal_statements_uploader",
            help="You can upload mixed providers/formats in one run.",
        )
        process_clicked = st.button(
            "Process analysis",
            type="primary",
            disabled=uploaded_file is None,
            use_container_width=True,
        )
        run_to_show = None  # set after processing or from last_run when returning from another page

        if uploaded_file is not None and process_clicked:
            try:
                # Read and process file
                uploaded_file.seek(0)
                raw_bytes = uploaded_file.getvalue()

                try:
                    string_data = raw_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    try:
                        string_data = raw_bytes.decode("utf-16")
                    except UnicodeDecodeError:
                        string_data = raw_bytes.decode("latin-1")
                
                if not string_data.strip():
                    st.error("Uploaded file is empty")
                    return

                json_data = json.loads(string_data)

                analysis_result = analyse_open_banking_application(
                    json_data=json_data,
                    params=params,
                    analysis_period=analysis_period,
                    card_terminal_files=card_terminal_files,
                    callbacks=AnalysisCallbacks(
                        filter_data_by_period=filter_data_by_period,
                        assess_primary_account_signal=assess_primary_account_signal,
                        calculate_financial_metrics=calculate_financial_metrics,
                        apply_manual_outstanding_debt=apply_manual_outstanding_debt,
                        derive_card_processing_payload=derive_card_processing_payload,
                        calculate_all_scores_enhanced=calculate_all_scores_enhanced,
                        combine_mca_and_tu_decisions=combine_mca_and_tu_decisions,
                        calculate_revenue_insights=calculate_revenue_insights,
                    ),
                    source_upload_name=uploaded_file.name if uploaded_file else None,
                )

                df = analysis_result.df
                filtered_df = analysis_result.filtered_df
                params = analysis_result.params
                metrics = analysis_result.metrics
                scores = analysis_result.scores
                revenue_insights = analysis_result.revenue_insights
                card_processing_payload = analysis_result.card_processing_payload
                date_min_iso = analysis_result.date_min_iso
                date_max_iso = analysis_result.date_max_iso
                ingestion_metadata = analysis_result.ingestion_metadata

                amount_warning = ingestion_metadata.get("amount_convention_warning")
                junk_warning = ingestion_metadata.get("junk_transactions_warning")
                duplicate_warning = ingestion_metadata.get("duplicate_transactions_warning")
                balance_warning = ""
                if "balance_warning" in df.columns and not df["balance_warning"].dropna().empty:
                    balance_warning = str(df["balance_warning"].dropna().iloc[0])
                for warning in (amount_warning, junk_warning, duplicate_warning, balance_warning):
                    if warning:
                        st.warning(warning)

                # Display data info and export
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.success(f"Loaded {len(df)} transactions")
                    st.caption(f"Amount convention: {ingestion_metadata.get('amount_convention', 'unknown')}")
                    if "balance_source" in df.columns and not df["balance_source"].dropna().empty:
                        st.caption(
                            f"Balance: {df['balance_source'].dropna().iloc[0]} "
                            f"({df['balance_confidence'].dropna().iloc[0] if 'balance_confidence' in df.columns and not df['balance_confidence'].dropna().empty else 'unknown'})"
                        )
                with col2:
                    date_range = f"{df['date'].min().date()} to {df['date'].max().date()}"
                    st.info(f"Date range: {date_range}")
                with col3:
                    if analysis_period != 'All':
                        filtered_count = len(filter_data_by_period(df, analysis_period))
                        st.info(f"Period: {filtered_count} transactions")
                with col4:
                    csv_data = create_categorized_csv(df)
                    if csv_data:
                        st.download_button(
                            label="Export categorized CSV",
                            data=csv_data,
                            file_name=f"{company_name.replace(' ', '_')}_transactions_categorized.csv",
                            mime="text/csv",
                            help="Download all transaction data with our subcategorization",
                            type="primary",
                            key="csv_export_main"
                        )

                # Store last successful run (complete, so other pages / return to Main can use it)
                st.session_state["last_run"] = analysis_result.run
                del raw_bytes, string_data, json_data
                gc.collect()

                try:
                    from app.services.run_persistence import persist_scorecard_run

                    persist_scorecard_run(
                        company_name=company_name,
                        analysis_period=analysis_period,
                        params=params,
                        metrics=metrics,
                        scores=scores,
                        revenue_insights=revenue_insights,
                        source_upload_name=uploaded_file.name if uploaded_file else None,
                        txn_row_count=len(df),
                        date_min_iso=date_min_iso,
                        date_max_iso=date_max_iso,
                    )
                except Exception:
                    pass

                render_full_financial_dashboard(
                    company_name=company_name,
                    analysis_period=analysis_period,
                    df=df,
                    filtered_df=filtered_df,
                    params=params,
                    metrics=metrics,
                    scores=scores,
                    revenue_insights=revenue_insights,
                    card_terminal_files=card_terminal_files,
                    card_processing_payload=card_processing_payload,
                )
                render_saved_run_saver()
                    
            except Exception as e:
                st.error(f"Unexpected error during processing: {e}")
                import traceback
                full_traceback = traceback.format_exc()
                st.code(full_traceback)
                print(full_traceback)
        
        elif uploaded_file is not None and not process_clicked and not st.session_state.get("last_run"):
            st.info("Transaction file loaded. Press Process analysis to run the scorecard.")

        elif st.session_state.get("last_run"):
            run = st.session_state["last_run"]
            st.info(
                "Showing your last completed analysis (same session). "
                "Upload a new JSON transaction file to run a fresh score."
            )
            company_name = run["company_name"]
            analysis_period = run["analysis_period"]
            df = run["df"]
            filtered_df = run["filtered_df"]
            params = run["params"]
            metrics = run["metrics"]
            scores = run["scores"]
            revenue_insights = run.get("revenue_insights") or {}
            if not revenue_insights and not filtered_df.empty:
                revenue_insights = calculate_revenue_insights(filtered_df)
            card_terminal_files = run.get("card_terminal_files")
            card_processing_payload = run.get("card_processing_payload")
            if card_processing_payload is None:
                card_processing_payload = derive_card_processing_payload(filtered_df, card_terminal_files)
                metrics.update(card_processing_payload.get("insights") or {})

            render_full_financial_dashboard(
                company_name=company_name,
                analysis_period=analysis_period,
                df=df,
                filtered_df=filtered_df,
                params=params,
                metrics=metrics,
                scores=scores,
                revenue_insights=revenue_insights,
                card_terminal_files=card_terminal_files,
                card_processing_payload=card_processing_payload,
            )
            render_saved_run_saver()
        else:
            render_empty_state_main()

    except Exception as e:
        st.error(f"Application error: {str(e)}")
        import traceback
        full_traceback = traceback.format_exc()
        st.code(full_traceback)
        print(full_traceback)

if __name__ == "__main__":
    main()
# ADD THIS LINE AT THE VERY END OF THE FILE:
SubprimeScoringSystem = SubprimeScoring


