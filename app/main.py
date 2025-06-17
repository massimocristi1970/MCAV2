import streamlit as st
import pandas as pd
import joblib
import json
import re
from datetime import datetime, timedelta
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add debug info about file structure
def debug_file_structure():
    """Debug helper to understand the file structure"""
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

# Run debug (remove this after fixing)
debug_file_structure()

# Improved import with multiple fallback strategies
def import_subprime_scoring():
    """Import SubprimeScoring with comprehensive fallback strategies"""
    
    # Strategy 1: Direct import from services (if main.py is in app/)
    try:
        from services.subprime_scoring_system import SubprimeScoring
        print("✅ Imported SubprimeScoring from services.subprime_scoring_system")
        return SubprimeScoring, True
    except ImportError as e:
        print(f"❌ Failed import from services: {e}")
    
    # Strategy 2: Import from app.services (if main.py is in root/)
    try:
        from app.services.subprime_scoring_system import SubprimeScoring
        print("✅ Imported SubprimeScoring from app.services.subprime_scoring_system")
        return SubprimeScoring, True
    except ImportError as e:
        print(f"❌ Failed import from app.services: {e}")
    
    # Strategy 3: Add path and import
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        services_dir = os.path.join(current_dir, 'services')
        
        if os.path.exists(services_dir) and services_dir not in sys.path:
            sys.path.insert(0, services_dir)
            print(f"📁 Added to path: {services_dir}")
        
        from subprime_scoring_system import SubprimeScoring
        print("✅ Imported SubprimeScoring after adding services to path")
        return SubprimeScoring, True
    except ImportError as e:
        print(f"❌ Failed import after path addition: {e}")
    
    # Strategy 4: Check parent app/services directory
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        app_services_dir = os.path.join(parent_dir, 'app', 'services')
        
        if os.path.exists(app_services_dir) and app_services_dir not in sys.path:
            sys.path.insert(0, app_services_dir)
            print(f"📁 Added to path: {app_services_dir}")
        
        from subprime_scoring_system import SubprimeScoring
        print("✅ Imported SubprimeScoring from parent app/services")
        return SubprimeScoring, True
    except ImportError as e:
        print(f"❌ Failed import from parent app/services: {e}")
    
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
                print("✅ Imported SubprimeScoring using importlib")
                return SubprimeScoring, True
        
        print("❌ subprime_scoring_system.py not found in any expected location")
        
    except Exception as e:
        print(f"❌ Failed absolute import: {e}")
    
    # If all strategies fail, return None
    print("🚨 All import strategies failed!")
    return None, False

# Use the import function
SubprimeScoring, SUBPRIME_SCORING_AVAILABLE = import_subprime_scoring()

# Create fallback class if import failed
if not SUBPRIME_SCORING_AVAILABLE:
    print("⚠️ Creating fallback SubprimeScoring class")
    class SubprimeScoring:
        def calculate_subprime_score(self, metrics, params):
            return {
                'subprime_score': 0,
                'risk_tier': 'Import Failed',
                'pricing_guidance': {'suggested_rate': 'N/A'},
                'recommendation': 'Subprime scoring import failed - check file structure',
                'breakdown': ['Import failed - check console for details']
            }
# NEW IMPORT: Add adaptive scoring module with proper path handling
ADAPTIVE_SCORING_AVAILABLE = False

def setup_adaptive_scoring():
    """Setup adaptive scoring import with detailed error reporting"""
    global ADAPTIVE_SCORING_AVAILABLE
    
    print("🔍 Starting adaptive scoring import attempts...")
    
    try:
        # Attempt 1: Direct import (if already in same directory)
        print("  Attempt 1: Direct import...")
        from adaptive_score_calculation import get_improved_weighted_score, get_detailed_scoring_breakdown
        ADAPTIVE_SCORING_AVAILABLE = True
        print("✅ Adaptive scoring module loaded successfully (direct import)")
        return True
    except ImportError as e:
        print(f"  ❌ Direct import failed: {e}")
    except Exception as e:
        print(f"  ❌ Direct import failed with error: {e}")
    
    try:
        # Attempt 2: Import from app folder
        print("  Attempt 2: Import from app folder...")
        from app.adaptive_score_calculation import get_improved_weighted_score, get_detailed_scoring_breakdown
        ADAPTIVE_SCORING_AVAILABLE = True
        print("✅ Adaptive scoring module loaded successfully (from app folder)")
        return True
    except ImportError as e:
        print(f"  ❌ App folder import failed: {e}")
    except Exception as e:
        print(f"  ❌ App folder import failed with error: {e}")
    
    try:
        # Attempt 3: Add app folder to path and import
        print("  Attempt 3: Adding app folder to path...")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        app_dir = os.path.join(current_dir, 'app')
        if app_dir not in sys.path:
            sys.path.insert(0, app_dir)
            print(f"    Added to path: {app_dir}")
        
        from adaptive_score_calculation import get_improved_weighted_score, get_detailed_scoring_breakdown
        ADAPTIVE_SCORING_AVAILABLE = True
        print("✅ Adaptive scoring module loaded successfully (with app path adjustment)")
        return True
    except ImportError as e:
        print(f"  ❌ Path adjustment import failed: {e}")
    except Exception as e:
        print(f"  ❌ Path adjustment import failed with error: {e}")
    
    try:
        # Attempt 4: Try parent directory then app
        print("  Attempt 4: Trying parent directory...")
        parent_dir = os.path.dirname(current_dir)
        app_dir = os.path.join(parent_dir, 'app')
        if app_dir not in sys.path:
            sys.path.insert(0, app_dir)
            print(f"    Added to path: {app_dir}")
        
        from adaptive_score_calculation import get_improved_weighted_score, get_detailed_scoring_breakdown
        ADAPTIVE_SCORING_AVAILABLE = True
        print("✅ Adaptive scoring module loaded successfully (from parent/app path)")
        return True
    except ImportError as e:
        print(f"  ❌ Parent directory import failed: {e}")
    except Exception as e:
        print(f"  ❌ Parent directory import failed with error: {e}")
    
    try:
        # Attempt 5: Force reload and import (since file exists)
        print("  Attempt 5: Force reload...")
        import importlib.util
        import sys
        
        # Find the file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        module_path = os.path.join(current_dir, 'adaptive_score_calculation.py')
        
        if os.path.exists(module_path):
            print(f"    Found module at: {module_path}")
            spec = importlib.util.spec_from_file_location("adaptive_score_calculation", module_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules["adaptive_score_calculation"] = module
            spec.loader.exec_module(module)
            
            # Test the functions
            get_improved_weighted_score = module.get_improved_weighted_score
            get_detailed_scoring_breakdown = module.get_detailed_scoring_breakdown
            
            ADAPTIVE_SCORING_AVAILABLE = True
            print("✅ Adaptive scoring module loaded successfully (force reload)")
            return True
        else:
            print(f"    Module not found at expected path: {module_path}")
            
    except Exception as e:
        print(f"  ❌ Force reload failed with error: {e}")
        import traceback
        print(f"    Detailed error: {traceback.format_exc()}")
    
    # If all attempts fail
    ADAPTIVE_SCORING_AVAILABLE = False
    print("⚠️ All import attempts failed")
    print("   The adaptive_score_calculation.py file exists but cannot be imported")
    print("   This usually means there's a syntax error or missing dependency in the file")
    return False

# Run the setup
setup_adaptive_scoring()

st.set_page_config(
    page_title="Business Finance Scorecard",
    layout="wide"
)

# Complete Industry thresholds with all sectors
INDUSTRY_THRESHOLDS = dict(sorted({
    'Medical Practices (GPs, Clinics, Dentists)': {
        'Debt Service Coverage Ratio': 1.60, 'Net Income': 1000, 'Operating Margin': 0.10,
        'Revenue Growth Rate': 0.06, 'Cash Flow Volatility': 0.08, 'Gross Burn Rate': 16000,
        'Directors Score': 75, 'Sector Risk': 0, 'Average Month-End Balance': 900,
        'Average Negative Balance Days per Month': 0, 'Number of Bounced Payments': 0,
    },
    'Pharmacies (Independent or Small Chains)': {
        'Debt Service Coverage Ratio': 1.60, 'Net Income': 500, 'Operating Margin': 0.08,
        'Revenue Growth Rate': 0.05, 'Cash Flow Volatility': 0.08, 'Gross Burn Rate': 15000,
        'Directors Score': 75, 'Sector Risk': 0, 'Average Month-End Balance': 750,
        'Average Negative Balance Days per Month': 0, 'Number of Bounced Payments': 0,
    },
    'Business Consultants': {
        'Debt Service Coverage Ratio': 1.60, 'Net Income': 1000, 'Operating Margin': 0.09,
        'Revenue Growth Rate': 0.05, 'Cash Flow Volatility': 0.08, 'Gross Burn Rate': 14000,
        'Directors Score': 75, 'Sector Risk': 0, 'Average Month-End Balance': 750,
        'Average Negative Balance Days per Month': 0, 'Number of Bounced Payments': 0,
    },
    'IT Services and Support Companies': {
        'Debt Service Coverage Ratio': 1.55, 'Net Income': 500, 'Operating Margin': 0.12,
        'Revenue Growth Rate': 0.07, 'Cash Flow Volatility': 0.10, 'Gross Burn Rate': 13000,
        'Directors Score': 75, 'Sector Risk': 0, 'Average Month-End Balance': 700,
        'Average Negative Balance Days per Month': 0, 'Number of Bounced Payments': 0,
    },
    'Courier Services (Independent and Regional Operators)': {
        'Debt Service Coverage Ratio': 1.50, 'Net Income': 500, 'Operating Margin': 0.08,
        'Revenue Growth Rate': 0.06, 'Cash Flow Volatility': 0.10, 'Gross Burn Rate': 12000,
        'Directors Score': 75, 'Sector Risk': 0, 'Average Month-End Balance': 600,
        'Average Negative Balance Days per Month': 0, 'Number of Bounced Payments': 0,
    },
    'Grocery Stores and Mini-Markets': {
        'Debt Service Coverage Ratio': 1.50, 'Net Income': 500, 'Operating Margin': 0.07,
        'Revenue Growth Rate': 0.06, 'Cash Flow Volatility': 0.12, 'Gross Burn Rate': 10000,
        'Directors Score': 75, 'Sector Risk': 0, 'Average Month-End Balance': 600,
        'Average Negative Balance Days per Month': 1, 'Number of Bounced Payments': 0,
    },
    'Education': {
        'Debt Service Coverage Ratio': 1.45, 'Net Income': 1500, 'Operating Margin': 0.10,
        'Revenue Growth Rate': 0.05, 'Cash Flow Volatility': 0.09, 'Gross Burn Rate': 11500,
        'Directors Score': 75, 'Sector Risk': 0, 'Average Month-End Balance': 700,
        'Average Negative Balance Days per Month': 1, 'Number of Bounced Payments': 0,
    },
    'Engineering': {
        'Debt Service Coverage Ratio': 1.55, 'Net Income': 7000, 'Operating Margin': 0.07,
        'Revenue Growth Rate': 0.04, 'Cash Flow Volatility': 0.10, 'Gross Burn Rate': 13000,
        'Directors Score': 75, 'Sector Risk': 0, 'Average Month-End Balance': 650,
        'Average Negative Balance Days per Month': 1, 'Number of Bounced Payments': 0,
    },
    'Estate Agent': {
        'Debt Service Coverage Ratio': 1.50, 'Net Income': 4500, 'Operating Margin': 0.09,
        'Revenue Growth Rate': 0.04, 'Cash Flow Volatility': 0.10, 'Gross Burn Rate': 13000,
        'Directors Score': 75, 'Sector Risk': 0, 'Average Month-End Balance': 650,
        'Average Negative Balance Days per Month': 0, 'Number of Bounced Payments': 0,
    },
    'Food Service': {
        'Debt Service Coverage Ratio': 1.55, 'Net Income': 2500, 'Operating Margin': 0.06,
        'Revenue Growth Rate': 0.06, 'Cash Flow Volatility': 0.12, 'Gross Burn Rate': 11000,
        'Directors Score': 75, 'Sector Risk': 0, 'Average Month-End Balance': 700,
        'Average Negative Balance Days per Month': 1, 'Number of Bounced Payments': 0,
    },
    'Import / Export': {
        'Debt Service Coverage Ratio': 1.55, 'Net Income': 3000, 'Operating Margin': 0.07,
        'Revenue Growth Rate': 0.06, 'Cash Flow Volatility': 0.15, 'Gross Burn Rate': 11000,
        'Directors Score': 75, 'Sector Risk': 0, 'Average Month-End Balance': 700,
        'Average Negative Balance Days per Month': 1, 'Number of Bounced Payments': 0,
    },
    'Manufacturing': {
        'Debt Service Coverage Ratio': 1.60, 'Net Income': 1500, 'Operating Margin': 0.08,
        'Revenue Growth Rate': 0.05, 'Cash Flow Volatility': 0.11, 'Gross Burn Rate': 13500,
        'Directors Score': 75, 'Sector Risk': 0, 'Average Month-End Balance': 750,
        'Average Negative Balance Days per Month': 1, 'Number of Bounced Payments': 0,
    },
    'Marketing / Advertising / Design': {
        'Debt Service Coverage Ratio': 1.55, 'Net Income': 5000, 'Operating Margin': 0.11,
        'Revenue Growth Rate': 0.06, 'Cash Flow Volatility': 0.11, 'Gross Burn Rate': 13500,
        'Directors Score': 75, 'Sector Risk': 0, 'Average Month-End Balance': 750,
        'Average Negative Balance Days per Month': 1, 'Number of Bounced Payments': 0,
    },
    'Off-Licence Business': {
        'Debt Service Coverage Ratio': 1.55, 'Net Income': 4500, 'Operating Margin': 0.08,
        'Revenue Growth Rate': 0.06, 'Cash Flow Volatility': 0.11, 'Gross Burn Rate': 14000,
        'Directors Score': 75, 'Sector Risk': 0, 'Average Month-End Balance': 650,
        'Average Negative Balance Days per Month': 1, 'Number of Bounced Payments': 0,
    },
    'Telecommunications': {
        'Debt Service Coverage Ratio': 1.55, 'Net Income': 5000, 'Operating Margin': 0.11,
        'Revenue Growth Rate': 0.05, 'Cash Flow Volatility': 0.12, 'Gross Burn Rate': 13000,
        'Directors Score': 75, 'Sector Risk': 0, 'Average Month-End Balance': 700,
        'Average Negative Balance Days per Month': 1, 'Number of Bounced Payments': 0,
    },
    'Tradesman': {
        'Debt Service Coverage Ratio': 1.40, 'Net Income': 4000, 'Operating Margin': 0.08,
        'Revenue Growth Rate': 0.05, 'Cash Flow Volatility': 0.15, 'Gross Burn Rate': 11500,
        'Directors Score': 75, 'Sector Risk': 0, 'Average Month-End Balance': 700,
        'Average Negative Balance Days per Month': 2, 'Number of Bounced Payments': 0,
    },
    'Wholesaler / Distributor': {
        'Debt Service Coverage Ratio': 1.50, 'Net Income': 3500, 'Operating Margin': 0.10,
        'Revenue Growth Rate': 0.06, 'Cash Flow Volatility': 0.13, 'Gross Burn Rate': 13000,
        'Directors Score': 75, 'Sector Risk': 0, 'Average Month-End Balance': 700,
        'Average Negative Balance Days per Month': 1, 'Number of Bounced Payments': 0,
    },
    'Other': {
        'Debt Service Coverage Ratio': 1.50, 'Net Income': 3000, 'Operating Margin': 0.08,
        'Revenue Growth Rate': 0.05, 'Cash Flow Volatility': 0.13, 'Gross Burn Rate': 11000,
        'Directors Score': 75, 'Sector Risk': 1, 'Average Month-End Balance': 700,
        'Average Negative Balance Days per Month': 1, 'Number of Bounced Payments': 0,
    },
    'Personal Services': {
        'Debt Service Coverage Ratio': 1.50, 'Net Income': 2000, 'Operating Margin': 0.09,
        'Revenue Growth Rate': 0.05, 'Cash Flow Volatility': 0.12, 'Gross Burn Rate': 12000,
        'Directors Score': 75, 'Sector Risk': 1, 'Average Month-End Balance': 700,
        'Average Negative Balance Days per Month': 1, 'Number of Bounced Payments': 0,
    },
    'Restaurants and Cafes': {
        'Debt Service Coverage Ratio': 1.30, 'Net Income': 0, 'Operating Margin': 0.05,
        'Revenue Growth Rate': 0.04, 'Cash Flow Volatility': 0.16, 'Gross Burn Rate': 11000,
        'Directors Score': 75, 'Sector Risk': 1, 'Average Month-End Balance': 600,
        'Average Negative Balance Days per Month': 2, 'Number of Bounced Payments': 0,
    },
    'Bars and Pubs': {
        'Debt Service Coverage Ratio': 1.25, 'Net Income': 0, 'Operating Margin': 0.04,
        'Revenue Growth Rate': 0.03, 'Cash Flow Volatility': 0.18, 'Gross Burn Rate': 10000,
        'Directors Score': 75, 'Sector Risk': 1, 'Average Month-End Balance': 600,
        'Average Negative Balance Days per Month': 2, 'Number of Bounced Payments': 0,
    },
    'Beauty Salons and Spas': {
        'Debt Service Coverage Ratio': 1.30, 'Net Income': 500, 'Operating Margin': 0.06,
        'Revenue Growth Rate': 0.04, 'Cash Flow Volatility': 0.14, 'Gross Burn Rate': 9500,
        'Directors Score': 75, 'Sector Risk': 1, 'Average Month-End Balance': 550,
        'Average Negative Balance Days per Month': 2, 'Number of Bounced Payments': 0,
    },
    'E-Commerce Retailers': {
        'Debt Service Coverage Ratio': 1.35, 'Net Income': 1000, 'Operating Margin': 0.07,
        'Revenue Growth Rate': 0.05, 'Cash Flow Volatility': 0.14, 'Gross Burn Rate': 10000,
        'Directors Score': 75, 'Sector Risk': 1, 'Average Month-End Balance': 600,
        'Average Negative Balance Days per Month': 1, 'Number of Bounced Payments': 0,
    },
    'Event Planning and Management Firms': {
        'Debt Service Coverage Ratio': 1.30, 'Net Income': 500, 'Operating Margin': 0.05,
        'Revenue Growth Rate': 0.03, 'Cash Flow Volatility': 0.16, 'Gross Burn Rate': 10000,
        'Directors Score': 75, 'Sector Risk': 1, 'Average Month-End Balance': 600,
        'Average Negative Balance Days per Month': 1, 'Number of Bounced Payments': 0,
    },
    'Auto Repair Shops': {
        'Debt Service Coverage Ratio': 1.40, 'Net Income': 1000, 'Operating Margin': 0.08,
        'Revenue Growth Rate': 0.05, 'Cash Flow Volatility': 0.12, 'Gross Burn Rate': 9500,
        'Directors Score': 75, 'Sector Risk': 1, 'Average Month-End Balance': 650,
        'Average Negative Balance Days per Month': 1, 'Number of Bounced Payments': 0,
    },
    'Fitness Centres and Gyms': {
        'Debt Service Coverage Ratio': 1.35, 'Net Income': 500, 'Operating Margin': 0.06,
        'Revenue Growth Rate': 0.04, 'Cash Flow Volatility': 0.18, 'Gross Burn Rate': 10000,
        'Directors Score': 75, 'Sector Risk': 1, 'Average Month-End Balance': 600,
        'Average Negative Balance Days per Month': 1, 'Number of Bounced Payments': 0,
    },
    'Construction Firms': {
        'Debt Service Coverage Ratio': 1.30, 'Net Income': 1000, 'Operating Margin': 0.08,
        'Revenue Growth Rate': 0.04, 'Cash Flow Volatility': 0.16, 'Gross Burn Rate': 12500,
        'Directors Score': 75, 'Sector Risk': 1, 'Average Month-End Balance': 700,
        'Average Negative Balance Days per Month': 2, 'Number of Bounced Payments': 0,
    },
    'Printing / Publishing': {
        'Debt Service Coverage Ratio': 1.50, 'Net Income': 2500, 'Operating Margin': 0.08,
        'Revenue Growth Rate': 0.05, 'Cash Flow Volatility': 0.14, 'Gross Burn Rate': 11000,
        'Directors Score': 75, 'Sector Risk': 1, 'Average Month-End Balance': 650,
        'Average Negative Balance Days per Month': 1, 'Number of Bounced Payments': 0,
    },
    'Recruitment': {
        'Debt Service Coverage Ratio': 1.50, 'Net Income': 2000, 'Operating Margin': 0.09,
        'Revenue Growth Rate': 0.05, 'Cash Flow Volatility': 0.10, 'Gross Burn Rate': 13000,
        'Directors Score': 75, 'Sector Risk': 1, 'Average Month-End Balance': 600,
        'Average Negative Balance Days per Month': 1, 'Number of Bounced Payments': 0,
    },
    'Retail': {
        'Debt Service Coverage Ratio': 1.40, 'Net Income': 2500, 'Operating Margin': 0.09,
        'Revenue Growth Rate': 0.04, 'Cash Flow Volatility': 0.14, 'Gross Burn Rate': 11500,
        'Directors Score': 75, 'Sector Risk': 1, 'Average Month-End Balance': 620,
        'Average Negative Balance Days per Month': 1, 'Number of Bounced Payments': 0,
    },
}.items()))

# Scoring weights
WEIGHTS = {
    'Debt Service Coverage Ratio': 19, 'Net Income': 13, 'Operating Margin': 9,
    'Revenue Growth Rate': 5, 'Cash Flow Volatility': 12, 'Gross Burn Rate': 3,
    'Company Age (Months)': 4, 'Directors Score': 18, 'Sector Risk': 3,
    'Average Month-End Balance': 5, 'Average Negative Balance Days per Month': 6,
    'Number of Bounced Payments': 3,
}

PENALTIES = {
    "personal_default_12m": 3, "business_ccj": 5, "director_ccj": 3,
    'website_or_social_outdated': 3, 'uses_generic_email': 1, 'no_online_presence': 2
}

# Enhanced debug info function
def show_debug_info():
    """Show debug information about file structure and imports"""
    with st.expander("🔧 Debug Information", expanded=False):
        st.write(f"**Adaptive Scoring Available:** {ADAPTIVE_SCORING_AVAILABLE}")
        if ADAPTIVE_SCORING_AVAILABLE:
            st.success("✅ Enhanced adaptive scoring is enabled")
        else:
            st.warning("⚠️ Using original scoring only - adaptive_score_calculation.py not found")
        
        # Show current working directory and file structure
        current_dir = os.getcwd()
        st.write(f"**Current Working Directory:** {current_dir}")
        
        # List files in current directory
        try:
            files = [f for f in os.listdir('.') if f.endswith('.py')]
            st.write(f"**Python files in current directory:** {files}")
        except Exception as e:
            st.write(f"Could not list files in current directory: {e}")
        
        # Check app folder
        app_folder = os.path.join(current_dir, 'app')
        if os.path.exists(app_folder):
            try:
                app_files = [f for f in os.listdir(app_folder) if f.endswith('.py')]
                st.write(f"**Python files in app folder:** {app_files}")
                if 'adaptive_score_calculation.py' in app_files:
                    st.success("✅ Found adaptive_score_calculation.py in app folder!")
                else:
                    st.warning("⚠️ adaptive_score_calculation.py not found in app folder")
            except Exception as e:
                st.write(f"Could not list files in app folder: {e}")
        else:
            st.write("**App folder:** Not found in current directory")
        
        # Show Python path
        st.write("**Python Path Includes:**")
        for i, path in enumerate(sys.path[:5]):  # Show first 5 paths
            st.write(f"  {i+1}. {path}")
        if len(sys.path) > 5:
            st.write(f"  ... and {len(sys.path) - 5} more paths")

# NEW HELPER FUNCTIONS: Add scoring comparison functions
def calculate_both_weighted_scores(metrics, params, industry_thresholds):
    """Calculate both original and adaptive weighted scores for comparison"""
    
    # Original weighted score (keep existing logic)
    original_weighted_score = 0
    for metric, weight in WEIGHTS.items():
        if metric == 'Company Age (Months)':
            if params['company_age_months'] >= 6:
                original_weighted_score += weight
        elif metric == 'Directors Score':
            if params['directors_score'] >= industry_thresholds['Directors Score']:
                original_weighted_score += weight
        elif metric == 'Sector Risk':
            sector_risk = industry_thresholds['Sector Risk']
            if sector_risk <= industry_thresholds['Sector Risk']:
                original_weighted_score += weight
        elif metric in metrics:
            threshold = industry_thresholds.get(metric, 0)
            if metric in ['Cash Flow Volatility', 'Average Negative Balance Days per Month', 'Number of Bounced Payments']:
                if metrics[metric] <= threshold:
                    original_weighted_score += weight
            else:
                if metrics[metric] >= threshold:
                    original_weighted_score += weight
    
    # Apply penalties
    penalties = 0
    for flag, penalty in PENALTIES.items():
        if params.get(flag, False):
            penalties += penalty
    
    original_weighted_score = max(0, original_weighted_score - penalties)
    
    # New adaptive weighted score (if available)
    if ADAPTIVE_SCORING_AVAILABLE:
        try:
            # Import the functions dynamically
            if 'adaptive_score_calculation' in sys.modules:
                adaptive_module = sys.modules['adaptive_score_calculation']
            elif 'app.adaptive_score_calculation' in sys.modules:
                adaptive_module = sys.modules['app.adaptive_score_calculation']
            else:
                # Try to import again
                try:
                    import adaptive_score_calculation as adaptive_module
                except ImportError:
                    from app import adaptive_score_calculation as adaptive_module
            
            adaptive_weighted_score, raw_adaptive_score, scoring_details = adaptive_module.get_detailed_scoring_breakdown(
                metrics, params['directors_score'], industry_thresholds['Sector Risk'], industry_thresholds, 
                params['company_age_months'], params.get('personal_default_12m', False), 
                params.get('business_ccj', False), params.get('director_ccj', False), 
                params.get('website_or_social_outdated', False), params.get('uses_generic_email', False), 
                params.get('no_online_presence', False), PENALTIES
            )
            return original_weighted_score, adaptive_weighted_score, raw_adaptive_score, scoring_details
        except Exception as e:
            st.warning(f"Error in adaptive scoring calculation: {e}")
            return original_weighted_score, original_weighted_score, original_weighted_score, []
    else:
        return original_weighted_score, original_weighted_score, original_weighted_score, []

def load_models():
    """Load ML models"""
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        return None, None

def map_transaction_category(transaction):
    """Enhanced transaction categorization matching original version"""
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
    combined_text = f"{name} {description}"

    is_credit = amount < 0
    is_debit = amount > 0

    # Step 1: Custom keyword overrides
    if is_credit and re.search(
        r"(?i)\b("
        r"stripe|sumup|zettle|square|take\s*payments|shopify|card\s+settlement|daily\s+takings|payout"
        r"|paypal|go\s*cardless|klarna|worldpay|izettle|ubereats|just\s*eat|deliveroo|uber|bolt"
        r"|fresha|treatwell|taskrabbit|terminal|pos\s+deposit|revolut"
        r"|capital\s+on\s+tap|capital\s+one|evo\s*payments?|tink|teya(\s+solutions)?|talech"
        r"|barclaycard|elavon|adyen|payzone|verifone|ingenico"
        r"|nmi|trust\s+payments?|global\s+payments?|checkout\.com|epdq|santander|handepay"
        r"|dojo|valitor|paypoint|mypos|moneris"
        r"|merchant\s+services|payment\s+sense"
        r")\b", 
        combined_text
    ):
        return "Income"
    if is_credit and re.search(r"(you\s?lend|yl\s?ii|yl\s?ltd|yl\s?limited|yl\s?a\s?limited)(?!.*\b(fnd|fund|funding)\b)", combined_text):
        return "Income"
    if is_credit and re.search(r"(you\s?lend|yl\s?ii|yl\s?ltd|yl\s?limited|yl\s?a\s?limited).*\b(fnd|fund|funding)\b", combined_text):
        return "Loans"
    if is_credit and re.search(
        r"\biwoca\b|\bcapify\b|\bfundbox\b|\bgot[\s\-]?capital\b|\bfunding[\s\-]?circle\b|"
        r"\bfleximize\b|\bmarketfinance\b|\bliberis\b|\besme[\s\-]?loans\b|\bthincats\b|"
        r"\bwhite[\s\-]?oak\b|\bgrowth[\s\-]?street\b|\bnucleus[\s\-]?commercial[\s\-]?finance\b|"
        r"\bultimate[\s\-]?finance\b|\bjust[\s\-]?cash[\s\-]?flow\b|\bboost[\s\-]?capital\b|"
        r"\bmerchant[\s\-]?money\b|\bcapital[\s\-]?on[\s\-]?tap\b|\bkriya\b|\buncapped\b|"
        r"\blendingcrowd\b|\bfolk2folk\b|\bfunding[\s\-]?tree\b|\bstart[\s\-]?up[\s\-]?loans\b|"
        r"\bbcrs[\s\-]?business[\s\-]?loans\b|\bbusiness[\s\-]?enterprise[\s\-]?fund\b|"
        r"\bswig[\s\-]?finance\b|\benterprise[\s\-]?answers\b|\blet's[\s\-]?do[\s\-]?business[\s\-]?finance\b|"
        r"\bfinance[\s\-]?for[\s\-]?enterprise\b|\bdsl[\s\-]?business[\s\-]?finance\b|"
        r"\bbizcap[\s\-]?uk\b|\bsigma[\s\-]?lending\b|\bbizlend[\s\-]?ltd\b|\bloans?\b",
        combined_text
    ):
        return "Loans"

    if is_debit and re.search(
        r"\biwoca\b|\bcapify\b|\bfundbox\b|\bgot[\s\-]?capital\b|\bfunding[\s\-]?circle\b|\bfleximize\b|\bmarketfinance\b|\bliberis\b|"
        r"\besme[\s\-]?loans\b|\bthincats\b|\bwhite[\s\-]?oak\b|\bgrowth[\s\-]?street\b|\bnucleus[\s\-]?commercial[\s\-]?finance\b|"
        r"\bultimate[\s\-]?finance\b|\bjust[\s\-]?cash[\s\-]?flow\b|\bboost[\s\-]?capital\b|\bmerchant[\s\-]?money\b|"
        r"\bcapital[\s\-]?on[\s\-]?tap\b|\bkriya\b|\buncapped\b|\blendingcrowd\b|\bfolk2folk\b|\bfunding[\s\-]?tree\b|"
        r"\bstart[\s\-]?up[\s\-]?loans\b|\bbcrs[\s\-]?business[\s\-]?loans\b|\bbusiness[\s\-]?enterprise[\s\-]?fund\b|"
        r"\bswig[\s\-]?finance\b|\benterprise[\s\-]?answers\b|\blet's[\s\-]?do[\s\-]?business[\s\-]?finance\b|"
        r"\bfinance[\s\-]?for[\s\-]?enterprise\b|\bdsl[\s\-]?business[\s\-]?finance\b|\bbizcap[\s\-]?uk\b|"
        r"\bsigma[\s\-]?lending\b|\bbizlend[\s\-]?ltd\b|"
        r"\bloans?\b|\bdebt\b|\brepayment\b|\binstal?ments?\b|\bpay[\s\-]?back\b|\brepay(?:ing|ment|ed)?\b|\bcleared\b",
        combined_text
    ):
        return "Debt Repayments"

    # Step 2: Plaid category fallback
    plaid_map = {
        "income_wages": "Income",
        "income_other_income": "Income",
        "income_dividends": "Special Inflow",
        "income_interest_earned": "Special Inflow",
        "income_retirement_pension": "Special Inflow",
        "income_unemployment": "Special Inflow",
        "transfer_in_cash_advances_and_loans": "Loans",
        "loan_payments_credit_card_payment": "Debt Repayments",
        "loan_payments_personal_loan_payment": "Debt Repayments",
        "loan_payments_other_payment": "Debt Repayments",
        "loan_payments_car_payment": "Debt Repayments",
        "loan_payments_mortgage_payment": "Debt Repayments",
        "loan_payments_student_loan_payment": "Debt Repayments",
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
    }

    # Match exact key
    if category in plaid_map:
        return plaid_map[category]

    # Step 3: Fallback for Plaid broad categories
    broad_matchers = [
        ("Expenses", [
            "bank_fees_", "entertainment_", "food_and_drink_", "general_merchandise_",
            "general_services_", "government_and_non_profit_", "home_improvement_",
            "medical_", "personal_care_", "rent_and_utilities_", "transportation_", "travel_"
        ])
    ]

    for label, patterns in broad_matchers:
        if any(category.startswith(p) for p in patterns):
            return label

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
        data = categorize_transactions(data)
        
        # FIXED: Use absolute values for all amounts
        total_revenue = abs(data.loc[data['is_revenue'], 'amount'].sum()) if data['is_revenue'].any() else 0
        total_expenses = abs(data.loc[data['is_expense'], 'amount'].sum()) if data['is_expense'].any() else 0
        net_income = total_revenue - total_expenses
        total_debt_repayments = abs(data.loc[data['is_debt_repayment'], 'amount'].sum()) if data['is_debt_repayment'].any() else 0
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
        operating_margin = max(-1, min(1, net_income / total_revenue)) if total_revenue > 0 else -1  # Cap between -100% and 100%
        
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
                print(f"    Median change: {revenue_growth_rate:.3f} ({revenue_growth_rate*100:.1f}%)")
            else:
               revenue_growth_rate = 0
                
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
            failed_payment_keywords = ['unpaid', 'returned', 'bounced', 'insufficient', 'failed', 'declined']
            for keyword in failed_payment_keywords:
                bounced_payments += data['name_y'].str.contains(keyword, case=False, na=False).sum()
        
        # DEBUGGING: Print key values
        print(f"\n🔍 DEBUG - Financial Metrics:")
        print(f"  Total Revenue: £{total_revenue:,.2f}")
        print(f"  Total Expenses: £{total_expenses:,.2f}")
        print(f"  Net Income: £{net_income:,.2f}")
        print(f"  DSCR: {debt_service_coverage_ratio:.2f}")
        print(f"  Operating Margin: {operating_margin:.3f} ({operating_margin*100:.1f}%)")
        print(f"  Cash Flow Volatility: {cash_flow_volatility:.3f}")
        print(f"  Revenue Growth Rate: {revenue_growth_rate:.2f}%")
        print(f"  Avg Month-End Balance: £{avg_month_end_balance:,.2f}")
        print(f"  Avg Negative Days: {avg_negative_days}")
        print(f"  Bounced Payments: {bounced_payments}")
        
        return {
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
        
    except Exception as e:
        st.error(f"Error calculating metrics: {e}")
        import traceback
        print(f"Full error traceback: {traceback.format_exc()}")
        return {}

def calculate_all_scores_enhanced(metrics, params):
    """Enhanced scoring calculation with better debugging and subprime scoring"""
    industry_thresholds = INDUSTRY_THRESHOLDS[params['industry']]
    sector_risk = industry_thresholds['Sector Risk']
    
    print(f"\n🎯 DEBUG - Scoring Calculation:")
    print(f"  Industry: {params['industry']}")
    print(f"  Sector Risk: {sector_risk}")
    print(f"  Directors Score: {params['directors_score']}")
    print(f"  Company Age: {params['company_age_months']} months")

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
    
    # Calculate both weighted scores (original and adaptive)
    original_weighted_score, adaptive_weighted_score, raw_adaptive_score, scoring_details = calculate_both_weighted_scores(
        metrics, params, industry_thresholds
    )
    
    print(f"  Original Weighted Score: {original_weighted_score}/100")
    print(f"  Adaptive Weighted Score: {adaptive_weighted_score:.1f}%")
    
    # NEW: Subprime scoring - WITH ERROR HANDLING
    print(f"\n🎯 DEBUG - Attempting Subprime Scoring:")
    try:
        subprime_scorer = SubprimeScoring()
        print(f"  Subprime scorer created: {type(subprime_scorer)}")
        
        print(f"  Calling calculate_subprime_score with:")
        print(f"    Metrics keys: {list(metrics.keys())}")
        print(f"    Params keys: {list(params.keys())}")
        
        subprime_result = subprime_scorer.calculate_subprime_score(metrics, params)
        print(f"  Subprime calculation successful!")
        print(f"  Subprime Score: {subprime_result['subprime_score']:.1f}/100")
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
            
            print(f"  Raw ML Score: {ml_score:.1f}%")
            print(f"  Adjusted ML Score: {adjusted_ml_score:.1f}%")
            
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
    print(f"="*50)
    
    return {
        'weighted_score': original_weighted_score,
        'adaptive_weighted_score': adaptive_weighted_score,
        'raw_adaptive_score': raw_adaptive_score,
        'scoring_details': scoring_details,
        'industry_score': industry_score,
        'ml_score': ml_score,
        'adjusted_ml_score': adjusted_ml_score,
        'loan_risk': loan_risk,
        'score_breakdown': score_breakdown,
        # NEW: Subprime scoring results
        'subprime_score': subprime_result['subprime_score'],
        'subprime_tier': subprime_result['risk_tier'],
        'subprime_pricing': subprime_result['pricing_guidance'],
        'subprime_recommendation': subprime_result['recommendation'],
        'subprime_breakdown': subprime_result['breakdown']
    }

# Add this function to your main.py file

def adjust_ml_score_for_growth_business(raw_ml_score, metrics, params):
    """
    Adjust ML score for growth businesses that the traditional model undervalues.
    """
    
    if raw_ml_score is None:
        return None
    
    print(f"\n🔧 ML Score Adjustment Analysis:")
    print(f"  Raw ML Score: {raw_ml_score:.1f}%")
    
    # Initialize adjustment
    adjustment = 0
    adjustment_reasons = []
    
    # Factor 1: Strong DSCR despite losses
    dscr = metrics.get('Debt Service Coverage Ratio', 0)
    operating_margin = metrics.get('Operating Margin', 0)
    
    if dscr >= 3.0 and operating_margin < 0:
        adjustment += 15
        adjustment_reasons.append(f"Strong DSCR ({dscr:.1f}) despite losses (+15)")
    elif dscr >= 2.0 and operating_margin < 0:
        adjustment += 12
        adjustment_reasons.append(f"Good DSCR ({dscr:.1f}) despite losses (+12)")
    elif dscr >= 1.5 and operating_margin < 0:
        adjustment += 8
        adjustment_reasons.append(f"Adequate DSCR ({dscr:.1f}) despite losses (+8)")
    
    # Factor 2: High growth trajectory
    growth = metrics.get('Revenue Growth Rate', 0)
    if growth > 1:  # Convert percentage if needed
        growth = growth / 100
    
    if growth >= 0.2:  # 20%+ growth
        adjustment += 12
        adjustment_reasons.append(f"High growth ({growth*100:.1f}%) (+12)")
    elif growth >= 0.15:  # 15%+ growth
        adjustment += 8
        adjustment_reasons.append(f"Strong growth ({growth*100:.1f}%) (+8)")
    elif growth >= 0.1:  # 10%+ growth
        adjustment += 5
        adjustment_reasons.append(f"Good growth ({growth*100:.1f}%) (+5)")
    
    # Factor 3: Growth + DSCR combination (compounding effect)
    if growth >= 0.15 and dscr >= 2.0:
        adjustment += 10
        adjustment_reasons.append("Growth + Strong DSCR combination (+10)")
    elif growth >= 0.1 and dscr >= 1.5:
        adjustment += 5
        adjustment_reasons.append("Growth + Good DSCR combination (+5)")
    
    # Factor 4: Director reliability
    directors_score = params.get('directors_score', 0)
    if directors_score >= 80:
        adjustment += 3
        adjustment_reasons.append(f"Excellent director score ({directors_score}) (+3)")
    elif directors_score >= 70:
        adjustment += 2
        adjustment_reasons.append(f"Good director score ({directors_score}) (+2)")
    
    # Factor 5: Young company bonus (growth trajectory more important)
    company_age = params.get('company_age_months', 0)
    if company_age <= 36 and growth >= 0.15:  # Young + high growth
        adjustment += 5
        adjustment_reasons.append(f"Young high-growth company ({company_age}m, {growth*100:.1f}%) (+5)")
    
    # Factor 6: Revenue scale adjustment
    revenue = metrics.get('Total Revenue', 0)
    monthly_revenue = metrics.get('Monthly Average Revenue', 0)
    
    if monthly_revenue >= 10000:  # £10k+ monthly revenue
        adjustment += 3
        adjustment_reasons.append(f"Strong revenue scale (£{monthly_revenue:,.0f}/month) (+3)")
    elif monthly_revenue >= 5000:  # £5k+ monthly revenue
        adjustment += 2
        adjustment_reasons.append(f"Good revenue scale (£{monthly_revenue:,.0f}/month) (+2)")
    
    # Apply adjustment with cap
    adjusted_score = min(85, raw_ml_score + adjustment)  # Cap at 85%
    
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
    
    if adjusted_score >= 60:
        risk_level = "Low Risk"
        color = "🟢"
    elif adjusted_score >= 40:
        risk_level = "Moderate Risk" 
        color = "🟡"
    elif adjusted_score >= 25:
        risk_level = "Higher Risk"
        color = "🟠"
    else:
        risk_level = "High Risk"
        color = "🔴"
    
    interpretation = f"{color} **{risk_level}** (Adjusted: {adjusted_score:.1f}%, Raw: {raw_score:.1f}%)"
    
    if improvement >= 15:
        interpretation += f"\n  📈 **Significant upward adjustment** (+{improvement:.1f}) for growth business profile"
    elif improvement >= 8:
        interpretation += f"\n  📈 **Notable upward adjustment** (+{improvement:.1f}) for growth characteristics"
    elif improvement >= 3:
        interpretation += f"\n  📈 **Minor upward adjustment** (+{improvement:.1f}) for positive factors"
    else:
        interpretation += f"\n  ➡️ **Minimal adjustment** (+{improvement:.1f}) - standard risk profile"
    
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

def create_score_charts(scores, metrics):
    """Create clean bar charts for scores - ENHANCED with adaptive scoring"""
    
    # Score comparison chart
    fig_scores = go.Figure()
    
    score_data = {
        'Original Weighted': scores['weighted_score'],
        'Adaptive Weighted': scores.get('adaptive_weighted_score', scores['weighted_score']),
        'Industry Score': (scores['industry_score'] / 12) * 100,  # Convert to percentage
        'ML Probability': scores.get('adjusted_ml_score', scores.get('ml_score', 0))
    }
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    fig_scores.add_trace(go.Bar(
        x=list(score_data.keys()),
        y=list(score_data.values()),
        marker_color=colors,
        text=[f"{v:.1f}%" for v in score_data.values()],
        textposition='outside'
    ))
    
    fig_scores.update_layout(
        title="Enhanced Score Comparison",
        yaxis_title="Score (%)",
        showlegend=False,
        height=400,
        yaxis=dict(range=[0, 100])
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
        title="Financial Overview",
        yaxis_title="Amount (£)",
        showlegend=False,
        height=400
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
            line=dict(color='green', width=3)
        ))
        
        fig_trend.add_trace(go.Scatter(
            x=monthly_data['date'],
            y=monthly_data['monthly_expenses'],
            mode='lines+markers',
            name='Expenses',
            line=dict(color='red', width=3)
        ))
        
        fig_trend.update_layout(
            title="Monthly Revenue vs Expenses",
            xaxis_title="Month",
            yaxis_title="Amount (£)",
            height=400
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
        mode='markers',
        marker=dict(color='black', size=10, symbol='diamond'),
        line=dict(color='black', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="Actual vs Threshold Performance",
        xaxis_title="Metrics",
        yaxis_title="Values",
        height=500,
        xaxis_tickangle=-45
    )
    
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
        title="Monthly Transaction Counts by Category",
        xaxis_title="Month",
        yaxis_title="Number of Transactions",
        barmode='stack',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
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
        title="Monthly Transaction Amounts by Category",
        xaxis_title="Month",
        yaxis_title="Amount (£)",
        barmode='stack',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
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

def main():
    """Main application - ENHANCED with adaptive scoring"""
    st.title("🏦 Business Finance Scorecard")
    st.markdown("---")
    
    # Add debug information
    show_debug_info()
    
    # Sidebar inputs
    st.sidebar.header("Business Parameters")
    
    company_name = st.sidebar.text_input("Company Name", "Sample Business Ltd")
    industry = st.sidebar.selectbox("Industry", list(INDUSTRY_THRESHOLDS.keys()))
    requested_loan = st.sidebar.number_input("Requested Loan (£)", min_value=0.0, value=25000.0, step=1000.0)
    directors_score = st.sidebar.slider("Director Credit Score", 0, 100, 75)
    company_age_months = st.sidebar.number_input("Company Age (Months)", min_value=0, value=24, step=1)
    
    st.sidebar.subheader("Risk Factors")
    personal_default_12m = st.sidebar.checkbox("Personal Defaults (12m)")
    business_ccj = st.sidebar.checkbox("Business CCJs")
    director_ccj = st.sidebar.checkbox("Director CCJs")
    website_or_social_outdated = st.sidebar.checkbox("Outdated Web Presence")
    uses_generic_email = st.sidebar.checkbox("Generic Email")
    no_online_presence = st.sidebar.checkbox("No Online Presence")
    
    # Time period filter
    st.sidebar.subheader("📅 Analysis Period")
    analysis_period = st.sidebar.selectbox(
        "Select Time Period",
        ["All", "3", "6", "9", "12"],
        help="Choose how many months of data to analyze"
    )
    
    params = {
        'company_name': company_name,
        'industry': industry,
        'requested_loan': requested_loan,
        'directors_score': directors_score,
        'company_age_months': company_age_months,
        'personal_default_12m': personal_default_12m,
        'business_ccj': business_ccj,
        'director_ccj': director_ccj,
        'website_or_social_outdated': website_or_social_outdated,
        'uses_generic_email': uses_generic_email,
        'no_online_presence': no_online_presence
    }
    
    # File upload
    uploaded_file = st.file_uploader("Upload Transaction Data (JSON)", type=['json'])
    
    if uploaded_file is not None:
        try:
            # Read and process file
            uploaded_file.seek(0)
            string_data = uploaded_file.getvalue().decode("utf-8")
            
            if not string_data.strip():
                st.error("❌ Uploaded file is empty")
                return
            
            json_data = json.loads(string_data)
            transactions = json_data.get('transactions', [])
            
            if not transactions:
                st.error("❌ No transactions found in JSON file")
                return
            
            df = pd.json_normalize(transactions)
            required_columns = ['date', 'amount', 'name']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"❌ Missing required columns: {missing_columns}")
                return
            
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
            df = df.dropna(subset=['date', 'amount'])
            
            if df.empty:
                st.error("❌ No valid transactions after cleaning")
                return
            
            # Display data info and export
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.success(f"✅ Loaded {len(df)} transactions")
            with col2:
                date_range = f"{df['date'].min().date()} to {df['date'].max().date()}"
                st.info(f"📅 Date Range: {date_range}")
            with col3:
                if analysis_period != 'All':
                    filtered_count = len(filter_data_by_period(df, analysis_period))
                    st.info(f"📊 Period: {filtered_count} transactions")
            with col4:
                csv_data = create_categorized_csv(df)
                if csv_data:
                    st.download_button(
                        label="📥 Export Categorized CSV",
                        data=csv_data,
                        file_name=f"{company_name.replace(' ', '_')}_transactions_categorized.csv",
                        mime="text/csv",
                        help="Download all transaction data with our subcategorization",
                        type="primary",
                        key="csv_export_main"
                    )
            
            # Filter data and calculate metrics
            filtered_df = filter_data_by_period(df, analysis_period)
            metrics = calculate_financial_metrics(filtered_df, params['company_age_months'])
            scores = calculate_all_scores_enhanced(metrics, params)
            revenue_insights = calculate_revenue_insights(filtered_df)

            # ENHANCED DASHBOARD RENDERING with adaptive scoring
            period_label = f"Last {analysis_period} Months" if analysis_period != 'All' else "Full Period"
            st.header(f"📊 Financial Dashboard: {company_name} ({period_label})")

            # Enhanced Key Metrics with Subprime Scoring
            col1, col2, col3, col4, col5, col6 = st.columns(6)

            with col1:
                st.metric("Original Weighted Score", f"{scores['weighted_score']:.0f}/100")

            with col2:
                if ADAPTIVE_SCORING_AVAILABLE and 'adaptive_weighted_score' in scores:
                    adaptive_score = scores['adaptive_weighted_score']
                    delta = adaptive_score - scores['weighted_score']
                    st.metric("Adaptive Weighted Score", f"{adaptive_score:.1f}%", delta=f"{delta:+.1f}")
                else:
                    st.metric("Adaptive Weighted Score", "N/A")

            with col3:
                if scores.get('adjusted_ml_score'):
                    raw_ml = scores.get('ml_score', 0)
                    adjusted_ml = scores['adjusted_ml_score']
                    delta = adjusted_ml - raw_ml
                    st.metric("Adjusted ML Probability", f"{adjusted_ml:.1f}%", delta=f"+{delta:.1f} vs raw")
                elif scores['ml_score']:
                    st.metric("ML Probability", f"{scores['ml_score']:.1f}%")
                else:
                    st.metric("ML Probability", "N/A")

            with col4:
                # NEW: Subprime Score
                subprime_score = scores['subprime_score']
                st.metric("Subprime Score", f"{subprime_score:.1f}/100")

            with col5:
                # NEW: Risk Tier
                tier = scores['subprime_tier']
                tier_colors = {
                    "Tier 1": "🟢", "Tier 2": "🟡", "Tier 3": "🟠", 
                    "Tier 4": "🔴", "Decline": "⚫"
                }
                st.metric("Risk Tier", f"{tier_colors.get(tier, '⚪')} {tier}")

            with col6:
                st.metric("Monthly Revenue", f"£{metrics.get('Monthly Average Revenue', 0):,.0f}")

            # ENHANCED SCORING COMPARISON SECTION
            if ADAPTIVE_SCORING_AVAILABLE and 'adaptive_weighted_score' in scores:
                st.markdown("---")
                
                # Create the enhanced scoring analysis section
                st.subheader("🎯 Enhanced Scoring Analysis")
                
                # Get the scores for comparison
                original_score = scores['weighted_score']
                adaptive_score = scores['adaptive_weighted_score'] 
                ml_score = scores['ml_score'] if scores['ml_score'] else 0
                scoring_details = scores.get('scoring_details', [])
                
                # Main scoring metrics display
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="🏛️ Original Weighted Score", 
                        value=f"{original_score:.0f}/100",
                        help="Traditional binary threshold-based scoring"
                    )
                
                with col2:
                    difference = adaptive_score - original_score
                    st.metric(
                        label="🧠 Adaptive Weighted Score", 
                        value=f"{adaptive_score:.1f}%",
                        delta=f"{difference:+.1f} vs Original",
                        help="Continuous scoring aligned with ML model"
                    )
                
                with col3:
                    if ml_score > 0:
                        ml_difference = abs(adaptive_score - ml_score)
                        st.metric(
                            label="🤖 ML Probability Score", 
                            value=f"{ml_score:.1f}%",
                            delta=f"±{ml_difference:.1f} vs Adaptive",
                            help="Machine learning model prediction"
                        )
                    else:
                        st.metric(
                            label="🤖 ML Probability Score", 
                            value="N/A",
                            help="ML model not available"
                        )
                
                # Score alignment analysis
                if ml_score > 0:
                    ml_adaptive_diff = abs(adaptive_score - ml_score)
                    
                    # Create alignment status with color coding
                    if ml_adaptive_diff < 10:
                        st.success(f"✅ **Excellent Alignment**: Adaptive and ML scores within {ml_adaptive_diff:.1f} points")
                        alignment_color = "🟢"
                        alignment_text = "Excellent"
                    elif ml_adaptive_diff < 15:
                        st.info(f"ℹ️ **Good Alignment**: Adaptive and ML scores within {ml_adaptive_diff:.1f} points")
                        alignment_color = "🟡"
                        alignment_text = "Good"
                    elif ml_adaptive_diff < 25:
                        st.warning(f"⚠️ **Moderate Alignment**: Adaptive and ML scores differ by {ml_adaptive_diff:.1f} points")
                        alignment_color = "🟠"
                        alignment_text = "Moderate"
                    else:
                        st.error(f"🔍 **Poor Alignment**: Adaptive and ML scores differ by {ml_adaptive_diff:.1f} points - investigate")
                        alignment_color = "🔴"
                        alignment_text = "Poor"
                    
                    # Risk level comparison with enhanced display
                    def get_risk_level(score):
                        if score >= 70: return "Low Risk", "🟢"
                        elif score >= 50: return "Medium Risk", "🟡"
                        elif score >= 30: return "High Risk", "🟠"
                        else: return "Very High Risk", "🔴"
                    
                    original_risk, original_icon = get_risk_level(original_score)
                    adaptive_risk, adaptive_icon = get_risk_level(adaptive_score)
                    ml_risk, ml_icon = get_risk_level(ml_score)
                    
                    # Enhanced risk comparison table
                    st.markdown("### 🎯 Risk Level Analysis")
                    
                    risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
                    
                    with risk_col1:
                        st.markdown(f"""
                        **🏛️ Original Method**  
                        {original_icon} **{original_risk}**  
                        Score: {original_score:.0f}/100
                        """)
                    
                    with risk_col2:
                        st.markdown(f"""
                        **🧠 Adaptive Method**  
                        {adaptive_icon} **{adaptive_risk}**  
                        Score: {adaptive_score:.1f}%
                        """)
                    
                    with risk_col3:
                        st.markdown(f"""
                        **🤖 ML Prediction**  
                        {ml_icon} **{ml_risk}**  
                        Score: {ml_score:.1f}%
                        """)
                    
                    with risk_col4:
                        st.markdown(f"""
                        **📊 Alignment**  
                        {alignment_color} **{alignment_text}**  
                        Diff: ±{ml_adaptive_diff:.1f}%
                        """)
                    
                    # Consensus analysis
                    risk_levels = [original_risk, adaptive_risk, ml_risk]
                    unique_risks = set(risk_levels)
                    
                    if len(unique_risks) == 1:
                        st.success(f"🎯 **Perfect Consensus**: All three methods agree on **{list(unique_risks)[0]}**")
                    elif adaptive_risk == ml_risk:
                        st.info(f"🤝 **Adaptive-ML Agreement**: Both advanced methods agree on **{adaptive_risk}** (Original: {original_risk})")
                    elif original_risk == ml_risk:
                        st.info(f"🤝 **Original-ML Agreement**: Traditional and ML methods agree on **{original_risk}** (Adaptive: {adaptive_risk})")
                    else:
                        st.warning(f"⚖️ **Split Decision**: No consensus - Original: {original_risk}, Adaptive: {adaptive_risk}, ML: {ml_risk}")
                
                # Score improvement analysis
                st.markdown("### 📈 Scoring Methodology Comparison")
                
                improvement_col1, improvement_col2 = st.columns(2)
                
                with improvement_col1:
                    st.markdown("""
                    **🏛️ Original Weighted Scoring:**
                    - ✅ Simple and transparent
                    - ✅ Easy to understand thresholds
                    - ❌ Binary pass/fail (harsh on borderline cases)
                    - ❌ Sharp cutoffs can cause dramatic swings
                    """)
                
                with improvement_col2:
                    st.markdown("""
                    **🧠 Adaptive Weighted Scoring:**
                    - ✅ Gradual transitions near thresholds
                    - ✅ Partial credit for near-threshold performance
                    - ✅ Better alignment with ML predictions
                    - ✅ More nuanced risk assessment
                    """)
                
                # Detailed breakdown (enhanced expandable section)
                if scoring_details:
                    with st.expander("🔍 **Detailed Adaptive Scoring Breakdown**", expanded=False):
                        st.markdown("**Component Score Analysis:**")
                        
                        # Create a more structured display of scoring details
                        detail_data = []
                        for detail in scoring_details:
                            if " → " in detail and " vs " in detail:
                                parts = detail.split(" → ")
                                if len(parts) == 2:
                                    metric_part = parts[0]
                                    score_part = parts[1].replace(" pts", "")
                                    
                                    if ": " in metric_part and " vs " in metric_part:
                                        metric_name = metric_part.split(": ")[0]
                                        values_part = metric_part.split(": ")[1]
                                        
                                        if " vs " in values_part:
                                            actual_val = values_part.split(" vs ")[0]
                                            threshold_val = values_part.split(" vs ")[1]
                                            
                                            detail_data.append({
                                                'Component': metric_name,
                                                'Actual Value': actual_val,
                                                'Threshold': threshold_val,
                                                'Points Earned': score_part
                                            })
                            else:
                                # Handle penalty entries
                                detail_data.append({
                                    'Component': detail.split(":")[0] if ":" in detail else detail,
                                    'Actual Value': '-',
                                    'Threshold': '-',
                                    'Points Earned': detail.split(":")[1] if ":" in detail else '0'
                                })
                        
                        if detail_data:
                            breakdown_df = pd.DataFrame(detail_data)
                            st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
                        else:
                            # Fallback to simple list if parsing fails
                            for detail in scoring_details:
                                st.write(f"• {detail}")
                        
                        # Summary statistics
                        total_possible = 105  # Max possible adaptive score
                        current_raw = scores.get('raw_adaptive_score', 0)
                        utilization = (current_raw / total_possible) * 100
                        
                        st.markdown(f"""
                        **📊 Scoring Summary:**
                        - **Raw Score**: {current_raw:.1f}/{total_possible} points
                        - **Utilization**: {utilization:.1f}% of maximum possible
                        - **ML-Aligned Score**: {adaptive_score:.1f}% (after sigmoid transformation)
                        """)

            # Revenue Insights
            st.markdown("---")
            st.subheader("💰 Revenue Insights")

            rev_col1, rev_col2, rev_col3, rev_col4 = st.columns(4)
            with rev_col1:
                sources_count = revenue_insights.get('unique_revenue_sources', 0)
                st.metric("Unique Revenue Sources", f"{sources_count}")
                if sources_count == 1:
                    st.warning("⚠️ Single revenue source - consider diversification")
                elif sources_count <= 3:
                    st.info("ℹ️ Limited revenue sources - moderate concentration risk")
                else:
                    st.success("✅ Good revenue diversification")
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
            st.subheader("📈 Charts & Analysis")

            # Row 1: Enhanced Score and Financial Charts
            col1, col2 = st.columns(2)
            with col1:
                fig_scores = create_score_charts(scores, metrics)
                st.plotly_chart(fig_scores, use_container_width=True, key="enhanced_scores_chart")
            with col2:
                fig_financial, fig_trend = create_financial_charts(metrics)
                st.plotly_chart(fig_financial, use_container_width=True, key="main_financial_chart")
            
            # Row 2: Trend and Threshold Charts
            col1, col2 = st.columns(2)
            with col1:
                if fig_trend:
                    st.plotly_chart(fig_trend, use_container_width=True, key="main_trend_chart")
                else:
                    st.info("Monthly trend requires multiple months of data")
            with col2:
                fig_threshold = create_threshold_chart(scores['score_breakdown'])
                st.plotly_chart(fig_threshold, use_container_width=True, key="main_threshold_chart")
            
            # Monthly Breakdown Section
            st.markdown("---")
            st.subheader("📊 Monthly Breakdown by Category")
            
            pivot_counts, pivot_amounts = create_monthly_breakdown(filtered_df)
            
            if pivot_counts is not None and not pivot_counts.empty:
                # Create monthly breakdown charts
                fig_monthly_counts, fig_monthly_amounts = create_monthly_charts(pivot_counts, pivot_amounts)
                
                # Display charts
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig_monthly_counts, use_container_width=True, key="main_monthly_counts")
                with col2:
                    st.plotly_chart(fig_monthly_amounts, use_container_width=True, key="main_monthly_amounts")
                
                # Monthly summary table
                with st.expander("📋 Detailed Monthly Breakdown", expanded=False):
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
            st.subheader("💳 Transaction Analysis")
            
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
                    title="Transaction Distribution",
                    height=300,
                    showlegend=True,
                    legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.01)
                )
                
                st.plotly_chart(fig_pie, use_container_width=True, key="main_category_pie")
            
            # NEW: Loans and Debt Repayments Analysis
            display_loans_repayments_section(filtered_df, analysis_period)

            
            # Detailed Metrics Table
            st.markdown("---")
            st.subheader("📋 Detailed Financial Metrics")
            
            # Create metrics table
            metrics_data = []
            industry_thresholds = INDUSTRY_THRESHOLDS[params['industry']]
            
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
                        'Status': '✅ Pass' if meets_threshold else '❌ Fail'
                    })
            
            if metrics_data:
                df_metrics = pd.DataFrame(metrics_data)
                st.dataframe(df_metrics, use_container_width=True, hide_index=True)
            
            # Period Comparison (if applicable)
            if analysis_period != 'All':
                st.markdown("---")
                with st.expander(f"📈 Compare with Full Period Analysis", expanded=False):
                    full_metrics = calculate_financial_metrics(df, params['company_age_months'])
                    full_scores = calculate_all_scores_enhanced(full_metrics, params)
                    
                    st.write("**Full Period vs Selected Period Comparison:**")
                    comp_col1, comp_col2, comp_col3, comp_col4 = st.columns(4)
                    
                    with comp_col1:
                        delta_weighted = scores['weighted_score'] - full_scores['weighted_score']
                        st.metric("Full Period Weighted Score", f"{full_scores['weighted_score']:.0f}/100", 
                                delta=f"{delta_weighted:+.0f} difference")
                    
                    with comp_col2:
                        if ADAPTIVE_SCORING_AVAILABLE:
                            delta_adaptive = scores.get('adaptive_weighted_score', 0) - full_scores.get('adaptive_weighted_score', 0)
                            st.metric("Full Period Adaptive Score", f"{full_scores.get('adaptive_weighted_score', 0):.1f}%",
                                    delta=f"{delta_adaptive:+.1f}% difference")
                        else:
                            st.metric("Full Period Adaptive Score", "N/A")
                    
                    with comp_col3:
                        if full_scores['ml_score'] and scores['ml_score']:
                            delta_ml = scores['ml_score'] - full_scores['ml_score']
                            st.metric("Full Period ML Probability", f"{full_scores['ml_score']:.1f}%",
                                    delta=f"{delta_ml:+.1f}% difference")
                        else:
                            st.metric("Full Period ML Probability", "N/A")
                    
                    with comp_col4:
                        delta_revenue = metrics.get('Monthly Average Revenue', 0) - full_metrics.get('Monthly Average Revenue', 0)
                        st.metric("Full Period Monthly Revenue", f"£{full_metrics.get('Monthly Average Revenue', 0):,.0f}",
                                delta=f"£{delta_revenue:+,.0f} difference")
            # NEW: Subprime Lending Analysis Section
            st.markdown("---")
            st.subheader("🎯 Subprime Lending Analysis")

            # Subprime scoring overview
            subprime_col1, subprime_col2, subprime_col3 = st.columns(3)

            with subprime_col1:
                score = scores['subprime_score']
                if score >= 65:
                    st.success(f"✅ **Excellent Subprime Candidate**\nScore: {score:.1f}/100")
                elif score >= 50:
                    st.info(f"ℹ️ **Good Subprime Candidate**\nScore: {score:.1f}/100") 
                elif score >= 35:
                    st.warning(f"⚠️ **Conditional Approval**\nScore: {score:.1f}/100")
                else:
                    st.error(f"❌ **High Risk - Review Required**\nScore: {score:.1f}/100")

            with subprime_col2:
                st.write("**Pricing Guidance:**")
                pricing = scores['subprime_pricing']
                for key, value in pricing.items():
                    if key in ['suggested_rate', 'max_loan_multiple', 'term_range']:
                        st.write(f"• **{key.replace('_', ' ').title()}**: {value}")

            with subprime_col3:
                st.write("**Monitoring Requirements:**")
                monitoring = pricing.get('monitoring', 'Standard reviews')
                approval_prob = pricing.get('approval_probability', 'Unknown')
                st.write(f"• **Monitoring**: {monitoring}")
                st.write(f"• **Approval Probability**: {approval_prob}")

            # Subprime recommendation
            st.markdown("### 📋 Subprime Lending Recommendation")
            recommendation = scores['subprime_recommendation']
            if "APPROVE" in recommendation:
                st.success(f"**Recommendation**: {recommendation}")
            elif "CONDITIONAL" in recommendation:
                st.warning(f"**Recommendation**: {recommendation}")
            elif "SENIOR REVIEW" in recommendation:
                st.info(f"**Recommendation**: {recommendation}")
            else:
                st.error(f"**Recommendation**: {recommendation}")

            # Detailed subprime breakdown
            with st.expander("🔍 **Detailed Subprime Scoring Breakdown**", expanded=False):
                st.write("**Scoring Components:**")
                for line in scores['subprime_breakdown']:
                    st.write(f"• {line}")

            # Score comparison for subprime context
            st.markdown("### 📊 All Scoring Methods Comparison (Subprime Context)")

            comparison_col1, comparison_col2 = st.columns(2)

            with comparison_col1:
                # Create comparison chart
                fig_subprime_comparison = go.Figure()
                
                score_data = {
                    'Traditional Weighted': scores['weighted_score'],
                    'Adaptive Weighted': scores.get('adaptive_weighted_score', scores['weighted_score']),
                    'ML Probability': scores['ml_score'] if scores['ml_score'] else 0,
                    'Subprime Optimized': scores['subprime_score']
                }
                
                colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']
                
                fig_subprime_comparison.add_trace(go.Bar(
                    x=list(score_data.keys()),
                    y=list(score_data.values()),
                    marker_color=colors,
                    text=[f"{v:.1f}%" for v in score_data.values()],
                    textposition='outside'
                ))
                
                fig_subprime_comparison.update_layout(
                    title="Score Comparison (Subprime Lending Context)",
                    yaxis_title="Score (%)",
                    showlegend=False,
                    height=400,
                    yaxis=dict(range=[0, 100])
                )
                
                st.plotly_chart(fig_subprime_comparison, use_container_width=True, key="subprime_comparison_chart")

            with comparison_col2:
                st.write("**Interpretation for Subprime Lending:**")
                
                # ML Score interpretation for subprime
                ml_score = scores['ml_score'] if scores['ml_score'] else 0
                if ml_score >= 25:
                    ml_interpretation = "🟢 Excellent for subprime"
                elif ml_score >= 15:
                    ml_interpretation = "🟡 Good for subprime"
                elif ml_score >= 8:
                    ml_interpretation = "🟠 Acceptable for subprime"
                elif ml_score >= 5:
                    ml_interpretation = "🔴 High-risk subprime"
                else:
                    ml_interpretation = "⚫ Too risky even for subprime"
                
                st.write(f"• **ML Model ({ml_score:.1f}%)**: {ml_interpretation}")
                
                # Subprime score interpretation
                subprime_score = scores['subprime_score']
                if subprime_score >= 65:
                    subprime_interpretation = "🟢 Premium subprime candidate"
                elif subprime_score >= 50:
                    subprime_interpretation = "🟡 Standard subprime candidate"
                elif subprime_score >= 35:
                    subprime_interpretation = "🟠 High-risk but acceptable"
                else:
                    subprime_interpretation = "🔴 Decline recommended"
                
                st.write(f"• **Subprime Score ({subprime_score:.1f})**: {subprime_interpretation}")
                
                # Score convergence analysis
                scores_list = [scores['weighted_score'], scores.get('adaptive_weighted_score', 0), ml_score, subprime_score]
                score_range = max(scores_list) - min(scores_list)
                
                if score_range <= 15:
                    convergence = "🟢 High convergence - all methods agree"
                elif score_range <= 30:
                    convergence = "🟡 Moderate convergence - some disagreement"
                else:
                    convergence = "🔴 Low convergence - significant disagreement"
                
                st.write(f"• **Score Convergence**: {convergence}")
                st.write(f"• **Score Range**: {score_range:.1f} points")
                
                # Primary recommendation
                st.markdown("**🎯 Primary Recommendation (Subprime Context):**")
                if subprime_score >= 50:
                    st.success("✅ **APPROVE** with appropriate subprime pricing")
                elif subprime_score >= 35:
                    st.warning("⚠️ **CONDITIONAL APPROVAL** with enhanced monitoring")
                else:
                    st.error("❌ **DECLINE** - Risk too high even for subprime")

            st.markdown("---")
            if ADAPTIVE_SCORING_AVAILABLE:
                st.success("🎯 Enhanced Dashboard complete with Adaptive Scoring and Subprime Analysis - All sections rendered successfully")
            else:
                st.info("🎯 Dashboard complete with Subprime Analysis - Install adaptive_score_calculation.py for enhanced scoring features")
            
            
        except Exception as e:
            st.error(f"❌ Unexpected error during processing: {e}")
    
    else:
        st.info("👆 Upload a JSON transaction file to begin analysis")

# COPY THIS ENTIRE BLOCK and paste it at the very end of your app/main.py file
# (After all the existing functions but before the "if __name__ == "__main__":" line)

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
    analysis['repayment_ratio'] = (analysis['total_repayments_made'] / analysis['total_loans_received']) if analysis['total_loans_received'] > 0 else 0
    
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

def create_loans_repayments_charts(analysis):
    """Create charts for loans and repayments analysis"""
    charts = {}
    
    # 1. Monthly Loans vs Repayments
    if not analysis['monthly_net_borrowing'].empty:
        monthly_data = analysis['monthly_net_borrowing']
        
        fig_monthly = go.Figure()
        
        fig_monthly.add_trace(go.Bar(
            name='Loans Received',
            x=monthly_data['month_str'],
            y=monthly_data['loans'],
            marker_color='lightcoral',
            opacity=0.8
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
            title="Monthly Loans vs Repayments",
            xaxis_title="Month",
            yaxis_title="Amount (£)",
            barmode='relative',
            height=400
        )
        
        charts['monthly_comparison'] = fig_monthly
    
    # 2. Loans by Lender (Top 10)
    if not analysis['loans_by_lender'].empty:
        top_lenders = analysis['loans_by_lender'].head(10)
        
        fig_lenders = go.Figure(data=[
            go.Bar(
                x=top_lenders['sum'],
                y=[name.title()[:30] + '...' if len(name) > 30 else name.title() for name in top_lenders['lender_clean']],
                orientation='h',
                marker_color='lightcoral',
                text=[f"£{amount:,.0f}" for amount in top_lenders['sum']],
                textposition='auto'
            )
        ])
        
        fig_lenders.update_layout(
            title="Loans by Lender (Top 10)",
            xaxis_title="Total Amount (£)",
            yaxis_title="Lender",
            height=400,
            yaxis=dict(autorange="reversed")
        )
        
        charts['loans_by_lender'] = fig_lenders
    
    # 3. Repayments by Recipient (Top 10)
    if not analysis['repayments_by_recipient'].empty:
        top_recipients = analysis['repayments_by_recipient'].head(10)
        
        fig_recipients = go.Figure(data=[
            go.Bar(
                x=top_recipients['sum'],
                y=[name.title()[:30] + '...' if len(name) > 30 else name.title() for name in top_recipients['recipient_clean']],
                orientation='h',
                marker_color='lightblue',
                text=[f"£{amount:,.0f}" for amount in top_recipients['sum']],
                textposition='auto'
            )
        ])
        
        fig_recipients.update_layout(
            title="Repayments by Recipient (Top 10)",
            xaxis_title="Total Amount (£)",
            yaxis_title="Recipient",
            height=400,
            yaxis=dict(autorange="reversed")
        )
        
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
        
        fig_cumulative.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        fig_cumulative.update_layout(
            title="Cumulative Net Borrowing Position",
            xaxis_title="Month",
            yaxis_title="Cumulative Amount (£)",
            height=400
        )
        
        charts['cumulative_borrowing'] = fig_cumulative
    
    return charts

def display_loans_repayments_section(df, analysis_period):
    """Display the complete loans and repayments analysis section"""
    st.markdown("---")
    st.subheader("💰 Loans and Debt Repayments Analysis")
    
    # Filter data by period if needed
    filtered_df = filter_data_by_period(df, analysis_period)
    
    # Perform analysis
    analysis = analyze_loans_and_repayments(filtered_df)
    
    # Key Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Loans Received", 
            f"£{analysis['total_loans_received']:,.0f}",
            help=f"From {analysis['loan_count']} loan transactions"
        )
    
    with col2:
        st.metric(
            "Total Repayments Made", 
            f"£{analysis['total_repayments_made']:,.0f}",
            help=f"From {analysis['repayment_count']} repayment transactions"
        )
    
    with col3:
        net_borrowing = analysis['net_borrowing']
        st.metric(
            "Net Borrowing Position", 
            f"£{abs(net_borrowing):,.0f}",
            delta="Outstanding" if net_borrowing > 0 else "Net Repaid" if net_borrowing < 0 else "Balanced"
        )
    
    with col4:
        repayment_ratio = analysis['repayment_ratio'] * 100
        st.metric(
            "Repayment Ratio", 
            f"{repayment_ratio:.1f}%",
            help="Percentage of loans that have been repaid"
        )
    
    with col5:
        avg_loan = analysis['avg_loan_amount']
        st.metric(
            "Average Loan Amount", 
            f"£{avg_loan:,.0f}" if avg_loan > 0 else "N/A"
        )
    
    # Risk Assessment Row
    st.markdown("### 🎯 Risk Assessment")
    risk_col1, risk_col2, risk_col3 = st.columns(3)
    
    with risk_col1:
        if analysis['loan_count'] == 0:
            st.info("✅ **No External Debt** - Business operates without external financing")
        elif analysis['repayment_ratio'] >= 0.8:
            st.success("✅ **Good Repayment Behavior** - Consistently repays debt obligations")
        elif analysis['repayment_ratio'] >= 0.5:
            st.warning("⚠️ **Moderate Debt Management** - Some outstanding obligations")
        else:
            st.error("🚨 **High Debt Risk** - Low repayment ratio indicates potential issues")
    
    with risk_col2:
        if analysis['loan_count'] == 0:
            st.info("📊 **No Borrowing History** - Cannot assess borrowing patterns")
        elif analysis['loan_count'] <= 3:
            st.success("📊 **Conservative Borrowing** - Infrequent use of external financing")
        elif analysis['loan_count'] <= 10:
            st.warning("📊 **Moderate Borrowing** - Regular use of external financing")
        else:
            st.error("📊 **High Borrowing Frequency** - Heavy reliance on external financing")
    
    with risk_col3:
        if analysis['net_borrowing'] <= 0:
            st.success("💰 **Positive Net Position** - More repaid than borrowed")
        elif analysis['net_borrowing'] <= analysis['total_loans_received'] * 0.3:
            st.info("💰 **Manageable Outstanding** - Reasonable debt burden")
        else:
            st.warning("💰 **High Outstanding Debt** - Significant borrowing position")
    
    # Charts Section
    if analysis['loan_count'] > 0 or analysis['repayment_count'] > 0:
        st.markdown("### 📈 Borrowing and Repayment Patterns")
        
        charts = create_loans_repayments_charts(analysis)
        
        # Row 1: Monthly comparison and cumulative position
        if 'monthly_comparison' in charts and 'cumulative_borrowing' in charts:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(charts['monthly_comparison'], use_container_width=True, key="loans_monthly_comparison")
            with col2:
                st.plotly_chart(charts['cumulative_borrowing'], use_container_width=True, key="loans_cumulative")
        
        # Row 2: Lender and recipient analysis
        chart_row2_col1, chart_row2_col2 = st.columns(2)
        
        if 'loans_by_lender' in charts and charts['loans_by_lender'] is not None:
            with chart_row2_col1:
                st.plotly_chart(charts['loans_by_lender'], use_container_width=True, key="loans_by_lender")
        else:
            with chart_row2_col1:
                st.info("No loan data available for lender analysis")
        
        if 'repayments_by_recipient' in charts and charts['repayments_by_recipient'] is not None:
            with chart_row2_col2:
                st.plotly_chart(charts['repayments_by_recipient'], use_container_width=True, key="repayments_by_recipient")
        else:
            with chart_row2_col2:
                st.info("No repayment data available for recipient analysis")
    
    # Detailed Breakdown Tables
    with st.expander("📋 Detailed Loan and Repayment Breakdown", expanded=False):
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


if __name__ == "__main__":
    main()
