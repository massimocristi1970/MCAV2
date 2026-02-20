import sys
from pathlib import Path

# Ensure repo root is on Python path (so imports from repo root work when running app/main.py)
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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
import os
import base64
from io import BytesIO

from mca_scorecard_rules import decide_application, Thresholds
from build_training_dataset import _flatten_transactions, build_mca_features

# Import modular components from pages package
# These modules contain extracted and refactored functions from this file
# For new development, prefer using these modular imports
try:
    from app.pages import (
        # Scoring functions
        calculate_weighted_scores as modular_weighted_scores,
        load_models as modular_load_models,
        calculate_subprime_score as modular_subprime_score,
        adjust_ml_score_for_growth_business as modular_ml_adjustment,
        # Chart functions
        create_score_charts as modular_score_charts,
        create_financial_charts as modular_financial_charts,
        create_loans_repayments_charts as modular_loans_charts,
        # Transaction functions - canonical implementations
        map_transaction_category as _map_transaction_category,
        categorize_transactions as modular_categorize,
        filter_data_by_period as modular_filter_period,
        calculate_financial_metrics as modular_calc_metrics,
        calculate_revenue_insights as modular_revenue_insights,
        create_categorized_csv as modular_create_csv,
        analyze_loans_and_repayments as modular_analyze_loans,
        # Report functions
        DashboardExporter as ModularDashboardExporter,
    )
    MODULAR_IMPORTS_AVAILABLE = True
except ImportError as e:
    MODULAR_IMPORTS_AVAILABLE = False
    _map_transaction_category = None  # Will use local fallback
    print(f"Note: Modular imports not available ({e}). Using inline functions.")

# Import ensemble scorer for unified recommendations
try:
    from app.services.ensemble_scorer import get_ensemble_recommendation, Decision
    ENSEMBLE_SCORER_AVAILABLE = True
except ImportError as e:
    ENSEMBLE_SCORER_AVAILABLE = False
    get_ensemble_recommendation = None
    Decision = None
    print(f"Note: Ensemble scorer not available ({e}).")


# Debug mode - only enabled when DEBUG environment variable is set to 'true'
DEBUG_MODE = os.environ.get('DEBUG', 'false').lower() == 'true'

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
            self.scaler = joblib.load('scaler.pkl')
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
            
            print("✅ ML validation available (calibrated for your training data)")
            
        except Exception as e:
            self.has_scaler = False
            print(f"ℹ️ ML validation not available: {e}")
    
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
                'Sector_Risk': 1 if params.get('industry', '') in ['Restaurants and Cafes', 'Bars and Pubs', 'Other'] else 0
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
            recommendations.append("✅ Some differences indicate healthier metrics than training data")
        
        if concerning_outliers and len(concerning_outliers) <= 2:
            recommendations.append("⚠️ A few metrics need verification")
        elif len(concerning_outliers) > 2:
            recommendations.append("🔍 Multiple metrics need review")
        
        if avg_z <= 2.0:
            recommendations.append("💡 Business profile more stable than much of training data")
        elif avg_z > 4.0:
            recommendations.append("⚠️ Business very different - use rule-based scores")
        
        # ML usage guidance
        if avg_z <= 2.0 and len(concerning_outliers) <= 1:
            recommendations.append("✅ ML score likely reliable despite training data issues")
        else:
            recommendations.append("⚠️ Prioritize subprime and weighted scores over ML score")
        
        return recommendations

       

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
    "business_ccj": 5, "director_ccj": 3,
    'poor_or_no_online_presence': 3, 'uses_generic_email': 1
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
    
    # Fallback implementation
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
                
                # Safe formatting with None check
                if revenue_growth_rate is not None:
                    print(f"    Median change: {revenue_growth_rate:.3f} ({revenue_growth_rate*100:.1f}%)")
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
            failed_payment_keywords = ['unpaid', 'returned', 'bounced', 'insufficient', 'failed', 'declined', 'nsf', 'unp']
            for keyword in failed_payment_keywords:
                bounced_payments += data['name_y'].str.contains(keyword, case=False, na=False).sum()
        
        # DEBUGGING: Print key values
        print(f"\n🔍 DEBUG - Financial Metrics:")
        print(f"  Total Revenue: £{total_revenue:,.2f}" if total_revenue is not None else "  Total Revenue: N/A")
        print(f"  Total Expenses: £{total_expenses:,.2f}" if total_expenses is not None else "  Total Expenses: N/A")
        print(f"  Net Income: £{net_income:,.2f}" if net_income is not None else "  Net Income: N/A")
        print(f"  DSCR: {debt_service_coverage_ratio:.2f}" if debt_service_coverage_ratio is not None else "  DSCR: N/A")
        print(f"  Operating Margin: {operating_margin:.3f} ({operating_margin*100:.1f}%)" if operating_margin is not None else "  Operating Margin: N/A")
        print(f"  Cash Flow Volatility: {cash_flow_volatility:.3f}" if cash_flow_volatility is not None else "  Cash Flow Volatility: N/A")
        print(f"  Revenue Growth Rate: {revenue_growth_rate:.2f}%" if revenue_growth_rate is not None else "  Revenue Growth Rate: N/A")
        print(f"  Avg Month-End Balance: £{avg_month_end_balance:,.2f}" if avg_month_end_balance is not None else "  Avg Month-End Balance: N/A")
        print(f"  Avg Negative Days: {avg_negative_days}" if avg_negative_days is not None else "  Avg Negative Days: N/A")
        print(f"  Bounced Payments: {bounced_payments}" if bounced_payments is not None else "  Bounced Payments: N/A")
        
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
            # Weights based on predictive power:
            # - MCA Rule (transaction consistency): 40% - empirically validated
            # - Subprime Score: 40% - comprehensive micro-enterprise assessment  
            # - ML Score: 20% - data-driven probability (retrained, 0.922 ROC-AUC)
            ensemble_scores = {
                'subprime_score': subprime_result['subprime_score'],
                'ml_score': adjusted_ml_score or ml_score,
                'mca_score': mca_rule_score,
                'mca_decision': mca_rule_decision,
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
        'ensemble': ensemble_result
    }

def adjust_ml_score_for_growth_business(raw_ml_score, metrics, params):
    """
    Adjust ML score for growth businesses that the traditional model undervalues.
    """
    
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
    
    # Apply adjustment with cap
    adjusted_score = min(50, raw_ml_score + adjustment)  # Cap at 85%
    
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
        color = "🟢"
    elif adjusted_score >= 70:
        risk_level = "Moderate Risk" 
        color = "🟡"
    elif adjusted_score >= 60:
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
    """Create clean bar charts for scores - Updated for 3 scoring methods"""
    
    # Score comparison chart
    fig_scores = go.Figure()
    
    score_data = {
        'Subprime Score': scores.get('subprime_score', 0),
        'MCA Rule': scores.get('mca_rule_score', 0),
        'Adjusted ML': scores.get('adjusted_ml_score', scores.get('ml_score', 0))
    }
    
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e']  # Green, Blue, Orange
    
    fig_scores.add_trace(go.Bar(
        x=list(score_data.keys()),
        y=list(score_data.values()),
        marker_color=colors,
        text=[f"{v:.1f}" + ("%" if k == "Adjusted ML" else "/100") for k, v in score_data.items()],
        textposition='outside'
    ))
    
    fig_scores.update_layout(
        title="Primary Scoring Methods Comparison",
        yaxis_title="Score",
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
                'risk_factors':  {
                    'business_ccj': params. get('business_ccj', False),
                    'director_ccj': params.get('director_ccj', False),
                    'poor_or_no_online_presence': params.get('poor_or_no_online_presence', False),
                    'uses_generic_email': params.get('uses_generic_email', False)
                }
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
                'loan_risk': scores.get('loan_risk')
            },
            'revenue_insights': revenue_insights,
            'loans_analysis': loans_analysis or {}
        }
        
        return export_data
    
    def generate_html_report(self, export_data: dict) -> str:
        """Generate comprehensive HTML report."""
        
        # Helper function for score styling
        def get_score_class(score):
            if score >= 70:
                return "high"
            elif score >= 40:
                return "medium"
            else:
                return "low"
        
        # Generate loans section HTML if data exists
        loans_section = ""
        if export_data['loans_analysis'] and export_data['loans_analysis'].get('loan_count', 0) > 0:
            loans_section = f"""
            <div class="section">
                <h2>💰 Loans & Debt Analysis</h2>
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
                        <div>{export_data['loans_analysis'].get('repayment_ratio', 0)*100:.1f}%</div>
                    </div>
                </div>
            </div>
            """
        
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
                <h1>🏦 Business Finance Scorecard Report</h1>
                <h2>{export_data['export_info']['company_name']}</h2>
                <p><strong>Generated:</strong> {datetime.fromisoformat(export_data['export_info']['export_timestamp']).strftime('%B %d, %Y at %I:%M %p')}</p>
                <p><strong>Analysis Period:</strong> {export_data['export_info']['analysis_period']}</p>
                <p><strong>Industry:</strong> {export_data['business_parameters']['industry']}</p>
            </div>
            
            <!-- Executive Summary -->
            <div class="section">
                <h2>📊 Executive Summary</h2>
                <div class="metric-grid">
                    <div class="metric-card">
                        <h3>🎯 Subprime Score</h3>
                        <div class="score-{get_score_class(export_data['scoring_results']['subprime_score'])}">{export_data['scoring_results']['subprime_score']:.1f}/100</div>
                        <p>{export_data['scoring_results']['subprime_tier']}</p>
                    </div>
                    <div class="metric-card">
                        <h3>🏛️ MCA Rule (40%)</h3>
                        <div class="score-{get_score_class(export_data['scoring_results'].get('mca_rule_score', 0))}">{export_data['scoring_results'].get('mca_rule_score', 0):.0f}/100</div>
                    </div>
                    <div class="metric-card">
                        <h3>🤖 ML Score</h3>
                        <div class="score-{get_score_class(export_data['scoring_results'].get('adjusted_ml_score', 0))}">{export_data['scoring_results'].get('adjusted_ml_score', 0):.1f}%</div>
                    </div>
                    <div class="metric-card">
                        <h3>💰 Requested Loan</h3>
                        <div>£{export_data['business_parameters']['requested_loan']:,.0f}</div>
                        <p>{export_data['scoring_results']['loan_risk']}</p>
                    </div>
                    <div class="metric-card">
                        <h3>📌 MCA Rule</h3>
                        <div>{export_data['scoring_results'].get('mca_rule_decision', 'N/A')}</div>
                        <p>Score: {export_data['scoring_results'].get('mca_rule_score', 'N/A')}</p>
                    </div>
                </div>
                
                <h3>📋 Primary Recommendation</h3>
                <p><strong>{export_data['scoring_results']['subprime_recommendation']}</strong></p>

                <h3>📌 MCA Rule Decision (Transparent)</h3>
                <p><strong>{export_data['scoring_results'].get('mca_rule_decision', 'N/A')}</strong>
                &nbsp;&nbsp;|&nbsp;&nbsp; Score: <strong>{export_data['scoring_results'].get('mca_rule_score', 'N/A')}</strong></p>

                <p><strong>Reasons:</strong></p>
                <ul>
                {''.join([f"<li>{r}</li>" for r in export_data['scoring_results'].get('mca_rule_reasons', [])])}
                </ul>

                <h3>📌 Decision Stack Summary</h3>
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
                        <td>MCA Rule (40%)</td>
                        <td>{export_data['scoring_results'].get('mca_rule_score', 'N/A')}/100</td>
                    </tr>
                    <tr>
                        <td>ML Score</td>
                        <td>{export_data['scoring_results'].get('adjusted_ml_score', export_data['scoring_results'].get('ml_score', 'N/A'))}</td>
                    </tr>
                    <tr>
                        <td>Requested Loan</td>
                        <td>£{export_data['business_parameters'].get('requested_loan', 0):,.0f}</td>
                    </tr>
                </table>

                </div>

            
            <!-- Financial Metrics -->
            <div class="section">
                <h2>💰 Financial Performance</h2>
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
                <h2>📈 Revenue Insights</h2>
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
                <h2>⚠️ Risk Factors Assessment</h2>
                <div class="table-responsive">
                    <table>
                        <tr><th>Risk Factor</th><th>Status</th></tr>
                        <tr><td>Business CCJs</td><td>{'❌ Yes' if export_data['business_parameters']['risk_factors']['business_ccj'] else '✅ No'}</td></tr>
                        <tr><td>Director CCJs</td><td>{'❌ Yes' if export_data['business_parameters']['risk_factors']['director_ccj'] else '✅ No'}</td></tr>
                        <tr><td>Poor/No Online Presence</td><td>{'❌ Yes' if export_data['business_parameters']['risk_factors']['poor_or_no_online_presence'] else '✅ No'}</td></tr>
                        <tr><td>Generic Email</td><td>{'❌ Yes' if export_data['business_parameters']['risk_factors']['uses_generic_email'] else '✅ No'}</td></tr>
                    </table>
                </div>
            </div>
            
            <!-- Business Parameters -->
            <div class="section">
                <h2>🏢 Business Information</h2>
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
        st.subheader("📥 Export Dashboard Report")
        
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
                label="📄 Export HTML Report",
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
                label="📊 Export JSON Data",
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
                label="📈 Export CSV Metrics",
                data=csv_data,
                file_name=f"{company_name.replace(' ', '_')}_metrics_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                help="Download financial metrics as CSV"
            )
        
        # Export information
        st.info(f"""
        **📊 Export Options Available:**
        - **HTML Report**: ✅ Complete dashboard in professional web format
        - **JSON Data**: ✅ All data for external analysis and integration
        - **CSV Metrics**: ✅ Financial metrics for spreadsheet analysis
        
        **Export includes**: All scoring results, financial metrics, revenue insights, risk factors, loans analysis, and business parameters.
        """)

def main():
    """Main application"""
    try:
        st.title("Business Finance Scorecard")
        st.markdown("---")
           
        # Sidebar inputs
        st.sidebar.header("Business Parameters")
        
        company_name = st.sidebar.text_input("Company Name", "Sample Business Ltd")
        industry = st.sidebar.selectbox("Industry", list(INDUSTRY_THRESHOLDS.keys()))
        requested_loan = st.sidebar.number_input("Requested Loan (£)", min_value=0.0, value=5000.0, step=1000.0)
        directors_score = st.sidebar.slider("Director Credit Score", 0, 100, 75)
        company_age_months = st.sidebar.number_input("Company Age (Months)", min_value=0, value=12, step=1)

        st.sidebar.subheader("Risk Factors")
        business_ccj = st.sidebar.checkbox("Business CCJs")
        director_ccj = st.sidebar.checkbox("Director CCJs")
        poor_or_no_online_presence = st.sidebar.checkbox("Poor/No Online Presence")
        uses_generic_email = st.sidebar.checkbox("Generic Email")
        
        # Time period filter
        st.sidebar.subheader("Analysis Period")
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
            'business_ccj': business_ccj,
            'director_ccj': director_ccj,
            'poor_or_no_online_presence': poor_or_no_online_presence,
            'uses_generic_email': uses_generic_email
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

                # Handle both formats:  direct list OR dictionary with 'transactions' key
                if isinstance(json_data, list):
                    transactions = json_data  # Direct list format
                elif isinstance(json_data, dict):
                    transactions = json_data.get('transactions', [])  # Dictionary format
                else:
                    st.error("❌ Unexpected JSON format - expected list or dictionary")
                    return

                if not transactions:
                    st.error("❌ No transactions found in JSON file")
                    return

                # --- MCA rule-based decision (new) ---
                txns_for_scoring = _flatten_transactions(transactions)
                mca_features = build_mca_features(txns_for_scoring)
                mca_decision, mca_score, mca_reasons = decide_application(mca_features, t=Thresholds())

                # Store for later display / export (no other logic changes)
                params["mca_rule_decision"] = mca_decision
                params["mca_rule_score"] = mca_score
                params["mca_rule_reasons"] = mca_reasons
                params["mca_rule_signals"] = {
                    "inflow_days_30d": mca_features.get("inflow_days_30d"),
                    "max_inflow_gap_days": mca_features.get("max_inflow_gap_days"),
                    "inflow_cv": mca_features.get("inflow_cv"),
                    "months_covered": mca_features.get("months_covered"),
                    "txn_count_avg_month": mca_features.get("txn_count_avg_month"),
                }

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

                # ensure MCA rule outputs are part of scoring_results for export
                scores["mca_rule_decision"] = params.get("mca_rule_decision")
                scores["mca_rule_score"] = params.get("mca_rule_score")
                scores["mca_rule_reasons"] = params.get("mca_rule_reasons", [])

                # -------------------------------
                # FINAL DECISION RULES (NEW)
                # -------------------------------
                def _base_decision_from_subprime(recommendation_text: str) -> str:
                    s = (recommendation_text or "").upper()
                    if "APPROVE" in s:
                        return "APPROVE"
                    # Treat conditional / senior review as REFER
                    if "CONDITIONAL" in s or "SENIOR REVIEW" in s or "REVIEW" in s:
                        return "REFER"
                    return "DECLINE"

                base_decision = _base_decision_from_subprime(scores.get("subprime_recommendation", ""))
                mca_decision = (scores.get("mca_rule_decision") or "").upper().strip()

                final_decision = base_decision
                final_reasons = [f"Base decision from Subprime: {base_decision}"]

                if mca_decision == "DECLINE":
                    final_decision = "DECLINE"
                    final_reasons.append("MCA Rule override: DECLINE (hard stop)")
                elif mca_decision == "REFER" and base_decision != "DECLINE":
                    final_decision = "REFER"
                    final_reasons.append("MCA Rule override: REFER (manual review)")
                elif mca_decision == "APPROVE":
                    final_reasons.append("MCA Rule: APPROVE (no override)")

                # Persist for UI + exports
                scores["final_decision"] = final_decision
                scores["final_decision_reasons"] = final_reasons
                params["final_decision"] = final_decision
                params["final_decision_reasons"] = final_reasons

                revenue_insights = calculate_revenue_insights(filtered_df)

                # ENHANCED DASHBOARD RENDERING
                period_label = f"Last {analysis_period} Months" if analysis_period != 'All' else "Full Period"
                st.header(f"Financial Dashboard: {company_name} ({period_label})")

                # ============================================
                # 🎯 UNIFIED RECOMMENDATION (TOP OF DASHBOARD)
                # ============================================
                ensemble = scores.get('ensemble')
                if ensemble:
                    decision = ensemble.get('decision', 'REFER')
                    combined_score = ensemble.get('combined_score', 0)
                    confidence = ensemble.get('confidence', 0)
                    
                    # Main recommendation display with prominent styling
                    if decision == 'APPROVE':
                        st.success(f"""
                        ## 🎯 Recommendation: ✅ APPROVE
                        **Combined Score:** {combined_score:.1f}/100  |  **Confidence:** {confidence:.0f}%
                        
                        *{ensemble.get('primary_reason', '')}*
                        """)
                    elif decision == 'CONDITIONAL_APPROVE':
                        st.info(f"""
                        ## 🎯 Recommendation: ℹ️ CONDITIONAL APPROVE
                        **Combined Score:** {combined_score:.1f}/100  |  **Confidence:** {confidence:.0f}%
                        
                        *{ensemble.get('primary_reason', '')}*
                        """)
                    elif decision == 'REFER':
                        st.warning(f"""
                        ## 🎯 Recommendation: ⚠️ REFER FOR REVIEW
                        **Combined Score:** {combined_score:.1f}/100  |  **Confidence:** {confidence:.0f}%
                        
                        *{ensemble.get('primary_reason', '')}*
                        """)
                    elif decision == 'SENIOR_REVIEW':
                        st.warning(f"""
                        ## 🎯 Recommendation: 🔍 SENIOR REVIEW REQUIRED
                        **Combined Score:** {combined_score:.1f}/100  |  **Confidence:** {confidence:.0f}%
                        
                        *{ensemble.get('primary_reason', '')}*
                        """)
                    else:  # DECLINE
                        st.error(f"""
                        ## 🎯 Recommendation: ❌ DECLINE
                        **Combined Score:** {combined_score:.1f}/100  |  **Confidence:** {confidence:.0f}%
                        
                        *{ensemble.get('primary_reason', '')}*
                        """)
                    
                    # Contributing scores in compact row
                    contributing = ensemble.get('contributing_scores', {})
                    score_cols = st.columns(3)

                    with score_cols[0]:
                        mca_s = contributing.get('mca_score', params.get('mca_rule_score', 50))
                        st.metric("MCA Rule (40%)", f"{mca_s:.0f}")
                    
                    with score_cols[1]:
                        subprime_s = contributing.get('subprime_score', scores.get('subprime_score', 0))
                        st.metric("Subprime (40%)", f"{subprime_s:.1f}")
                    
                    with score_cols[2]:
                        ml_s = contributing.get('ml_score', scores.get('adjusted_ml_score') or scores.get('ml_score') or 0)
                        st.metric("ML Score (20%)", f"{ml_s:.1f}%" if ml_s else "N/A")
                    
                    # Score convergence indicator
                    convergence = ensemble.get('score_convergence', 'Unknown')
                    if 'High' in convergence:
                        st.success(f"📊 **Score Convergence:** {convergence} - All scoring methods agree")
                    elif 'Good' in convergence:
                        st.info(f"📊 **Score Convergence:** {convergence}")
                    elif 'Moderate' in convergence:
                        st.warning(f"📊 **Score Convergence:** {convergence} - Some disagreement between methods")
                    else:
                        st.error(f"📊 **Score Convergence:** {convergence} - Significant disagreement")
                    
                    # Pricing and details in expander
                    with st.expander("📋 Pricing Guidance & Risk Analysis", expanded=False):
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
                            st.markdown("**⚠️ Risk Factors:**")
                            risk_factors = ensemble.get('risk_factors', [])
                            if risk_factors:
                                for rf in risk_factors:
                                    st.write(f"• {rf}")
                            else:
                                st.write("• No significant risk factors identified")
                        
                        with positive_col:
                            st.markdown("**✅ Positive Factors:**")
                            positive_factors = ensemble.get('positive_factors', [])
                            if positive_factors:
                                for pf in positive_factors:
                                    st.write(f"• {pf}")
                            else:
                                st.write("• No notable positive factors")
                        
                        st.markdown("---")
                        st.markdown("**📝 Recommendations:**")
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
                st.subheader("Charts & Analysis")

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
                st.subheader("Monthly Breakdown by Category")
                
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
                st.subheader("Detailed Financial Metrics")
                
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
                    with st.expander(f"Compare with Full Period Analysis", expanded=False):
                        full_metrics = calculate_financial_metrics(df, params['company_age_months'])
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
                # 📊 DETAILED SCORING ANALYSIS (Consolidated)
                # ============================================
                st.markdown("---")
                with st.expander("📊 Detailed Scoring Analysis", expanded=False):
                    # Subprime scoring overview
                    subprime_col1, subprime_col2, subprime_col3 = st.columns(3)

                    with subprime_col1:
                        score = scores['subprime_score']
                        if score >= 75:
                            st.success(f"✅ **Excellent Candidate**\nScore: {score:.1f}/100")
                        elif score >= 60:
                            st.info(f"ℹ️ **Good Candidate**\nScore: {score:.1f}/100") 
                        elif score >= 45:
                            st.warning(f"⚠️ **Conditional**\nScore: {score:.1f}/100")
                        elif score >= 30:
                            st.warning(f"⚠️ **High Monitoring**\nScore: {score:.1f}/100")    
                        else:
                            st.error(f"❌ **High Risk**\nScore: {score:.1f}/100")

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
                            status = '✅ Pass' if data['meets'] else '❌ Fail'
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
                        st.markdown("**📈 Metric Performance:**")
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
                                
                                status_emoji = {'PASS': '✅', 'PARTIAL': '🟡', 'FAIL': '❌'}
                                
                                metric_data.append({
                                    'Metric': metric['metric'],
                                    'Actual': actual_str,
                                    'Target': threshold_str,
                                    'Points': f"{metric['points_earned']:.1f}/{metric['points_possible']}",
                                    'Status': f"{status_emoji.get(metric['status'], '⚪')}"
                                })
                            
                            if metric_data:
                                df_metrics = pd.DataFrame(metric_data)
                                st.dataframe(df_metrics, use_container_width=True, hide_index=True)
                        
                        # Key factors in columns
                        if diagnostics.get('top_negative_factors') or diagnostics.get('top_positive_factors'):
                            neg_col, pos_col = st.columns(2)
                            
                            with neg_col:
                                if diagnostics.get('top_negative_factors'):
                                    st.markdown("**🔴 Top Risk Factors:**")
                                    for factor in diagnostics['top_negative_factors'][:3]:
                                        st.write(f"• {factor['metric']}: -{factor['points_lost']:.1f} pts")
                            
                            with pos_col:
                                if diagnostics.get('top_positive_factors'):
                                    st.markdown("**🟢 Top Strengths:**")
                                    for factor in diagnostics['top_positive_factors'][:3]:
                                        st.write(f"• {factor['metric']}: +{factor['points_earned']:.1f} pts")
                        
                        # Improvement suggestions
                        if diagnostics.get('improvement_suggestions'):
                            st.markdown("**💡 Improvements:**")
                            for suggestion in diagnostics['improvement_suggestions'][:3]:
                                st.info(f"• {suggestion}")
                    
                    # ML Validation (if available)
                    ml_validation = scores.get('ml_validation', {})
                    if ml_validation.get('available', False):
                        st.markdown("---")
                        st.markdown("**🤖 ML Score Reliability:**")
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
                    st.error(f"❌ Export functionality error: {str(e)}")
                    st.success("Enhanced Dashboard complete (export disabled due to error)")
                    
            except Exception as e:
                st.error(f"❌ Unexpected error during processing: {e}")
                import traceback
                full_traceback = traceback.format_exc()
                st.code(full_traceback)
                print(full_traceback)
        
        else:
            st.info("Upload a JSON transaction file to begin analysis")

    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        import traceback
        full_traceback = traceback.format_exc()
        st.code(full_traceback)
        print(full_traceback)

if __name__ == "__main__":
    main()
# ADD THIS LINE AT THE VERY END OF THE FILE:
SubprimeScoringSystem = SubprimeScoring