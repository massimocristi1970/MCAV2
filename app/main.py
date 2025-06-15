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

# NEW IMPORT: Add adaptive scoring module
try:
    from adaptive_score_calculation import get_improved_weighted_score, get_detailed_scoring_breakdown
    ADAPTIVE_SCORING_AVAILABLE = True
except ImportError:
    ADAPTIVE_SCORING_AVAILABLE = False
    print("Adaptive scoring module not found - using original weighted score")

st.set_page_config(
    page_title="Business Finance Scorecard",
    layout="wide"
)

# Industry thresholds
INDUSTRY_THRESHOLDS = {
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
    'Restaurants and Cafes': {
        'Debt Service Coverage Ratio': 1.30, 'Net Income': 0, 'Operating Margin': 0.05,
        'Revenue Growth Rate': 0.04, 'Cash Flow Volatility': 0.16, 'Gross Burn Rate': 11000,
        'Directors Score': 75, 'Sector Risk': 1, 'Average Month-End Balance': 600,
        'Average Negative Balance Days per Month': 2, 'Number of Bounced Payments': 0,
    },
    'Retail': {
        'Debt Service Coverage Ratio': 1.40, 'Net Income': 2500, 'Operating Margin': 0.09,
        'Revenue Growth Rate': 0.04, 'Cash Flow Volatility': 0.14, 'Gross Burn Rate': 11500,
        'Directors Score': 75, 'Sector Risk': 1, 'Average Month-End Balance': 620,
        'Average Negative Balance Days per Month': 1, 'Number of Bounced Payments': 0,
    },
    'Construction Firms': {
        'Debt Service Coverage Ratio': 1.30, 'Net Income': 1000, 'Operating Margin': 0.08,
        'Revenue Growth Rate': 0.04, 'Cash Flow Volatility': 0.16, 'Gross Burn Rate': 12500,
        'Directors Score': 75, 'Sector Risk': 1, 'Average Month-End Balance': 700,
        'Average Negative Balance Days per Month': 2, 'Number of Bounced Payments': 0,
    },
    'Other': {
        'Debt Service Coverage Ratio': 1.50, 'Net Income': 3000, 'Operating Margin': 0.08,
        'Revenue Growth Rate': 0.05, 'Cash Flow Volatility': 0.13, 'Gross Burn Rate': 11000,
        'Directors Score': 75, 'Sector Risk': 1, 'Average Month-End Balance': 700,
        'Average Negative Balance Days per Month': 1, 'Number of Bounced Payments': 0,
    }
}

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
        adaptive_weighted_score, raw_adaptive_score, scoring_details = get_detailed_scoring_breakdown(
            metrics, params['directors_score'], industry_thresholds['Sector Risk'], industry_thresholds, 
            params['company_age_months'], params.get('personal_default_12m', False), 
            params.get('business_ccj', False), params.get('director_ccj', False), 
            params.get('website_or_social_outdated', False), params.get('uses_generic_email', False), 
            params.get('no_online_presence', False), PENALTIES
        )
        return original_weighted_score, adaptive_weighted_score, raw_adaptive_score, scoring_details
    else:
        return original_weighted_score, original_weighted_score, original_weighted_score, []

def display_scoring_comparison(original_score, adaptive_score, ml_probability, scoring_details=None):
    """Display comparison between different scoring methods"""
    
    st.subheader("📊 Enhanced Scoring Analysis")
    
    # Create columns for score display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Original Weighted Score", 
            value=f"{original_score:.1f}",
            help="Traditional binary threshold-based scoring"
        )
    
    with col2:
        if ADAPTIVE_SCORING_AVAILABLE:
            difference = adaptive_score - original_score
            st.metric(
                label="Adaptive Weighted Score", 
                value=f"{adaptive_score:.1f}%",
                delta=f"{difference:+.1f} vs Original",
                help="Continuous scoring aligned with ML model"
            )
        else:
            st.metric(
                label="Adaptive Weighted Score", 
                value="Not Available",
                help="Install adaptive_score_calculation.py to enable"
            )
    
    with col3:
        if ADAPTIVE_SCORING_AVAILABLE and ml_probability:
            ml_difference = abs(adaptive_score - ml_probability)
            st.metric(
                label="ML Probability Score", 
                value=f"{ml_probability:.1f}%",
                delta=f"±{ml_difference:.1f} vs Adaptive",
                help="Machine learning model prediction"
            )
        else:
            st.metric(
                label="ML Probability Score", 
                value=f"{ml_probability:.1f}%" if ml_probability else "N/A",
                help="Machine learning model prediction"
            )
    
    # Score alignment analysis
    if ADAPTIVE_SCORING_AVAILABLE and ml_probability:
        ml_adaptive_diff = abs(adaptive_score - ml_probability)
        
        if ml_adaptive_diff < 10:
            st.success(f"✅ Excellent alignment: Adaptive and ML scores within {ml_adaptive_diff:.1f} points")
        elif ml_adaptive_diff < 15:
            st.info(f"ℹ️ Good alignment: Adaptive and ML scores within {ml_adaptive_diff:.1f} points")
        elif ml_adaptive_diff < 25:
            st.warning(f"⚠️ Moderate alignment: Adaptive and ML scores differ by {ml_adaptive_diff:.1f} points")
        else:
            st.error(f"🔍 Poor alignment: Adaptive and ML scores differ by {ml_adaptive_diff:.1f} points - investigate")
        
        # Risk level comparison
        def get_risk_level(score):
            if score >= 70: return "Low Risk", "🟢"
            elif score >= 50: return "Medium Risk", "🟡"
            elif score >= 30: return "High Risk", "🟠"
            else: return "Very High Risk", "🔴"
        
        original_risk, original_icon = get_risk_level(original_score)
        adaptive_risk, adaptive_icon = get_risk_level(adaptive_score)
        ml_risk, ml_icon = get_risk_level(ml_probability)
        
        st.write("**Risk Level Comparison:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"{original_icon} Original: {original_risk}")
        with col2:
            st.write(f"{adaptive_icon} Adaptive: {adaptive_risk}")
        with col3:
            st.write(f"{ml_icon} ML Model: {ml_risk}")
        
        if adaptive_risk == ml_risk:
            st.success("✅ Adaptive and ML models agree on risk level")
        else:
            st.info("ℹ️ Risk level difference detected")
        
        # Detailed breakdown (expandable)
        if scoring_details:
            with st.expander("🔍 Detailed Adaptive Scoring Breakdown"):
                st.write("**Component Scores:**")
                for detail in scoring_details:
                    st.write(f"• {detail}")

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
    """Calculate comprehensive financial metrics"""
    if data.empty:
        return {}
    
    try:
        data = categorize_transactions(data)
        
        # Basic calculations
        total_revenue = abs(data.loc[data['is_revenue'], 'amount'].sum())
        total_expenses = abs(data.loc[data['is_expense'], 'amount'].sum())
        net_income = total_revenue - total_expenses
        total_debt_repayments = abs(data.loc[data['is_debt_repayment'], 'amount'].sum())
        total_debt = abs(data.loc[data['is_debt'], 'amount'].sum())
        
        # Time-based calculations
        data['date'] = pd.to_datetime(data['date'])
        data['year_month'] = data['date'].dt.to_period('M')
        unique_months = data['year_month'].nunique()
        months_count = max(unique_months, 1)
        
        monthly_avg_revenue = total_revenue / months_count
        
        # Financial ratios
        debt_to_income_ratio = (total_debt / total_revenue) if total_revenue > 0 else 0
        expense_to_revenue_ratio = (total_expenses / total_revenue) if total_revenue > 0 else 0
        operating_margin = (net_income / total_revenue) if total_revenue > 0 else 0
        debt_service_coverage_ratio = (total_revenue / total_debt_repayments) if total_debt_repayments > 0 else 0
        
        # Monthly analysis
        monthly_summary = data.groupby('year_month').agg({
            'amount': [
                lambda x: abs(x[data.loc[x.index, 'is_revenue']].sum()),
                lambda x: abs(x[data.loc[x.index, 'is_expense']].sum())
            ]
        }).round(2)
        
        monthly_summary.columns = ['monthly_revenue', 'monthly_expenses']
        
        # Volatility metrics
        if len(monthly_summary) > 1:
            revenue_mean = monthly_summary['monthly_revenue'].mean()
            cash_flow_volatility = (monthly_summary['monthly_revenue'].std() / revenue_mean) if revenue_mean > 0 else 0.1
            revenue_growth_rate = monthly_summary['monthly_revenue'].pct_change().median() * 100
            gross_burn_rate = monthly_summary['monthly_expenses'].mean()
        else:
            cash_flow_volatility = 0.1
            revenue_growth_rate = 5.0
            gross_burn_rate = total_expenses / months_count
        
        # Balance metrics (simplified for demo)
        avg_month_end_balance = max(1000, total_revenue * 0.1)
        avg_negative_days = min(2, max(0, int(cash_flow_volatility * 10)))
        bounced_payments = 0
        
        return {
            "Total Revenue": round(total_revenue, 2),
            "Monthly Average Revenue": round(monthly_avg_revenue, 2),
            "Total Expenses": round(total_expenses, 2),
            "Net Income": round(net_income, 2),
            "Total Debt Repayments": round(total_debt_repayments, 2),
            "Total Debt": round(total_debt, 2),
            "Debt-to-Income Ratio": round(debt_to_income_ratio, 2),
            "Expense-to-Revenue Ratio": round(expense_to_revenue_ratio, 2),
            "Operating Margin": round(operating_margin, 2),
            "Debt Service Coverage Ratio": round(debt_service_coverage_ratio, 2),
            "Gross Burn Rate": round(gross_burn_rate, 2),
            "Cash Flow Volatility": round(cash_flow_volatility, 2),
            "Revenue Growth Rate": round(revenue_growth_rate if not pd.isna(revenue_growth_rate) else 0, 2),
            "Average Month-End Balance": round(avg_month_end_balance, 2),
            "Average Negative Balance Days per Month": avg_negative_days,
            "Number of Bounced Payments": bounced_payments,
            "monthly_summary": monthly_summary
        }
        
    except Exception as e:
        st.error(f"Error calculating metrics: {e}")
        return {}

def calculate_all_scores(metrics, params):
    """Calculate all scoring methods - ENHANCED with adaptive scoring"""
    industry_thresholds = INDUSTRY_THRESHOLDS[params['industry']]
    sector_risk = industry_thresholds['Sector Risk']
    
    # Calculate both weighted scores (original and adaptive)
    original_weighted_score, adaptive_weighted_score, raw_adaptive_score, scoring_details = calculate_both_weighted_scores(
        metrics, params, industry_thresholds
    )
    
    # Industry Score (binary)
    industry_score = 0
    score_breakdown = {}
    
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
    
    # ML Score (if available)
    model, scaler = load_models()
    ml_score = None
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
        except:
            pass
    
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
    
    return {
        'weighted_score': original_weighted_score,
        'adaptive_weighted_score': adaptive_weighted_score,
        'raw_adaptive_score': raw_adaptive_score,
        'scoring_details': scoring_details,
        'industry_score': industry_score,
        'ml_score': ml_score,
        'loan_risk': loan_risk,
        'score_breakdown': score_breakdown
    }

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
        'ML Probability': scores['ml_score'] if scores['ml_score'] else 0
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
            
            # FIND THIS SECTION in your main.py (around line 650-700) and REPLACE IT:

# Filter data and calculate metrics
filtered_df = filter_data_by_period(df, analysis_period)
metrics = calculate_financial_metrics(filtered_df, params['company_age_months'])
scores = calculate_all_scores(metrics, params)
revenue_insights = calculate_revenue_insights(filtered_df)

# ENHANCED DASHBOARD RENDERING with adaptive scoring
period_label = f"Last {analysis_period} Months" if analysis_period != 'All' else "Full Period"
st.header(f"📊 Financial Dashboard: {company_name} ({period_label})")

# ENHANCED Key Metrics with Adaptive Scoring - FIXED VERSION
col1, col2, col3, col4, col5 = st.columns(5)
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
    if scores['ml_score']:
        st.metric("ML Probability", f"{scores['ml_score']:.1f}%")
    else:
        st.metric("ML Probability", "N/A")
with col4:
    risk_colors = {"Low Risk": "🟢", "Moderate Low Risk": "🟡", "Medium Risk": "🟠", "Moderate High Risk": "🔴", "High Risk": "🔴"}
    st.metric("Loan to Revenue Risk", f"{risk_colors.get(scores['loan_risk'], '⚪')} {scores['loan_risk']}")
with col5:
    st.metric("Monthly Revenue", f"£{metrics.get('Monthly Average Revenue', 0):,.0f}")

# ENHANCED SCORING COMPARISON SECTION - NEW VERSION
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

# CONTINUE WITH THE REST OF YOUR EXISTING CODE...
            
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
                    full_scores = calculate_all_scores(full_metrics, params)
                    
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
            
            st.markdown("---")
            if ADAPTIVE_SCORING_AVAILABLE:
                st.success("🎯 Enhanced Dashboard complete with Adaptive Scoring - All sections rendered successfully")
            else:
                st.info("🎯 Dashboard complete - Install adaptive_score_calculation.py for enhanced scoring features")
            
        except Exception as e:
            st.error(f"Error processing file: {e}")
    
    else:
        st.info("👆 Upload a JSON transaction file to begin analysis")

if __name__ == "__main__":
    main()