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

def load_models():
    """Load ML models"""
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        return None, None

def map_transaction_category(transaction):
    """Enhanced transaction categorization"""
    name = str(transaction.get("name", "")).lower()
    description = str(transaction.get("merchant_name", "")).lower()
    category = str(transaction.get("personal_finance_category.detailed", "")).lower().strip().replace(" ", "_")
    amount = transaction.get("amount", 0)
    combined_text = f"{name} {description}"

    is_credit = amount < 0
    is_debit = amount > 0

    # Income patterns
    if is_credit and re.search(r"stripe|sumup|zettle|square|shopify|paypal|klarna|worldpay|uber|deliveroo|just\s*eat", combined_text):
        return "Income"
    
    # YouLend special handling
    if is_credit:
        if re.search(r"you\s?lend.*(?!fund)", combined_text):
            return "Income"
        elif re.search(r"you\s?lend.*fund", combined_text):
            return "Loans"
    
    # Loan patterns
    if is_credit and re.search(r"iwoca|capify|fundbox|funding\s*circle|liberis|loan|advance", combined_text):
        return "Loans"
    
    # Debt repayment patterns
    if is_debit and re.search(r"loan\s*repay|debt\s*repay|installment|iwoca.*payment|capify.*payment", combined_text):
        return "Debt Repayments"
    
    # Plaid category mappings
    plaid_mappings = {
        "income_wages": "Income", "income_other_income": "Income", 
        "income_dividends": "Special Inflow", "transfer_in_cash_advances_and_loans": "Loans",
        "loan_payments_": "Debt Repayments", "bank_fees_": "Failed Payment"
    }
    
    for key, value in plaid_mappings.items():
        if category.startswith(key):
            return value
    
    # Default categorization
    if is_credit:
        return "Income" if "income" in category else "Special Inflow"
    elif is_debit:
        return "Expenses" if any(x in combined_text for x in ["food", "transport", "retail", "entertainment"]) else "Special Outflow"
    
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
    """Calculate all scoring methods"""
    industry_thresholds = INDUSTRY_THRESHOLDS[params['industry']]
    sector_risk = industry_thresholds['Sector Risk']
    
    # Weighted Score
    weighted_score = 0
    for metric, weight in WEIGHTS.items():
        if metric == 'Company Age (Months)':
            if params['company_age_months'] >= 6:
                weighted_score += weight
        elif metric == 'Directors Score':
            if params['directors_score'] >= industry_thresholds['Directors Score']:
                weighted_score += weight
        elif metric == 'Sector Risk':
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
        'weighted_score': weighted_score,
        'industry_score': industry_score,
        'ml_score': ml_score,
        'loan_risk': loan_risk,
        'score_breakdown': score_breakdown
    }

def create_score_charts(scores, metrics):
    """Create clean bar charts for scores"""
    
    # Score comparison chart
    fig_scores = go.Figure()
    
    score_data = {
        'Weighted Score': scores['weighted_score'],
        'Industry Score': (scores['industry_score'] / 12) * 100,  # Convert to percentage
        'ML Probability': scores['ml_score'] if scores['ml_score'] else 0
    }
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    fig_scores.add_trace(go.Bar(
        x=list(score_data.keys()),
        y=list(score_data.values()),
        marker_color=colors,
        text=[f"{v:.1f}%" for v in score_data.values()],
        textposition='outside'
    ))
    
    fig_scores.update_layout(
        title="Score Comparison",
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

def filter_data_by_period(df, period_months):
    """Filter data by time period"""
    if df.empty or period_months == 'All':
        return df
    
    df['date'] = pd.to_datetime(df['date'])
    latest_date = df['date'].max()
    start_date = latest_date - pd.DateOffset(months=int(period_months))
    
    return df[df['date'] >= start_date]

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
    """Main application"""
    st.title(" Business Finance Scorecard")
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
            # Process file
            json_data = json.load(uploaded_file)
            transactions = json_data.get('transactions', [])
            
            if not transactions:
                st.error("No transactions found in uploaded file")
                return
            
            df = pd.json_normalize(transactions)
            df['date'] = pd.to_datetime(df['date'])
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
            df = df.dropna(subset=['amount'])
            
            st.success(f"✅ Loaded {len(df)} transactions")
            
            # Calculate metrics and scores
            metrics = calculate_financial_metrics(df, params['company_age_months'])
            scores = calculate_all_scores(metrics, params)
            
            # Main Dashboard
            st.header(f"📊 Financial Dashboard: {company_name}")
            
            # Key Metrics Row
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Weighted Score", f"{scores['weighted_score']:.0f}/100")
            
            with col2:
                st.metric("Industry Score", f"{scores['industry_score']}/12")
            
            with col3:
                if scores['ml_score']:
                    st.metric("ML Probability", f"{scores['ml_score']:.1f}%")
                else:
                    st.metric("ML Probability", "N/A")
            
            with col4:
                risk_colors = {"Low Risk": "🟢", "Moderate Low Risk": "🟡", "Medium Risk": "🟠", "Moderate High Risk": "🔴", "High Risk": "🔴"}
                st.metric("Loan Risk", f"{risk_colors.get(scores['loan_risk'], '⚪')} {scores['loan_risk']}")
            
            with col5:
                st.metric("Monthly Revenue", f"£{metrics.get('Monthly Average Revenue', 0):,.0f}")
            
            # Charts Section
            st.markdown("---")
            
            # Row 1: Score and Financial Charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig_scores = create_score_charts(scores, metrics)
                st.plotly_chart(fig_scores, use_container_width=True)
            
            with col2:
                fig_financial, fig_trend = create_financial_charts(metrics)
                st.plotly_chart(fig_financial, use_container_width=True)
            
            # Row 2: Trend and Threshold Charts
            col1, col2 = st.columns(2)
            
            with col1:
                if fig_trend:
                    st.plotly_chart(fig_trend, use_container_width=True)
                else:
                    st.info("Monthly trend requires multiple months of data")
            
            with col2:
                fig_threshold = create_threshold_chart(scores['score_breakdown'])
                st.plotly_chart(fig_threshold, use_container_width=True)
            
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
                    
                    metrics_data.append({
                        'Metric': metric,
                        'Actual Value': value,
                        'Threshold': f"{comparison} {threshold}",
                        'Status': '✅ Pass' if meets_threshold else '❌ Fail'
                    })
            
            if metrics_data:
                df_metrics = pd.DataFrame(metrics_data)
                st.dataframe(df_metrics, use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.error(f"Error processing file: {e}")
    
    else:
        st.info("👆 Upload a JSON transaction file to begin analysis")
        
        # Show sample data format
        with st.expander("📋 Expected JSON Format"):
            st.code('''
{
  "transactions": [
    {
      "date": "2024-01-15",
      "amount": -150.00,
      "name": "STRIPE PAYMENT",
      "merchant_name": "Online Payment",
      "personal_finance_category": {
        "detailed": "income_other_income"
      }
    }
  ]
}
            ''', language='json')

if __name__ == "__main__":
    main()