"""
MCA v2 Batch Processing Application - COMPLETE FIXED VERSION
Run this as: streamlit run batch_processor_complete_fixed.py --server.port 8502

MAJOR FIXES:
1. FIXED fuzzy matching logic and company name extraction
2. FIXED debug information display and storage
3. FIXED CSV parameter mapping with proper column handling
4. FIXED error handling and comprehensive logging
5. ADDED detailed debugging throughout the process
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import zipfile
import tempfile
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from pathlib import Path
import joblib

# Try to import rapidfuzz, fallback if not available
try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
    print("✅ RapidFuzz imported successfully")
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    print("⚠️ RapidFuzz not available, using fallback matching")

st.set_page_config(
    page_title="MCA v2 Batch Processor (COMPLETE FIXED)", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# COMPLETE INDUSTRY THRESHOLDS
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

# TRANSACTION CATEGORIZATION FUNCTIONS
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

# FINANCIAL METRICS CALCULATION
def calculate_financial_metrics(data, company_age_months):
    """Calculate comprehensive financial metrics"""
    if data.empty:
        return {}
    
    try:
        data = categorize_transactions(data)
        
        # Use absolute values for all amounts
        total_revenue = abs(data.loc[data['is_revenue'], 'amount'].sum()) if data['is_revenue'].any() else 0
        total_expenses = abs(data.loc[data['is_expense'], 'amount'].sum()) if data['is_expense'].any() else 0
        net_income = total_revenue - total_expenses
        total_debt_repayments = abs(data.loc[data['is_debt_repayment'], 'amount'].sum()) if data['is_debt_repayment'].any() else 0
        total_debt = abs(data.loc[data['is_debt'], 'amount'].sum()) if data['is_debt'].any() else 0
        
        # Ensure minimum values to prevent division by zero
        total_revenue = max(total_revenue, 1)
        
        # Time-based calculations
        data['date'] = pd.to_datetime(data['date'])
        data['year_month'] = data['date'].dt.to_period('M')
        unique_months = data['year_month'].nunique()
        months_count = max(unique_months, 1)
        
        monthly_avg_revenue = total_revenue / months_count
        
        # Financial ratios
        debt_to_income_ratio = min(total_debt / total_revenue, 10) if total_revenue > 0 else 0
        expense_to_revenue_ratio = total_expenses / total_revenue if total_revenue > 0 else 1
        operating_margin = max(-1, min(1, net_income / total_revenue)) if total_revenue > 0 else -1
        
        # Debt Service Coverage Ratio calculation
        if total_debt_repayments > 0:
            debt_service_coverage_ratio = total_revenue / total_debt_repayments
        elif total_debt > 0:
            estimated_annual_payment = total_debt * 0.1
            debt_service_coverage_ratio = total_revenue / estimated_annual_payment if estimated_annual_payment > 0 else 0
        else:
            debt_service_coverage_ratio = 10
        
        debt_service_coverage_ratio = min(debt_service_coverage_ratio, 50)
        
        # Monthly analysis
        monthly_summary = data.groupby('year_month').agg({
            'amount': [
                lambda x: abs(x[data.loc[x.index, 'is_revenue']].sum()) if data.loc[x.index, 'is_revenue'].any() else 0,
                lambda x: abs(x[data.loc[x.index, 'is_expense']].sum()) if data.loc[x.index, 'is_expense'].any() else 0
            ]
        }).round(2)
        
        monthly_summary.columns = ['monthly_revenue', 'monthly_expenses']
        
        # Volatility metrics
        if len(monthly_summary) > 1:
            revenue_values = monthly_summary['monthly_revenue']
            revenue_mean = revenue_values.mean()
            
            if revenue_mean > 0:
                cash_flow_volatility = min(revenue_values.std() / revenue_mean, 2.0)
            else:
                cash_flow_volatility = 0.5
                
            revenue_growth_changes = revenue_values.pct_change().dropna()
            if len(revenue_growth_changes) > 0:
                revenue_growth_rate = revenue_growth_changes.median()
                revenue_growth_rate = max(-0.5, min(0.5, revenue_growth_rate))
            else:
                revenue_growth_rate = 0
                
            gross_burn_rate = monthly_summary['monthly_expenses'].mean()
        else:
            cash_flow_volatility = 0.1
            revenue_growth_rate = 0
            gross_burn_rate = total_expenses / months_count
        
        # Balance metrics
        if 'balances.available' in data.columns and not data['balances.available'].isna().all():
            avg_month_end_balance = data['balances.available'].mean()
        else:
            monthly_net = (total_revenue - total_expenses) / months_count
            avg_month_end_balance = max(1000, monthly_net * 0.5)
        
        # Negative balance days - estimated
        if cash_flow_volatility > 0.3:
            avg_negative_days = min(10, int(cash_flow_volatility * 10))
        elif operating_margin < 0:
            avg_negative_days = 3
        else:
            avg_negative_days = 0
        
        # Bounced payments
        bounced_payments = 0
        if 'name' in data.columns:
            failed_payment_keywords = ['unpaid', 'returned', 'bounced', 'insufficient', 'failed', 'declined']
            for keyword in failed_payment_keywords:
                bounced_payments += data['name'].str.contains(keyword, case=False, na=False).sum()
        
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
        print(f"Error calculating metrics: {e}")
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
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        return None, None

# TIGHTENED SUBPRIME SCORING CLASS - More Realistic Risk Assessment
class TightenedSubprimeScoring:
    """Enhanced scoring system with tightened thresholds for realistic subprime business lending."""
    
    def __init__(self):
        # TIGHTENED subprime weights - emphasize cash flow stability and debt management
        self.subprime_weights = {
            'Debt Service Coverage Ratio': 35,  # Increased from 28
            'Cash Flow Volatility': 15,         # Increased from 8 
            'Directors Score': 18,              # Increased from 16
            'Average Month-End Balance': 10,    # Reduced from 12
            'Revenue Growth Rate': 8,           # Reduced from 20 (less emphasis on growth)
            'Operating Margin': 6,              # Same
            'Average Negative Balance Days per Month': 6,  # Increased from 4
            'Company Age (Months)': 2,          # Same
        }
        
        # TIGHTENED industry multipliers - more conservative across the board
        self.industry_multipliers = {
            'Medical Practices (GPs, Clinics, Dentists)': 1.05,  # Reduced from 1.1
            'IT Services and Support Companies': 1.05,           # Reduced from 1.1
            'Pharmacies (Independent or Small Chains)': 1.03,    # New, conservative
            'Business Consultants': 1.0,                        # Reduced from 1.05
            'Education': 1.0,                                    # Reduced from 1.05
            'Engineering': 1.0,                                  # Reduced from 1.05
            'Telecommunications': 0.98,                          # Reduced from 1.05
            'Manufacturing': 0.95,                               # Reduced from 1.0
            'Retail': 0.93,                                      # Reduced from 1.0
            'Food Service': 0.90,                                # Reduced from 1.0
            'Tradesman': 0.88,                                   # Reduced from 1.0
            'Other': 0.85,                                       # Reduced from 1.0
            'Restaurants and Cafes': 0.80,                       # Reduced from 0.9
            'Construction Firms': 0.78,                          # Reduced from 0.9
            'Beauty Salons and Spas': 0.75,                      # Reduced from 0.9
            'Bars and Pubs': 0.70,                               # Reduced from 0.85
            'Event Planning and Management Firms': 0.65,         # Reduced from 0.8
        }
        
        # ENHANCED penalty system - much stricter
        self.enhanced_penalties = {
            'personal_default_12m': 8,           # Increased from 3
            'business_ccj': 12,                  # Increased from 5
            'director_ccj': 8,                   # Increased from 3
            'website_or_social_outdated': 5,     # Increased from 3
            'uses_generic_email': 3,             # Increased from 1
            'no_online_presence': 6,             # Increased from 2
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
        
        return {
            'subprime_score': round(final_score, 1),
            'risk_tier': risk_tier,
            'pricing_guidance': pricing_guidance,
            'breakdown': breakdown,
            'recommendation': self._generate_tightened_recommendation(risk_tier, metrics, params)
        }
    
    def _calculate_tightened_base_score(self, metrics, params):
        """Calculate base score with MUCH TIGHTER thresholds."""
        
        score = 0
        max_possible = sum(self.subprime_weights.values())
        
        # DEBT SERVICE COVERAGE RATIO (35 points) - MUCH TIGHTER
        dscr = metrics.get('Debt Service Coverage Ratio', 0)
        if dscr >= 3.5:        # Raised from 3.0
            score += self.subprime_weights['Debt Service Coverage Ratio']
        elif dscr >= 2.5:      # Raised from 2.0
            score += self.subprime_weights['Debt Service Coverage Ratio'] * 0.85  # Reduced from 0.9
        elif dscr >= 2.0:      # Raised from 1.5
            score += self.subprime_weights['Debt Service Coverage Ratio'] * 0.70  # Reduced from 0.8
        elif dscr >= 1.5:      # Raised from 1.2
            score += self.subprime_weights['Debt Service Coverage Ratio'] * 0.45  # Reduced from 0.6
        elif dscr >= 1.2:      # Raised from 1.0
            score += self.subprime_weights['Debt Service Coverage Ratio'] * 0.20  # Reduced from 0.3
        # Below 1.2 gets 0 points (was 1.0)
        
        # CASH FLOW VOLATILITY (15 points) - MUCH STRICTER
        volatility = metrics.get('Cash Flow Volatility', 1.0)
        if volatility <= 0.10:     # Tightened from 0.15
            score += self.subprime_weights['Cash Flow Volatility']
        elif volatility <= 0.20:   # Tightened from 0.3
            score += self.subprime_weights['Cash Flow Volatility'] * 0.75  # Reduced from 0.8
        elif volatility <= 0.30:   # Tightened from 0.5
            score += self.subprime_weights['Cash Flow Volatility'] * 0.50  # Reduced from 0.6
        elif volatility <= 0.45:   # Tightened from 0.8
            score += self.subprime_weights['Cash Flow Volatility'] * 0.25  # Reduced from 0.4
        # Above 0.45 gets 0 points (was 1.0)
        
        # DIRECTORS SCORE (18 points) - TIGHTER
        directors_score = params.get('directors_score', 0)
        if directors_score >= 80:      # Raised from 85
            score += self.subprime_weights['Directors Score']
        elif directors_score >= 70:    # Reduced from 75
            score += self.subprime_weights['Directors Score'] * 0.80  # Reduced from 0.9
        elif directors_score >= 60:    # Reduced from 65
            score += self.subprime_weights['Directors Score'] * 0.55  # Reduced from 0.7
        elif directors_score >= 50:    # Reduced from 55
            score += self.subprime_weights['Directors Score'] * 0.30  # Reduced from 0.5
        elif directors_score >= 40:    # Reduced from 45
            score += self.subprime_weights['Directors Score'] * 0.10  # Reduced from 0.3
        # Below 40 gets 0 points (was 45)
        
        # AVERAGE MONTH-END BALANCE (10 points) - TIGHTER
        balance = metrics.get('Average Month-End Balance', 0)
        if balance >= 15000:       # Raised from 10000
            score += self.subprime_weights['Average Month-End Balance']
        elif balance >= 8000:      # Raised from 5000
            score += self.subprime_weights['Average Month-End Balance'] * 0.70  # Reduced from 0.8
        elif balance >= 4000:      # Raised from 2000
            score += self.subprime_weights['Average Month-End Balance'] * 0.45  # Reduced from 0.6
        elif balance >= 1000:      # Raised from 500
            score += self.subprime_weights['Average Month-End Balance'] * 0.20  # Reduced from 0.4
        # Below £1000 gets 0 points (was £500)
        
        # REVENUE GROWTH RATE (8 points) - LESS EMPHASIS, TIGHTER
        growth = metrics.get('Revenue Growth Rate', 0)
        if growth >= 0.25:         # Raised from 0.3
            score += self.subprime_weights['Revenue Growth Rate']
        elif growth >= 0.15:       # Reduced from 0.2
            score += self.subprime_weights['Revenue Growth Rate'] * 0.80  # Reduced from 0.9
        elif growth >= 0.05:       # Reduced from 0.1
            score += self.subprime_weights['Revenue Growth Rate'] * 0.60  # Reduced from 0.7
        elif growth >= 0:          # Same as 0.05
            score += self.subprime_weights['Revenue Growth Rate'] * 0.40  # Reduced from 0.5
        elif growth >= -0.05:      # Tightened from -0.1
            score += self.subprime_weights['Revenue Growth Rate'] * 0.20  # Increased from 0.1
        # Below -5% gets 0 points (was -10%)
        
        # OPERATING MARGIN (6 points) - TIGHTER
        margin = metrics.get('Operating Margin', 0)
        if margin >= 0.12:         # Raised from 0.1
            score += self.subprime_weights['Operating Margin']
        elif margin >= 0.08:       # Raised from 0.05
            score += self.subprime_weights['Operating Margin'] * 0.75  # Reduced from 0.8
        elif margin >= 0.03:       # Raised from 0
            score += self.subprime_weights['Operating Margin'] * 0.50  # Reduced from 0.6
        elif margin >= 0:          # Same as -0.05
            score += self.subprime_weights['Operating Margin'] * 0.25  # Reduced from 0.4
        # Negative margins get 0 points (was -0.1)
        
        # NEGATIVE BALANCE DAYS (6 points) - STRICTER
        neg_days = metrics.get('Average Negative Balance Days per Month', 0)
        if neg_days == 0:          # Only perfect score gets full points
            score += self.subprime_weights['Average Negative Balance Days per Month']
        elif neg_days <= 1:        # Tightened from 1
            score += self.subprime_weights['Average Negative Balance Days per Month'] * 0.60  # Reduced from 0.7
        elif neg_days <= 2:        # Tightened from 3
            score += self.subprime_weights['Average Negative Balance Days per Month'] * 0.30  # Reduced from 0.4
        # Above 2 days gets 0 points (was 5)
        
        # COMPANY AGE (2 points) - SLIGHTLY TIGHTER
        age_months = params.get('company_age_months', 0)
        if age_months >= 36:       # Raised from 24
            score += self.subprime_weights['Company Age (Months)']
        elif age_months >= 18:     # Raised from 12
            score += self.subprime_weights['Company Age (Months)'] * 0.60  # Reduced from 0.7
        elif age_months >= 12:     # Raised from 6
            score += self.subprime_weights['Company Age (Months)'] * 0.30  # Reduced from 0.4
        # Below 12 months gets 0 points (was 6)
        
        # Convert to percentage
        return (score / max_possible) * 100
    
    def _apply_industry_adjustment(self, base_score, industry):
        """Apply TIGHTENED industry-specific risk adjustments."""
        multiplier = self.industry_multipliers.get(industry, 0.80)  # Default more conservative
        return base_score * multiplier
    
    def _calculate_conservative_growth_bonus(self, metrics):
        """Calculate REDUCED bonus for growth momentum."""
        bonus = 0
        growth = metrics.get('Revenue Growth Rate', 0)
        dscr = metrics.get('Debt Service Coverage Ratio', 0)
        volatility = metrics.get('Cash Flow Volatility', 1.0)
        
        # MUCH STRICTER criteria for bonuses
        if growth >= 0.30 and dscr >= 2.5 and volatility <= 0.15:  # Raised all thresholds
            bonus += 3  # Reduced from 5
        elif growth >= 0.20 and dscr >= 2.0 and volatility <= 0.25:  # Raised thresholds
            bonus += 2  # Reduced from 3
        elif growth >= 0.15 and dscr >= 1.8 and volatility <= 0.35:  # Much stricter
            bonus += 1  # Same
        
        return bonus
    
    def _calculate_enhanced_stability_penalty(self, metrics, params):
        """Calculate MUCH HIGHER penalty for instability."""
        penalty = 0
        volatility = metrics.get('Cash Flow Volatility', 0)
        operating_margin = metrics.get('Operating Margin', 0)
        neg_days = metrics.get('Average Negative Balance Days per Month', 0)
        dscr = metrics.get('Debt Service Coverage Ratio', 0)
        
        # VOLATILITY penalties - much stricter
        if volatility > 0.8:           # Reduced threshold from 1.0
            penalty += (volatility - 0.8) * 25  # Increased multiplier from 10
        elif volatility > 0.5:         # New intermediate penalty
            penalty += (volatility - 0.5) * 15
        
        # OPERATING MARGIN penalties - stricter
        if operating_margin < -0.05:   # Tightened from -0.15
            penalty += abs(operating_margin - (-0.05)) * 80  # Increased from 50
        
        # NEGATIVE BALANCE penalties - much stricter
        if neg_days > 2:               # Reduced from 5
            penalty += (neg_days - 2) * 4  # Increased from 2
        
        # DSCR penalties - new addition
        if dscr < 1.2:                 # New penalty for low DSCR
            penalty += (1.2 - dscr) * 15
        
        return min(penalty, 35)  # Increased cap from 20
    
    def _calculate_enhanced_risk_penalties(self, params):
        """Calculate MUCH HIGHER penalties for risk factors."""
        penalty = 0
        
        for factor, penalty_points in self.enhanced_penalties.items():
            if params.get(factor, False):
                penalty += penalty_points
        
        # COMPOUND penalty for multiple risk factors
        risk_count = sum(1 for factor in self.enhanced_penalties.keys() if params.get(factor, False))
        if risk_count >= 3:            # 3 or more risk factors
            penalty += risk_count * 3   # Additional compound penalty
        elif risk_count >= 2:          # 2 risk factors  
            penalty += risk_count * 1.5
        
        return min(penalty, 25)  # Cap total risk penalties
    
    def _determine_tightened_risk_tier(self, score, metrics, params):
        """Determine risk tier with MUCH TIGHTER criteria."""
        dscr = metrics.get('Debt Service Coverage Ratio', 0)
        growth = metrics.get('Revenue Growth Rate', 0)
        directors_score = params.get('directors_score', 0)
        volatility = metrics.get('Cash Flow Volatility', 1.0)
        operating_margin = metrics.get('Operating Margin', 0)
        
        # TIER 1 - MUCH STRICTER (Premium Subprime)
        if (score >= 80 and dscr >= 2.5 and growth >= 0.20 and directors_score >= 75 and 
            volatility <= 0.20 and operating_margin >= 0.08):
            return "Tier 1", {
                "risk_level": "Premium Subprime",
                "suggested_rate": "1.35-1.45 factor rate",  # Slightly better rates
                "max_loan_multiple": "5x monthly revenue",   # Reduced from 6x
                "term_range": "12-18 months",                # Reduced max term
                "monitoring": "Monthly reviews",             # Increased from quarterly
                "approval_probability": "Very High"
            }
        
        # TIER 2 - STRICTER (Standard Subprime)
        elif (score >= 65 and dscr >= 2.0 and directors_score >= 65 and volatility <= 0.35):
            return "Tier 2", {
                "risk_level": "Standard Subprime", 
                "suggested_rate": "1.5-1.65 factor rate",   # Increased from 1.5-1.6
                "max_loan_multiple": "3.5x monthly revenue", # Reduced from 4x
                "term_range": "6-15 months",                 # Reduced max term
                "monitoring": "Bi-weekly reviews",           # Increased from monthly
                "approval_probability": "High"
            }
        
        # TIER 3 - STRICTER (High-Risk Subprime)
        elif (score >= 50 and dscr >= 1.5 and directors_score >= 55 and volatility <= 0.50):
            return "Tier 3", {
                "risk_level": "High-Risk Subprime",
                "suggested_rate": "1.65-1.8 factor rate",   # Increased from 1.6-1.75
                "max_loan_multiple": "2.5x monthly revenue", # Reduced from 3x
                "term_range": "4-12 months",                 # Reduced range
                "monitoring": "Weekly reviews + alerts",     # Enhanced monitoring
                "approval_probability": "Moderate"
            }
        
        # TIER 4 - MUCH STRICTER (Enhanced Monitoring)
        elif (score >= 35 and dscr >= 1.3 and directors_score >= 45):
            return "Tier 4", {
                "risk_level": "Enhanced Monitoring Required",
                "suggested_rate": "1.8-2.1 factor rate",    # Increased from 1.75-2.0
                "max_loan_multiple": "2x monthly revenue",   # Same
                "term_range": "3-6 months",                  # Reduced max term
                "monitoring": "Daily balance + weekly calls", # Enhanced
                "approval_probability": "Low - Senior review + guarantees required"
            }
        
        # DECLINE - MUCH MORE LIKELY
        else:
            return "Decline", {
                "risk_level": "Decline",
                "suggested_rate": "N/A",
                "max_loan_multiple": "N/A",
                "term_range": "N/A", 
                "monitoring": "N/A",
                "approval_probability": "Decline - Risk exceeds acceptable parameters"
            }
    
    def _generate_tightened_breakdown(self, base_score, industry_score, growth_bonus, 
                                    stability_penalty, risk_penalty, final_score, metrics, params):
        """Generate detailed scoring breakdown with tightened criteria."""
        breakdown = [
            f"Base Tightened Score: {base_score:.1f}/100",
            f"Industry Adjustment: {industry_score - base_score:+.1f} points",
            f"Growth Momentum Bonus: +{growth_bonus:.1f} points",
            f"Stability Penalty: -{stability_penalty:.1f} points",
            f"Risk Factor Penalty: -{risk_penalty:.1f} points",
            f"Final Score: {final_score:.1f}/100",
            "",
            "Key Metrics (Tightened Thresholds):",
            f"• DSCR: {metrics.get('Debt Service Coverage Ratio', 0):.2f} (Need 2.0+ for good score)",
            f"• Revenue Growth: {metrics.get('Revenue Growth Rate', 0)*100:.1f}% (Need 15%+ for good score)",
            f"• Directors Score: {params.get('directors_score', 0)}/100 (Need 70+ for good score)", 
            f"• Cash Flow Volatility: {metrics.get('Cash Flow Volatility', 0):.3f} (Need <0.20 for good score)",
            f"• Operating Margin: {metrics.get('Operating Margin', 0)*100:.1f}% (Need 8%+ for good score)",
            f"• Negative Balance Days: {metrics.get('Average Negative Balance Days per Month', 0):.0f} (Need 0-1 for good score)"
        ]
        return breakdown
    
    def _generate_tightened_recommendation(self, risk_tier, metrics, params):
        """Generate lending recommendation with tightened criteria."""
        if risk_tier == "Tier 1":
            return "APPROVE - Excellent subprime candidate meeting tightened criteria."
        elif risk_tier == "Tier 2": 
            return "APPROVE - Good subprime candidate with enhanced monitoring."
        elif risk_tier == "Tier 3":
            return "CONDITIONAL APPROVE - Marginal case requiring strict terms and close monitoring."
        elif risk_tier == "Tier 4":
            return "SENIOR REVIEW - High risk requiring guarantees and daily monitoring."
        else:
            return "DECLINE - Does not meet tightened subprime lending criteria."


# UPDATED CALCULATION FUNCTION to use tightened scoring
def calculate_all_scores_tightened(metrics, params):
    """Enhanced scoring calculation with TIGHTENED subprime scoring"""
    industry_thresholds = INDUSTRY_THRESHOLDS[params['industry']]
    sector_risk = industry_thresholds['Sector Risk']
    
    # Calculate weighted scores (unchanged)
    weighted_score = calculate_weighted_scores(metrics, params, industry_thresholds)
    
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
    
    # ML Score (unchanged - load models if available)
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
    
    return {
        'weighted_score': weighted_score,
        'industry_score': industry_score,
        'ml_score': ml_score,
        'loan_risk': loan_risk,
        'score_breakdown': score_breakdown,
        'subprime_score': subprime_result['subprime_score'],
        'subprime_tier': subprime_result['risk_tier'],
        'subprime_pricing': subprime_result['pricing_guidance'],
        'subprime_recommendation': subprime_result['recommendation'],
        'subprime_breakdown': subprime_result['breakdown']
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
    
    def clean_company_name(self, name):
        """Simple cleaning that preserves matching"""
        if not name:
            return ""
    
        # Convert to string and lowercase
        clean_name = str(name).lower().strip()
    
        # Remove ONLY obvious filename junk
        clean_name = re.sub(r'_\d+,\d+\.json$', '', clean_name)
        clean_name = re.sub(r'_\d+,\d+$', '', clean_name) 
        clean_name = re.sub(r'\.json$', '', clean_name)
        clean_name = re.sub(r'\s+transaction\s+report$', '', clean_name)
    
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
            
            # Try multiple strategies
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
                    
                print(f"DEBUG: No match found for '{clean_search}' against first CSV company '{list(clean_csv_companies.keys())[0] if clean_csv_companies else 'None'}'")
        else:
            # Fallback exact matching
            debug_info['fuzzy_match_debug'] = "Using fallback exact matching"
            
            if clean_search in clean_csv_companies:
                original_company = clean_csv_companies[clean_search]
                best_match = (original_company, 100)
                best_score = 100
                best_strategy = "exact_fallback"
        
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
            # Step 1: Validate transaction data
            # Handle case where json_data is a list
            if isinstance(json_data, list):
                if len(json_data) > 0 and isinstance(json_data[0], dict):
                    json_data = json_data[0]  # Use first item
                else:
                    raise ValueError("JSON file contains a list but first item is not a dictionary")

            # Check if this is a single transaction file (no 'transactions' key)
            if 'transactions' in json_data:
                transactions = json_data.get('transactions', [])
            else:
                # This file IS a single transaction - wrap it in a list
                if 'transaction_id' in json_data or 'amount' in json_data:
                    transactions = [json_data]
                    print(f"DEBUG: Single transaction file detected: {filename}")
                else:
                    transactions = []

            if not transactions:
                raise ValueError("No transactions found in JSON")
            
            debug_info['transaction_count'] = len(transactions)
            debug_info['debug_step'] = f'Found {len(transactions)} transactions'
            
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
            
            debug_info['debug_step'] = f'Data cleaned: {len(df)} valid transactions'
            
            # Step 2: FIXED company name extraction
            params = default_params.copy() if default_params else {}
            
            # Extract company name with comprehensive tracking
            company_name, extraction_method = self.extract_company_name_from_json(json_data, filename)
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
            
            # Step 3: FIXED fuzzy matching with comprehensive CSV data handling
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
                    debug_info['debug_step'] = f'FUZZY MATCH SUCCESS: "{company_name}" -> "{matched_company}" ({score}% confidence via {strategy})'
                    
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
                    debug_info['debug_step'] += f', Applied {params_applied} CSV parameters: {list(applied_params.keys())}'
                    
                else:
                    debug_info['fuzzy_match_success'] = False
                    debug_info['debug_step'] = f'FUZZY MATCH FAILED: No match found for "{company_name}" in {len(csv_companies)} CSV companies'
                    debug_info['fuzzy_match_debug'] = f'Failed to find match for "{company_name}" against CSV companies'
                    
            elif not parameter_mapping:
                debug_info['debug_step'] = 'No CSV parameter mapping provided'
                debug_info['fuzzy_match_debug'] = 'No CSV parameter mapping available'
            elif not company_name:
                debug_info['debug_step'] = 'No company name extracted for matching'
                debug_info['fuzzy_match_debug'] = 'No company name available for matching'
            
            # Step 4: Try to extract application-specific data from JSON
            debug_info['debug_step'] = 'Checking for JSON metadata'
            
            # Check for application metadata in JSON
            metadata_found = {}
            app_data = json_data.get('application_data', {})
            if app_data:
                debug_info['json_application_data_found'] = True
                for key in ['industry', 'directors_score', 'requested_loan', 'company_age_months']:
                    if key in app_data and app_data[key] is not None:
                        params[key] = app_data[key]
                        metadata_found[key] = app_data[key]
            
            # Check for metadata in root JSON
            root_metadata = {}
            metadata_fields = {
                'industry': json_data.get('industry'),
                'directors_score': json_data.get('directors_score'), 
                'requested_loan': json_data.get('requested_loan'),
                'company_age_months': json_data.get('company_age_months')
            }
            
            for key, value in metadata_fields.items():
                if value is not None and pd.notna(value):
                    params[key] = value
                    root_metadata[key] = value
            
            debug_info['json_metadata_found'] = {**metadata_found, **root_metadata}
            debug_info['debug_step'] = f'JSON metadata extraction complete, found: {list(debug_info["json_metadata_found"].keys())}'
            
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
                
                # Convert boolean risk factors
                risk_factors = ['personal_default_12m', 'business_ccj', 'director_ccj', 
                               'website_or_social_outdated', 'uses_generic_email', 'no_online_presence']
                
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
            debug_info['final_parameters'] = {k: v for k, v in params.items() if k not in ['company_name']}  # Don't include potentially long company names
            
            # Step 8: Calculate financial metrics
            debug_info['debug_step'] = 'Calculating financial metrics'
            
            metrics = calculate_financial_metrics(df, params['company_age_months'])
            
            if not metrics:
                raise ValueError("Could not calculate financial metrics from transaction data")
            
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
                'weighted_score': scores.get('weighted_score', 0),
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
    
    def process_batch(self, files_data, default_params, parameter_mapping=None, progress_bar=None):
        """Process multiple applications with comprehensive tracking"""
        self.results = []
        self.processed_count = 0
        self.error_count = 0
        self.error_log = []
        self.debug_log = []
        
        total_files = len(files_data)
        
        print(f"🚀 Starting batch processing of {total_files} files...")
        print(f"📊 Parameter mapping available: {bool(parameter_mapping)}")
        if parameter_mapping:
            print(f"📋 CSV companies available: {len(parameter_mapping)}")
            print(f"📝 First few CSV companies: {list(parameter_mapping.keys())[:3]}")
        
        for i, (filename, json_data) in enumerate(files_data):
            if progress_bar:
                progress_bar.progress((i + 1) / total_files, text=f"Processing {filename}...")
            
            print(f"\n🔄 Processing {i+1}/{total_files}: {filename}")
            
            result = self.process_single_application(json_data, filename, default_params, parameter_mapping)
            
            if result:
                self.results.append(result)
                print(f"✅ SUCCESS: {filename}")
                if result.get('fuzzy_match_success'):
                    print(f"   🎯 Matched to: {result.get('fuzzy_match_company')} ({result.get('fuzzy_match_score')}%)")
                if result.get('using_defaults'):
                    print(f"   ⚠️ Using defaults for: {result.get('using_defaults_for', [])}")
            else:
                print(f"❌ FAILED: {filename}")
        
        print(f"\n🏁 Batch processing complete!")
        print(f"✅ Successful: {self.processed_count}")
        print(f"❌ Failed: {self.error_count}")
        
        return pd.DataFrame(self.results) if self.results else pd.DataFrame()

def load_json_files(uploaded_files):
    """Load JSON files from uploaded files"""
    files_data = []
    
    for uploaded_file in uploaded_files:
        try:
            if uploaded_file.name.endswith('.zip'):
                # Handle ZIP files
                with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                    for file_info in zip_ref.filelist:
                        if file_info.filename.endswith('.json') and not file_info.is_dir():
                            with zip_ref.open(file_info.filename) as json_file:
                                json_data = json.load(json_file)
                                files_data.append((file_info.filename, json_data))
            
            elif uploaded_file.name.endswith('.json'):
                # Handle individual JSON files
                json_data = json.load(uploaded_file)
                files_data.append((uploaded_file.name, json_data))
                
        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {e}")
    
    return files_data

def create_results_dashboard(results_df):
    """Create comprehensive dashboard for batch results"""
    
    if results_df.empty:
        st.warning("No results to display")
        return
    
    # Summary Statistics
    st.subheader("📊 Batch Processing Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Applications Processed", len(results_df))
    
    with col2:
        avg_subprime = results_df['subprime_score'].mean() if 'subprime_score' in results_df.columns else 0
        st.metric("Avg Subprime Score", f"{avg_subprime:.1f}")
    
    with col3:
        avg_weighted = results_df['weighted_score'].mean() if 'weighted_score' in results_df.columns else 0
        st.metric("Avg Weighted Score", f"{avg_weighted:.1f}")
    
    with col4:
        if 'ml_score' in results_df.columns and results_df['ml_score'].notna().any():
            avg_ml = results_df['ml_score'].mean()
            st.metric("Avg ML Score", f"{avg_ml:.1f}%")
        else:
            st.metric("Avg ML Score", "N/A")
    
    with col5:
        avg_revenue = results_df['Total Revenue'].mean() if 'Total Revenue' in results_df.columns else 0
        st.metric("Avg Revenue", f"£{avg_revenue:,.0f}")
    
    # Score Distribution Charts
    st.subheader("📈 Score Distributions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'subprime_score' in results_df.columns:
            fig_subprime = px.histogram(
                results_df, 
                x='subprime_score', 
                title="Subprime Score Distribution",
                labels={'subprime_score': 'Subprime Score', 'count': 'Number of Applications'}
            )
            fig_subprime.add_vline(x=60, line_dash="dash", line_color="red", 
                                  annotation_text="Approval Threshold (60)")
            st.plotly_chart(fig_subprime, use_container_width=True)
    
    with col2:
        if 'weighted_score' in results_df.columns:
            fig_weighted = px.histogram(
                results_df, 
                x='weighted_score', 
                title="Weighted Score Distribution",
                labels={'weighted_score': 'Weighted Score', 'count': 'Number of Applications'}
            )
            fig_weighted.add_vline(x=70, line_dash="dash", line_color="red", 
                                  annotation_text="Typical Threshold (70)")
            st.plotly_chart(fig_weighted, use_container_width=True)
    
    # Risk Tier Analysis
    if 'subprime_tier' in results_df.columns:
        st.subheader("🎯 Risk Tier Analysis")
        
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
            st.plotly_chart(fig_tier, use_container_width=True)
        
        with col2:
            st.write("**Risk Tier Breakdown:**")
            for tier, count in tier_counts.items():
                percentage = (count / len(results_df)) * 100
                st.write(f"• **{tier}**: {count} applications ({percentage:.1f}%)")
            
            # Approval rate calculation
            approval_tiers = ['Tier 1', 'Tier 2', 'Tier 3']
            approved = tier_counts[tier_counts.index.isin(approval_tiers)].sum()
            approval_rate = (approved / len(results_df)) * 100
            
            st.metric("Potential Approval Rate", f"{approval_rate:.1f}%")
    
    # Detailed Results Table
    st.subheader("📋 Detailed Results")
    
    # Create downloadable results
    display_columns = [
        'company_name', 'industry', 'subprime_score', 'subprime_tier', 
        'weighted_score', 'Total Revenue', 'Net Income', 'Operating Margin',
        'Debt Service Coverage Ratio', 'requested_loan'
    ]
    
    # Filter to available columns
    available_columns = [col for col in display_columns if col in results_df.columns]
    display_df = results_df[available_columns].copy()
    
    # Format numeric columns
    numeric_format_cols = {
        'Total Revenue': lambda x: f"£{x:,.0f}",
        'Net Income': lambda x: f"£{x:,.0f}",
        'requested_loan': lambda x: f"£{x:,.0f}",
        'Operating Margin': lambda x: f"{x*100:.1f}%",
        'subprime_score': lambda x: f"{x:.1f}",
        'weighted_score': lambda x: f"{x:.0f}",
        'Debt Service Coverage Ratio': lambda x: f"{x:.2f}"
    }
    
    for col, formatter in numeric_format_cols.items():
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(formatter)
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Download button
    csv_data = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Full Results (CSV)",
        data=csv_data,
        file_name=f"mcav2_batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        type="primary"
    )

def main():
    """Main application"""
    
    st.title("🏦 MCA v2 Batch Processing Dashboard")
    st.markdown("**COMPLETE FIXED VERSION** - Process multiple loan applications through the MCA v2 scoring system")
    
    st.success("✅ **FIXES APPLIED**: Fuzzy matching, company name extraction, debug tracking, CSV parameter mapping")
    
    # Debug info about rapidfuzz
    if RAPIDFUZZ_AVAILABLE:
        st.info("🔧 **RapidFuzz Available**: Using advanced fuzzy matching algorithms")
    else:
        st.warning("⚠️ **RapidFuzz Not Available**: Using fallback exact matching only")
    
    # Sidebar for parameters
    st.sidebar.header("📋 Parameter Sources")
    st.sidebar.markdown("""
    **Parameter Priority:**
    1. **Individual JSON files** (if they contain application data)
    2. **CSV mapping file** (upload a CSV with company-specific parameters)
    3. **Fallback defaults** (only used when data is missing)
    """)
    
    # Option to upload parameter mapping
    st.sidebar.subheader("📊 Upload Parameter Mapping (CSV)")
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
            st.sidebar.success(f"✅ Loaded parameters for {len(param_df)} companies")
            
            # Show available columns
            available_cols = list(param_df.columns)
            expected_cols = ['company_name', 'industry', 'directors_score', 'requested_loan', 'company_age_months',
                           'personal_default_12m', 'business_ccj', 'director_ccj']
            
            missing_cols = [col for col in expected_cols if col not in available_cols]
            if missing_cols:
                st.sidebar.warning(f"⚠️ Missing columns: {', '.join(missing_cols)}")
            
            # Convert to dictionary - use 'company_name' column as key
            if 'company_name' in param_df.columns:
                for _, row in param_df.iterrows():
                    company_name = row['company_name']
                    parameter_mapping[company_name] = row.to_dict()
                
                st.sidebar.info(f"📋 **CSV Company Examples**: {list(parameter_mapping.keys())[:3]}")
            else:
                st.sidebar.error("❌ CSV must have 'company_name' column")
            
            # Show preview
            with st.sidebar.expander("Preview Parameter CSV"):
                st.dataframe(param_df.head(), use_container_width=True)
                
            # Add diagnostic button here
                st.sidebar.markdown("---")
                st.sidebar.subheader("🔧 Diagnostic Tools")
                if st.sidebar.button("🔍 Diagnose Specific Cases"):
                    st.session_state['show_diagnostic'] = True
                
        except Exception as e:
            st.sidebar.error(f"Error reading parameter file: {e}")
            
    # NEW: Show diagnostic section here (after parameter_mapping is loaded)
    if st.session_state.get('show_diagnostic', False) and parameter_mapping:
        st.subheader("🔧 Direct Matching Diagnostic")
        st.markdown("Let's diagnose exactly what's happening with specific cases:")
    
        failing_cases = [("22LUSH LTD", "22Lush Limited Transaction Report_0,0.json")]
    
        processor = BatchProcessor()
        csv_companies = list(parameter_mapping.keys())
    
        for csv_name, json_filename in failing_cases:
            st.write(f"**CSV:** `{csv_name}` vs **JSON:** `{json_filename}`")
            extracted_name, method = processor.extract_company_name_from_json({}, json_filename)
            st.write(f"**Extracted:** `{extracted_name}`")
        
            # Test if simple_match_test function exists
            if hasattr(processor, 'simple_match_test'):
                st.success("✅ simple_match_test function exists")
                debug_info = {}
                matched_company, score, strategy, success = processor.simple_match_test(extracted_name, csv_companies, debug_info)
                if success:
                    st.success(f"✅ **FUNCTION WORKS!** Found: {matched_company}")
                else:
                    st.error("❌ Function exists but returned no match")
            else:
                st.error("❌ simple_match_test function missing from BatchProcessor class")
        
            break
    
        if st.button("❌ Close Diagnostic", key="close_diagnostic_1"):
            st.session_state['show_diagnostic'] = False
            st.experimental_rerun()
    
        st.markdown("---")
    
    # Fallback defaults
    st.sidebar.subheader("🔄 Fallback Defaults")
    st.sidebar.markdown("*Only used when data is missing from JSON or CSV*")
    
    default_industry = st.sidebar.selectbox(
        "Fallback Industry", 
        list(INDUSTRY_THRESHOLDS.keys()),
        index=list(INDUSTRY_THRESHOLDS.keys()).index('Other')
    )
    
    default_loan = st.sidebar.number_input("Fallback Requested Loan (£)", min_value=0.0, value=25000.0, step=1000.0)
    default_directors_score = st.sidebar.slider("Fallback Director Credit Score", 0, 100, 75)
    default_company_age = st.sidebar.number_input("Fallback Company Age (Months)", min_value=0, value=24, step=1)
    
    # Risk factors
    st.sidebar.subheader("🚨 Default Risk Factors")
    default_personal_default = st.sidebar.checkbox("Personal Defaults (12m)", False)
    default_business_ccj = st.sidebar.checkbox("Business CCJs", False)
    default_director_ccj = st.sidebar.checkbox("Director CCJs", False)
    default_outdated_web = st.sidebar.checkbox("Outdated Web Presence", False)
    default_generic_email = st.sidebar.checkbox("Generic Email", False)
    default_no_online = st.sidebar.checkbox("No Online Presence", False)
    
    default_params = {
        'industry': default_industry,
        'requested_loan': default_loan,
        'directors_score': default_directors_score,
        'company_age_months': default_company_age,
        'personal_default_12m': default_personal_default,
        'business_ccj': default_business_ccj,
        'director_ccj': default_director_ccj,
        'website_or_social_outdated': default_outdated_web,
        'uses_generic_email': default_generic_email,
        'no_online_presence': default_no_online
    }
    
    # NEW: Show diagnostic section here (after parameter_mapping is loaded)
    if st.session_state.get('show_diagnostic', False) and parameter_mapping:
        st.subheader("🔧 Direct Matching Diagnostic")
        st.markdown("Let's diagnose exactly what's happening with specific cases:")
    
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
                st.success("✅ **MATCH after simple cleaning**")
            else:
                st.error(f"❌ **NO MATCH** - '{csv_lower}' vs '{extracted_lower}'")
    
            break  # Just show first case for now
    
        if st.button("❌ Close Diagnostic"):
            st.session_state['show_diagnostic'] = False
            st.experimental_rerun()
    
        st.markdown("---")
    
    # File upload section
    st.header("📁 Upload Applications")
    st.markdown("Upload individual JSON files or ZIP archives containing multiple JSON files:")
    
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
            st.success(f"✅ Loaded {len(files_data)} JSON files")
            
            # Show file list
            with st.expander("📋 Loaded Files", expanded=False):
                for filename, _ in files_data:
                    st.write(f"• {filename}")
            
            # Enhanced file matching analysis
            if parameter_mapping:
                st.subheader("🔍 File Matching Analysis")
                
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
                    st.success(f"✅ **{len(potential_matches)} potential matches found**")
                    
                    with st.expander(f"🎯 Potential Matches ({len(potential_matches)})", expanded=True):
                        match_df = pd.DataFrame(potential_matches)
                        match_df['score'] = match_df['score'].apply(lambda x: f"{x:.1f}%")
                        match_df.columns = ['CSV Company', 'JSON File', 'Match Score']
                        st.dataframe(match_df, use_container_width=True, hide_index=True)
                
                # Show missing JSON files
                if missing_jsons:
                    st.warning(f"⚠️ **{len(missing_jsons)} companies from CSV don't have matching JSON files**")
                    
                    with st.expander(f"📄 Missing JSON Files ({len(missing_jsons)})", expanded=False):
                        missing_df = pd.DataFrame({
                            'Company Name (from CSV)': missing_jsons,
                            'Status': ['No matching JSON file found'] * len(missing_jsons)
                        })
                        st.dataframe(missing_df, use_container_width=True, hide_index=True)
                
                # Show extra JSON files
                if extra_jsons:
                    with st.expander(f"📁 Extra JSON Files ({len(extra_jsons)})", expanded=False):
                        st.markdown("**JSON files uploaded but not in CSV (will use default parameters):**")
                        for filename in extra_jsons:
                            st.write(f"• {filename}.json")
            
            else:
                st.info("💡 Upload a CSV file to see file matching analysis.")
            
            # Process button
            if st.button("🚀 Process All Applications", type="primary"):
                
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
                    st.metric("✅ Processed Successfully", processor.processed_count)
                with col2:
                    st.metric("❌ Processing Errors", processor.error_count)
                with col3:
                    success_rate = (processor.processed_count / len(files_data)) * 100
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                
                # ENHANCED: Show fuzzy matching results
                if not results_df.empty and 'fuzzy_match_success' in results_df.columns:
                    st.subheader("🔍 Fuzzy Matching Results")
                    
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
                        with st.expander("🎯 Detailed Fuzzy Matching Results", expanded=False):
                            
                            # Successful matches
                            successful_df = results_df[results_df['fuzzy_match_success'] == True]
                            if not successful_df.empty:
                                st.write("**✅ Successful Matches:**")
                                match_details = successful_df[['original_filename', 'extracted_company_name', 'fuzzy_match_company', 'fuzzy_match_score', 'fuzzy_match_strategy', 'parameters_applied_from_csv']].copy()
                                match_details['fuzzy_match_score'] = match_details['fuzzy_match_score'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
                                match_details.columns = ['JSON File', 'Extracted Name', 'CSV Matched Company', 'Match Score', 'Strategy', 'Params Applied']
                                st.dataframe(match_details, use_container_width=True, hide_index=True)
                            
                            # Failed matches
                            failed_df = results_df[results_df['fuzzy_match_success'] == False]
                            if not failed_df.empty:
                                st.write("**❌ Failed Matches:**")
                                failed_details = failed_df[['original_filename', 'extracted_company_name', 'fuzzy_match_debug']].copy()
                                failed_details.columns = ['JSON File', 'Extracted Company Name', 'Debug Info']
                                st.dataframe(failed_details, use_container_width=True, hide_index=True)
                    
                    # Show parameter source breakdown
                    st.subheader("📊 Parameter Source Analysis")
                    
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
                    st.subheader("❌ Processing Errors Analysis")
                    
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
                            st.write(f"• **{category}**: {len(errors)} files")
                    
                    with col2:
                        st.write("**Most Common Errors:**")
                        sorted_categories = sorted(error_categories.items(), key=lambda x: len(x[1]), reverse=True)
                        for category, errors in sorted_categories[:3]:
                            percentage = (len(errors) / len(processor.error_log)) * 100
                            st.write(f"• {category}: {percentage:.1f}%")
                    
                    # Detailed error breakdown
                    with st.expander(f"📋 Detailed Error Breakdown ({len(processor.error_log)} errors)", expanded=False):
                        error_df = pd.DataFrame(processor.error_log)
                        st.dataframe(error_df, use_container_width=True, hide_index=True)
                        
                        # Download error log
                        error_csv = error_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Error Log (CSV)",
                            data=error_csv,
                            file_name=f"processing_errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                
                # Display results dashboard
                if not results_df.empty:
                    st.markdown("---")
                    
                    # COMPREHENSIVE DEBUG SECTION - Show all debug information
                    st.subheader("🔍 Comprehensive Debug Information")
                    
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
                            label="📥 Download Debug Log (CSV)",
                            data=debug_csv,
                            file_name=f"debug_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    # Quick debug stats
                    if debug_summary:
                        st.write("**🔧 Quick Debug Stats:**")
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
                        
                        st.write("**📊 Company Name Extraction Methods:**")
                        for method, count in extraction_methods.items():
                            percentage = (count / total_apps) * 100
                            st.write(f"• **{method}**: {count} files ({percentage:.1f}%)")
                    
                    st.markdown("---")
                    
                    create_results_dashboard(results_df)
                    
                    # Store results in session state for potential re-use
                    st.session_state['batch_results'] = results_df
                
                else:
                    st.error("❌ No applications were processed successfully")
        
        else:
            st.error("❌ No valid JSON files found in uploaded files")
    
    else:
        st.info("👆 Upload JSON files or ZIP archives to begin batch processing")
        
        # Example section
        st.markdown("---")
        st.subheader("📖 Usage Instructions")
        
        st.markdown("""
        **🎯 FIXED VERSION - Key Improvements:**
        - ✅ **Fixed fuzzy matching** - Now properly matches CSV companies to JSON files
        - ✅ **Enhanced company name extraction** - Tries multiple methods to find company names
        - ✅ **Comprehensive debug tracking** - Full visibility into processing steps
        - ✅ **Better CSV parameter handling** - Properly applies parameters from CSV mapping
        - ✅ **Detailed error reporting** - Know exactly why any file failed to process
        
        **Parameter Sources (in priority order):**
        1. **Individual JSON files** - If your JSON files contain application metadata
        2. **CSV parameter mapping** - Upload a CSV file with application-specific parameters  
        3. **Fallback defaults** - Used only when other sources don't have the data
        
        **CSV Parameter Mapping Format:**
        ```
        company_name,industry,directors_score,requested_loan,personal_default_12m,director_ccj
        ABC Manufacturing Ltd,Manufacturing,78,35000,FALSE,FALSE
        Smith's Restaurant,Restaurants and Cafes,65,50000,TRUE,FALSE
        Tech Solutions UK,IT Services and Support Companies,82,75000,FALSE,TRUE
        ```
        
        **Processing Features:**
        - **Fuzzy Matching**: Automatically matches company names between CSV and JSON files
        - **Multiple Extraction Methods**: Tries accounts, metadata, and filename extraction
        - **Comprehensive Debugging**: Full visibility into what happened with each file
        - **Error Categorization**: Understand why files failed and how to fix them
        - **Parameter Source Tracking**: Know where each parameter came from
        
        **Dashboard Insights:**
        - Score distributions and risk tier analysis
        - Industry performance comparison
        - Fuzzy matching success rates and details
        - Parameter source breakdown
        - Detailed error analysis with suggestions
        
        **Debug Information:**
        - Company name extraction methods
        - Fuzzy matching scores and strategies
        - CSV parameter application tracking
        - Processing step-by-step status
        - Full error logs with context
        """)

if __name__ == "__main__":
    main()