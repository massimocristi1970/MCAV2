# app/pages/transactions.py
"""Transaction processing and categorization functions."""

import pandas as pd
import numpy as np
import re
from typing import Dict, Any, Optional, Tuple


def map_transaction_category(transaction: Dict[str, Any]) -> str:
    """
    Enhanced transaction categorization using pattern matching.
    
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
    combined_text = f"{name} {description}"

    is_credit = amount < 0
    is_debit = amount > 0

    # Step 1: Payment processors and income sources (credits)
    if is_credit and re.search(
        r"(?i)\b("
        r"stripe|sumup|zettle|square|take\s*payments|shopify|card\s+settlement|daily\s+takings|payout"
        r"|paypal|go\s*cardless|klarna|worldpay|izettle|ubereats|just\s*eat|deliveroo|uber|bolt"
        r"|fresha|treatwell|taskrabbit|terminal|pos\s+deposit|revolut"
        r"|capital\s+one|evo\s*payments?|tink|teya(\s+solutions)?|talech"
        r"|barclaycard|elavon|adyen|payzone|verifone|ingenico"
        r"|nmi|trust\s+payments?|global\s+payments?|checkout\.com|epdq|santander|handepay"
        r"|dojo|paymentsense|first\s+data|fiserv|clover|lightspeed"
        r"|revel\s+systems?|touchbistro|vend|orderin|tide"
        r")\b", combined_text
    ):
        return "Income"

    # Step 2: Loan providers (credits = loans received)
    if is_credit and re.search(
        r"(?i)\b("
        r"iwoca|capify|fundbox|got[\s\-]?capital|funding[\s\-]?circle"
        r"|fleximize|marketfinance|liberis|esme[\s\-]?loans|thincats"
        r"|white[\s\-]?oak|growth[\s\-]?street|nucleus[\s\-]?commercial[\s\-]?finance"
        r"|ultimate[\s\-]?finance|just[\s\-]?cash[\s\-]?flow|boost[\s\-]?capital"
        r"|merchant[\s\-]?money|capital[\s\-]?on[\s\-]?tap|kriya|uncapped"
        r"|lendingcrowd|folk2folk|funding[\s\-]?tree|start[\s\-]?up[\s\-]?loans"
        r")\b", combined_text
    ):
        return "Loans"

    # Step 3: YouLend special handling
    if is_credit and re.search(r"(?i)\b(you\s?lend|yl\s?ii|yl\s?ltd|yl\s?limited|yl\s?a\s?limited)\b", combined_text):
        if re.search(r"(?i)\b(fnd|fund|funding)\b", combined_text):
            return "Loans"
        else:
            return "Income"

    # Step 4: Loan repayments (debits to loan providers)
    if is_debit and re.search(
        r"(?i)\b("
        r"iwoca|capify|fundbox|got[\s\-]?capital|funding[\s\-]?circle"
        r"|fleximize|marketfinance|liberis|esme[\s\-]?loans|thincats"
        r"|white[\s\-]?oak|growth[\s\-]?street|nucleus"
        r"|ultimate[\s\-]?finance|just[\s\-]?cash[\s\-]?flow|boost[\s\-]?capital"
        r"|merchant[\s\-]?money|capital[\s\-]?on[\s\-]?tap|kriya|uncapped"
        r"|you\s?lend|yl\s?ii|yl\s?ltd"
        r"|loan|repay|instalment|installment"
        r")\b", combined_text
    ):
        return "Debt Repayments"

    # Step 5: Failed/bounced payments
    if re.search(
        r"(?i)\b(unpaid|returned|bounced|insufficient|nsf|failed|declined|reversed|chargeback)\b",
        combined_text
    ):
        return "Failed Payment"

    # Step 6: Plaid category mapping
    plaid_income_categories = [
        "income_wages", "income_other_income", "income_dividends",
        "transfer_in_cash_advances_and_loans"
    ]
    plaid_expense_categories = [
        "entertainment_", "food_and_drink_", "general_merchandise_",
        "general_services_", "rent_and_utilities_", "transportation_"
    ]
    plaid_loan_categories = [
        "loan_payments_credit_card_payment", "loan_payments_personal_loan_payment",
        "loan_payments_other_payment"
    ]
    plaid_failed_categories = [
        "bank_fees_insufficient_funds", "bank_fees_late_payment"
    ]

    if any(category.startswith(cat) for cat in plaid_income_categories):
        return "Income" if is_credit else "Expenses"
    
    if any(category.startswith(cat) for cat in plaid_failed_categories):
        return "Failed Payment"
    
    if any(category.startswith(cat) for cat in plaid_loan_categories):
        return "Debt Repayments"
    
    if any(category.startswith(cat) for cat in plaid_expense_categories):
        return "Expenses"

    # Step 7: Default classification based on amount direction
    if is_credit:
        return "Income"
    elif is_debit:
        return "Expenses"
    else:
        return "Uncategorised"


def categorize_transactions(data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply categorization to transaction DataFrame.
    
    Args:
        data: DataFrame with transaction data
        
    Returns:
        DataFrame with categorization columns added
    """
    if data.empty:
        return data
    
    data = data.copy()
    data['subcategory'] = data.apply(map_transaction_category, axis=1)
    data['is_revenue'] = data['subcategory'].isin(['Income', 'Special Inflow'])
    data['is_expense'] = data['subcategory'].isin(['Expenses', 'Special Outflow'])
    data['is_debt_repayment'] = data['subcategory'].isin(['Debt Repayments'])
    data['is_debt'] = data['subcategory'].isin(['Loans'])
    
    return data


def filter_data_by_period(df: pd.DataFrame, period_months: str) -> pd.DataFrame:
    """
    Filter data to specified time period.
    
    Args:
        df: DataFrame with transaction data
        period_months: Period string ('All', '3', '6', '9', '12')
        
    Returns:
        Filtered DataFrame
    """
    if df.empty or period_months == 'All':
        return df
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    latest_date = df['date'].max()
    start_date = latest_date - pd.DateOffset(months=int(period_months))
    
    return df[df['date'] >= start_date]


def calculate_financial_metrics(data: pd.DataFrame, company_age_months: int) -> Dict[str, Any]:
    """
    Calculate comprehensive financial metrics from transaction data.
    
    Args:
        data: DataFrame with transaction data
        company_age_months: Age of the company in months
        
    Returns:
        Dictionary of financial metrics
    """
    if data.empty:
        return {}
    
    try:
        data = categorize_transactions(data)
        
        # Calculate totals using absolute values
        total_revenue = abs(data.loc[data['is_revenue'], 'amount'].sum()) if data['is_revenue'].any() else 0
        total_expenses = abs(data.loc[data['is_expense'], 'amount'].sum()) if data['is_expense'].any() else 0
        net_income = total_revenue - total_expenses
        total_debt_repayments = abs(data.loc[data['is_debt_repayment'], 'amount'].sum()) if data['is_debt_repayment'].any() else 0
        total_debt = abs(data.loc[data['is_debt'], 'amount'].sum()) if data['is_debt'].any() else 0
        
        # Minimum revenue to prevent division by zero
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
        
        # Debt Service Coverage Ratio
        if total_debt_repayments > 0:
            debt_service_coverage_ratio = total_revenue / total_debt_repayments
        elif total_debt > 0:
            estimated_annual_payment = total_debt * 0.1
            debt_service_coverage_ratio = total_revenue / estimated_annual_payment if estimated_annual_payment > 0 else 0
        else:
            debt_service_coverage_ratio = 10  # No debt = excellent coverage
        
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
            
            # Revenue growth calculation
            revenue_growth_changes = revenue_values.pct_change().dropna()
            if len(revenue_growth_changes) > 0:
                revenue_growth_rate = revenue_growth_changes.median()
                revenue_growth_rate = max(-0.5, min(0.5, revenue_growth_rate))
            else:
                revenue_growth_rate = 0
        else:
            cash_flow_volatility = 0.5
            revenue_growth_rate = 0
        
        # Gross burn rate
        gross_burn_rate = total_expenses / months_count if months_count > 0 else 0
        
        # Balance analysis
        if 'balances.available' in data.columns:
            data['balances.available'] = pd.to_numeric(data['balances.available'], errors='coerce').fillna(0)
            avg_month_end_balance = data.groupby('year_month')['balances.available'].last().mean()
            
            # Negative balance days
            data['is_negative'] = data['balances.available'] < 0
            negative_days_per_month = data.groupby('year_month')['is_negative'].sum()
            avg_negative_days = negative_days_per_month.mean() if len(negative_days_per_month) > 0 else 0
        else:
            avg_month_end_balance = 1000  # Default
            avg_negative_days = 0
        
        # Bounced payments count
        if 'name' in data.columns:
            bounced_keywords = ['unpaid', 'returned', 'bounced', 'insufficient', 'nsf', 'failed', 'declined']
            bounced_payments = data['name'].str.lower().str.contains('|'.join(bounced_keywords), na=False).sum()
        else:
            bounced_payments = 0
        
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
            "Revenue Growth Rate": round(revenue_growth_rate, 4),
            "Average Month-End Balance": round(avg_month_end_balance, 2),
            "Average Negative Balance Days per Month": round(avg_negative_days, 1),
            "Number of Bounced Payments": int(bounced_payments),
            "monthly_summary": monthly_summary
        }
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        import traceback
        print(f"Full error traceback: {traceback.format_exc()}")
        return {}


def calculate_revenue_insights(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate revenue-specific insights.
    
    Args:
        df: DataFrame with transaction data
        
    Returns:
        Dictionary of revenue insights
    """
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
    
    # Determine name column
    name_column = 'name' if 'name' in revenue_data.columns else 'name_y' if 'name_y' in revenue_data.columns else None
    
    if name_column is None:
        unique_revenue_sources = 0
    else:
        unique_revenue_sources = revenue_data[name_column].nunique()
    
    # Calculate daily metrics
    revenue_data['date'] = pd.to_datetime(revenue_data['date'])
    revenue_data['date_only'] = revenue_data['date'].dt.date
    
    daily_revenue = revenue_data.groupby('date_only').agg({
        'amount': ['count', lambda x: abs(x).sum()]
    }).round(2)
    
    daily_revenue.columns = ['daily_transaction_count', 'daily_revenue_amount']
    
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


def create_monthly_breakdown(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Create monthly breakdown by subcategory.
    
    Args:
        df: DataFrame with transaction data
        
    Returns:
        Tuple of (pivot_counts, pivot_amounts) DataFrames
    """
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
    
    # Pivot tables
    pivot_counts = monthly_breakdown.pivot(index='year_month', columns='subcategory', values='Transaction_Count').fillna(0)
    pivot_amounts = monthly_breakdown.pivot(index='year_month', columns='subcategory', values='Total_Amount').fillna(0)
    
    return pivot_counts, pivot_amounts


def create_categorized_csv(df: pd.DataFrame) -> Optional[str]:
    """
    Create CSV string with categorization.
    
    Args:
        df: DataFrame with transaction data
        
    Returns:
        CSV string or None if data is empty
    """
    if df.empty:
        return None
    
    # Apply categorization
    categorized_df = categorize_transactions(df.copy())
    
    # Select and order columns for export
    export_columns = [
        'date', 'name', 'amount', 'subcategory',
        'is_revenue', 'is_expense', 'is_debt_repayment', 'is_debt'
    ]
    
    # Add additional columns if they exist
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


def analyze_loans_and_repayments(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Comprehensive analysis of loans received and debt repayments.
    
    Args:
        df: DataFrame with transaction data
        
    Returns:
        Dictionary of loans and repayments analysis
    """
    if df.empty:
        return {}
    
    # Apply categorization
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
    
    # Loans analysis
    if not loans_data.empty:
        analysis['total_loans_received'] = loans_data['amount_abs'].sum()
        analysis['loan_count'] = len(loans_data)
        analysis['avg_loan_amount'] = loans_data['amount_abs'].mean()
        analysis['largest_loan'] = loans_data['amount_abs'].max()
        analysis['smallest_loan'] = loans_data['amount_abs'].min()
        
        # Monthly loans
        loans_by_month = loans_data.groupby('month')['amount_abs'].agg(['count', 'sum']).reset_index()
        loans_by_month['month_str'] = loans_by_month['month'].astype(str)
        analysis['loans_by_month'] = loans_by_month
        
        # Lender analysis
        if 'name' in loans_data.columns:
            loans_data['lender_clean'] = loans_data['name'].str.lower().str.strip()
            loans_by_lender = loans_data.groupby('lender_clean')['amount_abs'].agg(['count', 'sum']).reset_index()
            loans_by_lender = loans_by_lender.sort_values('sum', ascending=False)
            analysis['loans_by_lender'] = loans_by_lender
    else:
        analysis.update({
            'total_loans_received': 0, 'loan_count': 0, 'avg_loan_amount': 0,
            'largest_loan': 0, 'smallest_loan': 0, 'loans_by_month': pd.DataFrame(),
            'loans_by_lender': pd.DataFrame()
        })
    
    # Repayments analysis
    if not repayments_data.empty:
        analysis['total_repayments_made'] = repayments_data['amount_abs'].sum()
        analysis['repayment_count'] = len(repayments_data)
        analysis['avg_repayment_amount'] = repayments_data['amount_abs'].mean()
        
        # Monthly repayments
        repayments_by_month = repayments_data.groupby('month')['amount_abs'].agg(['count', 'sum']).reset_index()
        repayments_by_month['month_str'] = repayments_by_month['month'].astype(str)
        analysis['repayments_by_month'] = repayments_by_month
    else:
        analysis.update({
            'total_repayments_made': 0, 'repayment_count': 0, 'avg_repayment_amount': 0,
            'repayments_by_month': pd.DataFrame()
        })
    
    # Calculated metrics
    total_loans = analysis.get('total_loans_received', 0)
    total_repaid = analysis.get('total_repayments_made', 0)
    
    analysis['net_borrowing'] = total_loans - total_repaid
    analysis['repayment_ratio'] = total_repaid / total_loans if total_loans > 0 else 0
    
    return analysis
