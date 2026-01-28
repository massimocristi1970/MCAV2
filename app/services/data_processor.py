# app/services/data_processor.py
"""Enhanced data processing service with improved categorization and validation."""

import pandas as pd
import numpy as np
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, date
from rapidfuzz import fuzz

from ..core.exceptions import FileProcessingError, DataValidationError
from ..core.logger import get_logger, log_performance
from ..core.cache import CacheManager, DataCache
from ..core.validators import DataValidator, SecurityValidator

logger = get_logger("data_processor")

class TransactionCategorizer:
    """Advanced transaction categorization with ML-like capabilities."""
    
    def __init__(self):
        self.categorization_rules = self._load_categorization_rules()
        self.confidence_threshold = 0.8
    
    def _load_categorization_rules(self) -> Dict[str, Any]:
        """Load categorization rules and patterns."""
        
        return {
            'income_patterns': {
                'payment_processors': [
                    r'stripe', r'sumup', r'zettle', r'square', r'take\s*payments',
                    r'shopify', r'card\s+settlement', r'daily\s+takings', r'payout',
                    r'paypal', r'go\s*cardless', r'klarna', r'worldpay', r'izettle',
                    r'ubereats', r'just\s*eat', r'deliveroo', r'uber', r'bolt',
                    r'fresha', r'treatwell', r'taskrabbit', r'terminal', r'pos\s+deposit',
                    r'revolut', r'capital\s+on\s+tap', r'evo\s*payments?', r'tink',
                    r'teya(\s+solutions)?', r'talech', r'barclaycard', r'elavon', r'adyen'
                ],
                'direct_revenue': [
                    r'sales', r'revenue', r'income', r'payment\s+received',
                    r'customer\s+payment', r'invoice\s+payment', r'service\s+fee'
                ],
                'special_cases': [
                    (r'you\s?lend|yl\s?ii|yl\s?ltd|yl\s?limited|yl\s?a\s?limited', 
                     lambda text: 'Loans' if re.search(r'\b(fnd|fund|funding)\b', text) else 'Income')
                ]
            },
            'loan_patterns': [
                r'iwoca', r'capify', r'fundbox', r'got[\s\-]?capital', r'funding[\s\-]?circle',
                r'fleximize', r'marketfinance', r'liberis', r'esme[\s\-]?loans', r'thincats',
                r'white[\s\-]?oak', r'growth[\s\-]?street', r'nucleus[\s\-]?commercial[\s\-]?finance',
                r'ultimate[\s\-]?finance', r'just[\s\-]?cash[\s\-]?flow', r'boost[\s\-]?capital',
                r'merchant[\s\-]?money', r'capital[\s\-]?on[\s\-]?tap', r'kriya', r'uncapped',
                r'lendingcrowd', r'folk2folk', r'funding[\s\-]?tree', r'start[\s\-]?up[\s\-]?loans',
                r'loan', r'advance', r'financing'
            ],
            'debt_repayment_patterns': [
                r'repayment', r'loan\s+payment', r'debt\s+service', r'installment',
                r'instalment', r'payback', r'repay', r'amortization'
            ],
            'expense_patterns': {
                'operational': [
                    r'rent', r'utilities', r'insurance', r'supplies', r'inventory',
                    r'marketing', r'advertising', r'professional\s+services', r'legal',
                    r'accounting', r'consulting', r'office', r'equipment'
                ],
                'payroll': [
                    r'salary', r'wages', r'payroll', r'employee', r'staff',
                    r'pension', r'benefits', r'tax\s+payment', r'hmrc'
                ],
                'travel': [
                    r'travel', r'hotel', r'flight', r'transport', r'fuel',
                    r'parking', r'taxi', r'uber', r'train'
                ]
            },
            'special_inflow_patterns': [
                r'refund', r'rebate', r'grant', r'subsidy', r'dividend',
                r'interest\s+earned', r'investment\s+income', r'tax\s+refund'
            ],
            'special_outflow_patterns': [
                r'withdrawal', r'transfer\s+out', r'investment', r'capital\s+expenditure',
                r'equipment\s+purchase', r'asset\s+acquisition'
            ],
            'failed_payment_patterns': [
                r'unpaid', r'returned', r'bounced', r'insufficient\s+funds',
                r'nsf', r'declined', r'failed', r'reversed', r'chargeback'
            ]
        }
    
    @log_performance(logger)
    def categorize_transaction(self, transaction: Dict[str, Any]) -> Tuple[str, float]:
        """
        Categorize a single transaction with confidence score.
        
        IMPORTANT: The order of checks matters! Reversal/failed payment detection
        must happen BEFORE income/loan pattern matching to prevent misclassification
        of transactions like "STRIPE REVERSAL" being categorized as income.
        
        Returns:
            Tuple of (category, confidence_score)
        """
        
        # Extract transaction details
        name = str(transaction.get("name_y", transaction.get("name", ""))).lower()
        merchant_name = str(transaction.get("merchant_name", "")).lower()
        category = str(transaction.get("personal_finance_category.detailed", "")).lower()
        amount = transaction.get("amount_1", transaction.get("amount", 0))
        
        combined_text = f"{name} {merchant_name}"
        
        # Determine transaction direction
        is_credit = amount < 0  # Money coming in
        is_debit = amount > 0   # Money going out
        
        # STEP 1 (CRITICAL): Check for failed payment/reversal patterns FIRST!
        # This must happen before income/loan checks to prevent "STRIPE REVERSAL" 
        # being categorized as income
        failed_category, confidence = self._check_failed_payment_patterns(combined_text, category)
        if confidence > self.confidence_threshold:
            return failed_category, confidence
        
        # STEP 2: Check for refund indicators on credits
        if is_credit:
            refund_category, confidence = self._check_refund_patterns(combined_text)
            if confidence > self.confidence_threshold:
                return refund_category, confidence
        
        # STEP 3: Check for special income patterns (only for credits)
        if is_credit:
            income_category, confidence = self._check_income_patterns(combined_text)
            if confidence > self.confidence_threshold:
                return income_category, confidence
        
        # STEP 4: Check for loan patterns
        loan_category, confidence = self._check_loan_patterns(combined_text, is_credit)
        if confidence > self.confidence_threshold:
            return loan_category, confidence
        
        # STEP 5: Check for debt repayment patterns (only for debits)
        if is_debit:
            debt_category, confidence = self._check_debt_patterns(combined_text)
            if confidence > self.confidence_threshold:
                return debt_category, confidence
        
        # STEP 6: Use Plaid category as fallback (with credit/debit awareness)
        plaid_category, confidence = self._map_plaid_category(category, is_credit, is_debit)
        if confidence > 0.5:
            return plaid_category, confidence
        
        # STEP 7: Basic fallback based on amount direction
        if is_credit:
            return "Uncategorised", 0.3
        else:
            return "Expenses", 0.3
    
    def _check_income_patterns(self, text: str) -> Tuple[str, float]:
        """Check for income-related patterns."""
        
        # Payment processors (high confidence)
        for pattern in self.categorization_rules['income_patterns']['payment_processors']:
            if re.search(pattern, text, re.IGNORECASE):
                return "Income", 0.95
        
        # Direct revenue indicators
        for pattern in self.categorization_rules['income_patterns']['direct_revenue']:
            if re.search(pattern, text, re.IGNORECASE):
                return "Income", 0.85
        
        # Special cases with conditional logic
        for pattern, condition_func in self.categorization_rules['income_patterns']['special_cases']:
            if re.search(pattern, text, re.IGNORECASE):
                result = condition_func(text)
                return result, 0.9
        
        return "Unknown", 0.0
    
    def _check_loan_patterns(self, text: str, is_credit: bool) -> Tuple[str, float]:
        """Check for loan-related patterns."""
        
        for pattern in self.categorization_rules['loan_patterns']:
            if re.search(pattern, text, re.IGNORECASE):
                if is_credit:
                    return "Loans", 0.9
                else:
                    return "Debt Repayments", 0.9
        
        return "Unknown", 0.0
    
    def _check_debt_patterns(self, text: str) -> Tuple[str, float]:
        """Check for debt repayment patterns."""
        
        for pattern in self.categorization_rules['debt_repayment_patterns']:
            if re.search(pattern, text, re.IGNORECASE):
                return "Debt Repayments", 0.85
        
        return "Unknown", 0.0
    
    def _check_failed_payment_patterns(self, text: str, category: str = "") -> Tuple[str, float]:
        """
        Check for failed payment patterns.
        
        This should be called FIRST in the categorization pipeline to catch
        reversals before they match income/loan provider patterns.
        """
        
        # Extended patterns to catch more reversal/failed payment scenarios
        extended_patterns = [
            r'reversal', r'reversed', r'chargeback', r'dispute',
            r'refund\s+fee', r'rejected', r'cancelled\s+payment',
            r'payment\s+returned'
        ]
        
        # Check base patterns
        for pattern in self.categorization_rules['failed_payment_patterns']:
            if re.search(pattern, text, re.IGNORECASE):
                return "Failed Payment", 0.95
        
        # Check extended patterns
        for pattern in extended_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return "Failed Payment", 0.95
        
        # Check Plaid category for failed payments
        failed_plaid_categories = [
            'bank_fees_insufficient_funds', 'bank_fees_late_payment',
            'bank_fees_overdraft', 'bank_fees_returned_payment'
        ]
        if category.lower() in failed_plaid_categories:
            return "Failed Payment", 0.95
        
        return "Unknown", 0.0
    
    def _check_refund_patterns(self, text: str) -> Tuple[str, float]:
        """Check for refund/rebate patterns on credit transactions."""
        
        refund_patterns = [
            r'refund', r'rebate', r'credit\s+adj', r'adjustment',
            r'cashback', r'reimburs', r'money\s+back', r'return\s+credit'
        ]
        
        for pattern in refund_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return "Special Inflow", 0.9
        
        return "Unknown", 0.0
    
    def _map_plaid_category(self, category: str, is_credit: bool = False, is_debit: bool = True) -> Tuple[str, float]:
        """
        Map Plaid categories to our categories.
        
        CRITICAL FIX: Takes into account whether transaction is credit or debit.
        Credits with expense-like Plaid categories are likely refunds, not expenses.
        
        Args:
            category: The Plaid category string
            is_credit: Whether this is a credit (money coming in)
            is_debit: Whether this is a debit (money going out)
        
        Returns:
            Tuple of (category, confidence_score)
        """
        
        plaid_mapping = {
            "income_wages": ("Income", 0.8),
            "income_other_income": ("Income", 0.7),
            "income_dividends": ("Special Inflow", 0.8),
            "income_interest_earned": ("Special Inflow", 0.8),
            "transfer_in_cash_advances_and_loans": ("Loans", 0.9),
            "loan_payments_credit_card_payment": ("Debt Repayments", 0.9),
            "loan_payments_personal_loan_payment": ("Debt Repayments", 0.9),
            "loan_payments_other_payment": ("Debt Repayments", 0.8),
            "bank_fees_insufficient_funds": ("Failed Payment", 0.95),
            "bank_fees_late_payment": ("Failed Payment", 0.95),
            "bank_fees_overdraft": ("Failed Payment", 0.95),
            "bank_fees_returned_payment": ("Failed Payment", 0.95),
        }
        
        # Exact match
        if category in plaid_mapping:
            return plaid_mapping[category]
        
        # Partial matches
        if category.startswith("income_"):
            return "Income", 0.6
        elif category.startswith("loan_payments_"):
            return "Debt Repayments", 0.7
        elif category.startswith("bank_fees_"):
            return "Failed Payment", 0.8
        elif category.startswith("transfer_in_"):
            return "Special Inflow", 0.6
        elif category.startswith("transfer_out_"):
            return "Special Outflow", 0.6
        
        # CRITICAL FIX: Expense-like categories must respect credit/debit direction
        expense_prefixes = [
            "entertainment_", "food_and_drink_", "general_merchandise_",
            "general_services_", "rent_and_utilities_", "transportation_",
            "travel_", "home_improvement_", "medical_", "personal_care_",
            "government_and_non_profit_"
        ]
        
        if any(category.startswith(prefix) for prefix in expense_prefixes):
            if is_debit:
                return "Expenses", 0.7
            else:
                # Credit with expense-like category = likely a refund
                return "Special Inflow", 0.6
        
        return "Uncategorised", 0.1

class DataProcessor:
    """Enhanced data processing service."""
    
    def __init__(self):
        self.categorizer = TransactionCategorizer()
        self.validator = DataValidator()
        self.security_validator = SecurityValidator()
    
    @log_performance(logger)
    #@CacheManager.cache_data(ttl=1800)
    def process_json_data(
        self, 
        json_data: Dict[str, Any],
        include_confidence: bool = False
    ) -> pd.DataFrame:
        """
        Process JSON transaction data with enhanced categorization.
        
        Args:
            json_data: Raw JSON data from Plaid or file upload
            include_confidence: Whether to include categorization confidence scores
            
        Returns:
            Processed DataFrame with categorized transactions
        """
        try:
            # Validate JSON structure
            json_data = self.validator.validate_json_structure(json_data)
            
            accounts = json_data.get('accounts', [])
            transactions = json_data.get('transactions', [])
            
            # Normalize nested JSON structures
            accounts_df = pd.json_normalize(accounts)
            transactions_df = pd.json_normalize(transactions)
            
            # Validate required columns
            required_account_cols = ['account_id']
            required_transaction_cols = ['account_id', 'amount', 'date']
            
            missing_account_cols = [col for col in required_account_cols if col not in accounts_df.columns]
            missing_transaction_cols = [col for col in required_transaction_cols if col not in transactions_df.columns]
            
            if missing_account_cols:
                raise DataValidationError(f"Missing account columns: {missing_account_cols}")
            if missing_transaction_cols:
                raise DataValidationError(f"Missing transaction columns: {missing_transaction_cols}")
            
            # Merge accounts and transactions
            data = pd.merge(accounts_df, transactions_df, on="account_id", how="right")
            
            if data.empty:
                raise DataValidationError("No transactions found after merging with accounts")
            
            # Select and rename columns for consistency
            column_mapping = {
                'name_y': 'transaction_name',
                'name_x': 'account_name',
                'authorized_date': 'authorized_date',
                'date': 'date'
            }
            
            # Ensure we have the essential columns
            essential_columns = [
                'account_id', 'balances.available', 'amount', 'transaction_name',
                'authorized_date', 'date', 'merchant_name', 'website',
                'category', 'payment_channel', 'personal_finance_category.confidence_level',
                'personal_finance_category.detailed', 'personal_finance_category.primary'
            ]
            
            # Create missing columns with default values
            for col in essential_columns:
                if col not in data.columns:
                    if col == 'transaction_name':
                        data[col] = data.get('name_y', data.get('name', 'Unknown Transaction'))
                    elif col == 'authorized_date':
                        data[col] = data.get('date', pd.NaT)
                    else:
                        data[col] = None
            
            # Clean and process the data
            data = self._clean_data(data)
            
            # Enhanced transaction categorization
            data = self._categorize_transactions(data, include_confidence)
            
            # Add derived columns
            data = self._add_derived_columns(data)
            
            # Final validation
            data = self.validator.validate_transaction_data(data)
            
            logger.info(f"Processed {len(data)} transactions successfully")
            return data
            
        except Exception as e:
            logger.error(f"Error processing JSON data: {str(e)}")
            raise FileProcessingError(f"Failed to process transaction data: {str(e)}")
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize data."""
        
        # Convert dates
        date_columns = ['date', 'authorized_date', 'authorized_datetime']
        for col in date_columns:
            if col in data.columns:
                data[col] = pd.to_datetime(data[col], errors='coerce')
        
        # Clean amounts
        if 'amount' in data.columns:
            data['amount'] = pd.to_numeric(data['amount'], errors='coerce')
            data['amount_original'] = data['amount']  # Keep original for balance calculations
            data['amount'] = data['amount'].abs()  # Use absolute values for calculations
        
        # Clean text fields
        text_columns = ['transaction_name', 'merchant_name', 'account_name']
        for col in text_columns:
            if col in data.columns:
                data[col] = data[col].astype(str).str.strip()
                data[col] = data[col].replace('nan', '')
        
        # Sort by date (most recent first)
        if 'date' in data.columns:
            data = data.sort_values('date', ascending=False).reset_index(drop=True)
        
        # Recalculate running balances
        data = self._recalculate_balances(data)
        
        return data
    
    def _recalculate_balances(self, data: pd.DataFrame) -> pd.DataFrame:
        """Recalculate running account balances."""
        
        if 'balances.available' not in data.columns or 'amount_original' not in data.columns:
            return data
        
        try:
            # Convert balance to numeric
            data['balances.available'] = pd.to_numeric(data['balances.available'], errors='coerce').fillna(0)
            
            # Use the most recent balance as starting point
            current_balance = data.iloc[0]['balances.available'] if len(data) > 0 else 0
            updated_balances = [current_balance]
            
            # Calculate running balance (working backwards from most recent)
            for i in range(1, len(data)):
                # Add back the transaction amount (since we're going backwards in time)
                current_balance += data.iloc[i]['amount_original']
                updated_balances.append(current_balance)
            
            data['calculated_balance'] = updated_balances
            
        except Exception as e:
            logger.warning(f"Error recalculating balances: {str(e)}")
            data['calculated_balance'] = data.get('balances.available', 0)
        
        return data
    
    def _categorize_transactions(self, data: pd.DataFrame, include_confidence: bool) -> pd.DataFrame:
        """Categorize transactions using enhanced categorization."""
        
        categories = []
        confidences = []
        
        for _, transaction in data.iterrows():
            category, confidence = self.categorizer.categorize_transaction(transaction.to_dict())
            categories.append(category)
            confidences.append(confidence)
        
        data['subcategory'] = categories
        
        if include_confidence:
            data['categorization_confidence'] = confidences
        
        # Add boolean columns for easy filtering
        data['is_revenue'] = data['subcategory'].isin(['Income', 'Special Inflow'])
        data['is_expense'] = data['subcategory'].isin(['Expenses', 'Special Outflow'])
        data['is_debt_repayment'] = data['subcategory'].isin(['Debt Repayments'])
        data['is_debt'] = data['subcategory'].isin(['Loans'])
        data['is_failed_payment'] = data['subcategory'].isin(['Failed Payment'])
        
        return data
    
    def _add_derived_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add derived columns for analysis."""
        
        # Add time-based columns
        if 'date' in data.columns:
            data['year'] = data['date'].dt.year
            data['month'] = data['date'].dt.month
            data['quarter'] = data['date'].dt.quarter
            data['weekday'] = data['date'].dt.dayofweek
            data['is_weekend'] = data['weekday'].isin([5, 6])
            data['month_name'] = data['date'].dt.strftime('%B')
            data['year_month'] = data['date'].dt.to_period('M')
        
        # Add amount ranges for analysis
        if 'amount' in data.columns:
            data['amount_range'] = pd.cut(
                data['amount'], 
                bins=[0, 50, 200, 1000, 5000, float('inf')],
                labels=['Small (£0-50)', 'Medium (£50-200)', 'Large (£200-1K)', 
                       'Very Large (£1K-5K)', 'Exceptional (£5K+)']
            )
        
        # Add transaction frequency features
        if 'transaction_name' in data.columns:
            # Count transactions per merchant/name
            name_counts = data['transaction_name'].value_counts()
            data['merchant_frequency'] = data['transaction_name'].map(name_counts)
            
            # Identify recurring transactions
            data['is_recurring'] = data['merchant_frequency'] >= 3
        
        return data
    
    @log_performance(logger)
    def process_uploaded_file(
        self, 
        uploaded_file, 
        start_date: Optional[date] = None, 
        end_date: Optional[date] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process uploaded transaction file with date filtering.
        
        Returns:
            Tuple of (account_df, transaction_df)
        """
        
        try:
            # Security validation
            if hasattr(uploaded_file, 'name'):
                self.security_validator.validate_file_extension(
                    uploaded_file.name, ['json']
                )
            
            if hasattr(uploaded_file, 'size'):
                self.security_validator.validate_file_size(uploaded_file.size)
            
            # Load JSON data
            json_data = json.load(uploaded_file)
            
            # Extract accounts and transactions
            accounts = json_data.get('accounts', [])
            transactions = json_data.get('transactions', [])
            
            # Date filtering
            if start_date and end_date:
                transactions = self._filter_transactions_by_date(
                    transactions, start_date, end_date
                )
            
            # Process account data
            account_df = self._process_account_data(accounts)
            
            # Process transaction data
            filtered_json = {
                'accounts': accounts,
                'transactions': transactions
            }
            
            transaction_df = self.process_json_data(filtered_json, include_confidence=True)
            
            # Add account routing information
            transaction_df = self._add_account_routing_info(transaction_df, accounts)
            
            logger.info(f"Processed uploaded file: {len(account_df)} accounts, {len(transaction_df)} transactions")
            
            return account_df, transaction_df
            
        except Exception as e:
            logger.error(f"Error processing uploaded file: {str(e)}")
            raise FileProcessingError(f"Failed to process uploaded file: {str(e)}")
    
    def _filter_transactions_by_date(
        self, 
        transactions: List[Dict], 
        start_date: date, 
        end_date: date
    ) -> List[Dict]:
        """Filter transactions by date range."""
        
        filtered_transactions = []
        
        for txn in transactions:
            try:
                txn_date = pd.to_datetime(txn.get('date')).date()
                if start_date <= txn_date <= end_date:
                    filtered_transactions.append(txn)
            except Exception as e:
                logger.warning(f"Skipping transaction with invalid date: {txn.get('date')} - {str(e)}")
                continue
        
        return filtered_transactions
    
    def _process_account_data(self, accounts: List[Dict]) -> pd.DataFrame:
        """Process account data into DataFrame."""
        
        account_summaries = []
        
        for acct in accounts:
            account_summaries.append({
                'account_id': acct.get('account_id'),
                'account_name': acct.get('name', 'Unknown'),
                'account_type': acct.get('type', 'Unknown'),
                'account_subtype': acct.get('subtype', 'Unknown'),
                'balance_available': acct.get('balances', {}).get('available', 0),
                'balance_current': acct.get('balances', {}).get('current', 0),
                'sort_code': acct.get('sort_code', 'N/A'),
                'account_number': acct.get('account', 'N/A'),
                'currency_code': acct.get('balances', {}).get('iso_currency_code', 'GBP')
            })
        
        return pd.DataFrame(account_summaries)
    
    def _add_account_routing_info(self, data: pd.DataFrame, accounts: List[Dict]) -> pd.DataFrame:
        """Add account routing information to transaction data."""
        
        # Create routing lookup
        routing_data = {}
        for acct in accounts:
            account_id = acct.get('account_id')
            routing_data[account_id] = {
                'sort_code': acct.get('sort_code', 'N/A'),
                'account_number': acct.get('account', 'N/A'),
                'account_name': acct.get('name', 'Unnamed Account')
            }
        
        # Add routing information
        data['is_authorised_account'] = data['account_id'].isin(routing_data.keys())
        data['sort_code'] = data['account_id'].map(
            lambda x: routing_data.get(x, {}).get('sort_code', 'N/A')
        )
        data['account_number'] = data['account_id'].map(
            lambda x: routing_data.get(x, {}).get('account_number', 'N/A')
        )
        data['linked_account_name'] = data['account_id'].map(
            lambda x: routing_data.get(x, {}).get('account_name', 'Unnamed Account')
        )
        
        return data
    
    def filter_data_by_timeframe(self, data: pd.DataFrame, months: int) -> pd.DataFrame:
        """Filter data to last N months."""
        
        if data.empty or 'date' not in data.columns:
            return data
        
        data = data.copy()
        data['date'] = pd.to_datetime(data['date'])
        
        latest_date = data['date'].max()
        start_date = (latest_date - pd.DateOffset(months=months)).replace(day=1)
        
        filtered_data = data[data['date'] >= start_date]
        
        logger.info(f"Filtered data to last {months} months: {len(filtered_data)} transactions")
        
        return filtered_data
    
    def get_data_quality_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate a data quality report."""
        
        if data.empty:
            return {"error": "No data to analyze"}
        
        # Basic statistics
        total_transactions = len(data)
        date_range = {
            'start': data['date'].min().isoformat() if 'date' in data.columns else None,
            'end': data['date'].max().isoformat() if 'date' in data.columns else None,
            'days_covered': (data['date'].max() - data['date'].min()).days if 'date' in data.columns else 0
        }
        
        # Missing data analysis
        missing_data = {}
        for col in data.columns:
            missing_count = data[col].isna().sum()
            if missing_count > 0:
                missing_data[col] = {
                    'count': int(missing_count),
                    'percentage': round((missing_count / total_transactions) * 100, 2)
                }
        
        # Categorization quality
        categorization_quality = {}
        if 'subcategory' in data.columns:
            category_counts = data['subcategory'].value_counts()
            uncategorised_count = category_counts.get('Uncategorised', 0)
            
            categorization_quality = {
                'total_categories': len(category_counts),
                'categorization_rate': round(((total_transactions - uncategorised_count) / total_transactions) * 100, 2),
                'category_distribution': category_counts.to_dict()
            }
        
        # Data anomalies
        anomalies = []
        
        # Check for duplicate transactions
        if total_transactions > 0:
            duplicates = data.duplicated(subset=['date', 'amount', 'transaction_name']).sum()
            if duplicates > 0:
                anomalies.append(f"{duplicates} potential duplicate transactions")
        
        # Check for extreme amounts
        if 'amount' in data.columns and not data['amount'].empty:
            q99 = data['amount'].quantile(0.99)
            extreme_amounts = (data['amount'] > q99 * 5).sum()
            if extreme_amounts > 0:
                anomalies.append(f"{extreme_amounts} transactions with extremely high amounts")
        
        # Check for future dates
        if 'date' in data.columns:
            future_dates = (data['date'] > pd.Timestamp.now()).sum()
            if future_dates > 0:
                anomalies.append(f"{future_dates} transactions with future dates")
        
        return {
            'total_transactions': total_transactions,
            'date_range': date_range,
            'missing_data': missing_data,
            'categorization_quality': categorization_quality,
            'anomalies': anomalies,
            'data_completeness_score': round((1 - len(missing_data) / len(data.columns)) * 100, 2) if data.columns.any() else 0
        }

# Global data processor instance
data_processor = DataProcessor()