# app/core/validators.py
"""Input validation utilities for the business finance application."""

import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, date
import re
from .exceptions import DataValidationError

class DataValidator:
    """Comprehensive data validation for business finance application."""
    
    @staticmethod
    def validate_loan_amount(amount: float) -> float:
        """Validate loan amount input."""
        if not isinstance(amount, (int, float)):
            raise DataValidationError("Loan amount must be a number")
        
        if amount < 0:
            raise DataValidationError("Loan amount must be positive")
        
        if amount > 10_000_000:  # £10M limit
            raise DataValidationError("Loan amount exceeds maximum limit of £10,000,000")
        
        return float(amount)
    
    @staticmethod
    def validate_director_score(score: int) -> int:
        """Validate director score input."""
        if not isinstance(score, (int, float)):
            raise DataValidationError("Director score must be a number")
        
        score = int(score)
        if not 0 <= score <= 100:
            raise DataValidationError("Director score must be between 0 and 100")
        
        return score
    
    @staticmethod
    def validate_company_age(age_months: int) -> int:
        """Validate company age in months."""
        if not isinstance(age_months, (int, float)):
            raise DataValidationError("Company age must be a number")
        
        age_months = int(age_months)
        if age_months < 0:
            raise DataValidationError("Company age cannot be negative")
        
        if age_months > 1200:  # 100 years
            raise DataValidationError("Company age seems unrealistic (>100 years)")
        
        return age_months
    
    @staticmethod
    def validate_date_range(start_date: date, end_date: date) -> tuple:
        """Validate date range inputs."""
        if not isinstance(start_date, date) or not isinstance(end_date, date):
            raise DataValidationError("Dates must be valid date objects")
        
        if start_date > end_date:
            raise DataValidationError("Start date must be before end date")
        
        if end_date > date.today():
            raise DataValidationError("End date cannot be in the future")
        
        # Check for reasonable date range (not more than 5 years)
        if (end_date - start_date).days > 1825:
            raise DataValidationError("Date range cannot exceed 5 years")
        
        return start_date, end_date
    
    @staticmethod
    def validate_industry(industry: str, valid_industries: List[str]) -> str:
        """Validate industry selection."""
        if not isinstance(industry, str):
            raise DataValidationError("Industry must be a string")
        
        if industry not in valid_industries:
            raise DataValidationError(f"Invalid industry. Must be one of: {', '.join(valid_industries)}")
        
        return industry
    
    @staticmethod
    def validate_transaction_data(data: pd.DataFrame) -> pd.DataFrame:
        """Validate transaction data DataFrame."""
        if not isinstance(data, pd.DataFrame):
            raise DataValidationError("Transaction data must be a pandas DataFrame")
        
        if data.empty:
            raise DataValidationError("Transaction data cannot be empty")
        
        # Required columns
        required_columns = ['amount', 'date', 'name_y']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise DataValidationError(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Validate data types
        try:
            data['date'] = pd.to_datetime(data['date'])
        except Exception:
            raise DataValidationError("Invalid date format in transaction data")
        
        try:
            data['amount'] = pd.to_numeric(data['amount'], errors='coerce')
        except Exception:
            raise DataValidationError("Invalid amount format in transaction data")
        
        # Check for null values in critical columns
        if data['amount'].isna().any():
            raise DataValidationError("Transaction amounts cannot be null")
        
        if data['date'].isna().any():
            raise DataValidationError("Transaction dates cannot be null")
        
        return data
    
    @staticmethod
    def validate_json_structure(json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate JSON structure for transaction data."""
        if not isinstance(json_data, dict):
            raise DataValidationError("JSON data must be a dictionary")
        
        required_keys = ['accounts', 'transactions']
        missing_keys = [key for key in required_keys if key not in json_data]
        
        if missing_keys:
            raise DataValidationError(f"Missing required keys in JSON: {', '.join(missing_keys)}")
        
        if not isinstance(json_data['accounts'], list):
            raise DataValidationError("'accounts' must be a list")
        
        if not isinstance(json_data['transactions'], list):
            raise DataValidationError("'transactions' must be a list")
        
        if len(json_data['accounts']) == 0:
            raise DataValidationError("At least one account is required")
        
        if len(json_data['transactions']) == 0:
            raise DataValidationError("At least one transaction is required")
        
        return json_data

class SecurityValidator:
    """Security-related validation utilities."""
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent path traversal attacks."""
        if not isinstance(filename, str):
            raise DataValidationError("Filename must be a string")
        
        # Remove path separators and dangerous characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)
        sanitized = sanitized.replace('..', '')
        
        if not sanitized:
            raise DataValidationError("Invalid filename")
        
        return sanitized
    
    @staticmethod
    def validate_file_size(file_size: int, max_size_mb: int = 50) -> bool:
        """Validate file size limits."""
        max_size_bytes = max_size_mb * 1024 * 1024
        
        if file_size > max_size_bytes:
            raise DataValidationError(f"File size exceeds {max_size_mb}MB limit")
        
        return True
    
    @staticmethod
    def validate_file_extension(filename: str, allowed_extensions: List[str]) -> bool:
        """Validate file extension."""
        if not filename:
            raise DataValidationError("Filename cannot be empty")
        
        extension = filename.lower().split('.')[-1]
        
        if extension not in [ext.lower() for ext in allowed_extensions]:
            raise DataValidationError(f"File extension '{extension}' not allowed. Allowed: {', '.join(allowed_extensions)}")
        
        return True

# Validation decorators
def validate_inputs(**validators):
    """Decorator to validate function inputs."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Validate keyword arguments
            for param_name, validator in validators.items():
                if param_name in kwargs:
                    try:
                        kwargs[param_name] = validator(kwargs[param_name])
                    except Exception as e:
                        raise DataValidationError(f"Validation failed for {param_name}: {str(e)}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator