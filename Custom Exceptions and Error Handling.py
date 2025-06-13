# app/core/exceptions.py
"""Custom exceptions for the business finance application."""

class BusinessFinanceException(Exception):
    """Base exception for business finance application."""
    
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

class DataValidationError(BusinessFinanceException):
    """Raised when data validation fails."""
    pass

class PlaidAPIError(BusinessFinanceException):
    """Raised when Plaid API operations fail."""
    pass

class ModelPredictionError(BusinessFinanceException):
    """Raised when ML model prediction fails."""
    pass

class FileProcessingError(BusinessFinanceException):
    """Raised when file processing fails."""
    pass

class ConfigurationError(BusinessFinanceException):
    """Raised when configuration is invalid."""
    pass

class InsufficientDataError(BusinessFinanceException):
    """Raised when there's insufficient data for analysis."""
    pass

class TransactionCategorizationError(BusinessFinanceException):
    """Raised when transaction categorization fails."""
    pass

# Error handling utilities
def handle_exception(func):
    """Decorator to handle exceptions and provide user-friendly error messages."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except BusinessFinanceException as e:
            # Log the error and re-raise with context
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Business logic error in {func.__name__}: {e.message}", 
                        extra=e.details)
            raise e
        except Exception as e:
            # Convert unexpected errors to BusinessFinanceException
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            raise BusinessFinanceException(
                f"An unexpected error occurred in {func.__name__}",
                {"original_error": str(e), "function": func.__name__}
            )
    return wrapper