# app/core/logger.py
"""Logging configuration for the business finance application."""

import logging
import logging.config
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Dict, Any
from ..config.settings import settings

# Check if python-json-logger is available
try:
    from pythonjsonlogger import jsonlogger
    JSON_LOGGING_AVAILABLE = True
except ImportError:
    JSON_LOGGING_AVAILABLE = False

def setup_logging() -> None:
    """Set up logging configuration."""
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Build formatters dict
    formatters: Dict[str, Any] = {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "detailed": {
            "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
    }
    
    # Add JSON formatter only if python-json-logger is available
    if JSON_LOGGING_AVAILABLE:
        formatters["json"] = {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s"
        }
    
    # Logging configuration
    logging_config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": formatters,
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "standard",
                "stream": sys.stdout
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "filename": "logs/app.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "detailed",
                "filename": "logs/errors.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5
            }
        },
        "loggers": {
            "app": {
                "level": settings.LOG_LEVEL,
                "handlers": ["console", "file", "error_file"],
                "propagate": False
            },
            "app.plaid": {
                "level": "INFO",
                "handlers": ["file"],
                "propagate": False
            },
            "app.ml": {
                "level": "INFO",
                "handlers": ["file"],
                "propagate": False
            }
        },
        "root": {
            "level": "WARNING",
            "handlers": ["console"]
        }
    }
    
    # Apply configuration
    logging.config.dictConfig(logging_config)

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(f"app.{name}")

# Performance logging decorator
def log_performance(logger: logging.Logger):
    """Decorator to log function performance."""
    import time
    from functools import wraps
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(f"{func.__name__} completed in {duration:.3f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"{func.__name__} failed after {duration:.3f}s: {str(e)}")
                raise
        return wrapper
    return decorator

# Audit logging
class AuditLogger:
    """Audit logging for sensitive operations."""
    
    def __init__(self):
        self.logger = logging.getLogger("app.audit")
        
        # Create audit-specific handler
        handler = logging.handlers.RotatingFileHandler(
            "logs/audit.log",
            maxBytes=10485760,
            backupCount=10
        )
        
        formatter = logging.Formatter(
            "%(asctime)s [AUDIT] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_data_access(self, user_id: str, data_type: str, action: str):
        """Log data access events."""
        self.logger.info(f"User {user_id} performed {action} on {data_type}")
    
    def log_prediction_request(self, loan_amount: float, company_id: str):
        """Log ML prediction requests."""
        self.logger.info(f"Prediction requested for company {company_id}, loan amount: £{loan_amount}")
    
    def log_error(self, error_type: str, details: str):
        """Log application errors."""
        self.logger.error(f"Error: {error_type} - {details}")

# Global audit logger instance
audit_logger = AuditLogger()

# Initialize logging when module is imported
setup_logging()