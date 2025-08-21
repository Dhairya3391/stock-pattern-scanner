"""
Comprehensive Error Handling and Logging System

This module provides robust error handling, logging, and monitoring capabilities
for the Advanced Pattern Scanner system. It includes:
- Structured logging with different levels
- Error categorization and handling strategies
- Performance monitoring and alerting
- Graceful degradation mechanisms
"""

import logging
import traceback
import functools
import time
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Union
from pathlib import Path
from enum import Enum
import json

from loguru import logger as loguru_logger


class ErrorCategory(Enum):
    """Error categories for classification and handling."""
    DATA_ERROR = "data_error"
    MODEL_ERROR = "model_error"
    PATTERN_ERROR = "pattern_error"
    SYSTEM_ERROR = "system_error"
    NETWORK_ERROR = "network_error"
    VALIDATION_ERROR = "validation_error"
    CONFIGURATION_ERROR = "configuration_error"


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PatternScannerError(Exception):
    """Base exception for pattern scanner errors."""
    
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.SYSTEM_ERROR,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM, context: Optional[Dict] = None):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging."""
        return {
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "traceback": traceback.format_exc()
        }


class DataError(PatternScannerError):
    """Data-related errors."""
    
    def __init__(self, message: str, symbol: Optional[str] = None, **kwargs):
        context = kwargs.get('context', {})
        if symbol:
            context['symbol'] = symbol
        super().__init__(message, ErrorCategory.DATA_ERROR, context=context, **kwargs)


class ModelError(PatternScannerError):
    """ML model-related errors."""
    
    def __init__(self, message: str, model_name: Optional[str] = None, **kwargs):
        context = kwargs.get('context', {})
        if model_name:
            context['model_name'] = model_name
        super().__init__(message, ErrorCategory.MODEL_ERROR, context=context, **kwargs)


class PatternError(PatternScannerError):
    """Pattern detection-related errors."""
    
    def __init__(self, message: str, pattern_type: Optional[str] = None, **kwargs):
        context = kwargs.get('context', {})
        if pattern_type:
            context['pattern_type'] = pattern_type
        super().__init__(message, ErrorCategory.PATTERN_ERROR, context=context, **kwargs)


class NetworkError(PatternScannerError):
    """Network and API-related errors."""
    
    def __init__(self, message: str, endpoint: Optional[str] = None, **kwargs):
        context = kwargs.get('context', {})
        if endpoint:
            context['endpoint'] = endpoint
        super().__init__(message, ErrorCategory.NETWORK_ERROR, context=context, **kwargs)


class LoggingManager:
    """
    Centralized logging manager with structured logging capabilities.
    
    Provides different logging levels, formatters, and output destinations
    with performance monitoring and error tracking.
    """
    
    def __init__(self, log_dir: str = "logs", log_level: str = "INFO"):
        """
        Initialize the logging manager.
        
        Args:
            log_dir: Directory for log files
            log_level: Minimum logging level
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.log_level = log_level
        self.error_counts = {}
        self.performance_metrics = {}
        
        self._setup_loguru()
        self._setup_standard_logging()
    
    def _setup_loguru(self):
        """Setup loguru logger with custom configuration."""
        # Remove default handler
        loguru_logger.remove()
        
        # Console handler with colors
        loguru_logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<level>{message}</level>",
            level=self.log_level,
            colorize=True
        )
        
        # File handler for general logs
        loguru_logger.add(
            self.log_dir / "pattern_scanner.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level=self.log_level,
            rotation="10 MB",
            retention="30 days",
            compression="zip"
        )
        
        # Error-specific file handler
        loguru_logger.add(
            self.log_dir / "errors.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level="ERROR",
            rotation="5 MB",
            retention="60 days",
            compression="zip"
        )
        
        # Performance metrics handler
        loguru_logger.add(
            self.log_dir / "performance.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
            filter=lambda record: "PERFORMANCE" in record["extra"],
            rotation="5 MB",
            retention="30 days"
        )
    
    def _setup_standard_logging(self):
        """Setup standard Python logging for compatibility."""
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(self.log_dir / "standard.log")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, self.log_level))
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.log_level))
        root_logger.addHandler(file_handler)
    
    def log_error(self, error: PatternScannerError, extra_context: Optional[Dict] = None):
        """
        Log a structured error with full context.
        
        Args:
            error: PatternScannerError instance
            extra_context: Additional context information
        """
        context = error.context.copy()
        if extra_context:
            context.update(extra_context)
        
        # Update error counts
        error_key = f"{error.category.value}_{error.severity.value}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Log with appropriate level
        if error.severity == ErrorSeverity.CRITICAL:
            loguru_logger.critical(f"{error.message} | Context: {context}")
        elif error.severity == ErrorSeverity.HIGH:
            loguru_logger.error(f"{error.message} | Context: {context}")
        elif error.severity == ErrorSeverity.MEDIUM:
            loguru_logger.warning(f"{error.message} | Context: {context}")
        else:
            loguru_logger.info(f"{error.message} | Context: {context}")
    
    def log_performance(self, operation: str, duration: float, 
                       success: bool = True, extra_metrics: Optional[Dict] = None):
        """
        Log performance metrics for operations.
        
        Args:
            operation: Name of the operation
            duration: Duration in seconds
            success: Whether the operation succeeded
            extra_metrics: Additional metrics to log
        """
        metrics = {
            "operation": operation,
            "duration_seconds": duration,
            "success": success,
            "timestamp": datetime.now().isoformat()
        }
        
        if extra_metrics:
            metrics.update(extra_metrics)
        
        # Store metrics
        if operation not in self.performance_metrics:
            self.performance_metrics[operation] = []
        self.performance_metrics[operation].append(metrics)
        
        # Log performance
        loguru_logger.bind(PERFORMANCE=True).info(json.dumps(metrics))
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of logged errors."""
        return {
            "error_counts": self.error_counts,
            "total_errors": sum(self.error_counts.values()),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics."""
        summary = {}
        
        for operation, metrics in self.performance_metrics.items():
            durations = [m["duration_seconds"] for m in metrics]
            success_rate = sum(1 for m in metrics if m["success"]) / len(metrics)
            
            summary[operation] = {
                "total_calls": len(metrics),
                "avg_duration": sum(durations) / len(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "success_rate": success_rate
            }
        
        return summary


class ErrorHandler:
    """
    Centralized error handling with graceful degradation strategies.
    
    Provides different handling strategies based on error type and severity,
    with automatic fallback mechanisms and recovery procedures.
    """
    
    def __init__(self, logging_manager: LoggingManager):
        """
        Initialize error handler.
        
        Args:
            logging_manager: LoggingManager instance
        """
        self.logging_manager = logging_manager
        self.fallback_strategies = {}
        self.recovery_procedures = {}
        
        self._setup_default_strategies()
    
    def _setup_default_strategies(self):
        """Setup default error handling strategies."""
        # Data error strategies
        self.fallback_strategies[ErrorCategory.DATA_ERROR] = self._handle_data_error
        
        # Model error strategies
        self.fallback_strategies[ErrorCategory.MODEL_ERROR] = self._handle_model_error
        
        # Pattern error strategies
        self.fallback_strategies[ErrorCategory.PATTERN_ERROR] = self._handle_pattern_error
        
        # Network error strategies
        self.fallback_strategies[ErrorCategory.NETWORK_ERROR] = self._handle_network_error
    
    def handle_error(self, error: PatternScannerError, 
                    context: Optional[Dict] = None) -> Any:
        """
        Handle an error with appropriate strategy.
        
        Args:
            error: PatternScannerError instance
            context: Additional context for handling
            
        Returns:
            Result of fallback strategy or None
        """
        # Log the error
        self.logging_manager.log_error(error, context)
        
        # Apply fallback strategy
        if error.category in self.fallback_strategies:
            try:
                return self.fallback_strategies[error.category](error, context)
            except Exception as fallback_error:
                # Log fallback failure
                fallback_err = PatternScannerError(
                    f"Fallback strategy failed: {fallback_error}",
                    ErrorCategory.SYSTEM_ERROR,
                    ErrorSeverity.HIGH,
                    {"original_error": error.to_dict()}
                )
                self.logging_manager.log_error(fallback_err)
        
        return None
    
    def _handle_data_error(self, error: DataError, context: Optional[Dict]) -> Any:
        """Handle data-related errors with fallback strategies."""
        if "symbol" in error.context:
            symbol = error.context["symbol"]
            loguru_logger.warning(f"Data error for {symbol}, attempting fallback data source")
            
            # Could implement fallback to different data provider
            # For now, return None to indicate failure
            return None
        
        return None
    
    def _handle_model_error(self, error: ModelError, context: Optional[Dict]) -> Any:
        """Handle model-related errors with fallback strategies."""
        if "model_name" in error.context:
            model_name = error.context["model_name"]
            loguru_logger.warning(f"Model error for {model_name}, falling back to traditional methods")
            
            # Return indication to use traditional pattern detection
            return {"use_traditional": True}
        
        return None
    
    def _handle_pattern_error(self, error: PatternError, context: Optional[Dict]) -> Any:
        """Handle pattern detection errors with fallback strategies."""
        if "pattern_type" in error.context:
            pattern_type = error.context["pattern_type"]
            loguru_logger.warning(f"Pattern detection error for {pattern_type}, skipping pattern")
            
            # Return empty pattern list
            return []
        
        return None
    
    def _handle_network_error(self, error: NetworkError, context: Optional[Dict]) -> Any:
        """Handle network-related errors with retry and fallback strategies."""
        if "endpoint" in error.context:
            endpoint = error.context["endpoint"]
            loguru_logger.warning(f"Network error for {endpoint}, will retry later")
            
            # Could implement retry logic here
            return None
        
        return None


def error_handler(category: ErrorCategory = ErrorCategory.SYSTEM_ERROR,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 fallback_value: Any = None):
    """
    Decorator for automatic error handling and logging.
    
    Args:
        category: Error category for classification
        severity: Error severity level
        fallback_value: Value to return on error
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except PatternScannerError as e:
                # Re-raise custom errors
                raise
            except Exception as e:
                # Convert to custom error
                error = PatternScannerError(
                    f"Error in {func.__name__}: {str(e)}",
                    category,
                    severity,
                    {
                        "function": func.__name__,
                        "args": str(args)[:200],  # Truncate for logging
                        "kwargs": str(kwargs)[:200]
                    }
                )
                
                # Get global error handler if available
                try:
                    from .logging_manager import get_global_logging_manager
                    logging_manager = get_global_logging_manager()
                    error_handler_instance = ErrorHandler(logging_manager)
                    result = error_handler_instance.handle_error(error)
                    
                    if result is not None:
                        return result
                except:
                    # Fallback to basic logging
                    loguru_logger.error(f"Error in {func.__name__}: {str(e)}")
                
                return fallback_value
        
        return wrapper
    return decorator


def performance_monitor(operation_name: Optional[str] = None):
    """
    Decorator for automatic performance monitoring.
    
    Args:
        operation_name: Name of the operation (defaults to function name)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            start_time = time.time()
            success = True
            result = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration = time.time() - start_time
                
                # Log performance
                try:
                    from .logging_manager import get_global_logging_manager
                    logging_manager = get_global_logging_manager()
                    logging_manager.log_performance(op_name, duration, success)
                except:
                    # Fallback logging
                    loguru_logger.info(f"Performance: {op_name} took {duration:.3f}s (success: {success})")
        
        return wrapper
    return decorator


# Global instances
_global_logging_manager = None
_global_error_handler = None


def get_global_logging_manager() -> LoggingManager:
    """Get the global logging manager instance."""
    global _global_logging_manager
    if _global_logging_manager is None:
        _global_logging_manager = LoggingManager()
    return _global_logging_manager


def get_global_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        logging_manager = get_global_logging_manager()
        _global_error_handler = ErrorHandler(logging_manager)
    return _global_error_handler


def setup_global_error_handling():
    """Setup global error handling for the application."""
    # Initialize global instances
    get_global_logging_manager()
    get_global_error_handler()
    
    # Setup global exception handler
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        error = PatternScannerError(
            f"Uncaught exception: {exc_value}",
            ErrorCategory.SYSTEM_ERROR,
            ErrorSeverity.CRITICAL,
            {
                "exception_type": exc_type.__name__,
                "traceback": ''.join(traceback.format_tb(exc_traceback))
            }
        )
        
        error_handler = get_global_error_handler()
        error_handler.handle_error(error)
    
    sys.excepthook = handle_exception
    loguru_logger.info("Global error handling setup complete")