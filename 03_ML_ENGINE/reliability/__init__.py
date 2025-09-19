# 03_ML_ENGINE/reliability/__init__.py
"""
ML Reliability Module - Phase 2, Step 4
Enhanced error handling and circuit breaker patterns for ML operations

This module provides:
- ML-specific circuit breakers with model failure protection
- Comprehensive error classification and recovery strategies
- Performance optimization with sub-20ms targets
- Health monitoring and recovery mechanisms
"""

from .ml_circuit_breaker import (
    MLCircuitBreaker,
    MLCircuitBreakerRegistry,
    MLModelType,
    ErrorType,
    ml_circuit_breaker,
    ml_circuit_registry,
    get_ml_system_health_dashboard
)

from .error_handler import (
    MLErrorHandler,
    ErrorSeverity,
    ErrorCategory,
    ErrorContext,
    ml_error_handler,
    error_recovery_strategy,
    get_error_analysis_report
)

__all__ = [
    'MLCircuitBreaker',
    'MLCircuitBreakerRegistry',
    'MLModelType',
    'ErrorType',
    'ml_circuit_breaker',
    'ml_circuit_registry',
    'get_ml_system_health_dashboard',
    'MLErrorHandler',
    'ErrorSeverity',
    'ErrorCategory',
    'ErrorContext',
    'ml_error_handler',
    'error_recovery_strategy',
    'get_error_analysis_report'
]