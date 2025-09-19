# 03_ML_ENGINE/reliability/error_handler.py
"""
Enhanced Error Handling System for ML Model Integration - Phase 2, Step 4
Comprehensive error classification, recovery strategies, and monitoring

Location: #03_ML_ENGINE/reliability/error_handler.py

This module provides:
- Comprehensive error classification and analysis
- Recovery strategies for different error types
- Error pattern detection and prevention
- Graceful degradation mechanisms
- Error reporting and monitoring
- Context-aware error handling for ML operations
"""

import logging
import traceback
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union, Type, Tuple
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
import json
import re
from pathlib import Path
import functools
import inspect

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for ML operations"""
    CRITICAL = "critical"  # System failure, requires immediate attention
    HIGH = "high"  # Major functionality impacted
    MEDIUM = "medium"  # Degraded performance
    LOW = "low"  # Minor issues, system continues
    INFO = "info"  # Informational, not really an error


class ErrorCategory(Enum):
    """Categories of errors in ML system"""
    MODEL_FAILURE = "model_failure"  # ML model prediction failures
    DATA_QUALITY = "data_quality"  # Invalid or poor quality input data
    PERFORMANCE = "performance"  # Performance related issues
    RESOURCE = "resource"  # Memory, CPU, disk issues
    NETWORK = "network"  # Network connectivity issues
    CONFIGURATION = "configuration"  # Configuration errors
    DEPENDENCY = "dependency"  # External dependency failures
    VALIDATION = "validation"  # Input/output validation errors
    TIMEOUT = "timeout"  # Operation timeout errors
    INTEGRATION = "integration"  # Integration between components


class RecoveryAction(Enum):
    """Recovery actions that can be taken"""
    RETRY = "retry"  # Retry the operation
    FALLBACK = "fallback"  # Use fallback mechanism
    GRACEFUL_DEGRADE = "graceful_degrade"  # Continue with reduced functionality
    CIRCUIT_BREAK = "circuit_break"  # Open circuit breaker
    ESCALATE = "escalate"  # Escalate to human intervention
    IGNORE = "ignore"  # Log and continue
    RESTART = "restart"  # Restart component
    CACHE_FALLBACK = "cache_fallback"  # Use cached results


@dataclass
class ErrorContext:
    """Context information for error analysis"""
    timestamp: datetime
    function_name: str
    module_name: str
    input_data: Optional[Dict[str, Any]] = None
    system_state: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, float]] = None
    user_context: Optional[Dict[str, Any]] = None


@dataclass
class ErrorPattern:
    """Detected error pattern"""
    pattern_id: str
    category: ErrorCategory
    frequency: int
    first_occurrence: datetime
    last_occurrence: datetime
    error_messages: List[str] = field(default_factory=list)
    affected_functions: List[str] = field(default_factory=list)
    suggested_actions: List[RecoveryAction] = field(default_factory=list)


@dataclass
class ErrorReport:
    """Comprehensive error analysis report"""
    error_id: str
    severity: ErrorSeverity
    category: ErrorCategory
    error_message: str
    context: ErrorContext
    recovery_actions: List[RecoveryAction]
    pattern_matches: List[str]
    recommended_fixes: List[str]
    escalation_required: bool = False


class MLErrorHandler:
    """Advanced error handler for ML operations with pattern detection"""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize error handler with pattern detection"""
        self.lock = threading.Lock()
        self.error_patterns: Dict[str, ErrorPattern] = {}
        self.error_history: List[ErrorReport] = []
        self.max_history = 1000

        # Error classification rules
        self._init_classification_rules()

        # Recovery strategies
        self._init_recovery_strategies()

        # Database setup
        self._init_error_database(db_path)

        logger.info("🔍 ML Error Handler initialized with pattern detection")

    def _init_error_database(self, db_path: Optional[str] = None):
        """Initialize SQLite database for error tracking"""
        try:
            if db_path is None:
                db_dir = Path("10_DATA_STORAGE/ml_reliability")
                db_dir.mkdir(parents=True, exist_ok=True)
                db_path = str(db_dir / "error_tracking.db")

            self.db_path = db_path

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS error_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        error_id TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        category TEXT NOT NULL,
                        function_name TEXT NOT NULL,
                        error_message TEXT NOT NULL,
                        recovery_action TEXT,
                        pattern_id TEXT,
                        escalation_required INTEGER,
                        context_data TEXT,
                        stack_trace TEXT
                    )
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS error_patterns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pattern_id TEXT UNIQUE NOT NULL,
                        category TEXT NOT NULL,
                        frequency INTEGER NOT NULL,
                        first_occurrence TEXT NOT NULL,
                        last_occurrence TEXT NOT NULL,
                        error_signature TEXT NOT NULL,
                        suggested_actions TEXT
                    )
                """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_error_timestamp 
                    ON error_logs(timestamp)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_error_category 
                    ON error_logs(category)
                """)
        except Exception as e:
            logger.warning(f"Failed to initialize error database: {e}")
            self.db_path = None

    def _init_classification_rules(self):
        """Initialize error classification rules"""
        self.classification_rules = {
            # Model failure patterns
            'model_failure': [
                (re.compile(r'model.*not.*found', re.I), ErrorCategory.MODEL_FAILURE, ErrorSeverity.HIGH),
                (re.compile(r'prediction.*failed', re.I), ErrorCategory.MODEL_FAILURE, ErrorSeverity.MEDIUM),
                (re.compile(r'model.*corrupted', re.I), ErrorCategory.MODEL_FAILURE, ErrorSeverity.CRITICAL),
                (re.compile(r'ensemble.*error', re.I), ErrorCategory.MODEL_FAILURE, ErrorSeverity.HIGH),
            ],

            # Data quality patterns
            'data_quality': [
                (re.compile(r'invalid.*input', re.I), ErrorCategory.DATA_QUALITY, ErrorSeverity.MEDIUM),
                (re.compile(r'missing.*features?', re.I), ErrorCategory.DATA_QUALITY, ErrorSeverity.HIGH),
                (re.compile(r'data.*shape.*mismatch', re.I), ErrorCategory.DATA_QUALITY, ErrorSeverity.MEDIUM),
                (re.compile(r'nan.*values?', re.I), ErrorCategory.DATA_QUALITY, ErrorSeverity.LOW),
                (re.compile(r'empty.*dataset', re.I), ErrorCategory.DATA_QUALITY, ErrorSeverity.HIGH),
            ],

            # Performance patterns
            'performance': [
                (re.compile(r'timeout', re.I), ErrorCategory.PERFORMANCE, ErrorSeverity.MEDIUM),
                (re.compile(r'slow.*response', re.I), ErrorCategory.PERFORMANCE, ErrorSeverity.LOW),
                (re.compile(r'latency.*high', re.I), ErrorCategory.PERFORMANCE, ErrorSeverity.MEDIUM),
            ],

            # Resource patterns
            'resource': [
                (re.compile(r'out of memory', re.I), ErrorCategory.RESOURCE, ErrorSeverity.CRITICAL),
                (re.compile(r'memory.*error', re.I), ErrorCategory.RESOURCE, ErrorSeverity.HIGH),
                (re.compile(r'disk.*full', re.I), ErrorCategory.RESOURCE, ErrorSeverity.CRITICAL),
                (re.compile(r'cpu.*overload', re.I), ErrorCategory.RESOURCE, ErrorSeverity.HIGH),
            ],

            # Network patterns
            'network': [
                (re.compile(r'connection.*failed', re.I), ErrorCategory.NETWORK, ErrorSeverity.MEDIUM),
                (re.compile(r'network.*error', re.I), ErrorCategory.NETWORK, ErrorSeverity.MEDIUM),
                (re.compile(r'api.*unavailable', re.I), ErrorCategory.NETWORK, ErrorSeverity.HIGH),
            ],

            # Configuration patterns
            'configuration': [
                (re.compile(r'config.*not.*found', re.I), ErrorCategory.CONFIGURATION, ErrorSeverity.HIGH),
                (re.compile(r'invalid.*config', re.I), ErrorCategory.CONFIGURATION, ErrorSeverity.MEDIUM),
                (re.compile(r'missing.*parameter', re.I), ErrorCategory.CONFIGURATION, ErrorSeverity.MEDIUM),
            ]
        }

    def _init_recovery_strategies(self):
        """Initialize recovery strategies for different error types"""
        self.recovery_strategies = {
            ErrorCategory.MODEL_FAILURE: [
                RecoveryAction.FALLBACK,
                RecoveryAction.CIRCUIT_BREAK,
                RecoveryAction.ESCALATE
            ],
            ErrorCategory.DATA_QUALITY: [
                RecoveryAction.GRACEFUL_DEGRADE,
                RecoveryAction.CACHE_FALLBACK,
                RecoveryAction.RETRY
            ],
            ErrorCategory.PERFORMANCE: [
                RecoveryAction.RETRY,
                RecoveryAction.CIRCUIT_BREAK,
                RecoveryAction.GRACEFUL_DEGRADE
            ],
            ErrorCategory.RESOURCE: [
                RecoveryAction.RESTART,
                RecoveryAction.GRACEFUL_DEGRADE,
                RecoveryAction.ESCALATE
            ],
            ErrorCategory.NETWORK: [
                RecoveryAction.RETRY,
                RecoveryAction.FALLBACK,
                RecoveryAction.CACHE_FALLBACK
            ],
            ErrorCategory.CONFIGURATION: [
                RecoveryAction.ESCALATE,
                RecoveryAction.RESTART,
                RecoveryAction.FALLBACK
            ],
            ErrorCategory.DEPENDENCY: [
                RecoveryAction.RETRY,
                RecoveryAction.FALLBACK,
                RecoveryAction.CIRCUIT_BREAK
            ],
            ErrorCategory.VALIDATION: [
                RecoveryAction.GRACEFUL_DEGRADE,
                RecoveryAction.IGNORE,
                RecoveryAction.ESCALATE
            ],
            ErrorCategory.TIMEOUT: [
                RecoveryAction.RETRY,
                RecoveryAction.CIRCUIT_BREAK,
                RecoveryAction.FALLBACK
            ],
            ErrorCategory.INTEGRATION: [
                RecoveryAction.RETRY,
                RecoveryAction.FALLBACK,
                RecoveryAction.ESCALATE
            ]
        }

    def classify_error(self, exception: Exception, context: Optional[ErrorContext] = None) -> ErrorReport:
        """Classify error and generate comprehensive report"""
        error_message = str(exception)
        stack_trace = traceback.format_exc()

        # Classify error
        category, severity = self._classify_error_type(exception, error_message)

        # Generate unique error ID
        error_id = f"{category.value}_{hash(error_message) % 10000:04d}_{int(time.time())}"

        # Determine recovery actions
        recovery_actions = self.recovery_strategies.get(category, [RecoveryAction.ESCALATE])

        # Detect patterns
        pattern_matches = self._detect_patterns(error_message, category, context)

        # Generate recommended fixes
        recommended_fixes = self._generate_recommendations(category, error_message, pattern_matches)

        # Determine if escalation is required
        escalation_required = self._should_escalate(severity, category, pattern_matches)

        # Create error report
        error_report = ErrorReport(
            error_id=error_id,
            severity=severity,
            category=category,
            error_message=error_message,
            context=context or self._create_default_context(),
            recovery_actions=recovery_actions,
            pattern_matches=pattern_matches,
            recommended_fixes=recommended_fixes,
            escalation_required=escalation_required
        )

        # Store error
        self._store_error(error_report, stack_trace)

        # Update patterns
        self._update_patterns(error_report)

        return error_report

    def _classify_error_type(self, exception: Exception, error_message: str) -> Tuple[ErrorCategory, ErrorSeverity]:
        """Classify error type and severity"""
        # Check exception type first
        if isinstance(exception, MemoryError):
            return ErrorCategory.RESOURCE, ErrorSeverity.CRITICAL
        elif isinstance(exception, TimeoutError):
            return ErrorCategory.PERFORMANCE, ErrorSeverity.MEDIUM
        elif isinstance(exception, ConnectionError):
            return ErrorCategory.NETWORK, ErrorSeverity.MEDIUM
        elif isinstance(exception, (ValueError, TypeError)):
            return ErrorCategory.DATA_QUALITY, ErrorSeverity.MEDIUM
        elif isinstance(exception, FileNotFoundError):
            return ErrorCategory.CONFIGURATION, ErrorSeverity.HIGH
        elif isinstance(exception, ImportError):
            return ErrorCategory.DEPENDENCY, ErrorSeverity.HIGH

        # Check patterns in error message
        for rule_group in self.classification_rules.values():
            for pattern, category, severity in rule_group:
                if pattern.search(error_message):
                    return category, severity

        # Default classification
        return ErrorCategory.INTEGRATION, ErrorSeverity.MEDIUM

    def _detect_patterns(self, error_message: str, category: ErrorCategory,
                         context: Optional[ErrorContext]) -> List[str]:
        """Detect error patterns and return matching pattern IDs"""
        pattern_matches = []

        with self.lock:
            for pattern_id, pattern in self.error_patterns.items():
                if pattern.category == category:
                    # Simple pattern matching - could be enhanced with ML
                    for stored_message in pattern.error_messages[-5:]:  # Check last 5 messages
                        if self._messages_similar(error_message, stored_message):
                            pattern_matches.append(pattern_id)
                            break

        return pattern_matches

    def _messages_similar(self, msg1: str, msg2: str, threshold: float = 0.7) -> bool:
        """Check if two error messages are similar"""
        # Simple similarity check - could use more advanced NLP
        words1 = set(msg1.lower().split())
        words2 = set(msg2.lower().split())

        if not words1 or not words2:
            return False

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        similarity = len(intersection) / len(union)
        return similarity >= threshold

    def _generate_recommendations(self, category: ErrorCategory, error_message: str,
                                  pattern_matches: List[str]) -> List[str]:
        """Generate recommended fixes based on error analysis"""
        recommendations = []

        if category == ErrorCategory.MODEL_FAILURE:
            recommendations.extend([
                "Check if ML model files exist and are accessible",
                "Verify model configuration parameters",
                "Consider retraining or updating the model",
                "Implement fallback prediction mechanism"
            ])
        elif category == ErrorCategory.DATA_QUALITY:
            recommendations.extend([
                "Validate input data format and structure",
                "Check for missing or invalid feature values",
                "Implement data preprocessing and cleaning",
                "Add input validation checks"
            ])
        elif category == ErrorCategory.PERFORMANCE:
            recommendations.extend([
                "Optimize model inference time",
                "Implement caching for repeated predictions",
                "Consider model quantization or pruning",
                "Add performance monitoring"
            ])
        elif category == ErrorCategory.RESOURCE:
            recommendations.extend([
                "Monitor and optimize memory usage",
                "Consider batch processing for large datasets",
                "Implement resource cleanup mechanisms",
                "Scale infrastructure if needed"
            ])
        elif category == ErrorCategory.NETWORK:
            recommendations.extend([
                "Implement retry logic with exponential backoff",
                "Add network connectivity checks",
                "Consider offline mode or caching",
                "Monitor API rate limits"
            ])
        elif category == ErrorCategory.CONFIGURATION:
            recommendations.extend([
                "Verify configuration file paths and formats",
                "Add configuration validation at startup",
                "Implement configuration defaults",
                "Document required configuration parameters"
            ])

        # Add pattern-specific recommendations
        if pattern_matches:
            recommendations.append(f"This error matches {len(pattern_matches)} known patterns")
            recommendations.append("Consider implementing permanent fix for recurring issue")

        return recommendations

    def _should_escalate(self, severity: ErrorSeverity, category: ErrorCategory,
                         pattern_matches: List[str]) -> bool:
        """Determine if error should be escalated"""
        # Always escalate critical errors
        if severity == ErrorSeverity.CRITICAL:
            return True

        # Escalate if error is recurring (has pattern matches)
        if len(pattern_matches) > 0:
            with self.lock:
                for pattern_id in pattern_matches:
                    pattern = self.error_patterns.get(pattern_id)
                    if pattern and pattern.frequency > 5:  # Occurred more than 5 times
                        return True

        # Escalate high severity model failures
        if severity == ErrorSeverity.HIGH and category == ErrorCategory.MODEL_FAILURE:
            return True

        return False

    def _create_default_context(self) -> ErrorContext:
        """Create default error context"""
        return ErrorContext(
            timestamp=datetime.now(),
            function_name="unknown",
            module_name="unknown"
        )

    def _store_error(self, error_report: ErrorReport, stack_trace: str):
        """Store error in database"""
        if not self.db_path:
            return

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO error_logs 
                    (timestamp, error_id, severity, category, function_name, error_message,
                     recovery_action, pattern_id, escalation_required, context_data, stack_trace)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    error_report.context.timestamp.isoformat(),
                    error_report.error_id,
                    error_report.severity.value,
                    error_report.category.value,
                    error_report.context.function_name,
                    error_report.error_message,
                    ','.join([action.value for action in error_report.recovery_actions]),
                    ','.join(error_report.pattern_matches),
                    1 if error_report.escalation_required else 0,
                    json.dumps({
                        'input_data': error_report.context.input_data,
                        'system_state': error_report.context.system_state,
                        'performance_metrics': error_report.context.performance_metrics
                    }),
                    stack_trace
                ))
        except Exception as e:
            logger.warning(f"Failed to store error in database: {e}")

    def _update_patterns(self, error_report: ErrorReport):
        """Update error patterns based on new error"""
        with self.lock:
            # Create pattern signature
            pattern_signature = f"{error_report.category.value}_{hash(error_report.error_message[:100]) % 1000:03d}"

            if pattern_signature in self.error_patterns:
                # Update existing pattern
                pattern = self.error_patterns[pattern_signature]
                pattern.frequency += 1
                pattern.last_occurrence = error_report.context.timestamp
                pattern.error_messages.append(error_report.error_message)
                pattern.affected_functions.append(error_report.context.function_name)

                # Keep only recent messages
                if len(pattern.error_messages) > 10:
                    pattern.error_messages = pattern.error_messages[-10:]

            else:
                # Create new pattern
                pattern = ErrorPattern(
                    pattern_id=pattern_signature,
                    category=error_report.category,
                    frequency=1,
                    first_occurrence=error_report.context.timestamp,
                    last_occurrence=error_report.context.timestamp,
                    error_messages=[error_report.error_message],
                    affected_functions=[error_report.context.function_name],
                    suggested_actions=error_report.recovery_actions
                )
                self.error_patterns[pattern_signature] = pattern

            # Store pattern in database
            self._store_pattern(pattern)

    def _store_pattern(self, pattern: ErrorPattern):
        """Store error pattern in database"""
        if not self.db_path:
            return

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO error_patterns
                    (pattern_id, category, frequency, first_occurrence, last_occurrence,
                     error_signature, suggested_actions)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern.pattern_id,
                    pattern.category.value,
                    pattern.frequency,
                    pattern.first_occurrence.isoformat(),
                    pattern.last_occurrence.isoformat(),
                    pattern.error_messages[-1] if pattern.error_messages else "",
                    ','.join([action.value for action in pattern.suggested_actions])
                ))
        except Exception as e:
            logger.warning(f"Failed to store pattern in database: {e}")

    def get_error_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get error statistics for the specified time period"""
        if not self.db_path:
            return {'error': 'Database not available'}

        cutoff_time = datetime.now() - timedelta(hours=hours)

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get error counts by category
                category_stats = conn.execute("""
                    SELECT category, COUNT(*) as count
                    FROM error_logs 
                    WHERE timestamp > ?
                    GROUP BY category
                    ORDER BY count DESC
                """, (cutoff_time.isoformat(),)).fetchall()

                # Get error counts by severity
                severity_stats = conn.execute("""
                    SELECT severity, COUNT(*) as count
                    FROM error_logs 
                    WHERE timestamp > ?
                    GROUP BY severity
                    ORDER BY count DESC
                """, (cutoff_time.isoformat(),)).fetchall()

                # Get top error patterns
                pattern_stats = conn.execute("""
                    SELECT pattern_id, category, frequency
                    FROM error_patterns
                    ORDER BY frequency DESC
                    LIMIT 10
                """).fetchall()

                # Get escalation stats
                escalation_stats = conn.execute("""
                    SELECT 
                        SUM(escalation_required) as escalated,
                        COUNT(*) - SUM(escalation_required) as not_escalated
                    FROM error_logs 
                    WHERE timestamp > ?
                """, (cutoff_time.isoformat(),)).fetchone()

                return {
                    'time_period_hours': hours,
                    'category_distribution': dict(category_stats),
                    'severity_distribution': dict(severity_stats),
                    'top_error_patterns': [
                        {'pattern_id': p[0], 'category': p[1], 'frequency': p[2]}
                        for p in pattern_stats
                    ],
                    'escalation_stats': {
                        'escalated': escalation_stats[0] or 0,
                        'not_escalated': escalation_stats[1] or 0
                    }
                }

        except Exception as e:
            logger.error(f"Failed to get error statistics: {e}")
            return {'error': str(e)}

    def get_recovery_recommendations(self, category: ErrorCategory) -> List[str]:
        """Get recovery recommendations for error category"""
        base_recommendations = {
            action.value: self._get_action_description(action)
            for action in self.recovery_strategies.get(category, [])
        }

        return list(base_recommendations.values())

    def _get_action_description(self, action: RecoveryAction) -> str:
        """Get human-readable description of recovery action"""
        descriptions = {
            RecoveryAction.RETRY: "Retry the failed operation with exponential backoff",
            RecoveryAction.FALLBACK: "Use alternative method or cached results",
            RecoveryAction.GRACEFUL_DEGRADE: "Continue with reduced functionality",
            RecoveryAction.CIRCUIT_BREAK: "Temporarily disable failing component",
            RecoveryAction.ESCALATE: "Alert human operators for manual intervention",
            RecoveryAction.IGNORE: "Log error and continue normal operation",
            RecoveryAction.RESTART: "Restart the affected component or service",
            RecoveryAction.CACHE_FALLBACK: "Use previously cached successful results"
        }
        return descriptions.get(action, f"Execute {action.value} recovery strategy")


# Global error handler instance
ml_error_handler = MLErrorHandler()


def ml_error_handler_decorator(
        context_data: Optional[Dict[str, Any]] = None,
        recovery_strategy: Optional[List[RecoveryAction]] = None,
        escalate_on: Optional[List[ErrorSeverity]] = None
):
    """
    Decorator for ML functions with comprehensive error handling

    Usage:
        @ml_error_handler_decorator(
            context_data={'model_type': 'alpha'},
            escalate_on=[ErrorSeverity.CRITICAL]
        )
        def predict_model(data):
            return model.predict(data)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create context
            context = ErrorContext(
                timestamp=datetime.now(),
                function_name=func.__name__,
                module_name=func.__module__,
                input_data=context_data or {},
                system_state={'args_count': len(args), 'kwargs_keys': list(kwargs.keys())}
            )

            try:
                return func(*args, **kwargs)

            except Exception as e:
                # Classify and handle error
                error_report = ml_error_handler.classify_error(e, context)

                logger.error(f"🚨 Error in {func.__name__}: {error_report.error_message}")
                logger.info(f"📊 Error ID: {error_report.error_id}")
                logger.info(f"🔍 Category: {error_report.category.value}")
                logger.info(f"⚠️ Severity: {error_report.severity.value}")

                # Log recommendations
                if error_report.recommended_fixes:
                    logger.info("💡 Recommendations:")
                    for i, fix in enumerate(error_report.recommended_fixes[:3], 1):
                        logger.info(f"   {i}. {fix}")

                # Check if should escalate
                escalate_severities = escalate_on or [ErrorSeverity.CRITICAL]
                if error_report.severity in escalate_severities or error_report.escalation_required:
                    logger.critical(f"🚨 ESCALATION REQUIRED for error {error_report.error_id}")

                # Re-raise exception after logging
                raise e

        return wrapper

    return decorator


def error_recovery_strategy(category: ErrorCategory) -> Callable:
    """
    Get recovery strategy function for specific error category

    Usage:
        recovery_func = error_recovery_strategy(ErrorCategory.MODEL_FAILURE)
        recovery_func(exception, context)
    """

    def recovery_function(exception: Exception, context: Optional[Dict] = None) -> Any:
        """Execute recovery strategy"""
        error_report = ml_error_handler.classify_error(exception)

        for action in error_report.recovery_actions:
            if action == RecoveryAction.FALLBACK:
                logger.info("🔄 Executing fallback strategy")
                return {'status': 'fallback', 'confidence': 0.3, 'method': 'error_recovery'}
            elif action == RecoveryAction.CACHE_FALLBACK:
                logger.info("💾 Using cached fallback")
                return {'status': 'cached', 'confidence': 0.5, 'method': 'cache_fallback'}
            elif action == RecoveryAction.GRACEFUL_DEGRADE:
                logger.info("⚠️ Graceful degradation activated")
                return {'status': 'degraded', 'confidence': 0.2, 'method': 'degraded_mode'}

        # Default recovery
        logger.warning("🛡️ Using default recovery strategy")
        return {'status': 'default_recovery', 'confidence': 0.1, 'method': 'emergency'}

    return recovery_function


def get_error_analysis_report() -> str:
    """Generate comprehensive error analysis report"""
    stats = ml_error_handler.get_error_statistics(24)

    if 'error' in stats:
        return f"❌ Error retrieving statistics: {stats['error']}"

    report = f"""
{'=' * 70}
🔍 ML ERROR ANALYSIS REPORT (Last 24 Hours)
{'=' * 70}

📊 Error Distribution by Category:
"""

    for category, count in stats['category_distribution'].items():
        report += f"   {category.upper()}: {count}\n"

    report += f"\n⚠️ Error Distribution by Severity:\n"
    for severity, count in stats['severity_distribution'].items():
        report += f"   {severity.upper()}: {count}\n"

    report += f"\n🔄 Escalation Statistics:\n"
    escalation = stats['escalation_stats']
    report += f"   Escalated: {escalation['escalated']}\n"
    report += f"   Handled Automatically: {escalation['not_escalated']}\n"

    if stats['top_error_patterns']:
        report += f"\n🎯 Top Error Patterns:\n"
        for pattern in stats['top_error_patterns'][:5]:
            report += f"   {pattern['pattern_id']} ({pattern['category']}): {pattern['frequency']} occurrences\n"

    report += f"\n{'=' * 70}"
    return report


# Example usage and testing
if __name__ == "__main__":
    # Example: Test error handling with different error types

    @ml_error_handler_decorator(
        context_data={'model_type': 'test', 'component': 'prediction'},
        escalate_on=[ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]
    )
    def test_model_failure():
        """Test function that simulates model failure"""
        raise ValueError("Model prediction failed - invalid ensemble output")


    @ml_error_handler_decorator(
        context_data={'model_type': 'test', 'component': 'data_processing'}
    )
    def test_data_quality():
        """Test function that simulates data quality issues"""
        raise KeyError("Missing features in input data")


    @ml_error_handler_decorator()
    def test_resource_error():
        """Test function that simulates resource issues"""
        raise MemoryError("Out of memory during model training")


    print("🧪 Testing ML Error Handler")
    print("=" * 50)

    # Test different error types
    test_functions = [
        ("Model Failure", test_model_failure),
        ("Data Quality", test_data_quality),
        ("Resource Error", test_resource_error)
    ]

    for test_name, test_func in test_functions:
        print(f"\n🔬 Testing {test_name}:")
        try:
            test_func()
        except Exception as e:
            print(f"   ✓ Error handled: {type(e).__name__}")

        time.sleep(0.5)  # Small delay between tests

    # Generate and display analysis report
    time.sleep(1)  # Allow time for database writes
    print("\n" + get_error_analysis_report())