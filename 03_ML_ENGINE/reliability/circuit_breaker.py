# 03_ML_ENGINE/reliability/ml_circuit_breaker.py
"""
Advanced Circuit Breaker Implementation for ML Model Integration - Phase 2, Step 4
Enhanced error handling with circuit breakers, retry logic, and fallback mechanisms

Location: #03_ML_ENGINE/reliability/ml_circuit_breaker.py

This module provides:
- Circuit breaker patterns for ML model failure protection
- Exponential backoff retry logic for transient failures
- Comprehensive fallback mechanisms for various ML model failures
- Health monitoring and recovery strategies specifically for ML operations
- Failure rate tracking and automatic recovery
- Model-specific error handling and classification
- Performance optimization with sub-20ms targets
"""

import time
import logging
import threading
import functools
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random
import json
import sqlite3
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states for ML models"""
    CLOSED = "closed"  # Normal operation - models working
    OPEN = "open"  # Circuit is open - models failing, using fallbacks
    HALF_OPEN = "half_open"  # Testing if models have recovered


class MLModelType(Enum):
    """Types of ML models that can be protected"""
    ALPHA_MODEL = "alpha_model"
    LSTM_MODEL = "lstm_model"
    ENSEMBLE = "ensemble"
    TECHNICAL_ANALYSIS = "technical_analysis"
    SENTIMENT = "sentiment"


class ErrorType(Enum):
    """Classification of ML model errors"""
    TRANSIENT = "transient"  # Network issues, temporary unavailability
    PERMANENT = "permanent"  # Model corruption, invalid configuration
    RESOURCE = "resource"  # Memory issues, CPU overload
    DATA = "data"  # Invalid input data, missing features
    TIMEOUT = "timeout"  # Model taking too long to respond
    PREDICTION = "prediction"  # Model returning invalid predictions


@dataclass
class MLCircuitBreakerConfig:
    """Configuration for ML-specific circuit breaker behavior"""
    failure_threshold: int = 5  # Number of failures before opening
    success_threshold: int = 3  # Number of successes to close from half-open
    timeout_seconds: int = 30  # Time to wait before trying half-open
    request_volume_threshold: int = 10  # Minimum requests before considering failure rate
    failure_rate_threshold: float = 0.6  # Failure rate threshold (60%)
    monitoring_window_seconds: int = 300  # Time window for monitoring (5 minutes)

    # ML-specific configurations
    model_timeout_ms: int = 15000  # Model prediction timeout (15 seconds)
    max_prediction_time_ms: int = 20  # Target max prediction time (20ms)
    memory_limit_mb: int = 500  # Memory limit for ML operations
    fallback_confidence_threshold: float = 0.3  # Minimum confidence for fallback predictions

    # Performance thresholds
    latency_p95_threshold_ms: float = 50.0  # 95th percentile latency threshold
    latency_p99_threshold_ms: float = 100.0  # 99th percentile latency threshold
    error_spike_threshold: float = 0.8  # Error rate spike detection


@dataclass
class MLRequestResult:
    """Result of a single ML model request"""
    success: bool
    timestamp: datetime
    execution_time_ms: float
    model_type: MLModelType
    error_type: Optional[ErrorType] = None
    error_message: Optional[str] = None
    prediction_confidence: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    retry_attempt: int = 0
    fallback_used: bool = False


@dataclass
class MLCircuitMetrics:
    """Comprehensive metrics for ML circuit breaker performance"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    circuit_opens: int = 0
    circuit_closes: int = 0
    fallback_executions: int = 0
    avg_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    failure_rate: float = 0.0
    avg_confidence: float = 0.0
    memory_usage_mb: float = 0.0
    recent_requests: List[MLRequestResult] = field(default_factory=list)


class MLRetryStrategy:
    """Advanced retry strategy for ML operations with error-specific logic"""

    def __init__(self, max_attempts: int = 3, base_delay: float = 0.5,
                 max_delay: float = 30.0, jitter_factor: float = 0.1):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter_factor = jitter_factor

        # Error-specific retry policies
        self.error_retry_policies = {
            ErrorType.TRANSIENT: True,  # Always retry transient errors
            ErrorType.RESOURCE: True,  # Retry resource issues with backoff
            ErrorType.TIMEOUT: True,  # Retry timeouts with longer delay
            ErrorType.DATA: False,  # Don't retry data errors
            ErrorType.PERMANENT: False,  # Don't retry permanent failures
            ErrorType.PREDICTION: True  # Retry prediction errors once
        }

    def should_retry(self, attempt: int, error_type: ErrorType, exception: Exception) -> bool:
        """Determine if we should retry based on error type and attempt count"""
        if attempt >= self.max_attempts:
            return False

        return self.error_retry_policies.get(error_type, False)

    def get_delay(self, attempt: int, error_type: ErrorType) -> float:
        """Calculate delay with error-type specific adjustments"""
        base_delay = self.base_delay

        # Adjust base delay based on error type
        if error_type == ErrorType.RESOURCE:
            base_delay *= 2  # Longer delay for resource issues
        elif error_type == ErrorType.TIMEOUT:
            base_delay *= 3  # Even longer for timeouts

        # Exponential backoff with jitter
        delay = min(base_delay * (2 ** (attempt - 1)), self.max_delay)
        jitter = delay * self.jitter_factor * random.random()

        return delay + jitter


class MLFallbackStrategy:
    """Intelligent fallback mechanisms for ML model failures"""

    def __init__(self, model_type: MLModelType):
        self.model_type = model_type
        self.successful_predictions = []
        self.max_stored_predictions = 100
        self.lock = threading.Lock()

    def store_successful_prediction(self, input_data: Dict, prediction: Any, confidence: float):
        """Store successful prediction for fallback use"""
        with self.lock:
            self.successful_predictions.append({
                'timestamp': datetime.now(),
                'input_data': input_data,
                'prediction': prediction,
                'confidence': confidence
            })

            # Keep only recent predictions
            if len(self.successful_predictions) > self.max_stored_predictions:
                self.successful_predictions.pop(0)

    def get_fallback_prediction(self, input_data: Dict) -> Dict[str, Any]:
        """Generate fallback prediction based on historical data"""
        with self.lock:
            if not self.successful_predictions:
                return self._get_default_fallback()

            # Use most recent similar prediction or weighted average
            recent_predictions = [p for p in self.successful_predictions
                                  if (datetime.now() - p['timestamp']).seconds < 3600]

            if recent_predictions:
                # Return weighted average of recent predictions
                weights = [p['confidence'] for p in recent_predictions]
                total_weight = sum(weights)

                if self.model_type == MLModelType.ALPHA_MODEL:
                    # For alpha model, average the probability of profit
                    avg_pop = sum(p['prediction'].get('ensemble_pop', 0.5) * p['confidence']
                                  for p in recent_predictions) / total_weight
                    return {
                        'ensemble_pop': avg_pop,
                        'confidence': 'LOW',
                        'method': 'FALLBACK',
                        'fallback_reason': 'circuit_breaker_open'
                    }
                elif self.model_type == MLModelType.LSTM_MODEL:
                    # For LSTM, use trend continuation
                    return {
                        'predicted_direction': 'HOLD',
                        'confidence': 0.3,
                        'method': 'FALLBACK',
                        'fallback_reason': 'circuit_breaker_open'
                    }

            return self._get_default_fallback()

    def _get_default_fallback(self) -> Dict[str, Any]:
        """Get default conservative fallback prediction"""
        if self.model_type == MLModelType.ALPHA_MODEL:
            return {
                'ensemble_pop': 0.5,  # Neutral probability
                'confidence': 'LOW',
                'method': 'DEFAULT_FALLBACK',
                'fallback_reason': 'no_historical_data'
            }
        elif self.model_type == MLModelType.LSTM_MODEL:
            return {
                'predicted_direction': 'HOLD',
                'confidence': 0.2,
                'method': 'DEFAULT_FALLBACK',
                'fallback_reason': 'no_historical_data'
            }
        else:
            return {
                'signal': 'HOLD',
                'confidence': 0.1,
                'method': 'DEFAULT_FALLBACK',
                'fallback_reason': 'unknown_model_type'
            }


class MLCircuitBreaker:
    """Advanced circuit breaker specifically designed for ML model operations"""

    def __init__(self, name: str, model_type: MLModelType,
                 config: Optional[MLCircuitBreakerConfig] = None):
        self.name = name
        self.model_type = model_type
        self.config = config or MLCircuitBreakerConfig()

        # Circuit breaker state
        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.last_failure_time = datetime.now()
        self.lock = threading.Lock()

        # Metrics and monitoring
        self.metrics = MLCircuitMetrics()
        self.response_times = []
        self.max_response_times = 1000  # Keep last 1000 response times

        # Fallback strategy
        self.fallback = MLFallbackStrategy(model_type)

        # Database for persistence
        self._init_metrics_db()

        logger.info(f"🔧 ML Circuit Breaker '{name}' initialized for {model_type.value}")

    def _init_metrics_db(self):
        """Initialize SQLite database for metrics persistence"""
        try:
            db_dir = Path("10_DATA_STORAGE/ml_reliability")
            db_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str(db_dir / f"circuit_breaker_{self.name}.db")

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS circuit_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        state TEXT NOT NULL,
                        execution_time_ms REAL,
                        success INTEGER,
                        error_type TEXT,
                        error_message TEXT,
                        fallback_used INTEGER,
                        confidence REAL
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_timestamp 
                    ON circuit_metrics(timestamp)
                """)
        except Exception as e:
            logger.warning(f"Failed to initialize metrics database: {e}")
            self.db_path = None

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        with self.lock:
            # Check if circuit is open
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    logger.info(f"🔄 Circuit {self.name} entering HALF-OPEN state")
                else:
                    logger.warning(f"⚡ Circuit {self.name} OPEN - using fallback")
                    return self._execute_fallback(*args, **kwargs)

        # Attempt to execute the function
        start_time = time.time()
        try:
            # Execute with timeout
            result = self._execute_with_timeout(func, *args, **kwargs)
            execution_time_ms = (time.time() - start_time) * 1000

            # Record success
            self._record_success(execution_time_ms, result)
            return result

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            error_type = self._classify_error(e)

            # Record failure
            self._record_failure(execution_time_ms, error_type, str(e))

            # Decide whether to use fallback or re-raise
            if self.state == CircuitState.OPEN:
                return self._execute_fallback(*args, **kwargs)
            else:
                raise e

    def _execute_with_timeout(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with timeout protection"""
        timeout_seconds = self.config.model_timeout_ms / 1000.0

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=timeout_seconds)
            except asyncio.TimeoutError:
                raise TimeoutError(f"Model prediction exceeded timeout of {self.config.model_timeout_ms}ms")

    def _classify_error(self, exception: Exception) -> ErrorType:
        """Classify error type for appropriate handling"""
        error_str = str(exception).lower()

        if isinstance(exception, TimeoutError):
            return ErrorType.TIMEOUT
        elif isinstance(exception, MemoryError):
            return ErrorType.RESOURCE
        elif isinstance(exception, (ValueError, TypeError, KeyError)):
            return ErrorType.DATA
        elif 'connection' in error_str or 'network' in error_str:
            return ErrorType.TRANSIENT
        elif 'prediction' in error_str or 'confidence' in error_str:
            return ErrorType.PREDICTION
        else:
            return ErrorType.PERMANENT

    def _record_success(self, execution_time_ms: float, result: Any):
        """Record successful request"""
        self.metrics.total_requests += 1
        self.metrics.successful_requests += 1
        self.consecutive_successes += 1
        self.consecutive_failures = 0

        # Update response times
        self.response_times.append(execution_time_ms)
        if len(self.response_times) > self.max_response_times:
            self.response_times.pop(0)

        # Update performance metrics
        self._update_performance_metrics()

        # Store successful prediction for fallback
        if hasattr(result, 'get') and result.get('confidence', 0) > 0:
            try:
                input_data = {'timestamp': datetime.now().isoformat()}
                self.fallback.store_successful_prediction(
                    self.model_type, input_data, result, result.get('confidence', 0)
                )
            except Exception as e:
                logger.warning(f"Failed to store successful prediction: {e}")

        # Check if circuit should close
        if self.state == CircuitState.HALF_OPEN:
            if self.consecutive_successes >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.metrics.circuit_closes += 1
                logger.info(f"✅ Circuit {self.name} closed - service recovered")

    def _record_failure(self, execution_time_ms: float, error_type: ErrorType, error_msg: str):
        """Record failed request"""
        self.metrics.total_requests += 1
        self.metrics.failed_requests += 1
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self.last_failure_time = datetime.now()

        # Update metrics
        self._update_performance_metrics()

        # Check if circuit should open
        if (self.consecutive_failures >= self.config.failure_threshold and
                self.metrics.total_requests >= self.config.request_volume_threshold):

            if self.state != CircuitState.OPEN:
                self.state = CircuitState.OPEN
                self.metrics.circuit_opens += 1
                logger.error(f"🚨 Circuit {self.name} opened after {self.consecutive_failures} failures")

        # Store metrics in database
        self._store_metric_in_db(execution_time_ms, False, error_type, error_msg, False)

    def _update_performance_metrics(self):
        """Update computed performance metrics"""
        if self.metrics.total_requests > 0:
            self.metrics.failure_rate = self.metrics.failed_requests / self.metrics.total_requests

        if self.response_times:
            self.metrics.avg_response_time_ms = np.mean(self.response_times)
            self.metrics.p95_response_time_ms = np.percentile(self.response_times, 95)
            self.metrics.p99_response_time_ms = np.percentile(self.response_times, 99)

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        time_since_failure = datetime.now() - self.last_failure_time
        return time_since_failure.total_seconds() > self.config.timeout_seconds

    def _execute_fallback(self, *args, **kwargs) -> Any:
        """Execute fallback strategy"""
        self.metrics.fallback_executions += 1

        try:
            # Create input data from args/kwargs for fallback
            input_data = {
                'args_count': len(args),
                'kwargs_keys': list(kwargs.keys()) if kwargs else [],
                'timestamp': datetime.now().isoformat()
            }

            fallback_result = self.fallback.get_fallback_prediction(input_data)
            self._store_metric_in_db(0.0, True, None, None, True)

            logger.info(f"🔄 Fallback executed for {self.name}: {fallback_result.get('method', 'unknown')}")
            return fallback_result

        except Exception as e:
            logger.error(f"❌ Fallback failed for {self.name}: {e}")
            # Return ultra-conservative fallback
            return {
                'signal': 'HOLD',
                'confidence': 0.1,
                'method': 'EMERGENCY_FALLBACK',
                'error': str(e)
            }

    def _store_metric_in_db(self, execution_time_ms: float, success: bool,
                            error_type: Optional[ErrorType], error_message: Optional[str],
                            fallback_used: bool):
        """Store metric in SQLite database"""
        if not self.db_path:
            return

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO circuit_metrics 
                    (timestamp, state, execution_time_ms, success, error_type, error_message, fallback_used, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    self.state.value,
                    execution_time_ms,
                    1 if success else 0,
                    error_type.value if error_type else None,
                    error_message,
                    1 if fallback_used else 0,
                    None  # Confidence - could be extracted from result
                ))
        except Exception as e:
            logger.warning(f"Failed to store metric in database: {e}")

    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        with self.lock:
            # Calculate health score (0-100)
            health_score = 100.0

            # Reduce score based on failure rate
            if self.metrics.failure_rate > 0:
                health_score -= min(self.metrics.failure_rate * 50, 40)

            # Reduce score based on response times
            if self.metrics.p95_response_time_ms > self.config.max_prediction_time_ms:
                health_score -= min((self.metrics.p95_response_time_ms - self.config.max_prediction_time_ms) / 10, 30)

            # Reduce score based on circuit state
            if self.state == CircuitState.OPEN:
                health_score -= 30
            elif self.state == CircuitState.HALF_OPEN:
                health_score -= 15

            # Ensure health score is between 0 and 100
            health_score = max(0.0, min(100.0, health_score))

            return {
                'name': self.name,
                'model_type': self.model_type.value,
                'state': self.state.value,
                'health_score': round(health_score, 1),
                'metrics': {
                    'total_requests': self.metrics.total_requests,
                    'success_rate': round((1 - self.metrics.failure_rate) * 100, 1),
                    'avg_response_time_ms': round(self.metrics.avg_response_time_ms, 2),
                    'p95_response_time_ms': round(self.metrics.p95_response_time_ms, 2),
                    'circuit_opens': self.metrics.circuit_opens,
                    'fallback_executions': self.metrics.fallback_executions
                },
                'thresholds': {
                    'failure_threshold': self.config.failure_threshold,
                    'max_prediction_time_ms': self.config.max_prediction_time_ms
                }
            }


class MLCircuitBreakerRegistry:
    """Global registry for managing ML circuit breakers"""

    def __init__(self):
        self.breakers: Dict[str, MLCircuitBreaker] = {}
        self.lock = threading.Lock()

    def create(self, name: str, model_type: MLModelType,
               config: Optional[MLCircuitBreakerConfig] = None) -> MLCircuitBreaker:
        """Create and register a new circuit breaker"""
        with self.lock:
            if name in self.breakers:
                logger.warning(f"Circuit breaker '{name}' already exists, returning existing")
                return self.breakers[name]

            breaker = MLCircuitBreaker(name, model_type, config)
            self.breakers[name] = breaker
            logger.info(f"🔧 Registered ML circuit breaker: {name}")
            return breaker

    def get(self, name: str) -> Optional[MLCircuitBreaker]:
        """Get circuit breaker by name"""
        with self.lock:
            return self.breakers.get(name)

    def remove(self, name: str) -> bool:
        """Remove circuit breaker"""
        with self.lock:
            if name in self.breakers:
                del self.breakers[name]
                logger.info(f"🗑️ Removed ML circuit breaker: {name}")
                return True
            return False

    def list_all(self) -> List[str]:
        """List all registered circuit breakers"""
        with self.lock:
            return list(self.breakers.keys())

    def get_all_health_reports(self) -> Dict[str, Dict[str, Any]]:
        """Get health reports for all circuit breakers"""
        with self.lock:
            return {name: breaker.get_health_report()
                    for name, breaker in self.breakers.items()}

    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary"""
        with self.lock:
            if not self.breakers:
                return {'status': 'no_breakers', 'message': 'No ML circuit breakers registered'}

        health_reports = self.get_all_health_reports()
        health_scores = [report['health_score'] for report in health_reports.values()]

        avg_health = np.mean(health_scores)
        min_health = np.min(health_scores)

        # Count breakers by state
        states = [report['state'] for report in health_reports.values()]
        state_counts = {state: states.count(state) for state in set(states)}

        # Determine overall status
        if min_health < 50:
            status = 'critical'
        elif min_health < 80:
            status = 'degraded'
        else:
            status = 'healthy'

        return {
            'status': status,
            'health_score': avg_health,
            'min_health_score': min_health,
            'total_breakers': len(self.breakers),
            'state_distribution': state_counts,
            'breakers_critical': sum(1 for score in health_scores if score < 50),
            'breakers_degraded': sum(1 for score in health_scores if 50 <= score < 80),
            'breakers_healthy': sum(1 for score in health_scores if score >= 80)
        }


# Global registry
ml_circuit_registry = MLCircuitBreakerRegistry()


def ml_circuit_breaker(name: str, model_type: MLModelType,
                       config: Optional[MLCircuitBreakerConfig] = None,
                       retry_strategy: Optional[MLRetryStrategy] = None):
    """
    Decorator for ML model functions with circuit breaker protection

    Usage:
        @ml_circuit_breaker('alpha_model', MLModelType.ALPHA_MODEL)
        def predict_alpha(market_data):
            return model.predict(market_data)
    """

    def decorator(func: Callable) -> Callable:
        # Get or create circuit breaker
        breaker = ml_circuit_registry.get(name)
        if breaker is None:
            breaker = ml_circuit_registry.create(name, model_type, config)

        retry = retry_strategy or MLRetryStrategy()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(retry.max_attempts):
                try:
                    return breaker.call(func, *args, **kwargs)

                except Exception as e:
                    last_exception = e
                    error_type = breaker._classify_error(e)

                    if not retry.should_retry(attempt + 1, error_type, e):
                        break

                    delay = retry.get_delay(attempt + 1, error_type)
                    if delay > 0:
                        logger.warning(f"🔄 Retrying {name} after {delay:.2f}s (attempt {attempt + 1})")
                        time.sleep(delay)

                    if breaker.state == CircuitState.OPEN:
                        break

            # If we get here, all retries failed
            logger.error(f"❌ All retries failed for {name}, using emergency fallback")
            return breaker._execute_fallback(*args, **kwargs)

        return wrapper

    return decorator


def get_ml_system_health_dashboard() -> str:
    """Get formatted system health dashboard for ML components"""
    summary = ml_circuit_registry.get_system_health_summary()

    if summary['status'] == 'no_breakers':
        return "⚠️ No ML circuit breakers registered"

    # Status icons
    status_icons = {
        'healthy': '✅',
        'degraded': '⚠️',
        'critical': '🚨'
    }

    icon = status_icons.get(summary['status'], '❓')

    dashboard = f"""
{'=' * 70}
🧠 ML SYSTEM HEALTH DASHBOARD
{'=' * 70}
{icon} Overall Status: {summary['status'].upper()} ({summary['health_score']:.1f}/100)

📊 Circuit Breaker Summary:
   Total Breakers: {summary['total_breakers']}
   🟢 Healthy: {summary['breakers_healthy']} (≥80 health)
   🟡 Degraded: {summary['breakers_degraded']} (50-79 health)
   🔴 Critical: {summary['breakers_critical']} (<50 health)

🔄 Circuit States:
"""

    for state, count in summary['state_distribution'].items():
        dashboard += f"   {state.upper()}: {count}\n"

    # Individual breaker status
    health_reports = ml_circuit_registry.get_all_health_reports()
    if health_reports:
        dashboard += f"\n🔧 Individual Breaker Status:\n"
        for name, report in health_reports.items():
            health_score = report['health_score']
            if health_score >= 80:
                status_icon = '🟢'
            elif health_score >= 50:
                status_icon = '🟡'
            else:
                status_icon = '🔴'

            metrics = report['metrics']
            dashboard += f"   {status_icon} {name} ({report['model_type']}): "
            dashboard += f"{health_score:.1f}/100 | "
            dashboard += f"Success: {metrics['success_rate']:.1f}% | "
            dashboard += f"Avg: {metrics['avg_response_time_ms']:.1f}ms | "
            dashboard += f"Opens: {metrics['circuit_opens']}\n"

    dashboard += f"\n{'=' * 70}"
    return dashboard


# Example usage and testing functions
if __name__ == "__main__":
    # Example: Create circuit breakers for different ML models

    # Alpha model circuit breaker
    alpha_config = MLCircuitBreakerConfig(
        failure_threshold=3,
        max_prediction_time_ms=15
    )


    @ml_circuit_breaker('alpha_model', MLModelType.ALPHA_MODEL, alpha_config)
    def predict_alpha(market_data):
        """Example alpha model prediction function"""
        time.sleep(0.005)  # Simulate ML processing
        if random.random() < 0.1:  # 10% failure rate for testing
            raise ValueError("Model prediction failed")
        return {'ensemble_pop': 0.65, 'confidence': 'HIGH', 'method': 'ML'}


    # LSTM model circuit breaker
    @ml_circuit_breaker('lstm_model', MLModelType.LSTM_MODEL)
    def predict_lstm(time_series_data):
        """Example LSTM prediction function"""
        time.sleep(0.008)  # Simulate LSTM processing
        if random.random() < 0.05:  # 5% failure rate
            raise TimeoutError("LSTM prediction timeout")
        return {'predicted_direction': 'BUY', 'confidence': 0.8}


    # Test the circuit breakers
    print("🧪 Testing ML Circuit Breakers")
    print("=" * 50)

    # Test alpha model
    for i in range(20):
        try:
            result = predict_alpha({'symbol': 'AAPL', 'price': 150.0})
            print(f"Alpha prediction {i + 1}: {result['ensemble_pop']:.2f}")
        except Exception as e:
            print(f"Alpha prediction {i + 1} failed: {e}")

        time.sleep(0.1)

    # Test LSTM model
    for i in range(15):
        try:
            result = predict_lstm([100, 101, 102, 103])
            print(f"LSTM prediction {i + 1}: {result['predicted_direction']}")
        except Exception as e:
            print(f"LSTM prediction {i + 1} failed: {e}")

        time.sleep(0.1)

    # Display health dashboard
    print("\n" + get_ml_system_health_dashboard())