# 03_ML_ENGINE/integration/unified_model_integration.py
"""
Unified Model Integration Framework with Circuit Breaker Orchestration
MarketPulse Phase 2, Step 5 - Complete Model Integration

This module unifies all ML models with centralized circuit breaker management:
- Real Alpha Model Integration with profitability prediction
- Real LSTM Model Integration with time-series forecasting
- Circuit breaker orchestration across all models
- Performance optimization and tier-based fallback
- Comprehensive health monitoring and metrics

Location: #03_ML_ENGINE/integration/unified_model_integration.py
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import sqlite3
import json
import warnings
import time
from dataclasses import dataclass
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup paths
current_dir = Path(__file__).parent
root_dir = current_dir.parent.parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir.parent / "models"))
sys.path.append(str(current_dir.parent / "reliability"))
sys.path.append(str(current_dir.parent / "optimization"))

# Import reliability components
try:
    from ml_circuit_breaker import ml_circuit_breaker, CircuitBreakerRegistry
    from error_handler import ErrorHandler, ErrorClassifier
    from latency_optimizer import performance_monitor, LatencyOptimizer

    RELIABILITY_AVAILABLE = True
    logger.info("✅ Reliability components imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Reliability components not available: {e}")
    RELIABILITY_AVAILABLE = False


    # Fallback decorators
    def ml_circuit_breaker(**kwargs):
        def decorator(func):
            return func

        return decorator


    def performance_monitor(**kwargs):
        def decorator(func):
            return func

        return decorator

# Import model components
try:
    from real_alpha_model_integration import RealAlphaModelCore, create_real_alpha_model
    from real_lstm_model_integration import RealLSTMModelCore, create_real_lstm_model

    REAL_MODELS_AVAILABLE = True
    logger.info("✅ Real model components imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Real model components not available: {e}")
    REAL_MODELS_AVAILABLE = False


class SystemTier(Enum):
    """System-wide performance tiers"""
    PREMIUM = "premium"  # Sub-10ms, all models active
    STANDARD = "standard"  # Sub-20ms, essential models active
    ECONOMIC = "economic"  # Sub-50ms, basic models only
    FALLBACK = "fallback"  # Sub-10ms, heuristics only


class ModelHealth(Enum):
    """Model health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


@dataclass
class ModelStatus:
    """Model status tracking"""
    name: str
    health: ModelHealth
    success_rate: float
    avg_latency: float
    last_prediction: Optional[datetime]
    error_count: int
    circuit_breaker_state: str


class UnifiedModelIntegration:
    """
    Unified Model Integration Framework

    Orchestrates multiple ML models with:
    - Centralized circuit breaker management
    - Performance optimization across all models
    - Tier-based resource allocation
    - Health monitoring and automatic recovery
    - Intelligent fallback strategies
    """

    def __init__(self, db_path: str = "marketpulse_production.db", tier: SystemTier = SystemTier.STANDARD):
        """
        Initialize Unified Model Integration Framework

        Args:
            db_path: Path to SQLite database
            tier: System performance tier
        """
        self.db_path = db_path
        self.tier = tier
        self.model_version = "v2.0.0-unified"

        # Model instances
        self.alpha_model: Optional[RealAlphaModelCore] = None
        self.lstm_model: Optional[RealLSTMModelCore] = None

        # Model status tracking
        self.model_status: Dict[str, ModelStatus] = {}

        # System performance tracking
        self.system_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'average_latency': 0.0,
            'alpha_model_usage': 0,
            'lstm_model_usage': 0,
            'fallback_usage': 0,
            'concurrent_requests': 0,
            'peak_concurrent': 0
        }

        # Thread safety
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ModelExecution")

        # Configuration based on tier
        self.config = self._get_tier_config(tier)

        # Initialize components
        self._initialize_database()
        self._initialize_models()

        if RELIABILITY_AVAILABLE:
            self.error_handler = ErrorHandler()
            self.latency_optimizer = LatencyOptimizer()
            logger.info("✅ Reliability components initialized")

        logger.info(f"✅ UnifiedModelIntegration initialized with {tier.value} tier")

    def _get_tier_config(self, tier: SystemTier) -> Dict:
        """Get configuration based on system tier"""
        configs = {
            SystemTier.PREMIUM: {
                'max_latency_ms': 10.0,
                'enable_alpha_model': True,
                'enable_lstm_model': True,
                'enable_concurrent': True,
                'cache_ttl': 180,  # 3 minutes
                'max_concurrent': 8
            },
            SystemTier.STANDARD: {
                'max_latency_ms': 20.0,
                'enable_alpha_model': True,
                'enable_lstm_model': True,
                'enable_concurrent': True,
                'cache_ttl': 300,  # 5 minutes
                'max_concurrent': 4
            },
            SystemTier.ECONOMIC: {
                'max_latency_ms': 50.0,
                'enable_alpha_model': True,
                'enable_lstm_model': False,
                'enable_concurrent': False,
                'cache_ttl': 600,  # 10 minutes
                'max_concurrent': 2
            },
            SystemTier.FALLBACK: {
                'max_latency_ms': 10.0,
                'enable_alpha_model': False,
                'enable_lstm_model': False,
                'enable_concurrent': False,
                'cache_ttl': 60,  # 1 minute
                'max_concurrent': 1
            }
        }
        return configs[tier]

    def _initialize_database(self):
        """Initialize SQLite database with unified model tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Create unified_predictions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS unified_predictions (
                        id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        request_type TEXT NOT NULL,
                        alpha_prediction TEXT,
                        lstm_prediction TEXT,
                        unified_result TEXT NOT NULL,
                        system_tier TEXT NOT NULL,
                        total_latency_ms REAL NOT NULL,
                        models_used TEXT NOT NULL,
                        circuit_breaker_states TEXT,
                        performance_metrics TEXT
                    )
                ''')

                # Create system_health table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_health (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        system_tier TEXT NOT NULL,
                        overall_health_score REAL NOT NULL,
                        alpha_model_health TEXT,
                        lstm_model_health TEXT,
                        performance_stats TEXT NOT NULL,
                        active_issues TEXT
                    )
                ''')

                conn.commit()
                logger.info("✅ Unified model database tables initialized successfully")

        except Exception as e:
            logger.error(f"❌ Unified database initialization failed: {e}")
            raise

    def _initialize_models(self):
        """Initialize model instances based on configuration"""
        try:
            if not REAL_MODELS_AVAILABLE:
                logger.warning("⚠️ Real model components not available - using fallback mode")
                return

            # Initialize Alpha model if enabled
            if self.config['enable_alpha_model']:
                try:
                    alpha_tier = "premium" if self.tier == SystemTier.PREMIUM else "standard"
                    self.alpha_model = create_real_alpha_model(tier=alpha_tier, db_path=self.db_path)

                    self.model_status['alpha'] = ModelStatus(
                        name='alpha',
                        health=ModelHealth.HEALTHY,
                        success_rate=100.0,
                        avg_latency=0.0,
                        last_prediction=None,
                        error_count=0,
                        circuit_breaker_state='CLOSED'
                    )

                    logger.info("✅ Alpha model initialized successfully")

                except Exception as e:
                    logger.error(f"❌ Alpha model initialization failed: {e}")
                    self.model_status['alpha'] = ModelStatus(
                        name='alpha',
                        health=ModelHealth.UNAVAILABLE,
                        success_rate=0.0,
                        avg_latency=0.0,
                        last_prediction=None,
                        error_count=1,
                        circuit_breaker_state='OPEN'
                    )

            # Initialize LSTM model if enabled
            if self.config['enable_lstm_model']:
                try:
                    lstm_tier = "premium" if self.tier == SystemTier.PREMIUM else "standard"
                    horizons = [5, 15] if self.tier == SystemTier.PREMIUM else [5]
                    self.lstm_model = create_real_lstm_model(
                        tier=lstm_tier,
                        db_path=self.db_path,
                        horizons=horizons
                    )

                    self.model_status['lstm'] = ModelStatus(
                        name='lstm',
                        health=ModelHealth.HEALTHY,
                        success_rate=100.0,
                        avg_latency=0.0,
                        last_prediction=None,
                        error_count=0,
                        circuit_breaker_state='CLOSED'
                    )

                    logger.info("✅ LSTM model initialized successfully")

                except Exception as e:
                    logger.error(f"❌ LSTM model initialization failed: {e}")
                    self.model_status['lstm'] = ModelStatus(
                        name='lstm',
                        health=ModelHealth.UNAVAILABLE,
                        success_rate=0.0,
                        avg_latency=0.0,
                        last_prediction=None,
                        error_count=1,
                        circuit_breaker_state='OPEN'
                    )

        except Exception as e:
            logger.error(f"❌ Model initialization failed: {e}")

    @ml_circuit_breaker(
        failure_threshold=5,
        recovery_timeout=60,
        timeout=20.0
    )
    @performance_monitor(
        target_latency_ms=20.0,
        enable_caching=True,
        cache_ttl_seconds=300
    )
    def get_enhanced_prediction(self, symbol: str, market_data: pd.DataFrame,
                                include_alpha: bool = True, include_lstm: bool = True,
                                lstm_horizon: int = 5) -> Dict[str, Any]:
        """
        Get enhanced prediction using multiple models with circuit breaker protection

        Args:
            symbol: Stock/asset symbol
            market_data: Historical OHLCV data
            include_alpha: Whether to include Alpha model prediction
            include_lstm: Whether to include LSTM model prediction
            lstm_horizon: LSTM prediction horizon in minutes

        Returns:
            Unified prediction with circuit breaker protection
        """
        start_time = time.time()
        request_id = f"unified_{symbol}_{int(time.time() * 1000)}"

        with self.lock:
            self.system_stats['total_requests'] += 1
            self.system_stats['concurrent_requests'] += 1
            self.system_stats['peak_concurrent'] = max(
                self.system_stats['peak_concurrent'],
                self.system_stats['concurrent_requests']
            )

        try:
            logger.info(f"🚀 Enhanced prediction request: {symbol} (Alpha: {include_alpha}, LSTM: {include_lstm})")

            # Validate inputs
            if market_data is None or len(market_data) == 0:
                raise ValueError("Market data cannot be empty")

            # Prepare prediction results
            predictions = {}
            models_used = []
            circuit_breaker_states = {}

            # Execute predictions concurrently if enabled
            if self.config['enable_concurrent'] and include_alpha and include_lstm:
                predictions = self._execute_concurrent_predictions(
                    symbol, market_data, lstm_horizon
                )
                models_used = [k for k in predictions.keys() if predictions[k] is not None]

            else:
                # Sequential execution
                if include_alpha and self.alpha_model and self._is_model_healthy('alpha'):
                    try:
                        alpha_features = self._extract_alpha_features(symbol, market_data)
                        alpha_result = self.alpha_model.predict_profitability(alpha_features)
                        predictions['alpha'] = alpha_result
                        models_used.append('alpha')
                        self.system_stats['alpha_model_usage'] += 1
                        self._update_model_status('alpha', True, alpha_result.get('latency_ms', 0))

                    except Exception as e:
                        logger.warning(f"Alpha model failed: {e}")
                        self._update_model_status('alpha', False, 0)
                        predictions['alpha'] = None

                if include_lstm and self.lstm_model and self._is_model_healthy('lstm'):
                    try:
                        lstm_result = self.lstm_model.predict_profitability(
                            symbol, market_data, target_horizon=lstm_horizon
                        )
                        predictions['lstm'] = lstm_result
                        models_used.append('lstm')
                        self.system_stats['lstm_model_usage'] += 1
                        self._update_model_status('lstm', True, lstm_result.get('latency_ms', 0))

                    except Exception as e:
                        logger.warning(f"LSTM model failed: {e}")
                        self._update_model_status('lstm', False, 0)
                        predictions['lstm'] = None

            # Create unified result
            unified_result = self._create_unified_result(predictions, symbol, market_data)

            # Calculate total latency
            total_latency = (time.time() - start_time) * 1000

            # Update system stats
            with self.lock:
                self.system_stats['successful_requests'] += 1
                self.system_stats['concurrent_requests'] -= 1
                self._update_average_latency(total_latency)

            # Collect circuit breaker states
            for model_name in ['alpha', 'lstm']:
                if model_name in self.model_status:
                    circuit_breaker_states[model_name] = self.model_status[model_name].circuit_breaker_state

            # Enhance result with metadata
            enhanced_result = {
                **unified_result,
                'request_id': request_id,
                'system_tier': self.tier.value,
                'models_used': models_used,
                'individual_predictions': predictions,
                'total_latency_ms': round(total_latency, 2),
                'circuit_breaker_states': circuit_breaker_states,
                'system_health_score': self._calculate_health_score(),
                'timestamp': datetime.now().isoformat()
            }

            # Log prediction to database
            self._log_unified_prediction(request_id, enhanced_result)

            logger.info(f"✅ Enhanced prediction completed: {unified_result.get('final_signal', 'HOLD')} "
                        f"confidence ({unified_result.get('confidence', 'MEDIUM')}) - {total_latency:.1f}ms")

            return enhanced_result

        except Exception as e:
            logger.error(f"❌ Enhanced prediction failed: {e}")

            with self.lock:
                self.system_stats['concurrent_requests'] -= 1
                self.system_stats['fallback_usage'] += 1

            # Return safe fallback prediction
            fallback_result = self._get_fallback_prediction(symbol, market_data)
            fallback_result.update({
                'request_id': request_id,
                'system_tier': self.tier.value,
                'models_used': ['fallback'],
                'total_latency_ms': (time.time() - start_time) * 1000,
                'circuit_breaker_status': 'OPEN',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })

            return fallback_result

    def _execute_concurrent_predictions(self, symbol: str, market_data: pd.DataFrame,
                                        lstm_horizon: int) -> Dict[str, Any]:
        """Execute Alpha and LSTM predictions concurrently"""
        predictions = {'alpha': None, 'lstm': None}
        futures = {}

        try:
            # Submit Alpha prediction
            if self.alpha_model and self._is_model_healthy('alpha'):
                alpha_features = self._extract_alpha_features(symbol, market_data)
                futures['alpha'] = self.executor.submit(
                    self.alpha_model.predict_profitability, alpha_features
                )

            # Submit LSTM prediction
            if self.lstm_model and self._is_model_healthy('lstm'):
                futures['lstm'] = self.executor.submit(
                    self.lstm_model.predict_profitability, symbol, market_data, lstm_horizon
                )

            # Collect results with timeout
            timeout = self.config['max_latency_ms'] / 1000.0

            for name, future in futures.items():
                try:
                    result = future.result(timeout=timeout)
                    predictions[name] = result
                    self.system_stats[f'{name}_model_usage'] += 1
                    self._update_model_status(name, True, result.get('latency_ms', 0))

                except Exception as e:
                    logger.warning(f"Concurrent {name} model failed: {e}")
                    predictions[name] = None
                    self._update_model_status(name, False, 0)

        except Exception as e:
            logger.error(f"Concurrent execution failed: {e}")

        return predictions

    def _create_unified_result(self, predictions: Dict[str, Any], symbol: str,
                               market_data: pd.DataFrame) -> Dict[str, Any]:
        """Create unified result from individual model predictions"""
        try:
            alpha_pred = predictions.get('alpha')
            lstm_pred = predictions.get('lstm')

            # Initialize base result
            unified_result = {
                'symbol': symbol,
                'final_signal': 'HOLD',
                'confidence': 'MEDIUM',
                'confidence_score': 0.5,
                'prediction_method': 'UNIFIED'
            }

            # Case 1: Both models available
            if alpha_pred and lstm_pred:
                alpha_pop = alpha_pred.get('ensemble_pop', 0.5)
                lstm_change = lstm_pred.get('price_change_pct', 0.0)

                # Convert LSTM price change to signal
                lstm_signal_strength = abs(lstm_change) / 2.0  # Normalize to 0-1 range
                lstm_pop = 0.5 + (lstm_change / 100.0) * 2  # Convert % change to probability
                lstm_pop = np.clip(lstm_pop, 0.1, 0.9)

                # Weighted ensemble (Alpha 60%, LSTM 40%)
                combined_pop = alpha_pop * 0.6 + lstm_pop * 0.4

                # Determine signal
                if combined_pop > 0.65:
                    unified_result['final_signal'] = 'BUY'
                elif combined_pop < 0.35:
                    unified_result['final_signal'] = 'SELL'
                else:
                    unified_result['final_signal'] = 'HOLD'

                # Confidence based on model agreement
                agreement = 1.0 - abs(alpha_pop - lstm_pop)
                if agreement > 0.8 and (combined_pop > 0.7 or combined_pop < 0.3):
                    unified_result['confidence'] = 'VERY_HIGH'
                elif agreement > 0.6:
                    unified_result['confidence'] = 'HIGH'
                elif agreement > 0.4:
                    unified_result['confidence'] = 'MEDIUM'
                else:
                    unified_result['confidence'] = 'LOW'

                unified_result['confidence_score'] = combined_pop
                unified_result['model_agreement'] = agreement
                unified_result['alpha_contribution'] = alpha_pop * 0.6
                unified_result['lstm_contribution'] = lstm_pop * 0.4

            # Case 2: Only Alpha available
            elif alpha_pred:
                alpha_pop = alpha_pred.get('ensemble_pop', 0.5)

                if alpha_pop > 0.65:
                    unified_result['final_signal'] = 'BUY'
                elif alpha_pop < 0.35:
                    unified_result['final_signal'] = 'SELL'

                unified_result['confidence'] = alpha_pred.get('confidence', 'MEDIUM')
                unified_result['confidence_score'] = alpha_pop
                unified_result['prediction_method'] = 'ALPHA_ONLY'

            # Case 3: Only LSTM available
            elif lstm_pred:
                lstm_change = lstm_pred.get('price_change_pct', 0.0)

                if lstm_change > 1.0:
                    unified_result['final_signal'] = 'BUY'
                elif lstm_change < -1.0:
                    unified_result['final_signal'] = 'SELL'

                unified_result['confidence'] = lstm_pred.get('confidence', 'MEDIUM')
                unified_result['confidence_score'] = 0.5 + (lstm_change / 100.0)
                unified_result['prediction_method'] = 'LSTM_ONLY'
                unified_result['predicted_price'] = lstm_pred.get('predicted_price')
                unified_result['price_change_pct'] = lstm_change

            # Case 4: No models available - fallback
            else:
                fallback = self._get_fallback_prediction(symbol, market_data)
                unified_result.update(fallback)
                unified_result['prediction_method'] = 'FALLBACK'

            return unified_result

        except Exception as e:
            logger.error(f"Unified result creation failed: {e}")
            return self._get_fallback_prediction(symbol, market_data)

    def _extract_alpha_features(self, symbol: str, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Extract features for Alpha model from market data"""
        try:
            if len(market_data) == 0:
                return {'symbol': symbol}

            latest = market_data.iloc[-1]

            # Basic OHLCV features
            features = {
                'symbol': symbol,
                'close': float(latest.get('close', 100)),
                'volume': float(latest.get('volume', 1000)),
                'high': float(latest.get('high', latest.get('close', 100))),
                'low': float(latest.get('low', latest.get('close', 100))),
                'timestamp': datetime.now().isoformat()
            }

            # Calculate technical indicators if enough data
            if len(market_data) >= 14:
                close_prices = market_data['close'].astype(float)

                # RSI
                delta = close_prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                features['rsi_14'] = float(100 - (100 / (1 + rs.iloc[-1])))

                # Moving averages
                features['sma_20'] = float(close_prices.tail(20).mean())
                features['ema_12'] = float(close_prices.ewm(span=12).mean().iloc[-1])

                # Price momentum
                if len(close_prices) >= 5:
                    features['price_momentum_5'] = float(
                        (close_prices.iloc[-1] - close_prices.iloc[-5]) / close_prices.iloc[-5]
                    )

                # Volatility
                if len(close_prices) >= 20:
                    returns = close_prices.pct_change().dropna()
                    features['volatility_20'] = float(returns.tail(20).std())

                # Volume ratio
                if 'volume' in market_data.columns and len(market_data) >= 20:
                    volume_ma = market_data['volume'].tail(20).mean()
                    features['volume_ratio'] = float(latest['volume'] / volume_ma) if volume_ma > 0 else 1.0

            return features

        except Exception as e:
            logger.error(f"Alpha feature extraction failed: {e}")
            return {'symbol': symbol, 'close': 100.0}

    def _get_fallback_prediction(self, symbol: str, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Get intelligent fallback prediction when all models fail"""
        try:
            if len(market_data) == 0:
                return {
                    'final_signal': 'HOLD',
                    'confidence': 'LOW',
                    'confidence_score': 0.5,
                    'prediction_method': 'NO_DATA_FALLBACK'
                }

            # Simple trend analysis
            close_prices = market_data['close'].tail(10)
            if len(close_prices) >= 2:
                trend = (close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0]

                if trend > 0.02:  # 2% upward trend
                    return {
                        'final_signal': 'BUY',
                        'confidence': 'LOW',
                        'confidence_score': 0.6,
                        'prediction_method': 'TREND_FALLBACK',
                        'trend_strength': float(trend)
                    }
                elif trend < -0.02:  # 2% downward trend
                    return {
                        'final_signal': 'SELL',
                        'confidence': 'LOW',
                        'confidence_score': 0.4,
                        'prediction_method': 'TREND_FALLBACK',
                        'trend_strength': float(trend)
                    }

            return {
                'final_signal': 'HOLD',
                'confidence': 'LOW',
                'confidence_score': 0.5,
                'prediction_method': 'NEUTRAL_FALLBACK'
            }

        except Exception as e:
            logger.error(f"Fallback prediction failed: {e}")
            return {
                'final_signal': 'HOLD',
                'confidence': 'LOW',
                'confidence_score': 0.5,
                'prediction_method': 'ERROR_FALLBACK',
                'error': str(e)
            }

    def _is_model_healthy(self, model_name: str) -> bool:
        """Check if a specific model is healthy"""
        if model_name not in self.model_status:
            return False

        status = self.model_status[model_name]
        return (status.health in [ModelHealth.HEALTHY, ModelHealth.DEGRADED] and
                status.circuit_breaker_state != 'OPEN')

    def _update_model_status(self, model_name: str, success: bool, latency: float):
        """Update model status after prediction"""
        if model_name not in self.model_status:
            return

        status = self.model_status[model_name]

        # Update success rate (exponential moving average)
        if success:
            status.success_rate = status.success_rate * 0.9 + 100.0 * 0.1
            status.error_count = max(0, status.error_count - 1)
            status.circuit_breaker_state = 'CLOSED'
        else:
            status.success_rate = status.success_rate * 0.9
            status.error_count += 1
            if status.error_count > 5:
                status.circuit_breaker_state = 'OPEN'

        # Update latency (exponential moving average)
        status.avg_latency = status.avg_latency * 0.8 + latency * 0.2
        status.last_prediction = datetime.now()

        # Update health status
        if status.success_rate > 90 and status.avg_latency < self.config['max_latency_ms']:
            status.health = ModelHealth.HEALTHY
        elif status.success_rate > 70:
            status.health = ModelHealth.DEGRADED
        else:
            status.health = ModelHealth.FAILED

    def _update_average_latency(self, new_latency: float):
        """Update system average latency"""
        total_requests = self.system_stats['total_requests']
        if total_requests == 1:
            self.system_stats['average_latency'] = new_latency
        else:
            # Exponential moving average
            alpha = 0.1
            self.system_stats['average_latency'] = (
                    alpha * new_latency +
                    (1 - alpha) * self.system_stats['average_latency']
            )

    def _calculate_health_score(self) -> float:
        """Calculate overall system health score (0-100)"""
        try:
            if not self.model_status:
                return 50.0  # No models = neutral health

            total_score = 0.0
            total_weight = 0.0

            for model_name, status in self.model_status.items():
                # Model weight
                weight = 1.0
                if model_name == 'alpha':
                    weight = 0.6  # Alpha model more important
                elif model_name == 'lstm':
                    weight = 0.4

                # Model score based on health
                if status.health == ModelHealth.HEALTHY:
                    model_score = 100.0
                elif status.health == ModelHealth.DEGRADED:
                    model_score = 60.0
                elif status.health == ModelHealth.FAILED:
                    model_score = 20.0
                else:  # UNAVAILABLE
                    model_score = 0.0

                # Adjust for success rate and latency
                model_score *= (status.success_rate / 100.0)

                if status.avg_latency > self.config['max_latency_ms']:
                    model_score *= 0.7  # Penalty for high latency

                total_score += model_score * weight
                total_weight += weight

            overall_score = total_score / total_weight if total_weight > 0 else 0.0

            # System-level adjustments
            success_rate = (
                                   self.system_stats['successful_requests'] /
                                   max(self.system_stats['total_requests'], 1)
                           ) * 100

            overall_score *= (success_rate / 100.0)

            return round(np.clip(overall_score, 0.0, 100.0), 1)

        except Exception as e:
            logger.error(f"Health score calculation failed: {e}")
            return 0.0

    def _log_unified_prediction(self, request_id: str, prediction: Dict[str, Any]):
        """Log unified prediction to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO unified_predictions 
                    (id, timestamp, symbol, request_type, alpha_prediction, 
                     lstm_prediction, unified_result, system_tier, total_latency_ms,
                     models_used, circuit_breaker_states, performance_metrics)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    request_id,
                    prediction.get('timestamp', datetime.now().isoformat()),
                    prediction.get('symbol', ''),
                    'enhanced_prediction',
                    json.dumps(prediction.get('individual_predictions', {}).get('alpha')),
                    json.dumps(prediction.get('individual_predictions', {}).get('lstm')),
                    json.dumps({k: v for k, v in prediction.items()
                                if k not in ['individual_predictions', 'circuit_breaker_states']}),
                    prediction.get('system_tier', ''),
                    prediction.get('total_latency_ms', 0),
                    json.dumps(prediction.get('models_used', [])),
                    json.dumps(prediction.get('circuit_breaker_states', {})),
                    json.dumps(self.get_system_stats())
                ))
                conn.commit()

        except Exception as e:
            logger.error(f"Failed to log unified prediction: {e}")

    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        success_rate = (
                               self.system_stats['successful_requests'] /
                               max(self.system_stats['total_requests'], 1)
                       ) * 100

        stats = {
            **self.system_stats,
            'success_rate_pct': round(success_rate, 2),
            'system_tier': self.tier.value,
            'health_score': self._calculate_health_score(),
            'model_status': {name: {
                'health': status.health.value,
                'success_rate': round(status.success_rate, 2),
                'avg_latency': round(status.avg_latency, 2),
                'circuit_breaker_state': status.circuit_breaker_state,
                'error_count': status.error_count
            } for name, status in self.model_status.items()},
            'last_updated': datetime.now().isoformat()
        }

        return stats

    def health_check(self) -> Dict[str, Any]:
        """Comprehensive system health check"""
        health_status = {
            'status': 'HEALTHY',
            'timestamp': datetime.now().isoformat(),
            'system_tier': self.tier.value,
            'components': {},
            'performance': self.get_system_stats(),
            'issues': []
        }

        try:
            # Check database connectivity
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM unified_predictions")
                health_status['components']['database'] = 'OK'

        except Exception as e:
            health_status['components']['database'] = 'ERROR'
            health_status['issues'].append(f"Database: {str(e)}")
            health_status['status'] = 'DEGRADED'

        # Check individual models
        for model_name, status in self.model_status.items():
            if status.health == ModelHealth.HEALTHY:
                health_status['components'][f'{model_name}_model'] = 'OK'
            elif status.health == ModelHealth.DEGRADED:
                health_status['components'][f'{model_name}_model'] = 'DEGRADED'
                health_status['issues'].append(f"{model_name.title()} model degraded")
                if health_status['status'] == 'HEALTHY':
                    health_status['status'] = 'DEGRADED'
            else:
                health_status['components'][f'{model_name}_model'] = 'FAILED'
                health_status['issues'].append(f"{model_name.title()} model failed")
                health_status['status'] = 'DEGRADED'

        # Check system performance
        if self.system_stats['average_latency'] > self.config['max_latency_ms']:
            health_status['issues'].append("High system latency detected")
            health_status['status'] = 'DEGRADED'

        overall_health = self._calculate_health_score()
        if overall_health < 70:
            health_status['issues'].append("Low overall system health")
            health_status['status'] = 'DEGRADED'
        elif overall_health < 50:
            health_status['status'] = 'FAILED'

        return health_status

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


# Factory function for easy instantiation
def create_unified_model_integration(tier: str = "standard",
                                     db_path: str = "marketpulse_production.db") -> UnifiedModelIntegration:
    """
    Factory function to create UnifiedModelIntegration

    Args:
        tier: System tier ("premium", "standard", "economic", "fallback")
        db_path: Path to SQLite database

    Returns:
        Configured UnifiedModelIntegration instance
    """
    tier_enum = SystemTier(tier.lower())
    return UnifiedModelIntegration(db_path=db_path, tier=tier_enum)


# Usage example and testing
if __name__ == "__main__":
    print("🚀 UnifiedModelIntegration - Production Integration Test")
    print("=" * 60)

    try:
        # Create unified integration
        unified_system = create_unified_model_integration(tier="standard")

        # Test health check
        health = unified_system.health_check()
        print(f"✅ Health Status: {health['status']}")
        print(f"   Health Score: {health['performance']['health_score']}")

        # Create sample market data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='T')
        sample_data = pd.DataFrame({
            'timestamp': dates,
            'open': 150 + np.random.randn(100) * 2,
            'high': 152 + np.random.randn(100) * 2,
            'low': 148 + np.random.randn(100) * 2,
            'close': 150 + np.random.randn(100) * 2,
            'volume': 1000 + np.random.randint(0, 500, 100)
        })

        print("\n🧠 Testing unified prediction...")
        result = unified_system.get_enhanced_prediction('AAPL', sample_data)
        print(f"✅ Final Signal: {result['final_signal']} confidence ({result['confidence']})")
        print(f"   Models Used: {', '.join(result['models_used'])}")
        print(f"   Total Latency: {result['total_latency_ms']:.1f}ms")
        print(f"   Health Score: {result['system_health_score']}")

        # Test system stats
        print("\n📊 System Statistics:")
        stats = unified_system.get_system_stats()
        for key, value in stats.items():
            if key == 'model_status':
                continue
            if isinstance(value, float):
                print(f"   {key}: {value:.2f}")
            else:
                print(f"   {key}: {value}")

        print("\n✅ UnifiedModelIntegration test completed successfully!")

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback

        traceback.print_exc()