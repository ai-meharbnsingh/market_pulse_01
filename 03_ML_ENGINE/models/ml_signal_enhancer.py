"""
MarketPulse ML Signal Enhancer - Phase 2, Step 4 ENHANCED
Integrates ML models with advanced error handling, circuit breakers, and performance optimization
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import enhanced error handling and optimization components
try:
    from ..reliability.ml_circuit_breaker import (
        ml_circuit_breaker, MLModelType, MLCircuitBreakerConfig, ml_circuit_registry
    )
    from ..reliability.error_handler import ml_error_handler, ErrorCategory
    from ..optimization.performance_optimizer import (
        optimize_prediction, get_performance_optimizer, OptimizationConfig
    )
    from ..integration.real_model_framework import real_model_manager, ModelConfig, ModelType
    from ..performance.performance_logger import performance_monitor

    ENHANCED_FEATURES_AVAILABLE = True
    logger.info("✅ Enhanced error handling and optimization features loaded")

except ImportError as e:
    ENHANCED_FEATURES_AVAILABLE = False
    logger.warning(f"⚠️ Enhanced features not available: {e}")

    # Fallback decorators and classes for compatibility
    from enum import Enum


    def ml_circuit_breaker(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


    def ml_error_handler(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


    def performance_monitor(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


    class MLModelType(Enum):
        ALPHA_MODEL = "alpha_model"
        LSTM_MODEL = "lstm_model"
        ENSEMBLE = "ensemble"


    class ErrorCategory(Enum):
        MODEL_FAILURE = "model_failure"


    class MLCircuitBreakerConfig:
        def __init__(self, **kwargs):
            pass


    class OptimizationConfig:
        def __init__(self, **kwargs):
            pass


    # Fallback functions
    def optimize_prediction(func, *args, **kwargs):
        return func(*args, **kwargs)


    def get_performance_optimizer():
        return None


    # Mock objects
    class MockModelManager:
        def get_model(self, *args, **kwargs):
            return None


    real_model_manager = MockModelManager()
    ml_circuit_registry = MockModelManager()


    # Fallback decorators for compatibility
    def ml_circuit_breaker(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


    def ml_error_handler(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


    def optimize_prediction(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


    def performance_monitor(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedMockAlphaModel:
    """Enhanced mock alpha model with error handling and circuit breaker protection"""

    def __init__(self):
        self.failure_rate = 0.05  # 5% failure rate for testing
        self.call_count = 0

    @ml_circuit_breaker('mock_alpha', MLModelType.ALPHA_MODEL,
                        config=MLCircuitBreakerConfig(max_prediction_time_ms=30))
    @ml_error_handler(ErrorCategory.MODEL_FAILURE)
    @optimize_prediction(target_ms=20.0)
    @performance_monitor("mock_alpha_prediction")
    def predict_profitability(self, symbol: str, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced alpha prediction with full error handling"""
        self.call_count += 1

        # Simulate occasional failures for testing
        if np.random.random() < self.failure_rate:
            if self.call_count % 10 == 0:
                raise TimeoutError("Alpha model timeout")
            elif self.call_count % 15 == 0:
                raise ValueError("Invalid market data format")

        # Simulate processing time
        time.sleep(0.002 + np.random.random() * 0.003)  # 2-5ms

        # Enhanced prediction logic
        if isinstance(market_data, pd.DataFrame) and not market_data.empty:
            close_price = market_data['close'].iloc[-1] if 'close' in market_data else 100.0

            # More sophisticated mock prediction
            price_momentum = (close_price - market_data['close'].iloc[0]) / market_data['close'].iloc[0] if len(
                market_data) > 1 else 0
            volatility = market_data['close'].std() / market_data['close'].mean() if len(market_data) > 5 else 0.02

            base_prob = 0.5 + (price_momentum * 0.3)  # Momentum factor
            volatility_adj = min(0.1, volatility * 2)  # Volatility adjustment

            signal_strength = abs(base_prob - 0.5)
            confidence_level = min(0.9, 0.5 + signal_strength + (0.2 * (1 - volatility_adj)))

        else:
            base_prob = 0.5 + (np.random.random() - 0.5) * 0.4
            confidence_level = np.random.uniform(0.5, 0.85)

        signal = 'BUY' if base_prob > 0.55 else 'SELL' if base_prob < 0.45 else 'HOLD'

        return {
            'signal': signal,
            'ensemble_pop': base_prob,  # Probability of profit
            'confidence': confidence_level,
            'factors': {
                'momentum': price_momentum if 'price_momentum' in locals() else np.random.uniform(-0.1, 0.1),
                'volatility': volatility if 'volatility' in locals() else np.random.uniform(0.01, 0.05),
                'signal_strength': signal_strength if 'signal_strength' in locals() else abs(base_prob - 0.5)
            },
            'alpha_score': base_prob,
            'method': 'ENHANCED_MOCK_ALPHA',
            'call_count': self.call_count
        }


class EnhancedMockLSTMModel:
    """Enhanced mock LSTM model with error handling and circuit breaker protection"""

    def __init__(self):
        self.failure_rate = 0.03  # 3% failure rate
        self.call_count = 0

    @ml_circuit_breaker('mock_lstm', MLModelType.LSTM_MODEL,
                        config=MLCircuitBreakerConfig(max_prediction_time_ms=40))
    @ml_error_handler(ErrorCategory.MODEL_FAILURE)
    @optimize_prediction(target_ms=25.0)
    @performance_monitor("mock_lstm_prediction")
    def predict_profitability(self, symbol: str, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced LSTM prediction with full error handling"""
        self.call_count += 1

        # Simulate occasional failures
        if np.random.random() < self.failure_rate:
            if self.call_count % 20 == 0:
                raise MemoryError("LSTM model memory allocation failed")
            elif self.call_count % 25 == 0:
                raise RuntimeError("LSTM sequence processing error")

        # Simulate LSTM processing time (slightly longer)
        time.sleep(0.003 + np.random.random() * 0.005)  # 3-8ms

        # Enhanced LSTM prediction logic
        if isinstance(market_data, pd.DataFrame) and not market_data.empty:
            close_prices = market_data['close'].values if 'close' in market_data else np.array([100.0])

            # Time series analysis
            if len(close_prices) >= 5:
                recent_trend = (close_prices[-1] - close_prices[-5]) / close_prices[-5]
                price_volatility = np.std(close_prices[-10:]) / np.mean(close_prices[-10:]) if len(
                    close_prices) >= 10 else 0.02

                # LSTM-style prediction based on sequence
                sequence_momentum = np.mean(np.diff(close_prices[-5:])) if len(close_prices) >= 5 else 0
                trend_strength = abs(recent_trend)

                # Predict future direction
                prediction_prob = 0.5 + (recent_trend * 0.4) + (sequence_momentum * 0.2)
                prediction_prob = max(0.1, min(0.9, prediction_prob))

                confidence = min(0.9, 0.4 + trend_strength * 2 + (0.2 * (1 - price_volatility)))

                # Price prediction
                predicted_price = close_prices[-1] * (1 + recent_trend + np.random.normal(0, price_volatility * 0.5))
                price_change_pct = (predicted_price - close_prices[-1]) / close_prices[-1]

            else:
                prediction_prob = 0.5 + (np.random.random() - 0.5) * 0.3
                confidence = np.random.uniform(0.4, 0.8)
                predicted_price = close_prices[-1] * np.random.uniform(0.98, 1.02)
                price_change_pct = np.random.uniform(-0.02, 0.02)

        else:
            prediction_prob = 0.5 + (np.random.random() - 0.5) * 0.3
            confidence = np.random.uniform(0.4, 0.8)
            predicted_price = 100.0 * np.random.uniform(0.98, 1.02)
            price_change_pct = np.random.uniform(-0.02, 0.02)

        signal = 'BUY' if prediction_prob > 0.6 else 'SELL' if prediction_prob < 0.4 else 'HOLD'

        return {
            'signal': signal,
            'ensemble_pop': prediction_prob,
            'confidence': confidence,
            'predicted_price': predicted_price,
            'price_change_pct': price_change_pct,
            'horizon': 5,
            'method': 'ENHANCED_MOCK_LSTM',
            'sequence_analysis': {
                'trend_strength': trend_strength if 'trend_strength' in locals() else 0.5,
                'volatility': price_volatility if 'price_volatility' in locals() else 0.02
            },
            'call_count': self.call_count
        }


class EnhancedMLSignalEnhancer:
    """
    Enhanced ML Signal Enhancer with circuit breakers, error handling, and performance optimization
    """

    def __init__(self, config=None):
        """
        Initialize Enhanced ML Signal Enhancer with advanced capabilities
        """
        # Default configuration with enhanced ensemble settings
        self.config = {
            'ensemble': {
                'min_models_agree': 2,
                'confidence_boost': 0.15,
                'risk_threshold': 0.6,
                'fallback_enabled': True,
                'performance_weighting': True
            },
            'alpha_model': {
                'min_observations': 50,
                'default_confidence': 0.5,
                'circuit_breaker_enabled': True
            },
            'lstm': {
                'sequence_length': 30,
                'forecast_horizon': 5,
                'circuit_breaker_enabled': True
            },
            'performance': {
                'target_prediction_time_ms': 20.0,
                'enable_caching': True,
                'enable_batch_processing': True,
                'optimization_level': 'high'
            },
            'reliability': {
                'enable_circuit_breakers': True,
                'enable_error_handling': True,
                'fallback_strategy': 'statistical',
                'max_failures_before_fallback': 3
            }
        }

        # Override default config if custom config provided
        if config:
            self._update_config(config)

        # Initialize enhanced models with error handling
        logger.info("🚀 Initializing Enhanced ML Signal Enhancer...")

        self.alpha_model = EnhancedMockAlphaModel()
        self.lstm_forecaster = EnhancedMockLSTMModel()

        # Performance tracking and model management
        self.model_performance = {
            'alpha': {'calls': 0, 'failures': 0, 'avg_time_ms': 0.0, 'success_rate': 1.0},
            'lstm': {'calls': 0, 'failures': 0, 'avg_time_ms': 0.0, 'success_rate': 1.0},
            'ensemble': {'calls': 0, 'failures': 0, 'avg_time_ms': 0.0, 'success_rate': 1.0}
        }

        # Enhanced signal weighting with performance-based adjustments
        self.base_signal_weights = {
            'technical': 0.25,
            'alpha_model': 0.40,
            'lstm_forecaster': 0.35
        }
        self.current_signal_weights = self.base_signal_weights.copy()

        # Initialize performance optimizer if available
        if ENHANCED_FEATURES_AVAILABLE:
            try:
                optimizer_config = OptimizationConfig(
                    target_prediction_time_ms=self.config['performance']['target_prediction_time_ms'],
                    enable_async_processing=True,
                    enable_batch_processing=self.config['performance']['enable_batch_processing']
                )
                self.optimizer = get_performance_optimizer(optimizer_config)
                logger.info("⚡ Performance optimizer initialized")
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize optimizer: {e}")
                self.optimizer = None
        else:
            self.optimizer = None

        # Circuit breaker registry tracking
        self.circuit_breakers = []

        logger.info("✅ Enhanced ML Signal Enhancer initialized successfully")

    def _update_config(self, custom_config):
        """Update configuration with custom settings"""
        for key, value in custom_config.items():
            if key in self.config and isinstance(self.config[key], dict):
                self.config[key].update(value)
            else:
                self.config[key] = value

    @ml_circuit_breaker('ml_signal_enhancer', MLModelType.ENSEMBLE,
                        config=MLCircuitBreakerConfig(max_prediction_time_ms=50))
    @ml_error_handler(ErrorCategory.MODEL_FAILURE)
    @optimize_prediction(target_ms=20.0)
    @performance_monitor("enhanced_signal_processing")
    def enhance_signal(self, symbol: str, technical_signal: Dict, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Enhanced signal processing with full error handling and optimization
        """
        enhancement_start = time.time()

        try:
            logger.debug(f"🎯 Enhancing signal for {symbol}")

            # Input validation with error handling
            if not self._validate_inputs(symbol, technical_signal, market_data):
                return self._get_fallback_signal(symbol, technical_signal, "Invalid inputs")

            # Get ML predictions with error handling
            predictions = self._get_ml_predictions(symbol, market_data)

            # Ensemble processing with performance weighting
            enhanced_result = self._process_ensemble_signals(
                symbol, technical_signal, predictions, market_data
            )

            # Update performance metrics
            processing_time = (time.time() - enhancement_start) * 1000
            self._update_performance_metrics('ensemble', processing_time, success=True)

            # Add enhancement metadata
            enhanced_result.update({
                'enhancement_time_ms': processing_time,
                'models_used': len([p for p in predictions.values() if p is not None]),
                'reliability_score': self._calculate_reliability_score(predictions),
                'optimizer_enabled': self.optimizer is not None,
                'circuit_breaker_status': self._get_circuit_breaker_status()
            })

            logger.debug(f"✅ Signal enhanced for {symbol} in {processing_time:.1f}ms")
            return enhanced_result

        except Exception as e:
            processing_time = (time.time() - enhancement_start) * 1000
            self._update_performance_metrics('ensemble', processing_time, success=False)

            logger.error(f"❌ Signal enhancement failed for {symbol}: {e}")
            return self._get_fallback_signal(symbol, technical_signal, str(e))

    def _validate_inputs(self, symbol: str, technical_signal: Dict, market_data: pd.DataFrame) -> bool:
        """Validate input parameters"""
        try:
            if not symbol or not isinstance(symbol, str):
                return False

            if not technical_signal or not isinstance(technical_signal, dict):
                return False

            if not isinstance(market_data, pd.DataFrame) or market_data.empty:
                return False

            required_fields = ['action', 'confidence']
            if not all(field in technical_signal for field in required_fields):
                return False

            return True

        except Exception as e:
            logger.warning(f"⚠️ Input validation error: {e}")
            return False

    def _get_ml_predictions(self, symbol: str, market_data: pd.DataFrame) -> Dict[str, Optional[Dict]]:
        """Get predictions from all ML models with error handling"""
        predictions = {
            'alpha': None,
            'lstm': None
        }

        # Alpha model prediction
        if self.config['alpha_model'].get('circuit_breaker_enabled', True):
            try:
                predictions['alpha'] = self._get_alpha_prediction(symbol, market_data)
            except Exception as e:
                logger.warning(f"⚠️ Alpha model failed: {e}")
                self._update_performance_metrics('alpha', 0, success=False)

        # LSTM model prediction
        if self.config['lstm'].get('circuit_breaker_enabled', True):
            try:
                predictions['lstm'] = self._get_lstm_prediction(symbol, market_data)
            except Exception as e:
                logger.warning(f"⚠️ LSTM model failed: {e}")
                self._update_performance_metrics('lstm', 0, success=False)

        return predictions

    def _get_alpha_prediction(self, symbol: str, market_data: pd.DataFrame) -> Optional[Dict]:
        """Get prediction from Enhanced Alpha Model with error handling"""
        if not self.alpha_model or len(market_data) < self.config['alpha_model']['min_observations']:
            return None

        try:
            pred_start = time.time()
            prediction = self.alpha_model.predict_profitability(symbol, market_data)
            pred_time = (time.time() - pred_start) * 1000

            self._update_performance_metrics('alpha', pred_time, success=True)

            logger.debug(f"🧠 Alpha prediction for {symbol}: {prediction.get('ensemble_pop', 0.5):.3f}")
            return prediction

        except Exception as e:
            logger.error(f"❌ Alpha model prediction failed: {e}")
            self._update_performance_metrics('alpha', 0, success=False)
            raise  # Re-raise for circuit breaker handling

    def _get_lstm_prediction(self, symbol: str, market_data: pd.DataFrame) -> Optional[Dict]:
        """Get prediction from Enhanced LSTM Model with error handling"""
        if not self.lstm_forecaster or len(market_data) < self.config['lstm']['sequence_length']:
            return None

        try:
            pred_start = time.time()
            prediction = self.lstm_forecaster.predict_profitability(symbol, market_data)
            pred_time = (time.time() - pred_start) * 1000

            self._update_performance_metrics('lstm', pred_time, success=True)

            logger.debug(f"🔮 LSTM prediction for {symbol}: {prediction.get('ensemble_pop', 0.5):.3f}")
            return prediction

        except Exception as e:
            logger.error(f"❌ LSTM model prediction failed: {e}")
            self._update_performance_metrics('lstm', 0, success=False)
            raise  # Re-raise for circuit breaker handling
            # Alpha model expects specific data format
            prediction = self.alpha_model.predict_profitability(symbol, market_data)

            return {
                'signal': prediction.get('signal', 'HOLD'),
                'confidence': prediction.get('confidence', 0.5),
                'factors': prediction.get('factors', {}),
                'alpha_score': prediction.get('alpha_score', 0.0),
                'model': 'alpha_model'
            }

        except Exception as e:
            logger.warning(f"Alpha model prediction failed for {symbol}: {e}")
            return None

    def _get_lstm_prediction(self, symbol: str, market_data: pd.DataFrame) -> Optional[Dict]:
        """
        Get prediction from LSTM Forecaster
        """
        if not self.lstm_forecaster or len(market_data) < self.config['lstm']['sequence_length']:
            return None

        try:
            # LSTM forecaster uses predict_profitability method
            prediction = self.lstm_forecaster.predict_profitability(symbol, market_data)

            return {
                'signal': prediction.get('signal', 'HOLD'),
                'confidence': prediction.get('confidence', 0.5),
                'predicted_price': prediction.get('predicted_price', None),
                'price_change_pct': prediction.get('price_change_pct', 0.0),
                'forecast_horizon': prediction.get('horizon', 5),
                'model': 'lstm_forecaster'
            }

        except Exception as e:
            logger.warning(f"LSTM prediction failed for {symbol}: {e}")
            return None

    def enhance_signal(self, symbol: str, technical_signal: Dict, market_data: pd.DataFrame) -> Dict:
        """
        Enhance trading signal by integrating multiple ML models
        """
        enhanced_signal = {
            'symbol': symbol,
            'ml_predictions': {},
            'models_used': [],
            'ensemble_signal': 'HOLD',
            'confidence': technical_signal.get('confidence', 0.5)
        }

        try:
            # Get predictions from Alpha Model
            alpha_prediction = self._get_alpha_prediction(symbol, market_data)
            if alpha_prediction:
                enhanced_signal['ml_predictions']['alpha'] = alpha_prediction
                enhanced_signal['models_used'].append('alpha')

            # Get predictions from LSTM Forecaster
            lstm_prediction = self._get_lstm_prediction(symbol, market_data)
            if lstm_prediction:
                enhanced_signal['ml_predictions']['lstm'] = lstm_prediction
                enhanced_signal['models_used'].append('lstm')

            # Generate ensemble signal
            ensemble_result = self._generate_ensemble_signal(
                technical_signal,
                enhanced_signal['ml_predictions']
            )

            enhanced_signal.update(ensemble_result)

            # Add prediction details for transparency
            enhanced_signal['prediction_details'] = self._create_prediction_summary(
                technical_signal, enhanced_signal['ml_predictions']
            )

            logger.info(f"✅ Signal enhanced - Final: {enhanced_signal['ensemble_signal']} "
                        f"(Confidence: {enhanced_signal['confidence']:.2%})")

        except Exception as e:
            logger.error(f"❌ Error enhancing signal for {symbol}: {e}")
            # Fallback to technical signal only
            enhanced_signal['ensemble_signal'] = technical_signal.get('signal', 'HOLD')
            enhanced_signal['confidence'] = technical_signal.get('confidence', 0.5)

        return enhanced_signal

    def _generate_ensemble_signal(self,
                                  technical_signal: Dict,
                                  ml_predictions: Dict) -> Dict:
        """
        Generate ensemble signal from all available predictions
        """
        signals = []

        # Add technical signal
        tech_signal = technical_signal.get('signal', 'HOLD')
        tech_confidence = technical_signal.get('confidence', 0.5)
        signals.append((tech_signal, tech_confidence, self.signal_weights['technical']))

        # Add ML predictions
        for model_name, prediction in ml_predictions.items():
            signal = prediction.get('signal', 'HOLD')
            confidence = prediction.get('confidence', 0.5)
            weight = self.signal_weights.get(model_name, 0.1)
            signals.append((signal, confidence, weight))

        # Calculate weighted signal
        signal_scores = {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0}
        total_weight = 0.0
        weighted_confidence = 0.0

        for signal, confidence, weight in signals:
            signal_scores[signal] += weight * confidence
            total_weight += weight
            weighted_confidence += weight * confidence

        # Determine final signal
        final_signal = max(signal_scores, key=signal_scores.get)
        final_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.5

        # Apply ensemble rules
        models_agreeing = sum(1 for s, _, _ in signals if s == final_signal)

        # Boost confidence if multiple models agree
        if models_agreeing >= self.config['ensemble']['min_models_agree']:
            final_confidence += self.config['ensemble']['confidence_boost']
            final_confidence = min(final_confidence, 1.0)

        # Risk adjustment - reduce confidence for HOLD signals
        risk_adjusted_confidence = final_confidence
        if final_signal == 'HOLD':
            risk_adjusted_confidence *= 0.8

        return {
            'ensemble_signal': final_signal,
            'confidence': final_confidence,
            'risk_adjusted_confidence': risk_adjusted_confidence,
            'signal_scores': signal_scores,
            'models_agreeing': models_agreeing,
            'total_models': len(signals)
        }

    def _create_prediction_summary(self,
                                   technical_signal: Dict,
                                   ml_predictions: Dict) -> Dict:
        """
        Create detailed prediction summary for analysis
        """
        summary = {
            'technical_analysis': {
                'signal': technical_signal.get('signal', 'HOLD'),
                'confidence': technical_signal.get('confidence', 0.5),
                'indicators': technical_signal.get('indicators', {})
            },
            'ml_models': {},
            'consensus': {
                'strongest_signal': None,
                'agreement_pct': 0.0
            },
            'divergences': []
        }

        # Add ML model details
        for model_name, prediction in ml_predictions.items():
            summary['ml_models'][model_name] = {
                'signal': prediction.get('signal', 'HOLD'),
                'confidence': prediction.get('confidence', 0.5),
                'key_factors': prediction.get('factors', {})
            }

        # Compute consensus and divergences
        model_signals = [pred.get('signal', 'HOLD') for pred in ml_predictions.values()]
        signal_counts = {}
        for signal in model_signals:
            signal_counts[signal] = signal_counts.get(signal, 0) + 1

        # Determine strongest signal
        if signal_counts:
            summary['consensus']['strongest_signal'] = max(signal_counts, key=signal_counts.get)
            summary['consensus']['agreement_pct'] = max(signal_counts.values()) / len(model_signals) * 100

        # Detect divergences
        unique_signals = set(model_signals)
        if len(unique_signals) > 1:
            summary['divergences'] = list(unique_signals)

        return summary

    def update_performance(self, symbol: str, expected_signal: str, actual_signal: str, models_used: List[str]):
        """
        Update model performance tracking
        """
        for model in models_used:
            # Initialize model performance if not exists
            if model not in self.model_performance:
                self.model_performance[model] = {
                    'total_predictions': 0,
                    'accuracy': 0.0
                }

            # Update tracking
            self.model_performance[model]['total_predictions'] += 1

            # Calculate accuracy
            if expected_signal == actual_signal:
                current_accuracy = self.model_performance[model]['accuracy']
                total = self.model_performance[model]['total_predictions']

                # Running accuracy calculation
                new_accuracy = ((current_accuracy * (total - 1)) + 1) / total
                self.model_performance[model]['accuracy'] = new_accuracy

    def get_performance_summary(self) -> Dict:
        """
        Generate comprehensive performance summary
        """
        summary = {
            'model_performance': self.model_performance.copy(),
            'total_predictions': sum(m['total_predictions'] for m in self.model_performance.values()),
            'average_accuracy': np.mean(
                [m['accuracy'] for m in self.model_performance.values()]) if self.model_performance else 0.0,
            'models_available': len(self.model_performance)
        }

        return summary


# Export alias for backwards compatibility
MLSignalEnhancer = EnhancedMLSignalEnhancer

# Example usage
if __name__ == "__main__":
    print("🧪 Testing Enhanced ML Signal Enhancer")

    # Create enhancer
    enhancer = MLSignalEnhancer()

    # Test signal features
    test_features = {
        'price': 150.0,
        'volume': 1000000,
        'rsi': 65.0,
        'macd': 0.5,
        'bb_position': 0.7
    }

    # Test enhancement
    result = enhancer.enhance_signal('BUY', test_features, 0.7)
    print(f"Enhanced signal: {result}")

    print("✅ Enhanced ML Signal Enhancer test completed")