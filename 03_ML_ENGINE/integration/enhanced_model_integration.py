# 03_ML_ENGINE/integration/enhanced_model_integration.py
"""
Enhanced Model Integration Framework - Phase 2, Step 4
Production-ready framework for integrating ML models with circuit breakers and error handling

Location: #03_ML_ENGINE/integration/enhanced_model_integration.py

This framework provides:
- Production-ready ML model integration
- Circuit breaker protected model calls
- Intelligent fallback mechanisms
- Performance optimization for sub-20ms targets
- Health monitoring and automatic recovery
- A/B testing capabilities for model versions
"""

import time
import logging
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import numpy as np
import pandas as pd

# Import reliability components
try:
    from ..reliability.ml_circuit_breaker import (
        ml_circuit_breaker, MLModelType, MLCircuitBreakerConfig, ml_circuit_registry
    )
    from ..reliability.error_handler import ml_error_handler_decorator
    from ..performance.performance_logger import performance_monitor

    HAS_RELIABILITY = True
except ImportError:
    # Fallback for direct testing
    HAS_RELIABILITY = False


    def ml_circuit_breaker(*args, **kwargs):
        def decorator(func): return func

        return decorator


    def ml_error_handler_decorator(*args, **kwargs):
        def decorator(func): return func

        return decorator


    def performance_monitor(*args, **kwargs):
        def decorator(func): return func

        return decorator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTier(Enum):
    """Model performance tiers"""
    PREMIUM = "premium"  # <10ms, high accuracy
    STANDARD = "standard"  # <20ms, good accuracy
    ECONOMIC = "economic"  # <50ms, acceptable accuracy
    FALLBACK = "fallback"  # >50ms, basic heuristics


class PredictionQuality(Enum):
    """Quality levels of model predictions"""
    EXCELLENT = "excellent"  # Confidence >0.8, low latency
    GOOD = "good"  # Confidence >0.6, acceptable latency
    ACCEPTABLE = "acceptable"  # Confidence >0.4, higher latency
    POOR = "poor"  # Confidence <0.4, high latency


@dataclass
class ModelPerformanceProfile:
    """Performance profile for ML models"""
    model_name: str
    tier: ModelTier
    target_latency_ms: float
    max_latency_ms: float
    target_accuracy: float
    memory_limit_mb: int
    cpu_cores: int = 1

    # Circuit breaker settings
    failure_threshold: int = 3
    success_threshold: int = 2
    timeout_seconds: int = 30


@dataclass
class PredictionResult:
    """Structured prediction result with metadata"""
    prediction: Any
    confidence: float
    quality: PredictionQuality
    latency_ms: float
    model_name: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnhancedModelManager:
    """Enhanced model manager with reliability and performance optimization"""

    def __init__(self):
        """Initialize enhanced model manager"""
        self.models = {}
        self.performance_profiles = {}
        self.model_health = {}
        self.lock = threading.Lock()

        # Performance tracking
        self.prediction_history = []
        self.max_history = 1000

        # Model tier configurations
        self._setup_performance_profiles()

        logger.info("🚀 Enhanced Model Manager initialized")

    def _setup_performance_profiles(self):
        """Setup performance profiles for different model tiers"""
        self.performance_profiles = {
            ModelTier.PREMIUM: ModelPerformanceProfile(
                model_name="premium_ensemble",
                tier=ModelTier.PREMIUM,
                target_latency_ms=8.0,
                max_latency_ms=15.0,
                target_accuracy=0.85,
                memory_limit_mb=200,
                failure_threshold=2,
                success_threshold=3
            ),
            ModelTier.STANDARD: ModelPerformanceProfile(
                model_name="standard_model",
                tier=ModelTier.STANDARD,
                target_latency_ms=15.0,
                max_latency_ms=25.0,
                target_accuracy=0.75,
                memory_limit_mb=150,
                failure_threshold=3,
                success_threshold=2
            ),
            ModelTier.ECONOMIC: ModelPerformanceProfile(
                model_name="economic_model",
                tier=ModelTier.ECONOMIC,
                target_latency_ms=30.0,
                max_latency_ms=50.0,
                target_accuracy=0.65,
                memory_limit_mb=100,
                failure_threshold=5,
                success_threshold=2
            ),
            ModelTier.FALLBACK: ModelPerformanceProfile(
                model_name="fallback_heuristics",
                tier=ModelTier.FALLBACK,
                target_latency_ms=5.0,
                max_latency_ms=10.0,
                target_accuracy=0.55,
                memory_limit_mb=50,
                failure_threshold=10,
                success_threshold=1
            )
        }

    def register_model(self, tier: ModelTier, model_callable: Callable,
                       custom_profile: Optional[ModelPerformanceProfile] = None):
        """Register a model with the manager"""
        profile = custom_profile or self.performance_profiles[tier]

        # Create circuit breaker configuration
        circuit_config = MLCircuitBreakerConfig(
            failure_threshold=profile.failure_threshold,
            success_threshold=profile.success_threshold,
            max_prediction_time_ms=profile.max_latency_ms
        )

        # Wrap model with reliability components
        @ml_circuit_breaker(profile.model_name, MLModelType.ENSEMBLE, circuit_config)
        @ml_error_handler_decorator(profile.model_name)
        @performance_monitor(track_memory=True)
        def protected_model(*args, **kwargs):
            return model_callable(*args, **kwargs)

        with self.lock:
            self.models[tier] = protected_model
            self.model_health[tier] = {'status': 'healthy', 'last_check': datetime.now()}

        logger.info(f"✅ Registered {tier.value} model: {profile.model_name}")

    def predict_with_fallback(self, input_data: Dict,
                              preferred_tier: ModelTier = ModelTier.STANDARD) -> PredictionResult:
        """Make prediction with intelligent fallback through tiers"""
        start_time = time.time()

        # Try models in order of preference, fallback to lower tiers
        tier_order = self._get_fallback_order(preferred_tier)

        for tier in tier_order:
            if tier not in self.models:
                continue

            try:
                model = self.models[tier]
                profile = self.performance_profiles[tier]

                # Attempt prediction
                prediction_start = time.time()
                prediction = model(input_data)
                latency_ms = (time.time() - prediction_start) * 1000

                # Determine prediction quality
                quality = self._assess_prediction_quality(prediction, latency_ms, profile)

                # Create result
                result = PredictionResult(
                    prediction=prediction,
                    confidence=prediction.get('confidence', 0.5),
                    quality=quality,
                    latency_ms=latency_ms,
                    model_name=profile.model_name,
                    timestamp=datetime.now(),
                    metadata={
                        'tier': tier.value,
                        'total_latency_ms': (time.time() - start_time) * 1000,
                        'fallback_used': tier != preferred_tier
                    }
                )

                # Record successful prediction
                self._record_prediction(result)

                logger.debug(f"✅ Prediction successful using {tier.value} model: "
                             f"{latency_ms:.1f}ms, confidence: {result.confidence:.2f}")

                return result

            except Exception as e:
                logger.warning(f"⚠️ {tier.value} model failed: {e}")
                self._update_model_health(tier, 'degraded')
                continue

        # If all models failed, create emergency fallback
        return self._emergency_fallback(input_data, start_time)

    def _get_fallback_order(self, preferred_tier: ModelTier) -> List[ModelTier]:
        """Get fallback order starting from preferred tier"""
        all_tiers = [ModelTier.PREMIUM, ModelTier.STANDARD, ModelTier.ECONOMIC, ModelTier.FALLBACK]

        # Start from preferred tier and work down
        preferred_index = all_tiers.index(preferred_tier)
        return all_tiers[preferred_index:]

    def _assess_prediction_quality(self, prediction: Dict, latency_ms: float,
                                   profile: ModelPerformanceProfile) -> PredictionQuality:
        """Assess prediction quality based on confidence and latency"""
        confidence = prediction.get('confidence', 0.5)

        # Quality assessment matrix
        if confidence >= 0.8 and latency_ms <= profile.target_latency_ms:
            return PredictionQuality.EXCELLENT
        elif confidence >= 0.6 and latency_ms <= profile.max_latency_ms:
            return PredictionQuality.GOOD
        elif confidence >= 0.4:
            return PredictionQuality.ACCEPTABLE
        else:
            return PredictionQuality.POOR

    def _record_prediction(self, result: PredictionResult):
        """Record prediction for performance tracking"""
        with self.lock:
            self.prediction_history.append(result)

            # Keep history size manageable
            if len(self.prediction_history) > self.max_history:
                self.prediction_history = self.prediction_history[-self.max_history:]

    def _update_model_health(self, tier: ModelTier, status: str):
        """Update model health status"""
        with self.lock:
            if tier in self.model_health:
                self.model_health[tier] = {
                    'status': status,
                    'last_check': datetime.now()
                }

    def _emergency_fallback(self, input_data: Dict, start_time: float) -> PredictionResult:
        """Emergency fallback when all models fail"""
        logger.error("🚨 All models failed - using emergency fallback")

        # Simple heuristic-based prediction
        fallback_prediction = {
            'signal': 'HOLD',  # Conservative default
            'confidence': 0.3,
            'reason': 'emergency_fallback'
        }

        latency_ms = (time.time() - start_time) * 1000

        return PredictionResult(
            prediction=fallback_prediction,
            confidence=0.3,
            quality=PredictionQuality.POOR,
            latency_ms=latency_ms,
            model_name='emergency_fallback',
            timestamp=datetime.now(),
            metadata={'emergency_fallback': True}
        )

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        with self.lock:
            if not self.prediction_history:
                return {'status': 'no_data'}

            recent_predictions = [p for p in self.prediction_history
                                  if (datetime.now() - p.timestamp).seconds < 300]  # Last 5 minutes

            if not recent_predictions:
                return {'status': 'no_recent_data'}

            # Calculate metrics
            latencies = [p.latency_ms for p in recent_predictions]
            confidences = [p.confidence for p in recent_predictions]

            quality_counts = {}
            for quality in PredictionQuality:
                quality_counts[quality.value] = sum(1 for p in recent_predictions
                                                    if p.quality == quality)

            tier_usage = {}
            for p in recent_predictions:
                tier = p.metadata.get('tier', 'unknown')
                tier_usage[tier] = tier_usage.get(tier, 0) + 1

            return {
                'total_predictions': len(recent_predictions),
                'avg_latency_ms': np.mean(latencies),
                'p95_latency_ms': np.percentile(latencies, 95),
                'p99_latency_ms': np.percentile(latencies, 99),
                'avg_confidence': np.mean(confidences),
                'quality_distribution': quality_counts,
                'tier_usage': tier_usage,
                'sub_20ms_percentage': sum(1 for l in latencies if l < 20) / len(latencies) * 100,
                'model_health': dict(self.model_health)
            }

    def optimize_for_latency(self):
        """Optimize model selection for latency"""
        metrics = self.get_performance_metrics()

        if metrics.get('status') in ['no_data', 'no_recent_data']:
            return

        sub_20ms_percentage = metrics.get('sub_20ms_percentage', 0)

        if sub_20ms_percentage < 80:  # Target 80% sub-20ms predictions
            logger.info(f"⚡ Optimizing for latency: {sub_20ms_percentage:.1f}% sub-20ms")

            # Recommend premium tier for better latency
            logger.info("📊 Recommendation: Use premium tier models for better latency")

    async def predict_async(self, input_data: Dict,
                            preferred_tier: ModelTier = ModelTier.STANDARD) -> PredictionResult:
        """Async prediction for high-throughput scenarios"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.predict_with_fallback,
                                          input_data, preferred_tier)

    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of all models"""
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'models': {},
            'performance': self.get_performance_metrics()
        }

        critical_issues = 0

        for tier, health in self.model_health.items():
            model_status = {
                'tier': tier.value,
                'status': health['status'],
                'last_check': health['last_check'].isoformat()
            }

            if tier in self.models:
                # Get circuit breaker health if available
                if HAS_RELIABILITY:
                    try:
                        profile = self.performance_profiles[tier]
                        breaker = ml_circuit_registry.get(profile.model_name)
                        if breaker:
                            breaker_health = breaker.get_health_report()
                            model_status['circuit_breaker'] = breaker_health
                    except Exception as e:
                        logger.debug(f"Could not get circuit breaker health: {e}")

                model_status['registered'] = True
            else:
                model_status['registered'] = False
                if tier != ModelTier.FALLBACK:  # Fallback is optional
                    critical_issues += 1

            health_status['models'][tier.value] = model_status

        # Determine overall status
        if critical_issues > 0:
            health_status['overall_status'] = 'critical'
        elif any(h['status'] == 'degraded' for h in self.model_health.values()):
            health_status['overall_status'] = 'degraded'

        return health_status


# Global enhanced model manager instance
enhanced_model_manager = EnhancedModelManager()


# Example model implementations for testing
def create_premium_alpha_model():
    """Create premium alpha model with high performance"""

    def premium_alpha_predict(input_data: Dict) -> Dict:
        # Simulate premium model processing
        time.sleep(0.008)  # 8ms processing time

        # High confidence predictions
        confidence = np.random.normal(0.85, 0.1)
        confidence = max(0.6, min(0.95, confidence))  # Clamp to reasonable range

        signal = np.random.choice(['BUY', 'SELL', 'HOLD'], p=[0.4, 0.35, 0.25])

        return {
            'signal': signal,
            'confidence': confidence,
            'ensemble_pop': confidence,
            'method': 'premium_alpha',
            'features_used': ['price', 'volume', 'momentum', 'volatility']
        }

    return premium_alpha_predict


def create_standard_lstm_model():
    """Create standard LSTM model"""

    def standard_lstm_predict(input_data: Dict) -> Dict:
        # Simulate LSTM processing
        time.sleep(0.015)  # 15ms processing time

        confidence = np.random.normal(0.75, 0.15)
        confidence = max(0.4, min(0.9, confidence))

        # Price prediction
        current_price = input_data.get('price', 100)
        price_change_pct = np.random.normal(0, 0.02)  # ±2% typical
        predicted_price = current_price * (1 + price_change_pct)

        return {
            'predicted_price': predicted_price,
            'confidence': confidence,
            'direction': 'UP' if price_change_pct > 0 else 'DOWN',
            'method': 'lstm_neural_network',
            'timeframe': '1h'
        }

    return standard_lstm_predict


def create_economic_ensemble_model():
    """Create economic ensemble model"""

    def economic_ensemble_predict(input_data: Dict) -> Dict:
        # Simulate ensemble processing
        time.sleep(0.025)  # 25ms processing time

        confidence = np.random.normal(0.65, 0.1)
        confidence = max(0.35, min(0.8, confidence))

        # Simple ensemble of technical indicators
        signals = ['BUY', 'SELL', 'HOLD']
        weights = [0.3, 0.25, 0.45]  # Conservative bias toward HOLD

        signal = np.random.choice(signals, p=weights)

        return {
            'signal': signal,
            'confidence': confidence,
            'method': 'economic_ensemble',
            'components': ['rsi', 'macd', 'bb']
        }

    return economic_ensemble_predict


def setup_example_models():
    """Setup example models for testing"""
    # Register models with the enhanced manager
    enhanced_model_manager.register_model(ModelTier.PREMIUM, create_premium_alpha_model())
    enhanced_model_manager.register_model(ModelTier.STANDARD, create_standard_lstm_model())
    enhanced_model_manager.register_model(ModelTier.ECONOMIC, create_economic_ensemble_model())

    logger.info("✅ Example models registered successfully")


if __name__ == "__main__":
    # Demo the enhanced model integration
    print("🚀 Enhanced Model Integration Demo")
    print("=" * 50)

    # Setup example models
    setup_example_models()

    # Test predictions with different tiers
    test_data = {
        'symbol': 'AAPL',
        'price': 150.0,
        'volume': 1000000,
        'timestamp': datetime.now().isoformat()
    }

    print("\n🧪 Testing predictions with different tiers:")

    for preferred_tier in [ModelTier.PREMIUM, ModelTier.STANDARD, ModelTier.ECONOMIC]:
        try:
            result = enhanced_model_manager.predict_with_fallback(test_data, preferred_tier)
            print(f"✅ {preferred_tier.value}: {result.model_name} - "
                  f"{result.latency_ms:.1f}ms - {result.quality.value} - "
                  f"confidence: {result.confidence:.2f}")
        except Exception as e:
            print(f"❌ {preferred_tier.value}: {e}")

    # Show performance metrics
    print("\n📊 Performance Metrics:")
    metrics = enhanced_model_manager.get_performance_metrics()
    if metrics.get('status') != 'no_data':
        print(f"  Average Latency: {metrics.get('avg_latency_ms', 0):.1f}ms")
        print(f"  Sub-20ms Rate: {metrics.get('sub_20ms_percentage', 0):.1f}%")
        print(f"  Average Confidence: {metrics.get('avg_confidence', 0):.2f}")
        print(f"  Total Predictions: {metrics.get('total_predictions', 0)}")

    # Health check
    print("\n🏥 System Health Check:")
    health = enhanced_model_manager.health_check()
    print(f"  Overall Status: {health['overall_status'].upper()}")
    for tier, model_health in health['models'].items():
        print(f"  {tier}: {model_health['status']}")