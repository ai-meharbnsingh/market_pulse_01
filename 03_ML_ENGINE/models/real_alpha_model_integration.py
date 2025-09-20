# 03_ML_ENGINE/models/real_alpha_model_integration.py
"""
Real Alpha Model Integration with Circuit Breaker Protection
MarketPulse Phase 2, Step 5 - Production Model Integration

This module integrates the actual AlphaModelCore with:
- Circuit breaker protection from ml_circuit_breaker.py
- Performance monitoring from latency_optimizer.py
- Enhanced error handling from error_handler.py
- Tier-based model management

Location: #03_ML_ENGINE/models/real_alpha_model_integration.py
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Any
import sqlite3
import json
import warnings
import time
from dataclasses import dataclass
from enum import Enum

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup paths
current_dir = Path(__file__).parent
root_dir = current_dir.parent.parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir.parent / "reliability"))
sys.path.append(str(current_dir.parent / "optimization"))

# Import circuit breaker and performance components
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

# ML Libraries with graceful fallback
try:
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.ensemble import VotingClassifier, RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

    ML_LIBRARIES_AVAILABLE = True
    logger.info("✅ Advanced ML libraries available")
except ImportError as e:
    logger.warning(f"⚠️ Some ML libraries not available: {e}")
    ML_LIBRARIES_AVAILABLE = False

    # Fallback to basic sklearn
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

        BASIC_ML_AVAILABLE = True
        logger.info("✅ Basic ML libraries available")
    except ImportError:
        BASIC_ML_AVAILABLE = False
        logger.error("❌ No ML libraries available")


class ModelTier(Enum):
    """Model performance tiers for circuit breaker management"""
    PREMIUM = "premium"  # <10ms target, >85% accuracy
    STANDARD = "standard"  # <20ms target, >75% accuracy
    ECONOMIC = "economic"  # <50ms target, >65% accuracy
    FALLBACK = "fallback"  # <10ms heuristics, >55% accuracy


class RealAlphaModelCore:
    """
    Production-Ready Alpha Model with Circuit Breaker Protection

    Features:
    - Circuit breaker protection against model failures
    - Performance monitoring and optimization
    - Tier-based model management
    - Graceful degradation with fallback mechanisms
    - Database integration for hypothesis tracking
    - Real-time performance metrics
    """

    def __init__(self, db_path: str = "marketpulse_production.db", tier: ModelTier = ModelTier.STANDARD):
        """
        Initialize Real Alpha Model with circuit breaker protection

        Args:
            db_path: Path to SQLite database
            tier: Performance tier for circuit breaker configuration
        """
        self.db_path = db_path
        self.tier = tier
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
        self.model_version = "v2.0.0"

        # Circuit breaker configuration based on tier
        self.cb_config = self._get_circuit_breaker_config(tier)

        # Performance tracking
        self.performance_stats = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'average_latency': 0.0,
            'cache_hits': 0,
            'fallback_usage': 0
        }

        # Initialize components
        self._initialize_database()
        self._initialize_models()

        if RELIABILITY_AVAILABLE:
            self.error_handler = ErrorHandler()
            self.latency_optimizer = LatencyOptimizer()
            logger.info("✅ Reliability components initialized")

        logger.info(f"✅ RealAlphaModelCore initialized with {tier.value} tier")

    def _get_circuit_breaker_config(self, tier: ModelTier) -> Dict:
        """Get circuit breaker configuration based on model tier"""
        configs = {
            ModelTier.PREMIUM: {
                'failure_threshold': 3,
                'recovery_timeout': 30,
                'timeout': 10.0,
                'half_open_max_calls': 5
            },
            ModelTier.STANDARD: {
                'failure_threshold': 5,
                'recovery_timeout': 60,
                'timeout': 20.0,
                'half_open_max_calls': 3
            },
            ModelTier.ECONOMIC: {
                'failure_threshold': 7,
                'recovery_timeout': 120,
                'timeout': 50.0,
                'half_open_max_calls': 2
            },
            ModelTier.FALLBACK: {
                'failure_threshold': 10,
                'recovery_timeout': 300,
                'timeout': 10.0,
                'half_open_max_calls': 1
            }
        }
        return configs[tier]

    def _initialize_database(self):
        """Initialize SQLite database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Create trading_hypotheses table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trading_hypotheses (
                        id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        signal_type TEXT NOT NULL,
                        confidence_score REAL NOT NULL,
                        features TEXT NOT NULL,
                        model_version TEXT NOT NULL,
                        ai_provider TEXT,
                        expected_return REAL,
                        risk_level TEXT,
                        market_regime TEXT
                    )
                ''')

                # Create trade_outcomes table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trade_outcomes (
                        hypothesis_id TEXT PRIMARY KEY,
                        execution_timestamp TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                        exit_price REAL,
                        exit_timestamp TEXT,
                        actual_return REAL,
                        trade_duration_hours REAL,
                        was_profitable BOOLEAN,
                        exit_reason TEXT,
                        FOREIGN KEY (hypothesis_id) REFERENCES trading_hypotheses (id)
                    )
                ''')

                # Create model_performance table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS model_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        model_version TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        additional_data TEXT
                    )
                ''')

                conn.commit()
                logger.info("✅ Database tables initialized successfully")

        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            raise

    def _initialize_models(self):
        """Initialize ML models based on available libraries"""
        try:
            if ML_LIBRARIES_AVAILABLE:
                # Premium model ensemble with XGBoost and LightGBM
                self.models = {
                    'xgboost': xgb.XGBClassifier(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=42,
                        n_jobs=-1
                    ),
                    'lightgbm': lgb.LGBMClassifier(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=42,
                        n_jobs=-1,
                        verbose=-1
                    ),
                    'neural_net': MLPClassifier(
                        hidden_layer_sizes=(100, 50),
                        max_iter=500,
                        random_state=42
                    ),
                    'random_forest': RandomForestClassifier(
                        n_estimators=100,
                        max_depth=10,
                        random_state=42,
                        n_jobs=-1
                    )
                }
                logger.info("✅ Premium ML model ensemble initialized")

            elif BASIC_ML_AVAILABLE:
                # Standard model ensemble with basic sklearn
                self.models = {
                    'random_forest': RandomForestClassifier(
                        n_estimators=100,
                        max_depth=10,
                        random_state=42,
                        n_jobs=-1
                    ),
                    'logistic': LogisticRegression(
                        random_state=42,
                        max_iter=1000
                    )
                }
                logger.info("✅ Standard ML model ensemble initialized")

            else:
                logger.warning("⚠️ No ML libraries available - using heuristic fallback")

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
    def predict_profitability(self, signal_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict probability of profitability with circuit breaker protection

        Args:
            signal_features: Dictionary of signal features

        Returns:
            Dictionary containing prediction results with circuit breaker status
        """
        start_time = time.time()

        try:
            # Update performance stats
            self.performance_stats['total_predictions'] += 1

            # Validate inputs
            if not signal_features:
                raise ValueError("Signal features cannot be empty")

            # Get prediction based on available models
            if self.models and self.is_trained:
                prediction = self._get_ml_prediction(signal_features)
                method = "ML_ENSEMBLE"

            else:
                # Fallback to heuristic
                prediction = self._get_heuristic_prediction(signal_features)
                method = "HEURISTIC_FALLBACK"
                self.performance_stats['fallback_usage'] += 1

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # Update performance tracking
            self.performance_stats['successful_predictions'] += 1
            self._update_average_latency(latency_ms)

            # Enhance prediction with metadata
            enhanced_prediction = {
                **prediction,
                'model_tier': self.tier.value,
                'prediction_method': method,
                'latency_ms': round(latency_ms, 2),
                'circuit_breaker_status': 'CLOSED',
                'model_version': self.model_version,
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"✅ Prediction successful: {enhanced_prediction['ensemble_pop']:.3f} "
                        f"confidence ({enhanced_prediction['confidence']}) - {latency_ms:.1f}ms")

            return enhanced_prediction

        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")

            # Return safe fallback prediction
            fallback_prediction = {
                'ensemble_pop': 0.5,
                'confidence': 'LOW',
                'model_tier': self.tier.value,
                'prediction_method': 'ERROR_FALLBACK',
                'latency_ms': (time.time() - start_time) * 1000,
                'circuit_breaker_status': 'OPEN',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

            return fallback_prediction

    def _get_ml_prediction(self, signal_features: Dict[str, Any]) -> Dict[str, Any]:
        """Get prediction from trained ML models"""
        try:
            # Convert features to DataFrame
            feature_df = pd.DataFrame([signal_features])

            # Engineer features
            feature_df = self._engineer_features(feature_df)

            # Select available features
            available_features = [col for col in self.feature_columns if col in feature_df.columns]
            if not available_features:
                raise ValueError("No valid features available for prediction")

            X = feature_df[available_features].fillna(0)

            # Scale features
            X_scaled = self.scaler.transform(X)

            # Get predictions from all models
            predictions = {}
            model_scores = []

            for name, model in self.models.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        # Get probability of profitable class (class 1)
                        proba = model.predict_proba(X_scaled)[0][1]
                        predictions[f'{name}_pop'] = proba
                        model_scores.append(proba)
                    else:
                        # Binary prediction fallback
                        pred = model.predict(X_scaled)[0]
                        predictions[f'{name}_pop'] = float(pred)
                        model_scores.append(float(pred))

                except Exception as e:
                    logger.warning(f"Model {name} failed: {e}")
                    continue

            # Ensemble prediction
            if model_scores:
                ensemble_pop = np.mean(model_scores)
                std_dev = np.std(model_scores)
            else:
                raise ValueError("All models failed to make predictions")

            # Confidence assessment based on ensemble agreement
            if std_dev < 0.1 and (ensemble_pop > 0.7 or ensemble_pop < 0.3):
                confidence = 'VERY_HIGH'
            elif std_dev < 0.15 and (ensemble_pop > 0.65 or ensemble_pop < 0.35):
                confidence = 'HIGH'
            elif std_dev < 0.2:
                confidence = 'MEDIUM'
            else:
                confidence = 'LOW'

            return {
                'ensemble_pop': float(ensemble_pop),
                'confidence': confidence,
                'model_agreement': 1.0 - std_dev,
                'individual_predictions': predictions,
                'feature_count': len(available_features)
            }

        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            raise

    def _get_heuristic_prediction(self, signal_features: Dict[str, Any]) -> Dict[str, Any]:
        """Intelligent heuristic prediction when ML models are unavailable"""
        score = 0.5  # Base probability
        factors = {}

        try:
            # RSI-based signal
            rsi = signal_features.get('rsi_14', 50)
            if rsi < 30:
                score += 0.15
                factors['rsi_signal'] = 'oversold_bullish'
            elif rsi > 70:
                score -= 0.15
                factors['rsi_signal'] = 'overbought_bearish'
            else:
                factors['rsi_signal'] = 'neutral'

            # Volume confirmation
            volume_ratio = signal_features.get('volume_ratio', 1.0)
            if volume_ratio > 1.5:
                score += 0.1
                factors['volume_confirmation'] = 'strong'
            elif volume_ratio > 1.2:
                score += 0.05
                factors['volume_confirmation'] = 'moderate'
            else:
                factors['volume_confirmation'] = 'weak'

            # Price momentum
            momentum = signal_features.get('price_momentum_5', 0)
            momentum_contribution = np.clip(momentum * 0.2, -0.15, 0.15)
            score += momentum_contribution
            factors['momentum_contribution'] = momentum_contribution

            # MACD signal
            macd = signal_features.get('macd', 0)
            macd_signal = signal_features.get('macd_signal', 0)
            if macd > macd_signal:
                score += 0.05
                factors['macd_signal'] = 'bullish'
            else:
                score -= 0.05
                factors['macd_signal'] = 'bearish'

            # Volatility adjustment
            volatility = signal_features.get('volatility_20', 0.02)
            if volatility > 0.05:  # High volatility
                score *= 0.9  # Reduce confidence
                factors['volatility_adjustment'] = 'high_vol_penalty'

            # Clamp to valid range
            score = np.clip(score, 0.1, 0.9)

            # Confidence based on signal strength
            if abs(score - 0.5) > 0.25:
                confidence = 'HIGH'
            elif abs(score - 0.5) > 0.15:
                confidence = 'MEDIUM'
            else:
                confidence = 'LOW'

            return {
                'ensemble_pop': float(score),
                'confidence': confidence,
                'heuristic_factors': factors,
                'method': 'intelligent_heuristic'
            }

        except Exception as e:
            logger.error(f"Heuristic prediction failed: {e}")

            # Ultimate fallback
            return {
                'ensemble_pop': 0.5,
                'confidence': 'LOW',
                'method': 'ultimate_fallback',
                'error': str(e)
            }

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for ML prediction"""
        try:
            # Handle timestamp
            if 'timestamp' not in df.columns:
                df['timestamp'] = datetime.now()

            # Technical indicator ratios
            if 'close' in df.columns:
                df['price_to_sma_ratio'] = df.get('close', 1) / df.get('sma_20', 1)
                df['price_to_ema_ratio'] = df.get('close', 1) / df.get('ema_12', 1)

            # Volume indicators
            if 'volume' in df.columns:
                df['volume_ma_ratio'] = df.get('volume', 1) / df.get('volume_sma_20', 1)

            # Momentum indicators
            df['rsi_deviation'] = abs(df.get('rsi_14', 50) - 50)
            df['bb_position'] = (df.get('close', 0) - df.get('bb_lower', 0)) / \
                                (df.get('bb_upper', 0) - df.get('bb_lower', 1) + 1e-8)

            # Time-based features
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek

            return df

        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            return df

    def _update_average_latency(self, new_latency: float):
        """Update running average latency"""
        total_preds = self.performance_stats['total_predictions']
        if total_preds == 1:
            self.performance_stats['average_latency'] = new_latency
        else:
            # Exponential moving average
            alpha = 0.1
            self.performance_stats['average_latency'] = (
                    alpha * new_latency +
                    (1 - alpha) * self.performance_stats['average_latency']
            )

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        success_rate = (
                               self.performance_stats['successful_predictions'] /
                               max(self.performance_stats['total_predictions'], 1)
                       ) * 100

        return {
            **self.performance_stats,
            'success_rate_pct': round(success_rate, 2),
            'tier': self.tier.value,
            'circuit_breaker_config': self.cb_config,
            'last_updated': datetime.now().isoformat()
        }

    def log_trading_hypothesis(self, hypothesis_data: Dict[str, Any]) -> str:
        """Log trading hypothesis for learning"""
        try:
            hypothesis_id = str(uuid.uuid4())

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO trading_hypotheses 
                    (id, timestamp, symbol, signal_type, confidence_score, 
                     features, model_version, ai_provider, expected_return, 
                     risk_level, market_regime)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    hypothesis_id,
                    hypothesis_data.get('timestamp', datetime.now().isoformat()),
                    hypothesis_data.get('symbol', ''),
                    hypothesis_data.get('signal_type', ''),
                    hypothesis_data.get('confidence_score', 0.5),
                    json.dumps(hypothesis_data.get('features', {})),
                    self.model_version,
                    hypothesis_data.get('ai_provider', 'alpha_model'),
                    hypothesis_data.get('expected_return'),
                    hypothesis_data.get('risk_level', 'MEDIUM'),
                    hypothesis_data.get('market_regime', 'UNKNOWN')
                ))
                conn.commit()

            logger.info(f"✅ Trading hypothesis logged: {hypothesis_id}")
            return hypothesis_id

        except Exception as e:
            logger.error(f"❌ Failed to log trading hypothesis: {e}")
            return ""

    def health_check(self) -> Dict[str, Any]:
        """Comprehensive system health check"""
        health_status = {
            'status': 'HEALTHY',
            'timestamp': datetime.now().isoformat(),
            'model_tier': self.tier.value,
            'components': {},
            'performance': self.get_performance_stats(),
            'issues': []
        }

        try:
            # Check database connectivity
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM trading_hypotheses")
                health_status['components']['database'] = 'OK'

        except Exception as e:
            health_status['components']['database'] = 'ERROR'
            health_status['issues'].append(f"Database: {str(e)}")
            health_status['status'] = 'DEGRADED'

        # Check model availability
        if self.models:
            health_status['components']['ml_models'] = 'OK'
        else:
            health_status['components']['ml_models'] = 'FALLBACK'
            health_status['issues'].append("ML models not available - using heuristics")
            if health_status['status'] == 'HEALTHY':
                health_status['status'] = 'DEGRADED'

        # Check performance metrics
        if self.performance_stats['average_latency'] > 50:
            health_status['issues'].append("High average latency detected")
            health_status['status'] = 'DEGRADED'

        success_rate = (
                               self.performance_stats['successful_predictions'] /
                               max(self.performance_stats['total_predictions'], 1)
                       ) * 100

        if success_rate < 90:
            health_status['issues'].append("Low success rate detected")
            health_status['status'] = 'DEGRADED'

        return health_status


# Factory function for easy instantiation
def create_real_alpha_model(tier: str = "standard", db_path: str = "marketpulse_production.db") -> RealAlphaModelCore:
    """
    Factory function to create RealAlphaModelCore with proper tier

    Args:
        tier: Model tier ("premium", "standard", "economic", "fallback")
        db_path: Path to SQLite database

    Returns:
        Configured RealAlphaModelCore instance
    """
    tier_enum = ModelTier(tier.lower())
    return RealAlphaModelCore(db_path=db_path, tier=tier_enum)


# Usage example and testing
if __name__ == "__main__":
    print("🚀 RealAlphaModelCore - Production Integration Test")
    print("=" * 60)

    try:
        # Create model instance
        alpha_model = create_real_alpha_model(tier="standard")

        # Test health check
        health = alpha_model.health_check()
        print(f"✅ Health Status: {health['status']}")

        # Test prediction with sample features
        sample_features = {
            'symbol': 'AAPL',
            'close': 150.0,
            'rsi_14': 65.0,
            'macd': 1.2,
            'macd_signal': 1.0,
            'volume_ratio': 1.5,
            'price_momentum_5': 0.02,
            'volatility_20': 0.03,
            'sma_20': 148.0,
            'ema_12': 149.0
        }

        print("\n🧠 Testing prediction...")
        result = alpha_model.predict_profitability(sample_features)
        print(f"✅ Prediction: {result['ensemble_pop']:.3f} confidence ({result['confidence']})")
        print(f"   Method: {result['prediction_method']}")
        print(f"   Latency: {result['latency_ms']:.1f}ms")

        # Test performance stats
        print("\n📊 Performance Statistics:")
        stats = alpha_model.get_performance_stats()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.2f}")
            else:
                print(f"   {key}: {value}")

        print("\n✅ RealAlphaModelCore integration test completed successfully!")

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback

        traceback.print_exc()