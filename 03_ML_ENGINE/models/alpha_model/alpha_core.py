"""
MarketPulse ML Signal Enhancer - Phase 2, Step 1
Integrates existing ML models with technical trading signals

Location: #03_ML_ENGINE/models/ml_signal_enhancer.py

This module acts as the bridge between:
- Existing technical analysis signals (from integrated_trading_system.py)
- Alpha model predictions (alpha_model.py)
- LSTM intraday forecasts (lstm_intraday.py)
- Enhanced confidence scoring for trading decisions
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Any
import warnings

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Setup paths
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import ML models
try:
    from alpha_model import AlphaModel

    print("✅ Alpha Model imported successfully")
except ImportError as e:
    print(f"⚠️ Alpha Model not found: {e}")
    AlphaModel = None

try:
    from lstm_intraday import TimeSeriesForecaster

    print("✅ LSTM Forecaster imported successfully")
except ImportError as e:
    print(f"⚠️ LSTM Forecaster not found: {e}")
    TimeSeriesForecaster = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLSignalEnhancer:
    """
    Enhanced ML Signal Generator

    Combines multiple prediction sources:
    1. Technical analysis signals (existing)
    2. Alpha model factor analysis
    3. LSTM price predictions
    4. Ensemble voting system
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize ML Signal Enhancer"""

        self.config = config or self._default_config()

        # Initialize models
        self.alpha_model = None
        self.lstm_forecaster = None
        self._init_models()

        # Signal weights for ensemble
        self.signal_weights = {
            'technical': 0.4,  # Base technical analysis
            'alpha': 0.3,  # Alpha model factors
            'lstm': 0.3  # LSTM predictions
        }

        # Confidence thresholds
        self.confidence_thresholds = {
            'high': 0.75,
            'medium': 0.6,
            'low': 0.45
        }

        # Performance tracking
        self.prediction_history = []
        self.model_performance = {
            'technical': {'accuracy': 0.0, 'total_predictions': 0},
            'alpha': {'accuracy': 0.0, 'total_predictions': 0},
            'lstm': {'accuracy': 0.0, 'total_predictions': 0},
            'ensemble': {'accuracy': 0.0, 'total_predictions': 0}
        }

        logger.info("🧠 ML Signal Enhancer initialized")

    def _default_config(self) -> Dict:
        """Default configuration for ML models"""
        return {
            'alpha_model': {
                'lookback_period': 252,  # 1 year of trading days
                'min_observations': 60,
                'factor_weights': {
                    'momentum': 0.25,
                    'mean_reversion': 0.25,
                    'volatility': 0.20,
                    'volume': 0.15,
                    'fundamental': 0.15
                }
            },
            'lstm': {
                'sequence_length': 60,
                'prediction_horizon': 5,  # 5 periods ahead
                'confidence_threshold': 0.6
            },
            'ensemble': {
                'min_models_agree': 2,  # At least 2 models must agree
                'confidence_boost': 0.1  # Boost when all models agree
            }
        }

    def _init_models(self):
        """Initialize ML models if available"""

        # Initialize Alpha Model
        if AlphaModel:
            try:
                self.alpha_model = AlphaModel()  # Remove config parameter
                logger.info("✅ Alpha Model initialized")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize Alpha Model: {e}")

        # Initialize LSTM Forecaster
        if TimeSeriesForecaster:
            try:
                self.lstm_forecaster = TimeSeriesForecaster()  # Remove config parameter
                logger.info("✅ LSTM Forecaster initialized")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize LSTM Forecaster: {e}")

    def enhance_signal(self,
                       symbol: str,
                       technical_signal: Dict,
                       market_data: pd.DataFrame) -> Dict:
        """
        Enhance technical signal with ML predictions

        Args:
            symbol: Stock symbol
            technical_signal: Base technical analysis signal
            market_data: Historical market data (OHLCV)

        Returns:
            Enhanced signal with ML predictions and confidence
        """

        logger.info(f"🔍 Enhancing signal for {symbol}")

        # Base signal from technical analysis
        enhanced_signal = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'technical_signal': technical_signal,
            'ml_predictions': {},
            'ensemble_signal': 'HOLD',
            'confidence': 0.0,
            'risk_adjusted_confidence': 0.0,
            'models_used': [],
            'prediction_details': {}
        }

        try:
            # Get Alpha Model prediction
            alpha_prediction = self._get_alpha_prediction(symbol, market_data)
            if alpha_prediction:
                enhanced_signal['ml_predictions']['alpha'] = alpha_prediction
                enhanced_signal['models_used'].append('alpha')

            # Get LSTM prediction
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

    def _get_alpha_prediction(self, symbol: str, market_data: pd.DataFrame) -> Optional[Dict]:
        """Get prediction from Alpha Model"""

        if not self.alpha_model or len(market_data) < self.config['alpha_model']['min_observations']:
            return None

        try:
            # Alpha model expects specific data format
            prediction = self.alpha_model.predict(symbol, market_data)

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
        """Get prediction from LSTM Forecaster"""

        if not self.lstm_forecaster or len(market_data) < self.config['lstm']['sequence_length']:
            return None

        try:
            # LSTM forecaster expects price series
            prediction = self.lstm_forecaster.predict(symbol, market_data)

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

    def _generate_ensemble_signal(self,
                                  technical_signal: Dict,
                                  ml_predictions: Dict) -> Dict:
        """
        Generate ensemble signal from all available predictions

        Uses weighted voting system with confidence adjustment
        """

        signals = []
        confidences = []

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
        """Create detailed prediction summary for analysis"""

        summary = {
            'technical_analysis': {
                'signal': technical_signal.get('signal', 'HOLD'),
                'confidence': technical_signal.get('confidence', 0.5),
                'indicators': technical_signal.get('indicators', {})
            },
            'ml_models': {},
            'consensus': {},
            'divergences': []
        }

        # Add ML model details
        for model_name, prediction in ml_predictions.items():
            summary['ml_models'][model_name] = {
                'signal': prediction.get('signal', 'HOLD'),
                'confidence': prediction.get('confidence', 0.5),
                'key_factors': self._extract_key_factors(prediction)
            }

        # Identify consensus and divergences
        all_signals = [technical_signal.get('signal', 'HOLD')]
        all_signals.extend([p.get('signal', 'HOLD') for p in ml_predictions.values()])

        signal_counts = {s: all_signals.count(s) for s in ['BUY', 'SELL', 'HOLD']}
        summary['consensus'] = {
            'strongest_signal': max(signal_counts, key=signal_counts.get),
            'agreement_pct': max(signal_counts.values()) / len(all_signals) * 100,
            'total_models': len(all_signals)
        }

        # Find divergences
        if len(set(all_signals)) > 1:
            summary['divergences'] = list(set(all_signals))

        return summary

    def _extract_key_factors(self, prediction: Dict) -> Dict:
        """Extract key factors from model prediction"""

        key_factors = {}

        if 'factors' in prediction:
            # Alpha model factors
            factors = prediction['factors']
            key_factors.update(factors)

        if 'predicted_price' in prediction:
            # LSTM price prediction
            key_factors['predicted_price'] = prediction['predicted_price']
            key_factors['price_change_pct'] = prediction.get('price_change_pct', 0.0)

        return key_factors

    def get_signal_strength(self, confidence: float) -> str:
        """Categorize signal strength based on confidence"""

        if confidence >= self.confidence_thresholds['high']:
            return 'HIGH'
        elif confidence >= self.confidence_thresholds['medium']:
            return 'MEDIUM'
        else:
            return 'LOW'

    def update_performance(self,
                           symbol: str,
                           predicted_signal: str,
                           actual_outcome: str,
                           models_used: List[str]):
        """Update model performance tracking"""

        for model in models_used:
            if model in self.model_performance:
                self.model_performance[model]['total_predictions'] += 1

                # Simple accuracy: did signal direction match outcome?
                if predicted_signal == actual_outcome:
                    correct = 1
                else:
                    correct = 0

                current_accuracy = self.model_performance[model]['accuracy']
                total = self.model_performance[model]['total_predictions']

                # Update running accuracy
                new_accuracy = ((current_accuracy * (total - 1)) + correct) / total
                self.model_performance[model]['accuracy'] = new_accuracy

        logger.info(f"📊 Performance updated for {symbol} - Models: {models_used}")

    def get_performance_summary(self) -> Dict:
        """Get performance summary for all models"""

        summary = {
            'model_performance': self.model_performance.copy(),
            'total_predictions': sum(m['total_predictions'] for m in self.model_performance.values()),
            'average_accuracy': np.mean([m['accuracy'] for m in self.model_performance.values()]),
            'models_available': len([m for m in [self.alpha_model, self.lstm_forecaster] if m is not None])
        }

        return summary


def test_ml_signal_enhancer():
    """Test the ML Signal Enhancer"""

    print("\n=== Testing ML Signal Enhancer ===\n")

    # Initialize enhancer
    enhancer = MLSignalEnhancer()

    # Create sample technical signal
    technical_signal = {
        'signal': 'BUY',
        'confidence': 0.65,
        'indicators': {
            'rsi': 35.5,
            'macd_signal': 'bullish',
            'bb_position': 'lower'
        }
    }

    # Create sample market data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    prices = 100 + np.cumsum(np.random.randn(100) * 0.02)

    market_data = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.randn(100) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(100)) * 0.005),
        'low': prices * (1 - np.abs(np.random.randn(100)) * 0.005),
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 100)
    })

    # Test signal enhancement
    symbol = 'RELIANCE'
    enhanced_signal = enhancer.enhance_signal(symbol, technical_signal, market_data)

    # Display results
    print(f"Symbol: {enhanced_signal['symbol']}")
    print(f"Original Signal: {technical_signal['signal']} (Confidence: {technical_signal['confidence']:.2%})")
    print(f"Enhanced Signal: {enhanced_signal['ensemble_signal']} (Confidence: {enhanced_signal['confidence']:.2%})")
    print(f"Models Used: {enhanced_signal['models_used']}")
    print(f"Signal Strength: {enhancer.get_signal_strength(enhanced_signal['confidence'])}")

    # Show prediction details
    if enhanced_signal['prediction_details']:
        details = enhanced_signal['prediction_details']
        print(f"\nConsensus: {details['consensus']['strongest_signal']} "
              f"({details['consensus']['agreement_pct']:.1f}% agreement)")

        if details['divergences']:
            print(f"Divergences: {details['divergences']}")

    # Test performance tracking
    enhancer.update_performance(symbol, 'BUY', 'BUY', enhanced_signal['models_used'])
    performance = enhancer.get_performance_summary()
    print(f"\nPerformance Summary:")
    print(f"  Average Accuracy: {performance['average_accuracy']:.2%}")
    print(f"  Total Predictions: {performance['total_predictions']}")
    print(f"  Models Available: {performance['models_available']}")

    print("\n✅ ML Signal Enhancer Test Complete!")


if __name__ == "__main__":
    test_ml_signal_enhancer()