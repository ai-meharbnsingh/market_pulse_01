"""
MarketPulse ML Signal Enhancer - Phase 2, Step 1
Integrates existing ML models with technical trading signals
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

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MockAlphaModel:
    def predict_profitability(self, symbol, market_data):
        return {
            'signal': 'BUY' if np.random.random() > 0.5 else 'SELL',
            'confidence': np.random.uniform(0.5, 0.9),
            'factors': {'momentum': np.random.uniform(-1, 1)},
            'alpha_score': np.random.uniform(0, 1)
        }

class MockLSTMModel:
    def predict_profitability(self, symbol, market_data):
        return {
            'signal': 'BUY' if np.random.random() > 0.5 else 'SELL',
            'confidence': np.random.uniform(0.5, 0.9),
            'predicted_price': market_data['close'].iloc[-1] * np.random.uniform(0.9, 1.1),
            'price_change_pct': np.random.uniform(-0.05, 0.05),
            'horizon': 5
        }

class MLSignalEnhancer:
    def __init__(self, config=None):
        """
        Initialize ML Signal Enhancer with configurable parameters
        """
        # Default configuration with enhanced ensemble settings
        self.config = {
            'ensemble': {
                'min_models_agree': 2,
                'confidence_boost': 0.15,
                'risk_threshold': 0.6
            },
            'alpha_model': {
                'min_observations': 50,
                'default_confidence': 0.5
            },
            'lstm': {
                'sequence_length': 30,
                'forecast_horizon': 5
            }
        }

        # Override default config if custom config provided
        if config:
            self._update_config(config)

        # Initialize models and performance tracking
        self.alpha_model = MockAlphaModel()
        self.lstm_forecaster = MockLSTMModel()
        self.model_performance = {}

        # Signal weighting strategy
        self.signal_weights = {
            'technical': 0.3,
            'alpha_model': 0.4,
            'lstm_forecaster': 0.3
        }

    def _update_config(self, custom_config):
        """
        Update configuration with custom settings
        """
        for key, value in custom_config.items():
            if key in self.config:
                self.config[key].update(value)
            else:
                self.config[key] = value

    def _get_alpha_prediction(self, symbol: str, market_data: pd.DataFrame) -> Optional[Dict]:
        """
        Get prediction from Alpha Model
        """
        if not self.alpha_model or len(market_data) < self.config['alpha_model']['min_observations']:
            return None

        try:
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
            'average_accuracy': np.mean([m['accuracy'] for m in self.model_performance.values()]) if self.model_performance else 0.0,
            'models_available': len(self.model_performance)
        }

        return summary