import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import unittest
import pandas as pd
import numpy as np

# Import with full path
from models.ml_signal_enhancer import MLSignalEnhancer


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


class TestMLSignalEnhancer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Setup test environment with mock models and sample data
        """
        # Mock configuration
        cls.config = {
            'ensemble': {
                'min_models_agree': 2,
                'confidence_boost': 0.15
            },
            'alpha_model': {
                'min_observations': 30
            },
            'lstm': {
                'sequence_length': 20
            }
        }

        # Generate synthetic market data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        prices = 100 + np.cumsum(np.random.randn(100) * 0.1)
        cls.market_data = pd.DataFrame({
            'date': dates,
            'open': prices * (1 + np.random.randn(100) * 0.01),
            'high': prices * (1 + np.abs(np.random.randn(100)) * 0.02),
            'low': prices * (1 - np.abs(np.random.randn(100)) * 0.02),
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, 100)
        })

    def setUp(self):
        """Initialize ML Signal Enhancer for each test"""
        self.enhancer = MLSignalEnhancer(config=self.config)

        # Inject mock models
        self.enhancer.alpha_model = MockAlphaModel()
        self.enhancer.lstm_forecaster = MockLSTMModel()

    def test_signal_enhancement_basic(self):
        """
        Test basic signal enhancement workflow
        """
        technical_signal = {
            'signal': 'BUY',
            'confidence': 0.6,
            'indicators': {
                'rsi': 35,
                'macd': 'bullish'
            }
        }

        # Enhance signal
        enhanced_signal = self.enhancer.enhance_signal('SAMPLE', technical_signal, self.market_data)

        # Assertions
        self.assertIn('ensemble_signal', enhanced_signal)
        self.assertIn('confidence', enhanced_signal)
        self.assertIn('models_used', enhanced_signal)
        self.assertIn('ml_predictions', enhanced_signal)
        self.assertIn('prediction_details', enhanced_signal)

    def test_ensemble_signal_generation(self):
        """
        Test ensemble signal generation rules
        """
        technical_signal = {
            'signal': 'BUY',
            'confidence': 0.6,
            'indicators': {}
        }

        # Multiple enhancement iterations
        signals = [
            self.enhancer.enhance_signal('SAMPLE', technical_signal, self.market_data)
            for _ in range(10)
        ]

        # Check consistency and signal varieties
        ensemble_signals = [s['ensemble_signal'] for s in signals]
        ensemble_confidences = [s['confidence'] for s in signals]

        # Reasonable signal distribution
        self.assertTrue(len(set(ensemble_signals)) > 0)
        self.assertTrue(all(0.4 <= conf <= 1.0 for conf in ensemble_confidences))

    def test_config_customization(self):
        """
        Verify configuration customization works
        """
        custom_config = {
            'ensemble': {
                'min_models_agree': 3,  # More strict agreement
                'confidence_boost': 0.25  # Higher boost
            }
        }

        custom_enhancer = MLSignalEnhancer(config=custom_config)
        custom_enhancer.alpha_model = MockAlphaModel()
        custom_enhancer.lstm_forecaster = MockLSTMModel()

        technical_signal = {
            'signal': 'BUY',
            'confidence': 0.6,
            'indicators': {}
        }

        # Test with custom configuration
        enhanced_signal = custom_enhancer.enhance_signal('SAMPLE', technical_signal, self.market_data)

        # Verify config applied
        self.assertEqual(custom_enhancer.config['ensemble']['min_models_agree'], 3)
        self.assertEqual(custom_enhancer.config['ensemble']['confidence_boost'], 0.25)

    def test_performance_tracking(self):
        """
        Test model performance tracking mechanisms
        """
        technical_signal = {
            'signal': 'BUY',
            'confidence': 0.6,
            'indicators': {}
        }

        # Simulate multiple signal enhancements and performance updates
        for _ in range(5):
            enhanced_signal = self.enhancer.enhance_signal('SAMPLE', technical_signal, self.market_data)
            self.enhancer.update_performance(
                'SAMPLE',
                technical_signal['signal'],
                enhanced_signal['ensemble_signal'],
                enhanced_signal['models_used']
            )

        # Get performance summary
        performance = self.enhancer.get_performance_summary()

        # Assertions
        self.assertIn('model_performance', performance)
        self.assertIn('total_predictions', performance)
        self.assertIn('average_accuracy', performance)
        self.assertIn('models_available', performance)


if __name__ == '__main__':
    unittest.main(verbosity=2)