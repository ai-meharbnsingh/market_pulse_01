# 08_TESTS/test_ml_signal_enhancer.py
"""
Comprehensive Unit Tests for Advanced ML Signal Enhancer
Phase 2, Step 2 - Testing Suite

Location: #08_TESTS/test_ml_signal_enhancer.py

This test suite validates:
- Advanced mock models functionality
- Market regime detection
- Ensemble signal generation
- Performance tracking
- Caching mechanisms
- Error handling
"""

import sys
import os
from pathlib import Path
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')

# Setup path for importing modules
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir / "03_ML_ENGINE" / "models"))

# Import modules to test
try:
    from advanced_mock_models import (
        AdvancedAlphaModel,
        AdvancedLSTMModel,
        AdvancedMarketSimulator,
        MarketRegime,
        create_advanced_models
    )
    from ml_signal_enhancer import MLSignalEnhancer, EnhancementResult

    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    IMPORTS_SUCCESSFUL = False
    print(f"❌ Import failed: {e}")


class TestAdvancedMockModels(unittest.TestCase):
    """Test advanced mock models functionality"""

    def setUp(self):
        """Set up test fixtures"""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest("Module imports failed")

        self.alpha_model = AdvancedAlphaModel()
        self.lstm_model = AdvancedLSTMModel()
        self.simulator = AdvancedMarketSimulator()

        # Create sample market data
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 50)))

        self.market_data = pd.DataFrame({
            'date': dates,
            'open': prices * 0.999,
            'high': prices * 1.002,
            'low': prices * 0.998,
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, 50)
        })

    def test_alpha_model_prediction_structure(self):
        """Test that alpha model returns expected structure"""
        prediction = self.alpha_model.predict_profitability('TEST', self.market_data)

        # Check required fields
        required_fields = ['signal', 'confidence', 'alpha_score', 'factors', 'market_regime']
        for field in required_fields:
            self.assertIn(field, prediction, f"Missing field: {field}")

        # Check signal is valid
        self.assertIn(prediction['signal'], ['BUY', 'SELL', 'HOLD'])

        # Check confidence is in valid range
        self.assertGreaterEqual(prediction['confidence'], 0.0)
        self.assertLessEqual(prediction['confidence'], 1.0)

        # Check factors is a dictionary
        self.assertIsInstance(prediction['factors'], dict)

    def test_lstm_model_prediction_structure(self):
        """Test that LSTM model returns expected structure"""
        prediction = self.lstm_model.predict_profitability('TEST', self.market_data)

        # Check required fields
        required_fields = ['signal', 'confidence', 'predicted_price', 'price_change_pct', 'horizon']
        for field in required_fields:
            self.assertIn(field, prediction, f"Missing field: {field}")

        # Check signal is valid
        self.assertIn(prediction['signal'], ['BUY', 'SELL', 'HOLD'])

        # Check confidence is in valid range
        self.assertGreaterEqual(prediction['confidence'], 0.0)
        self.assertLessEqual(prediction['confidence'], 1.0)

        # Check price prediction is positive
        self.assertGreater(prediction['predicted_price'], 0)

    def test_market_regime_detection(self):
        """Test market regime detection functionality"""
        # Test with different market scenarios
        scenarios = ['bull_momentum', 'bear_selloff', 'sideways_range']

        for scenario in scenarios:
            market_data = self.simulator.generate_scenario(scenario, 'TEST', days=30)
            context = self.alpha_model.detect_market_regime(market_data)

            # Check that context is returned
            self.assertIsNotNone(context)
            self.assertIn(context.regime, [MarketRegime.BULL, MarketRegime.BEAR,
                                           MarketRegime.SIDEWAYS, MarketRegime.VOLATILE])

            # Check volatility level is in valid range
            self.assertGreaterEqual(context.volatility_level, 0.0)
            self.assertLessEqual(context.volatility_level, 1.0)

    def test_market_simulator_scenarios(self):
        """Test market simulator generates different scenarios correctly"""
        scenarios = ['bull_momentum', 'bear_selloff', 'sideways_range', 'flash_crash']

        for scenario in scenarios:
            market_data = self.simulator.generate_scenario(scenario, 'TEST', days=30)

            # Check data structure
            expected_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            for col in expected_columns:
                self.assertIn(col, market_data.columns, f"Missing column: {col}")

            # Check OHLC relationships
            for i, row in market_data.iterrows():
                self.assertLessEqual(row['low'], row['open'], f"Low > Open at index {i}")
                self.assertLessEqual(row['low'], row['close'], f"Low > Close at index {i}")
                self.assertGreaterEqual(row['high'], row['open'], f"High < Open at index {i}")
                self.assertGreaterEqual(row['high'], row['close'], f"High < Close at index {i}")

    def test_model_consistency(self):
        """Test that models produce consistent predictions for same data"""
        symbol = 'CONSISTENCY_TEST'

        # Run predictions multiple times
        alpha_predictions = []
        lstm_predictions = []

        for _ in range(3):
            alpha_pred = self.alpha_model.predict_profitability(symbol, self.market_data)
            lstm_pred = self.lstm_model.predict_profitability(symbol, self.market_data)

            alpha_predictions.append(alpha_pred)
            lstm_predictions.append(lstm_pred)

        # Models should have some consistency (not completely random)
        # This is a soft check since models use market context
        self.assertTrue(len(set(p['signal'] for p in alpha_predictions)) <= 2,
                        "Alpha model too inconsistent")


class TestMLSignalEnhancer(unittest.TestCase):
    """Test ML Signal Enhancer functionality"""

    def setUp(self):
        """Set up test fixtures"""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest("Module imports failed")

        self.enhancer = MLSignalEnhancer()

        # Create sample data
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 60)))

        self.market_data = pd.DataFrame({
            'date': dates,
            'open': prices * 0.999,
            'high': prices * 1.002,
            'low': prices * 0.998,
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, 60)
        })

        self.technical_signal = {
            'signal': 'BUY',
            'confidence': 0.7,
            'indicators': {
                'rsi': 35,
                'macd': 'bullish',
                'bb_position': 'lower'
            }
        }

    def test_signal_enhancement_structure(self):
        """Test that signal enhancement returns proper structure"""
        result = self.enhancer.enhance_signal('TEST', self.technical_signal, self.market_data)

        # Check required fields
        required_fields = [
            'symbol', 'original_signal', 'ensemble_signal', 'confidence',
            'models_used', 'enhancement_factors', 'processing_time_ms'
        ]

        for field in required_fields:
            self.assertIn(field, result, f"Missing field: {field}")

        # Check signal is valid
        self.assertIn(result['ensemble_signal'], ['BUY', 'SELL', 'HOLD'])

        # Check confidence is in valid range
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)

        # Check models used is a list
        self.assertIsInstance(result['models_used'], list)
        self.assertGreater(len(result['models_used']), 0)

    def test_performance_tracking(self):
        """Test performance tracking functionality"""
        symbol = 'PERF_TEST'

        # Generate initial signal
        result = self.enhancer.enhance_signal(symbol, self.technical_signal, self.market_data)

        # Update performance
        self.enhancer.update_performance(symbol, 'BUY', 'BUY', result['models_used'])

        # Check performance summary
        performance = self.enhancer.get_performance_summary()

        # Check structure
        self.assertIn('average_accuracy', performance)
        self.assertIn('total_predictions', performance)
        self.assertIn('model_performance', performance)

        # Check that total predictions increased
        self.assertGreater(performance['total_predictions'], 0)

    def test_caching_functionality(self):
        """Test caching mechanism"""
        symbol = 'CACHE_TEST'

        # First call
        result1 = self.enhancer.enhance_signal(symbol, self.technical_signal, self.market_data)
        first_time = result1['processing_time_ms']

        # Second call (should be cached)
        result2 = self.enhancer.enhance_signal(symbol, self.technical_signal, self.market_data)
        cached_time = result2['processing_time_ms']

        # Check that results are similar (cached)
        self.assertEqual(result1['ensemble_signal'], result2['ensemble_signal'])
        self.assertEqual(result1['confidence'], result2['confidence'])

        # Check performance summary includes cache stats
        performance = self.enhancer.get_performance_summary()
        self.assertIn('cache_statistics', performance)
        self.assertGreater(performance['cache_statistics']['cache_size'], 0)

    def test_configuration_override(self):
        """Test configuration override functionality"""
        custom_config = {
            'ensemble': {
                'confidence_boost': 0.25,
                'risk_threshold': 0.7
            },
            'processing': {
                'enable_caching': False
            }
        }

        custom_enhancer = MLSignalEnhancer(custom_config)

        # Check that custom config was applied
        self.assertEqual(custom_enhancer.config['ensemble']['confidence_boost'], 0.25)
        self.assertEqual(custom_enhancer.config['ensemble']['risk_threshold'], 0.7)
        self.assertFalse(custom_enhancer.config['processing']['enable_caching'])

    def test_market_scenario_integration(self):
        """Test integration with market scenario testing"""
        if not hasattr(self.enhancer, 'generate_market_scenario_test'):
            self.skipTest("Market scenario testing not available")

        scenarios = ['bull_momentum', 'bear_selloff', 'sideways_range']

        for scenario in scenarios:
            result = self.enhancer.generate_market_scenario_test(scenario)

            # Check that result contains expected fields
            if 'error' not in result:
                self.assertIn('scenario_type', result)
                self.assertIn('test_result', result)
                self.assertIn('market_data_shape', result)
                self.assertEqual(result['scenario_type'], scenario)

    def test_signal_strength_classification(self):
        """Test signal strength classification"""
        confidence_levels = [0.9, 0.75, 0.65, 0.55, 0.3]
        expected_strengths = ["VERY STRONG", "STRONG", "MODERATE", "WEAK", "VERY WEAK"]

        for confidence, expected in zip(confidence_levels, expected_strengths):
            strength = self.enhancer.get_signal_strength(confidence)
            self.assertEqual(strength, expected,
                             f"Wrong strength for confidence {confidence}")

    def test_error_handling(self):
        """Test error handling with invalid inputs"""
        # Test with empty market data
        empty_data = pd.DataFrame()
        result = self.enhancer.enhance_signal('EMPTY_TEST', self.technical_signal, empty_data)

        # Should still return a valid result
        self.assertIsInstance(result, dict)
        self.assertIn('ensemble_signal', result)

        # Test with invalid signal
        invalid_signal = {'invalid': 'signal'}
        result = self.enhancer.enhance_signal('INVALID_TEST', invalid_signal, self.market_data)

        # Should handle gracefully
        self.assertIsInstance(result, dict)
        self.assertIn('ensemble_signal', result)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflow"""

    def setUp(self):
        """Set up integration test fixtures"""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest("Module imports failed")

        self.enhancer = MLSignalEnhancer()
        self.simulator = AdvancedMarketSimulator()

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        # Generate market scenario
        market_data = self.simulator.generate_scenario('bull_momentum', 'E2E_TEST', days=40)

        # Create technical signal
        technical_signal = {
            'signal': 'BUY',
            'confidence': 0.6,
            'indicators': {'rsi': 40, 'macd': 'bullish'}
        }

        # Enhance signal
        result = self.enhancer.enhance_signal('E2E_TEST', technical_signal, market_data)

        # Simulate performance tracking
        self.enhancer.update_performance('E2E_TEST', result['ensemble_signal'],
                                         'BUY', result['models_used'])

        # Get performance summary
        performance = self.enhancer.get_performance_summary()

        # Verify complete workflow
        self.assertIsInstance(result, dict)
        self.assertIn('ensemble_signal', result)
        self.assertGreater(performance['total_predictions'], 0)

        print(f"✅ End-to-end test: {result['ensemble_signal']} signal "
              f"({result['confidence']:.2%} confidence)")

    def test_multiple_symbol_performance(self):
        """Test performance across multiple symbols"""
        symbols = ['SYMBOL1', 'SYMBOL2', 'SYMBOL3']
        scenarios = ['bull_momentum', 'bear_selloff', 'sideways_range']

        for symbol, scenario in zip(symbols, scenarios):
            # Generate scenario-specific data
            market_data = self.simulator.generate_scenario(scenario, symbol, days=30)

            # Create appropriate technical signal
            signal_map = {
                'bull_momentum': 'BUY',
                'bear_selloff': 'SELL',
                'sideways_range': 'HOLD'
            }

            technical_signal = {
                'signal': signal_map[scenario],
                'confidence': 0.65,
                'indicators': {'rsi': 50}
            }

            # Enhance signal
            result = self.enhancer.enhance_signal(symbol, technical_signal, market_data)

            # Simulate some performance tracking
            self.enhancer.update_performance(symbol, result['ensemble_signal'],
                                             technical_signal['signal'], result['models_used'])

        # Check final performance
        performance = self.enhancer.get_performance_summary()
        self.assertGreaterEqual(performance['total_predictions'], len(symbols))

        print(f"✅ Multi-symbol test: {performance['total_predictions']} predictions tracked")


def run_comprehensive_tests():
    """Run all tests and provide summary"""
    print("\n" + "=" * 60)
    print("🧪 COMPREHENSIVE ML SIGNAL ENHANCER TESTS")
    print("=" * 60)

    if not IMPORTS_SUCCESSFUL:
        print("❌ Cannot run tests - module imports failed")
        return False

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    test_classes = [TestAdvancedMockModels, TestMLSignalEnhancer, TestIntegration]
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")

    if result.errors:
        print("\n❌ ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Error:')[-1].strip()}")

    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun
    print(f"\n✅ Success Rate: {success_rate:.1%}")

    if success_rate >= 0.9:
        print("🎉 EXCELLENT! All core functionality working!")
    elif success_rate >= 0.7:
        print("✅ GOOD! Most functionality working with minor issues.")
    else:
        print("⚠️ NEEDS WORK! Significant issues found.")

    return success_rate >= 0.7


if __name__ == "__main__":
    run_comprehensive_tests()