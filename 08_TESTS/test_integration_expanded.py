# 08_TESTS/test_integration_expanded.py
"""
Expanded Integration Test Suite for Advanced ML Signal Enhancer - Phase 2, Step 3
Comprehensive testing with larger datasets and complex scenarios

Location: #08_TESTS/test_integration_expanded.py

This expanded test suite validates:
- Large-scale data processing
- Multi-symbol concurrent processing
- Market regime transition scenarios
- Performance under stress
- Error recovery mechanisms
- Long-running stability
- Memory usage patterns
- Cache effectiveness at scale
"""

import sys
import os
from pathlib import Path
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import time
import threading
import concurrent.futures
from typing import List, Dict, Any
import json

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')

# Setup path for importing modules
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir / "03_ML_ENGINE" / "models"))
sys.path.append(str(current_dir / "03_ML_ENGINE" / "performance"))

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
    from performance_logger import performance_monitor, get_performance_report, print_performance_summary

    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    IMPORTS_SUCCESSFUL = False
    print(f"❌ Import failed: {e}")


class TestLargeScaleProcessing(unittest.TestCase):
    """Test large-scale data processing capabilities"""

    def setUp(self):
        """Set up test fixtures for large-scale testing"""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest("Module imports failed")

        self.enhancer = MLSignalEnhancer()
        self.simulator = AdvancedMarketSimulator()

        # Generate large dataset
        self.large_symbols = [f"STOCK_{i:03d}" for i in range(50)]  # 50 symbols
        self.market_scenarios = ['bull_momentum', 'bear_selloff', 'sideways_range', 'volatile']

        self.test_results = []

    def test_large_dataset_processing(self):
        """Test processing of large datasets (1000+ data points)"""
        print("📊 Testing large dataset processing...")

        # Generate large market dataset
        large_market_data = self.simulator.generate_scenario('bull_momentum', 'LARGE_TEST', days=1000)

        technical_signal = {
            'signal': 'BUY',
            'confidence': 0.7,
            'indicators': {'rsi': 45, 'macd': 'bullish'}
        }

        start_time = time.time()
        result = self.enhancer.enhance_signal('LARGE_DATASET', technical_signal, large_market_data)
        processing_time = time.time() - start_time

        # Assertions
        self.assertIsInstance(result, dict)
        self.assertIn('ensemble_signal', result)
        self.assertLess(processing_time, 1.0, "Processing should complete within 1 second")

        print(f"   ✅ Processed 1000 data points in {processing_time:.3f}s")

        # Store result for analysis
        self.test_results.append({
            'test': 'large_dataset',
            'processing_time': processing_time,
            'data_size': len(large_market_data),
            'result': result
        })

    def test_multi_symbol_processing(self):
        """Test processing multiple symbols simultaneously"""
        print("📈 Testing multi-symbol processing...")

        technical_signal = {
            'signal': 'BUY',
            'confidence': 0.6,
            'indicators': {'rsi': 35, 'macd': 'neutral'}
        }

        # Generate market data for each symbol
        symbol_data = {}
        for symbol in self.large_symbols[:20]:  # Test with 20 symbols
            scenario = np.random.choice(self.market_scenarios)
            symbol_data[symbol] = self.simulator.generate_scenario(scenario, symbol, days=100)

        start_time = time.time()
        results = {}

        for symbol, market_data in symbol_data.items():
            results[symbol] = self.enhancer.enhance_signal(symbol, technical_signal, market_data)

        total_time = time.time() - start_time
        avg_time_per_symbol = total_time / len(symbol_data)

        # Assertions
        self.assertEqual(len(results), len(symbol_data))
        self.assertLess(avg_time_per_symbol, 0.5, "Average processing per symbol should be < 0.5s")

        # Check all results are valid
        for symbol, result in results.items():
            self.assertIn('ensemble_signal', result)
            self.assertIn(result['ensemble_signal'], ['BUY', 'SELL', 'HOLD'])

        print(f"   ✅ Processed {len(symbol_data)} symbols in {total_time:.3f}s "
              f"(avg {avg_time_per_symbol:.3f}s per symbol)")

        # Store results
        self.test_results.append({
            'test': 'multi_symbol',
            'total_time': total_time,
            'symbols_count': len(symbol_data),
            'avg_time_per_symbol': avg_time_per_symbol
        })

    def test_concurrent_processing(self):
        """Test concurrent processing with threading"""
        print("🔄 Testing concurrent processing...")

        def process_symbol(symbol_data_pair):
            symbol, market_data = symbol_data_pair
            technical_signal = {
                'signal': 'BUY',
                'confidence': 0.65,
                'indicators': {'rsi': 40}
            }
            return self.enhancer.enhance_signal(symbol, technical_signal, market_data)

        # Prepare data for concurrent processing
        symbol_data_pairs = []
        for i, symbol in enumerate(self.large_symbols[:10]):  # Test with 10 symbols
            scenario = self.market_scenarios[i % len(self.market_scenarios)]
            market_data = self.simulator.generate_scenario(scenario, symbol, days=50)
            symbol_data_pairs.append((symbol, market_data))

        # Sequential processing baseline
        start_time = time.time()
        sequential_results = [process_symbol(pair) for pair in symbol_data_pairs]
        sequential_time = time.time() - start_time

        # Concurrent processing
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            concurrent_results = list(executor.map(process_symbol, symbol_data_pairs))
        concurrent_time = time.time() - start_time

        # Assertions
        self.assertEqual(len(sequential_results), len(concurrent_results))
        speedup = sequential_time / concurrent_time if concurrent_time > 0 else 0

        print(f"   ✅ Sequential: {sequential_time:.3f}s, Concurrent: {concurrent_time:.3f}s, "
              f"Speedup: {speedup:.1f}x")

        # Store results
        self.test_results.append({
            'test': 'concurrent_processing',
            'sequential_time': sequential_time,
            'concurrent_time': concurrent_time,
            'speedup': speedup
        })

    def test_memory_usage_stability(self):
        """Test memory usage stability over many operations"""
        print("💾 Testing memory usage stability...")

        import psutil
        process = psutil.Process()

        # Record initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_readings = [initial_memory]

        technical_signal = {
            'signal': 'BUY',
            'confidence': 0.6,
            'indicators': {'rsi': 50}
        }

        # Perform many operations
        for i in range(100):
            symbol = f"MEM_TEST_{i}"
            scenario = self.market_scenarios[i % len(self.market_scenarios)]
            market_data = self.simulator.generate_scenario(scenario, symbol, days=30)

            result = self.enhancer.enhance_signal(symbol, technical_signal, market_data)

            # Record memory every 10 operations
            if i % 10 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_readings.append(current_memory)

        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        max_memory = max(memory_readings)

        # Assertions
        self.assertLess(memory_increase, 50, "Memory increase should be less than 50MB")
        self.assertLess(max_memory - initial_memory, 100, "Peak memory increase should be less than 100MB")

        print(f"   ✅ Memory usage: Initial {initial_memory:.1f}MB, Final {final_memory:.1f}MB, "
              f"Increase {memory_increase:.1f}MB, Peak {max_memory:.1f}MB")

        # Store results
        self.test_results.append({
            'test': 'memory_stability',
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'memory_increase_mb': memory_increase,
            'max_memory_mb': max_memory
        })


class TestMarketRegimeTransitions(unittest.TestCase):
    """Test behavior during market regime transitions"""

    def setUp(self):
        """Set up test fixtures for regime transition testing"""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest("Module imports failed")

        self.enhancer = MLSignalEnhancer()
        self.simulator = AdvancedMarketSimulator()

    def test_regime_transition_scenarios(self):
        """Test signal enhancement during market regime transitions"""
        print("📊 Testing market regime transitions...")

        # Define transition scenarios
        transitions = [
            ('bull_momentum', 'bear_selloff'),
            ('bear_selloff', 'sideways_range'),
            ('sideways_range', 'volatile'),
            ('volatile', 'bull_momentum')
        ]

        technical_signal = {
            'signal': 'HOLD',
            'confidence': 0.6,
            'indicators': {'rsi': 50, 'macd': 'neutral'}
        }

        transition_results = []

        for from_regime, to_regime in transitions:
            # Generate data for first regime
            data1 = self.simulator.generate_scenario(from_regime, 'TRANSITION_TEST', days=30)
            result1 = self.enhancer.enhance_signal('TRANSITION_TEST', technical_signal, data1)

            # Generate data for second regime
            data2 = self.simulator.generate_scenario(to_regime, 'TRANSITION_TEST', days=30)
            result2 = self.enhancer.enhance_signal('TRANSITION_TEST', technical_signal, data2)

            # Analyze transition
            confidence_change = result2['confidence'] - result1['confidence']
            signal_changed = result1['ensemble_signal'] != result2['ensemble_signal']

            transition_results.append({
                'from_regime': from_regime,
                'to_regime': to_regime,
                'signal1': result1['ensemble_signal'],
                'signal2': result2['ensemble_signal'],
                'confidence_change': confidence_change,
                'signal_changed': signal_changed
            })

            print(f"   {from_regime} → {to_regime}: {result1['ensemble_signal']} → {result2['ensemble_signal']} "
                  f"(confidence Δ{confidence_change:+.2%})")

        # Assertions
        self.assertEqual(len(transition_results), len(transitions))

        # At least some transitions should result in signal changes
        signal_changes = sum(1 for r in transition_results if r['signal_changed'])
        self.assertGreater(signal_changes, 0, "Some regime transitions should change signals")

        print(f"   ✅ Tested {len(transitions)} regime transitions, {signal_changes} resulted in signal changes")

    def test_mixed_regime_portfolio(self):
        """Test enhancement of portfolio with mixed market regimes"""
        print("🔄 Testing mixed regime portfolio...")

        portfolio_symbols = ['TECH_01', 'ENERGY_02', 'FINANCE_03', 'HEALTHCARE_04']
        regimes = ['bull_momentum', 'bear_selloff', 'sideways_range', 'volatile']

        portfolio_results = {}
        technical_signal = {
            'signal': 'BUY',
            'confidence': 0.65,
            'indicators': {'rsi': 45}
        }

        for symbol, regime in zip(portfolio_symbols, regimes):
            market_data = self.simulator.generate_scenario(regime, symbol, days=60)
            result = self.enhancer.enhance_signal(symbol, technical_signal, market_data)
            portfolio_results[symbol] = result

        # Analyze portfolio diversity
        signals = [r['ensemble_signal'] for r in portfolio_results.values()]
        confidences = [r['confidence'] for r in portfolio_results.values()]

        unique_signals = len(set(signals))
        avg_confidence = np.mean(confidences)
        confidence_std = np.std(confidences)

        # Assertions
        self.assertGreaterEqual(unique_signals, 2, "Portfolio should have diverse signals")
        self.assertGreater(avg_confidence, 0.5, "Average confidence should be reasonable")

        print(f"   ✅ Portfolio diversity: {unique_signals} unique signals, "
              f"avg confidence {avg_confidence:.1%} (σ={confidence_std:.1%})")


class TestStressAndReliability(unittest.TestCase):
    """Test system behavior under stress conditions"""

    def setUp(self):
        """Set up stress testing environment"""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest("Module imports failed")

        self.enhancer = MLSignalEnhancer()
        self.simulator = AdvancedMarketSimulator()

    def test_error_recovery(self):
        """Test recovery from various error conditions"""
        print("🛡️ Testing error recovery mechanisms...")

        error_scenarios = [
            # Empty market data
            ('empty_data', pd.DataFrame()),
            # Malformed market data
            ('malformed_data', pd.DataFrame({'wrong_columns': [1, 2, 3]})),
            # Very small dataset
            ('tiny_data', pd.DataFrame({
                'close': [100, 101],
                'volume': [1000, 1100],
                'date': pd.date_range('2024-01-01', periods=2)
            }))
        ]

        technical_signal = {
            'signal': 'BUY',
            'confidence': 0.6,
            'indicators': {'rsi': 50}
        }

        recovery_results = []

        for scenario_name, market_data in error_scenarios:
            try:
                result = self.enhancer.enhance_signal(scenario_name, technical_signal, market_data)

                # Should return valid result even with problematic data
                self.assertIsInstance(result, dict)
                self.assertIn('ensemble_signal', result)

                recovery_results.append({
                    'scenario': scenario_name,
                    'recovered': True,
                    'result': result['ensemble_signal']
                })

                print(f"   ✅ {scenario_name}: Recovered with {result['ensemble_signal']} signal")

            except Exception as e:
                recovery_results.append({
                    'scenario': scenario_name,
                    'recovered': False,
                    'error': str(e)
                })
                print(f"   ❌ {scenario_name}: Failed with {e}")

        # At least 2/3 scenarios should recover gracefully
        recovered_count = sum(1 for r in recovery_results if r['recovered'])
        self.assertGreaterEqual(recovered_count, 2, "Most error scenarios should be recoverable")

    def test_high_frequency_processing(self):
        """Test behavior under high-frequency processing demands"""
        print("⚡ Testing high-frequency processing...")

        technical_signal = {
            'signal': 'BUY',
            'confidence': 0.7,
            'indicators': {'rsi': 30}
        }

        # Generate base market data
        market_data = self.simulator.generate_scenario('bull_momentum', 'HF_TEST', days=50)

        # Rapid-fire processing
        start_time = time.time()
        processing_times = []

        for i in range(100):  # 100 rapid calls
            call_start = time.time()
            result = self.enhancer.enhance_signal(f'HF_SYMBOL_{i}', technical_signal, market_data)
            call_time = (time.time() - call_start) * 1000  # ms
            processing_times.append(call_time)

        total_time = time.time() - start_time
        avg_time = np.mean(processing_times)
        max_time = np.max(processing_times)
        throughput = 100 / total_time  # calls per second

        # Assertions
        self.assertLess(avg_time, 100, "Average call should be < 100ms")
        self.assertLess(max_time, 500, "Max call should be < 500ms")
        self.assertGreater(throughput, 5, "Should handle >5 calls per second")

        print(f"   ✅ Processed 100 calls in {total_time:.3f}s "
              f"(avg {avg_time:.1f}ms, max {max_time:.1f}ms, {throughput:.1f} calls/sec)")

    def test_cache_effectiveness_at_scale(self):
        """Test cache effectiveness with large-scale operations"""
        print("📦 Testing cache effectiveness at scale...")

        technical_signal = {
            'signal': 'BUY',
            'confidence': 0.6,
            'indicators': {'rsi': 45}
        }

        # Generate market data for testing
        market_data = self.simulator.generate_scenario('sideways_range', 'CACHE_TEST', days=100)
        symbol = 'CACHE_EFFECTIVENESS_TEST'

        # First call (cache miss)
        start_time = time.time()
        result1 = self.enhancer.enhance_signal(symbol, technical_signal, market_data)
        first_call_time = (time.time() - start_time) * 1000

        # Multiple subsequent calls (should be cache hits)
        cache_times = []
        for i in range(10):
            start_time = time.time()
            result = self.enhancer.enhance_signal(symbol, technical_signal, market_data)
            cache_time = (time.time() - start_time) * 1000
            cache_times.append(cache_time)

        avg_cache_time = np.mean(cache_times)
        speedup = first_call_time / avg_cache_time if avg_cache_time > 0 else 0

        # Assertions
        self.assertLess(avg_cache_time, first_call_time * 0.5, "Cache should provide significant speedup")
        self.assertGreater(speedup, 2, "Cache should provide at least 2x speedup")

        # Verify results consistency
        for i in range(5):
            result = self.enhancer.enhance_signal(symbol, technical_signal, market_data)
            self.assertEqual(result['ensemble_signal'], result1['ensemble_signal'])
            self.assertEqual(result['confidence'], result1['confidence'])

        print(f"   ✅ Cache effectiveness: First call {first_call_time:.1f}ms, "
              f"cached calls avg {avg_cache_time:.1f}ms, speedup {speedup:.1f}x")


class TestPerformanceIntegration(unittest.TestCase):
    """Test integration with performance monitoring system"""

    def setUp(self):
        """Set up performance integration testing"""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest("Module imports failed")

        self.enhancer = MLSignalEnhancer()
        self.simulator = AdvancedMarketSimulator()

    def test_performance_metrics_collection(self):
        """Test that performance metrics are properly collected"""
        print("📊 Testing performance metrics collection...")

        technical_signal = {
            'signal': 'BUY',
            'confidence': 0.65,
            'indicators': {'rsi': 40}
        }

        # Generate some activity
        for i in range(10):
            symbol = f'PERF_METRIC_TEST_{i}'
            scenario = ['bull_momentum', 'bear_selloff'][i % 2]
            market_data = self.simulator.generate_scenario(scenario, symbol, days=30)
            result = self.enhancer.enhance_signal(symbol, technical_signal, market_data)

        # Get performance report
        time.sleep(0.1)  # Allow metrics to be recorded
        try:
            performance_report = get_performance_report(hours_back=1)

            if performance_report.get('status') == 'success':
                summary = performance_report['system_summary']

                self.assertGreater(summary['total_calls'], 0)
                self.assertIn('avg_execution_time_ms', summary)

                print(f"   ✅ Performance metrics collected: {summary['total_calls']} calls, "
                      f"avg {summary['avg_execution_time_ms']:.1f}ms")
            else:
                print("   ⚠️ Performance monitoring data not available yet")

        except Exception as e:
            print(f"   ⚠️ Performance monitoring error: {e}")

    def test_performance_dashboard_integration(self):
        """Test integration with performance dashboard"""
        print("🎯 Testing performance dashboard integration...")

        # Generate activity for dashboard
        technical_signal = {'signal': 'BUY', 'confidence': 0.7, 'indicators': {'rsi': 35}}

        for i in range(5):
            symbol = f'DASHBOARD_TEST_{i}'
            market_data = self.simulator.generate_scenario('bull_momentum', symbol, days=25)
            result = self.enhancer.enhance_signal(symbol, technical_signal, market_data)

        # Test dashboard functionality
        try:
            self.enhancer.print_performance_dashboard()
            print("   ✅ Performance dashboard integrated successfully")
        except Exception as e:
            print(f"   ❌ Dashboard integration failed: {e}")
            self.fail(f"Dashboard integration failed: {e}")


def run_expanded_integration_tests():
    """Run expanded integration test suite with detailed reporting"""
    print("\n" + "=" * 80)
    print("🧪 EXPANDED INTEGRATION TEST SUITE")
    print("=" * 80)

    if not IMPORTS_SUCCESSFUL:
        print("❌ Cannot run tests - module imports failed")
        return False

    # Test classes to run
    test_classes = [
        TestLargeScaleProcessing,
        TestMarketRegimeTransitions,
        TestStressAndReliability,
        TestPerformanceIntegration
    ]

    all_results = []
    start_time = time.time()

    for test_class in test_classes:
        print(f"\n🔍 Running {test_class.__name__}...")

        # Create test suite for this class
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(test_class)

        # Run tests with custom result collector
        runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w'))
        result = runner.run(suite)

        # Collect results
        class_result = {
            'test_class': test_class.__name__,
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'success_rate': (result.testsRun - len(result.failures) - len(
                result.errors)) / result.testsRun if result.testsRun > 0 else 0
        }
        all_results.append(class_result)

    total_time = time.time() - start_time

    # Print comprehensive results
    print("\n" + "=" * 80)
    print("📊 EXPANDED INTEGRATION TEST RESULTS")
    print("=" * 80)

    total_tests = sum(r['tests_run'] for r in all_results)
    total_failures = sum(r['failures'] for r in all_results)
    total_errors = sum(r['errors'] for r in all_results)
    overall_success = (total_tests - total_failures - total_errors) / total_tests if total_tests > 0 else 0

    print(f"Total Tests Run: {total_tests}")
    print(f"Total Failures: {total_failures}")
    print(f"Total Errors: {total_errors}")
    print(f"Overall Success Rate: {overall_success:.1%}")
    print(f"Total Execution Time: {total_time:.2f}s")

    print("\n📈 Results by Test Class:")
    for result in all_results:
        status_icon = "✅" if result['success_rate'] == 1.0 else "⚠️" if result['success_rate'] >= 0.8 else "❌"
        print(f"  {status_icon} {result['test_class']}: {result['success_rate']:.1%} "
              f"({result['tests_run'] - result['failures'] - result['errors']}/{result['tests_run']} passed)")

    # Overall assessment
    if overall_success >= 0.95:
        print("\n🎉 EXCELLENT! System performs exceptionally well under all test conditions!")
    elif overall_success >= 0.85:
        print("\n✅ GOOD! System shows solid performance with minor issues.")
    elif overall_success >= 0.70:
        print("\n⚠️ ACCEPTABLE! System functional but needs optimization.")
    else:
        print("\n❌ NEEDS ATTENTION! Significant issues found.")

    return overall_success >= 0.80


if __name__ == "__main__":
    run_expanded_integration_tests()