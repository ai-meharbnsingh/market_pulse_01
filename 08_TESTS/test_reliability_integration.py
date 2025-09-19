# 08_TESTS/test_reliability_integration.py
"""
Enhanced Reliability Integration Testing - Phase 2, Step 4
Comprehensive testing of circuit breakers, error handling, and ML model integration

Location: #08_TESTS/test_reliability_integration.py

This module tests:
- ML model circuit breaker functionality
- Error handling and recovery strategies
- Performance monitoring with reliability components
- Integration of all reliability systems
- Stress testing and failure scenarios
"""

import time
import random
import logging
import unittest
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import reliability components
try:
    sys.path.append('03_ML_ENGINE/reliability')
    sys.path.append('03_ML_ENGINE/performance')
    sys.path.append('03_ML_ENGINE/models')

    from ml_circuit_breaker import (
        ml_circuit_breaker,
        MLModelType,
        MLCircuitBreakerConfig,
        ml_circuit_registry,
        get_ml_system_health_dashboard
    )

    from error_handler import (
        ml_error_handler_decorator,
        ml_error_handler,
        get_error_analysis_report
    )

    from performance_logger import (
        performance_monitor,
        get_system_performance_dashboard
    )

    from ml_signal_enhancer import MLSignalEnhancer

    HAS_COMPONENTS = True

except ImportError as e:
    logger.warning(f"⚠️ Could not import components: {e}")
    HAS_COMPONENTS = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestReliabilityIntegration(unittest.TestCase):
    """Test suite for reliability integration"""

    def setUp(self):
        """Set up test environment"""
        if not HAS_COMPONENTS:
            self.skipTest("Reliability components not available")

        self.start_time = datetime.now()
        self.test_data = {
            'symbol': 'AAPL',
            'price': 150.0,
            'volume': 1000000,
            'timestamp': self.start_time.isoformat()
        }

        # Reset circuit breakers
        try:
            ml_circuit_registry.reset_all()
        except Exception:
            pass

    def tearDown(self):
        """Clean up after tests"""
        pass

    def test_circuit_breaker_basic_functionality(self):
        """Test basic circuit breaker functionality"""
        logger.info("🧪 Testing circuit breaker basic functionality")

        success_count = 0
        failure_count = 0

        # Create a test function with circuit breaker
        @ml_circuit_breaker('test_model', MLModelType.ALPHA_MODEL)
        def test_prediction(data: Dict) -> Dict:
            """Test prediction function with 30% failure rate"""
            if random.random() < 0.3:  # 30% failure rate
                raise ValueError("Model prediction failed")
            return {'signal': 'BUY', 'confidence': 0.7}

        # Test multiple predictions
        for i in range(20):
            try:
                result = test_prediction(self.test_data)
                success_count += 1
                self.assertIn('signal', result)
            except Exception as e:
                failure_count += 1
                logger.debug(f"Expected failure {i + 1}: {e}")

        # Verify circuit breaker recorded the attempts
        health_report = ml_circuit_registry.get('test_model').get_health_report()
        total_requests = health_report['metrics']['total_requests']

        self.assertGreater(total_requests, 15, "Circuit breaker should record requests")
        self.assertEqual(success_count + failure_count, 20, "All requests should be accounted for")

        logger.info(f"✅ Circuit breaker test: {success_count} success, {failure_count} failures")

    def test_error_handler_integration(self):
        """Test error handler integration with circuit breakers"""
        logger.info("🧪 Testing error handler integration")

        error_count = 0

        @ml_error_handler_decorator('test_error_handler')
        @ml_circuit_breaker('error_test_model', MLModelType.LSTM_MODEL)
        def error_prone_function(data: Dict) -> Dict:
            """Function that throws different types of errors"""
            error_type = random.choice(['timeout', 'data', 'prediction', 'resource'])

            if error_type == 'timeout':
                time.sleep(0.01)  # Small delay to simulate timeout
                raise TimeoutError("Model timeout")
            elif error_type == 'data':
                raise ValueError("Invalid input data")
            elif error_type == 'prediction':
                raise RuntimeError("Prediction confidence too low")
            elif error_type == 'resource':
                raise MemoryError("Insufficient memory")
            else:
                return {'prediction': 'success', 'confidence': 0.8}

        # Test error handling
        for i in range(15):
            try:
                result = error_prone_function(self.test_data)
                logger.debug(f"Success {i + 1}: {result}")
            except Exception as e:
                error_count += 1
                logger.debug(f"Error {i + 1}: {type(e).__name__}: {e}")

        # Verify error handler captured errors
        error_report = get_error_analysis_report()
        self.assertIsNotNone(error_report, "Error handler should generate report")

        logger.info(f"✅ Error handler test: {error_count} errors handled")

    def test_performance_monitoring_integration(self):
        """Test performance monitoring with reliability components"""
        logger.info("🧪 Testing performance monitoring integration")

        @performance_monitor(track_memory=True)
        @ml_circuit_breaker('perf_test_model', MLModelType.ENSEMBLE)
        def monitored_prediction(data: Dict) -> Dict:
            """Performance monitored prediction function"""
            # Simulate some processing time
            time.sleep(random.uniform(0.001, 0.020))  # 1-20ms

            # Random failure 10% of the time
            if random.random() < 0.1:
                raise RuntimeError("Random prediction failure")

            return {
                'signal': random.choice(['BUY', 'SELL', 'HOLD']),
                'confidence': random.uniform(0.5, 0.9)
            }

        # Run multiple predictions to gather performance data
        successful_predictions = 0
        for i in range(50):
            try:
                result = monitored_prediction(self.test_data)
                successful_predictions += 1
            except Exception as e:
                logger.debug(f"Performance test failure {i + 1}: {e}")

        # Verify performance data was captured
        perf_dashboard = get_system_performance_dashboard()
        self.assertIsNotNone(perf_dashboard, "Performance dashboard should be available")

        # Verify circuit breaker health
        health_report = ml_circuit_registry.get('perf_test_model').get_health_report()
        self.assertGreater(health_report['metrics']['total_requests'], 40,
                           "Should have processed most requests")

        logger.info(f"✅ Performance monitoring test: {successful_predictions}/50 successful")

    def test_ml_signal_enhancer_reliability(self):
        """Test ML Signal Enhancer with reliability components"""
        logger.info("🧪 Testing ML Signal Enhancer reliability integration")

        # Create ML Signal Enhancer instance
        enhancer = MLSignalEnhancer()

        # Test with various market data scenarios
        test_scenarios = [
            {'symbol': 'AAPL', 'price': 150.0, 'volume': 1000000},
            {'symbol': 'GOOGL', 'price': 2500.0, 'volume': 500000},
            {'symbol': 'TSLA', 'price': 200.0, 'volume': 2000000},
            {'symbol': 'MSFT', 'price': 300.0, 'volume': 800000}
        ]

        enhanced_signals = 0
        total_attempts = 0

        for scenario in test_scenarios:
            for i in range(10):  # 10 attempts per scenario
                try:
                    total_attempts += 1

                    # Create signal features
                    signal_features = {
                        'price': scenario['price'],
                        'volume': scenario['volume'],
                        'rsi': random.uniform(30, 70),
                        'macd': random.uniform(-0.5, 0.5),
                        'bb_position': random.uniform(0, 1)
                    }

                    # Enhance the signal
                    enhanced_signal = enhancer.enhance_signal(
                        base_signal='BUY',
                        signal_features=signal_features,
                        confidence=0.6
                    )

                    self.assertIsNotNone(enhanced_signal, "Enhanced signal should not be None")
                    self.assertIn('final_signal', enhanced_signal)
                    enhanced_signals += 1

                except Exception as e:
                    logger.debug(f"ML Signal enhancer error: {e}")

        success_rate = enhanced_signals / total_attempts
        self.assertGreater(success_rate, 0.7, "Signal enhancement should have >70% success rate")

        logger.info(f"✅ ML Signal Enhancer reliability test: {success_rate:.1%} success rate")

    def test_stress_testing_circuit_breakers(self):
        """Stress test circuit breakers under high load"""
        logger.info("🧪 Stress testing circuit breakers")

        # Create multiple circuit breakers
        configs = {
            'high_performance': MLCircuitBreakerConfig(
                failure_threshold=3,
                max_prediction_time_ms=5
            ),
            'standard': MLCircuitBreakerConfig(
                failure_threshold=5,
                max_prediction_time_ms=20
            ),
            'tolerant': MLCircuitBreakerConfig(
                failure_threshold=10,
                max_prediction_time_ms=50
            )
        }

        results = {}

        for config_name, config in configs.items():
            logger.info(f"Testing {config_name} configuration")

            @ml_circuit_breaker(f'stress_test_{config_name}', MLModelType.ALPHA_MODEL, config)
            def stress_test_function(data: Dict) -> Dict:
                # Simulate variable processing time and failures
                processing_time = random.exponential(0.010)  # Average 10ms
                time.sleep(processing_time)

                if random.random() < 0.15:  # 15% failure rate
                    raise RuntimeError("Stress test failure")

                return {'result': 'success', 'processing_time': processing_time}

            # Run stress test
            successes = 0
            failures = 0

            for i in range(100):  # 100 requests per configuration
                try:
                    result = stress_test_function(self.test_data)
                    successes += 1
                except Exception:
                    failures += 1

            results[config_name] = {
                'successes': successes,
                'failures': failures,
                'success_rate': successes / (successes + failures)
            }

        # Verify all configurations handled the stress
        for config_name, result in results.items():
            self.assertGreater(result['success_rate'], 0.6,
                               f"{config_name} should have >60% success under stress")

        logger.info("✅ Stress test completed:")
        for config_name, result in results.items():
            logger.info(f"  {config_name}: {result['success_rate']:.1%} success rate")

    def test_system_health_dashboard(self):
        """Test system health dashboard functionality"""
        logger.info("🧪 Testing system health dashboard")

        # Create some circuit breakers with activity
        @ml_circuit_breaker('dashboard_test_1', MLModelType.ALPHA_MODEL)
        def test_model_1(data):
            if random.random() < 0.1:
                raise ValueError("Test failure")
            return {'result': 'success'}

        @ml_circuit_breaker('dashboard_test_2', MLModelType.LSTM_MODEL)
        def test_model_2(data):
            time.sleep(0.005)  # 5ms processing
            return {'result': 'success'}

        # Generate some activity
        for i in range(30):
            try:
                test_model_1(self.test_data)
                test_model_2(self.test_data)
            except Exception:
                pass

        # Get dashboard
        dashboard = get_ml_system_health_dashboard()
        self.assertIsNotNone(dashboard, "Dashboard should be available")
        self.assertIn('HEALTH DASHBOARD', dashboard)
        self.assertIn('dashboard_test_1', dashboard)
        self.assertIn('dashboard_test_2', dashboard)

        logger.info("✅ System health dashboard test passed")

    def test_concurrent_reliability_operations(self):
        """Test reliability components under concurrent load"""
        logger.info("🧪 Testing concurrent reliability operations")

        results = {'successes': 0, 'failures': 0}
        results_lock = threading.Lock()

        @ml_circuit_breaker('concurrent_test', MLModelType.ENSEMBLE)
        def concurrent_prediction(thread_id: int, data: Dict) -> Dict:
            """Thread-safe prediction function"""
            # Simulate processing time
            time.sleep(random.uniform(0.001, 0.010))

            if random.random() < 0.05:  # 5% failure rate
                raise RuntimeError(f"Thread {thread_id} prediction failed")

            return {'thread_id': thread_id, 'result': 'success'}

        def worker_thread(thread_id: int):
            """Worker thread function"""
            for i in range(20):
                try:
                    result = concurrent_prediction(thread_id, self.test_data)
                    with results_lock:
                        results['successes'] += 1
                except Exception:
                    with results_lock:
                        results['failures'] += 1

        # Start multiple worker threads
        threads = []
        for i in range(5):  # 5 concurrent threads
            thread = threading.Thread(target=worker_thread, args=(i,))
            thread.start()
            threads.append(thread)

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify thread safety
        total_operations = results['successes'] + results['failures']
        self.assertEqual(total_operations, 100, "All operations should be accounted for")

        success_rate = results['successes'] / total_operations
        self.assertGreater(success_rate, 0.8, "Concurrent operations should have >80% success rate")

        logger.info(f"✅ Concurrent operations test: {success_rate:.1%} success rate")


def run_reliability_integration_tests():
    """Run all reliability integration tests"""
    print("🧪 Running Enhanced Reliability Integration Tests")
    print("=" * 60)

    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestReliabilityIntegration)

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 60)
    print("🎯 TEST RESULTS SUMMARY")
    print("=" * 60)

    if result.wasSuccessful():
        print("✅ ALL RELIABILITY INTEGRATION TESTS PASSED!")
        print(f"📊 Tests run: {result.testsRun}")
        print("🚀 Enhanced error handling system is working correctly!")
    else:
        print("❌ Some tests failed:")
        print(f"📊 Tests run: {result.testsRun}")
        print(f"❌ Failures: {len(result.failures)}")
        print(f"⚠️ Errors: {len(result.errors)}")

        for test, traceback in result.failures + result.errors:
            print(f"\n❌ {test}: {traceback}")

    # Show system health dashboard
    print("\n" + "=" * 60)
    print("🔧 SYSTEM HEALTH DASHBOARD")
    print("=" * 60)
    try:
        print(get_ml_system_health_dashboard())
    except Exception as e:
        print(f"⚠️ Could not display dashboard: {e}")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_reliability_integration_tests()
    exit(0 if success else 1)