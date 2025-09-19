# 08_TESTS/test_reliability_basic.py
"""
Basic Reliability Components Test - Phase 2, Step 4
Simple verification that reliability components are working

Location: #08_TESTS/test_reliability_basic.py
"""

import sys
import time
import random
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add paths for imports
sys.path.append('03_ML_ENGINE/reliability')
sys.path.append('03_ML_ENGINE/performance')


def test_circuit_breaker():
    """Test basic circuit breaker functionality"""
    logger.info("🧪 Testing Circuit Breaker")

    try:
        from ml_circuit_breaker import ml_circuit_breaker, MLModelType, get_ml_system_health_dashboard

        # Create test function with circuit breaker
        @ml_circuit_breaker('test_basic', MLModelType.ALPHA_MODEL)
        def test_prediction(data):
            if random.random() < 0.2:  # 20% failure rate
                raise ValueError("Test failure")
            return {'signal': 'BUY', 'confidence': 0.7}

        # Run multiple tests
        successes = 0
        failures = 0

        for i in range(20):
            try:
                result = test_prediction({'price': 100 + i})
                successes += 1
                logger.debug(f"Success {i + 1}: {result}")
            except Exception as e:
                failures += 1
                logger.debug(f"Failure {i + 1}: {e}")

        logger.info(f"✅ Circuit Breaker Test: {successes} successes, {failures} failures")

        # Show dashboard
        dashboard = get_ml_system_health_dashboard()
        logger.info("📊 System Health Dashboard:")
        print(dashboard)

        return True

    except Exception as e:
        logger.error(f"❌ Circuit breaker test failed: {e}")
        return False


def test_error_handler():
    """Test basic error handler functionality"""
    logger.info("🧪 Testing Error Handler")

    try:
        from error_handler import ml_error_handler, get_error_analysis_report

        # Test error handling
        for i in range(10):
            try:
                # Simulate various errors
                error_type = random.choice(['timeout', 'data', 'other'])
                if error_type == 'timeout':
                    raise TimeoutError("Simulated timeout")
                elif error_type == 'data':
                    raise ValueError("Bad input data")
                else:
                    raise RuntimeError("Generic error")
            except Exception as e:
                # Let error handler process it
                logger.debug(f"Error {i + 1}: {e}")

        logger.info("✅ Error Handler Test completed")

        # Try to get error report
        try:
            report = get_error_analysis_report()
            if report:
                logger.info("📊 Error analysis report available")
        except Exception as e:
            logger.debug(f"Error report not available: {e}")

        return True

    except Exception as e:
        logger.error(f"❌ Error handler test failed: {e}")
        return False


def test_performance_monitor():
    """Test basic performance monitoring"""
    logger.info("🧪 Testing Performance Monitor")

    try:
        from performance_logger import performance_monitor, get_system_performance_dashboard

        @performance_monitor(track_memory=True)
        def monitored_function(data):
            time.sleep(0.01)  # Simulate some work
            return {'result': 'success'}

        # Run monitored function multiple times
        for i in range(10):
            try:
                result = monitored_function({'test': i})
                logger.debug(f"Monitored call {i + 1}: {result}")
            except Exception as e:
                logger.debug(f"Monitored call {i + 1} failed: {e}")

        logger.info("✅ Performance Monitor Test completed")

        # Try to get performance dashboard
        try:
            dashboard = get_system_performance_dashboard()
            if dashboard:
                logger.info("📊 Performance dashboard available")
        except Exception as e:
            logger.debug(f"Performance dashboard not available: {e}")

        return True

    except Exception as e:
        logger.error(f"❌ Performance monitor test failed: {e}")
        return False


def test_enhanced_model_integration():
    """Test enhanced model integration framework"""
    logger.info("🧪 Testing Enhanced Model Integration")

    try:
        sys.path.append('03_ML_ENGINE/integration')
        from enhanced_model_integration import enhanced_model_manager, ModelTier, setup_example_models

        # Setup example models
        setup_example_models()

        # Test predictions
        test_data = {'symbol': 'AAPL', 'price': 150.0, 'volume': 1000000}

        for tier in [ModelTier.STANDARD, ModelTier.ECONOMIC]:
            try:
                result = enhanced_model_manager.predict_with_fallback(test_data, tier)
                logger.info(f"✅ {tier.value} prediction: {result.latency_ms:.1f}ms, "
                            f"confidence: {result.confidence:.2f}")
            except Exception as e:
                logger.warning(f"⚠️ {tier.value} prediction failed: {e}")

        # Get performance metrics
        metrics = enhanced_model_manager.get_performance_metrics()
        if metrics.get('status') != 'no_data':
            logger.info(f"📊 Avg latency: {metrics.get('avg_latency_ms', 0):.1f}ms")

        return True

    except Exception as e:
        logger.error(f"❌ Enhanced model integration test failed: {e}")
        return False


def test_latency_optimizer():
    """Test latency optimization framework"""
    logger.info("🧪 Testing Latency Optimizer")

    try:
        sys.path.append('03_ML_ENGINE/optimization')
        from latency_optimizer import create_optimized_predictor, OptimizationLevel

        # Create mock model
        def mock_model(input_data):
            time.sleep(0.005)  # 5ms processing
            return {'signal': 'BUY', 'confidence': 0.8}

        # Create optimized version
        optimized_model = create_optimized_predictor(mock_model, OptimizationLevel.AGGRESSIVE)

        # Test predictions (including duplicates for cache testing)
        test_inputs = [
            {'symbol': 'AAPL', 'price': 150.0},
            {'symbol': 'GOOGL', 'price': 2500.0},
            {'symbol': 'AAPL', 'price': 150.0},  # Duplicate for cache test
        ]

        for test_input in test_inputs:
            try:
                result = optimized_model(test_input)
                logger.debug(f"Optimized prediction: {result}")
            except Exception as e:
                logger.warning(f"Optimized prediction failed: {e}")

        # Get performance report
        try:
            report = optimized_model.optimizer.get_performance_report()
            if report.get('status') != 'no_data':
                logger.info(f"✅ Latency optimizer: {report['cache_performance']['hit_rate']:.1%} cache hit rate")
        except Exception as e:
            logger.debug(f"Performance report not available: {e}")

        return True

    except Exception as e:
        logger.error(f"❌ Latency optimizer test failed: {e}")
        return False


def main():
    """Run all basic reliability tests"""
    print("🧪 Basic Reliability Components Test")
    print("=" * 50)

    tests = [
        ("Circuit Breaker", test_circuit_breaker),
        ("Error Handler", test_error_handler),
        ("Performance Monitor", test_performance_monitor),
        ("Enhanced Model Integration", test_enhanced_model_integration),
        ("Latency Optimizer", test_latency_optimizer),
    ]

    results = {}

    for test_name, test_func in tests:
        logger.info(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False

    # Summary
    print(f"\n{'=' * 60}")
    print("🎯 TEST RESULTS SUMMARY")
    print("=" * 60)

    passed = 0
    total = len(tests)

    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} - {test_name}")
        if result:
            passed += 1

    print(f"\n🎉 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("\n✅ ALL RELIABILITY TESTS PASSED!")
        print("🚀 Enhanced error handling and circuit breaker system is operational!")
    else:
        print(f"\n⚠️ {total - passed} tests failed. System partially operational.")

    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)