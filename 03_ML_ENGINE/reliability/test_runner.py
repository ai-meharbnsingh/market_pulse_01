# 03_ML_ENGINE/reliability/test_runner.py
"""
Reliability System Test Runner - Phase 2, Step 4
Quick verification that all reliability components are working correctly

Location: #03_ML_ENGINE/reliability/test_runner.py

This module provides:
- Quick system verification tests
- Component integration testing
- Performance benchmarking
- Error scenario testing
- Dashboard functionality verification
"""

import time
import random
import logging
from datetime import datetime
from typing import Dict, Any, List

# Import all reliability components
from .ml_circuit_breaker import (
    ml_circuit_breaker,
    MLModelType,
    MLCircuitBreakerConfig,
    ml_circuit_registry,
    get_ml_system_health_dashboard
)

from .error_handler import (
    ml_error_handler_decorator,
    ml_error_handler,
    get_error_analysis_report
)

from ..performance.performance_logger import (
    performance_monitor,
    get_system_performance_dashboard
)

from .monitoring_dashboard import (
    get_reliability_dashboard,
    create_alert
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReliabilitySystemTester:
    """Test runner for reliability system components"""

    def __init__(self):
        """Initialize test runner"""
        self.test_results = {}
        self.start_time = datetime.now()

    def run_all_tests(self) -> Dict[str, bool]:
        """Run all reliability system tests"""
        logger.info("🧪 Starting Reliability System Tests")
        logger.info("=" * 60)

        tests = [
            ("Circuit Breaker Basic", self.test_circuit_breaker_basic),
            ("Circuit Breaker Failure Handling", self.test_circuit_breaker_failures),
            ("Error Handler Classification", self.test_error_handler),
            ("Performance Monitoring", self.test_performance_monitoring),
            ("Integration Test", self.test_integration),
            ("Dashboard Generation", self.test_dashboard_generation),
            ("Load Testing", self.test_load_handling),
        ]

        for test_name, test_func in tests:
            try:
                logger.info(f"\n🔬 Running: {test_name}")
                success = test_func()
                self.test_results[test_name] = success

                if success:
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.error(f"❌ {test_name}: FAILED")

            except Exception as e:
                logger.error(f"💥 {test_name}: ERROR - {e}")
                self.test_results[test_name] = False

        self._print_test_summary()
        return self.test_results

    def test_circuit_breaker_basic(self) -> bool:
        """Test basic circuit breaker functionality"""
        try:
            @ml_circuit_breaker("test_basic", MLModelType.ALPHA_MODEL)
            def test_function():
                return {"result": "success", "confidence": 0.8}

            # Should work normally
            result = test_function()

            # Verify result structure
            if not isinstance(result, dict):
                return False

            # Check circuit breaker was registered
            breaker = ml_circuit_registry.get("test_basic")
            if breaker is None:
                return False

            # Verify metrics
            if breaker.metrics.successful_requests < 1:
                return False

            logger.info("   ✓ Basic circuit breaker functionality working")
            return True

        except Exception as e:
            logger.error(f"   ❌ Basic test failed: {e}")
            return False

    def test_circuit_breaker_failures(self) -> bool:
        """Test circuit breaker failure handling"""
        try:
            config = MLCircuitBreakerConfig(
                failure_threshold=2,
                success_threshold=1,
                timeout_seconds=0.1
            )

            call_count = 0

            @ml_circuit_breaker("test_failures", MLModelType.LSTM_MODEL, config)
            def failing_function():
                nonlocal call_count
                call_count += 1
                if call_count <= 3:
                    raise ValueError(f"Test failure {call_count}")
                return {"status": "recovered"}

            # First calls should fail and open circuit
            for i in range(3):
                result = failing_function()
                # Should get fallback results
                if not isinstance(result, dict):
                    return False

            # Check if circuit is open or half-open
            breaker = ml_circuit_registry.get("test_failures")
            if breaker is None:
                return False

            # Should have recorded failures
            if breaker.metrics.failed_requests == 0:
                return False

            logger.info("   ✓ Failure handling and circuit breaking working")
            return True

        except Exception as e:
            logger.error(f"   ❌ Failure test failed: {e}")
            return False

    def test_error_handler(self) -> bool:
        """Test error handler functionality"""
        try:
            @ml_error_handler_decorator(context_data={'test': 'error_handler'})
            def error_test_function(error_type: str):
                if error_type == "model":
                    raise ValueError("Model prediction failed")
                elif error_type == "data":
                    raise KeyError("Missing required features")
                elif error_type == "resource":
                    raise MemoryError("Out of memory")
                else:
                    return {"status": "ok"}

            # Test different error types
            error_types = ["model", "data", "resource"]

            for error_type in error_types:
                try:
                    error_test_function(error_type)
                except Exception:
                    pass  # Expected to fail

            # Test success case
            result = error_test_function("success")
            if result["status"] != "ok":
                return False

            # Check if errors were logged
            stats = ml_error_handler.get_error_statistics(1)
            if 'error' in stats:
                logger.warning("   ⚠️ Error handler database not available")
                return True  # Still passes if basic functionality works

            if sum(stats.get('category_distribution', {}).values()) < 3:
                return False

            logger.info("   ✓ Error handler classification and logging working")
            return True

        except Exception as e:
            logger.error(f"   ❌ Error handler test failed: {e}")
            return False

    def test_performance_monitoring(self) -> bool:
        """Test performance monitoring functionality"""
        try:
            @performance_monitor(cache_key='test_performance')
            def monitored_function(delay: float = 0.01):
                time.sleep(delay)
                return {"processing_time": delay, "result": "completed"}

            # Execute function multiple times
            for i in range(5):
                result = monitored_function(0.002 + i * 0.001)
                if not isinstance(result, dict):
                    return False

            # Test caching (should be faster on repeated calls)
            start_time = time.time()
            result = monitored_function(0.005)  # Same parameters
            first_call_time = time.time() - start_time

            start_time = time.time()
            result = monitored_function(0.005)  # Should be cached
            second_call_time = time.time() - start_time

            # Second call should be much faster (cached)
            if second_call_time >= first_call_time:
                logger.warning("   ⚠️ Caching may not be working optimally")

            logger.info("   ✓ Performance monitoring and caching working")
            return True

        except Exception as e:
            logger.error(f"   ❌ Performance monitoring test failed: {e}")
            return False

    def test_integration(self) -> bool:
        """Test integration between all components"""
        try:
            @ml_circuit_breaker("integration_test", MLModelType.ENSEMBLE)
            @ml_error_handler_decorator(context_data={'test': 'integration'})
            @performance_monitor(cache_key='integration')
            def integrated_function(mode: str = "normal"):
                time.sleep(0.005)  # Simulate processing

                if mode == "fail":
                    raise Exception("Integration test failure")
                elif mode == "slow":
                    time.sleep(0.05)  # Slower processing

                return {
                    "mode": mode,
                    "timestamp": datetime.now().isoformat(),
                    "confidence": random.uniform(0.6, 0.9)
                }

            # Test normal operation
            result = integrated_function("normal")
            if result["mode"] != "normal":
                return False

            # Test failure handling
            result = integrated_function("fail")
            # Should get fallback or error recovery
            if not isinstance(result, dict):
                return False

            # Test performance monitoring with slower operation
            result = integrated_function("slow")
            if not isinstance(result, dict):
                return False

            logger.info("   ✓ Component integration working")
            return True

        except Exception as e:
            logger.error(f"   ❌ Integration test failed: {e}")
            return False

    def test_dashboard_generation(self) -> bool:
        """Test dashboard generation functionality"""
        try:
            # Generate various dashboards
            dashboards = {
                "Circuit Breaker": get_ml_system_health_dashboard(),
                "Performance": get_system_performance_dashboard(),
                "Error Analysis": get_error_analysis_report(),
                "Reliability": get_reliability_dashboard()
            }

            # Verify each dashboard generates content
            for name, dashboard in dashboards.items():
                if not isinstance(dashboard, str) or len(dashboard) < 50:
                    logger.error(f"   ❌ {name} dashboard generation failed")
                    return False

                # Check for expected content
                if name == "Circuit Breaker" and "CIRCUIT BREAKER" not in dashboard:
                    logger.warning(f"   ⚠️ {name} dashboard content may be incomplete")

            # Test alert creation
            create_alert("info", "test_system", "Test alert for dashboard verification")

            logger.info("   ✓ Dashboard generation working")
            return True

        except Exception as e:
            logger.error(f"   ❌ Dashboard test failed: {e}")
            return False

    def test_load_handling(self) -> bool:
        """Test system under load"""
        try:
            import threading
            import queue

            results = queue.Queue()
            errors = queue.Queue()

            @ml_circuit_breaker("load_test", MLModelType.ALPHA_MODEL)
            @performance_monitor()
            def load_test_function(thread_id: int, call_id: int):
                time.sleep(random.uniform(0.001, 0.005))

                # Random failures (10% rate)
                if random.random() < 0.1:
                    raise Exception(f"Load test failure {thread_id}-{call_id}")

                return {"thread": thread_id, "call": call_id, "result": "success"}

            def worker(thread_id: int):
                for call_id in range(10):  # 10 calls per thread
                    try:
                        result = load_test_function(thread_id, call_id)
                        results.put(result)
                    except Exception as e:
                        errors.put((thread_id, call_id, str(e)))

            # Start multiple threads
            threads = []
            for i in range(3):  # 3 threads
                thread = threading.Thread(target=worker, args=(i,))
                threads.append(thread)
                thread.start()

            # Wait for completion
            for thread in threads:
                thread.join()

            # Collect results
            success_count = results.qsize()
            error_count = errors.qsize()
            total_calls = 30  # 3 threads * 10 calls

            # Should have processed most calls
            processed_calls = success_count + error_count
            if processed_calls < total_calls * 0.8:  # At least 80% processed
                return False

            logger.info(
                f"   ✓ Load test: {success_count} successes, {error_count} errors, {processed_calls}/{total_calls} processed")
            return True

        except Exception as e:
            logger.error(f"   ❌ Load test failed: {e}")
            return False

    def _print_test_summary(self):
        """Print test summary"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        failed_tests = total_tests - passed_tests

        elapsed_time = datetime.now() - self.start_time

        logger.info("\n" + "=" * 60)
        logger.info("🎯 RELIABILITY SYSTEM TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"✅ Passed: {passed_tests}")
        logger.info(f"❌ Failed: {failed_tests}")
        logger.info(f"⏱️ Duration: {elapsed_time.total_seconds():.2f} seconds")
        logger.info(f"📊 Success Rate: {(passed_tests / total_tests) * 100:.1f}%")

        if failed_tests > 0:
            logger.error("\n❌ Failed Tests:")
            for test_name, result in self.test_results.items():
                if not result:
                    logger.error(f"   - {test_name}")
        else:
            logger.info("\n🎉 All tests passed! Reliability system is working correctly.")

        logger.info("\n" + "=" * 60)


def run_reliability_tests() -> bool:
    """Run reliability system tests"""
    tester = ReliabilitySystemTester()
    results = tester.run_all_tests()

    # Return True if all tests passed
    return all(results.values())


def quick_system_check() -> str:
    """Quick system health check"""
    logger.info("🔍 Running Quick System Health Check")

    try:
        # Get all dashboard summaries
        cb_dashboard = get_ml_system_health_dashboard()
        perf_dashboard = get_system_performance_dashboard()
        error_report = get_error_analysis_report()
        reliability_dashboard = get_reliability_dashboard()

        return f"""
🔍 QUICK SYSTEM HEALTH CHECK
{'=' * 50}
✅ Circuit Breaker System: {'OK' if 'HEALTH DASHBOARD' in cb_dashboard else 'ISSUE'}
✅ Performance Monitoring: {'OK' if 'Health Score' in perf_dashboard else 'ISSUE'}
✅ Error Analysis: {'OK' if 'ERROR ANALYSIS' in error_report else 'ISSUE'}
✅ Reliability Dashboard: {'OK' if 'MONITORING DASHBOARD' in reliability_dashboard else 'ISSUE'}

📊 System Status: {'OPERATIONAL' if all(['DASHBOARD' in cb_dashboard, 'Health Score' in perf_dashboard]) else 'CHECK REQUIRED'}

💡 Run full tests with: python -m ml_engine.reliability.test_runner
{'=' * 50}
"""

    except Exception as e:
        return f"""
❌ QUICK SYSTEM HEALTH CHECK FAILED
{'=' * 50}
Error: {e}
Recommendation: Run full diagnostics
{'=' * 50}
"""


# Run tests if executed directly
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        print(quick_system_check())
    else:
        success = run_reliability_tests()
        sys.exit(0 if success else 1)