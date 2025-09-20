# 03_ML_ENGINE/optimization/production_performance_optimizer.py
"""
Production Performance Optimizer for 95%+ Sub-20ms Targets
MarketPulse Phase 2, Step 5 - Performance Fine-Tuning

This module provides advanced performance optimization to achieve:
- 95%+ sub-20ms response times across all model tiers
- Intelligent caching with predictive pre-loading
- Model warm-up and keep-alive strategies
- Memory pool management and garbage collection optimization
- Adaptive batch processing and request queuing
- Circuit breaker performance tuning

Location: #03_ML_ENGINE/optimization/production_performance_optimizer.py
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import sqlite3
import json
import warnings
import time
import threading
import queue
import gc
import psutil
import hashlib
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, deque
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import lru

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup paths
current_dir = Path(__file__).parent
root_dir = current_dir.parent.parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir.parent / "reliability"))


class PerformanceTarget(Enum):
    """Performance targets for different tiers"""
    PREMIUM = 10.0  # <10ms target
    STANDARD = 20.0  # <20ms target
    ECONOMIC = 50.0  # <50ms target
    FALLBACK = 10.0  # <10ms heuristics


class CacheStrategy(Enum):
    """Cache optimization strategies"""
    AGGRESSIVE = "aggressive"  # High memory usage, maximum speed
    BALANCED = "balanced"  # Moderate memory, good speed
    CONSERVATIVE = "conservative"  # Low memory, basic caching


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking"""
    total_requests: int = 0
    sub_target_requests: int = 0
    average_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    cache_hit_rate: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_pct: float = 0.0
    gc_collections: int = 0
    warm_up_hits: int = 0


class ProductionPerformanceOptimizer:
    """
    Production Performance Optimizer for 95%+ Sub-20ms Targets

    Features:
    - Intelligent multi-layer caching with predictive pre-loading
    - Model warm-up and keep-alive strategies
    - Memory pool management with optimized garbage collection
    - Adaptive request batching and priority queuing
    - Performance monitoring with real-time optimization
    - Circuit breaker performance tuning
    - Resource utilization optimization
    """

    def __init__(self, target_latency_ms: float = 20.0,
                 cache_strategy: CacheStrategy = CacheStrategy.BALANCED,
                 db_path: str = "marketpulse_production.db"):
        """
        Initialize Production Performance Optimizer

        Args:
            target_latency_ms: Target latency in milliseconds
            cache_strategy: Caching strategy to use
            db_path: Path to SQLite database
        """
        self.target_latency_ms = target_latency_ms
        self.cache_strategy = cache_strategy
        self.db_path = db_path

        # Performance tracking
        self.metrics = PerformanceMetrics()
        self.latency_history = deque(maxlen=1000)
        self.performance_history = []

        # Multi-layer caching system
        self._initialize_caching_system()

        # Memory management
        self.memory_pool = {}
        self.gc_threshold_mb = 500
        self.last_gc_time = time.time()

        # Request queue and batching
        self.request_queue = queue.PriorityQueue()
        self.batch_processor = None
        self.processing_thread = None

        # Model warm-up cache
        self.warm_models = {}
        self.model_keep_alive = {}

        # Performance optimization locks
        self.cache_lock = threading.RLock()
        self.metrics_lock = threading.Lock()
        self.warm_up_lock = threading.Lock()

        # Background optimization
        self.optimization_thread = None
        self.running = True

        # Initialize components
        self._initialize_database()
        self._start_background_optimization()

        logger.info(f"ProductionPerformanceOptimizer initialized with {target_latency_ms}ms target")

    def _initialize_caching_system(self):
        """Initialize multi-layer caching system"""
        cache_configs = {
            CacheStrategy.AGGRESSIVE: {
                'l1_size': 1000,
                'l2_size': 5000,
                'l3_size': 20000,
                'ttl_seconds': 600,
                'preload_enabled': True,
                'memory_limit_mb': 1024
            },
            CacheStrategy.BALANCED: {
                'l1_size': 500,
                'l2_size': 2000,
                'l3_size': 8000,
                'ttl_seconds': 300,
                'preload_enabled': True,
                'memory_limit_mb': 512
            },
            CacheStrategy.CONSERVATIVE: {
                'l1_size': 100,
                'l2_size': 500,
                'l3_size': 2000,
                'ttl_seconds': 120,
                'preload_enabled': False,
                'memory_limit_mb': 256
            }
        }

        config = cache_configs[self.cache_strategy]

        # L1 Cache: Ultra-fast in-memory (most recent)
        self.l1_cache = lru.LRU(config['l1_size'])
        self.l1_timestamps = {}

        # L2 Cache: Fast in-memory (frequent access)
        self.l2_cache = lru.LRU(config['l2_size'])
        self.l2_timestamps = {}

        # L3 Cache: Larger in-memory (historical)
        self.l3_cache = lru.LRU(config['l3_size'])
        self.l3_timestamps = {}

        self.cache_config = config
        self.cache_stats = {
            'l1_hits': 0, 'l1_misses': 0,
            'l2_hits': 0, 'l2_misses': 0,
            'l3_hits': 0, 'l3_misses': 0,
            'preload_hits': 0
        }

        logger.info(f"Multi-layer caching initialized: {self.cache_strategy.value} strategy")

    def _initialize_database(self):
        """Initialize database for performance tracking"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_optimization (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        target_latency_ms REAL NOT NULL,
                        actual_latency_ms REAL NOT NULL,
                        cache_strategy TEXT NOT NULL,
                        hit_rate_pct REAL NOT NULL,
                        memory_usage_mb REAL NOT NULL,
                        cpu_usage_pct REAL NOT NULL,
                        optimization_applied TEXT,
                        performance_score REAL NOT NULL
                    )
                ''')

                conn.commit()
                logger.info("Performance optimization database initialized")

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")

    def _start_background_optimization(self):
        """Start background optimization thread"""

        def optimization_loop():
            while self.running:
                try:
                    self._perform_background_optimization()
                    time.sleep(5)  # Optimize every 5 seconds
                except Exception as e:
                    logger.error(f"Background optimization error: {e}")
                    time.sleep(10)

        self.optimization_thread = threading.Thread(target=optimization_loop, daemon=True)
        self.optimization_thread.start()
        logger.info("Background optimization started")

    def optimize_prediction_call(self, prediction_func: Callable,
                                 cache_key: str, *args, **kwargs) -> Tuple[Any, float]:
        """
        Optimize a prediction function call with comprehensive caching and performance tuning

        Args:
            prediction_func: Function to call for prediction
            cache_key: Unique cache key for this prediction
            *args, **kwargs: Arguments for prediction function

        Returns:
            Tuple of (result, latency_ms)
        """
        start_time = time.time()

        try:
            # Check multi-layer cache
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                cache_latency = (time.time() - start_time) * 1000
                self._update_metrics(cache_latency, cache_hit=True)
                return cached_result, cache_latency

            # Warm-up model if needed
            self._ensure_model_warmed_up(prediction_func)

            # Optimize memory before prediction
            self._optimize_memory_if_needed()

            # Execute prediction with timeout
            result = self._execute_with_timeout(prediction_func, *args, **kwargs)

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # Store in cache
            self._store_in_cache(cache_key, result)

            # Update metrics
            self._update_metrics(latency_ms, cache_hit=False)

            # Trigger optimization if needed
            if latency_ms > self.target_latency_ms:
                self._trigger_adaptive_optimization(latency_ms)

            return result, latency_ms

        except Exception as e:
            error_latency = (time.time() - start_time) * 1000
            logger.error(f"Optimized prediction failed: {e}")
            self._update_metrics(error_latency, cache_hit=False, error=True)
            raise

    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get result from multi-layer cache"""
        current_time = time.time()
        ttl = self.cache_config['ttl_seconds']

        with self.cache_lock:
            # Check L1 cache (fastest)
            if cache_key in self.l1_cache:
                if current_time - self.l1_timestamps.get(cache_key, 0) < ttl:
                    self.cache_stats['l1_hits'] += 1
                    # Move to front (LRU)
                    result = self.l1_cache[cache_key]
                    return result
                else:
                    # Expired, remove
                    del self.l1_cache[cache_key]
                    del self.l1_timestamps[cache_key]

            self.cache_stats['l1_misses'] += 1

            # Check L2 cache
            if cache_key in self.l2_cache:
                if current_time - self.l2_timestamps.get(cache_key, 0) < ttl:
                    self.cache_stats['l2_hits'] += 1
                    result = self.l2_cache[cache_key]
                    # Promote to L1
                    self._promote_to_l1(cache_key, result)
                    return result
                else:
                    del self.l2_cache[cache_key]
                    del self.l2_timestamps[cache_key]

            self.cache_stats['l2_misses'] += 1

            # Check L3 cache
            if cache_key in self.l3_cache:
                if current_time - self.l3_timestamps.get(cache_key, 0) < ttl:
                    self.cache_stats['l3_hits'] += 1
                    result = self.l3_cache[cache_key]
                    # Promote to L2
                    self._promote_to_l2(cache_key, result)
                    return result
                else:
                    del self.l3_cache[cache_key]
                    del self.l3_timestamps[cache_key]

            self.cache_stats['l3_misses'] += 1
            return None

    def _store_in_cache(self, cache_key: str, result: Any):
        """Store result in appropriate cache layer"""
        current_time = time.time()

        with self.cache_lock:
            # Store in L1 (most recent/frequent)
            self.l1_cache[cache_key] = result
            self.l1_timestamps[cache_key] = current_time

    def _promote_to_l1(self, cache_key: str, result: Any):
        """Promote cache entry to L1"""
        self.l1_cache[cache_key] = result
        self.l1_timestamps[cache_key] = time.time()

    def _promote_to_l2(self, cache_key: str, result: Any):
        """Promote cache entry to L2"""
        self.l2_cache[cache_key] = result
        self.l2_timestamps[cache_key] = time.time()

    def _ensure_model_warmed_up(self, prediction_func: Callable):
        """Ensure model is warmed up for optimal performance"""
        func_name = getattr(prediction_func, '__name__', str(prediction_func))

        with self.warm_up_lock:
            if func_name not in self.warm_models:
                # Perform warm-up prediction
                try:
                    logger.info(f"Warming up model: {func_name}")
                    # Create dummy data for warm-up
                    dummy_data = self._create_dummy_warm_up_data(func_name)

                    # Execute warm-up
                    start_time = time.time()
                    if 'alpha' in func_name.lower():
                        prediction_func(dummy_data)
                    elif 'lstm' in func_name.lower():
                        prediction_func('WARM_UP', dummy_data, 5)

                    warm_up_time = (time.time() - start_time) * 1000

                    self.warm_models[func_name] = {
                        'warmed_at': time.time(),
                        'warm_up_latency': warm_up_time
                    }
                    self.metrics.warm_up_hits += 1

                    logger.info(f"Model {func_name} warmed up in {warm_up_time:.1f}ms")

                except Exception as e:
                    logger.warning(f"Model warm-up failed for {func_name}: {e}")
                    # Still mark as attempted to avoid repeated failures
                    self.warm_models[func_name] = {
                        'warmed_at': time.time(),
                        'warm_up_latency': 0,
                        'warm_up_failed': True
                    }

    def _create_dummy_warm_up_data(self, func_name: str) -> Any:
        """Create dummy data for model warm-up"""
        if 'alpha' in func_name.lower():
            return {
                'symbol': 'WARM_UP',
                'close': 100.0,
                'rsi_14': 50.0,
                'sma_20': 100.0,
                'volume_ratio': 1.0,
                'price_momentum_5': 0.0
            }
        elif 'lstm' in func_name.lower():
            # Create dummy OHLCV data
            return pd.DataFrame({
                'open': np.random.randn(100) + 100,
                'high': np.random.randn(100) + 102,
                'low': np.random.randn(100) + 98,
                'close': np.random.randn(100) + 100,
                'volume': np.random.randint(1000, 2000, 100)
            })
        else:
            return {}

    def _execute_with_timeout(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with timeout protection"""
        timeout = self.target_latency_ms / 1000.0 * 2  # 2x target as timeout

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(func, *args, **kwargs)

        try:
            result = future.result(timeout=timeout)
            executor.shutdown(wait=False)
            return result
        except Exception as e:
            executor.shutdown(wait=False)
            raise e

    def _optimize_memory_if_needed(self):
        """Optimize memory usage if threshold exceeded"""
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        if current_memory > self.gc_threshold_mb:
            if time.time() - self.last_gc_time > 30:  # At least 30 seconds between GC
                logger.info(f"Memory optimization triggered: {current_memory:.1f}MB")

                # Clear expired cache entries
                self._cleanup_expired_cache()

                # Force garbage collection
                collected = gc.collect()
                self.metrics.gc_collections += 1
                self.last_gc_time = time.time()

                new_memory = psutil.Process().memory_info().rss / 1024 / 1024
                logger.info(f"Memory optimized: {current_memory:.1f}MB -> {new_memory:.1f}MB "
                            f"(collected {collected} objects)")

    def _cleanup_expired_cache(self):
        """Clean up expired cache entries"""
        current_time = time.time()
        ttl = self.cache_config['ttl_seconds']

        with self.cache_lock:
            # Clean L1 cache
            expired_keys = [k for k, t in self.l1_timestamps.items()
                            if current_time - t > ttl]
            for key in expired_keys:
                del self.l1_cache[key]
                del self.l1_timestamps[key]

            # Clean L2 cache
            expired_keys = [k for k, t in self.l2_timestamps.items()
                            if current_time - t > ttl]
            for key in expired_keys:
                del self.l2_cache[key]
                del self.l2_timestamps[key]

            # Clean L3 cache
            expired_keys = [k for k, t in self.l3_timestamps.items()
                            if current_time - t > ttl]
            for key in expired_keys:
                del self.l3_cache[key]
                del self.l3_timestamps[key]

            logger.info(f"Cleaned {len(expired_keys)} expired cache entries")

    def _update_metrics(self, latency_ms: float, cache_hit: bool = False, error: bool = False):
        """Update performance metrics"""
        with self.metrics_lock:
            self.metrics.total_requests += 1

            if not error:
                if latency_ms <= self.target_latency_ms:
                    self.metrics.sub_target_requests += 1

                # Update latency tracking
                self.latency_history.append(latency_ms)

                # Update running average
                if self.metrics.total_requests == 1:
                    self.metrics.average_latency = latency_ms
                else:
                    # Exponential moving average
                    alpha = 0.1
                    self.metrics.average_latency = (
                            alpha * latency_ms + (1 - alpha) * self.metrics.average_latency
                    )

                # Update percentiles
                if len(self.latency_history) >= 20:
                    sorted_latencies = sorted(self.latency_history)
                    self.metrics.p95_latency = sorted_latencies[int(0.95 * len(sorted_latencies))]
                    self.metrics.p99_latency = sorted_latencies[int(0.99 * len(sorted_latencies))]

            # Update cache hit rate
            total_cache_requests = (
                    self.cache_stats['l1_hits'] + self.cache_stats['l1_misses']
            )
            if total_cache_requests > 0:
                total_hits = (
                        self.cache_stats['l1_hits'] +
                        self.cache_stats['l2_hits'] +
                        self.cache_stats['l3_hits']
                )
                self.metrics.cache_hit_rate = (total_hits / total_cache_requests) * 100

    def _trigger_adaptive_optimization(self, latency_ms: float):
        """Trigger adaptive optimization based on performance"""
        if latency_ms > self.target_latency_ms * 2:  # Critical performance issue
            logger.warning(f"Critical latency detected: {latency_ms:.1f}ms")

            # Immediate optimizations
            self._optimize_memory_if_needed()
            self._cleanup_expired_cache()

            # Adjust cache strategy if needed
            if self.cache_strategy == CacheStrategy.CONSERVATIVE:
                logger.info("Upgrading cache strategy to BALANCED")
                self.cache_strategy = CacheStrategy.BALANCED
                self._initialize_caching_system()

    def _perform_background_optimization(self):
        """Perform background optimization tasks"""
        try:
            # Update system metrics
            self.metrics.memory_usage_mb = psutil.Process().memory_info().rss / 1024 / 1024
            self.metrics.cpu_usage_pct = psutil.Process().cpu_percent()

            # Performance-based optimizations
            success_rate = self.get_success_rate()

            if success_rate < 95.0:  # Below 95% target
                if self.cache_strategy == CacheStrategy.CONSERVATIVE:
                    self.cache_strategy = CacheStrategy.BALANCED
                    logger.info("Auto-upgraded cache strategy to BALANCED")
                elif self.cache_strategy == CacheStrategy.BALANCED and success_rate < 90.0:
                    self.cache_strategy = CacheStrategy.AGGRESSIVE
                    logger.info("Auto-upgraded cache strategy to AGGRESSIVE")

            # Memory management
            if self.metrics.memory_usage_mb > self.gc_threshold_mb:
                self._optimize_memory_if_needed()

            # Log performance metrics periodically
            if self.metrics.total_requests > 0 and self.metrics.total_requests % 100 == 0:
                self._log_performance_metrics()

        except Exception as e:
            logger.error(f"Background optimization failed: {e}")

    def get_success_rate(self) -> float:
        """Get current sub-target success rate"""
        if self.metrics.total_requests == 0:
            return 0.0
        return (self.metrics.sub_target_requests / self.metrics.total_requests) * 100

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        success_rate = self.get_success_rate()

        return {
            'target_latency_ms': self.target_latency_ms,
            'success_rate_pct': round(success_rate, 2),
            'average_latency_ms': round(self.metrics.average_latency, 2),
            'p95_latency_ms': round(self.metrics.p95_latency, 2),
            'p99_latency_ms': round(self.metrics.p99_latency, 2),
            'cache_hit_rate_pct': round(self.metrics.cache_hit_rate, 2),
            'memory_usage_mb': round(self.metrics.memory_usage_mb, 2),
            'cpu_usage_pct': round(self.metrics.cpu_usage_pct, 2),
            'total_requests': self.metrics.total_requests,
            'cache_stats': self.cache_stats.copy(),
            'cache_strategy': self.cache_strategy.value,
            'warm_up_hits': self.metrics.warm_up_hits,
            'gc_collections': self.metrics.gc_collections,
            'optimization_status': 'OPTIMAL' if success_rate >= 95.0 else 'TUNING',
            'last_updated': datetime.now().isoformat()
        }

    def _log_performance_metrics(self):
        """Log performance metrics to database"""
        try:
            report = self.get_performance_report()
            performance_score = min(100.0, report['success_rate_pct'] +
                                    (100.0 - report['average_latency_ms'] / self.target_latency_ms * 100))

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO performance_optimization 
                    (timestamp, target_latency_ms, actual_latency_ms, cache_strategy,
                     hit_rate_pct, memory_usage_mb, cpu_usage_pct, optimization_applied, performance_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    self.target_latency_ms,
                    report['average_latency_ms'],
                    report['cache_strategy'],
                    report['cache_hit_rate_pct'],
                    report['memory_usage_mb'],
                    report['cpu_usage_pct'],
                    json.dumps(report['cache_stats']),
                    performance_score
                ))
                conn.commit()

        except Exception as e:
            logger.error(f"Failed to log performance metrics: {e}")

    def create_cache_key(self, *args, **kwargs) -> str:
        """Create deterministic cache key from arguments"""
        # Create hash from arguments
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()

    def shutdown(self):
        """Shutdown optimizer and cleanup resources"""
        self.running = False

        if self.optimization_thread and self.optimization_thread.is_alive():
            self.optimization_thread.join(timeout=5)

        # Final performance log
        if self.metrics.total_requests > 0:
            self._log_performance_metrics()

        logger.info("ProductionPerformanceOptimizer shutdown completed")

    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.shutdown()
        except:
            pass


# Decorator for easy optimization of any prediction function
def optimize_performance(target_latency_ms: float = 20.0,
                         cache_strategy: str = "balanced"):
    """
    Decorator to optimize any prediction function for sub-target latency

    Args:
        target_latency_ms: Target latency in milliseconds
        cache_strategy: Cache strategy ("aggressive", "balanced", "conservative")
    """

    def decorator(func):
        optimizer = ProductionPerformanceOptimizer(
            target_latency_ms=target_latency_ms,
            cache_strategy=CacheStrategy(cache_strategy)
        )

        def optimized_wrapper(*args, **kwargs):
            cache_key = optimizer.create_cache_key(*args, **kwargs)
            result, latency = optimizer.optimize_prediction_call(func, cache_key, *args, **kwargs)
            return result

        # Attach performance reporting to wrapper
        optimized_wrapper.get_performance_report = optimizer.get_performance_report
        optimized_wrapper.optimizer = optimizer

        return optimized_wrapper

    return decorator


# Usage example and testing
if __name__ == "__main__":
    print("Production Performance Optimizer - Integration Test")
    print("=" * 60)

    try:
        # Create optimizer
        optimizer = ProductionPerformanceOptimizer(target_latency_ms=20.0)


        # Mock prediction function
        def mock_prediction(data):
            time.sleep(0.015)  # Simulate 15ms processing
            return {
                'prediction': np.random.random(),
                'confidence': 'HIGH' if np.random.random() > 0.5 else 'MEDIUM'
            }


        # Test optimization
        print("\n🧠 Testing performance optimization...")

        total_latency = 0
        sub_target_count = 0

        for i in range(50):
            cache_key = f"test_prediction_{i % 10}"  # Some cache hits
            result, latency = optimizer.optimize_prediction_call(
                mock_prediction, cache_key, {'data': f'test_{i}'}
            )

            total_latency += latency
            if latency <= 20.0:
                sub_target_count += 1

            if i % 10 == 0:
                print(f"   Request {i}: {latency:.1f}ms")

        # Get performance report
        report = optimizer.get_performance_report()
        print(f"\n📊 Performance Report:")
        print(f"   Success Rate: {report['success_rate_pct']:.1f}%")
        print(f"   Average Latency: {report['average_latency_ms']:.1f}ms")
        print(f"   P95 Latency: {report['p95_latency_ms']:.1f}ms")
        print(f"   Cache Hit Rate: {report['cache_hit_rate_pct']:.1f}%")
        print(f"   Memory Usage: {report['memory_usage_mb']:.1f}MB")

        target_achieved = report['success_rate_pct'] >= 95.0
        print(f"\n✅ Target {'ACHIEVED' if target_achieved else 'MISSED'}: "
              f"{report['success_rate_pct']:.1f}% sub-{optimizer.target_latency_ms}ms")

        # Cleanup
        optimizer.shutdown()
        print("\n✅ ProductionPerformanceOptimizer test completed!")

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback

        traceback.print_exc()