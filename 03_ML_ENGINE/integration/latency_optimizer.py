# 03_ML_ENGINE/optimization/latency_optimizer.py
"""
Latency Optimization Framework - Phase 2, Step 4
Advanced performance optimization for achieving sub-20ms ML prediction targets

Location: #03_ML_ENGINE/optimization/latency_optimizer.py

This framework provides:
- Sub-20ms latency optimization strategies
- Intelligent caching with prediction freshness
- Model warm-up and pre-loading techniques
- Batch processing optimization
- Memory management and garbage collection optimization
- CPU and threading optimization for ML operations
"""

import time
import threading
import logging
import gc
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict
import numpy as np
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizationLevel(Enum):
    """Performance optimization levels"""
    CONSERVATIVE = "conservative"  # Safe optimizations
    AGGRESSIVE = "aggressive"  # More aggressive, potential trade-offs
    EXTREME = "extreme"  # Maximum performance, minimal safety margins


@dataclass
class LatencyTarget:
    """Latency performance targets"""
    p50_target_ms: float = 15.0  # 50th percentile target
    p95_target_ms: float = 25.0  # 95th percentile target
    p99_target_ms: float = 40.0  # 99th percentile target
    max_acceptable_ms: float = 100.0  # Absolute maximum


@dataclass
class CacheEntry:
    """Smart cache entry with prediction metadata"""
    prediction: Any
    timestamp: datetime
    input_hash: str
    confidence: float
    access_count: int = 0
    last_access: datetime = field(default_factory=datetime.now)
    freshness_score: float = 1.0


class SmartPredictionCache:
    """Intelligent prediction cache with freshness scoring"""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        """Initialize smart cache"""
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.RLock()

        # Freshness parameters
        self.market_hours_multiplier = 0.8  # Predictions less fresh during market hours
        self.volatility_multiplier = 0.9  # Predictions less fresh in volatile markets

    def _hash_input(self, input_data: Dict) -> str:
        """Create hash for input data"""
        # Create stable hash from input data
        serialized = pickle.dumps(sorted(input_data.items()), protocol=2)
        return hashlib.md5(serialized).hexdigest()

    def _calculate_freshness(self, entry: CacheEntry) -> float:
        """Calculate prediction freshness score (0-1)"""
        age_seconds = (datetime.now() - entry.timestamp).total_seconds()

        # Base freshness decay
        base_freshness = max(0, 1 - (age_seconds / self.ttl_seconds))

        # Apply market conditions multipliers
        current_hour = datetime.now().hour
        if 9 <= current_hour <= 16:  # Market hours (approximate)
            base_freshness *= self.market_hours_multiplier

        return base_freshness

    def get(self, input_data: Dict, min_confidence: float = 0.0) -> Optional[Any]:
        """Get cached prediction if fresh enough"""
        input_hash = self._hash_input(input_data)

        with self.lock:
            if input_hash in self.cache:
                entry = self.cache[input_hash]

                # Check if entry is still valid
                freshness = self._calculate_freshness(entry)

                if freshness > 0.1 and entry.confidence >= min_confidence:  # Minimum freshness threshold
                    # Move to end (LRU)
                    entry.access_count += 1
                    entry.last_access = datetime.now()
                    entry.freshness_score = freshness

                    self.cache.move_to_end(input_hash)
                    self.hit_count += 1

                    logger.debug(f"Cache HIT: freshness={freshness:.2f}, confidence={entry.confidence:.2f}")
                    return entry.prediction
                else:
                    # Remove stale entry
                    del self.cache[input_hash]

            self.miss_count += 1
            return None

    def put(self, input_data: Dict, prediction: Any, confidence: float):
        """Cache a new prediction"""
        input_hash = self._hash_input(input_data)

        with self.lock:
            # Remove oldest entries if cache is full
            while len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)

            entry = CacheEntry(
                prediction=prediction,
                timestamp=datetime.now(),
                input_hash=input_hash,
                confidence=confidence
            )

            self.cache[input_hash] = entry

    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0

        with self.lock:
            avg_freshness = np.mean([e.freshness_score for e in self.cache.values()]) if self.cache else 0

            return {
                'total_requests': total_requests,
                'hit_count': self.hit_count,
                'miss_count': self.miss_count,
                'hit_rate': hit_rate,
                'cache_size': len(self.cache),
                'avg_freshness': avg_freshness,
                'speedup_estimate': 1 + (hit_rate * 10)  # Estimate 10x speedup on cache hit
            }


class ModelWarmupManager:
    """Manages model warm-up to reduce cold start latency"""

    def __init__(self):
        """Initialize warmup manager"""
        self.warmed_models = set()
        self.warmup_data_cache = {}
        self.executor = ThreadPoolExecutor(max_workers=2)

    def warmup_model(self, model_func: Callable, sample_input: Dict,
                     warmup_iterations: int = 3):
        """Warm up a model with sample predictions"""
        model_name = getattr(model_func, '__name__', 'unknown')

        if model_name in self.warmed_models:
            return  # Already warmed up

        logger.info(f"🔥 Warming up model: {model_name}")

        try:
            # Perform warm-up predictions
            warmup_times = []
            for i in range(warmup_iterations):
                start_time = time.time()
                try:
                    result = model_func(sample_input)
                    warmup_time = (time.time() - start_time) * 1000
                    warmup_times.append(warmup_time)
                    logger.debug(f"Warmup {i + 1}: {warmup_time:.1f}ms")
                except Exception as e:
                    logger.warning(f"Warmup iteration {i + 1} failed: {e}")

            if warmup_times:
                avg_warmup_time = np.mean(warmup_times)
                logger.info(f"✅ Model {model_name} warmed up: {avg_warmup_time:.1f}ms average")
                self.warmed_models.add(model_name)

        except Exception as e:
            logger.error(f"❌ Failed to warm up model {model_name}: {e}")

    def warmup_all_models_async(self, models_and_samples: List[Tuple[Callable, Dict]]):
        """Warm up multiple models asynchronously"""
        futures = []

        for model_func, sample_input in models_and_samples:
            future = self.executor.submit(self.warmup_model, model_func, sample_input)
            futures.append(future)

        # Wait for all warmups to complete
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Async warmup failed: {e}")


class LatencyOptimizer:
    """Main latency optimization engine"""

    def __init__(self, optimization_level: OptimizationLevel = OptimizationLevel.CONSERVATIVE,
                 targets: Optional[LatencyTarget] = None):
        """Initialize latency optimizer"""
        self.optimization_level = optimization_level
        self.targets = targets or LatencyTarget()

        # Optimization components
        self.cache = SmartPredictionCache()
        self.warmup_manager = ModelWarmupManager()

        # Performance tracking
        self.latency_history = []
        self.max_history = 1000
        self.lock = threading.Lock()

        # Memory management
        self.gc_threshold = 100  # MB
        self.last_gc = datetime.now()

        logger.info(f"⚡ Latency Optimizer initialized: {optimization_level.value} mode")

    def optimize_prediction(self, model_func: Callable, input_data: Dict,
                            min_confidence: float = 0.5) -> Tuple[Any, Dict[str, Any]]:
        """Optimize a single prediction with caching and performance monitoring"""
        start_time = time.time()

        # Try cache first
        cached_result = self.cache.get(input_data, min_confidence)
        if cached_result is not None:
            latency_ms = (time.time() - start_time) * 1000
            self._record_latency(latency_ms, cached=True)

            return cached_result, {
                'latency_ms': latency_ms,
                'cached': True,
                'optimization': 'cache_hit'
            }

        # Memory management
        self._manage_memory()

        # Make actual prediction
        prediction_start = time.time()
        try:
            prediction = model_func(input_data)
            prediction_latency = (time.time() - prediction_start) * 1000

            # Cache the result
            confidence = prediction.get('confidence', 0.5) if isinstance(prediction, dict) else 0.5
            self.cache.put(input_data, prediction, confidence)

            total_latency = (time.time() - start_time) * 1000
            self._record_latency(total_latency, cached=False)

            return prediction, {
                'latency_ms': total_latency,
                'prediction_latency_ms': prediction_latency,
                'cached': False,
                'optimization': self._get_optimization_strategy(total_latency)
            }

        except Exception as e:
            error_latency = (time.time() - start_time) * 1000
            logger.error(f"Prediction failed after {error_latency:.1f}ms: {e}")
            raise

    def batch_optimize_predictions(self, model_func: Callable,
                                   input_batch: List[Dict]) -> List[Tuple[Any, Dict]]:
        """Batch optimization for multiple predictions"""
        if len(input_batch) <= 1:
            # Use single prediction for small batches
            return [self.optimize_prediction(model_func, input_batch[0]) if input_batch else []]

        logger.debug(f"🔄 Batch processing {len(input_batch)} predictions")
        start_time = time.time()

        # Separate cached and non-cached inputs
        cached_results = {}
        uncached_inputs = []

        for i, input_data in enumerate(input_batch):
            cached_result = self.cache.get(input_data)
            if cached_result is not None:
                cached_results[i] = (cached_result, {'cached': True})
            else:
                uncached_inputs.append((i, input_data))

        # Process uncached inputs
        if uncached_inputs:
            if len(uncached_inputs) == 1:
                # Single prediction
                idx, input_data = uncached_inputs[0]
                try:
                    prediction = model_func(input_data)
                    confidence = prediction.get('confidence', 0.5) if isinstance(prediction, dict) else 0.5
                    self.cache.put(input_data, prediction, confidence)
                    cached_results[idx] = (prediction, {'cached': False})
                except Exception as e:
                    logger.error(f"Single prediction in batch failed: {e}")
                    cached_results[idx] = (None, {'error': str(e)})
            else:
                # True batch processing would go here
                # For now, process individually but in parallel
                with ThreadPoolExecutor(max_workers=min(4, len(uncached_inputs))) as executor:
                    future_to_idx = {}

                    for idx, input_data in uncached_inputs:
                        future = executor.submit(model_func, input_data)
                        future_to_idx[future] = (idx, input_data)

                    for future in as_completed(future_to_idx):
                        idx, input_data = future_to_idx[future]
                        try:
                            prediction = future.result()
                            confidence = prediction.get('confidence', 0.5) if isinstance(prediction, dict) else 0.5
                            self.cache.put(input_data, prediction, confidence)
                            cached_results[idx] = (prediction, {'cached': False})
                        except Exception as e:
                            logger.error(f"Batch prediction {idx} failed: {e}")
                            cached_results[idx] = (None, {'error': str(e)})

        # Assemble results in original order
        results = []
        for i in range(len(input_batch)):
            if i in cached_results:
                results.append(cached_results[i])
            else:
                results.append((None, {'error': 'missing_result'}))

        total_latency = (time.time() - start_time) * 1000
        logger.debug(f"✅ Batch completed: {total_latency:.1f}ms for {len(input_batch)} predictions")

        return results

    def _record_latency(self, latency_ms: float, cached: bool = False):
        """Record latency measurement"""
        with self.lock:
            self.latency_history.append({
                'latency_ms': latency_ms,
                'timestamp': datetime.now(),
                'cached': cached
            })

            # Keep history size manageable
            if len(self.latency_history) > self.max_history:
                self.latency_history = self.latency_history[-self.max_history:]

    def _get_optimization_strategy(self, latency_ms: float) -> str:
        """Determine which optimization strategy was most effective"""
        if latency_ms <= self.targets.p50_target_ms:
            return 'optimal'
        elif latency_ms <= self.targets.p95_target_ms:
            return 'acceptable'
        elif latency_ms <= self.targets.p99_target_ms:
            return 'degraded'
        else:
            return 'poor'

    def _manage_memory(self):
        """Intelligent memory management"""
        current_time = datetime.now()

        # Check if we should run garbage collection
        if (current_time - self.last_gc).seconds > 60:  # Every minute
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            if memory_usage > self.gc_threshold:
                logger.debug(f"🗑️ Running GC: memory usage {memory_usage:.1f}MB")
                gc.collect()
                self.last_gc = current_time

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        with self.lock:
            if not self.latency_history:
                return {'status': 'no_data'}

            # Calculate latency statistics
            latencies = [entry['latency_ms'] for entry in self.latency_history]
            cached_latencies = [entry['latency_ms'] for entry in self.latency_history if entry['cached']]
            uncached_latencies = [entry['latency_ms'] for entry in self.latency_history if not entry['cached']]

            # Performance metrics
            p50 = np.percentile(latencies, 50)
            p95 = np.percentile(latencies, 95)
            p99 = np.percentile(latencies, 99)

            # Target achievement
            sub_20ms_count = sum(1 for l in latencies if l < 20)
            sub_20ms_percentage = (sub_20ms_count / len(latencies)) * 100

            # Cache statistics
            cache_stats = self.cache.get_stats()

            return {
                'total_predictions': len(self.latency_history),
                'latency_stats': {
                    'p50_ms': p50,
                    'p95_ms': p95,
                    'p99_ms': p99,
                    'mean_ms': np.mean(latencies),
                    'std_ms': np.std(latencies)
                },
                'target_achievement': {
                    'p50_target_met': p50 <= self.targets.p50_target_ms,
                    'p95_target_met': p95 <= self.targets.p95_target_ms,
                    'p99_target_met': p99 <= self.targets.p99_target_ms,
                    'sub_20ms_percentage': sub_20ms_percentage
                },
                'cache_performance': cache_stats,
                'optimization_level': self.optimization_level.value,
                'cached_predictions': len(cached_latencies),
                'uncached_predictions': len(uncached_latencies),
                'avg_cached_latency_ms': np.mean(cached_latencies) if cached_latencies else 0,
                'avg_uncached_latency_ms': np.mean(uncached_latencies) if uncached_latencies else 0
            }

    def tune_for_latency_target(self, target_ms: float):
        """Auto-tune optimizer for specific latency target"""
        logger.info(f"🎯 Tuning optimizer for {target_ms}ms target")

        current_performance = self.get_performance_report()

        if current_performance.get('status') == 'no_data':
            logger.warning("⚠️ No performance data available for tuning")
            return

        current_p95 = current_performance['latency_stats']['p95_ms']

        if current_p95 > target_ms:
            # Need to be more aggressive
            if target_ms <= 15:
                self.optimization_level = OptimizationLevel.EXTREME
                self.cache.ttl_seconds = 600  # Longer cache
                logger.info("⚡ Switching to EXTREME optimization mode")
            elif target_ms <= 25:
                self.optimization_level = OptimizationLevel.AGGRESSIVE
                self.cache.ttl_seconds = 450
                logger.info("🔥 Switching to AGGRESSIVE optimization mode")

        # Update targets
        self.targets.p95_target_ms = target_ms
        self.targets.p50_target_ms = target_ms * 0.6
        self.targets.p99_target_ms = target_ms * 1.6


def create_optimized_predictor(model_func: Callable,
                               optimization_level: OptimizationLevel = OptimizationLevel.CONSERVATIVE) -> Callable:
    """Create an optimized version of a prediction function"""
    optimizer = LatencyOptimizer(optimization_level)

    def optimized_predictor(input_data: Dict) -> Any:
        """Optimized prediction function"""
        result, metadata = optimizer.optimize_prediction(model_func, input_data)

        # Log performance if debugging
        if logger.isEnabledFor(logging.DEBUG):
            optimization = metadata.get('optimization', 'unknown')
            latency = metadata.get('latency_ms', 0)
            cached = metadata.get('cached', False)
            cache_indicator = '🟢' if cached else '🔴'

            logger.debug(f"{cache_indicator} Prediction: {latency:.1f}ms - {optimization}")

        return result

    # Attach optimizer for external access
    optimized_predictor.optimizer = optimizer

    return optimized_predictor


if __name__ == "__main__":
    # Demo the latency optimization framework
    print("⚡ Latency Optimization Framework Demo")
    print("=" * 50)


    # Create a mock model function for testing
    def mock_ml_model(input_data: Dict) -> Dict:
        """Mock ML model with variable latency"""
        # Simulate variable processing time
        processing_time = np.random.exponential(0.015)  # Average 15ms
        time.sleep(processing_time)

        return {
            'signal': np.random.choice(['BUY', 'SELL', 'HOLD']),
            'confidence': np.random.uniform(0.4, 0.9),
            'processing_time_ms': processing_time * 1000
        }


    # Create optimized version
    optimized_model = create_optimized_predictor(mock_ml_model, OptimizationLevel.AGGRESSIVE)

    # Test with various inputs
    test_inputs = [
        {'symbol': 'AAPL', 'price': 150.0},
        {'symbol': 'GOOGL', 'price': 2500.0},
        {'symbol': 'AAPL', 'price': 150.0},  # Duplicate for cache test
        {'symbol': 'MSFT', 'price': 300.0},
        {'symbol': 'AAPL', 'price': 150.0},  # Another duplicate
    ]

    print("\n🧪 Testing optimized predictions:")
    for i, test_input in enumerate(test_inputs):
        start_time = time.time()
        try:
            result = optimized_model(test_input)
            latency_ms = (time.time() - start_time) * 1000
            print(f"  {i + 1}. {test_input['symbol']}: {latency_ms:.1f}ms - "
                  f"{result['signal']} (conf: {result['confidence']:.2f})")
        except Exception as e:
            print(f"  {i + 1}. {test_input['symbol']}: ERROR - {e}")

    # Show performance report
    print("\n📊 Performance Report:")
    report = optimized_model.optimizer.get_performance_report()

    if report.get('status') != 'no_data':
        print(f"  Total Predictions: {report['total_predictions']}")
        print(f"  P50 Latency: {report['latency_stats']['p50_ms']:.1f}ms")
        print(f"  P95 Latency: {report['latency_stats']['p95_ms']:.1f}ms")
        print(f"  Sub-20ms Rate: {report['target_achievement']['sub_20ms_percentage']:.1f}%")
        print(f"  Cache Hit Rate: {report['cache_performance']['hit_rate']:.1%}")
        print(f"  Cache Speedup: {report['cache_performance']['speedup_estimate']:.1f}x")

    print("\n✅ Latency optimization demo completed!")