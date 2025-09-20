# 03_ML_ENGINE/performance/performance_logger.py
"""
Performance Logging Decorator System for ML Signal Enhancer - Phase 2, Step 3
Comprehensive performance monitoring and metrics collection

Location: #03_ML_ENGINE/performance/performance_logger.py

This module provides:
- Method execution time tracking
- Memory usage monitoring
- Model accuracy tracking
- System resource monitoring
- Performance trend analysis
- Anomaly detection
- Automated reporting
"""

import functools
import time
import psutil
import logging
import json
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceLevel(Enum):
    """Performance level classifications"""
    EXCELLENT = "excellent"  # < 1ms
    GOOD = "good"  # 1-5ms
    ACCEPTABLE = "acceptable"  # 5-20ms
    SLOW = "slow"  # 20-100ms
    CRITICAL = "critical"  # > 100ms


@dataclass
class PerformanceMetric:
    """Individual performance measurement"""
    timestamp: datetime
    method_name: str
    execution_time_ms: float
    memory_usage_mb: float
    cpu_percent: float
    success: bool
    error_message: Optional[str] = None
    input_size: Optional[int] = None
    output_size: Optional[int] = None
    custom_metrics: Optional[Dict[str, Any]] = None


@dataclass
class PerformanceReport:
    """Performance analysis report"""
    method_name: str
    total_calls: int
    avg_execution_time_ms: float
    min_execution_time_ms: float
    max_execution_time_ms: float
    success_rate: float
    performance_level: PerformanceLevel
    memory_stats: Dict[str, float]
    trend_analysis: Dict[str, Any]
    anomalies_detected: List[str]


class PerformanceDatabase:
    """SQLite database for performance metrics storage"""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize performance database"""
        if db_path is None:
            # Create in project's data storage
            db_dir = Path("10_DATA_STORAGE/performance")
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(db_dir / "marketpulse_performance.db")

        self.db_path = db_path
        self._init_database()
        logger.info(f"📊 Performance database initialized: {db_path}")

    def _init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    method_name TEXT NOT NULL,
                    execution_time_ms REAL NOT NULL,
                    memory_usage_mb REAL NOT NULL,
                    cpu_percent REAL NOT NULL,
                    success INTEGER NOT NULL,
                    error_message TEXT,
                    input_size INTEGER,
                    output_size INTEGER,
                    custom_metrics TEXT
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_method_timestamp 
                ON performance_metrics(method_name, timestamp)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON performance_metrics(timestamp)
            """)

    def store_metric(self, metric: PerformanceMetric):
        """Store performance metric in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                custom_metrics_json = json.dumps(metric.custom_metrics) if metric.custom_metrics else None

                conn.execute("""
                    INSERT INTO performance_metrics (
                        timestamp, method_name, execution_time_ms, memory_usage_mb,
                        cpu_percent, success, error_message, input_size, output_size, custom_metrics
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metric.timestamp.isoformat(),
                    metric.method_name,
                    metric.execution_time_ms,
                    metric.memory_usage_mb,
                    metric.cpu_percent,
                    1 if metric.success else 0,
                    metric.error_message,
                    metric.input_size,
                    metric.output_size,
                    custom_metrics_json
                ))
        except Exception as e:
            logger.error(f"❌ Failed to store performance metric: {e}")

    def get_metrics(self, method_name: Optional[str] = None,
                    hours_back: int = 24) -> List[PerformanceMetric]:
        """Retrieve performance metrics from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cutoff_time = datetime.now() - timedelta(hours=hours_back)

                if method_name:
                    cursor = conn.execute("""
                        SELECT * FROM performance_metrics 
                        WHERE method_name = ? AND timestamp >= ?
                        ORDER BY timestamp DESC
                    """, (method_name, cutoff_time.isoformat()))
                else:
                    cursor = conn.execute("""
                        SELECT * FROM performance_metrics 
                        WHERE timestamp >= ?
                        ORDER BY timestamp DESC
                    """, (cutoff_time.isoformat(),))

                metrics = []
                for row in cursor.fetchall():
                    custom_metrics = json.loads(row[10]) if row[10] else None

                    metric = PerformanceMetric(
                        timestamp=datetime.fromisoformat(row[1]),
                        method_name=row[2],
                        execution_time_ms=row[3],
                        memory_usage_mb=row[4],
                        cpu_percent=row[5],
                        success=bool(row[6]),
                        error_message=row[7],
                        input_size=row[8],
                        output_size=row[9],
                        custom_metrics=custom_metrics
                    )
                    metrics.append(metric)

                return metrics

        except Exception as e:
            logger.error(f"❌ Failed to retrieve performance metrics: {e}")
            return []


class PerformanceAnalyzer:
    """Analyze performance metrics and generate insights"""

    def __init__(self, database: PerformanceDatabase):
        """Initialize performance analyzer"""
        self.database = database
        logger.info("📈 Performance analyzer initialized")

    def analyze_method_performance(self, method_name: str, hours_back: int = 24) -> PerformanceReport:
        """Analyze performance for a specific method"""
        metrics = self.database.get_metrics(method_name, hours_back)

        if not metrics:
            return PerformanceReport(
                method_name=method_name,
                total_calls=0,
                avg_execution_time_ms=0.0,
                min_execution_time_ms=0.0,
                max_execution_time_ms=0.0,
                success_rate=0.0,
                performance_level=PerformanceLevel.GOOD,
                memory_stats={},
                trend_analysis={},
                anomalies_detected=[]
            )

        # Basic statistics
        execution_times = [m.execution_time_ms for m in metrics]
        memory_usage = [m.memory_usage_mb for m in metrics]
        success_count = sum(1 for m in metrics if m.success)

        avg_time = np.mean(execution_times)
        min_time = np.min(execution_times)
        max_time = np.max(execution_times)
        success_rate = success_count / len(metrics)

        # Performance level classification
        if avg_time < 1:
            perf_level = PerformanceLevel.EXCELLENT
        elif avg_time < 5:
            perf_level = PerformanceLevel.GOOD
        elif avg_time < 20:
            perf_level = PerformanceLevel.ACCEPTABLE
        elif avg_time < 100:
            perf_level = PerformanceLevel.SLOW
        else:
            perf_level = PerformanceLevel.CRITICAL

        # Memory statistics
        memory_stats = {
            'avg_memory_mb': np.mean(memory_usage),
            'min_memory_mb': np.min(memory_usage),
            'max_memory_mb': np.max(memory_usage),
            'memory_std_mb': np.std(memory_usage)
        }

        # Trend analysis
        trend_analysis = self._analyze_trends(metrics)

        # Anomaly detection
        anomalies = self._detect_anomalies(metrics)

        return PerformanceReport(
            method_name=method_name,
            total_calls=len(metrics),
            avg_execution_time_ms=avg_time,
            min_execution_time_ms=min_time,
            max_execution_time_ms=max_time,
            success_rate=success_rate,
            performance_level=perf_level,
            memory_stats=memory_stats,
            trend_analysis=trend_analysis,
            anomalies_detected=anomalies
        )

    def _analyze_trends(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        if len(metrics) < 10:
            return {'trend': 'insufficient_data'}

        # Sort by timestamp
        sorted_metrics = sorted(metrics, key=lambda m: m.timestamp)

        # Calculate moving averages
        execution_times = [m.execution_time_ms for m in sorted_metrics]
        recent_avg = np.mean(execution_times[-10:])  # Last 10 calls
        older_avg = np.mean(execution_times[:-10])  # Earlier calls

        # Determine trend direction
        if recent_avg > older_avg * 1.1:
            trend = 'deteriorating'
        elif recent_avg < older_avg * 0.9:
            trend = 'improving'
        else:
            trend = 'stable'

        # Calculate volatility
        volatility = np.std(execution_times) / np.mean(execution_times)

        return {
            'trend': trend,
            'recent_avg_ms': recent_avg,
            'older_avg_ms': older_avg,
            'change_percent': ((recent_avg - older_avg) / older_avg) * 100,
            'volatility': volatility
        }

    def _detect_anomalies(self, metrics: List[PerformanceMetric]) -> List[str]:
        """Detect performance anomalies"""
        anomalies = []

        if not metrics:
            return anomalies

        execution_times = [m.execution_time_ms for m in metrics]

        # Statistical anomaly detection (outliers beyond 2 standard deviations)
        mean_time = np.mean(execution_times)
        std_time = np.std(execution_times)

        outliers = [t for t in execution_times if abs(t - mean_time) > 2 * std_time]
        if outliers:
            anomalies.append(f"Performance outliers detected: {len(outliers)} calls beyond 2σ")

        # Error rate anomaly
        error_rate = 1 - (sum(1 for m in metrics if m.success) / len(metrics))
        if error_rate > 0.05:  # More than 5% error rate
            anomalies.append(f"High error rate: {error_rate:.1%}")

        # Memory usage spike
        memory_usage = [m.memory_usage_mb for m in metrics]
        mean_memory = np.mean(memory_usage)
        max_memory = np.max(memory_usage)

        if max_memory > mean_memory * 2:
            anomalies.append(f"Memory usage spike detected: {max_memory:.1f}MB")

        # Performance degradation
        if len(metrics) >= 20:
            recent_times = execution_times[-10:]
            older_times = execution_times[:10]

            if np.mean(recent_times) > np.mean(older_times) * 1.5:
                anomalies.append("Performance degradation trend detected")

        return anomalies

    def generate_system_report(self, hours_back: int = 24) -> Dict[str, Any]:
        """Generate comprehensive system performance report"""
        all_metrics = self.database.get_metrics(hours_back=hours_back)

        if not all_metrics:
            return {'status': 'no_data', 'message': 'No performance data available'}

        # Group by method
        methods = {}
        for metric in all_metrics:
            if metric.method_name not in methods:
                methods[metric.method_name] = []
            methods[metric.method_name].append(metric)

        # Analyze each method
        method_reports = {}
        for method_name, method_metrics in methods.items():
            method_reports[method_name] = self.analyze_method_performance(method_name, hours_back)

        # System-wide statistics
        total_calls = len(all_metrics)
        avg_system_time = np.mean([m.execution_time_ms for m in all_metrics])
        system_success_rate = sum(1 for m in all_metrics if m.success) / total_calls

        # System health score (0-100)
        health_score = self._calculate_health_score(method_reports)

        return {
            'status': 'success',
            'report_generated': datetime.now().isoformat(),
            'time_period_hours': hours_back,
            'system_summary': {
                'total_calls': total_calls,
                'avg_execution_time_ms': avg_system_time,
                'success_rate': system_success_rate,
                'health_score': health_score,
                'unique_methods': len(method_reports)
            },
            'method_reports': {name: asdict(report) for name, report in method_reports.items()},
            'recommendations': self._generate_recommendations(method_reports)
        }

    def _calculate_health_score(self, method_reports: Dict[str, PerformanceReport]) -> int:
        """Calculate system health score (0-100)"""
        if not method_reports:
            return 50

        scores = []

        for report in method_reports.values():
            # Performance score based on level
            perf_scores = {
                PerformanceLevel.EXCELLENT: 100,
                PerformanceLevel.GOOD: 85,
                PerformanceLevel.ACCEPTABLE: 70,
                PerformanceLevel.SLOW: 50,
                PerformanceLevel.CRITICAL: 20
            }

            perf_score = perf_scores.get(report.performance_level, 50)

            # Success rate score
            success_score = report.success_rate * 100

            # Anomaly penalty
            anomaly_penalty = len(report.anomalies_detected) * 5

            method_score = max(0, min(100, (perf_score + success_score) / 2 - anomaly_penalty))
            scores.append(method_score)

        return int(np.mean(scores))

    def _generate_recommendations(self, method_reports: Dict[str, PerformanceReport]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []

        for method_name, report in method_reports.items():
            if report.performance_level in [PerformanceLevel.SLOW, PerformanceLevel.CRITICAL]:
                recommendations.append(
                    f"Optimize {method_name}: avg {report.avg_execution_time_ms:.1f}ms "
                    f"({report.performance_level.value})"
                )

            if report.success_rate < 0.95:
                recommendations.append(
                    f"Improve reliability for {method_name}: "
                    f"{report.success_rate:.1%} success rate"
                )

            if report.anomalies_detected:
                recommendations.append(
                    f"Investigate {method_name}: {len(report.anomalies_detected)} anomalies detected"
                )

        if not recommendations:
            recommendations.append("System performing well - no immediate optimizations needed")

        return recommendations


class PerformanceLogger:
    """Main performance logging coordinator"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, db_path: Optional[str] = None):
        """Singleton pattern for performance logger"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, db_path: Optional[str] = None):
        """Initialize performance logger (singleton)"""
        if self._initialized:
            return

        self.database = PerformanceDatabase(db_path)
        self.analyzer = PerformanceAnalyzer(self.database)
        self._process = psutil.Process()
        self._initialized = True

        logger.info("🚀 Performance logger initialized")

    def log_performance(self, method_name: str, execution_time_ms: float,
                        success: bool = True, error_message: Optional[str] = None,
                        input_size: Optional[int] = None, output_size: Optional[int] = None,
                        custom_metrics: Optional[Dict[str, Any]] = None):
        """Log performance metric"""
        try:
            # Get system metrics
            memory_info = self._process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB
            cpu_percent = self._process.cpu_percent()

            metric = PerformanceMetric(
                timestamp=datetime.now(),
                method_name=method_name,
                execution_time_ms=execution_time_ms,
                memory_usage_mb=memory_mb,
                cpu_percent=cpu_percent,
                success=success,
                error_message=error_message,
                input_size=input_size,
                output_size=output_size,
                custom_metrics=custom_metrics
            )

            self.database.store_metric(metric)

        except Exception as e:
            logger.error(f"❌ Failed to log performance metric: {e}")

    def get_report(self, method_name: Optional[str] = None, hours_back: int = 24) -> Dict[str, Any]:
        """Get performance report"""
        if method_name:
            report = self.analyzer.analyze_method_performance(method_name, hours_back)
            return asdict(report)
        else:
            return self.analyzer.generate_system_report(hours_back)


# Global performance logger instance
performance_logger = PerformanceLogger()


def performance_monitor(method_name: Optional[str] = None,
                        track_input_size: bool = False,
                        track_output_size: bool = False,
                        custom_metrics_extractor: Optional[Callable] = None):
    """
    Performance monitoring decorator

    Args:
        method_name: Custom method name (defaults to function name)
        track_input_size: Whether to track input data size
        track_output_size: Whether to track output data size
        custom_metrics_extractor: Function to extract custom metrics from result

    Usage:
        @performance_monitor()
        def my_function():
            pass

        @performance_monitor(track_input_size=True, track_output_size=True)
        def process_data(data):
            return processed_data
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Start timing
            start_time = time.time()

            # Determine method name
            func_name = method_name or f"{func.__module__}.{func.__qualname__}"

            # Track input size if requested
            input_size = None
            if track_input_size and args:
                try:
                    if hasattr(args[0], '__len__'):
                        input_size = len(args[0])
                    elif hasattr(args[0], 'shape'):  # numpy arrays, pandas dataframes
                        input_size = args[0].shape[0] if args[0].shape else 0
                except:
                    input_size = None

            success = True
            error_message = None
            result = None

            try:
                # Execute function
                result = func(*args, **kwargs)

            except Exception as e:
                success = False
                error_message = str(e)
                raise  # Re-raise the exception

            finally:
                # Calculate execution time
                execution_time_ms = (time.time() - start_time) * 1000

                # Track output size if requested and function succeeded
                output_size = None
                if track_output_size and success and result is not None:
                    try:
                        if hasattr(result, '__len__'):
                            output_size = len(result)
                        elif hasattr(result, 'shape'):
                            output_size = result.shape[0] if result.shape else 0
                    except:
                        output_size = None

                # Extract custom metrics if provided
                custom_metrics = None
                if custom_metrics_extractor and success and result is not None:
                    try:
                        custom_metrics = custom_metrics_extractor(result)
                    except Exception as e:
                        logger.warning(f"Custom metrics extraction failed: {e}")

                # Log performance
                performance_logger.log_performance(
                    method_name=func_name,
                    execution_time_ms=execution_time_ms,
                    success=success,
                    error_message=error_message,
                    input_size=input_size,
                    output_size=output_size,
                    custom_metrics=custom_metrics
                )

            return result

        return wrapper

    return decorator


def get_performance_report(method_name: Optional[str] = None, hours_back: int = 24) -> Dict[str, Any]:
    """Get performance report for analysis"""
    return performance_logger.get_report(method_name, hours_back)


def print_performance_summary():
    """Print a formatted performance summary"""
    report = get_performance_report()

    if report.get('status') != 'success':
        print("❌ No performance data available")
        return

    summary = report['system_summary']

    print("\n" + "=" * 60)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Health Score: {summary['health_score']}/100")
    print(f"Total Calls: {summary['total_calls']}")
    print(f"Average Time: {summary['avg_execution_time_ms']:.1f}ms")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"Methods Monitored: {summary['unique_methods']}")

    print("\n📈 RECOMMENDATIONS:")
    for rec in report['recommendations']:
        print(f"  • {rec}")

    print("\n🔍 METHOD DETAILS:")
    for method_name, method_report in report['method_reports'].items():
        level = method_report['performance_level']
        avg_time = method_report['avg_execution_time_ms']
        success_rate = method_report['success_rate']

        level_icons = {
            'excellent': '🟢',
            'good': '🟡',
            'acceptable': '🟠',
            'slow': '🔴',
            'critical': '💀'
        }

        icon = level_icons.get(level, '❓')
        print(f"  {icon} {method_name}: {avg_time:.1f}ms ({success_rate:.1%} success)")


if __name__ == "__main__":
    # Test the performance logging system
    import random

    print("🧪 Testing Performance Logger...")


    @performance_monitor(track_input_size=True, track_output_size=True)
    def test_function(data_size: int):
        """Test function for performance monitoring"""
        import time
        import random

        # Simulate some processing time
        time.sleep(random.uniform(0.001, 0.01))

        # Return some data
        return list(range(data_size))


    @performance_monitor()
    def failing_function():
        """Test function that fails"""
        raise ValueError("Test error")


    # Run some test calls
    print("Running test functions...")

    for i in range(10):
        test_function(random.randint(10, 100))

    # Test error handling
    for i in range(2):
        try:
            failing_function()
        except:
            pass

    # Print performance summary
    time.sleep(0.1)  # Allow time for logging
    print_performance_summary()

    print("\n✅ Performance logger test complete!")