# 03_ML_ENGINE/monitoring/persistent_monitoring_dashboard.py
"""
Persistent Monitoring Dashboard with Database Storage
MarketPulse Phase 2, Step 5 - Complete Monitoring Integration

This module provides comprehensive monitoring with:
- Persistent storage for circuit breaker states and performance metrics
- Real-time visual dashboards for system health monitoring
- Historical trend analysis and alerting
- Circuit breaker state persistence across restarts
- Performance degradation detection and recovery tracking
- Unified system health scoring and recommendations

Location: #03_ML_ENGINE/monitoring/persistent_monitoring_dashboard.py
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import sqlite3
import json
import warnings
import time
import threading
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import psutil

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup paths
current_dir = Path(__file__).parent
root_dir = current_dir.parent.parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir.parent / "reliability"))
sys.path.append(str(current_dir.parent / "optimization"))


class HealthLevel(Enum):
    """System health levels"""
    EXCELLENT = "excellent"  # 90-100 score
    GOOD = "good"  # 70-89 score
    DEGRADED = "degraded"  # 50-69 score
    POOR = "poor"  # 30-49 score
    CRITICAL = "critical"  # 0-29 score


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class CircuitBreakerState:
    """Circuit breaker persistent state"""
    name: str
    state: str  # CLOSED, OPEN, HALF_OPEN
    failure_count: int
    last_failure_time: Optional[str]
    last_success_time: Optional[str]
    total_requests: int
    successful_requests: int
    average_latency: float
    failure_rate: float
    recovery_attempts: int
    created_at: str
    updated_at: str


@dataclass
class SystemHealthSnapshot:
    """System health snapshot"""
    timestamp: str
    overall_health_score: float
    health_level: str
    component_health: Dict[str, float]
    performance_metrics: Dict[str, float]
    circuit_breaker_states: Dict[str, str]
    active_alerts: List[Dict[str, Any]]
    recommendations: List[str]
    system_uptime: float
    memory_usage_mb: float
    cpu_usage_pct: float


class PersistentMonitoringDashboard:
    """
    Persistent Monitoring Dashboard with Database Storage

    Features:
    - Circuit breaker state persistence across system restarts
    - Real-time performance monitoring with historical trends
    - Visual dashboard generation with health scoring
    - Automated alerting for performance degradation
    - Recovery tracking and recommendation system
    - Comprehensive system health analysis
    """

    def __init__(self, db_path: str = "marketpulse_production.db",
                 monitoring_interval: int = 10,
                 alert_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize Persistent Monitoring Dashboard

        Args:
            db_path: Path to SQLite database
            monitoring_interval: Monitoring update interval in seconds
            alert_thresholds: Custom alert thresholds
        """
        self.db_path = db_path
        self.monitoring_interval = monitoring_interval
        self.start_time = datetime.now()

        # Alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'health_score_critical': 30.0,
            'health_score_poor': 50.0,
            'health_score_degraded': 70.0,
            'latency_warning_ms': 50.0,
            'latency_critical_ms': 100.0,
            'success_rate_warning': 90.0,
            'success_rate_critical': 80.0,
            'memory_warning_mb': 1024,
            'memory_critical_mb': 2048,
            'cpu_warning_pct': 80.0,
            'cpu_critical_pct': 95.0
        }

        # In-memory state
        self.circuit_breaker_states: Dict[str, CircuitBreakerState] = {}
        self.performance_history = deque(maxlen=1000)
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        self.health_history = deque(maxlen=100)

        # Thread safety
        self.lock = threading.RLock()

        # Background monitoring
        self.monitoring_thread = None
        self.running = True

        # Initialize components
        self._initialize_database()
        self._load_persistent_state()
        self._start_background_monitoring()

        logger.info("PersistentMonitoringDashboard initialized successfully")

    def _initialize_database(self):
        """Initialize database tables for persistent monitoring"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Circuit breaker states table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS circuit_breaker_states (
                        name TEXT PRIMARY KEY,
                        state TEXT NOT NULL,
                        failure_count INTEGER NOT NULL,
                        last_failure_time TEXT,
                        last_success_time TEXT,
                        total_requests INTEGER NOT NULL,
                        successful_requests INTEGER NOT NULL,
                        average_latency REAL NOT NULL,
                        failure_rate REAL NOT NULL,
                        recovery_attempts INTEGER NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                ''')

                # System health history table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_health_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        overall_health_score REAL NOT NULL,
                        health_level TEXT NOT NULL,
                        component_health TEXT NOT NULL,
                        performance_metrics TEXT NOT NULL,
                        circuit_breaker_states TEXT NOT NULL,
                        active_alerts TEXT NOT NULL,
                        recommendations TEXT NOT NULL,
                        system_uptime REAL NOT NULL,
                        memory_usage_mb REAL NOT NULL,
                        cpu_usage_pct REAL NOT NULL
                    )
                ''')

                # Performance metrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_metrics_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        component TEXT NOT NULL,
                        additional_data TEXT
                    )
                ''')

                # Alerts history table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS alerts_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        alert_id TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        component TEXT NOT NULL,
                        message TEXT NOT NULL,
                        threshold_value REAL,
                        actual_value REAL,
                        resolution_time TEXT,
                        status TEXT NOT NULL
                    )
                ''')

                conn.commit()
                logger.info("Monitoring database tables initialized successfully")

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    def _load_persistent_state(self):
        """Load persistent circuit breaker states from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM circuit_breaker_states")
                rows = cursor.fetchall()

                for row in rows:
                    cb_state = CircuitBreakerState(
                        name=row[0],
                        state=row[1],
                        failure_count=row[2],
                        last_failure_time=row[3],
                        last_success_time=row[4],
                        total_requests=row[5],
                        successful_requests=row[6],
                        average_latency=row[7],
                        failure_rate=row[8],
                        recovery_attempts=row[9],
                        created_at=row[10],
                        updated_at=row[11]
                    )
                    self.circuit_breaker_states[cb_state.name] = cb_state

                logger.info(f"Loaded {len(self.circuit_breaker_states)} circuit breaker states")

        except Exception as e:
            logger.error(f"Failed to load persistent state: {e}")

    def _start_background_monitoring(self):
        """Start background monitoring thread"""

        def monitoring_loop():
            while self.running:
                try:
                    self._collect_system_metrics()
                    self._update_health_status()
                    self._check_alerts()
                    self._persist_current_state()
                    time.sleep(self.monitoring_interval)
                except Exception as e:
                    logger.error(f"Background monitoring error: {e}")
                    time.sleep(self.monitoring_interval * 2)

        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Background monitoring started")

    def update_circuit_breaker_state(self, name: str, state: str,
                                     failure_count: int = 0, latency: float = 0.0,
                                     success: bool = True):
        """Update circuit breaker state with persistence"""
        current_time = datetime.now().isoformat()

        with self.lock:
            if name not in self.circuit_breaker_states:
                # Create new circuit breaker state
                self.circuit_breaker_states[name] = CircuitBreakerState(
                    name=name,
                    state=state,
                    failure_count=failure_count,
                    last_failure_time=None,
                    last_success_time=None,
                    total_requests=0,
                    successful_requests=0,
                    average_latency=0.0,
                    failure_rate=0.0,
                    recovery_attempts=0,
                    created_at=current_time,
                    updated_at=current_time
                )

            cb_state = self.circuit_breaker_states[name]

            # Update state
            cb_state.state = state
            cb_state.failure_count = failure_count
            cb_state.total_requests += 1
            cb_state.updated_at = current_time

            if success:
                cb_state.successful_requests += 1
                cb_state.last_success_time = current_time
            else:
                cb_state.last_failure_time = current_time
                if state == "HALF_OPEN":
                    cb_state.recovery_attempts += 1

            # Update metrics
            if cb_state.total_requests > 0:
                cb_state.failure_rate = (
                                                (cb_state.total_requests - cb_state.successful_requests) /
                                                cb_state.total_requests
                                        ) * 100

            # Update average latency (exponential moving average)
            if latency > 0:
                if cb_state.average_latency == 0:
                    cb_state.average_latency = latency
                else:
                    alpha = 0.1
                    cb_state.average_latency = (
                            alpha * latency + (1 - alpha) * cb_state.average_latency
                    )

            # Persist to database
            self._persist_circuit_breaker_state(cb_state)

    def _persist_circuit_breaker_state(self, cb_state: CircuitBreakerState):
        """Persist circuit breaker state to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO circuit_breaker_states 
                    (name, state, failure_count, last_failure_time, last_success_time,
                     total_requests, successful_requests, average_latency, failure_rate,
                     recovery_attempts, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    cb_state.name, cb_state.state, cb_state.failure_count,
                    cb_state.last_failure_time, cb_state.last_success_time,
                    cb_state.total_requests, cb_state.successful_requests,
                    cb_state.average_latency, cb_state.failure_rate,
                    cb_state.recovery_attempts, cb_state.created_at, cb_state.updated_at
                ))
                conn.commit()

        except Exception as e:
            logger.error(f"Failed to persist circuit breaker state: {e}")

    def _collect_system_metrics(self):
        """Collect current system metrics"""
        try:
            # System resource metrics
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            cpu_usage = psutil.Process().cpu_percent()
            uptime = (datetime.now() - self.start_time).total_seconds()

            # Performance metrics
            performance_metrics = {
                'memory_usage_mb': memory_usage,
                'cpu_usage_pct': cpu_usage,
                'system_uptime': uptime,
                'active_circuit_breakers': len(self.circuit_breaker_states),
                'failed_circuit_breakers': sum(1 for cb in self.circuit_breaker_states.values()
                                               if cb.state == 'OPEN'),
                'average_cb_latency': np.mean([cb.average_latency for cb in self.circuit_breaker_states.values()])
                if self.circuit_breaker_states else 0.0,
                'total_cb_requests': sum(cb.total_requests for cb in self.circuit_breaker_states.values()),
                'overall_success_rate': self._calculate_overall_success_rate()
            }

            self.performance_history.append({
                'timestamp': datetime.now().isoformat(),
                'metrics': performance_metrics
            })

        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")

    def _calculate_overall_success_rate(self) -> float:
        """Calculate overall system success rate"""
        if not self.circuit_breaker_states:
            return 100.0

        total_requests = sum(cb.total_requests for cb in self.circuit_breaker_states.values())
        if total_requests == 0:
            return 100.0

        total_successful = sum(cb.successful_requests for cb in self.circuit_breaker_states.values())
        return (total_successful / total_requests) * 100

    def _update_health_status(self):
        """Update overall system health status"""
        try:
            health_score = self._calculate_health_score()
            health_level = self._determine_health_level(health_score)

            # Component health breakdown
            component_health = {
                'circuit_breakers': self._calculate_circuit_breaker_health(),
                'performance': self._calculate_performance_health(),
                'resources': self._calculate_resource_health()
            }

            # Get current performance metrics
            current_metrics = self.performance_history[-1]['metrics'] if self.performance_history else {}

            # Create health snapshot
            snapshot = SystemHealthSnapshot(
                timestamp=datetime.now().isoformat(),
                overall_health_score=health_score,
                health_level=health_level.value,
                component_health=component_health,
                performance_metrics=current_metrics,
                circuit_breaker_states={name: cb.state for name, cb in self.circuit_breaker_states.items()},
                active_alerts=list(self.active_alerts.values()),
                recommendations=self._generate_recommendations(health_score, component_health),
                system_uptime=current_metrics.get('system_uptime', 0.0),
                memory_usage_mb=current_metrics.get('memory_usage_mb', 0.0),
                cpu_usage_pct=current_metrics.get('cpu_usage_pct', 0.0)
            )

            self.health_history.append(snapshot)

            # Persist to database
            self._persist_health_snapshot(snapshot)

        except Exception as e:
            logger.error(f"Failed to update health status: {e}")

    def _calculate_health_score(self) -> float:
        """Calculate overall system health score (0-100)"""
        try:
            scores = []
            weights = []

            # Circuit breaker health (40% weight)
            cb_health = self._calculate_circuit_breaker_health()
            scores.append(cb_health)
            weights.append(0.4)

            # Performance health (35% weight)
            perf_health = self._calculate_performance_health()
            scores.append(perf_health)
            weights.append(0.35)

            # Resource health (25% weight)
            resource_health = self._calculate_resource_health()
            scores.append(resource_health)
            weights.append(0.25)

            # Weighted average
            weighted_score = sum(score * weight for score, weight in zip(scores, weights))
            return min(100.0, max(0.0, weighted_score))

        except Exception as e:
            logger.error(f"Health score calculation failed: {e}")
            return 50.0

    def _calculate_circuit_breaker_health(self) -> float:
        """Calculate circuit breaker component health"""
        if not self.circuit_breaker_states:
            return 100.0

        total_score = 0.0
        for cb in self.circuit_breaker_states.values():
            # Base score from success rate
            success_rate = (cb.successful_requests / max(cb.total_requests, 1)) * 100
            score = success_rate

            # Penalties for failures
            if cb.state == 'OPEN':
                score *= 0.3  # Heavy penalty for open circuit breakers
            elif cb.state == 'HALF_OPEN':
                score *= 0.7  # Moderate penalty for half-open

            # Latency penalty
            if cb.average_latency > 50:
                score *= 0.8
            elif cb.average_latency > 100:
                score *= 0.6

            total_score += score

        return total_score / len(self.circuit_breaker_states)

    def _calculate_performance_health(self) -> float:
        """Calculate performance component health"""
        if not self.performance_history:
            return 100.0

        recent_metrics = self.performance_history[-1]['metrics']
        score = 100.0

        # Success rate impact (50% of performance health)
        success_rate = recent_metrics.get('overall_success_rate', 100.0)
        score = score * 0.5 + success_rate * 0.5

        # Latency impact (30% of performance health)
        avg_latency = recent_metrics.get('average_cb_latency', 0.0)
        if avg_latency > 100:
            score *= 0.6
        elif avg_latency > 50:
            score *= 0.8
        elif avg_latency > 20:
            score *= 0.9

        # Failed circuit breakers impact (20% of performance health)
        failed_cbs = recent_metrics.get('failed_circuit_breakers', 0)
        total_cbs = recent_metrics.get('active_circuit_breakers', 1)
        if failed_cbs > 0:
            failure_ratio = failed_cbs / total_cbs
            score *= (1.0 - failure_ratio * 0.5)

        return min(100.0, max(0.0, score))

    def _calculate_resource_health(self) -> float:
        """Calculate resource component health"""
        if not self.performance_history:
            return 100.0

        recent_metrics = self.performance_history[-1]['metrics']
        score = 100.0

        # Memory usage impact
        memory_mb = recent_metrics.get('memory_usage_mb', 0.0)
        if memory_mb > 2048:
            score *= 0.5
        elif memory_mb > 1024:
            score *= 0.7
        elif memory_mb > 512:
            score *= 0.9

        # CPU usage impact
        cpu_pct = recent_metrics.get('cpu_usage_pct', 0.0)
        if cpu_pct > 95:
            score *= 0.4
        elif cpu_pct > 80:
            score *= 0.7
        elif cpu_pct > 60:
            score *= 0.9

        return min(100.0, max(0.0, score))

    def _determine_health_level(self, health_score: float) -> HealthLevel:
        """Determine health level from score"""
        if health_score >= 90:
            return HealthLevel.EXCELLENT
        elif health_score >= 70:
            return HealthLevel.GOOD
        elif health_score >= 50:
            return HealthLevel.DEGRADED
        elif health_score >= 30:
            return HealthLevel.POOR
        else:
            return HealthLevel.CRITICAL

    def _generate_recommendations(self, health_score: float,
                                  component_health: Dict[str, float]) -> List[str]:
        """Generate system recommendations based on health analysis"""
        recommendations = []

        try:
            # Overall health recommendations
            if health_score < 50:
                recommendations.append("URGENT: System health is critical - immediate attention required")
            elif health_score < 70:
                recommendations.append("System health is degraded - investigate performance issues")

            # Circuit breaker recommendations
            cb_health = component_health.get('circuit_breakers', 100)
            if cb_health < 70:
                failed_cbs = [name for name, cb in self.circuit_breaker_states.items() if cb.state == 'OPEN']
                if failed_cbs:
                    recommendations.append(
                        f"Circuit breakers OPEN: {', '.join(failed_cbs)} - check underlying services")

            # Performance recommendations
            perf_health = component_health.get('performance', 100)
            if perf_health < 70:
                if self.performance_history:
                    recent = self.performance_history[-1]['metrics']
                    if recent.get('average_cb_latency', 0) > 50:
                        recommendations.append(
                            "High latency detected - optimize model performance or increase resources")
                    if recent.get('overall_success_rate', 100) < 90:
                        recommendations.append("Low success rate - investigate model failures and error handling")

            # Resource recommendations
            resource_health = component_health.get('resources', 100)
            if resource_health < 70:
                if self.performance_history:
                    recent = self.performance_history[-1]['metrics']
                    if recent.get('memory_usage_mb', 0) > 1024:
                        recommendations.append("High memory usage - consider memory optimization or garbage collection")
                    if recent.get('cpu_usage_pct', 0) > 80:
                        recommendations.append("High CPU usage - consider scaling resources or optimizing algorithms")

            # Specific circuit breaker recommendations
            for name, cb in self.circuit_breaker_states.items():
                if cb.failure_rate > 20:
                    recommendations.append(
                        f"High failure rate ({cb.failure_rate:.1f}%) in {name} - investigate model reliability")
                if cb.recovery_attempts > 5:
                    recommendations.append(f"Multiple recovery attempts for {name} - may need manual intervention")

        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            recommendations.append("Error generating recommendations - check system logs")

        return recommendations

    def _check_alerts(self):
        """Check for alert conditions and manage active alerts"""
        try:
            current_time = datetime.now()

            if not self.performance_history:
                return

            recent_metrics = self.performance_history[-1]['metrics']

            # Check health score alerts
            health_score = self._calculate_health_score()
            self._check_threshold_alert(
                'health_score_critical', health_score,
                self.alert_thresholds['health_score_critical'],
                AlertSeverity.CRITICAL, 'system',
                f"System health critical: {health_score:.1f}%", False
            )

            self._check_threshold_alert(
                'health_score_poor', health_score,
                self.alert_thresholds['health_score_poor'],
                AlertSeverity.ERROR, 'system',
                f"System health poor: {health_score:.1f}%", False
            )

            # Check latency alerts
            avg_latency = recent_metrics.get('average_cb_latency', 0.0)
            self._check_threshold_alert(
                'latency_critical', avg_latency,
                self.alert_thresholds['latency_critical_ms'],
                AlertSeverity.CRITICAL, 'performance',
                f"Critical latency: {avg_latency:.1f}ms", True
            )

            # Check success rate alerts
            success_rate = recent_metrics.get('overall_success_rate', 100.0)
            self._check_threshold_alert(
                'success_rate_critical', success_rate,
                self.alert_thresholds['success_rate_critical'],
                AlertSeverity.CRITICAL, 'performance',
                f"Critical success rate: {success_rate:.1f}%", False
            )

            # Check resource alerts
            memory_mb = recent_metrics.get('memory_usage_mb', 0.0)
            self._check_threshold_alert(
                'memory_critical', memory_mb,
                self.alert_thresholds['memory_critical_mb'],
                AlertSeverity.ERROR, 'resources',
                f"High memory usage: {memory_mb:.1f}MB", True
            )

            # Check circuit breaker alerts
            for name, cb in self.circuit_breaker_states.items():
                if cb.state == 'OPEN':
                    alert_id = f"circuit_breaker_open_{name}"
                    if alert_id not in self.active_alerts:
                        self._create_alert(
                            alert_id, AlertSeverity.ERROR, 'circuit_breaker',
                            f"Circuit breaker OPEN: {name}", 0, 0
                        )

            # Clean up resolved alerts
            self._clean_resolved_alerts()

        except Exception as e:
            logger.error(f"Alert checking failed: {e}")

    def _check_threshold_alert(self, alert_type: str, current_value: float,
                               threshold: float, severity: AlertSeverity,
                               component: str, message: str, greater_than: bool = True):
        """Check threshold-based alert condition"""
        alert_triggered = (
                (greater_than and current_value > threshold) or
                (not greater_than and current_value < threshold)
        )

        alert_id = f"{alert_type}_{component}"

        if alert_triggered:
            if alert_id not in self.active_alerts:
                self._create_alert(alert_id, severity, component, message, threshold, current_value)
        else:
            if alert_id in self.active_alerts:
                self._resolve_alert(alert_id)

    def _create_alert(self, alert_id: str, severity: AlertSeverity,
                      component: str, message: str, threshold: float, actual: float):
        """Create new alert"""
        alert = {
            'id': alert_id,
            'severity': severity.value,
            'component': component,
            'message': message,
            'threshold_value': threshold,
            'actual_value': actual,
            'created_at': datetime.now().isoformat(),
            'status': 'active'
        }

        self.active_alerts[alert_id] = alert

        # Persist to database
        self._persist_alert(alert)

        logger.warning(f"ALERT CREATED: {severity.value.upper()} - {message}")

    def _resolve_alert(self, alert_id: str):
        """Resolve active alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert['status'] = 'resolved'
            alert['resolution_time'] = datetime.now().isoformat()

            # Persist resolution to database
            self._persist_alert(alert)

            logger.info(f"ALERT RESOLVED: {alert['message']}")
            del self.active_alerts[alert_id]

    def _clean_resolved_alerts(self):
        """Clean up resolved alerts that no longer apply"""
        resolved_alerts = []

        for alert_id, alert in self.active_alerts.items():
            # Check if circuit breaker alerts are still valid
            if 'circuit_breaker_open_' in alert_id:
                cb_name = alert_id.replace('circuit_breaker_open_', '')
                if cb_name in self.circuit_breaker_states:
                    if self.circuit_breaker_states[cb_name].state != 'OPEN':
                        resolved_alerts.append(alert_id)

        for alert_id in resolved_alerts:
            self._resolve_alert(alert_id)

    def _persist_alert(self, alert: Dict[str, Any]):
        """Persist alert to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO alerts_history
                    (timestamp, alert_id, severity, component, message, 
                     threshold_value, actual_value, resolution_time, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    alert.get('created_at', datetime.now().isoformat()),
                    alert['id'], alert['severity'], alert['component'],
                    alert['message'], alert.get('threshold_value'),
                    alert.get('actual_value'), alert.get('resolution_time'),
                    alert['status']
                ))
                conn.commit()

        except Exception as e:
            logger.error(f"Failed to persist alert: {e}")

    def _persist_health_snapshot(self, snapshot: SystemHealthSnapshot):
        """Persist health snapshot to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO system_health_history
                    (timestamp, overall_health_score, health_level, component_health,
                     performance_metrics, circuit_breaker_states, active_alerts,
                     recommendations, system_uptime, memory_usage_mb, cpu_usage_pct)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    snapshot.timestamp, snapshot.overall_health_score, snapshot.health_level,
                    json.dumps(snapshot.component_health), json.dumps(snapshot.performance_metrics),
                    json.dumps(snapshot.circuit_breaker_states), json.dumps(snapshot.active_alerts),
                    json.dumps(snapshot.recommendations), snapshot.system_uptime,
                    snapshot.memory_usage_mb, snapshot.cpu_usage_pct
                ))
                conn.commit()

        except Exception as e:
            logger.error(f"Failed to persist health snapshot: {e}")

    def _persist_current_state(self):
        """Persist all current states to database"""
        # Circuit breaker states are persisted individually
        # Health snapshot is persisted in _update_health_status
        pass

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        try:
            current_health = self.health_history[-1] if self.health_history else None
            recent_performance = list(self.performance_history)[-10:] if self.performance_history else []

            return {
                'current_health': asdict(current_health) if current_health else None,
                'circuit_breakers': {name: asdict(cb) for name, cb in self.circuit_breaker_states.items()},
                'active_alerts': list(self.active_alerts.values()),
                'performance_history': recent_performance,
                'health_trend': [h.overall_health_score for h in list(self.health_history)[-20:]],
                'system_stats': {
                    'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
                    'monitoring_interval': self.monitoring_interval,
                    'total_cb_states': len(self.circuit_breaker_states),
                    'active_alerts_count': len(self.active_alerts)
                },
                'last_updated': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get dashboard data: {e}")
            return {}

    def get_health_report(self) -> Dict[str, Any]:
        """Get current health report"""
        if not self.health_history:
            return {'status': 'No data available'}

        current_health = self.health_history[-1]
        return {
            'overall_health_score': current_health.overall_health_score,
            'health_level': current_health.health_level,
            'component_health': current_health.component_health,
            'active_alerts_count': len(current_health.active_alerts),
            'recommendations': current_health.recommendations,
            'circuit_breaker_summary': {
                'total': len(self.circuit_breaker_states),
                'healthy': sum(1 for cb in self.circuit_breaker_states.values() if cb.state == 'CLOSED'),
                'degraded': sum(1 for cb in self.circuit_breaker_states.values() if cb.state == 'HALF_OPEN'),
                'failed': sum(1 for cb in self.circuit_breaker_states.values() if cb.state == 'OPEN')
            },
            'timestamp': current_health.timestamp
        }

    def shutdown(self):
        """Shutdown monitoring dashboard"""
        self.running = False

        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=10)

        # Final state persistence
        for cb in self.circuit_breaker_states.values():
            self._persist_circuit_breaker_state(cb)

        logger.info("PersistentMonitoringDashboard shutdown completed")

    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.shutdown()
        except:
            pass


# Usage example and testing
if __name__ == "__main__":
    print("Persistent Monitoring Dashboard - Integration Test")
    print("=" * 60)

    try:
        # Create monitoring dashboard
        dashboard = PersistentMonitoringDashboard(monitoring_interval=5)

        # Simulate circuit breaker updates
        print("\n🔄 Simulating circuit breaker updates...")
        dashboard.update_circuit_breaker_state('alpha_model', 'CLOSED', 0, 15.5, True)
        dashboard.update_circuit_breaker_state('lstm_model', 'CLOSED', 0, 22.1, True)
        dashboard.update_circuit_breaker_state('ensemble_model', 'HALF_OPEN', 2, 45.2, False)

        time.sleep(6)  # Let background monitoring run

        # Get health report
        health_report = dashboard.get_health_report()
        print(f"\n📊 Health Report:")
        print(f"   Overall Health: {health_report['overall_health_score']:.1f} ({health_report['health_level']})")
        print(f"   Circuit Breakers: {health_report['circuit_breaker_summary']}")
        print(f"   Active Alerts: {health_report['active_alerts_count']}")

        if health_report.get('recommendations'):
            print(f"   Recommendations: {len(health_report['recommendations'])}")
            for rec in health_report['recommendations'][:3]:
                print(f"     • {rec}")

        # Get dashboard data
        dashboard_data = dashboard.get_dashboard_data()
        print(f"\n💻 Dashboard Data Available:")
        print(f"   Current Health: {'Available' if dashboard_data.get('current_health') else 'None'}")
        print(f"   Circuit Breakers: {len(dashboard_data.get('circuit_breakers', {}))}")
        print(f"   Performance History: {len(dashboard_data.get('performance_history', []))}")
        print(f"   Health Trend Points: {len(dashboard_data.get('health_trend', []))}")

        print("\n✅ PersistentMonitoringDashboard test completed!")

        # Cleanup
        dashboard.shutdown()

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback

        traceback.print_exc()