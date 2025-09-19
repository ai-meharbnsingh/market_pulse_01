# 03_ML_ENGINE/reliability/monitoring_dashboard.py
"""
Comprehensive Reliability Monitoring Dashboard - Phase 2, Step 4
Real-time monitoring and visualization of ML system reliability

Location: #03_ML_ENGINE/reliability/monitoring_dashboard.py

This module provides:
- Real-time system health monitoring
- Circuit breaker status visualization
- Error pattern analysis and alerts
- Performance metrics dashboard
- Recovery strategy recommendations
- Historical trend analysis
- Alert management and escalation
"""

import time
import json
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path

# Import reliability components
from .ml_circuit_breaker import ml_circuit_registry, MLModelType
from .error_handler import ml_error_handler

try:
    from ..performance.performance_logger import performance_logger
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))
    from performance.performance_logger import performance_logger

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class SystemStatus(Enum):
    """Overall system status"""
    HEALTHY = "healthy"  # All systems operational
    DEGRADED = "degraded"  # Some issues but functioning
    CRITICAL = "critical"  # Major issues affecting functionality
    EMERGENCY = "emergency"  # System failure requiring immediate attention


@dataclass
class Alert:
    """System alert"""
    id: str
    timestamp: datetime
    level: AlertLevel
    component: str
    message: str
    details: Dict[str, Any]
    acknowledged: bool = False
    resolved: bool = False


@dataclass
class SystemMetrics:
    """Comprehensive system metrics"""
    timestamp: datetime
    circuit_breaker_status: Dict[str, Any]
    error_statistics: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    health_score: float
    status: SystemStatus
    active_alerts: List[Alert]


class ReliabilityMonitoringDashboard:
    """Comprehensive reliability monitoring dashboard"""

    def __init__(self, alert_thresholds: Optional[Dict[str, float]] = None):
        """Initialize monitoring dashboard"""
        self.alert_thresholds = alert_thresholds or self._get_default_thresholds()
        self.alerts: List[Alert] = []
        self.max_alerts = 1000
        self.lock = threading.Lock()

        # Initialize database for alert storage
        self._init_alert_database()

        # Start background monitoring
        self.monitoring_active = True
        self.monitoring_interval = 30  # 30 seconds
        self.monitoring_thread = threading.Thread(target=self._background_monitoring)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

        logger.info("🖥️ Reliability Monitoring Dashboard initialized")

    def _get_default_thresholds(self) -> Dict[str, float]:
        """Get default alert thresholds"""
        return {
            'health_score_warning': 80.0,
            'health_score_critical': 60.0,
            'failure_rate_warning': 0.15,
            'failure_rate_critical': 0.30,
            'response_time_warning': 50.0,  # ms
            'response_time_critical': 100.0,  # ms
            'circuit_breaker_opens_warning': 5,
            'circuit_breaker_opens_critical': 10,
            'error_spike_threshold': 3.0,  # multiplier over baseline
            'memory_usage_warning': 80.0,  # percentage
            'memory_usage_critical': 95.0  # percentage
        }

    def _init_alert_database(self):
        """Initialize SQLite database for alert storage"""
        try:
            db_dir = Path("10_DATA_STORAGE/ml_reliability")
            db_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str(db_dir / "monitoring_alerts.db")

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS monitoring_alerts (
                        id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        level TEXT NOT NULL,
                        component TEXT NOT NULL,
                        message TEXT NOT NULL,
                        details TEXT NOT NULL,
                        acknowledged INTEGER DEFAULT 0,
                        resolved INTEGER DEFAULT 0,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_alert_timestamp 
                    ON monitoring_alerts(timestamp)
                """)
        except Exception as e:
            logger.warning(f"Failed to initialize alert database: {e}")
            self.db_path = None

    def get_system_status(self) -> SystemMetrics:
        """Get comprehensive system status"""
        timestamp = datetime.now()

        # Get circuit breaker status
        cb_summary = ml_circuit_registry.get_system_health_summary()
        cb_health_reports = ml_circuit_registry.get_all_health_reports()

        # Get error statistics
        error_stats = ml_error_handler.get_error_statistics(1)  # Last hour

        # Get performance metrics
        perf_health = performance_logger.analyzer.get_system_health_score()

        # Calculate overall health score
        health_score = self._calculate_overall_health_score(
            cb_summary, error_stats, perf_health
        )

        # Determine system status
        status = self._determine_system_status(health_score, cb_summary, error_stats)

        # Get active alerts
        active_alerts = self._get_active_alerts()

        return SystemMetrics(
            timestamp=timestamp,
            circuit_breaker_status=cb_summary,
            error_statistics=error_stats,
            performance_metrics=perf_health,
            health_score=health_score,
            status=status,
            active_alerts=active_alerts
        )

    def _calculate_overall_health_score(self, cb_summary: Dict, error_stats: Dict,
                                        perf_health: Dict) -> float:
        """Calculate overall system health score"""
        scores = []

        # Circuit breaker health (weight: 40%)
        if cb_summary.get('status') != 'no_breakers':
            cb_score = cb_summary.get('health_score', 0)
            scores.append((cb_score, 0.4))

        # Performance health (weight: 40%)
        perf_score = perf_health.get('health_score', 0)
        scores.append((perf_score, 0.4))

        # Error rate health (weight: 20%)
        error_score = self._calculate_error_health_score(error_stats)
        scores.append((error_score, 0.2))

        # Weighted average
        if scores:
            weighted_sum = sum(score * weight for score, weight in scores)
            total_weight = sum(weight for _, weight in scores)
            return weighted_sum / total_weight
        else:
            return 0.0

    def _calculate_error_health_score(self, error_stats: Dict) -> float:
        """Calculate health score based on error statistics"""
        if 'error' in error_stats:
            return 50.0  # Neutral score if no data

        total_errors = sum(error_stats.get('category_distribution', {}).values())

        if total_errors == 0:
            return 100.0

        # Penalize based on error count and severity
        severity_weights = {'critical': 10, 'high': 5, 'medium': 2, 'low': 1, 'info': 0.5}
        severity_dist = error_stats.get('severity_distribution', {})

        weighted_errors = sum(
            count * severity_weights.get(severity.lower(), 1)
            for severity, count in severity_dist.items()
        )

        # Score decreases with more weighted errors
        # 100 at 0 errors, decreasing to 0 at high error counts
        score = max(0, 100 - (weighted_errors * 2))
        return score

    def _determine_system_status(self, health_score: float, cb_summary: Dict,
                                 error_stats: Dict) -> SystemStatus:
        """Determine overall system status"""
        # Check for emergency conditions
        if health_score < 30:
            return SystemStatus.EMERGENCY

        # Check for critical conditions
        critical_conditions = [
            health_score < 50,
            cb_summary.get('breakers_critical', 0) > 0,
            self._has_critical_errors(error_stats)
        ]

        if any(critical_conditions):
            return SystemStatus.CRITICAL

        # Check for degraded conditions
        degraded_conditions = [
            health_score < 80,
            cb_summary.get('breakers_degraded', 0) > 0,
            self._has_high_error_rate(error_stats)
        ]

        if any(degraded_conditions):
            return SystemStatus.DEGRADED

        return SystemStatus.HEALTHY

    def _has_critical_errors(self, error_stats: Dict) -> bool:
        """Check if there are critical errors"""
        if 'error' in error_stats:
            return False

        severity_dist = error_stats.get('severity_distribution', {})
        return severity_dist.get('critical', 0) > 0

    def _has_high_error_rate(self, error_stats: Dict) -> bool:
        """Check if there's a high error rate"""
        if 'error' in error_stats:
            return False

        total_errors = sum(error_stats.get('category_distribution', {}).values())
        return total_errors > 10  # More than 10 errors in the last hour

    def _background_monitoring(self):
        """Background monitoring thread"""
        while self.monitoring_active:
            try:
                self._check_system_health()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in background monitoring: {e}")
                time.sleep(self.monitoring_interval)

    def _check_system_health(self):
        """Check system health and generate alerts"""
        system_metrics = self.get_system_status()

        # Check for health score alerts
        self._check_health_score_alerts(system_metrics.health_score)

        # Check circuit breaker alerts
        self._check_circuit_breaker_alerts(system_metrics.circuit_breaker_status)

        # Check error alerts
        self._check_error_alerts(system_metrics.error_statistics)

        # Check performance alerts
        self._check_performance_alerts(system_metrics.performance_metrics)

    def _check_health_score_alerts(self, health_score: float):
        """Check health score and generate alerts if needed"""
        if health_score < self.alert_thresholds['health_score_critical']:
            self._create_alert(
                AlertLevel.CRITICAL,
                "system_health",
                f"System health score critically low: {health_score:.1f}/100",
                {"health_score": health_score, "threshold": self.alert_thresholds['health_score_critical']}
            )
        elif health_score < self.alert_thresholds['health_score_warning']:
            self._create_alert(
                AlertLevel.WARNING,
                "system_health",
                f"System health score below warning threshold: {health_score:.1f}/100",
                {"health_score": health_score, "threshold": self.alert_thresholds['health_score_warning']}
            )

    def _check_circuit_breaker_alerts(self, cb_status: Dict):
        """Check circuit breaker status and generate alerts"""
        if cb_status.get('status') == 'no_breakers':
            return

        critical_breakers = cb_status.get('breakers_critical', 0)
        if critical_breakers > 0:
            self._create_alert(
                AlertLevel.CRITICAL,
                "circuit_breakers",
                f"{critical_breakers} circuit breakers in critical state",
                {"critical_breakers": critical_breakers, "status": cb_status}
            )

        # Check for too many circuit breaker opens
        health_reports = ml_circuit_registry.get_all_health_reports()
        total_opens = sum(
            report['metrics']['circuit_opens']
            for report in health_reports.values()
        )

        if total_opens > self.alert_thresholds['circuit_breaker_opens_critical']:
            self._create_alert(
                AlertLevel.CRITICAL,
                "circuit_breakers",
                f"Excessive circuit breaker opens: {total_opens}",
                {"total_opens": total_opens, "threshold": self.alert_thresholds['circuit_breaker_opens_critical']}
            )

    def _check_error_alerts(self, error_stats: Dict):
        """Check error statistics and generate alerts"""
        if 'error' in error_stats:
            return

        # Check for critical errors
        severity_dist = error_stats.get('severity_distribution', {})
        critical_errors = severity_dist.get('critical', 0)

        if critical_errors > 0:
            self._create_alert(
                AlertLevel.CRITICAL,
                "errors",
                f"{critical_errors} critical errors detected in the last hour",
                {"critical_errors": critical_errors, "severity_distribution": severity_dist}
            )

        # Check for high error rate
        total_errors = sum(error_stats.get('category_distribution', {}).values())
        if total_errors > 20:  # More than 20 errors per hour
            self._create_alert(
                AlertLevel.WARNING,
                "errors",
                f"High error rate: {total_errors} errors in the last hour",
                {"total_errors": total_errors, "category_distribution": error_stats.get('category_distribution')}
            )

    def _check_performance_alerts(self, perf_metrics: Dict):
        """Check performance metrics and generate alerts"""
        if 'error' in perf_metrics:
            return

        health_score = perf_metrics.get('health_score', 100)

        if health_score < 50:
            self._create_alert(
                AlertLevel.CRITICAL,
                "performance",
                f"Performance health critically low: {health_score:.1f}/100",
                {"performance_health": health_score, "metrics": perf_metrics}
            )

    def _create_alert(self, level: AlertLevel, component: str, message: str,
                      details: Dict[str, Any]):
        """Create and store an alert"""
        alert_id = f"{component}_{level.value}_{int(time.time())}"

        # Check if similar alert already exists (avoid spam)
        if self._similar_alert_exists(component, message):
            return

        alert = Alert(
            id=alert_id,
            timestamp=datetime.now(),
            level=level,
            component=component,
            message=message,
            details=details
        )

        with self.lock:
            self.alerts.append(alert)

            # Keep only recent alerts
            if len(self.alerts) > self.max_alerts:
                self.alerts = self.alerts[-self.max_alerts:]

        # Store in database
        self._store_alert(alert)

        # Log alert
        log_level = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.CRITICAL: logger.error,
            AlertLevel.EMERGENCY: logger.critical
        }.get(level, logger.info)

        log_level(f"🚨 ALERT [{level.value.upper()}] {component}: {message}")

    def _similar_alert_exists(self, component: str, message: str,
                              time_window: int = 300) -> bool:
        """Check if similar alert exists in recent time window"""
        cutoff_time = datetime.now() - timedelta(seconds=time_window)

        with self.lock:
            for alert in self.alerts:
                if (alert.timestamp > cutoff_time and
                        alert.component == component and
                        alert.message == message and
                        not alert.resolved):
                    return True

        return False

    def _store_alert(self, alert: Alert):
        """Store alert in database"""
        if not self.db_path:
            return

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO monitoring_alerts 
                    (id, timestamp, level, component, message, details, acknowledged, resolved)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    alert.id,
                    alert.timestamp.isoformat(),
                    alert.level.value,
                    alert.component,
                    alert.message,
                    json.dumps(alert.details),
                    int(alert.acknowledged),
                    int(alert.resolved)
                ))
        except Exception as e:
            logger.warning(f"Failed to store alert in database: {e}")

    def _get_active_alerts(self, hours: int = 24) -> List[Alert]:
        """Get active alerts from the last specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        with self.lock:
            return [
                alert for alert in self.alerts
                if alert.timestamp > cutoff_time and not alert.resolved
            ]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        with self.lock:
            for alert in self.alerts:
                if alert.id == alert_id:
                    alert.acknowledged = True
                    self._update_alert_status(alert_id, acknowledged=True)
                    logger.info(f"✅ Alert {alert_id} acknowledged")
                    return True

        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        with self.lock:
            for alert in self.alerts:
                if alert.id == alert_id:
                    alert.resolved = True
                    self._update_alert_status(alert_id, resolved=True)
                    logger.info(f"✅ Alert {alert_id} resolved")
                    return True

        return False

    def _update_alert_status(self, alert_id: str, acknowledged: bool = None,
                             resolved: bool = None):
        """Update alert status in database"""
        if not self.db_path:
            return

        try:
            with sqlite3.connect(self.db_path) as conn:
                updates = []
                values = []

                if acknowledged is not None:
                    updates.append("acknowledged = ?")
                    values.append(int(acknowledged))

                if resolved is not None:
                    updates.append("resolved = ?")
                    values.append(int(resolved))

                if updates:
                    values.append(alert_id)
                    query = f"UPDATE monitoring_alerts SET {', '.join(updates)} WHERE id = ?"
                    conn.execute(query, values)
        except Exception as e:
            logger.warning(f"Failed to update alert status: {e}")

    def get_dashboard_summary(self) -> str:
        """Get formatted dashboard summary"""
        system_metrics = self.get_system_status()

        # Status icons
        status_icons = {
            SystemStatus.HEALTHY: '✅',
            SystemStatus.DEGRADED: '⚠️',
            SystemStatus.CRITICAL: '🚨',
            SystemStatus.EMERGENCY: '🔥'
        }

        icon = status_icons.get(system_metrics.status, '❓')

        dashboard = f"""
{'=' * 80}
🖥️ RELIABILITY MONITORING DASHBOARD
{'=' * 80}
{icon} System Status: {system_metrics.status.value.upper()}
🏥 Health Score: {system_metrics.health_score:.1f}/100
📊 Timestamp: {system_metrics.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

🔧 Circuit Breaker Status:
"""

        cb_status = system_metrics.circuit_breaker_status
        if cb_status.get('status') == 'no_breakers':
            dashboard += "   No circuit breakers registered\n"
        else:
            dashboard += f"   Total: {cb_status['total_breakers']}\n"
            dashboard += f"   🟢 Healthy: {cb_status['breakers_healthy']}\n"
            dashboard += f"   🟡 Degraded: {cb_status['breakers_degraded']}\n"
            dashboard += f"   🔴 Critical: {cb_status['breakers_critical']}\n"

        dashboard += f"\n📈 Performance Metrics:\n"
        perf_metrics = system_metrics.performance_metrics
        dashboard += f"   Health Score: {perf_metrics.get('health_score', 0):.1f}/100\n"
        dashboard += f"   Status: {perf_metrics.get('status', 'unknown').upper()}\n"
        dashboard += f"   Methods: {perf_metrics.get('method_count', 0)}\n"
        dashboard += f"   Total Calls: {perf_metrics.get('total_calls', 0)}\n"

        dashboard += f"\n🚨 Error Statistics (Last Hour):\n"
        error_stats = system_metrics.error_statistics
        if 'error' in error_stats:
            dashboard += f"   Error retrieving statistics: {error_stats['error']}\n"
        else:
            category_dist = error_stats.get('category_distribution', {})
            total_errors = sum(category_dist.values())
            dashboard += f"   Total Errors: {total_errors}\n"

            if category_dist:
                for category, count in sorted(category_dist.items(), key=lambda x: x[1], reverse=True):
                    dashboard += f"   {category.upper()}: {count}\n"

        dashboard += f"\n🚨 Active Alerts ({len(system_metrics.active_alerts)}):\n"
        if not system_metrics.active_alerts:
            dashboard += "   No active alerts\n"
        else:
            for alert in system_metrics.active_alerts[-5:]:  # Show last 5
                alert_icon = {
                    AlertLevel.INFO: 'ℹ️',
                    AlertLevel.WARNING: '⚠️',
                    AlertLevel.CRITICAL: '🚨',
                    AlertLevel.EMERGENCY: '🔥'
                }.get(alert.level, '❓')

                status = "✅" if alert.acknowledged else "🔔"
                dashboard += f"   {alert_icon}{status} {alert.component}: {alert.message[:60]}...\n"

        # Recommendations
        recommendations = self._get_system_recommendations(system_metrics)
        if recommendations:
            dashboard += f"\n💡 Recommendations:\n"
            for i, rec in enumerate(recommendations[:3], 1):
                dashboard += f"   {i}. {rec}\n"

        dashboard += f"\n{'=' * 80}"
        return dashboard

    def _get_system_recommendations(self, metrics: SystemMetrics) -> List[str]:
        """Get system recommendations based on current metrics"""
        recommendations = []

        if metrics.health_score < 70:
            recommendations.append("Investigate system health issues - multiple components degraded")

        if metrics.status in [SystemStatus.CRITICAL, SystemStatus.EMERGENCY]:
            recommendations.append("Immediate attention required - system in critical state")

        cb_status = metrics.circuit_breaker_status
        if cb_status.get('breakers_critical', 0) > 0:
            recommendations.append("Fix critical circuit breaker issues to restore service")

        error_stats = metrics.error_statistics
        if not ('error' in error_stats):
            total_errors = sum(error_stats.get('category_distribution', {}).values())
            if total_errors > 10:
                recommendations.append("Investigate and reduce error rate")

        if len(metrics.active_alerts) > 5:
            recommendations.append("Address active alerts to improve system stability")

        perf_health = metrics.performance_metrics.get('health_score', 100)
        if perf_health < 70:
            recommendations.append("Optimize performance - response times or reliability issues detected")

        return recommendations

    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        logger.info("🛑 Reliability monitoring stopped")


# Global monitoring dashboard instance
reliability_dashboard = ReliabilityMonitoringDashboard()


def get_reliability_dashboard() -> str:
    """Get formatted reliability dashboard"""
    return reliability_dashboard.get_dashboard_summary()


def create_alert(level: str, component: str, message: str, details: Dict = None) -> str:
    """Create a custom alert"""
    alert_level = AlertLevel(level.lower())
    reliability_dashboard._create_alert(alert_level, component, message, details or {})
    return f"Alert created: {component} - {message}"


# Example usage and testing
if __name__ == "__main__":
    print("🖥️ Testing Reliability Monitoring Dashboard")
    print("=" * 60)

    # Display current dashboard
    print(get_reliability_dashboard())

    # Create test alerts
    create_alert("warning", "test_component", "Test warning alert")
    create_alert("critical", "test_model", "Test critical alert")

    # Wait a moment for background processing
    time.sleep(2)

    # Display updated dashboard
    print("\n" + "=" * 60)
    print("UPDATED DASHBOARD:")
    print("=" * 60)
    print(get_reliability_dashboard())

    # Stop monitoring
    reliability_dashboard.stop_monitoring()