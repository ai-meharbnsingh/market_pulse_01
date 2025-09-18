# antifragile_framework/core/resource_guard.py

import logging
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional
import sys
from pathlib import Path

# FIXED: Add proper path setup for telemetry imports
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent.parent.parent.parent  # Go up to project root
TELEMETRY_PATH = PROJECT_ROOT / "01_Framework_Core" / "telemetry"

sys.path.insert(0, str(TELEMETRY_PATH))

# Now import telemetry modules with the correct path
try:
    from telemetry import event_topics
    from telemetry.core_logger import UniversalEventSchema, core_logger
    from telemetry.event_bus import EventBus
except ImportError as e:
    # Fallback for testing environments
    import warnings

    warnings.warn(f"Could not import telemetry modules: {e}")


    class MockEventTopics:
        RESOURCE_PENALIZED = "resource.penalized"
        RESOURCE_HEALED = "resource.healed"


    event_topics = MockEventTopics()


    class MockLogger:
        def log_event(self, *args, **kwargs):
            pass


    core_logger = MockLogger()


    class MockEventBus:
        def publish(self, *args, **kwargs):
            pass


    EventBus = MockEventBus


    class MockEventSchema:
        def __init__(self, *args, **kwargs):
            pass


    UniversalEventSchema = MockEventSchema

from .exceptions import NoResourcesAvailableError

log = logging.getLogger(__name__)

class ResourceState(Enum):
    AVAILABLE = auto()
    IN_USE = auto()
    COOLING_DOWN = auto()
    DISABLED = auto()


class MonitoredResource:
    """Represents a single, monitored resource (e.g., an API key)."""

    def __init__(
        self,
        value: str,
        provider_name: str,
        cooldown_seconds: int = 300,
        penalty: float = 0.5,
        healing_interval_seconds: int = 3600,
        healing_increment: float = 0.1,
        event_bus: Optional[EventBus] = None,
    ):
        if not (0 < penalty <= 1):
            raise ValueError("Penalty must be between 0 and 1.")

        self.value = value
        self.provider_name = provider_name
        self.health_score: float = 1.0
        self.state: ResourceState = ResourceState.AVAILABLE
        self.last_failure_timestamp: float = 0.0
        self.last_health_update_timestamp: float = time.monotonic()
        self.last_reserved_timestamp: float = 0.0
        self.lock = threading.Lock()

        self._cooldown_seconds = cooldown_seconds
        self._penalty = penalty
        self._healing_interval_seconds = healing_interval_seconds
        self._healing_increment = healing_increment

        self.event_bus = event_bus
        self.logger = core_logger
        self.safe_value = (
            f"{self.value[:4]}...{self.value[-4:]}" if len(self.value) > 8 else "..."
        )

    def __repr__(self) -> str:
        return (
            f"<MonitoredResource(value='{self.safe_value}', score={self.health_score:.2f}, "
            f"state={self.state.name})>"
        )

    def _log_and_publish_event(
            self, event_name: str, severity: str, payload_data: Dict[str, Any]
    ):
        """Logs an event and publishes it to the event bus."""
        event_schema = UniversalEventSchema(
            event_type=event_name,
            event_topic="system.resources",  # ADD THIS LINE
            event_source=f"ResourceGuard.{self.provider_name}",
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            severity=severity,
            payload=payload_data,
        )
        self.logger.log_event(
            event_type=event_name,
            event_topic="system.resources",
            payload=payload_data,
            severity=severity
        )
        if self.event_bus:
            self.event_bus.publish(event_name, event_schema.model_dump())

    def _update_health(self):
        now = time.monotonic()
        if self.state == ResourceState.COOLING_DOWN:
            if now - self.last_failure_timestamp > self._cooldown_seconds:
                old_state = self.state.name
                self.state = ResourceState.AVAILABLE
                self.last_health_update_timestamp = now
                self._log_and_publish_event(
                    event_topics.RESOURCE_HEALTH_RESTORED,
                    "INFO",
                    {
                        "resource_id": self.safe_value,
                        "old_state": old_state,
                        "new_state": self.state.name,
                        "reason": "Cooldown expired",
                        "provider": self.provider_name,
                    },
                )

        if self.state == ResourceState.AVAILABLE and self.health_score < 1.0:
            time_since_last_update = now - self.last_health_update_timestamp
            if time_since_last_update > self._healing_interval_seconds:
                old_score = self.health_score
                intervals_passed = (
                    time_since_last_update // self._healing_interval_seconds
                )
                healing_amount = intervals_passed * self._healing_increment
                self.health_score = min(1.0, self.health_score + healing_amount)
                self.last_health_update_timestamp = now
                if self.health_score > old_score:
                    self._log_and_publish_event(
                        "resource.health.update",
                        "DEBUG",
                        {
                            "resource_id": self.safe_value,
                            "old_score": old_score,
                            "new_score": self.health_score,
                            "reason": "Periodic healing",
                            "provider": self.provider_name,
                        },
                    )

    def is_available(self) -> bool:
        with self.lock:
            self._update_health()
            return self.state == ResourceState.AVAILABLE

    def penalize(self):
        with self.lock:
            old_score = self.health_score
            self.health_score *= 1 - self._penalty
            if self.health_score < 0.01:
                self.health_score = 0.01
            old_state = self.state.name
            self.state = ResourceState.COOLING_DOWN
            now = time.monotonic()
            self.last_failure_timestamp = now
            self.last_health_update_timestamp = now
            self._log_and_publish_event(
                event_topics.RESOURCE_PENALIZED,
                "WARNING",
                {
                    "resource_id": self.safe_value,
                    "old_state": old_state,
                    "new_state": self.state.name,
                    "reason": "Penalty applied after failure",
                    "old_score": old_score,
                    "new_score": self.health_score,
                    "provider": self.provider_name,
                },
            )

    def release(self):
        with self.lock:
            if self.state == ResourceState.IN_USE:
                self.state = ResourceState.AVAILABLE


class ResourceGuard:
    def __init__(
        self,
        provider_name: str,
        api_keys: List[str],
        resource_config: Optional[Dict[str, Any]] = None,
        event_bus: Optional[EventBus] = None,
    ):
        self.provider_name = provider_name
        self.event_bus = event_bus
        config = resource_config or {}

        monitored_resource_params = {
            "cooldown_seconds": config.get("cooldown", 300),
            "penalty": config.get("penalty", 0.5),
            "healing_interval_seconds": config.get("healing_interval", 3600),
            "healing_increment": config.get("healing_increment", 0.1),
            "event_bus": self.event_bus,
        }

        self._resources: List[MonitoredResource] = [
            MonitoredResource(
                key, provider_name=provider_name, **monitored_resource_params
            )
            for key in api_keys
        ]
        if not self._resources:
            log.warning(
                f"ResourceGuard for '{provider_name}' initialized with no API keys."
            )
        self.lock = threading.Lock()

    def get_total_resource_count(self) -> int:
        return len(self._resources)

    # NEW METHOD: Check if any resource is healthy and available
    def has_healthy_resources(self) -> bool:
        """
        Checks if there is at least one healthy and available resource in the guard.
        This performs an internal update of resource states to reflect current cooldowns/healing.
        """
        with self.lock:
            return any(res.is_available() for res in self._resources)

    def _reserve_resource(self) -> Optional[MonitoredResource]:
        with self.lock:
            # Ensure all resources' states are updated before sorting and selecting
            for res in self._resources:
                res._update_health()  # Call internal update to reflect latest state

            available_resources = [res for res in self._resources if res.is_available()]
            if not available_resources:
                return None

            # Prioritize resources by health score (highest first)
            available_resources.sort(key=lambda r: r.health_score, reverse=True)

            healthiest_resource = available_resources[0]
            with healthiest_resource.lock:  # Acquire lock on the specific resource
                healthiest_resource.state = ResourceState.IN_USE
                healthiest_resource.last_reserved_timestamp = time.monotonic()
            return healthiest_resource

    @contextmanager
    def get_resource(self):
        resource = self._reserve_resource()
        if not resource:
            raise NoResourcesAvailableError(provider=self.provider_name)
        try:
            yield resource
        finally:
            resource.release()

    def penalize_resource(self, resource_value: str):
        with self.lock:
            for resource in self._resources:
                if resource.value == resource_value:
                    resource.penalize()
                    return
        log.warning(
            f"Attempted to penalize a resource value that was not found: {resource_value[:4]}..."
        )

    def get_all_resources(self) -> List[MonitoredResource]:
        with self.lock:
            # When getting all resources, ensure their states are up-to-date
            for res in self._resources:
                res._update_health()
            return list(self._resources)
