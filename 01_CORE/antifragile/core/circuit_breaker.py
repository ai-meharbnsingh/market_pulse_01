# antifragile_framework/core/circuit_breaker.py

import threading
import time
from enum import Enum, auto
from typing import Any, Dict


class CircuitBreakerState(Enum):
    """The possible states of the circuit breaker."""

    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()


class CircuitBreakerError(Exception):
    """Custom exception raised when a call is blocked by an open circuit breaker."""

    def __init__(self, service_name: str):
        self.service_name = service_name
        super().__init__(f"CircuitBreaker for '{service_name}' is open.")


class CircuitBreaker:
    """A stateful object that wraps calls to a service to prevent cascading failures."""

    def __init__(
        self,
        service_name: str,
        failure_threshold: int = 5,
        reset_timeout_seconds: int = 60,
    ):
        if failure_threshold <= 0:
            raise ValueError("Failure threshold must be positive.")
        if reset_timeout_seconds <= 0:
            raise ValueError("Reset timeout must be positive.")

        self.service_name = service_name
        self._failure_threshold = failure_threshold
        self._reset_timeout_seconds = reset_timeout_seconds

        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time: float = 0.0
        self.lock = threading.Lock()

    def _set_state(self, new_state: CircuitBreakerState):
        if self.state != new_state:
            self.state = new_state

    def check(self):
        with self.lock:
            if self.state == CircuitBreakerState.OPEN:
                time_since_failure = time.monotonic() - self.last_failure_time
                if time_since_failure > self._reset_timeout_seconds:
                    self._set_state(CircuitBreakerState.HALF_OPEN)
                else:
                    raise CircuitBreakerError(self.service_name)

    def record_failure(self):
        with self.lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self._trip()
            else:
                self.failure_count += 1
                if self.failure_count >= self._failure_threshold:
                    self._trip()

    def record_success(self):
        self.reset()

    def _trip(self):
        self._set_state(CircuitBreakerState.OPEN)
        self.last_failure_time = time.monotonic()
        self.failure_count = 0

    def reset(self):
        with self.lock:
            self._set_state(CircuitBreakerState.CLOSED)
            self.failure_count = 0
            self.last_failure_time = 0.0


class CircuitBreakerRegistry:
    """Manages a collection of CircuitBreaker instances, one for each service/provider."""

    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def get_breaker(self, service_name: str, **kwargs: Any) -> CircuitBreaker:
        if service_name in self._breakers:
            return self._breakers[service_name]
        with self._lock:
            if service_name not in self._breakers:
                self._breakers[service_name] = CircuitBreaker(
                    service_name=service_name, **kwargs
                )
            return self._breakers[service_name]
