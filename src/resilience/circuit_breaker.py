"""Circuit breaker primitives for resilient integrations."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")


class CircuitBreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpenError(RuntimeError):
    pass


@dataclass(frozen=True)
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout_seconds: float = 30.0
    half_open_max_calls: int = 1


class CircuitBreaker:
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._lock = threading.RLock()
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._opened_at = 0.0
        self._half_open_calls = 0

    @property
    def state(self) -> CircuitBreakerState:
        with self._lock:
            return self._state

    def _allow_call(self) -> bool:
        with self._lock:
            if self._state == CircuitBreakerState.CLOSED:
                return True
            if self._state == CircuitBreakerState.OPEN:
                elapsed = time.monotonic() - self._opened_at
                if elapsed >= self.config.recovery_timeout_seconds:
                    self._state = CircuitBreakerState.HALF_OPEN
                    self._half_open_calls = 0
                else:
                    return False
            if self._state == CircuitBreakerState.HALF_OPEN:
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
            return True

    def _record_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            self._half_open_calls = 0
            self._state = CircuitBreakerState.CLOSED

    def _record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            if self._failure_count >= self.config.failure_threshold:
                self._state = CircuitBreakerState.OPEN
                self._opened_at = time.monotonic()

    def call(self, operation: Callable[..., T], *args: Any, fallback: Optional[Callable[[Exception], T]] = None, **kwargs: Any) -> T:
        if not self._allow_call():
            err = CircuitBreakerOpenError(f"Circuit '{self.name}' is open")
            if fallback:
                return fallback(err)
            raise err

        try:
            result = operation(*args, **kwargs)
            self._record_success()
            return result
        except Exception as exc:
            self._record_failure()
            if fallback:
                return fallback(exc)
            raise
