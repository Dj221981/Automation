"""Advanced retry strategies with backoff, jitter, and retry budgets."""

from __future__ import annotations

import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, Generic, List, Optional, Type, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class RetryConfig:
    max_attempts: int = 3
    base_delay_seconds: float = 0.05
    max_delay_seconds: float = 2.0
    jitter_seconds: float = 0.05
    retry_budget_per_minute: int = 200
    adaptive_delays: Dict[Type[BaseException], float] | None = None


class RetryBudget:
    def __init__(self, max_events_per_minute: int):
        self._max_events = max_events_per_minute
        self._lock = threading.RLock()
        self._events: Deque[float] = deque()

    def consume(self) -> bool:
        now = time.monotonic()
        with self._lock:
            while self._events and now - self._events[0] > 60.0:
                self._events.popleft()
            if len(self._events) >= self._max_events:
                return False
            self._events.append(now)
            return True


class RetryExecutor:
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self._budget = RetryBudget(self.config.retry_budget_per_minute)

    def _delay_for(self, attempt: int, exc: BaseException) -> float:
        adaptive = self.config.adaptive_delays or {}
        for exc_type, delay in adaptive.items():
            if isinstance(exc, exc_type):
                return min(self.config.max_delay_seconds, max(0.0, delay))
        exp_delay = min(self.config.max_delay_seconds, self.config.base_delay_seconds * (2 ** max(0, attempt - 1)))
        jitter = random.uniform(0.0, max(0.0, self.config.jitter_seconds))
        return min(self.config.max_delay_seconds, exp_delay + jitter)

    def execute(self, operation: Callable[..., T], *args: Any, retryable: tuple[type[BaseException], ...] = (Exception,), **kwargs: Any) -> T:
        last_error: Optional[BaseException] = None
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                return operation(*args, **kwargs)
            except retryable as exc:
                last_error = exc
                if attempt >= self.config.max_attempts or not self._budget.consume():
                    break
                time.sleep(self._delay_for(attempt, exc))
        if last_error is not None:
            raise last_error
        raise RuntimeError("retry execution failed without captured exception")


class DeadLetterQueue(Generic[T]):
    def __init__(self, max_items: int = 5000):
        self._items: Deque[T] = deque(maxlen=max_items)
        self._lock = threading.RLock()

    def put(self, item: T) -> None:
        with self._lock:
            self._items.append(item)

    def items(self) -> List[T]:
        with self._lock:
            return list(self._items)
