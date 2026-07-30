"""Bulkhead-style resource isolation primitives."""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Set

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None


@dataclass(frozen=True)
class ThreadPoolBulkheadConfig:
    max_workers: int = 4
    max_queue_size: int = 100


class ThreadPoolBulkhead:
    def __init__(self, name: str, config: Optional[ThreadPoolBulkheadConfig] = None):
        self.name = name
        self.config = config or ThreadPoolBulkheadConfig()
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers, thread_name_prefix=f"bulkhead-{name}")
        self._lock = threading.RLock()
        self._inflight: Set[Future[Any]] = set()

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future[Any]:
        with self._lock:
            if len(self._inflight) >= self.config.max_queue_size:
                raise RuntimeError(f"Bulkhead '{self.name}' queue limit reached")
            future = self._executor.submit(fn, *args, **kwargs)
            self._inflight.add(future)
            future.add_done_callback(lambda f: self._inflight.discard(f))
            return future


class MemoryUsageLimiter:
    def __init__(self, max_memory_mb: float):
        self.max_memory_mb = max_memory_mb

    def allow(self) -> bool:
        if psutil is None:
            return True
        rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        return rss_mb <= self.max_memory_mb


class TaskQueueLimiter:
    def __init__(self, max_tasks_per_agent: int):
        self.max_tasks_per_agent = max_tasks_per_agent
        self._lock = threading.RLock()
        self._counts: Dict[str, int] = {}

    def try_acquire(self, agent_id: str) -> bool:
        with self._lock:
            current = self._counts.get(agent_id, 0)
            if current >= self.max_tasks_per_agent:
                return False
            self._counts[agent_id] = current + 1
            return True

    def release(self, agent_id: str) -> None:
        with self._lock:
            current = self._counts.get(agent_id, 0)
            if current <= 1:
                self._counts.pop(agent_id, None)
            else:
                self._counts[agent_id] = current - 1
