"""Performance tracking, profiling and bottleneck analysis."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from statistics import mean
from typing import Deque, Dict, Iterator, List

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None


@dataclass(frozen=True)
class PerformanceSample:
    operation: str
    duration_seconds: float
    timestamp_monotonic: float


class PerformanceTracker:
    def __init__(self, max_samples_per_operation: int = 1000):
        self.max_samples_per_operation = max_samples_per_operation
        self._lock = threading.RLock()
        self._samples: Dict[str, Deque[PerformanceSample]] = defaultdict(lambda: deque(maxlen=max_samples_per_operation))

    @contextmanager
    def track(self, operation: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            with self._lock:
                self._samples[operation].append(
                    PerformanceSample(operation=operation, duration_seconds=duration, timestamp_monotonic=time.monotonic())
                )

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            report: Dict[str, Dict[str, float]] = {}
            for op, samples in self._samples.items():
                durations = [sample.duration_seconds for sample in samples]
                if durations:
                    report[op] = {
                        "count": float(len(durations)),
                        "avg_duration": float(mean(durations)),
                        "max_duration": float(max(durations)),
                    }
            return report

    def identify_bottlenecks(self, top_n: int = 3) -> List[str]:
        snap = self.snapshot()
        ranked = sorted(snap.items(), key=lambda item: item[1].get("avg_duration", 0.0), reverse=True)
        return [name for name, _ in ranked[:top_n]]

    def resource_snapshot(self) -> Dict[str, float]:
        if psutil is None:
            return {"cpu_percent": 0.0, "memory_rss_bytes": 0.0}
        process = psutil.Process()
        return {
            "cpu_percent": float(psutil.cpu_percent(interval=0.0)),
            "memory_rss_bytes": float(process.memory_info().rss),
        }
