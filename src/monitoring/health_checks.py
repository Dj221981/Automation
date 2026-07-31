"""Health check probes for liveness/readiness and dependency status."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass(frozen=True)
class HealthCheckResult:
    name: str
    healthy: bool
    details: str


class HealthChecker:
    def __init__(self):
        self._checks: Dict[str, Callable[[], HealthCheckResult]] = {}

    def register(self, name: str, check: Callable[[], HealthCheckResult]) -> None:
        self._checks[name] = check

    def run(self) -> Dict[str, Any]:
        results: List[HealthCheckResult] = []
        for name, check in self._checks.items():
            try:
                results.append(check())
            except Exception as exc:
                results.append(HealthCheckResult(name=name, healthy=False, details=f"check failed: {exc}"))
        return {
            "healthy": all(item.healthy for item in results),
            "checks": [item.__dict__ for item in results],
        }


def agent_health_check(agent_system: Any) -> HealthCheckResult:
    responsive = bool(getattr(agent_system, "agents", {}))
    return HealthCheckResult("agent_health", responsive, "agent registry available" if responsive else "no agents available")


def database_health_check(task_store: Any) -> HealthCheckResult:
    try:
        task_store.list_tasks(None)
        return HealthCheckResult("database", True, "task store query succeeded")
    except Exception as exc:
        return HealthCheckResult("database", False, str(exc))


def redis_health_check(redis_client: Any) -> HealthCheckResult:
    if redis_client is None:
        return HealthCheckResult("redis", True, "redis check skipped (not configured)")
    try:
        redis_client.ping()
        return HealthCheckResult("redis", True, "redis ping succeeded")
    except Exception as exc:
        return HealthCheckResult("redis", False, str(exc))


def model_health_check(model: Any) -> HealthCheckResult:
    loaded = getattr(model, "network", None) is not None
    return HealthCheckResult("model", loaded, "model loaded" if loaded else "model not loaded")


def queue_health_check(agent_system: Any, warning_threshold: Optional[int] = None) -> HealthCheckResult:
    queue_depth = len(getattr(agent_system, "_task_index", []))
    max_size = getattr(agent_system, "max_queue_size", 0)
    threshold = warning_threshold if warning_threshold is not None else int(max_size * 0.9) if max_size else 1000
    healthy = queue_depth <= threshold
    return HealthCheckResult("task_queue", healthy, f"depth={queue_depth} threshold={threshold}")
