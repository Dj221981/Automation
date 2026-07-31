"""
Super Agentic Agents Framework
===============================

A sophisticated multi-agent system architecture for Ai-morphasis 2.0.
This module provides the core infrastructure for creating, managing,
and orchestrating intelligent agentic agents with evolved capabilities.

Features:
    - Hierarchical agent architecture
    - Agent memory and state management (with optional Redis persistence)
    - Inter-agent communication
    - Distributed async task execution with concurrency control
    - Dynamic capability evolution
    - Agent reasoning and decision-making (with optional LLM integration)
    - Retry & exponential backoff error recovery
    - Structured JSON logging & observability
    - Pydantic v2 input validation
    - Thread-safe synchronous execution with monitoring
    - Task persistence (in-memory, Redis, Postgres)
    - Health checks & performance tracking
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import heapq
import json
import logging
import os
import random
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type

from src.monitoring.health_checks import (
    HealthChecker,
    agent_health_check,
    database_health_check,
    queue_health_check,
    redis_health_check,
)
from src.monitoring.performance_tracker import PerformanceTracker
from src.monitoring.thresholds import ThresholdMonitor
from src.observability.metrics import get_metrics_registry
from src.observability.structured_logging import get_logger
from src.observability.tracing import get_tracing_manager
from src.resilience.bulkheads import TaskQueueLimiter
from src.resilience.circuit_breaker import CircuitBreaker

from .task_store import InMemoryTaskStore, StoredTask, TaskStore

# ---------------------------------------------------------------------------
# Optional dependency imports (graceful degradation)
# ---------------------------------------------------------------------------
try:
    import openai  # noqa: F401 - used via type hints only at runtime
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False

try:
    import redis.asyncio as aioredis
    _HAS_REDIS = True
except ImportError:
    _HAS_REDIS = False

try:
    from pydantic import BaseModel, Field, field_validator
    import pydantic
    _HAS_PYDANTIC = True
except ImportError:
    _HAS_PYDANTIC = False


# ---------------------------------------------------------------------------
# Structured observability helpers (Phase 1 hardening)
# ---------------------------------------------------------------------------


class _Counter:
    """Thread-safe integer counter for structured metrics."""

    def __init__(self) -> None:
        self._value = 0
        self._lock = threading.Lock()

    def increment(self, delta: int = 1) -> None:
        with self._lock:
            self._value += delta

    @property
    def value(self) -> int:
        with self._lock:
            return self._value


class _Timer:
    """Accumulates duration samples (seconds) for structured metrics."""

    def __init__(self) -> None:
        self._total = 0.0
        self._count = 0
        self._lock = threading.Lock()

    def record(self, elapsed: float) -> None:
        with self._lock:
            self._total += elapsed
            self._count += 1

    @property
    def avg(self) -> float:
        with self._lock:
            return self._total / self._count if self._count else 0.0

    @property
    def total(self) -> float:
        with self._lock:
            return self._total

    @property
    def count(self) -> int:
        with self._lock:
            return self._count


class SystemMetrics:
    """Collects structured counters and timers for an AgentSystem."""

    def __init__(self) -> None:
        self.tasks_created = _Counter()
        self.tasks_submitted = _Counter()
        self.tasks_completed = _Counter()
        self.tasks_failed = _Counter()
        self.tasks_dependency_blocked = _Counter()
        self.task_duration = _Timer()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tasks_created": self.tasks_created.value,
            "tasks_submitted": self.tasks_submitted.value,
            "tasks_completed": self.tasks_completed.value,
            "tasks_failed": self.tasks_failed.value,
            "tasks_dependency_blocked": self.tasks_dependency_blocked.value,
            "task_duration_avg_s": round(self.task_duration.avg, 6),
            "task_duration_total_s": round(self.task_duration.total, 6),
        }


# ---------------------------------------------------------------------------
# Structured Logger
# ---------------------------------------------------------------------------

class StructuredLogger:
    """Wraps :class:`logging.Logger` and auto-injects structured fields as JSON extras."""

    def __init__(self, name: str, agent_id: str = "", agent_name: str = "") -> None:
        self._logger = logging.getLogger(name)
        self._base_extra: Dict[str, Any] = {
            "agent_id": agent_id,
            "agent_name": agent_name,
        }

    def _extra(self, task_id: str = "", **kw: Any) -> Dict[str, Any]:
        extra = dict(self._base_extra)
        if task_id:
            extra["task_id"] = task_id
        extra.update(kw)
        return {"structured": extra}

    def info(self, msg: str, task_id: str = "", **kw: Any) -> None:
        self._logger.info(msg, extra=self._extra(task_id, **kw))

    def warning(self, msg: str, task_id: str = "", **kw: Any) -> None:
        self._logger.warning(msg, extra=self._extra(task_id, **kw))

    def error(self, msg: str, task_id: str = "", **kw: Any) -> None:
        self._logger.error(msg, extra=self._extra(task_id, **kw))

    def debug(self, msg: str, task_id: str = "", **kw: Any) -> None:
        self._logger.debug(msg, extra=self._extra(task_id, **kw))


logger = get_logger(__name__)
_TRACING = get_tracing_manager()
_METRICS = get_metrics_registry()

DEFAULT_CAPABILITY_TIMEOUT_SECONDS = 30.0
MIN_CAPABILITY_TIMEOUT_SECONDS = 0.01
MAX_CAPABILITY_TIMEOUT_SECONDS = 300.0
MIN_CAPABILITY_RATE_LIMIT = 1
MAX_CAPABILITY_RATE_LIMIT = 10000
DEFAULT_CAPABILITY_RETRY_ATTEMPTS = 1
MAX_CAPABILITY_RETRY_ATTEMPTS = 5
MAX_RETRY_BACKOFF_SECONDS = 5.0

DEFAULT_CLAIM_TTL_SECONDS = 60
DEFAULT_CLAIM_GRACE_SECONDS = 10
DEFAULT_CLAIM_HEARTBEAT_INTERVAL_SECONDS = 20
DEFAULT_MAX_QUEUE_SIZE = 10000
DEFAULT_MAX_PERSIST_RETRIES = 3


class AgentRole(Enum):
    """Defines the role/purpose of an agent."""

    ORCHESTRATOR = "orchestrator"
    EXECUTOR = "executor"
    ANALYZER = "analyzer"
    LEARNER = "learner"
    SUPERVISOR = "supervisor"
    SPECIALIZED = "specialized"


class AgentStatus(Enum):
    """Tracks the operational status of an agent."""

    IDLE = "idle"
    ACTIVE = "active"
    BUSY = "busy"
    LEARNING = "learning"
    ERROR = "error"
    SUSPENDED = "suspended"


class TaskStatus(Enum):
    """Lifecycle states for tasks."""

    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DEPENDENCY_BLOCKED = "dependency_blocked"


class DependencyError(RuntimeError):
    """Raised when a task cannot execute because its dependencies are unmet."""

    def __init__(self, task_id: str, unmet: List[str]) -> None:
        self.task_id = task_id
        self.unmet_dependencies = unmet
        super().__init__(
            f"Task {task_id} cannot execute: unmet dependencies {unmet}"
        )


class TaskPriority(Enum):
    """Defines task execution priority levels."""

    CRITICAL = 5
    HIGH = 4
    NORMAL = 3
    LOW = 2
    DEFERRED = 1


@dataclass(slots=True)
class AgentCapability:
    """Represents a capability an agent can perform."""

    name: str
    description: str
    func: Optional[Callable[..., Any]] = None
    confidence_score: float = 1.0
    requires_resources: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    safe_mode_only: bool = True
    allowed_roles: Set[AgentRole] = field(
        default_factory=lambda: {
            AgentRole.EXECUTOR,
            AgentRole.ANALYZER,
            AgentRole.LEARNER,
            AgentRole.ORCHESTRATOR,
        }
    )
    max_calls_per_minute: int = 60
    timeout_seconds: float = DEFAULT_CAPABILITY_TIMEOUT_SECONDS
    retry_attempts: int = DEFAULT_CAPABILITY_RETRY_ATTEMPTS
    retry_backoff_seconds: float = 0.0
    retry_jitter_seconds: float = 0.0
    non_retryable_exceptions: Tuple[Type[BaseException], ...] = (PermissionError,)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Capability name cannot be empty")
        if not self.description.strip():
            raise ValueError("Capability description cannot be empty")
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("Capability confidence_score must be between 0.0 and 1.0")
        if not MIN_CAPABILITY_RATE_LIMIT <= self.max_calls_per_minute <= MAX_CAPABILITY_RATE_LIMIT:
            raise ValueError(
                f"max_calls_per_minute must be between {MIN_CAPABILITY_RATE_LIMIT} and {MAX_CAPABILITY_RATE_LIMIT}"
            )
        if not MIN_CAPABILITY_TIMEOUT_SECONDS <= self.timeout_seconds <= MAX_CAPABILITY_TIMEOUT_SECONDS:
            raise ValueError(
                f"timeout_seconds must be between {MIN_CAPABILITY_TIMEOUT_SECONDS} and {MAX_CAPABILITY_TIMEOUT_SECONDS}"
            )
        if not DEFAULT_CAPABILITY_RETRY_ATTEMPTS <= self.retry_attempts <= MAX_CAPABILITY_RETRY_ATTEMPTS:
            raise ValueError(
                f"retry_attempts must be between {DEFAULT_CAPABILITY_RETRY_ATTEMPTS} and {MAX_CAPABILITY_RETRY_ATTEMPTS}"
            )
        if self.retry_backoff_seconds < 0 or self.retry_backoff_seconds > MAX_RETRY_BACKOFF_SECONDS:
            raise ValueError(f"retry_backoff_seconds must be between 0 and {MAX_RETRY_BACKOFF_SECONDS}")
        if self.retry_jitter_seconds < 0 or self.retry_jitter_seconds > MAX_RETRY_BACKOFF_SECONDS:
            raise ValueError(f"retry_jitter_seconds must be between 0 and {MAX_RETRY_BACKOFF_SECONDS}")

    def __repr__(self) -> str:
        return f"<Capability: {self.name} v{self.version} ({self.confidence_score:.2%})>"


@dataclass(slots=True)
class AgentMemory:
    """Represents agent memory with episodic and semantic storage.

    Supports an optional Redis backend for persistence.  When *redis_url* is
    provided the class will attempt to create an async Redis connection on first
    use; if Redis is unavailable it falls back transparently to in-process dicts
    and logs a warning.
    """

    agent_id: str
    episodic_memory: Dict[str, Any] = field(default_factory=dict)
    semantic_memory: Dict[str, Any] = field(default_factory=dict)
    procedural_memory: Dict[str, Callable[..., Any]] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    max_episodes: int = 1000
    redis_url: Optional[str] = field(default=None)
    _redis: Any = field(default=None, init=False, repr=False, compare=False)

    def store_episode(self, key: str, value: Any) -> None:
        """Store an episode in short-term memory (in-process)."""
        if len(self.episodic_memory) >= self.max_episodes:
            oldest_key = next(iter(self.episodic_memory))
            del self.episodic_memory[oldest_key]
        self.episodic_memory[key] = {"value": value, "timestamp": datetime.now()}
        self.last_accessed = datetime.now()

    def store_semantic(self, key: str, value: Any) -> None:
        """Store knowledge in long-term memory (in-process)."""
        self.semantic_memory[key] = {
            "value": value,
            "timestamp": datetime.now(),
            "access_count": 0,
        }
        self.last_accessed = datetime.now()

    def retrieve(self, key: str, memory_type: str = "auto") -> Optional[Any]:
        """Retrieve from memory (auto-selects best source, in-process only)."""
        if memory_type in ("auto", "episodic") and key in self.episodic_memory:
            self.last_accessed = datetime.now()
            return self.episodic_memory[key]["value"]
        if memory_type in ("auto", "semantic") and key in self.semantic_memory:
            self.semantic_memory[key]["access_count"] += 1
            self.last_accessed = datetime.now()
            return self.semantic_memory[key]["value"]
        return None

    # ------------------------------------------------------------------
    # Redis helpers (async, optional)
    # ------------------------------------------------------------------

    async def _get_redis(self) -> Optional[Any]:
        """Return a connected Redis client, or None if unavailable."""
        if not _HAS_REDIS or not self.redis_url:
            return None
        if self._redis is None:
            try:
                self._redis = aioredis.from_url(
                    self.redis_url, decode_responses=True
                )
                await self._redis.ping()
            except Exception as exc:
                logger.warning("Redis unavailable (%s); using in-memory store.", exc)
                self._redis = None
        return self._redis

    def _episodic_redis_key(self, key: str) -> str:
        return f"agent:{self.agent_id}:episodic:{key}"

    def _semantic_redis_key(self, key: str) -> str:
        return f"agent:{self.agent_id}:semantic:{key}"

    async def async_store_episode(self, key: str, value: Any) -> None:
        """Store an episode in short-term memory (with optional Redis TTL=3600s)."""
        self.store_episode(key, value)
        redis = await self._get_redis()
        if redis:
            try:
                payload = json.dumps({"value": value, "timestamp": datetime.now().isoformat()},
                                     default=str)
                await redis.set(self._episodic_redis_key(key), payload, ex=3600)
            except Exception as exc:
                logger.warning("Redis store_episode error: %s", exc)

    async def async_store_semantic(self, key: str, value: Any) -> None:
        """Store knowledge in long-term memory (with optional Redis, no expiry)."""
        self.store_semantic(key, value)
        redis = await self._get_redis()
        if redis:
            try:
                payload = json.dumps(
                    {"value": value, "timestamp": datetime.now().isoformat(), "access_count": 0},
                    default=str,
                )
                await redis.set(self._semantic_redis_key(key), payload)
            except Exception as exc:
                logger.warning("Redis store_semantic error: %s", exc)

    async def async_retrieve(self, key: str, memory_type: str = "auto") -> Optional[Any]:
        """Retrieve from memory, checking Redis when available."""
        local = self.retrieve(key, memory_type)
        if local is not None:
            return local
        redis = await self._get_redis()
        if redis:
            memory_types = ["episodic", "semantic"] if memory_type == "auto" else [memory_type]
            redis_key_fn = {
                "episodic": self._episodic_redis_key,
                "semantic": self._semantic_redis_key,
            }
            for kind in memory_types:
                rkey = redis_key_fn[kind](key)
                try:
                    raw = await redis.get(rkey)
                    if raw:
                        data = json.loads(raw)
                        return data.get("value")
                except Exception as exc:
                    logger.warning("Redis retrieve error: %s", exc)
        return None


@dataclass(slots=True)
class Task:
    """Represents a task for agents to execute."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    priority: TaskPriority = TaskPriority.NORMAL
    assigned_to: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.description.strip():
            raise ValueError("Task description cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "priority": self.priority.name,
            "assigned_to": self.assigned_to,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": _make_json_safe(self.result),
            "error": self.error,
            "parameters": _make_json_safe(self.parameters),
            "dependencies": list(self.dependencies),
            "metadata": _make_json_safe(self.metadata),
        }


@dataclass(order=True, slots=True)
class QueuedTask:
    """Represents a heap-backed queued task entry."""

    priority_rank: int
    created_at_ts: float
    sequence: int
    id: str = field(compare=False)

# ============================================================================
# Pydantic v2 Validation Models (graceful degradation if pydantic not installed)
# ============================================================================

if _HAS_PYDANTIC:
    class AgentConfig(BaseModel):
        """Validates inputs for creating an agent."""

        name: str = Field(..., min_length=1, max_length=100)
        role: AgentRole = AgentRole.EXECUTOR
        max_capabilities: int = Field(default=50, ge=1, le=200)
        max_retries: int = Field(default=3, ge=0, le=10)

    class TaskConfig(BaseModel):
        """Validates inputs for creating a task."""

        description: str = Field(..., min_length=1)
        priority: TaskPriority = TaskPriority.NORMAL
        parameters: Dict[str, Any] = Field(default_factory=dict)
        dependencies: List[str] = Field(default_factory=list)

        @field_validator("dependencies")
        @classmethod
        def validate_uuids(cls, v: List[str]) -> List[str]:
            import re
            uuid_re = re.compile(
                r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
                re.IGNORECASE,
            )
            for item in v:
                if not uuid_re.match(item):
                    raise ValueError(f"Invalid UUID in dependencies: {item}")
            return v

else:
    # Minimal fallback when pydantic is not available
    class AgentConfig:  # type: ignore[no-redef]
        """Fallback AgentConfig without Pydantic validation."""

        def __init__(
            self,
            name: str,
            role: AgentRole = AgentRole.EXECUTOR,
            max_capabilities: int = 50,
            max_retries: int = 3,
        ) -> None:
            if not name or len(name) > 100:
                raise ValueError("Agent name must be 1-100 characters")
            if not 1 <= max_capabilities <= 200:
                raise ValueError("max_capabilities must be 1-200")
            if not 0 <= max_retries <= 10:
                raise ValueError("max_retries must be 0-10")
            self.name = name
            self.role = role
            self.max_capabilities = max_capabilities
            self.max_retries = max_retries

    class TaskConfig:  # type: ignore[no-redef]
        """Fallback TaskConfig without Pydantic validation."""

        def __init__(
            self,
            description: str,
            priority: TaskPriority = TaskPriority.NORMAL,
            parameters: Optional[Dict[str, Any]] = None,
            dependencies: Optional[List[str]] = None,
        ) -> None:
            import re
            if not description:
                raise ValueError("Task description cannot be empty")
            uuid_re = re.compile(
                r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
                re.IGNORECASE,
            )
            for item in (dependencies or []):
                if not uuid_re.match(item):
                    raise ValueError(f"Invalid UUID in dependencies: {item}")
            self.description = description
            self.priority = priority
            self.parameters = parameters or {}
            self.dependencies = dependencies or []


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(
        self,
        name: str,
        role: AgentRole = AgentRole.EXECUTOR,
        max_capabilities: int = 50,
        *,
        redis_url: Optional[str] = None,
        llm_client: Optional[Any] = None,
    ):
        if not name.strip():
            raise ValueError("Agent name cannot be empty")
        if max_capabilities <= 0:
            raise ValueError("max_capabilities must be greater than 0")

        self.id = str(uuid.uuid4())
        self.name = name
        self.role = role
        self.status = AgentStatus.IDLE
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.capabilities: Dict[str, AgentCapability] = {}
        self.max_capabilities = max_capabilities
        self.memory = AgentMemory(agent_id=self.id, redis_url=redis_url)
        self.llm_client = llm_client
        self._slog = StructuredLogger(__name__, agent_id=self.id, agent_name=name)
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: List[Task] = []
        self.failed_tasks: List[Task] = []
        self.task_history: List[Task] = []
        self.parent_agent: Optional[str] = None
        self.child_agents: Set[str] = set()
        self.peer_agents: Set[str] = set()
        self._lock = threading.RLock()
        self.safety_mode = True
        self._capability_call_log: Dict[str, List[datetime]] = {}
        self._audit_log: List[Dict[str, Any]] = []
        self.max_audit_entries = 5000
        self._capability_executor = ThreadPoolExecutor(max_workers=max(4, max_capabilities), thread_name_prefix=f"cap-{self.name}")
        self.performance_metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_total": 0,
            "avg_task_time_success": 0.0,
            "avg_task_time_failure": 0.0,
            "avg_task_time_overall": 0.0,
            "success_rate": 1.0,
            "capability_timeouts": 0,
            "capability_retries": 0,
            "capability_failures": 0,
        }
        logger.info("Initialized %s agent: %s (ID: %s)", self.role.value, self.name, self.id)

    @abstractmethod
    def think(self, input_data: Any) -> Dict[str, Any]:
        """Core reasoning method - must be implemented by subclasses."""

    @abstractmethod
    def act(self, decision: Dict[str, Any]) -> Any:
        """Execution method - must be implemented by subclasses."""

    def _touch(self) -> None:
        self.last_activity = datetime.now()

    def _audit(self, event: str, payload: Dict[str, Any]) -> None:
        entry = {
            "ts": datetime.now().isoformat(),
            "agent_id": self.id,
            "agent_name": self.name,
            "event": event,
            "payload": _make_json_safe(payload),
        }
        self._audit_log.append(entry)
        if len(self._audit_log) > self.max_audit_entries:
            self._audit_log = self._audit_log[-self.max_audit_entries :]

    def _validate_capability_use(self, capability: AgentCapability) -> None:
        if self.role not in capability.allowed_roles:
            raise PermissionError(f"Role {self.role.value} cannot use capability {capability.name}")
        if self.safety_mode and not capability.safe_mode_only:
            raise PermissionError(f"Capability {capability.name} blocked by safety_mode")
        now = datetime.now()
        calls = self._capability_call_log.setdefault(capability.name, [])
        cutoff = now.timestamp() - 60
        calls[:] = [ts for ts in calls if ts.timestamp() >= cutoff]
        if len(calls) >= capability.max_calls_per_minute:
            raise RuntimeError(f"Rate limit exceeded for capability {capability.name}")
        calls.append(now)

    def _run_capability_with_timeout(self, capability: AgentCapability, kwargs: Dict[str, Any]) -> Any:
        future: Future[Any] = self._capability_executor.submit(capability.func, **kwargs)  # type: ignore[misc]
        try:
            return future.result(timeout=capability.timeout_seconds)
        except FutureTimeoutError as exc:
            future.cancel()
            raise TimeoutError(
                f"Capability {capability.name} timed out after {capability.timeout_seconds:.3f}s"
            ) from exc

    @staticmethod
    def _is_retryable_exception(capability: AgentCapability, exc: BaseException) -> bool:
        return not isinstance(exc, capability.non_retryable_exceptions)

    def execute_capability(self, capability_name: str, **kwargs: Any) -> Any:
        with self._lock:
            capability = self.capabilities.get(capability_name)
            if capability is None:
                raise KeyError(f"Capability not found: {capability_name}")
            if capability.func is None:
                raise ValueError(f"Capability {capability_name} has no bound function")
            self._validate_capability_use(capability)
            self._audit("capability_execute_start", {"capability": capability_name, "kwargs_keys": list(kwargs.keys())})
        last_error: Optional[BaseException] = None
        with _TRACING.start_span(
            "agent.capability.execute",
            attributes={"agent.id": self.id, "agent.role": self.role.value, "capability.name": capability_name},
        ):
            for attempt in range(1, capability.retry_attempts + 1):
                start = datetime.now()
                with self._lock:
                    self._audit(
                        "capability_execute_attempt",
                        {
                            "capability": capability_name,
                            "attempt": attempt,
                            "max_attempts": capability.retry_attempts,
                        },
                    )
                try:
                    result = self._run_capability_with_timeout(capability, kwargs)
                    elapsed = (datetime.now() - start).total_seconds()
                    with self._lock:
                        self._audit(
                            "capability_execute_success",
                            {
                                "capability": capability_name,
                                "attempt": attempt,
                                "elapsed_s": elapsed,
                            },
                        )
                    return result
                except Exception as exc:
                    last_error = exc
                    retryable = self._is_retryable_exception(capability, exc)
                    final_attempt = attempt >= capability.retry_attempts
                    with self._lock:
                        if isinstance(exc, TimeoutError):
                            self.performance_metrics["capability_timeouts"] += 1
                        self.performance_metrics["capability_failures"] += 1
                        self._audit(
                            "capability_execute_error",
                            {
                                "capability": capability_name,
                                "attempt": attempt,
                                "retryable": retryable,
                                "final_attempt": final_attempt,
                                "error_type": exc.__class__.__name__,
                                "error": str(exc),
                            },
                        )
                    if final_attempt or not retryable:
                        raise
                    sleep_for = capability.retry_backoff_seconds * attempt
                    if capability.retry_jitter_seconds:
                        sleep_for += random.uniform(0.0, capability.retry_jitter_seconds)
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                    with self._lock:
                        self.performance_metrics["capability_retries"] += 1

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Capability execution failed: {capability_name}")

    def register_capability(self, capability: AgentCapability) -> bool:
        with self._lock:
            if capability.name in self.capabilities:
                logger.warning(
                    "Capability '%s' already registered for %s", capability.name, self.name
                )
                return False
            if len(self.capabilities) >= self.max_capabilities:
                logger.warning("Agent %s has reached max capabilities limit", self.name)
                return False
            self.capabilities[capability.name] = capability
            self.memory.store_semantic(f"capability:{capability.name}", capability)
            self._touch()
            logger.info("Capability '%s' registered for %s", capability.name, self.name)
            return True

    def get_capability(self, name: str) -> Optional[AgentCapability]:
        return self.capabilities.get(name)

    def list_capabilities(self) -> List[str]:
        return list(self.capabilities.keys())

    def assign_task(self, task: Task) -> bool:
        with self._lock:
            if task.status not in {TaskStatus.PENDING, TaskStatus.ASSIGNED}:
                logger.warning(
                    "Task %s is not assignable because it is in status %s",
                    task.id,
                    task.status.value,
                )
                return False
            if task.id in self.active_tasks:
                logger.warning("Task %s is already active on agent %s", task.id, self.name)
                return False
            if task.assigned_to and task.assigned_to != self.id:
                logger.warning("Task %s is already assigned to another agent", task.id)
                return False

            self.active_tasks[task.id] = task
            task.assigned_to = self.id
            task.status = TaskStatus.ASSIGNED
            self.memory.store_episode(f"task:{task.id}", task)
            self._touch()
            logger.info("Task %s assigned to agent %s", task.id, self.name)
            return True

    def run_task(self, task: Task, completed_task_ids: Optional[Set[str]] = None) -> Any:
        """Execute an assigned task synchronously.

        Parameters
        ----------
        task:
            The task to run.  Must already be assigned to this agent.
        completed_task_ids:
            Optional set of task IDs that have already completed.  When
            provided, any unmet entries in ``task.dependencies`` cause a
            :class:`DependencyError` to be raised before execution begins.
            Pass ``None`` (default) to skip the check (backward-compatible).
        """
        # Dependency enforcement (Phase 1 hardening)
        if task.dependencies and completed_task_ids is not None:
            unmet = [dep for dep in task.dependencies if dep not in completed_task_ids]
            if unmet:
                task.status = TaskStatus.DEPENDENCY_BLOCKED
                task.error = f"Unmet dependencies: {unmet}"
                logger.warning(
                    "Task %s blocked by unmet dependencies: %s",
                    task.id,
                    unmet,
                    extra={"task_id": task.id, "unmet_dependencies": unmet},
                )
                raise DependencyError(task.id, unmet)

        with self._lock:
            if task.id not in self.active_tasks:
                raise ValueError(f"Task {task.id} must be assigned before execution")
            if task.assigned_to != self.id:
                raise ValueError(f"Task {task.id} is assigned to {task.assigned_to}, not {self.id}")
            start_time = datetime.now()
            self.status = AgentStatus.BUSY
            self._touch()
            logger.info("Agent %s running task %s", self.name, task.id)

        try:
            reasoning = self.think(task.parameters)
            result = self.act(reasoning)
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            self._record_task_outcome(task, success=True, start_time=start_time)
            logger.info("Task %s completed successfully", task.id)
            return result
        except Exception:
            task.status = TaskStatus.FAILED
            self._record_task_outcome(task, success=False, start_time=start_time)
            logger.exception("Task %s failed", task.id)
            raise

    def _record_task_outcome(self, task: Task, success: bool, start_time: datetime) -> None:
        with self._lock:
            self._update_metrics(success=success, start_time=start_time)
            target_collection = self.completed_tasks if success else self.failed_tasks
            if all(existing.id != task.id for existing in target_collection):
                target_collection.append(task)
            if all(existing.id != task.id for existing in self.task_history):
                self.task_history.append(task)
            self.active_tasks.pop(task.id, None)
            self.status = AgentStatus.IDLE if success else AgentStatus.ERROR
            self.memory.store_episode(f"task:{task.id}", task)
            self._touch()

    async def execute_task(self, task: "Task") -> Any:
        """Async wrapper around run_task for callers using async/await.

        Sets task.status to COMPLETED on success or FAILED on error, and
        records ``duration_ms`` in task.metadata for observability.
        """
        if task.id not in self.active_tasks:
            self.assign_task(task)
        start = datetime.now()
        try:
            result = self.run_task(task)
            task.status = TaskStatus.COMPLETED
            elapsed_ms = (datetime.now() - start).total_seconds() * 1000
            if isinstance(task.metadata, dict):
                task.metadata["duration_ms"] = elapsed_ms
            return result
        except Exception as exc:
            task.status = TaskStatus.FAILED
            task.error = str(exc)
            raise

    def release_task(self, task_id: str) -> None:
        with self._lock:
            self.active_tasks.pop(task_id, None)
            self._touch()

    def reset_status(self) -> None:
        with self._lock:
            if self.active_tasks:
                raise RuntimeError(f"Cannot reset agent {self.name} while tasks are active")
            self.status = AgentStatus.IDLE
            self._touch()
            logger.info("Agent %s status reset to idle", self.name)

    def _update_metrics(self, success: bool, start_time: datetime) -> None:
        elapsed = (datetime.now() - start_time).total_seconds()
        key = "tasks_completed" if success else "tasks_failed"
        self.performance_metrics[key] += 1
        self.performance_metrics["tasks_total"] += 1
        total = self.performance_metrics["tasks_total"]
        self.performance_metrics["success_rate"] = (
            self.performance_metrics["tasks_completed"] / total if total else 0
        )
        overall_prev = self.performance_metrics["avg_task_time_overall"]
        self.performance_metrics["avg_task_time_overall"] = overall_prev + (
            (elapsed - overall_prev) / total
        )
        if success:
            n = self.performance_metrics["tasks_completed"]
            prev = self.performance_metrics["avg_task_time_success"]
            self.performance_metrics["avg_task_time_success"] = prev + ((elapsed - prev) / n)
        else:
            n = self.performance_metrics["tasks_failed"]
            prev = self.performance_metrics["avg_task_time_failure"]
            self.performance_metrics["avg_task_time_failure"] = prev + ((elapsed - prev) / n)

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "id": self.id,
                "name": self.name,
                "role": self.role.name,
                "status": self.status.name,
                "capabilities": self.list_capabilities(),
                "active_tasks": len(self.active_tasks),
                "completed_tasks": len(self.completed_tasks),
                "failed_tasks": len(self.failed_tasks),
                "performance": _make_json_safe(self.performance_metrics),
                "created_at": self.created_at.isoformat(),
                "last_activity": self.last_activity.isoformat(),
            }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name} ({self.role.value})>"


class OrchestratorAgent(BaseAgent):
    def __init__(self, name: str = "Orchestrator", **kwargs: Any):
        super().__init__(name, role=AgentRole.ORCHESTRATOR, **kwargs)
        self.managed_agents: Dict[str, BaseAgent] = {}
        self.task_queue: List[Task] = []

    def think(self, input_data: Any) -> Dict[str, Any]:
        return {
            "analysis": "Task requires orchestration",
            "priority": "high",
            "execution_strategy": "parallel",
        }

    def act(self, decision: Dict[str, Any]) -> Any:
        logger.info(
            "Orchestrator %s executing strategy: %s", self.name, decision.get("execution_strategy")
        )
        return {"status": "orchestration_complete"}

    def register_agent(self, agent: BaseAgent) -> bool:
        if agent.id in self.managed_agents:
            logger.warning(
                "Agent %s is already registered under orchestrator %s", agent.name, self.name
            )
            return False
        self.managed_agents[agent.id] = agent
        agent.parent_agent = self.id
        self._touch()
        logger.info("Agent %s registered under orchestrator %s", agent.name, self.name)
        return True

    def distribute_task(self, task: Task, target_agent_id: Optional[str] = None) -> bool:
        if task.status not in {TaskStatus.PENDING, TaskStatus.ASSIGNED}:
            logger.warning("Task %s cannot be distributed in status %s", task.id, task.status.value)
            return False
        if target_agent_id and target_agent_id in self.managed_agents:
            return self.managed_agents[target_agent_id].assign_task(task)
        best_agent = self._select_best_agent(task)
        if best_agent:
            return best_agent.assign_task(task)
        logger.warning("No suitable agent found for task %s", task.id)
        return False

    def _select_best_agent(self, task: Task) -> Optional[BaseAgent]:
        available_agents = [
            a
            for a in self.managed_agents.values()
            if a.status not in {AgentStatus.SUSPENDED, AgentStatus.BUSY}
        ]
        if not available_agents:
            return None

        task_description = task.description.lower()

        def score(agent: BaseAgent) -> tuple[int, int, int]:
            capability_match = 0
            for capability in agent.capabilities.values():
                capability_name = capability.name.lower()
                capability_description = capability.description.lower()
                if capability_name in task_description or task_description in capability_name:
                    capability_match += 3
                elif any(
                    token and token in task_description for token in capability_description.split()
                ):
                    capability_match += 1
            return (
                capability_match,
                -len(agent.active_tasks),
                int(agent.performance_metrics["success_rate"] * 100),
            )

        return max(available_agents, key=score)

    def get_system_status(self) -> Dict[str, Any]:
        return {
            "orchestrator": self.get_status(),
            "managed_agents": [a.get_status() for a in self.managed_agents.values()],
            "total_agents": len(self.managed_agents),
            "pending_tasks": len(self.task_queue),
        }


class ExecutorAgent(BaseAgent):
    def __init__(self, name: str = "Executor", **kwargs: Any):
        super().__init__(name, role=AgentRole.EXECUTOR, **kwargs)
        self.execution_history: List[Dict[str, Any]] = []

    def think(self, input_data: Any) -> Dict[str, Any]:
        return {"action": "execute", "parameters": input_data, "validation": True}

    def act(self, decision: Dict[str, Any]) -> Any:
        params = decision.get("parameters", {})
        self.execution_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "decision": _make_json_safe(decision),
                "result": "executed",
            }
        )
        return {"execution": "successful", "parameters_processed": _make_json_safe(params)}


class AnalyzerAgent(BaseAgent):
    def __init__(self, name: str = "Analyzer", **kwargs: Any):
        super().__init__(name, role=AgentRole.ANALYZER, **kwargs)
        self.analysis_cache: Dict[str, Dict[str, Any]] = {}

    def think(self, input_data: Any) -> Dict[str, Any]:
        return {
            "data_received": bool(input_data),
            "analysis_type": "comprehensive",
            "insights_generated": True,
        }

    def act(self, decision: Dict[str, Any]) -> Any:
        return {
            "analysis_complete": True,
            "insights": _make_json_safe(decision),
            "timestamp": datetime.now().isoformat(),
        }


class LearnerAgent(BaseAgent):
    def __init__(self, name: str = "Learner", **kwargs: Any):
        super().__init__(name, role=AgentRole.LEARNER, **kwargs)
        self.learned_patterns: Dict[str, Any] = {}
        self.learning_history: List[Dict[str, Any]] = []

    def think(self, input_data: Any) -> Dict[str, Any]:
        return {"learning_mode": True, "input_analyzed": True, "patterns_identified": []}

    def act(self, decision: Dict[str, Any]) -> Any:
        self.learning_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "decision": _make_json_safe(decision),
                "patterns_learned": len(self.learned_patterns),
            }
        )
        return {"learning": "in_progress", "patterns": _make_json_safe(self.learned_patterns)}

    def learn_from_experience(self, experience: Dict[str, Any]) -> None:
        pattern_id = str(uuid.uuid4())
        self.learned_patterns[pattern_id] = {
            "experience": _make_json_safe(experience),
            "learned_at": datetime.now().isoformat(),
            "confidence": 0.5,
        }
        self.memory.store_semantic(f"pattern:{pattern_id}", self.learned_patterns[pattern_id])
        self._touch()
        logger.info("Learner %s learned pattern: %s", self.name, pattern_id)


def _task_from_dict(data: Dict[str, Any]) -> "Task":
    """Reconstruct a :class:`Task` from its :meth:`Task.to_dict` representation."""
    priority_name = data.get("priority", TaskPriority.NORMAL.name)
    try:
        priority = TaskPriority[priority_name]
    except KeyError:
        priority = TaskPriority.NORMAL

    status_value = data.get("status", TaskStatus.PENDING.value)
    try:
        status = TaskStatus(status_value)
    except ValueError:
        status = TaskStatus.PENDING

    created_at = datetime.now()
    raw_created = data.get("created_at")
    if raw_created:
        try:
            created_at = datetime.fromisoformat(raw_created)
        except ValueError:
            pass

    completed_at: Optional[datetime] = None
    raw_completed = data.get("completed_at")
    if raw_completed:
        try:
            completed_at = datetime.fromisoformat(raw_completed)
        except ValueError:
            pass

    return Task(
        id=data.get("id", str(uuid.uuid4())),
        description=data.get("description") or "(restored)",
        priority=priority,
        assigned_to=data.get("assigned_to"),
        status=status,
        created_at=created_at,
        completed_at=completed_at,
        result=data.get("result"),
        error=data.get("error"),
        parameters=data.get("parameters", {}),
        dependencies=data.get("dependencies", []),
        metadata=data.get("metadata", {}),
    )


class AgentSystem:
    _ALLOWED_TRANSITIONS = {
        TaskStatus.PENDING: {TaskStatus.ASSIGNED, TaskStatus.CANCELLED},
        TaskStatus.ASSIGNED: {TaskStatus.RUNNING, TaskStatus.CANCELLED, TaskStatus.PENDING},
        TaskStatus.RUNNING: {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.PENDING},
        TaskStatus.FAILED: {TaskStatus.PENDING},
        TaskStatus.COMPLETED: set(),
        TaskStatus.CANCELLED: set(),
        TaskStatus.DEPENDENCY_BLOCKED: {TaskStatus.PENDING},
    }

    def __init__(
        self,
        name: str = "Ai-morphasis",
        task_store: Optional[TaskStore] = None,
        *,
        claim_ttl_seconds: int = DEFAULT_CLAIM_TTL_SECONDS,
        claim_grace_seconds: int = DEFAULT_CLAIM_GRACE_SECONDS,
        claim_heartbeat_interval_seconds: int = DEFAULT_CLAIM_HEARTBEAT_INTERVAL_SECONDS,
        max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE,
        max_persist_retries: int = DEFAULT_MAX_PERSIST_RETRIES,
    ):
        if not name.strip():
            raise ValueError("System name cannot be empty")
        if claim_ttl_seconds <= 0:
            raise ValueError("claim_ttl_seconds must be greater than 0")
        if claim_grace_seconds < 0:
            raise ValueError("claim_grace_seconds must be greater than or equal to 0")
        if claim_heartbeat_interval_seconds <= 0:
            raise ValueError("claim_heartbeat_interval_seconds must be greater than 0")
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be greater than 0")
        if max_persist_retries <= 0:
            raise ValueError("max_persist_retries must be greater than 0")

        self.name = name
        self.id = str(uuid.uuid4())
        self.created_at = datetime.now()
        self.task_store = task_store or InMemoryTaskStore()
        self._lock = threading.RLock()
        self._task_versions: Dict[str, int] = {}
        self._max_persist_retries = max_persist_retries
        self.persistence_backoff_min_seconds = 0.01
        self.persistence_backoff_max_seconds = 0.25
        self.persistence_backoff_max_exponent = 6
        self.persistence_backoff_jitter_ratio = 0.1
        self._retry_random = random.Random()
        self.claim_ttl_seconds = claim_ttl_seconds
        self.claim_grace_seconds = claim_grace_seconds
        self.claim_heartbeat_interval_seconds = claim_heartbeat_interval_seconds
        self.claim_sweep_interval_seconds = 1.0
        self.worker_poll_interval_seconds = 0.05
        self.orchestrator = OrchestratorAgent(f"{name}-Orchestrator")
        self.agents: Dict[str, BaseAgent] = {self.orchestrator.id: self.orchestrator}
        self.max_queue_size = max_queue_size
        self.global_task_queue: List[QueuedTask] = []
        self._task_index: Set[str] = set()
        self._queue_sequence = 0
        self._execution_results: Dict[str, Dict[str, Any]] = {}
        self._inflight_execution_keys: Set[str] = set()
        self.dead_letter_queue: deque[Dict[str, Any]] = deque(maxlen=2000)
        self.max_retries_per_task = 3
        self._queue_limiter = TaskQueueLimiter(max_tasks_per_agent=200)
        self._task_execution_breaker = CircuitBreaker("agent_task_execution")
        self.performance_tracker = PerformanceTracker()
        self.threshold_monitor = ThresholdMonitor()
        self.health_checker = HealthChecker()
        self.completed_tasks: List[Task] = []
        self.failed_tasks: List[Task] = []
        self.event_log: List[Dict[str, Any]] = []
        self.max_events = 10000
        self._idempotency_index: Dict[str, str] = {}
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._worker_threads: List[threading.Thread] = []
        self._maintenance_thread: Optional[threading.Thread] = None
        self.system_metrics = {
            "total_agents": 1,
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "completed_tasks_total": 0,
            "avg_task_duration_success": 0.0,
            "avg_task_duration_failure": 0.0,
            "avg_task_duration_overall": 0.0,
            "queue_stale_pops": 0,
            "dependency_blocked_tasks": 0,
            "claim_reclaims": 0,
            "persistence_retry_attempts": 0,
            "persistence_failures": 0,
            "queue_overflow_drops": 0,
            "queue_stale_entries_pruned": 0,
            "claim_renew_success": 0,
            "claim_renew_failures": 0,
            "claim_validation_failures": 0,
            "persistence_conflicts": 0,
            "persistence_retries": 0,
            "idempotent_hits": 0,
        }
        self.health_checker.register("agent_health", lambda: agent_health_check(self))
        self.health_checker.register("database", lambda: database_health_check(self.task_store))
        self.health_checker.register("redis", lambda: redis_health_check(None))
        self.health_checker.register("queue", lambda: queue_health_check(self))
        # Structured observability metrics (Phase 1 hardening)
        self.metrics = SystemMetrics()
        logger.info("Initialized Agent System: %s", self.name)

    def _emit_event(self, event_type: str, task: Optional[Task] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        correlation_id = None
        if task is not None and isinstance(task.metadata, dict):
            correlation_id = task.metadata.get("correlation_id")
        elif extra and isinstance(extra, dict):
            correlation_id = extra.get("correlation_id")
        payload: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "system_id": self.id,
            "system_name": self.name,
            "correlation_id": correlation_id,
        }
        if task is not None:
            payload.update(
                {
                    "task_id": task.id,
                    "task_status": task.status.value,
                    "assigned_to": task.assigned_to,
                    "claimed_by": task.metadata.get("claimed_by") if isinstance(task.metadata, dict) else None,
                }
            )
        if extra:
            payload["extra"] = _make_json_safe(extra)
        self.event_log.append(payload)
        if len(self.event_log) > self.max_events:
            self.event_log = self.event_log[-self.max_events :]
        logger.info("event=%s payload=%s", event_type, json.dumps(_make_json_safe(payload)))

    def _get_task_version(self, task: Task) -> int:
        if not isinstance(task.metadata, dict):
            task.metadata = {}
        raw = task.metadata.get("_version", 0)
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 0

    def _set_task_version(self, task: Task, version: int) -> None:
        if not isinstance(task.metadata, dict):
            task.metadata = {}
        task.metadata["_version"] = int(version)

    def _ensure_task_metadata(self, task: Task) -> Dict[str, Any]:
        if not isinstance(task.metadata, dict):
            task.metadata = {}
        return task.metadata

    def _parse_datetime(self, raw: Any) -> Optional[datetime]:
        if not isinstance(raw, str) or not raw:
            return None
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return None
        if parsed.tzinfo is not None:
            return parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed

    def _synchronize_task(self, target: Task, source: Task) -> Task:
        target.description = source.description
        target.priority = source.priority
        target.assigned_to = source.assigned_to
        target.status = source.status
        target.created_at = source.created_at
        target.completed_at = source.completed_at
        target.result = copy.deepcopy(source.result)
        target.error = source.error
        target.parameters = copy.deepcopy(dict(source.parameters))
        target.dependencies = copy.deepcopy(list(source.dependencies))
        target.metadata = copy.deepcopy(dict(source.metadata))
        return target

    def _extract_idempotency_key(
        self, parameters: Optional[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        nested_metadata = parameters.get("metadata") if isinstance(parameters, dict) else None
        candidates = [
            metadata.get("idempotency_key") if isinstance(metadata, dict) else None,
            parameters.get("idempotency_key") if isinstance(parameters, dict) else None,
            nested_metadata.get("idempotency_key") if isinstance(nested_metadata, dict) else None,
        ]
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return None

    def _find_task_by_idempotency_key(self, idempotency_key: str) -> Optional[Task]:
        task_id = self._idempotency_index.get(idempotency_key)
        if task_id:
            task = self.load_task(task_id)
            if task is not None:
                return task
            self._idempotency_index.pop(idempotency_key, None)

        for task in self.list_persisted_tasks():
            metadata = task.metadata if isinstance(task.metadata, dict) else {}
            if metadata.get("idempotency_key") == idempotency_key:
                self._idempotency_index[idempotency_key] = task.id
                return task
        return None

    def _get_unmet_dependencies(self, task: Task) -> List[str]:
        unmet: List[str] = []
        for dependency_id in list(task.dependencies):
            if not isinstance(dependency_id, str) or not dependency_id.strip():
                continue
            dependency = self.load_task(dependency_id)
            if dependency is None or dependency.status != TaskStatus.COMPLETED:
                unmet.append(dependency_id)
        return unmet

    def add_agent(self, agent: BaseAgent) -> bool:
        with self._lock:
            if agent.id in self.agents:
                logger.warning("Agent %s is already present in system", agent.name)
                return False
            self.agents[agent.id] = agent
            if not self.orchestrator.register_agent(agent):
                self.agents.pop(agent.id, None)
                return False
            self.system_metrics["total_agents"] += 1
            logger.info("Agent %s added to system", agent.name)
            return True

    def remove_agent(self, agent_id: str) -> bool:
        with self._lock:
            if agent_id == self.orchestrator.id:
                logger.warning("Cannot remove the orchestrator agent from the system")
                return False
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                if agent.active_tasks:
                    logger.warning("Cannot remove agent %s while it has active tasks", agent.name)
                    return False
                self.agents.pop(agent_id)
                self.orchestrator.managed_agents.pop(agent_id, None)
                self.system_metrics["total_agents"] -= 1
                logger.info("Agent %s removed from system", agent.name)
                return True
            return False

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        return self.agents.get(agent_id)

    def create_task(
        self,
        description: str,
        parameters: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Task:
        with self._lock:
            safe_parameters = copy.deepcopy(parameters) if isinstance(parameters, dict) else {}
            safe_metadata = copy.deepcopy(metadata) if isinstance(metadata, dict) else {}
            parameter_dependencies = safe_parameters.get("dependencies", [])
            if dependencies is not None:
                safe_dependencies = list(dependencies)
            elif isinstance(parameter_dependencies, list):
                safe_dependencies = list(parameter_dependencies)
            else:
                safe_dependencies = []
            idempotency_key = self._extract_idempotency_key(safe_parameters, safe_metadata)
            if idempotency_key:
                existing = self._find_task_by_idempotency_key(idempotency_key)
                if existing is not None:
                    self.system_metrics["idempotent_hits"] += 1
                    self._emit_event("task_create_deduplicated", existing, {"idempotency_key": idempotency_key})
                    return existing

            task = Task(
                description=description,
                parameters=safe_parameters,
                priority=priority,
                dependencies=safe_dependencies,
                metadata=safe_metadata,
            )
            task_metadata = self._ensure_task_metadata(task)
            task_metadata.setdefault("correlation_id", str(uuid.uuid4()))
            task_metadata.setdefault("attempts", 0)
            task_metadata.setdefault("max_attempts", self.max_retries_per_task)
            if idempotency_key:
                task_metadata["idempotency_key"] = idempotency_key
                self._idempotency_index[idempotency_key] = task.id
            self._set_claim(task, None)
            self._enqueue_if_missing(task)
            self.system_metrics["total_tasks"] += 1
            self.metrics.tasks_created.increment()
            self._store_task(task)
            self._emit_event("task_created", task)
            logger.info("Task %s created: %s", task.id, description)
            return task

    def submit_task(self, task: Task, agent_id: Optional[str] = None) -> bool:
        self.metrics.tasks_submitted.increment()
        with self._lock:
            persisted = self.load_task(task.id)
            if persisted is not None and persisted is not task:
                self._synchronize_task(task, persisted)

            if task.status != TaskStatus.PENDING:
                logger.warning(
                    "Failed to submit task %s because it is in status %s",
                    task.id,
                    task.status.value,
                )
                return False
            unmet_dependencies = self._get_unmet_dependencies(task)
            if unmet_dependencies:
                self.system_metrics["dependency_blocked_tasks"] += 1
                self.metrics.tasks_dependency_blocked.increment()
                self._enqueue_if_missing(task)
                self._emit_event("task_dependency_blocked", task, {"dependencies": unmet_dependencies})
                logger.info("Task %s blocked by unmet dependencies: %s", task.id, unmet_dependencies)
                return False

            assigned = False
            if agent_id:
                agent = self.get_agent(agent_id)
                if agent:
                    assigned = agent.assign_task(task)
            else:
                assigned = self.orchestrator.distribute_task(task)

            if assigned:
                if task.assigned_to and not self._queue_limiter.try_acquire(task.assigned_to):
                    logger.warning("Agent %s queue limit reached; task %s rejected", task.assigned_to, task.id)
                    return False
                self._set_task_status(task, TaskStatus.ASSIGNED, assigned_to=task.assigned_to, claimed_by=task.assigned_to)
                self._dequeue_task(task.id)
                self._emit_event("task_assigned", task)
                logger.info("Task %s submitted successfully", task.id)
                return True

            logger.warning("Failed to submit task %s", task.id)
            self._enqueue_if_missing(task)
            return False

    def execute_task(self, task_id: str, agent_id: str) -> Any:
        limited_agent_id = agent_id
        with self._lock:
            agent = self.get_agent(agent_id)
            if not agent:
                raise KeyError(f"Agent not found: {agent_id}")
            task = agent.active_tasks.get(task_id)
            if not task:
                raise KeyError(f"Task {task_id} is not assigned to agent {agent_id}")
            unmet_dependencies = self._get_unmet_dependencies(task)
            if unmet_dependencies:
                self.system_metrics["dependency_blocked_tasks"] += 1
                self._release_task_from_agent(task.id, agent_id)
                self._set_task_status(task, TaskStatus.PENDING, assigned_to=None, claimed_by=None, completed_at=None)
                self._enqueue_if_missing(task)
                self._emit_event("task_dependency_blocked", task, {"dependencies": unmet_dependencies})
                raise ValueError(f"Task {task.id} blocked by unmet dependencies")
            self._ensure_claimed_by(task, agent_id)
            self._set_task_status(
                task, TaskStatus.RUNNING, assigned_to=agent_id, claimed_by=agent_id
            )
            self._emit_event("task_running", task)

        start_time = datetime.now()
        try:
            with _TRACING.start_span(
                "agent.task.execute",
                attributes={"agent.id": agent_id, "task.id": task.id, "task.description": task.description},
            ):
                with self.performance_tracker.track("agent_task_execution"):
                    result = self._task_execution_breaker.call(agent.run_task, task)
            with self._lock:
                self._set_task_status(
                    task,
                    TaskStatus.COMPLETED,
                    assigned_to=agent_id,
                    claimed_by=agent_id,
                    result=result,
                    error=None,
                    completed_at=datetime.now(),
                )
                self._append_unique_task(self.completed_tasks, task)
                self._update_system_metrics(success=True, start_time=start_time)
                elapsed = (datetime.now() - start_time).total_seconds()
                _METRICS.record_agent_task(success=True, duration_seconds=elapsed)
                self._emit_event("task_completed", task)
            return result
        except Exception as exc:
            with self._lock:
                self._set_task_status(
                    task,
                    TaskStatus.FAILED,
                    assigned_to=agent_id,
                    claimed_by=agent_id,
                    result=task.result,
                    error=str(exc),
                    completed_at=datetime.now(),
                )
                self._append_unique_task(self.failed_tasks, task)
                self._update_system_metrics(success=False, start_time=start_time)
                elapsed = (datetime.now() - start_time).total_seconds()
                _METRICS.record_agent_task(success=False, duration_seconds=elapsed)
                attempts = int(task.metadata.get("attempts", 0)) + 1 if isinstance(task.metadata, dict) else 1
                if isinstance(task.metadata, dict):
                    task.metadata["attempts"] = attempts
                max_attempts = (
                    int(task.metadata.get("max_attempts", self.max_retries_per_task))
                    if isinstance(task.metadata, dict)
                    else self.max_retries_per_task
                )
                if attempts >= max_attempts:
                    self.dead_letter_queue.append(
                        {
                            "task_id": task.id,
                            "failed_at": datetime.now().isoformat(),
                            "error": str(exc),
                            "attempts": attempts,
                        }
                    )
                    self._emit_event("task_dead_lettered", task, {"attempts": attempts})
                self._emit_event("task_failed", task, {"error": str(exc)})
            raise
        finally:
            self._queue_limiter.release(limited_agent_id)

    def cancel_task(self, task_id: str, reason: Optional[str] = None) -> Task:
        with self._lock:
            task = self.load_task(task_id)
            if task is None:
                raise KeyError(f"Task not found: {task_id}")
            if task.status not in {TaskStatus.PENDING, TaskStatus.ASSIGNED}:
                raise ValueError(
                    f"Task {task_id} in status {task.status.value} cannot be cancelled"
                )

            original_assigned_to = task.assigned_to
            self._release_task_from_agent(task.id, original_assigned_to)
            self._set_task_status(
                task,
                TaskStatus.CANCELLED,
                assigned_to=None,
                claimed_by=None,
                result=None,
                error=reason,
                completed_at=datetime.now(),
            )
            self._dequeue_task(task.id)
            self._emit_event("task_cancelled", task, {"reason": reason})
            return task

    def load_task(self, task_id: str) -> Optional[Task]:
        active_task = self._find_active_task(task_id)
        if active_task is not None:
            return active_task
        stored_task = self.task_store.get_task(task_id)
        if stored_task is None:
            return None
        return self._from_stored_task(stored_task)

    def list_persisted_tasks(self, status: Optional[TaskStatus] = None) -> List[Task]:
        stored_tasks = self.task_store.list_tasks(status.name if status else None)
        return [self._from_stored_task(task) for task in stored_tasks]

    def get_completed_task_ids(self) -> Set[str]:
        """Return the set of task IDs that have completed successfully."""
        with self._lock:
            return {t.id for t in self.completed_tasks}

    def recover_incomplete_tasks(self, reset_to: TaskStatus = TaskStatus.PENDING) -> int:
        with self._lock:
            if reset_to != TaskStatus.PENDING:
                raise ValueError("Only reset_to=TaskStatus.PENDING is supported for recovery")

            recovered = 0
            for status in (TaskStatus.ASSIGNED, TaskStatus.RUNNING):
                for task in self.list_persisted_tasks(status):
                    active_task = self._find_active_task(task.id)
                    working_task = active_task or task
                    original_assigned_to = working_task.assigned_to
                    self._release_task_from_agent(working_task.id, original_assigned_to)
                    working_task.result = None
                    working_task.error = None
                    working_task.completed_at = None
                    self._set_task_status(
                        working_task, TaskStatus.PENDING, assigned_to=None, claimed_by=None
                    )
                    self._enqueue_if_missing(working_task)
                    self._emit_event("task_recovered", working_task)
                    recovered += 1

            logger.info("Recovered %s incomplete tasks", recovered)
            return recovered

    def requeue_task(self, task_id: str) -> Task:
        with self._lock:
            task = self.load_task(task_id)
            if task is None:
                raise KeyError(f"Task not found: {task_id}")
            if not self._is_recoverable_status(task.status):
                raise ValueError(f"Task {task_id} in status {task.status.value} cannot be requeued")

            original_assigned_to = task.assigned_to
            self._release_task_from_agent(task.id, original_assigned_to)
            self._set_task_status(
                task,
                TaskStatus.PENDING,
                assigned_to=None,
                claimed_by=None,
                result=None,
                error=None,
                completed_at=None,
            )
            self._enqueue_if_missing(task)
            self._emit_event("task_requeued_manual", task)
            return task

    def _to_stored_task(self, task: Task) -> StoredTask:
        return StoredTask(
            id=task.id,
            description=task.description,
            priority=task.priority.name,
            assigned_to=task.assigned_to,
            status=task.status.name,
            created_at=task.created_at,
            completed_at=task.completed_at,
            result=_make_json_safe(task.result),
            error=task.error,
            parameters=_make_json_safe(task.parameters),
            dependencies=list(task.dependencies),
            metadata=_make_json_safe(task.metadata),
        )

    def _from_stored_task(self, stored_task: StoredTask) -> Task:
        return Task(
            id=stored_task.id,
            description=stored_task.description,
            priority=TaskPriority[stored_task.priority],
            assigned_to=stored_task.assigned_to,
            status=TaskStatus[stored_task.status],
            created_at=stored_task.created_at,
            completed_at=stored_task.completed_at,
            result=copy.deepcopy(stored_task.result),
            error=stored_task.error,
            parameters=copy.deepcopy(dict(stored_task.parameters)),
            dependencies=copy.deepcopy(list(stored_task.dependencies)),
            metadata=copy.deepcopy(dict(stored_task.metadata)),
        )

    def _store_task(self, task: Task) -> None:
        if self._get_task_version(task) <= 0:
            self._set_task_version(task, 1)
        self.task_store.create_task(self._to_stored_task(task))
        self._task_versions[task.id] = self._get_task_version(task)

    def _update_task_record(self, task: Task) -> None:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self._max_persist_retries + 1):
            try:
                stored = self.task_store.get_task(task.id)
                local_ver = self._get_task_version(task)
                if stored is None:
                    if local_ver <= 0:
                        self._set_task_version(task, 1)
                    self.task_store.create_task(self._to_stored_task(task))
                    self._task_versions[task.id] = self._get_task_version(task)
                    return
                remote_task = self._from_stored_task(stored)
                remote_ver = self._get_task_version(remote_task)
                if local_ver < remote_ver:
                    self.system_metrics["persistence_conflicts"] += 1
                    raise RuntimeError(
                        f"Stale task update detected for {task.id}: local={local_ver}, remote={remote_ver}"
                    )
                self._set_task_version(task, max(local_ver, remote_ver) + 1)
                self.task_store.update_task(self._to_stored_task(task))
                self._task_versions[task.id] = self._get_task_version(task)
                return
            except Exception as exc:
                last_exc = exc
                self.system_metrics["persistence_retry_attempts"] += 1
                self.system_metrics["persistence_retries"] += 1
                self._emit_event("task_persistence_retry", task, {"attempt": attempt, "error": str(exc)})
                if attempt >= self._max_persist_retries:
                    break
                exponent = min(attempt - 1, self.persistence_backoff_max_exponent)
                base_delay = min(
                    self.persistence_backoff_max_seconds,
                    self.persistence_backoff_min_seconds * (2**exponent),
                )
                jitter_factor = (self._retry_random.random() - 0.5) * 2
                jitter = base_delay * self.persistence_backoff_jitter_ratio * jitter_factor
                time.sleep(min(self.persistence_backoff_max_seconds, max(0.0, base_delay + jitter)))

        self.system_metrics["persistence_failures"] += 1
        self._emit_event("task_persistence_terminal_failure", task, {"error": str(last_exc) if last_exc else None})
        logger.exception(
            "Failed to persist task update after retries for task_id=%s assigned_to=%s status=%s",
            task.id,
            task.assigned_to,
            task.status.name,
        )
        raise RuntimeError(f"Task persistence failed for {task.id}") from last_exc

    def _validate_transition(self, current: TaskStatus, target: TaskStatus) -> None:
        allowed = self._ALLOWED_TRANSITIONS.get(current, set())
        if target not in allowed and current != target:
            raise ValueError(f"Illegal task transition: {current.value} -> {target.value}")

    def _set_task_status(
        self,
        task: Task,
        status: TaskStatus,
        *,
        assigned_to: Optional[str] = None,
        claimed_by: Optional[str] = None,
        result: Any = None,
        error: Optional[str] = None,
        completed_at: Optional[datetime] = None,
    ) -> None:
        self._validate_transition(task.status, status)
        task.status = status
        task.assigned_to = assigned_to
        task.result = result
        task.error = error
        task.completed_at = completed_at
        self._set_claim(task, claimed_by)
        self._update_task_record(task)

    def _set_claim(self, task: Task, claimed_by: Optional[str]) -> None:
        metadata = self._ensure_task_metadata(task)
        now = datetime.now()
        if claimed_by is None:
            metadata.pop("claimed_by", None)
            metadata.pop("claim_token", None)
            metadata.pop("claim_expires_at", None)
            metadata.pop("claim_heartbeat_at", None)
        else:
            current_claimed_by = metadata.get("claimed_by")
            existing_token = metadata.get("claim_token") if current_claimed_by == claimed_by else None
            expires_at = now + timedelta(seconds=self.claim_ttl_seconds)
            metadata["claimed_by"] = claimed_by
            metadata["claim_token"] = existing_token if isinstance(existing_token, str) and existing_token else str(uuid.uuid4())
            metadata["claim_heartbeat_at"] = now.isoformat()
            metadata["claim_expires_at"] = expires_at.isoformat()

    def _ensure_claimed_by(self, task: Task, agent_id: str) -> None:
        metadata = self._ensure_task_metadata(task)
        claimed_by = metadata.get("claimed_by")
        expires_at = self._parse_datetime(metadata.get("claim_expires_at"))
        if claimed_by and expires_at and expires_at <= datetime.now():
            self.system_metrics["claim_validation_failures"] += 1
            raise ValueError(f"Task {task.id} claim expired for {claimed_by}")
        if claimed_by and claimed_by != agent_id:
            self.system_metrics["claim_validation_failures"] += 1
            raise ValueError(f"Task {task.id} is claimed by {claimed_by}, not {agent_id}")

    def _is_recoverable_status(self, status: TaskStatus) -> bool:
        return status in {TaskStatus.ASSIGNED, TaskStatus.RUNNING, TaskStatus.FAILED}

    def _find_active_task(self, task_id: str) -> Optional[Task]:
        for agent in self.agents.values():
            task = agent.active_tasks.get(task_id)
            if task is not None:
                return task
        return None

    def _release_task_from_agent(self, task_id: str, agent_id: Optional[str]) -> None:
        if not agent_id:
            return
        agent = self.get_agent(agent_id)
        if agent is not None:
            agent.release_task(task_id)

    def _enqueue_if_missing(self, task: Task) -> None:
        if task.id in self._task_index:
            return
        if len(self._task_index) >= self.max_queue_size:
            self.system_metrics["queue_overflow_drops"] += 1
            raise OverflowError(f"Task queue full ({self.max_queue_size})")
        entry = QueuedTask(
            priority_rank=-int(task.priority.value),
            created_at_ts=task.created_at.timestamp(),
            sequence=self._queue_sequence,
            id=task.id,
        )
        self._queue_sequence += 1
        heapq.heappush(self.global_task_queue, entry)
        self._task_index.add(task.id)

    def _dequeue_task(self, task_id: str) -> None:
        self._task_index.discard(task_id)
        if not self.global_task_queue:
            return
        retained = [queued for queued in self.global_task_queue if queued.id != task_id]
        if len(retained) != len(self.global_task_queue):
            self.global_task_queue = retained
            heapq.heapify(self.global_task_queue)

    def _pop_next_valid_task_id(self) -> Optional[str]:
        while self.global_task_queue:
            queued = heapq.heappop(self.global_task_queue)
            if queued.id not in self._task_index:
                self.system_metrics["queue_stale_pops"] += 1
                self.system_metrics["queue_stale_entries_pruned"] += 1
                continue
            task = self.load_task(queued.id)
            if task is None or task.status != TaskStatus.PENDING:
                self._task_index.discard(queued.id)
                self.system_metrics["queue_stale_pops"] += 1
                self.system_metrics["queue_stale_entries_pruned"] += 1
                continue
            self._task_index.discard(queued.id)
            return queued.id
        return None

    def renew_task_claim(self, task_id: str, agent_id: Optional[str] = None) -> bool:
        """Refresh the lease heartbeat for an assigned or running task."""
        with self._lock:
            task = self._find_active_task(task_id) or self.load_task(task_id)
            if task is None or task.status not in {TaskStatus.ASSIGNED, TaskStatus.RUNNING}:
                self.system_metrics["claim_renew_failures"] += 1
                return False
            metadata = self._ensure_task_metadata(task)
            claimed_by = metadata.get("claimed_by")
            if not isinstance(claimed_by, str) or not claimed_by:
                self.system_metrics["claim_renew_failures"] += 1
                return False
            if agent_id is not None and claimed_by != agent_id:
                self.system_metrics["claim_renew_failures"] += 1
                return False
            expires_at = self._parse_datetime(metadata.get("claim_expires_at"))
            if expires_at is not None and expires_at + timedelta(seconds=self.claim_grace_seconds) <= datetime.now():
                self.system_metrics["claim_renew_failures"] += 1
                return False
            self._set_claim(task, claimed_by)
            self._update_task_record(task)
            self._emit_event("task_claim_renewed", task)
            self.system_metrics["claim_renew_success"] += 1
            return True

    def _renew_active_claims(self) -> int:
        renewed = 0
        with self._lock:
            now = datetime.now()
            for agent in self.agents.values():
                for task in agent.active_tasks.values():
                    if task.status not in {TaskStatus.ASSIGNED, TaskStatus.RUNNING}:
                        continue
                    metadata = self._ensure_task_metadata(task)
                    claimed_by = metadata.get("claimed_by")
                    expires_at = self._parse_datetime(metadata.get("claim_expires_at"))
                    if not isinstance(claimed_by, str) or not claimed_by:
                        continue
                    if expires_at is not None and expires_at + timedelta(seconds=self.claim_grace_seconds) <= now:
                        continue
                    self._set_claim(task, claimed_by)
                    self._update_task_record(task)
                    self._emit_event("task_claim_renewed", task)
                    renewed += 1
        return renewed

    def reclaim_expired_claims(self) -> int:
        """Reclaim expired task leases and return tasks to the pending queue."""
        with self._lock:
            reclaimed = 0
            candidates: Dict[str, Task] = {}
            for status in (TaskStatus.ASSIGNED, TaskStatus.RUNNING):
                for task in self.list_persisted_tasks(status):
                    candidates[task.id] = task
            for agent in self.agents.values():
                for task in agent.active_tasks.values():
                    candidates[task.id] = task

            now = datetime.now()
            for task in candidates.values():
                metadata = self._ensure_task_metadata(task)
                claimed_by = metadata.get("claimed_by")
                expires_at = self._parse_datetime(metadata.get("claim_expires_at"))
                if not isinstance(claimed_by, str) or expires_at is None:
                    continue
                if expires_at + timedelta(seconds=self.claim_grace_seconds) > now:
                    continue
                self._release_task_from_agent(task.id, task.assigned_to)
                self._set_task_status(
                    task,
                    TaskStatus.PENDING,
                    assigned_to=None,
                    claimed_by=None,
                    result=None,
                    error="claim expired",
                    completed_at=None,
                )
                self._enqueue_if_missing(task)
                self.system_metrics["claim_reclaims"] += 1
                reclaimed += 1
                self._emit_event("task_claim_reclaimed", task, {"previous_claimed_by": claimed_by})
            return reclaimed

    def process_next_pending_task(self) -> bool:
        """Assign and execute the next valid pending task from the queue."""
        with self._lock:
            task_id = self._pop_next_valid_task_id()
            if task_id is None:
                return False
            task = self.load_task(task_id)
            if task is None or task.status != TaskStatus.PENDING:
                return False

        if not self.submit_task(task):
            return False

        assigned_to = task.assigned_to
        if not assigned_to:
            return False
        self.execute_task(task.id, assigned_to)
        return True

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            if not self._pause_event.is_set():
                self._pause_event.wait(self.worker_poll_interval_seconds)
                continue
            try:
                processed = self.process_next_pending_task()
            except Exception:
                logger.exception("Worker loop failed while processing task")
                processed = False
            if not processed:
                self._stop_event.wait(self.worker_poll_interval_seconds)

    def _maintenance_loop(self) -> None:
        while not self._stop_event.wait(self.claim_sweep_interval_seconds):
            try:
                self._renew_active_claims()
                self.reclaim_expired_claims()
            except Exception:
                logger.exception("Maintenance loop failed")

    def start_workers(self, worker_count: int = 1) -> None:
        """Start background workers and lease maintenance threads."""
        if worker_count <= 0:
            raise ValueError("worker_count must be greater than 0")
        with self._lock:
            self._stop_event.clear()
            self._pause_event.set()
            if self._maintenance_thread is None or not self._maintenance_thread.is_alive():
                self._maintenance_thread = threading.Thread(
                    target=self._maintenance_loop,
                    name=f"{self.name}-maintenance",
                    daemon=True,
                )
                self._maintenance_thread.start()
            alive_workers = [thread for thread in self._worker_threads if thread.is_alive()]
            self._worker_threads = alive_workers
            while len(self._worker_threads) < worker_count:
                worker = threading.Thread(
                    target=self._worker_loop,
                    name=f"{self.name}-worker-{len(self._worker_threads) + 1}",
                    daemon=True,
                )
                worker.start()
                self._worker_threads.append(worker)

    def stop_workers(self, timeout_seconds: Optional[float] = None) -> None:
        """Request worker shutdown and optionally wait for threads to stop."""
        self._stop_event.set()
        self._pause_event.set()
        initial_timeout = None if timeout_seconds is None else max(timeout_seconds, 0.0)
        if self._maintenance_thread is not None:
            self._maintenance_thread.join(initial_timeout)
            if not self._maintenance_thread.is_alive():
                self._maintenance_thread = None
        remaining = initial_timeout
        for worker in list(self._worker_threads):
            start = time.monotonic()
            worker.join(remaining)
            if remaining is not None:
                remaining = max(0.0, remaining - (time.monotonic() - start))
        self._worker_threads = [thread for thread in self._worker_threads if thread.is_alive()]

    def pause_processing(self) -> None:
        """Pause worker task consumption without stopping maintenance."""
        self._pause_event.clear()

    def resume_processing(self) -> None:
        """Resume worker task consumption after a pause."""
        self._pause_event.set()

    def drain_and_shutdown(self, timeout_seconds: float = 5.0) -> bool:
        """Stop workers, wait for active tasks to settle, and persist current state."""
        timeout_seconds = max(timeout_seconds, 0.0)
        deadline = time.monotonic() + timeout_seconds
        self.stop_workers(timeout_seconds=timeout_seconds)
        while time.monotonic() < deadline:
            with self._lock:
                if not any(agent.active_tasks for agent in self.agents.values()):
                    break
            time.sleep(0.01)
        with self._lock:
            return not any(agent.active_tasks for agent in self.agents.values())

    def _append_unique_task(self, collection: List[Task], task: Task) -> None:
        if not any(existing.id == task.id for existing in collection):
            collection.append(task)

    def _update_system_metrics(self, success: bool, start_time: datetime) -> None:
        elapsed = (datetime.now() - start_time).total_seconds()
        key = "successful_tasks" if success else "failed_tasks"
        self.system_metrics[key] += 1
        self.system_metrics["completed_tasks_total"] += 1
        total = self.system_metrics["completed_tasks_total"]
        overall_prev = self.system_metrics["avg_task_duration_overall"]
        self.system_metrics["avg_task_duration_overall"] = overall_prev + (
            (elapsed - overall_prev) / total
        )
        if success:
            n = self.system_metrics["successful_tasks"]
            prev = self.system_metrics["avg_task_duration_success"]
            self.system_metrics["avg_task_duration_success"] = prev + ((elapsed - prev) / n)
            self.metrics.tasks_completed.increment()
        else:
            n = self.system_metrics["failed_tasks"]
            prev = self.system_metrics["avg_task_duration_failure"]
            self.system_metrics["avg_task_duration_failure"] = prev + ((elapsed - prev) / n)
            self.metrics.tasks_failed.increment()
        self.metrics.task_duration.record(elapsed)

    def get_system_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "system_name": self.name,
                "system_id": self.id,
                "created_at": self.created_at.isoformat(),
                "agents": {aid: agent.get_status() for aid, agent in self.agents.items()},
                "metrics": _make_json_safe(self.system_metrics),
                "structured_metrics": self.metrics.to_dict(),
                "pending_tasks": len(self._task_index),
                "completed_tasks": len(self.completed_tasks),
                "failed_tasks": len(self.failed_tasks),
                "dead_letter_tasks": len(self.dead_letter_queue),
                "processing_paused": not self._pause_event.is_set(),
                "worker_threads": len([thread for thread in self._worker_threads if thread.is_alive()]),
            }

    def get_observability_snapshot(self) -> Dict[str, Any]:
        perf_snapshot = self.performance_tracker.snapshot()
        resource_snapshot = self.performance_tracker.resource_snapshot()
        _METRICS.record_task_system(
            queue_depth=len(self._task_index),
            processing_time=float(self.system_metrics.get("avg_task_duration_overall", 0.0)),
            throughput=float(self.system_metrics.get("completed_tasks_total", 0.0)),
        )
        _METRICS.record_resources(
            memory_usage=resource_snapshot.get("memory_rss_bytes", 0.0),
            cpu_usage=resource_snapshot.get("cpu_percent", 0.0),
            model_size=0.0,
        )
        metric_snapshot = _METRICS.snapshot()
        return {
            "metrics": _make_json_safe(self.system_metrics),
            "prometheus_metrics": metric_snapshot,
            "recent_events": self.event_log[-200:],
            "queue_depth": len(self._task_index),
            "dead_letter_depth": len(self.dead_letter_queue),
            "worker_threads": len([thread for thread in self._worker_threads if thread.is_alive()]),
            "processing_paused": not self._pause_event.is_set(),
            "health": self.health_checker.run(),
            "performance": perf_snapshot,
            "resource_usage": resource_snapshot,
            "alerts": self.threshold_monitor.evaluate(metrics=metric_snapshot),
        }

    def to_json(self) -> str:
        return json.dumps(self.get_system_status(), indent=2, default=str)

    # ------------------------------------------------------------------
    # Persistence: JSON snapshot save / load (Phase 1 hardening)
    # ------------------------------------------------------------------

    def save_snapshot(self, filepath: str) -> None:
        """Persist AgentSystem state to a JSON file.

        Saves identity, metrics, task queue, and completed task history in a
        format that can be restored by :meth:`load_snapshot`.  Enums and
        datetimes are serialised to their string representations.

        Parameters
        ----------
        filepath:
            Destination file path.  Parent directories are created if missing.
        """
        if not filepath or not isinstance(filepath, str):
            raise ValueError("filepath must be a non-empty string")

        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with self._lock:
            completed = [t.to_dict() for t in self.completed_tasks]
            failed = [t.to_dict() for t in self.failed_tasks]
            queued = [
                self.load_task(qt.id).to_dict()
                for qt in self.global_task_queue
                if self.load_task(qt.id) is not None
            ]
            snapshot = {
                "schema_version": 1,
                "name": self.name,
                "id": self.id,
                "created_at": self.created_at.isoformat(),
                "system_metrics": _make_json_safe(dict(self.system_metrics)),
                "structured_metrics": self.metrics.to_dict(),
                "global_task_queue": queued,
                "completed_tasks": completed,
                "failed_tasks": failed,
            }

        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(snapshot, fh, indent=2, default=str)

        logger.info(
            "AgentSystem snapshot saved to %s",
            filepath,
            extra={"system_id": self.id, "filepath": filepath},
        )

    @classmethod
    def load_snapshot(cls, filepath: str) -> "AgentSystem":
        """Restore a lightweight AgentSystem from a JSON snapshot.

        Restores metrics, task history summary, and queue sizes so that
        operational dashboards and logging are meaningful immediately after
        restart.  Live agent objects cannot be fully reconstructed from JSON
        alone; concrete agent instances must be re-registered after loading.

        Parameters
        ----------
        filepath:
            Path to a snapshot file previously created by :meth:`save_snapshot`.
        """
        if not filepath or not isinstance(filepath, str):
            raise ValueError("filepath must be a non-empty string")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"snapshot file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        system = cls(name=data.get("name", "Ai-morphasis"))
        system.id = data.get("id", system.id)
        created_at_raw = data.get("created_at")
        if created_at_raw:
            try:
                system.created_at = datetime.fromisoformat(created_at_raw)
            except ValueError:
                pass

        # Restore legacy system_metrics dict (best-effort; unknown keys ignored)
        for key, value in data.get("system_metrics", {}).items():
            if key in system.system_metrics:
                system.system_metrics[key] = value

        # Restore completed task history
        for task_data in data.get("completed_tasks", []):
            task = _task_from_dict(task_data)
            system.completed_tasks.append(task)

        # Restore failed task history
        for task_data in data.get("failed_tasks", []):
            task = _task_from_dict(task_data)
            system.failed_tasks.append(task)

        logger.info(
            "AgentSystem snapshot loaded from %s (id=%s, completed=%d, queued_hints=%d)",
            filepath,
            system.id,
            len(system.completed_tasks),
            len(data.get("global_task_queue", [])),
        )
        return system

    def __repr__(self) -> str:
        return f"<AgentSystem: {self.name} ({len(self.agents)} agents)>"


class AgentFactory:
    _agent_templates = {
        "executor": ExecutorAgent,
        "analyzer": AnalyzerAgent,
        "learner": LearnerAgent,
        "orchestrator": OrchestratorAgent,
    }

    @classmethod
    def create_agent(cls, agent_type: str, name: str) -> Optional[BaseAgent]:
        agent_class = cls._agent_templates.get(agent_type.lower())
        if agent_class:
            return agent_class(name)
        logger.error("Unknown agent type: %s", agent_type)
        return None

    @classmethod
    def create_team(
        cls, team_config: Dict[str, int], task_store: Optional[TaskStore] = None
    ) -> AgentSystem:
        system = AgentSystem("Ai-morphasis-Team", task_store=task_store)
        for agent_type, count in team_config.items():
            if count < 0:
                raise ValueError(f"Agent count cannot be negative for type: {agent_type}")
            for i in range(count):
                agent = cls.create_agent(agent_type, f"{agent_type.title()}-{i + 1}")
                if agent:
                    system.add_agent(agent)
        logger.info("Agent team created with config: %s", team_config)
        return system


def _make_json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_make_json_safe(item) for item in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def example_usage() -> None:
    system = AgentSystem("Ai-morphasis-2.0")
    executor = ExecutorAgent("TaskExecutor-1")
    analyzer = AnalyzerAgent("DataAnalyzer-1")
    learner = LearnerAgent("SystemLearner-1")

    system.add_agent(executor)
    system.add_agent(analyzer)
    system.add_agent(learner)

    executor.register_capability(
        AgentCapability(
            name="file_processing",
            description="Process and manipulate files",
            confidence_score=0.95,
        )
    )

    analyzer.register_capability(
        AgentCapability(
            name="data_analysis",
            description="Analyze data and generate insights",
            confidence_score=0.88,
        )
    )

    task1 = system.create_task(
        description="Analyze performance metrics",
        parameters={"metric_type": "performance", "duration": "24h"},
    )

    system.submit_task(task1, executor.id)
    system.execute_task(task1.id, executor.id)

    print("\n" + "=" * 60)
    print("AGENT SYSTEM STATUS")
    print("=" * 60)
    print(system.to_json())


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    example_usage()
