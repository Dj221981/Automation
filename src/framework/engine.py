"""
Execution Engine
================

A thread-safe, lifecycle-managed execution engine that bridges the raw
:class:`~src.agents.super_agentic_agents.AgentSystem` machinery and the
higher-level framework abstractions.

The engine adds:

* A clean *start / stop / pause / resume* lifecycle independent of the agent
  system's worker threads.
* **Engine hooks** that let external code observe task events (started,
  completed, failed) without modifying the engine internals.
* A simple **task submission API** that wraps
  :class:`~src.agents.super_agentic_agents.Task` creation and tracking.
* Per-engine metrics (submitted, completed, failed, durations).

Usage::

    from src.framework.engine import ExecutionEngine, EngineConfig, EngineHook
    from src.agents.super_agentic_agents import AgentSystem, ExecutorAgent

    system = AgentSystem("demo")
    agent  = ExecutorAgent("worker")
    system.add_agent(agent)

    engine = ExecutionEngine(system, config=EngineConfig(default_agent_id=agent.id))
    engine.start()

    task_id = engine.submit("do something", parameters={"x": 1})
    result  = engine.run_task(task_id)
    engine.stop()
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from src.agents.super_agentic_agents import (
    AgentSystem,
    Task,
    TaskPriority,
    TaskStatus,
)

__all__ = [
    "EngineState",
    "EngineConfig",
    "EngineHook",
    "TaskRecord",
    "ExecutionEngine",
]


# ---------------------------------------------------------------------------
# Engine lifecycle state
# ---------------------------------------------------------------------------


class EngineState(Enum):
    """Lifecycle states for the execution engine."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EngineConfig:
    """Tunable parameters for :class:`ExecutionEngine`."""

    #: Agent ID to use when no explicit agent is supplied to :meth:`submit`.
    default_agent_id: Optional[str] = None
    #: Default task priority.
    default_priority: TaskPriority = TaskPriority.NORMAL
    #: Hard cap on concurrent in-flight tasks managed by this engine.
    max_concurrent_tasks: int = 100


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


@dataclass
class EngineHook:
    """Callbacks for engine lifecycle events.

    All callbacks are optional.  Each receives a :class:`TaskRecord` and
    may raise freely – the engine logs but does not propagate exceptions from
    hooks so that misbehaving hooks never crash the engine.
    """

    on_submitted: Optional[Callable[["TaskRecord"], None]] = None
    on_started: Optional[Callable[["TaskRecord"], None]] = None
    on_completed: Optional[Callable[["TaskRecord"], None]] = None
    on_failed: Optional[Callable[["TaskRecord"], None]] = None
    on_cancelled: Optional[Callable[["TaskRecord"], None]] = None


# ---------------------------------------------------------------------------
# Task record
# ---------------------------------------------------------------------------


@dataclass
class TaskRecord:
    """Lightweight tracking record for a task managed by the engine."""

    task_id: str
    description: str
    agent_id: str
    submitted_at: float = field(default_factory=time.monotonic)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.started_at is not None and self.finished_at is not None:
            return max(self.finished_at - self.started_at, 0.0)
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "agent_id": self.agent_id,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ExecutionEngine:
    """Wraps an :class:`~src.agents.super_agentic_agents.AgentSystem` with a
    clean lifecycle API, hook system, and per-engine metrics.

    The engine does **not** own the ``AgentSystem``; it is purely an
    orchestration façade.  Stopping the engine does not shut down the
    underlying system's worker threads.
    """

    def __init__(
        self,
        system: AgentSystem,
        config: Optional[EngineConfig] = None,
        hooks: Optional[List[EngineHook]] = None,
    ) -> None:
        if not isinstance(system, AgentSystem):
            raise TypeError("system must be an AgentSystem instance")
        self._system = system
        self._config = config or EngineConfig()
        self._hooks: List[EngineHook] = list(hooks or [])
        self._state = EngineState.IDLE
        self._lock = threading.RLock()
        self._records: Dict[str, TaskRecord] = {}
        # Metrics
        self._submitted = 0
        self._completed = 0
        self._failed = 0
        self._total_duration: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Transition the engine from *IDLE* or *PAUSED* to *RUNNING*."""
        with self._lock:
            if self._state == EngineState.STOPPED:
                raise RuntimeError("Cannot start a stopped engine; create a new instance.")
            self._state = EngineState.RUNNING

    def pause(self) -> None:
        """Suspend task submission.  In-flight tasks continue to completion."""
        with self._lock:
            if self._state == EngineState.RUNNING:
                self._state = EngineState.PAUSED

    def resume(self) -> None:
        """Resume a paused engine."""
        with self._lock:
            if self._state == EngineState.PAUSED:
                self._state = EngineState.RUNNING

    def stop(self) -> None:
        """Permanently stop the engine.  No new tasks may be submitted."""
        with self._lock:
            self._state = EngineState.STOPPED

    @property
    def state(self) -> EngineState:
        with self._lock:
            return self._state

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def add_hook(self, hook: EngineHook) -> None:
        """Register an additional :class:`EngineHook`."""
        with self._lock:
            self._hooks.append(hook)

    def _fire(self, event: str, record: TaskRecord) -> None:
        """Invoke all registered hooks for *event* (non-blocking, fault-tolerant)."""
        with self._lock:
            hooks = list(self._hooks)
        for h in hooks:
            cb = getattr(h, event, None)
            if cb is not None:
                try:
                    cb(record)
                except Exception:
                    pass  # hooks must not crash the engine

    # ------------------------------------------------------------------
    # Task submission
    # ------------------------------------------------------------------

    def submit(
        self,
        description: str,
        *,
        parameters: Optional[Dict[str, Any]] = None,
        priority: Optional[TaskPriority] = None,
        agent_id: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
    ) -> str:
        """Create and enqueue a task; return its task ID.

        Raises :exc:`RuntimeError` if the engine is paused or stopped.
        """
        with self._lock:
            if self._state != EngineState.RUNNING:
                raise RuntimeError(
                    f"Cannot submit tasks while engine is in state {self._state.value!r}. "
                    "Call engine.start() first."
                )
            if self._submitted >= self._config.max_concurrent_tasks + self._completed + self._failed:
                in_flight = self._submitted - self._completed - self._failed
                if in_flight >= self._config.max_concurrent_tasks:
                    raise RuntimeError(
                        f"Engine has reached the max_concurrent_tasks limit "
                        f"({self._config.max_concurrent_tasks})."
                    )

        effective_agent_id = agent_id or self._config.default_agent_id
        if effective_agent_id is None:
            # Pick the first available non-orchestrator agent
            with self._lock:
                candidates = [
                    aid for aid, ag in self._system.agents.items()
                    if ag.role.value != "orchestrator"
                ]
            effective_agent_id = candidates[0] if candidates else next(iter(self._system.agents))

        task = self._system.create_task(
            description=description,
            parameters=parameters or {},
            priority=priority or self._config.default_priority,
            dependencies=dependencies or [],
        )

        self._system.submit_task(task, effective_agent_id)

        record = TaskRecord(
            task_id=task.id,
            description=task.description,
            agent_id=effective_agent_id,
        )
        with self._lock:
            self._records[task.id] = record
            self._submitted += 1

        self._fire("on_submitted", record)
        return task.id

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run_task(self, task_id: str) -> TaskRecord:
        """Execute *task_id* synchronously and return the updated record.

        Delegates directly to :meth:`AgentSystem.execute_task`, which is a
        blocking call that either completes or fails the task before returning.
        """
        with self._lock:
            record = self._records.get(task_id)
        if record is None:
            raise KeyError(f"No task with id {task_id!r} tracked by this engine.")

        agent_id = record.agent_id
        record.started_at = time.monotonic()
        record.status = TaskStatus.RUNNING
        self._fire("on_started", record)

        final_status = TaskStatus.FAILED
        result = None
        error_msg: Optional[str] = None

        try:
            result = self._system.execute_task(task_id, agent_id)
            final_status = TaskStatus.COMPLETED
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            final_status = TaskStatus.FAILED

        record.finished_at = time.monotonic()
        record.status = final_status
        record.result = result
        record.error = error_msg

        with self._lock:
            if final_status == TaskStatus.COMPLETED:
                self._completed += 1
                if record.duration_seconds is not None:
                    self._total_duration += record.duration_seconds
            else:
                self._failed += 1

        event = "on_completed" if final_status == TaskStatus.COMPLETED else "on_failed"
        self._fire(event, record)
        return record

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def get_record(self, task_id: str) -> Optional[TaskRecord]:
        """Return the :class:`TaskRecord` for a submitted task, or *None*."""
        with self._lock:
            return self._records.get(task_id)

    def list_records(self) -> List[TaskRecord]:
        """Return all tracked task records."""
        with self._lock:
            return list(self._records.values())

    def metrics(self) -> Dict[str, Any]:
        """Return aggregate metrics for the engine."""
        with self._lock:
            in_flight = self._submitted - self._completed - self._failed
            avg_duration = (
                self._total_duration / self._completed if self._completed else 0.0
            )
            return {
                "state": self._state.value,
                "submitted": self._submitted,
                "completed": self._completed,
                "failed": self._failed,
                "in_flight": max(in_flight, 0),
                "avg_duration_seconds": round(avg_duration, 6),
            }

    def __repr__(self) -> str:
        m = self.metrics()
        return (
            f"<ExecutionEngine state={m['state']} "
            f"submitted={m['submitted']} completed={m['completed']} "
            f"failed={m['failed']}>"
        )
