"""
Super Agentic Agents Framework
===============================

A sophisticated multi-agent system architecture for Ai-morphasis 2.0.
This module provides the core infrastructure for creating, managing,
and orchestrating intelligent agentic agents with evolved capabilities.

Features:
    - Hierarchical agent architecture
    - Agent memory and state management
    - Inter-agent communication
    - Distributed task execution
    - Dynamic capability evolution
    - Agent reasoning and decision-making
"""

from __future__ import annotations

import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from threading import RLock
from time import sleep
from typing import Any, Callable, Dict, List, Optional, Set

from .task_store import InMemoryTaskStore, StoredTask, TaskStore

logger = logging.getLogger(__name__)


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


class TaskPriority(Enum):
    """Defines task execution priority levels."""

    CRITICAL = 5
    HIGH = 4
    NORMAL = 3
    LOW = 2
    DEFERRED = 1


ALLOWED_TASK_TRANSITIONS: Dict[TaskStatus, Set[TaskStatus]] = {
    TaskStatus.PENDING: {TaskStatus.ASSIGNED, TaskStatus.CANCELLED},
    TaskStatus.ASSIGNED: {TaskStatus.RUNNING, TaskStatus.PENDING, TaskStatus.CANCELLED},
    TaskStatus.RUNNING: {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.PENDING},
    TaskStatus.FAILED: {TaskStatus.PENDING},
    TaskStatus.COMPLETED: set(),
    TaskStatus.CANCELLED: set(),
}


@dataclass(slots=True)
class AgentCapability:
    """Represents a capability an agent can perform."""

    name: str
    description: str
    func: Optional[Callable[..., Any]] = None
    confidence_score: float = 1.0
    requires_resources: List[str] = field(default_factory=list)
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Capability name cannot be empty")
        if not self.description.strip():
            raise ValueError("Capability description cannot be empty")
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("Capability confidence_score must be between 0.0 and 1.0")

    def __repr__(self) -> str:
        return f"<Capability: {self.name} v{self.version} ({self.confidence_score:.2%})>"


@dataclass(slots=True)
class AgentMemory:
    """Represents agent memory with episodic and semantic storage."""

    agent_id: str
    episodic_memory: Dict[str, Any] = field(default_factory=dict)
    semantic_memory: Dict[str, Any] = field(default_factory=dict)
    procedural_memory: Dict[str, Callable[..., Any]] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    max_episodes: int = 1000
    _lock: RLock = field(default_factory=RLock, repr=False)

    def store_episode(self, key: str, value: Any) -> None:
        with self._lock:
            if len(self.episodic_memory) >= self.max_episodes:
                oldest_key = next(iter(self.episodic_memory))
                del self.episodic_memory[oldest_key]
            self.episodic_memory[key] = {"value": value, "timestamp": datetime.now()}
            self.last_accessed = datetime.now()

    def store_semantic(self, key: str, value: Any) -> None:
        with self._lock:
            self.semantic_memory[key] = {
                "value": value,
                "timestamp": datetime.now(),
                "access_count": 0,
            }
            self.last_accessed = datetime.now()

    def retrieve(self, key: str, memory_type: str = "auto") -> Optional[Any]:
        with self._lock:
            if memory_type in ("auto", "episodic") and key in self.episodic_memory:
                self.last_accessed = datetime.now()
                return self.episodic_memory[key]["value"]
            if memory_type in ("auto", "semantic") and key in self.semantic_memory:
                self.semantic_memory[key]["access_count"] += 1
                self.last_accessed = datetime.now()
                return self.semantic_memory[key]["value"]
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
        self.parameters = dict(self.parameters) if isinstance(self.parameters, dict) else {}
        if isinstance(self.dependencies, (list, tuple, set)):
            self.dependencies = list(self.dependencies)
        else:
            self.dependencies = []
        self.metadata = dict(self.metadata) if isinstance(self.metadata, dict) else {}

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


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(self, name: str, role: AgentRole = AgentRole.EXECUTOR, max_capabilities: int = 50):
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
        self._lock = RLock()
        self.capabilities: Dict[str, AgentCapability] = {}
        self.max_capabilities = max_capabilities
        self.memory = AgentMemory(agent_id=self.id)
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: List[Task] = []
        self.failed_tasks: List[Task] = []
        self.task_history: List[Task] = []
        self.parent_agent: Optional[str] = None
        self.child_agents: Set[str] = set()
        self.peer_agents: Set[str] = set()
        self.performance_metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "avg_task_time": 0.0,
            "success_rate": 1.0,
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

    def register_capability(self, capability: AgentCapability) -> bool:
        with self._lock:
            if capability.name in self.capabilities:
                logger.warning("Capability '%s' already registered for %s", capability.name, self.name)
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
        with self._lock:
            return self.capabilities.get(name)

    def list_capabilities(self) -> List[str]:
        with self._lock:
            return list(self.capabilities.keys())

    def assign_task(self, task: Task) -> bool:
        with self._lock:
            if task.status not in {TaskStatus.PENDING, TaskStatus.ASSIGNED}:
                logger.warning("Task %s is not assignable because it is in status %s", task.id, task.status.value)
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

    def run_task(
        self,
        task: Task,
        timeout_seconds: Optional[float] = None,
        heartbeat: Optional[Callable[[], None]] = None,
    ) -> Any:
        with self._lock:
            if task.id not in self.active_tasks:
                raise ValueError(f"Task {task.id} must be assigned before execution")
            if task.assigned_to != self.id:
                raise ValueError(f"Task {task.id} is assigned to {task.assigned_to}, not {self.id}")
        start_time = datetime.now()
        with self._lock:
            self.status = AgentStatus.BUSY
            self._touch()
        logger.info("Agent %s running task %s", self.name, task.id)

        try:
            if heartbeat is not None:
                heartbeat()
            reasoning = self.think(task.parameters)
            if heartbeat is not None:
                heartbeat()
            result = self.act(reasoning)
            elapsed = (datetime.now() - start_time).total_seconds()
            if timeout_seconds is not None and elapsed > timeout_seconds:
                raise TimeoutError(
                    f"Task {task.id} exceeded timeout ({elapsed:.3f}s > {timeout_seconds:.3f}s)"
                )
            self._record_task_outcome(task, success=True, start_time=start_time)
            logger.info("Task %s completed successfully", task.id)
            return result
        except Exception:
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

    def release_task(self, task_id: str) -> None:
        with self._lock:
            self.active_tasks.pop(task_id, None)
            self._touch()

    def get_active_task(self, task_id: str) -> Optional[Task]:
        with self._lock:
            return self.active_tasks.get(task_id)

    def has_active_task(self, task_id: str) -> bool:
        with self._lock:
            return task_id in self.active_tasks

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
        total = self.performance_metrics["tasks_completed"] + self.performance_metrics["tasks_failed"]
        self.performance_metrics["success_rate"] = self.performance_metrics["tasks_completed"] / total if total else 0
        previous_avg = self.performance_metrics["avg_task_time"]
        completed = self.performance_metrics["tasks_completed"]
        if success and completed:
            self.performance_metrics["avg_task_time"] = previous_avg + ((elapsed - previous_avg) / completed)

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "id": self.id,
                "name": self.name,
                "role": self.role.name,
                "status": self.status.name,
                "capabilities": list(self.capabilities.keys()),
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
    def __init__(self, name: str = "Orchestrator"):
        super().__init__(name, role=AgentRole.ORCHESTRATOR)
        self.managed_agents: Dict[str, BaseAgent] = {}
        self.task_queue: List[Task] = []

    def think(self, input_data: Any) -> Dict[str, Any]:
        return {"analysis": "Task requires orchestration", "priority": "high", "execution_strategy": "parallel"}

    def act(self, decision: Dict[str, Any]) -> Any:
        logger.info("Orchestrator %s executing strategy: %s", self.name, decision.get("execution_strategy"))
        return {"status": "orchestration_complete"}

    def register_agent(self, agent: BaseAgent) -> bool:
        with self._lock:
            if agent.id in self.managed_agents:
                logger.warning("Agent %s is already registered under orchestrator %s", agent.name, self.name)
                return False
            self.managed_agents[agent.id] = agent
            agent.parent_agent = self.id
            self._touch()
            logger.info("Agent %s registered under orchestrator %s", agent.name, self.name)
            return True

    def distribute_task(self, task: Task, target_agent_id: Optional[str] = None) -> bool:
        with self._lock:
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
        with self._lock:
            available_agents = [a for a in self.managed_agents.values() if a.status not in {AgentStatus.SUSPENDED, AgentStatus.BUSY}]
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
                    elif any(token and token in task_description for token in capability_description.split()):
                        capability_match += 1
                return (capability_match, -len(agent.active_tasks), int(agent.performance_metrics["success_rate"] * 100))

            return max(available_agents, key=score)

    def get_system_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "orchestrator": self.get_status(),
                "managed_agents": [a.get_status() for a in self.managed_agents.values()],
                "total_agents": len(self.managed_agents),
                "pending_tasks": len(self.task_queue),
            }


class ExecutorAgent(BaseAgent):
    def __init__(self, name: str = "Executor"):
        super().__init__(name, role=AgentRole.EXECUTOR)
        self.execution_history: List[Dict[str, Any]] = []

    def think(self, input_data: Any) -> Dict[str, Any]:
        return {"action": "execute", "parameters": input_data, "validation": True}

    def act(self, decision: Dict[str, Any]) -> Any:
        params = decision.get("parameters", {})
        self.execution_history.append({
            "timestamp": datetime.now().isoformat(),
            "decision": _make_json_safe(decision),
            "result": "executed",
        })
        return {"execution": "successful", "parameters_processed": _make_json_safe(params)}


class AnalyzerAgent(BaseAgent):
    def __init__(self, name: str = "Analyzer"):
        super().__init__(name, role=AgentRole.ANALYZER)
        self.analysis_cache: Dict[str, Dict[str, Any]] = {}

    def think(self, input_data: Any) -> Dict[str, Any]:
        return {"data_received": bool(input_data), "analysis_type": "comprehensive", "insights_generated": True}

    def act(self, decision: Dict[str, Any]) -> Any:
        return {"analysis_complete": True, "insights": _make_json_safe(decision), "timestamp": datetime.now().isoformat()}


class LearnerAgent(BaseAgent):
    def __init__(self, name: str = "Learner"):
        super().__init__(name, role=AgentRole.LEARNER)
        self.learned_patterns: Dict[str, Any] = {}
        self.learning_history: List[Dict[str, Any]] = []

    def think(self, input_data: Any) -> Dict[str, Any]:
        return {"learning_mode": True, "input_analyzed": True, "patterns_identified": []}

    def act(self, decision: Dict[str, Any]) -> Any:
        self.learning_history.append({
            "timestamp": datetime.now().isoformat(),
            "decision": _make_json_safe(decision),
            "patterns_learned": len(self.learned_patterns),
        })
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


class AgentSystem:
    def __init__(
        self,
        name: str = "Ai-morphasis",
        task_store: Optional[TaskStore] = None,
        *,
        default_lease_seconds: int = 300,
        execution_timeout_seconds: Optional[float] = None,
        persistence_retry_attempts: int = 3,
        persistence_retry_backoff_seconds: float = 0.05,
    ):
        if not name.strip():
            raise ValueError("System name cannot be empty")
        if default_lease_seconds <= 0:
            raise ValueError("default_lease_seconds must be greater than 0")
        if persistence_retry_attempts <= 0:
            raise ValueError("persistence_retry_attempts must be greater than 0")
        if persistence_retry_backoff_seconds < 0:
            raise ValueError("persistence_retry_backoff_seconds must be >= 0")

        self.name = name
        self.id = str(uuid.uuid4())
        self.created_at = datetime.now()
        self._lock = RLock()
        self.task_store = task_store or InMemoryTaskStore()
        self.default_lease_seconds = default_lease_seconds
        self.execution_timeout_seconds = execution_timeout_seconds
        self.persistence_retry_attempts = persistence_retry_attempts
        self.persistence_retry_backoff_seconds = persistence_retry_backoff_seconds
        self.orchestrator = OrchestratorAgent(f"{name}-Orchestrator")
        self.agents: Dict[str, BaseAgent] = {self.orchestrator.id: self.orchestrator}
        self.global_task_queue: List[Task] = []
        self.completed_tasks: List[Task] = []
        self.failed_tasks: List[Task] = []
        self.system_metrics = {
            "total_agents": 1,
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "avg_task_duration": 0.0,
        }
        logger.info("Initialized Agent System: %s", self.name)

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
                with agent._lock:
                    has_active_tasks = bool(agent.active_tasks)
                if has_active_tasks:
                    logger.warning("Cannot remove agent %s while it has active tasks", agent.name)
                    return False
                self.agents.pop(agent_id)
                self.orchestrator.managed_agents.pop(agent_id, None)
                self.system_metrics["total_agents"] -= 1
                logger.info("Agent %s removed from system", agent.name)
                return True
            return False

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        with self._lock:
            return self.agents.get(agent_id)

    def create_task(self, description: str, parameters: Dict[str, Any], priority: TaskPriority = TaskPriority.NORMAL) -> Task:
        with self._lock:
            task = Task(description=description, parameters=parameters, priority=priority)
            self._set_claim(task, None)
            self._enqueue_if_missing(task)
            self.system_metrics["total_tasks"] += 1
            self._store_task(task)
            logger.info("task_created task_id=%s priority=%s description=%s", task.id, priority.name, description)
            return task

    def submit_task(self, task: Task, agent_id: Optional[str] = None) -> bool:
        with self._lock:
            persisted = self.load_task(task.id)
            if persisted is not None:
                task = persisted
            if task.status == TaskStatus.ASSIGNED and task.assigned_to == agent_id and agent_id:
                agent = self.agents.get(agent_id)
                if agent and not agent.has_active_task(task.id):
                    agent.assign_task(task)
                self._dequeue_task(task.id)
                logger.info("task_submit_idempotent task_id=%s agent_id=%s status=%s", task.id, agent_id, task.status.name)
                return True
            if task.status != TaskStatus.PENDING:
                logger.warning("task_submit_rejected task_id=%s status=%s", task.id, task.status.name)
                return False

            assigned = False
            if agent_id:
                agent = self.agents.get(agent_id)
                if agent:
                    assigned = agent.assign_task(task)
            else:
                assigned = self.orchestrator.distribute_task(task)

            if assigned:
                self._set_task_status(task, TaskStatus.ASSIGNED, assigned_to=task.assigned_to, claimed_by=task.assigned_to)
                self._dequeue_task(task.id)
                logger.info("task_submitted task_id=%s agent_id=%s", task.id, task.assigned_to)
                return True

            logger.warning("task_submit_failed task_id=%s requested_agent_id=%s", task.id, agent_id)
            return False

    def execute_task(self, task_id: str, agent_id: str) -> Any:
        with self._lock:
            agent = self.agents.get(agent_id)
            if not agent:
                raise KeyError(f"Agent not found: {agent_id}")
            task = agent.get_active_task(task_id)
            if not task:
                raise KeyError(f"Task {task_id} is not assigned to agent {agent_id}")
            stored_task = self.task_store.get_task(task_id)
            if stored_task is not None:
                persisted_task = self._from_stored_task(stored_task)
                if (
                    persisted_task.assigned_to
                    and persisted_task.assigned_to != agent_id
                    and not self._is_lease_expired(persisted_task)
                ):
                    raise ValueError(
                        f"Task {task.id} is assigned to {persisted_task.assigned_to}, not {agent_id}"
                    )
                task.metadata = dict(persisted_task.metadata)
                task.assigned_to = persisted_task.assigned_to
                task.status = persisted_task.status
            self._ensure_claimed_by(task, agent_id)
            self._set_task_status(task, TaskStatus.RUNNING, assigned_to=agent_id, claimed_by=agent_id)
            timeout_seconds = self._resolve_execution_timeout(task)

        start_time = datetime.now()
        try:
            result = agent.run_task(task, timeout_seconds=timeout_seconds, heartbeat=lambda: self.heartbeat_task(task_id, agent_id))
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
            raise

    def cancel_task(self, task_id: str, reason: Optional[str] = None) -> Task:
        with self._lock:
            task = self.load_task(task_id)
            if task is None:
                raise KeyError(f"Task not found: {task_id}")
            if task.status not in {TaskStatus.PENDING, TaskStatus.ASSIGNED}:
                raise ValueError(f"Task {task_id} in status {task.status.value} cannot be cancelled")

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
            logger.info("task_cancelled task_id=%s reason=%s", task.id, reason)
            return task

    def load_task(self, task_id: str) -> Optional[Task]:
        with self._lock:
            active_task = self._find_active_task(task_id)
            if active_task is not None:
                return active_task
            stored_task = self.task_store.get_task(task_id)
            if stored_task is None:
                return None
            return self._from_stored_task(stored_task)

    def list_persisted_tasks(self, status: Optional[TaskStatus] = None) -> List[Task]:
        with self._lock:
            stored_tasks = self.task_store.list_tasks(status.name if status else None)
            return [self._from_stored_task(task) for task in stored_tasks]

    def recover_incomplete_tasks(self, reset_to: TaskStatus = TaskStatus.PENDING) -> int:
        if reset_to != TaskStatus.PENDING:
            raise ValueError("Only reset_to=TaskStatus.PENDING is supported for recovery")

        recovered = 0
        with self._lock:
            for status in (TaskStatus.ASSIGNED, TaskStatus.RUNNING):
                for task in self.list_persisted_tasks(status):
                    active_task = self._find_active_task(task.id)
                    working_task = active_task or task
                    if self._has_valid_lease(task):
                        logger.info(
                            "task_recovery_skipped_active_lease task_id=%s claimed_by=%s status=%s",
                            task.id,
                            self._get_claimed_by(task),
                            task.status.name,
                        )
                        continue
                    original_assigned_to = working_task.assigned_to
                    self._release_task_from_agent(working_task.id, original_assigned_to)
                    working_task.result = None
                    working_task.error = None
                    working_task.completed_at = None
                    self._set_task_status(working_task, TaskStatus.PENDING, assigned_to=None, claimed_by=None)
                    self._enqueue_if_missing(working_task)
                    recovered += 1

        logger.info("task_recovery_complete recovered=%s", recovered)
        return recovered

    def requeue_task(self, task_id: str) -> Task:
        with self._lock:
            task = self.load_task(task_id)
            if task is None:
                raise KeyError(f"Task not found: {task_id}")
            if not self._is_recoverable_status(task.status):
                raise ValueError(f"Task {task_id} in status {task.status.value} cannot be requeued")
            if task.status in {TaskStatus.ASSIGNED, TaskStatus.RUNNING} and self._has_valid_lease(task):
                raise ValueError(f"Task {task_id} has an active lease and cannot be requeued")

            previous_status = task.status
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
            logger.info("task_requeued task_id=%s previous_status=%s", task.id, previous_status.name)
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
        try:
            priority = TaskPriority[stored_task.priority]
        except KeyError:
            priority = TaskPriority.NORMAL
        try:
            status = TaskStatus[stored_task.status]
        except KeyError:
            status = TaskStatus.PENDING
        return Task(
            id=stored_task.id,
            description=stored_task.description,
            priority=priority,
            assigned_to=stored_task.assigned_to,
            status=status,
            created_at=stored_task.created_at,
            completed_at=stored_task.completed_at,
            result=stored_task.result,
            error=stored_task.error,
            parameters=dict(stored_task.parameters),
            dependencies=list(stored_task.dependencies),
            metadata=dict(stored_task.metadata),
        )

    def _store_task(self, task: Task) -> None:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.persistence_retry_attempts + 1):
            try:
                self.task_store.create_task(self._to_stored_task(task))
                return
            except ValueError:
                raise
            except Exception as exc:  # pragma: no cover - defensive for store backends
                last_error = exc
                if attempt == self.persistence_retry_attempts:
                    logger.exception("task_store_create_failed task_id=%s attempts=%s", task.id, attempt)
                    raise
                backoff = self.persistence_retry_backoff_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "task_store_create_retry task_id=%s attempt=%s max_attempts=%s backoff_seconds=%.3f error=%s",
                    task.id,
                    attempt,
                    self.persistence_retry_attempts,
                    backoff,
                    exc,
                )
                sleep(backoff)
        if last_error is not None:
            raise last_error

    def _update_task_record(self, task: Task) -> None:
        for attempt in range(1, self.persistence_retry_attempts + 1):
            try:
                self.task_store.update_task(self._to_stored_task(task))
                return
            except (KeyError, ValueError):
                raise
            except Exception as exc:
                if attempt == self.persistence_retry_attempts:
                    logger.exception(
                        "task_store_update_failed task_id=%s assigned_to=%s status=%s attempts=%s",
                        task.id,
                        task.assigned_to,
                        task.status.name,
                        attempt,
                    )
                    raise
                backoff = self.persistence_retry_backoff_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "task_store_update_retry task_id=%s status=%s attempt=%s max_attempts=%s backoff_seconds=%.3f error=%s",
                    task.id,
                    task.status.name,
                    attempt,
                    self.persistence_retry_attempts,
                    backoff,
                    exc,
                )
                sleep(backoff)

    def heartbeat_task(self, task_id: str, agent_id: str, lease_seconds: Optional[int] = None) -> Task:
        """Refresh a task lease while an agent is still actively working on it."""
        with self._lock:
            task = self.load_task(task_id)
            if task is None:
                raise KeyError(f"Task not found: {task_id}")
            if task.status not in {TaskStatus.ASSIGNED, TaskStatus.RUNNING}:
                raise ValueError(f"Task {task_id} in status {task.status.value} cannot refresh lease")
            self._ensure_claimed_by(task, agent_id)
            self._set_claim(task, claimed_by=agent_id, lease_seconds=lease_seconds)
            self._update_task_record(task)
            return task

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
        previous_status = task.status
        self._validate_task_transition(task.id, previous_status, status)
        task.status = status
        task.assigned_to = assigned_to
        task.result = result
        task.error = error
        task.completed_at = completed_at
        self._set_claim(task, claimed_by)
        self._update_task_record(task)
        self._log_status_transition(task, previous_status, status)

    def _set_claim(
        self,
        task: Task,
        claimed_by: Optional[str],
        *,
        claimed_at: Optional[datetime] = None,
        heartbeat_at: Optional[datetime] = None,
        lease_seconds: Optional[int] = None,
    ) -> None:
        task.metadata = dict(task.metadata)
        if claimed_by is None:
            task.metadata.pop("claimed_by", None)
            task.metadata.pop("claimed_at", None)
            task.metadata.pop("heartbeat_at", None)
            task.metadata.pop("lease_expires_at", None)
        else:
            now = datetime.now()
            lease_duration = lease_seconds or self.default_lease_seconds
            claim_time = claimed_at or now
            heartbeat_time = heartbeat_at or now
            task.metadata["claimed_by"] = claimed_by
            task.metadata["claimed_at"] = claim_time.isoformat()
            task.metadata["heartbeat_at"] = heartbeat_time.isoformat()
            task.metadata["lease_expires_at"] = (heartbeat_time + timedelta(seconds=lease_duration)).isoformat()

    def _ensure_claimed_by(self, task: Task, agent_id: str) -> None:
        claimed_by = self._get_claimed_by(task)
        if claimed_by and claimed_by != agent_id:
            if self._is_lease_expired(task):
                logger.warning(
                    "task_claim_stale task_id=%s stale_claimed_by=%s requested_by=%s lease_expires_at=%s",
                    task.id,
                    claimed_by,
                    agent_id,
                    self._get_claim_value(task, "lease_expires_at"),
                )
                return
            raise ValueError(f"Task {task.id} is claimed by {claimed_by}, not {agent_id}")

    def _is_recoverable_status(self, status: TaskStatus) -> bool:
        return status in {TaskStatus.ASSIGNED, TaskStatus.RUNNING, TaskStatus.FAILED}

    def _find_active_task(self, task_id: str) -> Optional[Task]:
        for agent in self.agents.values():
            task = agent.get_active_task(task_id)
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
        if not any(existing.id == task.id for existing in self.global_task_queue):
            self.global_task_queue.append(task)

    def _dequeue_task(self, task_id: str) -> None:
        self.global_task_queue = [queued_task for queued_task in self.global_task_queue if queued_task.id != task_id]

    def _append_unique_task(self, collection: List[Task], task: Task) -> None:
        if not any(existing.id == task.id for existing in collection):
            collection.append(task)

    def _update_system_metrics(self, success: bool, start_time: datetime) -> None:
        elapsed = (datetime.now() - start_time).total_seconds()
        key = "successful_tasks" if success else "failed_tasks"
        self.system_metrics[key] += 1
        completed = max(self.system_metrics["successful_tasks"], 0)
        previous_avg = self.system_metrics["avg_task_duration"]
        if success and completed:
            self.system_metrics["avg_task_duration"] = previous_avg + ((elapsed - previous_avg) / completed)

    def _resolve_execution_timeout(self, task: Task) -> Optional[float]:
        metadata_timeout = task.metadata.get("execution_timeout_seconds") if isinstance(task.metadata, dict) else None
        if metadata_timeout is None:
            return self.execution_timeout_seconds
        try:
            timeout = float(metadata_timeout)
        except (TypeError, ValueError):
            logger.warning("invalid_task_timeout task_id=%s timeout=%s", task.id, metadata_timeout)
            return self.execution_timeout_seconds
        return timeout if timeout > 0 else self.execution_timeout_seconds

    def _validate_task_transition(self, task_id: str, current: TaskStatus, new: TaskStatus) -> None:
        if current == new:
            return
        allowed = ALLOWED_TASK_TRANSITIONS.get(current, set())
        if new not in allowed:
            raise ValueError(f"Invalid task transition for {task_id}: {current.name} -> {new.name}")

    def _log_status_transition(self, task: Task, old_status: TaskStatus, new_status: TaskStatus) -> None:
        if old_status == new_status:
            return
        logger.info(
            "task_status_transition task_id=%s from=%s to=%s agent_id=%s claimed_by=%s lease_expires_at=%s error=%s",
            task.id,
            old_status.name,
            new_status.name,
            task.assigned_to,
            self._get_claimed_by(task),
            self._get_claim_value(task, "lease_expires_at"),
            task.error,
        )

    def _get_claim_value(self, task: Task, key: str) -> Optional[str]:
        if not isinstance(task.metadata, dict):
            return None
        value = task.metadata.get(key)
        return value if isinstance(value, str) else None

    def _get_claimed_by(self, task: Task) -> Optional[str]:
        return self._get_claim_value(task, "claimed_by")

    def _parse_claim_datetime(self, task: Task, key: str) -> Optional[datetime]:
        raw = self._get_claim_value(task, key)
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None

    def _is_lease_expired(self, task: Task, at_time: Optional[datetime] = None) -> bool:
        lease_expires_at = self._parse_claim_datetime(task, "lease_expires_at")
        if lease_expires_at is None:
            return False
        now = at_time or datetime.now()
        return lease_expires_at <= now

    def _has_valid_lease(self, task: Task) -> bool:
        claimed_by = self._get_claimed_by(task)
        if not claimed_by:
            return False
        return not self._is_lease_expired(task)

    def get_system_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "system_name": self.name,
                "system_id": self.id,
                "created_at": self.created_at.isoformat(),
                "agents": {aid: agent.get_status() for aid, agent in self.agents.items()},
                "metrics": _make_json_safe(self.system_metrics),
                "pending_tasks": len(self.global_task_queue),
                "completed_tasks": len(self.completed_tasks),
                "failed_tasks": len(self.failed_tasks),
            }

    def to_json(self) -> str:
        return json.dumps(self.get_system_status(), indent=2, default=str)

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
    def create_team(cls, team_config: Dict[str, int], task_store: Optional[TaskStore] = None) -> AgentSystem:
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    example_usage()
