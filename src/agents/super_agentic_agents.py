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
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

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


logger = logging.getLogger(__name__)


# ============================================================================
# Agent Enums and Data Models
# ============================================================================

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


class TaskPriority(Enum):
    """Defines task execution priority levels."""
    CRITICAL = 5
    HIGH = 4
    NORMAL = 3
    LOW = 2
    DEFERRED = 1


@dataclass
class AgentCapability:
    """Represents a capability an agent can perform."""
    name: str
    description: str
    func: Optional[Callable] = None
    confidence_score: float = 1.0
    requires_resources: List[str] = field(default_factory=list)
    version: str = "1.0.0"

    def __repr__(self) -> str:
        return f"<Capability: {self.name} v{self.version} ({self.confidence_score:.2%})>"


@dataclass
class AgentMemory:
    """Represents agent memory with episodic and semantic storage.

    Supports an optional Redis backend for persistence.  When *redis_url* is
    provided the class will attempt to create an async Redis connection on first
    use; if Redis is unavailable it falls back transparently to in-process dicts
    and logs a warning.
    """
    agent_id: str
    episodic_memory: Dict[str, Any] = field(default_factory=dict)  # Short-term
    semantic_memory: Dict[str, Any] = field(default_factory=dict)  # Long-term
    procedural_memory: Dict[str, Callable] = field(default_factory=dict)  # Skills
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    max_episodes: int = 1000
    redis_url: Optional[str] = None
    _redis: Any = field(default=None, init=False, repr=False, compare=False)

    # ------------------------------------------------------------------
    # Redis helpers
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

    # ------------------------------------------------------------------
    # Public API (sync wrappers call async internally where possible)
    # ------------------------------------------------------------------

    def store_episode(self, key: str, value: Any) -> None:
        """Store an episode in short-term memory (in-process)."""
        if len(self.episodic_memory) >= self.max_episodes:
            oldest_key = next(iter(self.episodic_memory))
            del self.episodic_memory[oldest_key]
        self.episodic_memory[key] = {
            "value": value,
            "timestamp": datetime.now()
        }
        self.last_accessed = datetime.now()

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

    def store_semantic(self, key: str, value: Any) -> None:
        """Store knowledge in long-term memory (in-process)."""
        self.semantic_memory[key] = {
            "value": value,
            "timestamp": datetime.now(),
            "access_count": 0
        }

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

    def retrieve(self, key: str, memory_type: str = "auto") -> Optional[Any]:
        """Retrieve from memory (auto-selects best source, in-process only)."""
        if memory_type in ("auto", "episodic") and key in self.episodic_memory:
            return self.episodic_memory[key]["value"]
        if memory_type in ("auto", "semantic") and key in self.semantic_memory:
            self.semantic_memory[key]["access_count"] += 1
            return self.semantic_memory[key]["value"]
        return None

    async def async_retrieve(self, key: str, memory_type: str = "auto") -> Optional[Any]:
        """Retrieve from memory, checking Redis when available."""
        # Try in-process first (fast path)
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


@dataclass
class Task:
    """Represents a task for agents to execute."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    priority: TaskPriority = TaskPriority.NORMAL
    assigned_to: Optional[str] = None
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "priority": self.priority.name,
            "assigned_to": self.assigned_to,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error,
            "parameters": self.parameters,
            "dependencies": self.dependencies,
            "metadata": self.metadata
        }


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
        def validate_uuid_dependencies(cls, deps: List[str]) -> List[str]:
            import re
            uuid_re = re.compile(
                r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
                re.IGNORECASE,
            )
            for dep in deps:
                if not uuid_re.match(dep):
                    raise ValueError(f"Dependency '{dep}' is not a valid UUID.")
            return deps

else:
    # Lightweight stubs when pydantic is not installed
    class AgentConfig:  # type: ignore[no-redef]
        def __init__(self, name: str, role: "AgentRole" = None,
                     max_capabilities: int = 50, max_retries: int = 3):
            if not name or len(name) > 100:
                raise ValueError("name must be 1–100 characters")
            if not 1 <= max_capabilities <= 200:
                raise ValueError("max_capabilities must be 1–200")
            if not 0 <= max_retries <= 10:
                raise ValueError("max_retries must be 0–10")
            self.name = name
            self.role = role
            self.max_capabilities = max_capabilities
            self.max_retries = max_retries

    class TaskConfig:  # type: ignore[no-redef]
        def __init__(self, description: str, priority: "TaskPriority" = None,
                     parameters: Dict[str, Any] = None,
                     dependencies: List[str] = None):
            import re
            if not description:
                raise ValueError("description must be non-empty")
            uuid_re = re.compile(
                r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
                re.IGNORECASE,
            )
            for dep in (dependencies or []):
                if not uuid_re.match(dep):
                    raise ValueError(f"Dependency '{dep}' is not a valid UUID.")
            self.description = description
            self.priority = priority or TaskPriority.NORMAL
            self.parameters = parameters or {}
            self.dependencies = dependencies or []


# ============================================================================
# Base Agent Classes
# ============================================================================

class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(
        self,
        name: str,
        role: AgentRole = AgentRole.EXECUTOR,
        max_capabilities: int = 50,
        llm_client: Optional[Any] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        concurrency_limit: int = 5,
        redis_url: Optional[str] = None,
    ):
        """Initialize a base agent.

        Args:
            name: Human-readable agent name.
            role: Agent role (default: EXECUTOR).
            max_capabilities: Maximum number of registered capabilities.
            llm_client: Optional ``openai.AsyncOpenAI``-compatible client.
            max_retries: Number of retry attempts on task failure (0–10).
            retry_delay: Base delay in seconds between retries.
            concurrency_limit: Maximum concurrent tasks via semaphore.
            redis_url: Optional Redis URL for persistent memory.
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.role = role
        self.status = AgentStatus.IDLE
        self.created_at = datetime.now()
        self.last_activity = datetime.now()

        self.capabilities: Dict[str, AgentCapability] = {}
        self.max_capabilities = max_capabilities
        self.memory = AgentMemory(agent_id=self.id, redis_url=redis_url)

        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: List[Task] = []
        self.task_history: List[Task] = []

        self.parent_agent: Optional[str] = None
        self.child_agents: Set[str] = set()
        self.peer_agents: Set[str] = set()

        # LLM integration
        self.llm_client = llm_client
        self.llm_model: str = "gpt-4o-mini"

        # Retry config
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Concurrency control
        self._semaphore = asyncio.Semaphore(concurrency_limit)

        # Structured logger
        self._slog = StructuredLogger(__name__, agent_id=self.id, agent_name=self.name)

        self.performance_metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "avg_task_time": 0.0,
            "success_rate": 1.0
        }

        self._slog.info(f"Initialized {self.role.value} agent: {self.name} (ID: {self.id})")

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    async def think(self, input_data: Any) -> Dict[str, Any]:
        """Core reasoning method - must be implemented by subclasses."""

    @abstractmethod
    async def act(self, decision: Dict[str, Any]) -> Any:
        """Execution method - must be implemented by subclasses."""

    # ------------------------------------------------------------------
    # Capability management
    # ------------------------------------------------------------------

    def register_capability(self, capability: AgentCapability) -> bool:
        """Register a new capability."""
        if len(self.capabilities) >= self.max_capabilities:
            self._slog.warning(f"Agent {self.name} has reached max capabilities limit")
            return False

        self.capabilities[capability.name] = capability
        self.memory.store_semantic(f"capability:{capability.name}", capability)
        self._slog.info(f"Capability '{capability.name}' registered for {self.name}")
        return True

    def get_capability(self, name: str) -> Optional[AgentCapability]:
        """Retrieve a registered capability."""
        return self.capabilities.get(name)

    def list_capabilities(self) -> List[str]:
        """Get list of all capability names."""
        return list(self.capabilities.keys())

    # ------------------------------------------------------------------
    # Task management
    # ------------------------------------------------------------------

    def assign_task(self, task: Task) -> bool:
        """Assign a task to this agent."""
        self.active_tasks[task.id] = task
        task.assigned_to = self.id
        task.status = "assigned"
        self.memory.store_episode(f"task:{task.id}", task)
        self._slog.info(f"Task {task.id} assigned to agent {self.name}", task_id=task.id)
        return True

    async def execute_task(self, task: Task) -> Any:
        """Execute an assigned task with retry and exponential backoff.

        On success, resets status to IDLE and records duration in
        ``task.metadata["duration_ms"]``.  After exhausting retries, sets
        status to ERROR and records the final error string.
        """
        async with self._semaphore:
            start_time = time.monotonic()
            task.metadata.setdefault("start_time", datetime.now().isoformat())

            last_error: Optional[Exception] = None
            for attempt in range(self.max_retries + 1):
                try:
                    self.status = AgentStatus.BUSY
                    self._slog.info(
                        f"Agent {self.name} executing task {task.id} (attempt {attempt + 1})",
                        task_id=task.id,
                    )

                    reasoning = await self.think(task.parameters)
                    result = await self.act(reasoning)

                    # Success path
                    duration_ms = (time.monotonic() - start_time) * 1000
                    task.metadata["duration_ms"] = duration_ms
                    task.status = "completed"
                    task.result = result
                    task.completed_at = datetime.now()

                    self._update_metrics(task, success=True)
                    self.active_tasks.pop(task.id, None)
                    self.completed_tasks.append(task)
                    self.task_history.append(task)

                    self.status = AgentStatus.IDLE
                    self.last_activity = datetime.now()
                    self._slog.info(
                        f"Task {task.id} completed in {duration_ms:.1f} ms",
                        task_id=task.id,
                    )
                    return result

                except Exception as exc:
                    last_error = exc
                    if attempt < self.max_retries:
                        delay = self.retry_delay * (2 ** attempt)
                        self._slog.warning(
                            f"Task {task.id} attempt {attempt + 1} failed: {exc}. "
                            f"Retrying in {delay:.1f}s…",
                            task_id=task.id,
                        )
                        await asyncio.sleep(delay)
                    else:
                        # Exhausted retries
                        duration_ms = (time.monotonic() - start_time) * 1000
                        task.metadata["duration_ms"] = duration_ms
                        task.status = "failed"
                        task.error = str(exc)
                        task.completed_at = datetime.now()
                        self._update_metrics(task, success=False)
                        self.status = AgentStatus.ERROR
                        self.last_activity = datetime.now()
                        self._slog.error(
                            f"Task {task.id} failed after {self.max_retries + 1} attempts: {exc}",
                            task_id=task.id,
                        )
                        raise

    def _update_metrics(self, task: Task, success: bool) -> None:
        """Update performance metrics."""
        if success:
            self.performance_metrics["tasks_completed"] += 1
        else:
            self.performance_metrics["tasks_failed"] += 1

        total = (
            self.performance_metrics["tasks_completed"]
            + self.performance_metrics["tasks_failed"]
        )
        self.performance_metrics["success_rate"] = (
            self.performance_metrics["tasks_completed"] / total if total > 0 else 0
        )

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status."""
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role.name,
            "status": self.status.name,
            "capabilities": self.list_capabilities(),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "performance": self.performance_metrics,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat()
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name} ({self.role.value})>"


# ============================================================================
# Specialized Agent Classes
# ============================================================================

class OrchestratorAgent(BaseAgent):
    """
    Master orchestrator agent that manages and coordinates other agents.
    Responsible for task distribution, monitoring, and system-level decisions.
    """

    def __init__(self, name: str = "Orchestrator", **kwargs: Any):
        super().__init__(name, role=AgentRole.ORCHESTRATOR, **kwargs)
        self.managed_agents: Dict[str, BaseAgent] = {}
        self.task_queue: List[Task] = []

    async def think(self, input_data: Any) -> Dict[str, Any]:
        """Analyze input and create execution plan.

        When an LLM client is configured, calls the chat completions API to
        produce a real execution plan.  Falls back to a hardcoded dict otherwise.
        """
        if self.llm_client is not None:
            try:
                prompt = (
                    f"You are an orchestration agent. Given the following task input, "
                    f"create a concise JSON execution plan with keys: analysis, priority, "
                    f"execution_strategy.\n\nTask input: {json.dumps(input_data, default=str)}"
                )
                response = await self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                return json.loads(content)
            except Exception as exc:
                self._slog.warning(f"LLM call failed, using fallback plan: {exc}")

        return {
            "analysis": "Task requires orchestration",
            "priority": "high",
            "execution_strategy": "parallel"
        }

    async def act(self, decision: Dict[str, Any]) -> Any:
        """Orchestrate agent actions based on decision."""
        self._slog.info(
            f"Orchestrator {self.name} executing strategy: {decision.get('execution_strategy')}"
        )
        return {"status": "orchestration_complete"}

    def register_agent(self, agent: BaseAgent) -> bool:
        """Register an agent under this orchestrator."""
        self.managed_agents[agent.id] = agent
        agent.parent_agent = self.id
        self._slog.info(f"Agent {agent.name} registered under orchestrator {self.name}")
        return True

    def distribute_task(self, task: Task, target_agent_id: Optional[str] = None) -> bool:
        """Distribute a task to appropriate agent."""
        if target_agent_id and target_agent_id in self.managed_agents:
            agent = self.managed_agents[target_agent_id]
            return agent.assign_task(task)

        best_agent = self._select_best_agent(task)
        if best_agent:
            return best_agent.assign_task(task)

        self._slog.warning(f"No suitable agent found for task {task.id}", task_id=task.id)
        return False

    def _select_best_agent(self, task: Task) -> Optional[BaseAgent]:
        """Select best agent for task based on capabilities."""
        available_agents = [
            a for a in self.managed_agents.values()
            if a.status != AgentStatus.SUSPENDED
        ]

        if not available_agents:
            return None

        return min(available_agents, key=lambda a: len(a.active_tasks))

    def get_system_status(self) -> Dict[str, Any]:
        """Get status of entire agent system."""
        return {
            "orchestrator": self.get_status(),
            "managed_agents": [a.get_status() for a in self.managed_agents.values()],
            "total_agents": len(self.managed_agents),
            "pending_tasks": len(self.task_queue)
        }


class ExecutorAgent(BaseAgent):
    """
    Executor agent that performs specific tasks and operations.
    Specialized for task execution and implementation.
    """

    def __init__(self, name: str = "Executor", **kwargs: Any):
        super().__init__(name, role=AgentRole.EXECUTOR, **kwargs)
        self.execution_history: List[Dict[str, Any]] = []

    async def think(self, input_data: Any) -> Dict[str, Any]:
        """Analyze task parameters and create execution plan."""
        return {
            "action": "execute",
            "parameters": input_data,
            "validation": True
        }

    async def act(self, decision: Dict[str, Any]) -> Any:
        """Execute the task based on decision."""
        params = decision.get("parameters", {})
        self.execution_history.append({
            "timestamp": datetime.now().isoformat(),
            "decision": decision,
            "result": "executed"
        })
        return {"execution": "successful", "parameters_processed": params}


class AnalyzerAgent(BaseAgent):
    """
    Analyzer agent that examines data, identifies patterns, and provides insights.
    Specialized for analysis and decision support.
    """

    def __init__(self, name: str = "Analyzer", **kwargs: Any):
        super().__init__(name, role=AgentRole.ANALYZER, **kwargs)
        self.analysis_cache: Dict[str, Dict[str, Any]] = {}

    async def think(self, input_data: Any) -> Dict[str, Any]:
        """Analyze input data and extract insights.

        Uses the LLM client when available; falls back to a heuristic dict.
        """
        if self.llm_client is not None:
            try:
                prompt = (
                    f"Analyze the following input data and return a JSON object with keys: "
                    f"data_received (bool), analysis_type (string), insights_generated (bool), "
                    f"summary (string).\n\nInput: {json.dumps(input_data, default=str)}"
                )
                response = await self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                return json.loads(content)
            except Exception as exc:
                self._slog.warning(f"LLM call failed, using fallback analysis: {exc}")

        return {
            "data_received": bool(input_data),
            "analysis_type": "comprehensive",
            "insights_generated": True
        }

    async def act(self, decision: Dict[str, Any]) -> Any:
        """Generate and return analysis results."""
        result = {
            "analysis_complete": True,
            "insights": decision,
            "timestamp": datetime.now().isoformat()
        }
        return result


class LearnerAgent(BaseAgent):
    """
    Learner agent that adapts and improves through experience.
    Specialized for learning, optimization, and capability evolution.
    """

    def __init__(self, name: str = "Learner", **kwargs: Any):
        super().__init__(name, role=AgentRole.LEARNER, **kwargs)
        self.learned_patterns: Dict[str, Any] = {}
        self.learning_history: List[Dict[str, Any]] = []

    async def think(self, input_data: Any) -> Dict[str, Any]:
        """Analyze input for learning opportunities."""
        return {
            "learning_mode": True,
            "input_analyzed": True,
            "patterns_identified": []
        }

    async def act(self, decision: Dict[str, Any]) -> Any:
        """Learn from decision and update internal models."""
        self.learning_history.append({
            "timestamp": datetime.now().isoformat(),
            "decision": decision,
            "patterns_learned": len(self.learned_patterns)
        })
        return {"learning": "in_progress", "patterns": self.learned_patterns}

    def learn_from_experience(self, experience: Dict[str, Any]) -> None:
        """Extract and store learning from an experience."""
        pattern_id = str(uuid.uuid4())
        self.learned_patterns[pattern_id] = {
            "experience": experience,
            "learned_at": datetime.now().isoformat(),
            "confidence": 0.5
        }
        self.memory.store_semantic(f"pattern:{pattern_id}", self.learned_patterns[pattern_id])
        self._slog.info(f"Learner {self.name} learned pattern: {pattern_id}")


# ============================================================================
# Agent System and Management
# ============================================================================

class AgentSystem:
    """
    Central management system for all agents.
    Handles agent lifecycle, orchestration, and inter-agent communication.
    """

    def __init__(self, name: str = "Ai-morphasis"):
        self.name = name
        self.id = str(uuid.uuid4())
        self.created_at = datetime.now()

        self.orchestrator = OrchestratorAgent(f"{name}-Orchestrator")
        self.agents: Dict[str, BaseAgent] = {self.orchestrator.id: self.orchestrator}
        self.global_task_queue: List[Task] = []
        self.completed_tasks: List[Task] = []

        self.system_metrics = {
            "total_agents": 1,
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "avg_task_duration": 0.0
        }

        logger.info(f"Initialized Agent System: {self.name}")

    def add_agent(self, agent: BaseAgent) -> bool:
        """Add an agent to the system."""
        self.agents[agent.id] = agent
        self.orchestrator.register_agent(agent)
        self.system_metrics["total_agents"] += 1
        logger.info(f"Agent {agent.name} added to system")
        return True

    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from the system."""
        if agent_id in self.agents:
            agent = self.agents.pop(agent_id)
            self.system_metrics["total_agents"] -= 1
            logger.info(f"Agent {agent.name} removed from system")
            return True
        return False

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Retrieve an agent by ID."""
        return self.agents.get(agent_id)

    def create_task(
        self,
        description: str,
        parameters: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        dependencies: Optional[List[str]] = None,
    ) -> Task:
        """Create a new task, validating inputs via TaskConfig when possible."""
        cfg = TaskConfig(
            description=description,
            priority=priority,
            parameters=parameters,
            dependencies=dependencies or [],
        )
        task = Task(
            description=cfg.description,
            parameters=cfg.parameters,
            priority=cfg.priority,
            dependencies=cfg.dependencies,
        )
        self.global_task_queue.append(task)
        self.system_metrics["total_tasks"] += 1
        logger.info(f"Task {task.id} created: {description}")
        return task

    def submit_task(self, task: Task, agent_id: Optional[str] = None) -> bool:
        """Submit a task to an agent for execution."""
        if agent_id:
            agent = self.get_agent(agent_id)
            if agent:
                return agent.assign_task(task)
        else:
            return self.orchestrator.distribute_task(task)

        logger.warning(f"Failed to submit task {task.id}")
        return False

    async def run_tasks(self, tasks: List[tuple]) -> List[Any]:
        """Run multiple (agent, task) pairs concurrently using asyncio.gather.

        Args:
            tasks: List of ``(agent, task)`` tuples.

        Returns:
            List of results (or exceptions for failed tasks).
        """
        coros = [agent.execute_task(task) for agent, task in tasks]
        return await asyncio.gather(*coros, return_exceptions=True)

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "system_name": self.name,
            "system_id": self.id,
            "created_at": self.created_at.isoformat(),
            "agents": {aid: agent.get_status() for aid, agent in self.agents.items()},
            "metrics": self.system_metrics,
            "pending_tasks": len(self.global_task_queue),
            "completed_tasks": len(self.completed_tasks)
        }

    def get_health(self) -> Dict[str, Any]:
        """Return a health-check dict for monitoring.

        Returns:
            A dict with keys: ``status``, ``agents_healthy``, ``agents_total``,
            ``uptime_seconds``, ``task_success_rate``.
        """
        total = len(self.agents)
        healthy = sum(
            1 for a in self.agents.values()
            if a.status not in (AgentStatus.ERROR, AgentStatus.SUSPENDED)
        )
        total_tasks = (
            self.system_metrics["successful_tasks"] + self.system_metrics["failed_tasks"]
        )
        success_rate = (
            self.system_metrics["successful_tasks"] / total_tasks if total_tasks > 0 else 1.0
        )
        uptime = (datetime.now() - self.created_at).total_seconds()

        error_rate = 1.0 - success_rate
        if error_rate < 0.05 and healthy == total:
            status = "healthy"
        elif error_rate < 0.30 and healthy >= total * 0.5:
            status = "degraded"
        else:
            status = "unhealthy"

        return {
            "status": status,
            "agents_healthy": healthy,
            "agents_total": total,
            "uptime_seconds": uptime,
            "task_success_rate": success_rate,
        }

    def to_json(self) -> str:
        """Serialize system status to JSON."""
        return json.dumps(self.get_system_status(), indent=2, default=str)

    def __repr__(self) -> str:
        return f"<AgentSystem: {self.name} ({len(self.agents)} agents)>"


# ============================================================================
# Factory and Utilities
# ============================================================================

class AgentFactory:
    """Factory for creating agents with standard configurations."""

    _agent_templates: Dict[str, type] = {
        "executor": ExecutorAgent,
        "analyzer": AnalyzerAgent,
        "learner": LearnerAgent,
        "orchestrator": OrchestratorAgent
    }

    @classmethod
    def create_agent(cls, agent_type: str, name: str, **kwargs: Any) -> Optional[BaseAgent]:
        """Create an agent from template."""
        agent_class = cls._agent_templates.get(agent_type.lower())
        if agent_class:
            return agent_class(name, **kwargs)
        logger.error(f"Unknown agent type: {agent_type}")
        return None

    @classmethod
    def create_team(cls, team_config: Dict[str, int], **kwargs: Any) -> AgentSystem:
        """Create a complete agent team from configuration."""
        system = AgentSystem("Ai-morphasis-Team")

        for agent_type, count in team_config.items():
            for i in range(count):
                agent = cls.create_agent(agent_type, f"{agent_type.title()}-{i+1}", **kwargs)
                if agent:
                    system.add_agent(agent)

        logger.info(f"Agent team created with config: {team_config}")
        return system


# ============================================================================
# Example Usage
# ============================================================================

async def _async_example() -> None:
    """Async implementation of the example usage."""
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
            confidence_score=0.95
        )
    )

    analyzer.register_capability(
        AgentCapability(
            name="data_analysis",
            description="Analyze data and generate insights",
            confidence_score=0.88
        )
    )

    task1 = system.create_task(
        description="Analyze performance metrics",
        parameters={"metric_type": "performance", "duration": "24h"}
    )

    system.submit_task(task1, executor.id)
    await executor.execute_task(task1)

    print("\n" + "="*60)
    print("AGENT SYSTEM STATUS")
    print("="*60)
    print(system.to_json())
    print("\nHEALTH CHECK")
    print("="*60)
    print(json.dumps(system.get_health(), indent=2))


def example_usage() -> None:
    """Demonstrate the super agentic agents framework."""
    asyncio.run(_async_example())


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    example_usage()
