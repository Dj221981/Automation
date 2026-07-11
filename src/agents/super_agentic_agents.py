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

import uuid
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

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


class TaskStatus(Enum):
    """Tracks the lifecycle of a task."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


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
    """Represents agent memory with episodic and semantic storage."""
    agent_id: str
    episodic_memory: Dict[str, Any] = field(default_factory=dict)  # Short-term
    semantic_memory: Dict[str, Any] = field(default_factory=dict)  # Long-term
    procedural_memory: Dict[str, Callable] = field(default_factory=dict)  # Skills
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    max_episodes: int = 1000

    def store_episode(self, key: str, value: Any) -> None:
        """Store an episode in short-term memory."""
        if len(self.episodic_memory) >= self.max_episodes:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.episodic_memory))
            del self.episodic_memory[oldest_key]
        self.episodic_memory[key] = {
            "value": value,
            "timestamp": datetime.now()
        }
        self.last_accessed = datetime.now()

    def store_semantic(self, key: str, value: Any) -> None:
        """Store knowledge in long-term memory."""
        self.semantic_memory[key] = {
            "value": value,
            "timestamp": datetime.now(),
            "access_count": 0
        }

    def retrieve(self, key: str, memory_type: str = "auto") -> Optional[Any]:
        """Retrieve from memory (auto-selects best source)."""
        if memory_type in ("auto", "episodic") and key in self.episodic_memory:
            return self.episodic_memory[key]["value"]
        if memory_type in ("auto", "semantic") and key in self.semantic_memory:
            self.semantic_memory[key]["access_count"] += 1
            return self.semantic_memory[key]["value"]
        return None


@dataclass
class Task:
    """Represents a task for agents to execute."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    priority: TaskPriority = TaskPriority.NORMAL
    assigned_to: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize task data."""
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("Task description must be a non-empty string")
        self.description = self.description.strip()

        if not isinstance(self.priority, TaskPriority):
            raise TypeError("Task priority must be a TaskPriority value")
        if not isinstance(self.parameters, dict):
            raise TypeError("Task parameters must be a dictionary")
        if not isinstance(self.metadata, dict):
            raise TypeError("Task metadata must be a dictionary")
        if not isinstance(self.dependencies, list):
            raise TypeError("Task dependencies must be a list")

        normalized_dependencies: List[str] = []
        for dependency in self.dependencies:
            if not isinstance(dependency, str) or not dependency.strip():
                raise ValueError("Task dependency identifiers must be non-empty strings")
            dependency_id = dependency.strip()
            if dependency_id not in normalized_dependencies:
                normalized_dependencies.append(dependency_id)
        self.dependencies = normalized_dependencies
        self.status = self._coerce_status(self.status)

        status_history = list(self.metadata.get("status_history", []))
        if not status_history:
            status_history.append({
                "from": None,
                "to": self.status.value,
                "timestamp": self._timestamp()
            })
        self.metadata["status_history"] = status_history

    @staticmethod
    def _timestamp() -> str:
        """Return a consistent ISO-formatted timestamp for task events."""
        return datetime.now().isoformat()

    @staticmethod
    def _coerce_status(status: Any) -> TaskStatus:
        """Convert a raw status value into TaskStatus."""
        if isinstance(status, TaskStatus):
            return status
        if isinstance(status, str):
            try:
                return TaskStatus(status.lower())
            except ValueError as exc:
                raise ValueError(f"Unsupported task status: {status}") from exc
        raise TypeError("Task status must be a TaskStatus or string value")

    @property
    def status_value(self) -> str:
        """Return the string value of the current task status."""
        return self.status.value

    def is_terminal(self) -> bool:
        """Return whether the task has reached a terminal state."""
        return self.status in {TaskStatus.COMPLETED, TaskStatus.FAILED}

    def transition_to(self, new_status: TaskStatus) -> None:
        """Move the task to a new valid lifecycle state."""
        new_status = self._coerce_status(new_status)
        if new_status == self.status:
            return

        valid_transitions = {
            TaskStatus.PENDING: {TaskStatus.ASSIGNED, TaskStatus.RUNNING, TaskStatus.FAILED},
            TaskStatus.ASSIGNED: {TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.FAILED},
            TaskStatus.RUNNING: {TaskStatus.COMPLETED, TaskStatus.FAILED},
            TaskStatus.COMPLETED: set(),
            TaskStatus.FAILED: set(),
        }
        if new_status not in valid_transitions[self.status]:
            raise ValueError(
                f"Invalid task status transition from {self.status.value} to {new_status.value}"
            )

        previous_status = self.status
        self.status = new_status
        self.metadata.setdefault("status_history", []).append({
            "from": previous_status.value,
            "to": new_status.value,
            "timestamp": self._timestamp()
        })

    def duration_seconds(self) -> Optional[float]:
        """Return task execution duration in seconds when available."""
        if self.started_at and self.completed_at:
            duration = (self.completed_at - self.started_at).total_seconds()
            if duration < 0:
                logger.warning(f"Task {self.id} reported negative duration {duration}; clamping to 0.0")
                return 0.0
            return duration
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "priority": self.priority.name,
            "assigned_to": self.assigned_to,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error,
            "parameters": self.parameters,
            "dependencies": self.dependencies,
            "metadata": self.metadata
        }


# ============================================================================
# Base Agent Classes
# ============================================================================

class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(
        self,
        name: str,
        role: AgentRole = AgentRole.EXECUTOR,
        max_capabilities: int = 50
    ):
        """Initialize a base agent."""
        if not isinstance(name, str):
            raise TypeError("Agent name must be a string")
        if not name.strip():
            raise ValueError("Agent name cannot be empty or whitespace-only")
        if not isinstance(role, AgentRole):
            raise TypeError("Agent role must be an AgentRole value")
        if not isinstance(max_capabilities, int) or max_capabilities <= 0:
            raise ValueError("Agent max_capabilities must be a positive integer")

        self.id = str(uuid.uuid4())
        self.name = name.strip()
        self.role = role
        self.status = AgentStatus.IDLE
        self.created_at = datetime.now()
        self.last_activity = datetime.now()

        self.capabilities: Dict[str, AgentCapability] = {}
        self.max_capabilities = max_capabilities
        self.memory = AgentMemory(agent_id=self.id)

        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: List[Task] = []
        self.task_history: List[Task] = []
        self.system: Optional["AgentSystem"] = None  # Optional when used outside AgentSystem.

        self.parent_agent: Optional[str] = None
        self.child_agents: Set[str] = set()
        self.peer_agents: Set[str] = set()

        self.performance_metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "avg_task_time": 0.0,
            "success_rate": 1.0
        }
        self._total_task_time = 0.0

        logger.info(f"Initialized {self.role.value} agent: {self.name} (ID: {self.id})")

    @abstractmethod
    def think(self, input_data: Any) -> Dict[str, Any]:
        """Core reasoning method - must be implemented by subclasses."""
        pass

    @abstractmethod
    def act(self, decision: Dict[str, Any]) -> Any:
        """Execution method - must be implemented by subclasses."""
        pass

    def register_capability(self, capability: AgentCapability) -> bool:
        """Register a new capability."""
        if not isinstance(capability, AgentCapability):
            raise TypeError("capability must be an AgentCapability instance")
        if not isinstance(capability.name, str) or not capability.name.strip():
            raise ValueError("Capability name must be a non-empty string")
        if not 0.0 <= capability.confidence_score <= 1.0:
            raise ValueError("Capability confidence_score must be between 0.0 and 1.0 inclusive")
        if len(self.capabilities) >= self.max_capabilities:
            logger.warning(f"Agent {self.name} has reached max capabilities limit")
            return False

        self.capabilities[capability.name] = capability
        self.memory.store_semantic(f"capability:{capability.name}", capability)
        self.last_activity = datetime.now()
        logger.info(f"Capability '{capability.name}' registered for {self.name}")
        return True

    def get_capability(self, name: str) -> Optional[AgentCapability]:
        """Retrieve a registered capability."""
        return self.capabilities.get(name)

    def list_capabilities(self) -> List[str]:
        """Get list of all capability names."""
        return list(self.capabilities.keys())

    def _prepare_task_for_assignment(self, task: Task) -> bool:
        """Validate task submission when the agent is attached to a system."""
        return True if self.system is None else self.system._prepare_task_for_assignment(task)

    def _notify_system(self, method_name: str, *args: Any, **kwargs: Any) -> None:
        """Notify the attached system, if any, about an agent lifecycle event."""
        if self.system is None:
            return
        getattr(self.system, method_name)(*args, **kwargs)

    def assign_task(self, task: Task) -> bool:
        """Assign a task to this agent."""
        if not isinstance(task, Task):
            raise TypeError("task must be a Task instance")
        if self.status == AgentStatus.SUSPENDED:
            logger.warning(f"Cannot assign task {task.id} to suspended agent {self.name}")
            return False
        if not self._prepare_task_for_assignment(task):
            return False
        if task.id in self.active_tasks:
            logger.warning(f"Task {task.id} is already active on agent {self.name}")
            return False
        if task.is_terminal():
            logger.warning(f"Cannot assign terminal task {task.id} to agent {self.name}")
            return False
        if task.assigned_to and task.assigned_to != self.id:
            logger.warning(
                f"Cannot assign task {task.id} to {self.name}; already assigned to {task.assigned_to}"
            )
            return False

        self.active_tasks[task.id] = task
        task.assigned_to = self.id
        task.error = None
        task.transition_to(TaskStatus.ASSIGNED)
        self.memory.store_episode(f"task:{task.id}", task)
        self.last_activity = datetime.now()
        if self.status != AgentStatus.BUSY:
            self.status = AgentStatus.ACTIVE
        self._notify_system("_on_task_assigned", self, task)
        logger.info(f"Task {task.id} assigned to agent {self.name}")
        return True

    def execute_task(self, task: Task) -> Any:
        """Execute an assigned task."""
        if not isinstance(task, Task):
            raise TypeError("task must be a Task instance")
        if task.id not in self.active_tasks:
            if task.status == TaskStatus.PENDING and task.assigned_to in (None, self.id):
                if not self.assign_task(task):
                    raise ValueError(f"Task {task.id} could not be assigned to agent {self.name}")
            else:
                raise ValueError(f"Task {task.id} is not assigned to agent {self.name}")
        if task.assigned_to != self.id:
            raise ValueError(f"Task {task.id} is assigned to a different agent")

        execution_started = datetime.now()
        success = False
        try:
            self.status = AgentStatus.BUSY
            self.last_activity = execution_started
            task.started_at = execution_started
            task.completed_at = None
            task.result = None
            task.error = None
            task.transition_to(TaskStatus.RUNNING)
            self._notify_system("_on_task_started", self, task)
            logger.info(f"Agent {self.name} executing task {task.id}")

            # Think phase
            reasoning = self.think(task.parameters)

            # Act phase
            result = self.act(reasoning)

            # Update task
            task.transition_to(TaskStatus.COMPLETED)
            task.result = result
            task.completed_at = datetime.now()
            success = True
            logger.info(f"Task {task.id} completed successfully")
            return result

        except Exception as e:
            task.transition_to(TaskStatus.FAILED)
            task.error = str(e)
            task.completed_at = datetime.now()
            self.status = AgentStatus.ERROR
            logger.exception(f"Task {task.id} failed on agent {self.name}")
            raise
        finally:
            if not task.completed_at:
                task.completed_at = datetime.now()
            self._finalize_task(task, success=success)

    def _update_metrics(self, task: Task, success: bool) -> None:
        """Update performance metrics."""
        if success:
            self.performance_metrics["tasks_completed"] += 1
        else:
            self.performance_metrics["tasks_failed"] += 1

        duration = task.duration_seconds()
        completed = self.performance_metrics["tasks_completed"]
        failed = self.performance_metrics["tasks_failed"]
        total = completed + failed
        if duration is not None and total > 0:
            self._total_task_time += duration
            self.performance_metrics["avg_task_time"] = self._total_task_time / total

        self.performance_metrics["success_rate"] = (
            completed / total if total > 0 else 0
        )

    def _finalize_task(self, task: Task, success: bool) -> None:
        """Clean up task state after execution and notify the system."""
        self.active_tasks.pop(task.id, None)
        if success and task not in self.completed_tasks:
            self.completed_tasks.append(task)
        if task not in self.task_history:
            self.task_history.append(task)

        self._update_metrics(task, success=success)
        self.last_activity = datetime.now()
        self._reset_status()
        self._notify_system("_on_task_completed", self, task, success=success)

    def _reset_status(self) -> None:
        """Reset the agent to a coherent post-execution state."""
        if self.status == AgentStatus.SUSPENDED:
            return
        self.status = AgentStatus.ACTIVE if self.active_tasks else AgentStatus.IDLE

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

    def __init__(self, name: str = "Orchestrator"):
        super().__init__(name, role=AgentRole.ORCHESTRATOR)
        self.managed_agents: Dict[str, BaseAgent] = {}
        self.task_queue: List[Task] = []

    def think(self, input_data: Any) -> Dict[str, Any]:
        """Analyze input and create execution plan."""
        return {
            "analysis": "Task requires orchestration",
            "priority": "high",
            "execution_strategy": "parallel"
        }

    def act(self, decision: Dict[str, Any]) -> Any:
        """Orchestrate agent actions based on decision."""
        logger.info(f"Orchestrator {self.name} executing strategy: {decision.get('execution_strategy')}")
        return {"status": "orchestration_complete"}

    def register_agent(self, agent: BaseAgent) -> bool:
        """Register an agent under this orchestrator."""
        if not isinstance(agent, BaseAgent):
            raise TypeError("agent must be a BaseAgent instance")
        if agent.id == self.id:
            logger.warning("Orchestrator cannot register itself as a managed agent")
            return False
        self.managed_agents[agent.id] = agent
        agent.parent_agent = self.id
        logger.info(f"Agent {agent.name} registered under orchestrator {self.name}")
        return True

    def distribute_task(self, task: Task, target_agent_id: Optional[str] = None) -> bool:
        """Distribute a task to appropriate agent."""
        if self.system and not self.system._prepare_task_for_assignment(task):
            return False
        if target_agent_id and target_agent_id in self.managed_agents:
            agent = self.managed_agents[target_agent_id]
            assigned = agent.assign_task(task)
            if assigned:
                logger.info(f"Task {task.id} distributed to target agent {agent.name}")
            return assigned

        # Auto-select best agent
        best_agent = self._select_best_agent(task)
        if best_agent:
            assigned = best_agent.assign_task(task)
            if assigned:
                logger.info(f"Task {task.id} distributed to selected agent {best_agent.name}")
            return assigned

        logger.warning(f"No suitable agent found for task {task.id}")
        return False

    def _select_best_agent(self, task: Task) -> Optional[BaseAgent]:
        """Select best agent for task based on capabilities."""
        available_agents = [
            a for a in self.managed_agents.values()
            if a.status != AgentStatus.SUSPENDED
        ]

        if not available_agents:
            return None

        # Prefer agents with fewer active tasks, then the least recently active agent
        # (older last_activity timestamp) to balance work while keeping the selection deterministic.
        return min(available_agents, key=lambda a: (len(a.active_tasks), a.last_activity, a.name))

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

    def __init__(self, name: str = "Executor"):
        super().__init__(name, role=AgentRole.EXECUTOR)
        self.execution_history: List[Dict[str, Any]] = []

    def think(self, input_data: Any) -> Dict[str, Any]:
        """Analyze task parameters and create execution plan."""
        return {
            "action": "execute",
            "parameters": input_data,
            "validation": True
        }

    def act(self, decision: Dict[str, Any]) -> Any:
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

    def __init__(self, name: str = "Analyzer"):
        super().__init__(name, role=AgentRole.ANALYZER)
        self.analysis_cache: Dict[str, Dict[str, Any]] = {}

    def think(self, input_data: Any) -> Dict[str, Any]:
        """Analyze input data and extract insights."""
        analysis = {
            "data_received": bool(input_data),
            "analysis_type": "comprehensive",
            "insights_generated": True
        }
        return analysis

    def act(self, decision: Dict[str, Any]) -> Any:
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

    def __init__(self, name: str = "Learner"):
        super().__init__(name, role=AgentRole.LEARNER)
        self.learned_patterns: Dict[str, Any] = {}
        self.learning_history: List[Dict[str, Any]] = []

    def think(self, input_data: Any) -> Dict[str, Any]:
        """Analyze input for learning opportunities."""
        return {
            "learning_mode": True,
            "input_analyzed": True,
            "patterns_identified": []
        }

    def act(self, decision: Dict[str, Any]) -> Any:
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
        logger.info(f"Learner {self.name} learned pattern: {pattern_id}")


# ============================================================================
# Agent System and Management
# ============================================================================

class AgentSystem:
    """
    Central management system for all agents.
    Handles agent lifecycle, orchestration, and inter-agent communication.
    """

    def __init__(self, name: str = "Ai-morphasis"):
        if not isinstance(name, str):
            raise TypeError("System name must be a string")
        if not name.strip():
            raise ValueError("System name cannot be empty or whitespace-only")
        self.name = name.strip()
        self.id = str(uuid.uuid4())
        self.created_at = datetime.now()

        self.orchestrator = OrchestratorAgent(f"{name}-Orchestrator")
        self.orchestrator.system = self
        self.agents: Dict[str, BaseAgent] = {self.orchestrator.id: self.orchestrator}
        self.global_task_queue: List[Task] = []
        self.completed_tasks: List[Task] = []
        self.failed_tasks: List[Task] = []
        self.active_tasks: Dict[str, Task] = {}
        self.task_history: List[Task] = []
        self.task_registry: Dict[str, Task] = {}
        self._task_sequence = 0

        self.system_metrics = {
            "total_agents": 1,
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "avg_task_duration": 0.0
        }
        self._total_task_duration = 0.0

        logger.info(f"Initialized Agent System: {self.name}")

    def add_agent(self, agent: BaseAgent) -> bool:
        """Add an agent to the system."""
        if not isinstance(agent, BaseAgent):
            raise TypeError("agent must be a BaseAgent instance")
        if agent.id in self.agents:
            logger.warning(f"Agent {agent.name} is already registered in system {self.name}")
            return False
        self.agents[agent.id] = agent
        agent.system = self
        if not self.orchestrator.register_agent(agent):
            self.agents.pop(agent.id, None)
            agent.system = None
            return False
        self.system_metrics["total_agents"] = len(self.agents)
        logger.info(f"Agent {agent.name} added to system")
        return True

    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from the system."""
        if agent_id == self.orchestrator.id:
            logger.warning("Cannot remove the system orchestrator")
            return False
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            if agent.active_tasks:
                logger.warning(
                    f"Cannot remove agent {agent.name}; {len(agent.active_tasks)} task(s) still active"
                )
                return False
            self.agents.pop(agent_id)
            self.orchestrator.managed_agents.pop(agent_id, None)
            agent.parent_agent = None
            agent.system = None
            self.system_metrics["total_agents"] = len(self.agents)
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
        metadata: Optional[Dict[str, Any]] = None
    ) -> Task:
        """Create a new task."""
        task = Task(
            description=description,
            parameters=parameters,
            priority=priority,
            dependencies=dependencies or [],
            metadata=metadata or {}
        )
        self._validate_dependencies(task.dependencies)
        self._assign_queue_order(task)
        self.task_registry[task.id] = task
        self._enqueue_task(task)
        self.system_metrics["total_tasks"] += 1
        logger.info(
            f"Task {task.id} created: {description} "
            f"(priority={task.priority.name}, dependencies={len(task.dependencies)})"
        )
        return task

    def submit_task(self, task: Task, agent_id: Optional[str] = None) -> bool:
        """Submit a task to an agent for execution."""
        if agent_id:
            agent = self.get_agent(agent_id)
            if agent:
                return agent.assign_task(task)
            logger.warning(f"Failed to submit task {task.id}; unknown agent {agent_id}")
        else:
            return self.orchestrator.distribute_task(task)

        logger.warning(f"Failed to submit task {task.id}")
        return False

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "system_name": self.name,
            "system_id": self.id,
            "created_at": self.created_at.isoformat(),
            "agents": {aid: agent.get_status() for aid, agent in self.agents.items()},
            "metrics": self.system_metrics,
            "pending_tasks": len(self.global_task_queue),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks)
        }

    def _assign_queue_order(self, task: Task) -> None:
        """Attach a stable insertion order for deterministic queue sorting."""
        if "queue_order" not in task.metadata:
            task.metadata["queue_order"] = self._task_sequence
            self._task_sequence += 1

    def _validate_dependencies(self, dependencies: List[str]) -> None:
        """Ensure dependencies reference known tasks."""
        unknown_dependencies = [
            dependency_id for dependency_id in dependencies
            if dependency_id not in self.task_registry
        ]
        if unknown_dependencies:
            raise ValueError(f"Unknown task dependencies: {unknown_dependencies}")

    def _sort_queue(self, queue: List[Task]) -> None:
        """Sort a task queue deterministically by priority, creation time, and id."""
        queue.sort(
            key=lambda item: (
                -item.priority.value,
                item.created_at,
                item.metadata.get("queue_order", 0)
            )
        )

    def _enqueue_task(self, task: Task) -> None:
        """Add a task to system-managed queues if it is pending."""
        if task.status != TaskStatus.PENDING:
            return
        if task not in self.global_task_queue:
            self.global_task_queue.append(task)
            self._sort_queue(self.global_task_queue)
        if task not in self.orchestrator.task_queue:
            self.orchestrator.task_queue.append(task)
            self._sort_queue(self.orchestrator.task_queue)

    def _remove_from_queues(self, task_id: str) -> None:
        """Remove a task from pending queues."""
        self.global_task_queue = [task for task in self.global_task_queue if task.id != task_id]
        self.orchestrator.task_queue = [
            task for task in self.orchestrator.task_queue if task.id != task_id
        ]

    def _register_external_task(self, task: Task) -> bool:
        """Register a task that was constructed outside the system."""
        existing_task = self.task_registry.get(task.id)
        if existing_task and existing_task is not task:
            logger.warning(f"Duplicate task id detected for task {task.id}")
            return False
        if existing_task is None:
            try:
                self._validate_dependencies(task.dependencies)
            except ValueError as exc:
                logger.warning(f"Task {task.id} has invalid dependencies: {exc}")
                return False
            self._assign_queue_order(task)
            self.task_registry[task.id] = task
            self.system_metrics["total_tasks"] += 1
            self._enqueue_task(task)
            logger.info(f"Registered external task {task.id} with system {self.name}")
        return True

    def _prepare_task_for_assignment(self, task: Task) -> bool:
        """Validate task state before assignment to an agent."""
        if not isinstance(task, Task):
            logger.warning("Ignoring non-Task submission")
            return False
        if not self._register_external_task(task):
            return False
        if task.is_terminal():
            logger.warning(f"Cannot submit terminal task {task.id}")
            return False
        if task.id in self.active_tasks or task.status in {TaskStatus.ASSIGNED, TaskStatus.RUNNING}:
            logger.warning(f"Task {task.id} is already in progress")
            return False

        for dependency_id in task.dependencies:
            dependency = self.task_registry.get(dependency_id)
            if dependency is None:
                logger.error(f"Task {task.id} lost dependency reference {dependency_id}")
                return False
            if dependency.status != TaskStatus.COMPLETED:
                logger.info(
                    f"Task {task.id} is waiting on dependency {dependency_id} "
                    f"(status={dependency.status.value})"
                )
                return False

        self._enqueue_task(task)
        return True

    def _on_task_assigned(self, agent: BaseAgent, task: Task) -> None:
        """Record that a task has been assigned to an agent."""
        self.active_tasks[task.id] = task
        self._remove_from_queues(task.id)
        logger.info(f"Task {task.id} assigned to agent {agent.name} at system level")

    def _on_task_started(self, agent: BaseAgent, task: Task) -> None:
        """Record task execution start."""
        self.active_tasks[task.id] = task
        logger.info(f"Task {task.id} started by agent {agent.name}")

    def _on_task_completed(self, agent: BaseAgent, task: Task, success: bool) -> None:
        """Record task completion or failure and update metrics."""
        self.active_tasks.pop(task.id, None)
        self._remove_from_queues(task.id)

        if success:
            if task not in self.completed_tasks:
                self.completed_tasks.append(task)
            self.system_metrics["successful_tasks"] += 1
        else:
            if task not in self.failed_tasks:
                self.failed_tasks.append(task)
            self.system_metrics["failed_tasks"] += 1

        if task not in self.task_history:
            self.task_history.append(task)

        total_terminal_tasks = (
            self.system_metrics["successful_tasks"] + self.system_metrics["failed_tasks"]
        )
        duration = task.duration_seconds()
        if duration is not None and total_terminal_tasks > 0:
            self._total_task_duration += duration
            self.system_metrics["avg_task_duration"] = (
                self._total_task_duration / total_terminal_tasks
            )

        logger.info(
            f"Task {task.id} finished on agent {agent.name} "
            f"(success={success}, duration={duration})"
        )

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

    _agent_templates = {
        "executor": ExecutorAgent,
        "analyzer": AnalyzerAgent,
        "learner": LearnerAgent,
        "orchestrator": OrchestratorAgent
    }

    @classmethod
    def create_agent(cls, agent_type: str, name: str) -> Optional[BaseAgent]:
        """Create an agent from template."""
        if not isinstance(agent_type, str):
            raise TypeError("agent_type must be a string")
        if not agent_type.strip():
            raise ValueError("agent_type cannot be empty or whitespace-only")
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if not name.strip():
            raise ValueError("name cannot be empty or whitespace-only")
        agent_class = cls._agent_templates.get(agent_type.strip().lower())
        if agent_class:
            return agent_class(name)
        logger.error(f"Unknown agent type: {agent_type}")
        return None

    @classmethod
    def create_team(cls, team_config: Dict[str, int]) -> AgentSystem:
        """Create a complete agent team from configuration."""
        if not isinstance(team_config, dict):
            raise TypeError("team_config must be a dictionary")
        system = AgentSystem("Ai-morphasis-Team")

        for agent_type, count in team_config.items():
            if not isinstance(count, int) or count < 0:
                raise ValueError("team_config counts must be non-negative integers")
            for i in range(count):
                agent = cls.create_agent(agent_type, f"{agent_type.title()}-{i+1}")
                if agent:
                    system.add_agent(agent)

        logger.info(f"Agent team created with config: {team_config}")
        return system


# ============================================================================
# Example Usage
# ============================================================================

def example_usage():
    """Demonstrate the super agentic agents framework."""
    # Create agent system
    system = AgentSystem("Ai-morphasis-2.0")

    # Create and add agents
    executor = ExecutorAgent("TaskExecutor-1")
    analyzer = AnalyzerAgent("DataAnalyzer-1")
    learner = LearnerAgent("SystemLearner-1")

    system.add_agent(executor)
    system.add_agent(analyzer)
    system.add_agent(learner)

    # Register capabilities
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

    # Create and submit tasks
    task1 = system.create_task(
        description="Analyze performance metrics",
        parameters={"metric_type": "performance", "duration": "24h"}
    )

    system.submit_task(task1, executor.id)

    # Print system status
    print("\n" + "="*60)
    print("AGENT SYSTEM STATUS")
    print("="*60)
    print(system.to_json())


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    example_usage()
