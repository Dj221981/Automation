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
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)
BASE_RETRY_BACKOFF_SECONDS = 1


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


@dataclass(slots=True)
class AgentCapability:
    """Represents a capability an agent can perform."""

    name: str
    description: str
    func: Optional[Callable[..., Any]] = None
    confidence_score: float = 1.0
    requires_resources: List[str] = field(default_factory=list)
    version: str = "1.0.0"

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

    def store_episode(self, key: str, value: Any) -> None:
        if len(self.episodic_memory) >= self.max_episodes:
            oldest_key = next(iter(self.episodic_memory))
            del self.episodic_memory[oldest_key]
        self.episodic_memory[key] = {"value": value, "timestamp": datetime.now()}
        self.last_accessed = datetime.now()

    def store_semantic(self, key: str, value: Any) -> None:
        self.semantic_memory[key] = {
            "value": value,
            "timestamp": datetime.now(),
            "access_count": 0,
        }
        self.last_accessed = datetime.now()

    def retrieve(self, key: str, memory_type: str = "auto") -> Optional[Any]:
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
    started_at: Optional[datetime] = None
    last_attempt_at: Optional[datetime] = None
    next_retry_at: Optional[datetime] = None
    retry_count: int = 0
    # max_retries is the number of retries after the initial execution attempt.
    # max_retries=0 means no retries are allowed (one total attempt).
    max_retries: int = 0
    result: Any = None
    error: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    _ALLOWED_TRANSITIONS = {
        TaskStatus.PENDING: {TaskStatus.ASSIGNED, TaskStatus.CANCELLED},
        TaskStatus.ASSIGNED: {TaskStatus.RUNNING, TaskStatus.CANCELLED},
        TaskStatus.RUNNING: {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED},
        TaskStatus.FAILED: {TaskStatus.PENDING, TaskStatus.CANCELLED},
        TaskStatus.COMPLETED: set(),
        TaskStatus.CANCELLED: set(),
    }

    @classmethod
    def can_transition(cls, current: TaskStatus, new: TaskStatus) -> bool:
        return new in cls._ALLOWED_TRANSITIONS.get(current, set())

    def transition_to(self, new_status: TaskStatus) -> None:
        if self.status == new_status:
            return
        if not self.can_transition(self.status, new_status):
            raise ValueError(
                f"Invalid task status transition for task {self.id}: {self.status.value} -> {new_status.value}"
            )
        self.status = new_status

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "priority": self.priority.name,
            "assigned_to": self.assigned_to,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_attempt_at": self.last_attempt_at.isoformat() if self.last_attempt_at else None,
            "next_retry_at": self.next_retry_at.isoformat() if self.next_retry_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "result": self.result,
            "error": self.error,
            "parameters": self.parameters,
            "dependencies": self.dependencies,
            "metadata": self.metadata,
        }


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(self, name: str, role: AgentRole = AgentRole.EXECUTOR, max_capabilities: int = 50):
        self.id = str(uuid.uuid4())
        self.name = name
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
        task.transition_to(TaskStatus.ASSIGNED)
        self.active_tasks[task.id] = task
        task.assigned_to = self.id
        self.memory.store_episode(f"task:{task.id}", task)
        self._touch()
        logger.info("Task %s assigned to agent %s", task.id, self.name)
        return True

    def execute_task(self, task: Task) -> Any:
        start_time = datetime.now()
        self.status = AgentStatus.BUSY
        task.last_attempt_at = start_time
        if task.started_at is None:
            task.started_at = start_time
        task.transition_to(TaskStatus.RUNNING)
        self._touch()
        logger.info("Agent %s executing task %s", self.name, task.id)

        try:
            reasoning = self.think(task.parameters)
            result = self.act(reasoning)
            task.transition_to(TaskStatus.COMPLETED)
            task.result = result
            task.completed_at = datetime.now()
            self._update_metrics(task, success=True, start_time=start_time)
            self.completed_tasks.append(task)
            self.task_history.append(task)
            logger.info("Task %s completed successfully", task.id)
            return result
        except Exception as exc:
            task.transition_to(TaskStatus.FAILED)
            task.error = str(exc)
            task.completed_at = datetime.now()
            self._update_metrics(task, success=False, start_time=start_time)
            logger.exception("Task %s failed", task.id)
            raise
        finally:
            self.active_tasks.pop(task.id, None)
            self.status = AgentStatus.IDLE if task.status == TaskStatus.COMPLETED else AgentStatus.ERROR
            self._touch()

    def _update_metrics(self, task: Task, success: bool, start_time: datetime) -> None:
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
        self.managed_agents[agent.id] = agent
        agent.parent_agent = self.id
        self._touch()
        logger.info("Agent %s registered under orchestrator %s", agent.name, self.name)
        return True

    def distribute_task(self, task: Task, target_agent_id: Optional[str] = None) -> bool:
        if target_agent_id and target_agent_id in self.managed_agents:
            return self.managed_agents[target_agent_id].assign_task(task)
        best_agent = self._select_best_agent(task)
        if best_agent:
            return best_agent.assign_task(task)
        logger.warning("No suitable agent found for task %s", task.id)
        return False

    def _select_best_agent(self, task: Task) -> Optional[BaseAgent]:
        available_agents = [a for a in self.managed_agents.values() if a.status != AgentStatus.SUSPENDED]
        if not available_agents:
            return None

        def score(agent: BaseAgent) -> tuple[int, int, float]:
            capability_match = 0
            for capability in agent.capabilities.values():
                if task.description.lower().find(capability.name.lower()) != -1:
                    capability_match += 1
            return (capability_match, -len(agent.active_tasks), int(agent.performance_metrics["success_rate"] * 100))

        return max(available_agents, key=score)

    def get_system_status(self) -> Dict[str, Any]:
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
            "decision": decision,
            "result": "executed",
        })
        return {"execution": "successful", "parameters_processed": params}


class AnalyzerAgent(BaseAgent):
    def __init__(self, name: str = "Analyzer"):
        super().__init__(name, role=AgentRole.ANALYZER)
        self.analysis_cache: Dict[str, Dict[str, Any]] = {}

    def think(self, input_data: Any) -> Dict[str, Any]:
        return {"data_received": bool(input_data), "analysis_type": "comprehensive", "insights_generated": True}

    def act(self, decision: Dict[str, Any]) -> Any:
        return {"analysis_complete": True, "insights": decision, "timestamp": datetime.now().isoformat()}


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
            "decision": decision,
            "patterns_learned": len(self.learned_patterns),
        })
        return {"learning": "in_progress", "patterns": self.learned_patterns}

    def learn_from_experience(self, experience: Dict[str, Any]) -> None:
        pattern_id = str(uuid.uuid4())
        self.learned_patterns[pattern_id] = {
            "experience": experience,
            "learned_at": datetime.now().isoformat(),
            "confidence": 0.5,
        }
        self.memory.store_semantic(f"pattern:{pattern_id}", self.learned_patterns[pattern_id])
        self._touch()
        logger.info("Learner %s learned pattern: %s", self.name, pattern_id)


class AgentSystem:
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
            "avg_task_duration": 0.0,
        }
        self._completed_task_count = 0
        self._completed_task_duration_total = 0.0
        logger.info("Initialized Agent System: %s", self.name)

    def add_agent(self, agent: BaseAgent) -> bool:
        self.agents[agent.id] = agent
        self.orchestrator.register_agent(agent)
        self.system_metrics["total_agents"] += 1
        logger.info("Agent %s added to system", agent.name)
        return True

    def remove_agent(self, agent_id: str) -> bool:
        if agent_id in self.agents:
            agent = self.agents.pop(agent_id)
            self.orchestrator.managed_agents.pop(agent_id, None)
            self.system_metrics["total_agents"] -= 1
            logger.info("Agent %s removed from system", agent.name)
            return True
        return False

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        return self.agents.get(agent_id)

    def create_task(self, description: str, parameters: Dict[str, Any], priority: TaskPriority = TaskPriority.NORMAL) -> Task:
        task = Task(description=description, parameters=parameters, priority=priority)
        self.global_task_queue.append(task)
        self.system_metrics["total_tasks"] += 1
        logger.info("Task %s created: %s", task.id, description)
        return task

    def submit_task(self, task: Task, agent_id: Optional[str] = None) -> bool:
        if agent_id:
            agent = self.get_agent(agent_id)
            if agent:
                return agent.assign_task(task)
        else:
            return self.orchestrator.distribute_task(task)
        logger.warning("Failed to submit task %s", task.id)
        return False

    def _remove_from_pending_queue(self, task: Task) -> None:
        self.global_task_queue = [queued for queued in self.global_task_queue if queued.id != task.id]

    def _record_terminal_task_outcome(self, task: Task, success: bool) -> None:
        if task.started_at is None and task.last_attempt_at is not None:
            logger.warning("Task %s missing started_at; falling back to last_attempt_at for duration", task.id)
        duration_start = task.started_at if task.started_at is not None else task.last_attempt_at
        if task.completed_at is not None and duration_start is not None:
            self._completed_task_duration_total += (task.completed_at - duration_start).total_seconds()
            self._completed_task_count += 1
            self.system_metrics["avg_task_duration"] = self._completed_task_duration_total / self._completed_task_count
        if success:
            self.system_metrics["successful_tasks"] += 1
        else:
            self.system_metrics["failed_tasks"] += 1

    def process_task(self, task_id: str, agent_id: Optional[str] = None) -> bool:
        task = next((queued for queued in self.global_task_queue if queued.id == task_id), None)
        if task is None:
            logger.warning("Task %s not found in pending queue", task_id)
            return False

        assigned_agent: Optional[BaseAgent] = None
        if agent_id:
            assigned_agent = self.get_agent(agent_id)
            if assigned_agent is None:
                raise ValueError(f"Agent '{agent_id}' not found")

        if task.assigned_to:
            assigned_agent = self.get_agent(task.assigned_to)
        elif assigned_agent:
            assigned_agent.assign_task(task)
        else:
            assigned = self.orchestrator.distribute_task(task)
            if assigned and task.assigned_to:
                assigned_agent = self.get_agent(task.assigned_to)

        if assigned_agent is None:
            logger.warning("Unable to assign task %s", task.id)
            return False

        try:
            assigned_agent.execute_task(task)
        except Exception:
            logger.exception("Task %s execution failed on agent %s", task.id, assigned_agent.id)
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                backoff_seconds = BASE_RETRY_BACKOFF_SECONDS * max(1, task.retry_count)
                task.next_retry_at = datetime.now() + timedelta(seconds=backoff_seconds)
                task.metadata["retry_backoff_seconds"] = backoff_seconds
                task.transition_to(TaskStatus.PENDING)
                task.assigned_to = None
                task.completed_at = None
                task.error = None
                return False

            self._remove_from_pending_queue(task)
            self.completed_tasks.append(task)
            self._record_terminal_task_outcome(task, success=False)
            return False

        self._remove_from_pending_queue(task)
        self.completed_tasks.append(task)
        self._record_terminal_task_outcome(task, success=True)
        return True

    def process_pending_tasks(self, max_tasks: Optional[int] = None) -> Dict[str, Any]:
        processed = 0
        success = 0
        terminal_failures = 0
        task_ids = [task.id for task in self.global_task_queue]

        for task_id in task_ids:
            if max_tasks is not None and processed >= max_tasks:
                break
            result = self.process_task(task_id)
            processed += 1
            if result:
                success += 1
            else:
                task = next((completed for completed in self.completed_tasks if completed.id == task_id), None)
                if task and task.status == TaskStatus.FAILED:
                    terminal_failures += 1

        return {
            "processed": processed,
            "successful": success,
            "failed": terminal_failures,
            "remaining_pending": len(self.global_task_queue),
        }

    def get_system_status(self) -> Dict[str, Any]:
        return {
            "system_name": self.name,
            "system_id": self.id,
            "created_at": self.created_at.isoformat(),
            "agents": {aid: agent.get_status() for aid, agent in self.agents.items()},
            "metrics": self.system_metrics,
            "pending_tasks": len(self.global_task_queue),
            "completed_tasks": len(self.completed_tasks),
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
    def create_team(cls, team_config: Dict[str, int]) -> AgentSystem:
        system = AgentSystem("Ai-morphasis-Team")
        for agent_type, count in team_config.items():
            for i in range(count):
                agent = cls.create_agent(agent_type, f"{agent_type.title()}-{i + 1}")
                if agent:
                    system.add_agent(agent)
        logger.info("Agent team created with config: %s", team_config)
        return system


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

    print("\n" + "=" * 60)
    print("AGENT SYSTEM STATUS")
    print("=" * 60)
    print(system.to_json())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    example_usage()
