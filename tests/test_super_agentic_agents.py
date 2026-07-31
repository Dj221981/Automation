"""
Tests for the production-ready super_agentic_agents framework.

Covers:
- Agent creation and capability registration
- Task creation, assignment, and execution
- AgentConfig and TaskConfig validation
- AgentSystem.get_system_status() output structure
- Task persistence lifecycle (create, assign, execute, fail, recover, requeue, cancel)
- Async execute_task wrapper on BaseAgent
- StructuredLogger
"""

import asyncio
import uuid
from datetime import datetime
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.super_agentic_agents import (
    AgentCapability,
    AgentConfig,
    AgentMemory,
    AgentRole,
    AgentStatus,
    AgentSystem,
    AnalyzerAgent,
    BaseAgent,
    ExecutorAgent,
    LearnerAgent,
    OrchestratorAgent,
    StructuredLogger,
    Task,
    TaskConfig,
    TaskPriority,
    TaskStatus,
    AgentFactory,
)
from src.agents.task_store import InMemoryTaskStore, StoredTask


# ============================================================================
# Helpers
# ============================================================================

def make_task(description: str = "Test task", priority: TaskPriority = TaskPriority.NORMAL) -> Task:
    return Task(description=description, priority=priority, parameters={"key": "value"})


# ============================================================================
# AgentConfig validation
# ============================================================================

class TestAgentConfig:
    def test_valid_config(self):
        cfg = AgentConfig(name="MyAgent", max_capabilities=10, max_retries=3)
        assert cfg.name == "MyAgent"
        assert cfg.max_capabilities == 10
        assert cfg.max_retries == 3

    def test_name_too_long(self):
        with pytest.raises((ValueError, Exception)):
            AgentConfig(name="x" * 101)

    def test_empty_name(self):
        with pytest.raises((ValueError, Exception)):
            AgentConfig(name="")

    def test_max_capabilities_out_of_range(self):
        with pytest.raises((ValueError, Exception)):
            AgentConfig(name="A", max_capabilities=0)
        with pytest.raises((ValueError, Exception)):
            AgentConfig(name="A", max_capabilities=201)

    def test_max_retries_out_of_range(self):
        with pytest.raises((ValueError, Exception)):
            AgentConfig(name="A", max_retries=-1)
        with pytest.raises((ValueError, Exception)):
            AgentConfig(name="A", max_retries=11)


# ============================================================================
# TaskConfig validation
# ============================================================================

class TestTaskConfig:
    def test_valid_task_config(self):
        cfg = TaskConfig(
            description="Do something",
            priority=TaskPriority.HIGH,
            parameters={"a": 1},
            dependencies=[],
        )
        assert cfg.description == "Do something"

    def test_empty_description(self):
        with pytest.raises((ValueError, Exception)):
            TaskConfig(description="")

    def test_invalid_dependency_uuid(self):
        with pytest.raises((ValueError, Exception)):
            TaskConfig(description="Task", dependencies=["not-a-uuid"])

    def test_valid_uuid_dependency(self):
        valid_uuid = str(uuid.uuid4())
        cfg = TaskConfig(description="Task", dependencies=[valid_uuid])
        assert cfg.dependencies == [valid_uuid]


# ============================================================================
# Agent creation and capability registration
# ============================================================================

class TestAgentCreation:
    def test_executor_agent_defaults(self):
        agent = ExecutorAgent("Exec-1")
        assert agent.name == "Exec-1"
        assert agent.role == AgentRole.EXECUTOR
        assert agent.status == AgentStatus.IDLE

    def test_analyzer_agent_defaults(self):
        agent = AnalyzerAgent("Analyzer-1")
        assert agent.role == AgentRole.ANALYZER

    def test_learner_agent_defaults(self):
        agent = LearnerAgent("Learner-1")
        assert agent.role == AgentRole.LEARNER

    def test_orchestrator_agent_defaults(self):
        agent = OrchestratorAgent("Orch-1")
        assert agent.role == AgentRole.ORCHESTRATOR

    def test_register_capability(self):
        agent = ExecutorAgent("Exec-2")
        cap = AgentCapability(name="my_cap", description="test cap", confidence_score=0.9)
        result = agent.register_capability(cap)
        assert result is True
        assert "my_cap" in agent.list_capabilities()

    def test_max_capabilities_limit(self):
        agent = ExecutorAgent("Exec-3", max_capabilities=2)
        for i in range(2):
            agent.register_capability(AgentCapability(name=f"cap_{i}", description="cap"))
        # Third registration should fail
        result = agent.register_capability(AgentCapability(name="cap_3", description="cap"))
        assert result is False

    def test_agent_repr(self):
        agent = ExecutorAgent("Exec-repr")
        assert "ExecutorAgent" in repr(agent)
        assert "Exec-repr" in repr(agent)

    def test_get_status_keys(self):
        agent = ExecutorAgent("Exec-status")
        status = agent.get_status()
        for key in ("id", "name", "role", "status", "capabilities", "performance"):
            assert key in status


# ============================================================================
# Task creation and assignment
# ============================================================================

class TestTaskManagement:
    def test_create_task(self):
        system = AgentSystem("TestSys")
        task = system.create_task("Test", parameters={"x": 1})
        assert task.description == "Test"
        assert task.id  # has a UUID

    def test_task_assignment(self):
        agent = ExecutorAgent("Exec-assign")
        task = make_task()
        result = agent.assign_task(task)
        assert result is True
        assert task.id in agent.active_tasks
        assert task.assigned_to == agent.id

    def test_task_to_dict(self):
        task = make_task()
        d = task.to_dict()
        assert d["status"] == "pending"
        assert "id" in d


# ============================================================================
# Async task execution (via BaseAgent.execute_task wrapper)
# ============================================================================

@pytest.mark.asyncio
class TestAsyncExecution:
    async def test_execute_task_success(self):
        agent = ExecutorAgent("Exec-async")
        task = make_task("Async task")
        agent.assign_task(task)
        result = await agent.execute_task(task)
        assert result["execution"] == "successful"
        assert task.status == TaskStatus.COMPLETED
        assert agent.status == AgentStatus.IDLE

    async def test_execute_task_sets_duration(self):
        agent = ExecutorAgent("Exec-duration")
        task = make_task()
        agent.assign_task(task)
        await agent.execute_task(task)
        assert "duration_ms" in task.metadata
        assert task.metadata["duration_ms"] >= 0

    async def test_analyzer_agent_execution(self):
        agent = AnalyzerAgent("Analyze-async")
        task = make_task("Analyze data")
        agent.assign_task(task)
        result = await agent.execute_task(task)
        assert result["analysis_complete"] is True

    async def test_learner_agent_execution(self):
        agent = LearnerAgent("Learn-async")
        task = make_task("Learn something")
        agent.assign_task(task)
        result = await agent.execute_task(task)
        assert "learning" in result

    async def test_orchestrator_agent_execution(self):
        agent = OrchestratorAgent("Orch-async")
        task = make_task("Orchestrate")
        agent.assign_task(task)
        result = await agent.execute_task(task)
        assert result["status"] == "orchestration_complete"

    async def test_execute_task_failure(self):
        class FailingAgent(ExecutorAgent):
            def act(self, decision: Dict[str, Any]) -> Any:
                raise RuntimeError("Execution failed")

        agent = FailingAgent("Fail-async")
        task = make_task("Failing task")
        agent.assign_task(task)

        with pytest.raises(RuntimeError, match="Execution failed"):
            await agent.execute_task(task)

        assert task.status == TaskStatus.FAILED
        assert "Execution failed" in (task.error or "")


# ============================================================================
# AgentMemory
# ============================================================================

class TestAgentMemory:
    def test_store_and_retrieve_episode(self):
        mem = AgentMemory(agent_id="test-agent")
        mem.store_episode("task:1", {"result": "ok"})
        assert mem.retrieve("task:1") == {"result": "ok"}

    def test_store_and_retrieve_semantic(self):
        mem = AgentMemory(agent_id="test-agent")
        mem.store_semantic("config:key", 42)
        assert mem.retrieve("config:key", "semantic") == 42

    def test_auto_retrieval_prefers_episodic(self):
        mem = AgentMemory(agent_id="test-agent")
        mem.store_episode("key", "episodic_value")
        mem.store_semantic("key", "semantic_value")
        result = mem.retrieve("key", "auto")
        assert result == "episodic_value"

    def test_missing_key_returns_none(self):
        mem = AgentMemory(agent_id="test-agent")
        assert mem.retrieve("nonexistent") is None

    def test_max_episodes_evicts_oldest(self):
        mem = AgentMemory(agent_id="test-agent", max_episodes=3)
        for i in range(3):
            mem.store_episode(f"k{i}", f"v{i}")
        mem.store_episode("k3", "v3")
        assert mem.retrieve("k0", "episodic") is None
        assert mem.retrieve("k3", "episodic") == "v3"


# ============================================================================
# StructuredLogger
# ============================================================================

class TestStructuredLogger:
    def test_logger_creation(self):
        slog = StructuredLogger("test_module", agent_id="abc", agent_name="TestAgent")
        assert slog._base_extra["agent_id"] == "abc"
        assert slog._base_extra["agent_name"] == "TestAgent"

    def test_extra_includes_task_id(self):
        slog = StructuredLogger("test_module")
        extra = slog._extra(task_id="task-123")
        assert extra["structured"]["task_id"] == "task-123"

    def test_log_methods_callable(self):
        import logging
        slog = StructuredLogger("test_module")
        with patch.object(slog._logger, "info") as mock_info:
            slog.info("Test message")
            mock_info.assert_called_once()


# ============================================================================
# AgentSystem status
# ============================================================================

class TestSystemStatus:
    def test_system_status_keys_present(self):
        system = AgentSystem("StatusSys")
        status = system.get_system_status()
        assert "system_name" in status
        assert "agents" in status
        assert "metrics" in status

    def test_fresh_system_has_one_orchestrator(self):
        system = AgentSystem("FreshSys")
        assert len(system.agents) == 1
        assert system.orchestrator.id in system.agents


# ============================================================================
# Task persistence lifecycle tests (from main)
# ============================================================================

class FailingExecutorAgent(ExecutorAgent):
    def act(self, decision: Dict[str, Any]) -> Any:
        raise RuntimeError("boom")


def test_inmemory_task_store_returns_defensive_copies():
    store = InMemoryTaskStore()
    task = StoredTask(id="task-1", description="Task 1", priority="NORMAL")

    store.create_task(task)
    loaded = store.get_task("task-1")
    assert loaded is not None

    loaded.metadata["changed"] = True
    loaded.dependencies.append("dep-1")

    reloaded = store.get_task("task-1")
    assert reloaded is not None
    assert reloaded.metadata == {}
    assert reloaded.dependencies == []


def test_agent_system_persists_task_lifecycle_to_completion():
    store = InMemoryTaskStore()
    system = AgentSystem("TestSystem", task_store=store)
    agent = ExecutorAgent("Executor-1")
    system.add_agent(agent)

    task = system.create_task("Process payload", {"value": 1}, priority=TaskPriority.HIGH)
    stored_created = store.get_task(task.id)
    assert stored_created is not None
    assert stored_created.status == "PENDING"

    assert system.submit_task(task, agent.id) is True
    stored_assigned = store.get_task(task.id)
    assert stored_assigned is not None
    assert stored_assigned.status == "ASSIGNED"
    assert stored_assigned.assigned_to == agent.id
    assert stored_assigned.metadata.get("claimed_by") == agent.id

    result = system.execute_task(task.id, agent.id)
    assert result["execution"] == "successful"

    stored_completed = store.get_task(task.id)
    assert stored_completed is not None
    assert stored_completed.status == "COMPLETED"
    assert stored_completed.assigned_to == agent.id
    assert stored_completed.metadata.get("claimed_by") == agent.id
    assert stored_completed.completed_at is not None


def test_agent_system_persists_failed_execution():
    store = InMemoryTaskStore()
    system = AgentSystem("FailureSystem", task_store=store)
    agent = FailingExecutorAgent("FailingExecutor-1")
    system.add_agent(agent)

    task = system.create_task("Explode", {"value": 1})
    assert system.submit_task(task, agent.id) is True

    try:
        system.execute_task(task.id, agent.id)
        assert False, "Expected RuntimeError"
    except RuntimeError as exc:
        assert str(exc) == "boom"

    stored_failed = store.get_task(task.id)
    assert stored_failed is not None
    assert stored_failed.status == "FAILED"
    assert stored_failed.error == "boom"
    assert stored_failed.assigned_to == agent.id


def test_recover_incomplete_tasks_resets_running_and_assigned_tasks():
    store = InMemoryTaskStore()
    system = AgentSystem("RecoverySystem", task_store=store)
    agent = ExecutorAgent("Executor-1")
    system.add_agent(agent)

    task = system.create_task("Recover me", {"value": 1})
    assert system.submit_task(task, agent.id) is True

    stored_assigned = store.get_task(task.id)
    assert stored_assigned is not None
    stored_assigned.status = "RUNNING"
    store.update_task(stored_assigned)

    recovered = system.recover_incomplete_tasks()
    assert recovered == 1

    reloaded = store.get_task(task.id)
    assert reloaded is not None
    assert reloaded.status == "PENDING"
    assert reloaded.assigned_to is None
    assert reloaded.metadata.get("claimed_by") is None
    assert task.id in system._task_index
    assert agent.active_tasks == {}


def test_requeue_task_moves_failed_task_back_to_pending():
    store = InMemoryTaskStore()
    system = AgentSystem("RequeueSystem", task_store=store)
    agent = FailingExecutorAgent("FailingExecutor-1")
    system.add_agent(agent)

    task = system.create_task("Retry me", {"value": 2})
    assert system.submit_task(task, agent.id) is True

    try:
        system.execute_task(task.id, agent.id)
    except RuntimeError:
        pass

    requeued = system.requeue_task(task.id)
    assert requeued.status == TaskStatus.PENDING

    stored = store.get_task(task.id)
    assert stored is not None
    assert stored.status == "PENDING"
    assert stored.assigned_to is None
    assert stored.error is None
    assert task.id in system._task_index


def test_claim_mismatch_blocks_execution():
    store = InMemoryTaskStore()
    system = AgentSystem("ClaimSystem", task_store=store)
    agent_one = ExecutorAgent("Executor-1")
    agent_two = ExecutorAgent("Executor-2")
    system.add_agent(agent_one)
    system.add_agent(agent_two)

    task = system.create_task("Claimed task", {"value": 3})
    assert system.submit_task(task, agent_one.id) is True

    # Load the persisted task (which has claimed_by = agent_one.id set)
    claimed_task = system.load_task(task.id)
    assert claimed_task is not None
    agent_two.active_tasks[claimed_task.id] = claimed_task

    try:
        system.execute_task(claimed_task.id, agent_two.id)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "claimed by" in str(exc)


def test_remove_agent_guards_and_idle_removal():
    store = InMemoryTaskStore()
    system = AgentSystem("GuardSystem", task_store=store)
    busy_agent = ExecutorAgent("BusyExecutor")
    idle_agent = ExecutorAgent("IdleExecutor")
    system.add_agent(busy_agent)
    system.add_agent(idle_agent)

    task = system.create_task("Active task", {"value": 4})
    assert system.submit_task(task, busy_agent.id) is True

    assert system.remove_agent(system.orchestrator.id) is False
    assert system.remove_agent(busy_agent.id) is False
    assert system.remove_agent(idle_agent.id) is True


def test_queue_dedup_and_cancel_task():
    store = InMemoryTaskStore()
    system = AgentSystem("QueueSystem", task_store=store)
    agent = ExecutorAgent("Executor-1")
    system.add_agent(agent)

    task = system.create_task("Queue task", {"value": 5})
    # task already enqueued by create_task; calling _enqueue_if_missing again is a no-op
    system._enqueue_if_missing(task)
    assert task.id in system._task_index

    assert system.submit_task(task, agent.id) is True
    assert task.id not in system._task_index

    requeued = system.requeue_task(task.id)
    assert requeued.status == TaskStatus.PENDING
    assert task.id in system._task_index

    cancelled = system.cancel_task(task.id, reason="no longer needed")
    assert cancelled.status == TaskStatus.CANCELLED

    stored = store.get_task(task.id)
    assert stored is not None
    assert stored.status == "CANCELLED"
    assert stored.error == "no longer needed"
    assert stored.assigned_to is None
    assert stored.metadata.get("claimed_by") is None
    assert task.id not in system._task_index


# ============================================================================
# PR #2 Hardening tests (adapted to main's execute_task architecture)
# ============================================================================

class FailingActExecutorAgent(ExecutorAgent):
    """ExecutorAgent whose act() always raises to simulate failures."""

    def act(self, decision: Dict[str, Any]) -> Any:
        raise RuntimeError("simulated failure")


def test_task_lifecycle_methods():
    """Task.is_terminal, transition_to, duration_seconds, and started_at work correctly."""
    task = make_task("Lifecycle check")
    assert not task.is_terminal()
    assert task.started_at is None
    assert task.status_value == "pending"
    assert task.duration_seconds() is None

    task.transition_to(TaskStatus.ASSIGNED)
    assert task.status == TaskStatus.ASSIGNED
    assert not task.is_terminal()

    task.transition_to(TaskStatus.RUNNING)
    assert task.status == TaskStatus.RUNNING

    from datetime import datetime
    task.started_at = datetime.now()
    task.transition_to(TaskStatus.COMPLETED)
    task.completed_at = datetime.now()
    assert task.is_terminal()
    assert task.duration_seconds() is not None
    assert task.duration_seconds() >= 0.0

    # Status history recorded
    history = task.metadata.get("status_history", [])
    assert len(history) >= 2
    assert history[-1]["to"] == TaskStatus.COMPLETED.value


def test_task_invalid_transition_raises():
    """transition_to raises ValueError for illegal transitions."""
    task = make_task("Invalid transition")
    task.transition_to(TaskStatus.ASSIGNED)
    task.transition_to(TaskStatus.RUNNING)
    task.transition_to(TaskStatus.COMPLETED)

    with pytest.raises(ValueError, match="Illegal task transition"):
        task.transition_to(TaskStatus.RUNNING)


def test_task_type_validation():
    """Task.__post_init__ raises TypeError for wrong parameter/metadata/dependency types."""
    with pytest.raises(TypeError, match="dictionary"):
        Task(description="Bad params", parameters=[])  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="dictionary"):
        Task(description="Bad meta", metadata="not-a-dict")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="list"):
        Task(description="Bad deps", dependencies="not-a-list")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="non-empty string"):
        Task(description="  ")


def test_task_description_stripped():
    """Task descriptions are stripped of surrounding whitespace."""
    task = Task(description="  trimmed  ", parameters={})
    assert task.description == "trimmed"


def test_task_success_updates_bookkeeping():
    """Successful execution via system.execute_task updates all lifecycle tracking."""
    store = InMemoryTaskStore()
    system = AgentSystem("LifecycleSystem", task_store=store)
    executor = ExecutorAgent("LifecycleExecutor")
    assert system.add_agent(executor)

    task = system.create_task(
        description="Process lifecycle metrics",
        parameters={"metric": "latency"},
        priority=TaskPriority.HIGH,
    )

    # Task is in the queue and _task_index before submission
    assert task.id in system._task_index

    assert system.submit_task(task, executor.id) is True
    assert task.status == TaskStatus.ASSIGNED
    assert task.id in executor.active_tasks
    # active_tasks property aggregates from agents
    assert task.id in system.active_tasks
    # Dequeued after assignment
    assert task.id not in system._task_index

    result = system.execute_task(task.id, executor.id)

    assert result["execution"] == "successful"
    assert task.status == TaskStatus.COMPLETED
    assert task.completed_at is not None
    assert task.started_at is not None
    assert executor.status == AgentStatus.IDLE
    assert task.id not in executor.active_tasks
    assert task.id not in system.active_tasks
    assert task in executor.completed_tasks
    assert task in executor.task_history
    assert task in system.completed_tasks
    assert task in system.task_history
    assert task not in system.failed_tasks
    assert executor.performance_metrics["tasks_completed"] == 1
    assert executor.performance_metrics["tasks_failed"] == 0
    assert system.system_metrics["successful_tasks"] == 1
    assert system.system_metrics["failed_tasks"] == 0
    assert system.system_metrics["avg_task_duration"] >= 0

    # Status history in metadata
    history = task.metadata.get("status_history", [])
    assert any(entry["to"] == TaskStatus.COMPLETED.value for entry in history)


def test_task_failure_updates_metrics(monkeypatch):
    """Execution failures update metrics and do not leave stale active-task state."""
    store = InMemoryTaskStore()
    system = AgentSystem("FailureSystem", task_store=store)
    executor = ExecutorAgent("FailureExecutor")
    assert system.add_agent(executor)

    task = system.create_task(
        description="Fail gracefully",
        parameters={"should_fail": True},
    )
    assert system.submit_task(task, executor.id) is True

    with pytest.raises(RuntimeError, match="simulated failure"):
        monkeypatch.setattr(executor, "act", lambda _d: (_ for _ in ()).throw(RuntimeError("simulated failure")))
        system.execute_task(task.id, executor.id)

    assert task.status == TaskStatus.FAILED
    assert executor.status == AgentStatus.IDLE
    assert task.id not in executor.active_tasks
    assert task.id not in system.active_tasks
    assert task not in system.completed_tasks
    assert task in system.failed_tasks
    assert task in system.task_history
    assert system.system_metrics["failed_tasks"] == 1
    assert system.system_metrics["successful_tasks"] == 0

    history = task.metadata.get("status_history", [])
    assert any(entry["to"] == TaskStatus.FAILED.value for entry in history)


def test_dependency_validation_at_creation_time():
    """create_task raises ValueError when dependency IDs are unknown to the system."""
    store = InMemoryTaskStore()
    system = AgentSystem("DependencySystem", task_store=store)

    with pytest.raises(ValueError, match="Unknown task dependencies"):
        system.create_task(
            description="Broken dependency task",
            parameters={},
            dependencies=["missing-task-id"],
        )


def test_known_dependency_accepted_at_creation():
    """create_task accepts dependency IDs that correspond to existing tasks."""
    store = InMemoryTaskStore()
    system = AgentSystem("DependencySystem", task_store=store)

    prerequisite = system.create_task(description="Prerequisite", parameters={})
    dependent = system.create_task(
        description="Dependent task",
        parameters={},
        dependencies=[prerequisite.id],
    )
    assert prerequisite.id in dependent.dependencies


def test_queue_ordering_high_before_low():
    """High-priority tasks appear before low-priority tasks in the queue heap."""
    store = InMemoryTaskStore()
    system = AgentSystem("OrderSystem", task_store=store)

    low = system.create_task("Low priority", parameters={}, priority=TaskPriority.LOW)
    high = system.create_task("High priority", parameters={}, priority=TaskPriority.HIGH)

    # heapq root is always the minimum element (highest priority = lowest priority_rank value)
    assert system.global_task_queue[0].id == high.id


def test_submit_task_blocked_by_unmet_dependency():
    """submit_task returns False when a task's dependency has not completed."""
    store = InMemoryTaskStore()
    system = AgentSystem("DepBlockSystem", task_store=store)
    executor = ExecutorAgent("Executor-dep")
    system.add_agent(executor)

    prerequisite = system.create_task("Prerequisite", parameters={})
    dependent = system.create_task(
        "Dependent", parameters={}, dependencies=[prerequisite.id]
    )

    # Cannot submit dependent while prerequisite is still pending
    assert system.submit_task(dependent, executor.id) is False
    assert dependent.status == TaskStatus.PENDING
    assert dependent.id in system._task_index

    # After completing prerequisite, dependent can be submitted
    assert system.submit_task(prerequisite, executor.id) is True
    system.execute_task(prerequisite.id, executor.id)
    assert system.submit_task(dependent, executor.id) is True


def test_agent_system_attribute_set_on_add_cleared_on_remove():
    """agent.system is set when added to a system and cleared when removed."""
    store = InMemoryTaskStore()
    system = AgentSystem("SystemAttrSystem", task_store=store)
    executor = ExecutorAgent("Executor-sys")
    assert executor.system is None

    assert system.add_agent(executor) is True
    assert executor.system is system

    assert system.remove_agent(executor.id) is True
    assert executor.system is None
    assert executor.id not in system.agents
    assert executor.id not in system.orchestrator.managed_agents


def test_remove_agent_rejects_while_task_active():
    """remove_agent returns False while the agent has an active assigned task."""
    store = InMemoryTaskStore()
    system = AgentSystem("RemovalSystem", task_store=store)
    executor = ExecutorAgent("Executor-busy")
    system.add_agent(executor)

    task = system.create_task("Work before removal", parameters={})
    assert system.submit_task(task, executor.id) is True

    # Cannot remove while task is active
    assert system.remove_agent(executor.id) is False

    # After completing the task via system, removal succeeds
    system.execute_task(task.id, executor.id)
    assert system.remove_agent(executor.id) is True


def test_invalid_agent_name_raises():
    """BaseAgent raises ValueError for empty or whitespace-only names."""
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        ExecutorAgent(" ")

    with pytest.raises(ValueError, match="empty or whitespace-only"):
        ExecutorAgent("")


def test_agent_name_stripped():
    """BaseAgent.name is stripped of surrounding whitespace."""
    agent = ExecutorAgent("  MyAgent  ")
    assert agent.name == "MyAgent"


def test_agent_factory_raises_for_unknown_type():
    """AgentFactory.create_agent raises ValueError for unknown or empty agent_type."""
    with pytest.raises(ValueError, match="agent_type"):
        AgentFactory.create_agent("", "NamedAgent")

    with pytest.raises(ValueError, match="agent_type"):
        AgentFactory.create_agent("unknown_type", "NamedAgent")


def test_task_cancelled_is_terminal():
    """CANCELLED is treated as a terminal state by is_terminal()."""
    task = make_task("Cancel me")
    task.transition_to(TaskStatus.ASSIGNED)
    task.transition_to(TaskStatus.CANCELLED)
    assert task.is_terminal()

    with pytest.raises(ValueError, match="Illegal task transition"):
        task.transition_to(TaskStatus.PENDING)


def test_task_dependency_blocked_recovers_to_pending():
    """DEPENDENCY_BLOCKED task can transition back to PENDING."""
    task = make_task("Blocked task")
    task.transition_to(TaskStatus.DEPENDENCY_BLOCKED)
    assert task.status == TaskStatus.DEPENDENCY_BLOCKED
    assert not task.is_terminal()

    task.transition_to(TaskStatus.PENDING)
    assert task.status == TaskStatus.PENDING


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
