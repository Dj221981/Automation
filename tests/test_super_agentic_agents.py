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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
