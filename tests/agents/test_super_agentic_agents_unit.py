import threading
import uuid

import pytest

from src.agents.super_agentic_agents import (
    AgentCapability,
    AgentSystem,
    ExecutorAgent,
    Task,
    TaskPriority,
    TaskStatus,
)


def test_create_task_sets_defaults_and_metadata():
    system = AgentSystem("test")
    task = system.create_task("do thing", {"x": 1}, priority=TaskPriority.HIGH)

    assert task.status == TaskStatus.PENDING
    assert task.priority == TaskPriority.HIGH
    assert "correlation_id" in task.metadata
    assert isinstance(uuid.UUID(task.metadata["correlation_id"]), uuid.UUID)
    assert task.metadata["attempts"] == 0
    assert task.metadata["max_attempts"] == system.max_retries_per_task


def test_register_and_execute_capability_success():
    agent = ExecutorAgent("exec")

    def cap_fn(x: int) -> int:
        return x + 1

    cap = AgentCapability(name="increment", description="inc", func=cap_fn)
    assert agent.register_capability(cap) is True

    result = agent.execute_capability("increment", x=41)
    assert result == 42


def test_capability_rate_limit_enforced():
    agent = ExecutorAgent("exec")

    cap = AgentCapability(
        name="limited",
        description="limited",
        func=lambda: True,
        max_calls_per_minute=1,
    )
    assert agent.register_capability(cap)

    assert agent.execute_capability("limited") is True
    with pytest.raises(RuntimeError, match="Rate limit exceeded"):
        agent.execute_capability("limited")


def test_capability_role_restriction_enforced():
    agent = ExecutorAgent("exec")

    cap = AgentCapability(
        name="restricted",
        description="restricted",
        func=lambda: True,
        allowed_roles=set(),
    )
    assert agent.register_capability(cap)

    with pytest.raises(PermissionError, match="cannot use capability"):
        agent.execute_capability("restricted")


def test_illegal_transition_rejected_completed_to_running():
    system = AgentSystem("test")
    agent = ExecutorAgent("worker")
    assert system.add_agent(agent)

    task = system.create_task("job", {})
    assert system.submit_task(task, agent.id)
    system.execute_task(task.id, agent.id)

    with pytest.raises(ValueError, match="Illegal task transition"):
        system._set_task_status(task, TaskStatus.RUNNING, assigned_to=agent.id, claimed_by=agent.id)


def test_queue_overflow_raises():
    system = AgentSystem("test")
    system.max_queue_size = 1
    system.create_task("one", {})

    with pytest.raises(OverflowError, match="Task queue full"):
        system.create_task("two", {})


def test_retry_delay_caps_exponent_and_max_delay():
    system = AgentSystem("test")
    system.retry_backoff_base_seconds = 5
    system.retry_backoff_max_seconds = 10_000

    assert system._calculate_retry_delay(1) == 5
    assert system._calculate_retry_delay(2) == 10
    assert system._calculate_retry_delay(25) == 5 * (2**10)

    system.retry_backoff_max_seconds = 120
    assert system._calculate_retry_delay(25) == 120


def test_sync_task_copies_state_without_sharing_references():
    system = AgentSystem("test")
    source = Task(
        description="source",
        priority=TaskPriority.HIGH,
        assigned_to="agent-1",
        status=TaskStatus.FAILED,
        result={"nested": {"ok": True}},
        error="boom",
        parameters={"payload": {"value": 1}},
        dependencies=["dep-1"],
        metadata={"retry": {"count": 2}},
    )
    target = Task(description="target", parameters={"old": True})

    system._sync_task(target, source)

    assert target.description == source.description
    assert target.priority == source.priority
    assert target.assigned_to == source.assigned_to
    assert target.status == source.status
    assert target.result == source.result
    assert target.error == source.error
    assert target.parameters == source.parameters
    assert target.dependencies == source.dependencies
    assert target.metadata == source.metadata

    source.result["nested"]["ok"] = False
    source.parameters["payload"]["value"] = 99
    source.dependencies.append("dep-2")
    source.metadata["retry"]["count"] = 3

    assert target.result["nested"]["ok"] is True
    assert target.parameters["payload"]["value"] == 1
    assert target.dependencies == ["dep-1"]
    assert target.metadata["retry"]["count"] == 2


def test_retry_backlog_increments_and_clears_on_manual_requeue():
    class FailingExecutorAgent(ExecutorAgent):
        def act(self, _decision):
            raise RuntimeError("boom")

    system = AgentSystem("test")
    agent = FailingExecutorAgent("worker")
    assert system.add_agent(agent)

    task = system.create_task("retry", {})
    assert system.submit_task(task, agent.id)

    with pytest.raises(RuntimeError, match="boom"):
        system.execute_task(task.id, agent.id)

    assert system.get_observability_snapshot()["retry_backlog"] == 1

    system.requeue_task(task.id)
    assert system.get_observability_snapshot()["retry_backlog"] == 0


def test_queue_compaction_discards_stale_entries():
    system = AgentSystem("test")
    system.max_queue_size = 1
    task = system.create_task("compact-me", {})

    for _ in range(4):
        system._dequeue_task(task.id)
        system._enqueue_if_missing(task)

    assert len(system._global_task_queue) == 1
    assert len(system.global_task_queue) == 1
    assert system.global_task_queue[0].id == task.id
