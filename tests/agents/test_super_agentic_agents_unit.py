import threading
import uuid

import pytest

from src.agents.super_agentic_agents import (
    AgentCapability,
    AgentSystem,
    ExecutorAgent,
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
