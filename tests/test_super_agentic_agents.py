"""Focused tests for the hardened super agentic agents framework."""

import pytest

from src.agents.super_agentic_agents import (
    AgentFactory,
    AgentStatus,
    AgentSystem,
    ExecutorAgent,
    Task,
    TaskPriority,
    TaskStatus,
)


def test_task_success_updates_bookkeeping():
    """Successful execution should keep task lifecycle and metrics consistent."""
    system = AgentSystem("LifecycleSystem")
    executor = ExecutorAgent("LifecycleExecutor")
    assert system.add_agent(executor)

    task = system.create_task(
        description="Process lifecycle metrics",
        parameters={"metric": "latency"},
        priority=TaskPriority.HIGH,
    )

    assert [queued_task.id for queued_task in system.global_task_queue] == [task.id]
    assert system.submit_task(task, executor.id) is True
    assert task.status == TaskStatus.ASSIGNED
    assert task.id in executor.active_tasks
    assert task.id in system.active_tasks
    assert task not in system.global_task_queue

    result = executor.execute_task(task)

    assert result["execution"] == "successful"
    assert task.status == TaskStatus.COMPLETED
    assert task.completed_at is not None
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
    assert executor.performance_metrics["avg_task_time"] >= 0
    assert system.system_metrics["successful_tasks"] == 1
    assert system.system_metrics["failed_tasks"] == 0
    assert system.system_metrics["avg_task_duration"] >= 0
    assert task.metadata["status_history"][-1]["to"] == TaskStatus.COMPLETED.value


def test_task_failure_updates_metrics(monkeypatch):
    """Execution failures should not leave stale task or agent state behind."""
    system = AgentSystem("FailureSystem")
    executor = ExecutorAgent("FailureExecutor")
    assert system.add_agent(executor)

    task = system.create_task(
        description="Fail gracefully",
        parameters={"should_fail": True},
    )
    assert system.submit_task(task, executor.id) is True

    def fail_task(_decision):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(executor, "act", fail_task)

    with pytest.raises(RuntimeError, match="simulated failure"):
        executor.execute_task(task)

    assert task.status == TaskStatus.FAILED
    assert task.error == "simulated failure"
    assert executor.status == AgentStatus.IDLE
    assert task.id not in executor.active_tasks
    assert task.id not in system.active_tasks
    assert task not in executor.completed_tasks
    assert task in executor.task_history
    assert task not in system.completed_tasks
    assert task in system.failed_tasks
    assert task in system.task_history
    assert executor.performance_metrics["tasks_failed"] == 1
    assert system.system_metrics["failed_tasks"] == 1
    assert task.metadata["status_history"][-1]["to"] == TaskStatus.FAILED.value


def test_dependency_validation_and_queue_ordering():
    """Dependencies should be validated and pending queue order should be stable."""
    system = AgentSystem("DependencySystem")
    executor = ExecutorAgent("DependencyExecutor")
    assert system.add_agent(executor)

    low_priority = system.create_task(
        description="Low priority task",
        parameters={},
        priority=TaskPriority.LOW,
    )
    high_priority = system.create_task(
        description="High priority task",
        parameters={},
        priority=TaskPriority.HIGH,
    )

    assert len(system.global_task_queue) == 2
    assert [task.id for task in system.global_task_queue] == [high_priority.id, low_priority.id]

    with pytest.raises(ValueError, match="Unknown task dependencies"):
        system.create_task(
            description="Broken dependency task",
            parameters={},
            dependencies=["missing-task"],
        )

    prerequisite = system.create_task(
        description="Prerequisite task",
        parameters={},
    )
    dependent = system.create_task(
        description="Dependent task",
        parameters={},
        dependencies=[prerequisite.id],
    )

    assert system.submit_task(dependent, executor.id) is False
    assert dependent.status == TaskStatus.PENDING
    assert dependent in system.global_task_queue

    assert system.submit_task(prerequisite, executor.id) is True
    executor.execute_task(prerequisite)
    assert system.submit_task(dependent, executor.id) is True


def test_remove_agent_rejects_active_tasks_and_succeeds_after_completion():
    """Agents with in-flight work should not be removable."""
    system = AgentSystem("RemovalSystem")
    executor = ExecutorAgent("RemovalExecutor")
    assert system.add_agent(executor)

    task = system.create_task(
        description="Work before removal",
        parameters={},
    )
    assert system.submit_task(task, executor.id) is True
    assert system.remove_agent(executor.id) is False

    executor.execute_task(task)

    assert system.remove_agent(executor.id) is True
    assert executor.system is None
    assert executor.id not in system.agents
    assert executor.id not in system.orchestrator.managed_agents


def test_invalid_inputs_are_rejected_early():
    """Invalid task and agent input should fail fast."""
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        ExecutorAgent(" ")

    with pytest.raises(TypeError, match="dictionary"):
        Task(
            description="Bad parameters",
            parameters=[],
            priority=TaskPriority.NORMAL,
        )

    with pytest.raises(ValueError, match="agent_type"):
        AgentFactory.create_agent("", "NamedAgent")
