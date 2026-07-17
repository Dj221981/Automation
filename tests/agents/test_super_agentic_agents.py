import pytest

from src.agents.super_agentic_agents import AgentRole, AgentSystem, BaseAgent, Task, TaskStatus


class SuccessfulAgent(BaseAgent):
    def __init__(self, name: str = "successful-agent"):
        super().__init__(name=name, role=AgentRole.EXECUTOR)

    def think(self, input_data):
        return {"input": input_data}

    def act(self, decision):
        return {"ok": True, "decision": decision}


class FailThenSucceedAgent(BaseAgent):
    def __init__(self, name: str = "retry-agent"):
        super().__init__(name=name, role=AgentRole.EXECUTOR)
        self.attempts = 0

    def think(self, input_data):
        return {"attempt": self.attempts + 1}

    def act(self, decision):
        self.attempts += 1
        if self.attempts == 1:
            raise RuntimeError("first attempt fails")
        return {"attempt": self.attempts, "ok": True}


class AlwaysFailAgent(BaseAgent):
    def __init__(self, name: str = "always-fail-agent"):
        super().__init__(name=name, role=AgentRole.EXECUTOR)

    def think(self, input_data):
        return {"input": input_data}

    def act(self, decision):
        raise RuntimeError("always fails")


def test_process_task_success_path_updates_terminal_metrics():
    system = AgentSystem("test-system-success")
    agent = SuccessfulAgent()
    system.add_agent(agent)
    task = system.create_task("run success task", {"value": 1})

    processed = system.process_task(task.id, agent.id)

    assert processed is True
    assert task.status == TaskStatus.COMPLETED
    assert task.started_at is not None
    assert task.last_attempt_at is not None
    assert task.completed_at is not None
    assert task.to_dict()["started_at"] is not None
    assert system.system_metrics["total_tasks"] == 1
    assert system.system_metrics["successful_tasks"] == 1
    assert system.system_metrics["failed_tasks"] == 0
    assert system.system_metrics["avg_task_duration"] >= 0
    assert len(system.global_task_queue) == 0
    assert any(completed.id == task.id for completed in system.completed_tasks)


def test_process_task_failure_then_retry_success():
    system = AgentSystem("test-system-retry-success")
    agent = FailThenSucceedAgent()
    system.add_agent(agent)
    task = system.create_task("run retry task", {"value": 1})
    task.max_retries = 1

    first_attempt = system.process_task(task.id, agent.id)

    assert first_attempt is False
    assert task.status == TaskStatus.PENDING
    assert task.retry_count == 1
    assert task.next_retry_at is not None
    assert system.system_metrics["successful_tasks"] == 0
    assert system.system_metrics["failed_tasks"] == 0
    assert len(system.global_task_queue) == 1

    second_attempt = system.process_task(task.id, agent.id)

    assert second_attempt is True
    assert task.status == TaskStatus.COMPLETED
    assert system.system_metrics["successful_tasks"] == 1
    assert system.system_metrics["failed_tasks"] == 0


def test_process_task_retry_exhausted_counts_terminal_failure_once():
    system = AgentSystem("test-system-retry-exhausted")
    agent = AlwaysFailAgent()
    system.add_agent(agent)
    task = system.create_task("run fail task", {"value": 1})
    task.max_retries = 1

    first_attempt = system.process_task(task.id, agent.id)
    assert first_attempt is False
    assert task.status == TaskStatus.PENDING
    assert task.retry_count == 1
    assert system.system_metrics["failed_tasks"] == 0

    second_attempt = system.process_task(task.id, agent.id)
    assert second_attempt is False
    assert task.status == TaskStatus.FAILED
    assert len(system.global_task_queue) == 0
    assert system.system_metrics["failed_tasks"] == 1
    assert system.system_metrics["successful_tasks"] == 0

    third_attempt = system.process_task(task.id, agent.id)
    assert third_attempt is False
    assert system.system_metrics["failed_tasks"] == 1


def test_invalid_transition_guard_raises_clear_error():
    task = Task(description="invalid transition")

    with pytest.raises(ValueError, match="Invalid task status transition"):
        task.transition_to(TaskStatus.COMPLETED)

    task.transition_to(TaskStatus.ASSIGNED)
    task.transition_to(TaskStatus.RUNNING)
    task.transition_to(TaskStatus.COMPLETED)
    with pytest.raises(ValueError, match="Invalid task status transition"):
        task.transition_to(TaskStatus.RUNNING)
