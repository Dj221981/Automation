from src.agents.super_agentic_agents import AgentSystem, ExecutorAgent, TaskPriority, TaskStatus
from src.agents.task_store import InMemoryTaskStore, StoredTask


class FailingExecutorAgent(ExecutorAgent):
    def act(self, decision):
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

    agent.release_task(task.id)

    recovered = system.recover_incomplete_tasks()
    assert recovered == 1

    reloaded = store.get_task(task.id)
    assert reloaded is not None
    assert reloaded.status == "PENDING"
    assert reloaded.assigned_to is None
    assert reloaded.metadata.get("claimed_by") is None
    assert any(queued.id == task.id for queued in system.global_task_queue)


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
    assert any(queued.id == task.id for queued in system.global_task_queue)
