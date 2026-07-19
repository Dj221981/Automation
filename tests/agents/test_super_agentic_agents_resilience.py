from src.agents.super_agentic_agents import AgentSystem, ExecutorAgent, TaskStatus
from src.agents.task_store import InMemoryTaskStore


def _make_failing_agent(name: str) -> ExecutorAgent:
    agent = ExecutorAgent(name)

    def boom(_):
        raise RuntimeError("boom")

    agent.act = boom  # type: ignore[assignment]
    return agent


def test_failed_execution_tracks_retry_metadata_without_auto_requeue():
    store = InMemoryTaskStore()
    system = AgentSystem("retry-metadata", task_store=store)
    agent = _make_failing_agent("worker")
    assert system.add_agent(agent)

    task = system.create_task("job", {})
    assert system.submit_task(task, agent.id)

    try:
        system.execute_task(task.id, agent.id)
        assert False, "Expected RuntimeError"
    except RuntimeError as exc:
        assert str(exc) == "boom"

    stored = store.get_task(task.id)
    assert stored is not None
    assert task.status == TaskStatus.FAILED
    assert stored.status == "FAILED"
    assert stored.metadata["attempts"] == 1
    assert stored.metadata["retry_backoff_seconds"] == system.retry_backoff_base_seconds
    assert stored.metadata.get("next_retry_at")
    assert stored.metadata.get("dead_lettered_at") is None
    assert task.id not in agent.active_tasks
    assert task.id not in system._task_index


def test_manual_requeue_then_second_failure_moves_task_to_dead_letter():
    store = InMemoryTaskStore()
    system = AgentSystem("dead-letter", task_store=store)
    system.max_retries_per_task = 2
    agent = _make_failing_agent("worker")
    assert system.add_agent(agent)

    task = system.create_task("job", {})
    assert system.submit_task(task, agent.id)

    for attempt in range(2):
        try:
            system.execute_task(task.id, agent.id)
            assert False, "Expected RuntimeError"
        except RuntimeError as exc:
            assert str(exc) == "boom"
        if attempt == 0:
            task = system.requeue_task(task.id)
            assert task.status == TaskStatus.PENDING
            assert system.submit_task(task, agent.id)

    stored = store.get_task(task.id)
    assert stored is not None
    assert stored.status == "FAILED"
    assert stored.metadata["attempts"] == 2
    assert stored.metadata.get("dead_lettered_at")
    assert stored.metadata.get("dead_letter_reason") == "boom"
    assert stored.metadata.get("next_retry_at") is None
    assert len(system.dead_letter_queue) == 1
    assert system.dead_letter_queue[-1]["task_id"] == task.id
    assert system.dead_letter_queue[-1]["attempts"] == 2

    snapshot = system.get_observability_snapshot()
    assert snapshot["dead_letter_depth"] == 1
    assert snapshot["retry_backlog"] == 0
    assert snapshot["recent_dead_letters"][-1]["task_id"] == task.id


def test_recover_incomplete_tasks_on_restart_restores_pending_queue():
    store = InMemoryTaskStore()
    initial_system = AgentSystem("initial", task_store=store)
    agent = ExecutorAgent("worker")
    assert initial_system.add_agent(agent)

    task = initial_system.create_task("recover-me", {})
    assert initial_system.submit_task(task, agent.id)

    stored = store.get_task(task.id)
    assert stored is not None
    stored.status = "RUNNING"
    store.update_task(stored)

    restarted_system = AgentSystem("restarted", task_store=store)
    recovered = restarted_system.recover_incomplete_tasks()

    assert recovered == 1
    recovered_task = restarted_system.load_task(task.id)
    assert recovered_task is not None
    assert recovered_task.status == TaskStatus.PENDING
    assert recovered_task.assigned_to is None
    assert any(entry.id == task.id for entry in restarted_system.global_task_queue)
