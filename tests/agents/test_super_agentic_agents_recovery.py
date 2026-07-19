from src.agents.super_agentic_agents import AgentSystem, ExecutorAgent, TaskStatus


def test_requeue_failed_task_returns_to_pending():
    system = AgentSystem("test")
    agent = ExecutorAgent("worker")
    assert system.add_agent(agent)

    task = system.create_task("job", {})
    assert system.submit_task(task, agent.id)

    # Force failure by monkey-patching act
    def boom(_decision):
        raise RuntimeError("boom")

    agent.act = boom  # type: ignore[assignment]

    try:
        system.execute_task(task.id, agent.id)
    except RuntimeError:
        pass

    # Depending on retry policy, it may already be PENDING or FAILED then requeue
    if task.status != TaskStatus.PENDING:
        task = system.requeue_task(task.id)

    assert task.status == TaskStatus.PENDING


def test_recover_incomplete_tasks_resets_running_to_pending():
    system = AgentSystem("test")
    agent = ExecutorAgent("worker")
    assert system.add_agent(agent)

    task = system.create_task("recover-me", {})
    assert system.submit_task(task, agent.id)

    # Simulate a stuck running task persisted state
    system._set_task_status(task, TaskStatus.RUNNING, assigned_to=agent.id, claimed_by=agent.id)

    recovered = system.recover_incomplete_tasks()
    assert recovered >= 1

    loaded = system.load_task(task.id)
    assert loaded is not None
    assert loaded.status == TaskStatus.PENDING
    assert loaded.assigned_to is None
