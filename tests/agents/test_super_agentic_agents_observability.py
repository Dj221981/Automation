from src.agents.super_agentic_agents import AgentSystem, ExecutorAgent


def test_observability_snapshot_contains_metrics_and_events():
    system = AgentSystem("obs")
    agent = ExecutorAgent("worker")
    assert system.add_agent(agent)

    task = system.create_task("obs task", {})
    assert system.submit_task(task, agent.id)
    system.execute_task(task.id, agent.id)

    snap = system.get_observability_snapshot()
    assert "metrics" in snap
    assert "recent_events" in snap
    assert "queue_depth" in snap
    assert "dead_letter_depth" in snap
    assert isinstance(snap["recent_events"], list)
    assert len(snap["recent_events"]) >= 1


def test_system_status_includes_dead_letter_count():
    system = AgentSystem("obs")
    status = system.get_system_status()
    assert "dead_letter_tasks" in status
