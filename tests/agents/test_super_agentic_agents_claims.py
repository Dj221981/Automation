import pytest

from src.agents.super_agentic_agents import AgentSystem, ExecutorAgent


def test_claim_expires_blocks_execution():
    system = AgentSystem("test")
    agent = ExecutorAgent("worker")
    assert system.add_agent(agent)

    task = system.create_task("job", {})
    assert system.submit_task(task, agent.id)

    # Expire claim manually
    task.metadata["claim_expires_at"] = "2000-01-01T00:00:00"

    with pytest.raises(ValueError, match="claim expired"):
        system.execute_task(task.id, agent.id)


def test_wrong_agent_cannot_execute_claimed_task():
    system = AgentSystem("test")
    a1 = ExecutorAgent("a1")
    a2 = ExecutorAgent("a2")
    assert system.add_agent(a1)
    assert system.add_agent(a2)

    task = system.create_task("job", {})
    assert system.submit_task(task, a1.id)

    with pytest.raises((KeyError, ValueError)):
        system.execute_task(task.id, a2.id)
