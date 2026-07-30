"""Basic performance benchmarks for agent and training pathways."""

from __future__ import annotations

import time

from src.agents.super_agentic_agents import AgentSystem, ExecutorAgent


def test_agent_task_execution_benchmark():
    system = AgentSystem("perf")
    agent = ExecutorAgent("worker")
    assert system.add_agent(agent)

    start = time.perf_counter()
    for _ in range(10):
        task = system.create_task("benchmark", {})
        assert system.submit_task(task, agent.id)
        system.execute_task(task.id, agent.id)
    elapsed = time.perf_counter() - start

    assert elapsed >= 0


def test_concurrent_agent_handling_smoke():
    system = AgentSystem("perf2")
    agents = [ExecutorAgent(f"worker-{i}") for i in range(3)]
    for agent in agents:
        assert system.add_agent(agent)

    created = [system.create_task(f"task-{i}", {}) for i in range(6)]
    for idx, task in enumerate(created):
        target = agents[idx % len(agents)]
        assert system.submit_task(task, target.id)
        system.execute_task(task.id, target.id)

    assert len(system.completed_tasks) == 6
