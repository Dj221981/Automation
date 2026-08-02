import threading

from src.agents.super_agentic_agents import AgentSystem, ExecutorAgent, TaskStatus


def test_submit_same_task_concurrently_only_one_assignment():
    system = AgentSystem("test")
    a1 = ExecutorAgent("a1")
    a2 = ExecutorAgent("a2")
    assert system.add_agent(a1)
    assert system.add_agent(a2)

    task = system.create_task("race", {})
    results = []

    def submit(agent_id: str):
        results.append(system.submit_task(task, agent_id))

    t1 = threading.Thread(target=submit, args=(a1.id,))
    t2 = threading.Thread(target=submit, args=(a2.id,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert results.count(True) == 1
    assert results.count(False) == 1
    assert task.status == TaskStatus.ASSIGNED


def test_parallel_task_creation_does_not_corrupt_queue_index():
    system = AgentSystem("test")

    def create(i: int):
        system.create_task(f"task-{i}", {})

    threads = [threading.Thread(target=create, args=(i,)) for i in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(system._task_index) == 100
    assert system.system_metrics["total_tasks"] == 100
