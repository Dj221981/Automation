import threading
import time

from src.agents.super_agentic_agents import AgentSystem, ExecutorAgent, TaskStatus


class BlockingExecutorAgent(ExecutorAgent):
    def __init__(self, name: str = "blocking") -> None:
        super().__init__(name)
        self.started = threading.Event()
        self.finished = threading.Event()

    def act(self, decision):
        self.started.set()
        self.finished.wait(1.0)
        return super().act(decision)


def test_queue_stale_entries_are_skipped_and_counted():
    system = AgentSystem("hardening")
    stale = system.create_task("stale", {})
    live = system.create_task("live", {})

    system._task_index.discard(stale.id)

    assert system._pop_next_valid_task_id() == live.id
    assert system.system_metrics["queue_stale_pops"] >= 1
    assert all(entry.id != stale.id for entry in system.global_task_queue)


def test_submit_task_blocks_until_dependencies_complete():
    system = AgentSystem("deps")
    agent = ExecutorAgent("worker")
    assert system.add_agent(agent)

    dependency = system.create_task("dependency", {})
    dependent = system.create_task("dependent", {}, dependencies=[dependency.id])

    assert system.submit_task(dependent, agent.id) is False
    assert dependent.status == TaskStatus.PENDING
    assert system.system_metrics["dependency_blocked_tasks"] == 1

    assert system.submit_task(dependency, agent.id) is True
    system.execute_task(dependency.id, agent.id)

    assert system.submit_task(dependent, agent.id) is True


def test_reclaim_expired_claims_requeues_task():
    system = AgentSystem("claims")
    agent = ExecutorAgent("worker")
    assert system.add_agent(agent)

    task = system.create_task("expiring", {})
    assert system.submit_task(task, agent.id) is True

    task.metadata["claim_expires_at"] = "2000-01-01T00:00:00"

    reclaimed = system.reclaim_expired_claims()

    assert reclaimed == 1
    assert task.status == TaskStatus.PENDING
    assert task.assigned_to is None
    assert any(entry.id == task.id for entry in system.global_task_queue)
    assert system.system_metrics["claim_reclaims"] == 1


def test_worker_loop_processes_pending_tasks():
    system = AgentSystem("workers")
    agent = ExecutorAgent("worker")
    assert system.add_agent(agent)

    task = system.create_task("background", {})
    system.start_workers(1)
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            loaded = system.load_task(task.id)
            if loaded is not None and loaded.status == TaskStatus.COMPLETED:
                break
            time.sleep(0.01)
        else:
            assert False, "task did not complete in time"
    finally:
        assert system.drain_and_shutdown(timeout_seconds=1.0) is True


def test_create_task_is_idempotent_with_reused_key():
    system = AgentSystem("idempotent")

    first = system.create_task("same", {"value": 1, "idempotency_key": "duplicate-key"})
    second = system.create_task("same", {"value": 1, "idempotency_key": "duplicate-key"})

    assert second.id == first.id
    assert system.system_metrics["total_tasks"] == 1


def test_drain_and_shutdown_waits_for_active_worker_task():
    system = AgentSystem("drain")
    agent = BlockingExecutorAgent("worker")
    assert system.add_agent(agent)

    task = system.create_task("blocking", {})
    system.start_workers(1)
    try:
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and not agent.started.is_set():
            time.sleep(0.01)
        assert agent.started.is_set()

        result = {}

        def drain():
            result["ok"] = system.drain_and_shutdown(timeout_seconds=1.0)

        shutdown_thread = threading.Thread(target=drain)
        shutdown_thread.start()
        time.sleep(0.05)
        agent.finished.set()
        shutdown_thread.join(timeout=2.0)

        assert result["ok"] is True
        assert system.load_task(task.id).status == TaskStatus.COMPLETED
    finally:
        agent.finished.set()
        system.stop_workers(timeout_seconds=0.1)
