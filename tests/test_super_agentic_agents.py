import threading
import time

from src.agents.super_agentic_agents import AgentSystem, ExecutorAgent, TaskPriority, TaskStatus
from src.agents.task_store import InMemoryTaskStore, StoredTask, ensure_valid_transition


class FailingExecutorAgent(ExecutorAgent):
    def act(self, decision):
        raise RuntimeError("boom")


class SlowExecutorAgent(ExecutorAgent):
    """Agent whose act() deliberately sleeps, used for timeout tests."""

    def __init__(self, name: str = "SlowExecutor", sleep_seconds: float = 0.5):
        super().__init__(name)
        self._sleep_seconds = sleep_seconds

    def act(self, decision):
        time.sleep(self._sleep_seconds)
        return super().act(decision)


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

    recovered = system.recover_incomplete_tasks()
    assert recovered == 1

    reloaded = store.get_task(task.id)
    assert reloaded is not None
    assert reloaded.status == "PENDING"
    assert reloaded.assigned_to is None
    assert reloaded.metadata.get("claimed_by") is None
    assert any(queued.id == task.id for queued in system.global_task_queue)
    assert agent.active_tasks == {}


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


def test_claim_mismatch_blocks_execution():
    store = InMemoryTaskStore()
    system = AgentSystem("ClaimSystem", task_store=store)
    agent_one = ExecutorAgent("Executor-1")
    agent_two = ExecutorAgent("Executor-2")
    system.add_agent(agent_one)
    system.add_agent(agent_two)

    task = system.create_task("Claimed task", {"value": 3})
    assert system.submit_task(task, agent_one.id) is True

    agent_two.active_tasks[task.id] = task

    try:
        system.execute_task(task.id, agent_two.id)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "claimed by" in str(exc)


def test_remove_agent_guards_and_idle_removal():
    store = InMemoryTaskStore()
    system = AgentSystem("GuardSystem", task_store=store)
    busy_agent = ExecutorAgent("BusyExecutor")
    idle_agent = ExecutorAgent("IdleExecutor")
    system.add_agent(busy_agent)
    system.add_agent(idle_agent)

    task = system.create_task("Active task", {"value": 4})
    assert system.submit_task(task, busy_agent.id) is True

    assert system.remove_agent(system.orchestrator.id) is False
    assert system.remove_agent(busy_agent.id) is False
    assert system.remove_agent(idle_agent.id) is True


def test_queue_dedup_and_cancel_task():
    store = InMemoryTaskStore()
    system = AgentSystem("QueueSystem", task_store=store)
    agent = ExecutorAgent("Executor-1")
    system.add_agent(agent)

    task = system.create_task("Queue task", {"value": 5})
    system._enqueue_if_missing(task)
    assert sum(1 for queued in system.global_task_queue if queued.id == task.id) == 1

    assert system.submit_task(task, agent.id) is True
    assert sum(1 for queued in system.global_task_queue if queued.id == task.id) == 0

    requeued = system.requeue_task(task.id)
    system._enqueue_if_missing(requeued)
    assert sum(1 for queued in system.global_task_queue if queued.id == task.id) == 1

    cancelled = system.cancel_task(task.id, reason="no longer needed")
    assert cancelled.status == TaskStatus.CANCELLED

    stored = store.get_task(task.id)
    assert stored is not None
    assert stored.status == "CANCELLED"
    assert stored.error == "no longer needed"
    assert stored.assigned_to is None
    assert stored.metadata.get("claimed_by") is None
    assert sum(1 for queued in system.global_task_queue if queued.id == task.id) == 0


# ---------------------------------------------------------------------------
# New tests: task transitions
# ---------------------------------------------------------------------------


def test_valid_task_transitions_accepted():
    """ensure_valid_transition should not raise for valid state progressions."""
    valid_pairs = [
        ("PENDING", "ASSIGNED"),
        ("PENDING", "CANCELLED"),
        ("ASSIGNED", "RUNNING"),
        ("ASSIGNED", "PENDING"),
        ("ASSIGNED", "CANCELLED"),
        ("RUNNING", "COMPLETED"),
        ("RUNNING", "FAILED"),
        ("RUNNING", "CANCELLED"),
        ("RUNNING", "PENDING"),
        ("FAILED", "PENDING"),
    ]
    for current, new in valid_pairs:
        ensure_valid_transition(current, new)  # should not raise


def test_invalid_task_transitions_rejected():
    """ensure_valid_transition should raise ValueError for illegal transitions."""
    invalid_pairs = [
        ("PENDING", "COMPLETED"),
        ("PENDING", "RUNNING"),
        ("PENDING", "FAILED"),
        ("COMPLETED", "PENDING"),
        ("COMPLETED", "RUNNING"),
        ("COMPLETED", "FAILED"),
        ("CANCELLED", "PENDING"),
        ("CANCELLED", "RUNNING"),
        ("FAILED", "COMPLETED"),
        ("FAILED", "RUNNING"),
    ]
    for current, new in invalid_pairs:
        try:
            ensure_valid_transition(current, new)
            assert False, f"Expected ValueError for {current} -> {new}"
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# New tests: idempotent submission
# ---------------------------------------------------------------------------


def test_idempotent_submit_same_agent_returns_true():
    """Submitting the same task to the same agent a second time should succeed (idempotent)."""
    store = InMemoryTaskStore()
    system = AgentSystem("IdempotentSystem", task_store=store)
    agent = ExecutorAgent("Executor-1")
    system.add_agent(agent)

    task = system.create_task("Idempotent task", {"value": 1})
    assert system.submit_task(task, agent.id) is True
    # Second submit with the same agent_id should be idempotent.
    assert system.submit_task(task, agent.id) is True

    stored = store.get_task(task.id)
    assert stored is not None
    assert stored.status == "ASSIGNED"
    assert stored.assigned_to == agent.id


def test_idempotent_submit_different_agent_returns_false():
    """Submitting an already-ASSIGNED task to a different agent should fail."""
    store = InMemoryTaskStore()
    system = AgentSystem("IdempotentSystem2", task_store=store)
    agent_one = ExecutorAgent("Executor-1")
    agent_two = ExecutorAgent("Executor-2")
    system.add_agent(agent_one)
    system.add_agent(agent_two)

    task = system.create_task("Idempotent task 2", {"value": 2})
    assert system.submit_task(task, agent_one.id) is True
    assert system.submit_task(task, agent_two.id) is False


# ---------------------------------------------------------------------------
# New tests: concurrent submit
# ---------------------------------------------------------------------------


def test_concurrent_submit_only_one_succeeds():
    """Concurrent submissions of the same task should result in exactly one success."""
    store = InMemoryTaskStore()
    system = AgentSystem("ConcurrentSystem", task_store=store)
    agents = [ExecutorAgent(f"Executor-{i}") for i in range(5)]
    for a in agents:
        system.add_agent(a)

    task = system.create_task("Concurrent task", {"value": 99})
    results = []
    errors = []

    def try_submit(agent):
        try:
            results.append(system.submit_task(task, agent.id))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=try_submit, args=(a,)) for a in agents]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Unexpected exceptions during concurrent submit: {errors}"
    # Exactly one True among the results (one agent got the task).
    assert results.count(True) == 1
    stored = store.get_task(task.id)
    assert stored is not None
    assert stored.status == "ASSIGNED"


# ---------------------------------------------------------------------------
# New tests: lease metadata
# ---------------------------------------------------------------------------


def test_submit_task_sets_lease_metadata():
    """When default_lease_seconds is set, submitted tasks should carry lease metadata."""
    from datetime import datetime, timedelta

    store = InMemoryTaskStore()
    system = AgentSystem("LeaseSystem", task_store=store, default_lease_seconds=60.0)
    agent = ExecutorAgent("Executor-1")
    system.add_agent(agent)

    task = system.create_task("Lease task", {"value": 7})
    assert system.submit_task(task, agent.id) is True

    stored = store.get_task(task.id)
    assert stored is not None
    assert stored.metadata.get("claimed_by") == agent.id
    assert stored.metadata.get("claimed_at") is not None
    assert stored.metadata.get("heartbeat_at") is not None
    lease_str = stored.metadata.get("lease_expires_at")
    assert lease_str is not None
    lease_dt = datetime.fromisoformat(lease_str)
    # Use a tolerance: lease should be at least 55s in the future (generous buffer for CI).
    assert lease_dt >= datetime.now() + timedelta(seconds=55)


def test_heartbeat_task_refreshes_lease():
    """heartbeat_task should update heartbeat_at and extend lease_expires_at."""
    from datetime import datetime, timedelta

    store = InMemoryTaskStore()
    system = AgentSystem("HeartbeatSystem", task_store=store, default_lease_seconds=5.0)
    agent = ExecutorAgent("Executor-1")
    system.add_agent(agent)

    task = system.create_task("Heartbeat task", {"value": 8})
    assert system.submit_task(task, agent.id) is True

    stored_before = store.get_task(task.id)
    hb_before = stored_before.metadata.get("heartbeat_at")

    time.sleep(0.05)
    system.heartbeat_task(task.id, agent.id)

    stored_after = store.get_task(task.id)
    hb_after = stored_after.metadata.get("heartbeat_at")
    assert hb_after != hb_before
    lease_after = datetime.fromisoformat(stored_after.metadata["lease_expires_at"])
    # Lease should be at least 4s in the future (generous buffer for CI on a 5s lease).
    assert lease_after >= datetime.now() + timedelta(seconds=4)


def test_heartbeat_task_wrong_agent_raises():
    """heartbeat_task should raise ValueError when the wrong agent tries to refresh."""
    store = InMemoryTaskStore()
    system = AgentSystem("HBGuardSystem", task_store=store, default_lease_seconds=60.0)
    agent_one = ExecutorAgent("Executor-1")
    agent_two = ExecutorAgent("Executor-2")
    system.add_agent(agent_one)
    system.add_agent(agent_two)

    task = system.create_task("HB guard task", {"value": 9})
    assert system.submit_task(task, agent_one.id) is True

    try:
        system.heartbeat_task(task.id, agent_two.id)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "claimed by" in str(exc)


# ---------------------------------------------------------------------------
# New tests: lease-aware recovery
# ---------------------------------------------------------------------------


def test_recover_skips_tasks_with_active_lease():
    """Tasks with a future lease_expires_at should NOT be recovered."""
    store = InMemoryTaskStore()
    system = AgentSystem("LeaseRecoverySystem", task_store=store, default_lease_seconds=300.0)
    agent = ExecutorAgent("Executor-1")
    system.add_agent(agent)

    task = system.create_task("Active lease task", {"value": 10})
    assert system.submit_task(task, agent.id) is True

    # Manually force the task to RUNNING in the store (simulates mid-execution).
    stored = store.get_task(task.id)
    stored.status = "RUNNING"
    store.update_task(stored)

    # Task has an active lease (300s from now), should be skipped.
    recovered = system.recover_incomplete_tasks()
    assert recovered == 0

    still_stored = store.get_task(task.id)
    assert still_stored.status == "RUNNING"


def test_recover_resets_tasks_with_expired_lease():
    """Tasks with an expired lease_expires_at should be recovered."""
    from datetime import datetime, timedelta

    store = InMemoryTaskStore()
    system = AgentSystem("ExpiredLeaseSystem", task_store=store)
    agent = ExecutorAgent("Executor-1")
    system.add_agent(agent)

    task = system.create_task("Expired lease task", {"value": 11})
    assert system.submit_task(task, agent.id) is True

    # Backdate the lease to an already-expired time.
    stored = store.get_task(task.id)
    stored.status = "RUNNING"
    stored.metadata["lease_expires_at"] = (datetime.now() - timedelta(seconds=10)).isoformat()
    store.update_task(stored)

    recovered = system.recover_incomplete_tasks()
    assert recovered == 1

    reloaded = store.get_task(task.id)
    assert reloaded.status == "PENDING"
    assert reloaded.assigned_to is None


# ---------------------------------------------------------------------------
# New tests: timeout failure persistence
# ---------------------------------------------------------------------------


def test_timeout_failure_is_persisted():
    """A task that exceeds execution_timeout_seconds should be marked FAILED in the store."""
    store = InMemoryTaskStore()
    # Slow agent sleeps 0.3s; timeout is 0.01s (post-execution check).
    system = AgentSystem("TimeoutSystem", task_store=store, execution_timeout_seconds=0.05)
    agent = SlowExecutorAgent("SlowExec", sleep_seconds=0.2)
    system.add_agent(agent)

    task = system.create_task("Slow task", {"value": 12})
    assert system.submit_task(task, agent.id) is True

    try:
        system.execute_task(task.id, agent.id)
        assert False, "Expected TimeoutError"
    except TimeoutError as exc:
        assert "exceeded timeout" in str(exc)

    stored = store.get_task(task.id)
    assert stored is not None
    assert stored.status == "FAILED"
    assert "exceeded timeout" in (stored.error or "")


# ---------------------------------------------------------------------------
# New tests: cancel correctness
# ---------------------------------------------------------------------------


def test_cancel_running_task_raises():
    """cancel_task should raise ValueError for RUNNING tasks."""
    store = InMemoryTaskStore()
    system = AgentSystem("CancelRunningSystem", task_store=store)
    agent = ExecutorAgent("Executor-1")
    system.add_agent(agent)

    task = system.create_task("Running task", {"value": 13})
    assert system.submit_task(task, agent.id) is True

    # Force the task to RUNNING in the store directly.
    stored = store.get_task(task.id)
    stored.status = "RUNNING"
    store.update_task(stored)

    try:
        system.cancel_task(task.id)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "cannot be cancelled" in str(exc)


def test_requeue_cancelled_task_raises():
    """requeue_task should raise ValueError for CANCELLED tasks (terminal state)."""
    store = InMemoryTaskStore()
    system = AgentSystem("RequeueCancelledSystem", task_store=store)
    agent = ExecutorAgent("Executor-1")
    system.add_agent(agent)

    task = system.create_task("Cancelled task", {"value": 14})
    assert system.submit_task(task, agent.id) is True

    system.cancel_task(task.id, reason="test")

    try:
        system.requeue_task(task.id)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "cannot be requeued" in str(exc)


# ---------------------------------------------------------------------------
# New tests: stale/invalid persisted task recovery
# ---------------------------------------------------------------------------


def test_from_stored_task_unknown_priority_falls_back_to_normal():
    """_from_stored_task should not raise for unknown priority values."""
    from src.agents.super_agentic_agents import TaskPriority

    store = InMemoryTaskStore()
    system = AgentSystem("FallbackSystem", task_store=store)

    # Insert a task with a bogus priority directly into the store.
    bad_task = StoredTask(id="bad-priority-1", description="Bad priority task", priority="ULTRA_HIGH")
    store.create_task(bad_task)

    tasks = system.list_persisted_tasks()
    assert any(t.id == "bad-priority-1" for t in tasks)
    task = next(t for t in tasks if t.id == "bad-priority-1")
    assert task.priority == TaskPriority.NORMAL


def test_from_stored_task_unknown_status_falls_back_to_pending():
    """_from_stored_task should not raise for unknown status values (e.g., legacy migration)."""
    from datetime import datetime as _dt

    store = InMemoryTaskStore()
    system = AgentSystem("FallbackStatusSystem", task_store=store)

    # Construct a StoredTask that bypasses constructor validation to simulate a
    # legacy record with a status value not present in the current TaskStatus enum.
    legacy_stored = object.__new__(StoredTask)
    legacy_stored.id = "bad-status-1"
    legacy_stored.description = "Bad status task"
    legacy_stored.priority = "NORMAL"
    legacy_stored.assigned_to = None
    legacy_stored.status = "LEGACY_PROCESSING"
    legacy_stored.created_at = _dt.now()
    legacy_stored.completed_at = None
    legacy_stored.result = None
    legacy_stored.error = None
    legacy_stored.parameters = {}
    legacy_stored.dependencies = []
    legacy_stored.metadata = {}

    task = system._from_stored_task(legacy_stored)
    assert task.status == TaskStatus.PENDING
