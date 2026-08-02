"""
Hardening unit tests for task_store module.

Covers:
- StoredTask validation (field constraints, status normalization)
- InMemoryTaskStore CRUD semantics and defensive copies
- Task status transition enforcement (valid and invalid paths)
- Concurrent write safety via threading
- list_tasks filtering and chronological ordering
- normalize_task_status and ensure_valid_transition helpers
- Edge cases: empty strings, None fields, boundary conditions
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import List

import pytest

from src.agents.task_store import (
    ALLOWED_TASK_STATUSES,
    ALLOWED_TASK_TRANSITIONS,
    InMemoryTaskStore,
    StoredTask,
    ensure_valid_transition,
    normalize_task_status,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(
    task_id: str = "t-1",
    description: str = "Test task",
    priority: str = "NORMAL",
    status: str = "PENDING",
) -> StoredTask:
    return StoredTask(id=task_id, description=description, priority=priority, status=status)


# ===========================================================================
# StoredTask validation
# ===========================================================================

class TestStoredTaskValidation:
    def test_valid_task_is_created(self):
        task = _make_task()
        assert task.id == "t-1"
        assert task.status == "PENDING"

    def test_empty_id_raises(self):
        with pytest.raises(ValueError, match="id cannot be empty"):
            StoredTask(id="   ", description="desc", priority="NORMAL")

    def test_empty_description_raises(self):
        with pytest.raises(ValueError, match="description cannot be empty"):
            StoredTask(id="t-1", description="   ", priority="NORMAL")

    def test_description_is_stripped(self):
        task = StoredTask(id="t-1", description="  hello  ", priority="NORMAL")
        assert task.description == "hello"

    def test_invalid_status_raises(self):
        with pytest.raises(ValueError, match="Invalid task status"):
            StoredTask(id="t-1", description="desc", priority="NORMAL", status="UNKNOWN")

    def test_status_is_normalised_to_uppercase(self):
        task = StoredTask(id="t-1", description="desc", priority="NORMAL", status="pending")
        assert task.status == "PENDING"

    def test_completed_at_before_created_at_raises(self):
        now = datetime.now()
        with pytest.raises(ValueError, match="completed_at cannot be earlier"):
            StoredTask(
                id="t-1",
                description="desc",
                priority="NORMAL",
                created_at=now,
                completed_at=now - timedelta(seconds=1),
            )

    def test_completed_at_equal_to_created_at_is_allowed(self):
        now = datetime.now()
        task = StoredTask(
            id="t-1",
            description="desc",
            priority="NORMAL",
            created_at=now,
            completed_at=now,
        )
        assert task.completed_at == now

    def test_all_valid_statuses_accepted(self):
        for status in ALLOWED_TASK_STATUSES:
            task = _make_task(status=status)
            assert task.status == status

    def test_defaults_are_set(self):
        task = _make_task()
        assert task.assigned_to is None
        assert task.completed_at is None
        assert task.result is None
        assert task.error is None
        assert task.parameters == {}
        assert task.dependencies == []
        assert task.metadata == {}


# ===========================================================================
# normalize_task_status
# ===========================================================================

class TestNormalizeTaskStatus:
    @pytest.mark.parametrize("raw", ["PENDING", "pending", " Pending ", "RUNNING"])
    def test_normalises_case_and_whitespace(self, raw: str):
        result = normalize_task_status(raw)
        assert result == raw.strip().upper()

    def test_invalid_status_raises(self):
        with pytest.raises(ValueError, match="Invalid task status"):
            normalize_task_status("MADE_UP")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            normalize_task_status("")


# ===========================================================================
# ensure_valid_transition
# ===========================================================================

class TestEnsureValidTransition:
    def test_same_status_is_allowed(self):
        for status in ALLOWED_TASK_STATUSES:
            ensure_valid_transition(status, status)  # must not raise

    @pytest.mark.parametrize("from_status,to_status", [
        ("PENDING", "ASSIGNED"),
        ("PENDING", "CANCELLED"),
        ("ASSIGNED", "RUNNING"),
        ("ASSIGNED", "PENDING"),
        ("ASSIGNED", "CANCELLED"),
        ("RUNNING", "COMPLETED"),
        ("RUNNING", "FAILED"),
        ("RUNNING", "PENDING"),
        ("RUNNING", "CANCELLED"),
        ("FAILED", "PENDING"),
    ])
    def test_allowed_transitions_do_not_raise(self, from_status: str, to_status: str):
        ensure_valid_transition(from_status, to_status)

    @pytest.mark.parametrize("from_status,to_status", [
        ("PENDING", "RUNNING"),
        ("PENDING", "COMPLETED"),
        ("PENDING", "FAILED"),
        ("COMPLETED", "PENDING"),
        ("COMPLETED", "RUNNING"),
        ("COMPLETED", "FAILED"),
        ("CANCELLED", "PENDING"),
        ("CANCELLED", "RUNNING"),
        ("FAILED", "RUNNING"),
        ("FAILED", "COMPLETED"),
    ])
    def test_illegal_transitions_raise(self, from_status: str, to_status: str):
        with pytest.raises(ValueError, match="Invalid task transition"):
            ensure_valid_transition(from_status, to_status)

    def test_transitions_table_is_complete(self):
        for status in ALLOWED_TASK_STATUSES:
            assert status in ALLOWED_TASK_TRANSITIONS, f"{status} missing from ALLOWED_TASK_TRANSITIONS"


# ===========================================================================
# InMemoryTaskStore – basic CRUD
# ===========================================================================

class TestInMemoryTaskStoreCRUD:
    def test_create_and_get(self):
        store = InMemoryTaskStore()
        task = _make_task()
        store.create_task(task)
        loaded = store.get_task("t-1")
        assert loaded is not None
        assert loaded.id == "t-1"
        assert loaded.status == "PENDING"

    def test_get_nonexistent_returns_none(self):
        store = InMemoryTaskStore()
        assert store.get_task("no-such-id") is None

    def test_create_duplicate_raises(self):
        store = InMemoryTaskStore()
        store.create_task(_make_task())
        with pytest.raises(ValueError, match="already exists"):
            store.create_task(_make_task())

    def test_update_existing_task(self):
        store = InMemoryTaskStore()
        store.create_task(_make_task())
        loaded = store.get_task("t-1")
        assert loaded is not None
        loaded.status = "ASSIGNED"
        store.update_task(loaded)
        reloaded = store.get_task("t-1")
        assert reloaded is not None
        assert reloaded.status == "ASSIGNED"

    def test_update_nonexistent_raises(self):
        store = InMemoryTaskStore()
        with pytest.raises(KeyError, match="not found"):
            store.update_task(_make_task(task_id="ghost"))

    def test_update_enforces_transition(self):
        store = InMemoryTaskStore()
        store.create_task(_make_task())
        loaded = store.get_task("t-1")
        assert loaded is not None
        loaded.status = "COMPLETED"
        with pytest.raises(ValueError, match="Invalid task transition"):
            store.update_task(loaded)

    def test_full_lifecycle_pending_to_completed(self):
        store = InMemoryTaskStore()
        store.create_task(_make_task())

        def _update(new_status: str) -> None:
            t = store.get_task("t-1")
            assert t is not None
            t.status = new_status
            store.update_task(t)

        _update("ASSIGNED")
        _update("RUNNING")
        t = store.get_task("t-1")
        assert t is not None
        t.status = "COMPLETED"
        t.completed_at = datetime.now()
        store.update_task(t)

        final = store.get_task("t-1")
        assert final is not None
        assert final.status == "COMPLETED"
        assert final.completed_at is not None


# ===========================================================================
# InMemoryTaskStore – defensive copies
# ===========================================================================

class TestInMemoryTaskStoreDefensiveCopies:
    def test_mutating_returned_task_does_not_affect_store(self):
        store = InMemoryTaskStore()
        store.create_task(_make_task())
        first = store.get_task("t-1")
        assert first is not None

        first.metadata["injected"] = True
        first.dependencies.append("dep-99")
        first.parameters["x"] = 42

        second = store.get_task("t-1")
        assert second is not None
        assert "injected" not in second.metadata
        assert "dep-99" not in second.dependencies
        assert "x" not in second.parameters

    def test_mutating_original_task_after_create_does_not_affect_store(self):
        store = InMemoryTaskStore()
        task = _make_task()
        store.create_task(task)
        task.metadata["poison"] = True

        loaded = store.get_task("t-1")
        assert loaded is not None
        assert "poison" not in loaded.metadata


# ===========================================================================
# InMemoryTaskStore – list_tasks
# ===========================================================================

class TestInMemoryTaskStoreListTasks:
    def _build_store(self) -> InMemoryTaskStore:
        store = InMemoryTaskStore()
        for i in range(5):
            t = StoredTask(
                id=f"t-{i}",
                description=f"Task {i}",
                priority="NORMAL",
                created_at=datetime.now() + timedelta(seconds=i),
            )
            store.create_task(t)
        return store

    def test_list_all_returns_all(self):
        store = self._build_store()
        tasks = store.list_tasks()
        assert len(tasks) == 5

    def test_list_all_ordered_newest_first(self):
        store = self._build_store()
        tasks = store.list_tasks()
        timestamps = [t.created_at for t in tasks]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_list_by_status_filters_correctly(self):
        store = InMemoryTaskStore()
        for i in range(3):
            store.create_task(_make_task(task_id=f"p-{i}", description=f"Task {i}"))

        # Assign the first task
        t = store.get_task("p-0")
        assert t is not None
        t.status = "ASSIGNED"
        store.update_task(t)

        pending = store.list_tasks(status="PENDING")
        assigned = store.list_tasks(status="ASSIGNED")
        assert len(pending) == 2
        assert len(assigned) == 1
        assert assigned[0].id == "p-0"

    def test_list_empty_store_returns_empty(self):
        store = InMemoryTaskStore()
        assert store.list_tasks() == []

    def test_list_by_status_case_insensitive(self):
        store = InMemoryTaskStore()
        store.create_task(_make_task())
        assert len(store.list_tasks(status="pending")) == 1
        assert len(store.list_tasks(status="PENDING")) == 1

    def test_list_by_invalid_status_raises(self):
        store = InMemoryTaskStore()
        with pytest.raises(ValueError, match="Invalid task status"):
            store.list_tasks(status="BOGUS")


# ===========================================================================
# InMemoryTaskStore – concurrency
# ===========================================================================

class TestInMemoryTaskStoreConcurrency:
    def test_concurrent_creates_only_first_succeeds(self):
        store = InMemoryTaskStore()
        errors: List[Exception] = []
        successes: List[int] = []

        def _try_create(idx: int) -> None:
            try:
                store.create_task(_make_task(task_id="shared-id", description=f"Worker {idx}"))
                successes.append(idx)
            except ValueError:
                errors.append(idx)  # type: ignore[arg-type]

        threads = [threading.Thread(target=_try_create, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(successes) == 1
        assert len(errors) == 9

    def test_concurrent_updates_are_serialised(self):
        store = InMemoryTaskStore()
        store.create_task(_make_task())

        barrier = threading.Barrier(2)
        results: List[str] = []

        def _worker(new_status: str) -> None:
            barrier.wait()
            t = store.get_task("t-1")
            assert t is not None
            t.status = new_status
            try:
                store.update_task(t)
                results.append(new_status)
            except (ValueError, KeyError):
                pass

        t1 = threading.Thread(target=_worker, args=("ASSIGNED",))
        t2 = threading.Thread(target=_worker, args=("CANCELLED",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Exactly one update must have won
        assert len(results) == 1
        final = store.get_task("t-1")
        assert final is not None
        assert final.status in {"ASSIGNED", "CANCELLED"}

    def test_many_tasks_created_by_different_threads(self):
        store = InMemoryTaskStore()
        n = 50

        def _create(idx: int) -> None:
            store.create_task(_make_task(task_id=f"task-{idx}", description=f"Task {idx}"))

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(_create, i) for i in range(n)]
            for f in as_completed(futures):
                f.result()

        assert len(store.list_tasks()) == n


# ===========================================================================
# InMemoryTaskStore – metadata and result fields
# ===========================================================================

class TestInMemoryTaskStoreFields:
    def test_metadata_is_stored_and_retrieved(self):
        store = InMemoryTaskStore()
        task = StoredTask(
            id="t-meta",
            description="meta task",
            priority="HIGH",
            metadata={"key": "value", "count": 1},
        )
        store.create_task(task)
        loaded = store.get_task("t-meta")
        assert loaded is not None
        assert loaded.metadata == {"key": "value", "count": 1}

    def test_dependencies_are_stored_and_retrieved(self):
        store = InMemoryTaskStore()
        task = StoredTask(
            id="t-deps",
            description="dep task",
            priority="NORMAL",
            dependencies=["dep-a", "dep-b"],
        )
        store.create_task(task)
        loaded = store.get_task("t-deps")
        assert loaded is not None
        assert loaded.dependencies == ["dep-a", "dep-b"]

    def test_result_and_error_persisted_on_update(self):
        store = InMemoryTaskStore()
        store.create_task(_make_task())

        t = store.get_task("t-1")
        assert t is not None
        t.status = "ASSIGNED"
        store.update_task(t)

        t = store.get_task("t-1")
        assert t is not None
        t.status = "RUNNING"
        store.update_task(t)

        t = store.get_task("t-1")
        assert t is not None
        t.status = "FAILED"
        t.error = "something went wrong"
        store.update_task(t)

        loaded = store.get_task("t-1")
        assert loaded is not None
        assert loaded.status == "FAILED"
        assert loaded.error == "something went wrong"

    def test_assigned_to_persisted(self):
        store = InMemoryTaskStore()
        store.create_task(_make_task())
        t = store.get_task("t-1")
        assert t is not None
        t.status = "ASSIGNED"
        t.assigned_to = "agent-42"
        store.update_task(t)

        loaded = store.get_task("t-1")
        assert loaded is not None
        assert loaded.assigned_to == "agent-42"


# ===========================================================================
# Transition coverage – terminal states cannot be left
# ===========================================================================

class TestTerminalStates:
    @pytest.mark.parametrize("terminal", ["COMPLETED", "CANCELLED"])
    def test_terminal_state_rejects_all_transitions(self, terminal: str):
        allowed = ALLOWED_TASK_TRANSITIONS[terminal]
        assert allowed == set(), f"{terminal} should have no allowed outgoing transitions"

    def test_failed_can_only_go_to_pending(self):
        assert ALLOWED_TASK_TRANSITIONS["FAILED"] == {"PENDING"}

    def test_completed_task_cannot_be_updated_to_any_other_status(self):
        store = InMemoryTaskStore()
        store.create_task(_make_task())

        def _advance(new_status: str) -> None:
            t = store.get_task("t-1")
            assert t is not None
            t.status = new_status
            store.update_task(t)

        _advance("ASSIGNED")
        _advance("RUNNING")
        _advance("COMPLETED")

        for bad_status in ("PENDING", "ASSIGNED", "RUNNING", "FAILED", "CANCELLED"):
            t = store.get_task("t-1")
            assert t is not None
            t.status = bad_status
            with pytest.raises(ValueError, match="Invalid task transition"):
                store.update_task(t)
