"""
Phase 1 hardening tests for super_agentic_agents.

Validates features added as part of PR #5 conflict resolution:
- TaskStatus.DEPENDENCY_BLOCKED and DependencyError
- SystemMetrics (_Counter, _Timer) counters and timers
- _task_from_dict helper
- AgentSystem.save_snapshot / load_snapshot persistence
- AgentSystem.metrics structured observability
- AgentSystem.get_completed_task_ids utility
- BaseAgent.run_task dependency enforcement via completed_task_ids
- ExperienceReplay thread-safety stress tests
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, Set

import numpy as np
import pytest

from src.agents.super_agentic_agents import (
    AgentFactory,
    AgentRole,
    AgentSystem,
    AnalyzerAgent,
    BaseAgent,
    DependencyError,
    ExecutorAgent,
    LearnerAgent,
    OrchestratorAgent,
    SystemMetrics,
    Task,
    TaskPriority,
    TaskStatus,
    _task_from_dict,
)
from src.models.neural_network import AgentLearningModel, ExperienceReplay


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


class _SimpleAgent(BaseAgent):
    """Minimal concrete agent for testing BaseAgent behaviour."""

    def __init__(self, name: str = "Simple", fail: bool = False) -> None:
        super().__init__(name, role=AgentRole.EXECUTOR)
        self.fail = fail

    def think(self, input_data: Any) -> Dict[str, Any]:
        return {"input": input_data}

    def act(self, decision: Dict[str, Any]) -> Any:
        if self.fail:
            raise RuntimeError("deliberate failure")
        return {"ok": True}


@pytest.fixture()
def system() -> AgentSystem:
    return AgentSystem("TestSystem")


@pytest.fixture()
def executor(system: AgentSystem) -> ExecutorAgent:
    agent = ExecutorAgent("Exec-1")
    system.add_agent(agent)
    return agent


# ---------------------------------------------------------------------------
# 1. DependencyError and DEPENDENCY_BLOCKED
# ---------------------------------------------------------------------------


class TestDependencyError:
    def test_dependency_error_carries_task_id_and_unmet(self) -> None:
        err = DependencyError("task-123", ["dep-a", "dep-b"])
        assert err.task_id == "task-123"
        assert "dep-a" in err.unmet_dependencies
        assert "dep-b" in err.unmet_dependencies

    def test_dependency_error_message_contains_task_id(self) -> None:
        err = DependencyError("task-xyz", ["missing"])
        assert "task-xyz" in str(err)

    def test_dependency_blocked_status_exists(self) -> None:
        assert TaskStatus.DEPENDENCY_BLOCKED.value == "dependency_blocked"

    def test_run_task_with_unmet_dependencies_raises(self) -> None:
        agent = _SimpleAgent("dep-fail")
        task = Task(description="child", dependencies=["missing-id"])
        agent.assign_task(task)
        with pytest.raises(DependencyError) as exc_info:
            agent.run_task(task, completed_task_ids=set())
        err = exc_info.value
        assert "missing-id" in err.unmet_dependencies
        assert err.task_id == task.id

    def test_run_task_sets_dependency_blocked_status(self) -> None:
        agent = _SimpleAgent("status-check")
        task = Task(description="child", dependencies=["x", "y"])
        agent.assign_task(task)
        with pytest.raises(DependencyError):
            agent.run_task(task, completed_task_ids={"x"})
        assert task.status == TaskStatus.DEPENDENCY_BLOCKED
        assert "y" in task.error

    def test_run_task_with_met_dependencies_succeeds(self) -> None:
        agent = _SimpleAgent("dep-ok")
        task = Task(description="child", dependencies=["dep-1"])
        agent.assign_task(task)
        result = agent.run_task(task, completed_task_ids={"dep-1"})
        assert result is not None
        assert task.status == TaskStatus.COMPLETED

    def test_run_task_without_completed_ids_skips_check(self) -> None:
        """Backward-compatible: completed_task_ids=None skips dependency check."""
        agent = _SimpleAgent("backward-compat")
        task = Task(description="task", dependencies=["would-be-unmet"])
        agent.assign_task(task)
        result = agent.run_task(task, completed_task_ids=None)
        assert task.status == TaskStatus.COMPLETED

    def test_run_task_empty_dependencies_always_succeeds(self) -> None:
        agent = _SimpleAgent("no-deps")
        task = Task(description="nodeps")
        agent.assign_task(task)
        result = agent.run_task(task, completed_task_ids=set())
        assert task.status == TaskStatus.COMPLETED


# ---------------------------------------------------------------------------
# 2. SystemMetrics, _Counter, _Timer
# ---------------------------------------------------------------------------


class TestSystemMetrics:
    def test_counter_starts_at_zero(self) -> None:
        m = SystemMetrics()
        assert m.tasks_created.value == 0

    def test_counter_increments(self) -> None:
        m = SystemMetrics()
        m.tasks_created.increment()
        m.tasks_created.increment(3)
        assert m.tasks_created.value == 4

    def test_timer_avg_zero_on_no_samples(self) -> None:
        m = SystemMetrics()
        assert m.task_duration.avg == 0.0

    def test_timer_records_samples(self) -> None:
        m = SystemMetrics()
        m.task_duration.record(1.0)
        m.task_duration.record(3.0)
        assert m.task_duration.total == pytest.approx(4.0)
        assert m.task_duration.avg == pytest.approx(2.0)
        assert m.task_duration.count == 2

    def test_to_dict_contains_all_keys(self) -> None:
        m = SystemMetrics()
        d = m.to_dict()
        for key in (
            "tasks_created",
            "tasks_submitted",
            "tasks_completed",
            "tasks_failed",
            "tasks_dependency_blocked",
            "task_duration_avg_s",
            "task_duration_total_s",
        ):
            assert key in d, f"missing key: {key}"

    def test_counter_thread_safety(self) -> None:
        m = SystemMetrics()
        threads = [threading.Thread(target=lambda: m.tasks_created.increment()) for _ in range(200)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert m.tasks_created.value == 200


# ---------------------------------------------------------------------------
# 3. AgentSystem.metrics integration
# ---------------------------------------------------------------------------


class TestAgentSystemMetrics:
    def test_metrics_attribute_exists(self, system: AgentSystem) -> None:
        assert hasattr(system, "metrics")
        assert isinstance(system.metrics, SystemMetrics)

    def test_create_task_increments_structured_counter(self, system: AgentSystem) -> None:
        before = system.metrics.tasks_created.value
        system.create_task("t", {})
        assert system.metrics.tasks_created.value == before + 1

    def test_submit_task_increments_submitted_counter(
        self, system: AgentSystem, executor: ExecutorAgent
    ) -> None:
        before = system.metrics.tasks_submitted.value
        task = system.create_task("count-test", {})
        system.submit_task(task, agent_id=executor.id)
        assert system.metrics.tasks_submitted.value == before + 1

    def test_submit_blocked_task_increments_dependency_blocked(
        self, system: AgentSystem, executor: ExecutorAgent
    ) -> None:
        dep = system.create_task("dep", {})
        child = system.create_task("child", {}, dependencies=[dep.id])
        before = system.metrics.tasks_dependency_blocked.value
        system.submit_task(child, agent_id=executor.id)
        assert system.metrics.tasks_dependency_blocked.value == before + 1

    def test_get_system_status_includes_structured_metrics(self, system: AgentSystem) -> None:
        status = system.get_system_status()
        assert "structured_metrics" in status
        d = status["structured_metrics"]
        assert "tasks_created" in d


# ---------------------------------------------------------------------------
# 4. AgentSystem.get_completed_task_ids
# ---------------------------------------------------------------------------


class TestGetCompletedTaskIds:
    def test_empty_initially(self, system: AgentSystem) -> None:
        ids = system.get_completed_task_ids()
        assert isinstance(ids, set)

    def test_returns_ids_of_completed_tasks(self, system: AgentSystem) -> None:
        done = Task(id="done-1", status=TaskStatus.COMPLETED, description="done")
        system.completed_tasks.append(done)
        ids = system.get_completed_task_ids()
        assert "done-1" in ids


# ---------------------------------------------------------------------------
# 5. _task_from_dict
# ---------------------------------------------------------------------------


class TestTaskFromDict:
    def test_roundtrip_via_to_dict(self) -> None:
        t = Task(description="roundtrip", priority=TaskPriority.HIGH, dependencies=["x"])
        restored = _task_from_dict(t.to_dict())
        assert restored.id == t.id
        assert restored.description == t.description
        assert restored.priority == TaskPriority.HIGH
        assert "x" in restored.dependencies

    def test_unknown_priority_defaults_to_normal(self) -> None:
        t = _task_from_dict({"priority": "DOES_NOT_EXIST"})
        assert t.priority == TaskPriority.NORMAL

    def test_unknown_status_defaults_to_pending(self) -> None:
        t = _task_from_dict({"status": "unknown_value"})
        assert t.status == TaskStatus.PENDING

    def test_missing_fields_get_defaults(self) -> None:
        t = _task_from_dict({})
        assert t.description == "(restored)"
        assert t.priority == TaskPriority.NORMAL
        assert t.status == TaskStatus.PENDING
        assert t.dependencies == []

    def test_completed_at_parsed(self) -> None:
        from datetime import datetime
        ts = "2024-01-15T10:30:00"
        t = _task_from_dict({"completed_at": ts})
        assert t.completed_at == datetime.fromisoformat(ts)


# ---------------------------------------------------------------------------
# 6. AgentSystem save_snapshot / load_snapshot
# ---------------------------------------------------------------------------


class TestAgentSystemPersistence:
    def test_save_creates_file(self, tmp_path: Any, system: AgentSystem) -> None:
        filepath = str(tmp_path / "snapshot.json")
        system.save_snapshot(filepath)
        assert os.path.exists(filepath)

    def test_save_file_is_valid_json(self, tmp_path: Any, system: AgentSystem) -> None:
        filepath = str(tmp_path / "snapshot.json")
        system.save_snapshot(filepath)
        with open(filepath) as fh:
            data = json.load(fh)
        assert "name" in data
        assert "system_metrics" in data
        assert "structured_metrics" in data

    def test_save_snapshot_includes_completed_tasks(
        self, tmp_path: Any, system: AgentSystem
    ) -> None:
        done = Task(id="done-x", status=TaskStatus.COMPLETED, description="done")
        system.completed_tasks.append(done)
        filepath = str(tmp_path / "snap.json")
        system.save_snapshot(filepath)
        with open(filepath) as fh:
            data = json.load(fh)
        ids = [t["id"] for t in data["completed_tasks"]]
        assert "done-x" in ids

    def test_load_snapshot_restores_name_and_id(
        self, tmp_path: Any, system: AgentSystem
    ) -> None:
        filepath = str(tmp_path / "snap.json")
        original_id = system.id
        system.save_snapshot(filepath)
        loaded = AgentSystem.load_snapshot(filepath)
        assert loaded.name == system.name
        assert loaded.id == original_id

    def test_load_snapshot_restores_system_metrics(
        self, tmp_path: Any, system: AgentSystem
    ) -> None:
        system.create_task("m1", {})
        system.create_task("m2", {})
        filepath = str(tmp_path / "snap.json")
        system.save_snapshot(filepath)
        loaded = AgentSystem.load_snapshot(filepath)
        assert loaded.system_metrics["total_tasks"] == 2

    def test_load_snapshot_restores_completed_tasks(
        self, tmp_path: Any, system: AgentSystem
    ) -> None:
        done = Task(id="r-done", status=TaskStatus.COMPLETED, description="restored")
        system.completed_tasks.append(done)
        filepath = str(tmp_path / "snap.json")
        system.save_snapshot(filepath)
        loaded = AgentSystem.load_snapshot(filepath)
        ids = [t.id for t in loaded.completed_tasks]
        assert "r-done" in ids

    def test_load_snapshot_missing_optional_fields(self, tmp_path: Any) -> None:
        """Minimal snapshot without optional fields should still load."""
        minimal = {"name": "MinimalSystem", "id": "abc-123"}
        filepath = str(tmp_path / "minimal.json")
        with open(filepath, "w") as fh:
            json.dump(minimal, fh)
        loaded = AgentSystem.load_snapshot(filepath)
        assert loaded.name == "MinimalSystem"
        assert loaded.id == "abc-123"

    def test_save_snapshot_invalid_filepath_raises(self, system: AgentSystem) -> None:
        with pytest.raises(ValueError):
            system.save_snapshot("")

    def test_load_snapshot_nonexistent_raises(self, tmp_path: Any) -> None:
        with pytest.raises(FileNotFoundError):
            AgentSystem.load_snapshot(str(tmp_path / "does_not_exist_xyz.json"))

    def test_load_snapshot_invalid_filepath_raises(self) -> None:
        with pytest.raises(ValueError):
            AgentSystem.load_snapshot("")

    def test_snapshot_roundtrip_dependencies_preserved(
        self, tmp_path: Any, system: AgentSystem
    ) -> None:
        dep = system.create_task("dep", {})
        system.create_task("child", {}, dependencies=[dep.id])
        filepath = str(tmp_path / "snap.json")
        system.save_snapshot(filepath)
        with open(filepath) as fh:
            data = json.load(fh)
        queue_items = data.get("global_task_queue", [])
        child_data = next(
            (t for t in queue_items if t.get("description") == "child"), None
        )
        assert child_data is not None
        assert dep.id in child_data["dependencies"]


# ---------------------------------------------------------------------------
# 7. ExperienceReplay thread-safety stress tests
# ---------------------------------------------------------------------------


class TestExperienceReplayConcurrency:
    """Validate that ExperienceReplay is safe to use from multiple threads."""

    def _make_replay(self, size: int = 100, seed: int = 0) -> ExperienceReplay:
        return ExperienceReplay(state_size=4, max_size=size, seed=seed)

    def _add_n(self, replay: ExperienceReplay, n: int, rng_seed: int = 42) -> None:
        rng = np.random.default_rng(rng_seed)
        for _ in range(n):
            s = rng.standard_normal(4).astype(np.float32)
            ns = rng.standard_normal(4).astype(np.float32)
            replay.add(s, 0, 1.0, ns, False)

    def test_thread_safe_concurrent_add(self) -> None:
        replay = self._make_replay(size=500)
        threads = [
            threading.Thread(target=self._add_n, args=(replay, 50, i))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(replay) <= 500

    def test_thread_safe_concurrent_sample(self) -> None:
        replay = self._make_replay(size=200)
        self._add_n(replay, 50)
        errors: list = []

        def sample_worker() -> None:
            try:
                for _ in range(10):
                    replay.sample(5)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=sample_worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"Concurrent sample raised errors: {errors}"


# ---------------------------------------------------------------------------
# 8. AgentFactory
# ---------------------------------------------------------------------------


class TestAgentFactory:
    def test_create_executor(self) -> None:
        agent = AgentFactory.create_agent("executor", "Exec")
        assert isinstance(agent, ExecutorAgent)

    def test_create_analyzer(self) -> None:
        agent = AgentFactory.create_agent("analyzer", "Ana")
        assert isinstance(agent, AnalyzerAgent)

    def test_create_learner(self) -> None:
        agent = AgentFactory.create_agent("learner", "Lear")
        assert isinstance(agent, LearnerAgent)

    def test_create_unknown_returns_none(self) -> None:
        agent = AgentFactory.create_agent("unknown_type_xyz", "X")
        assert agent is None

    def test_create_team(self) -> None:
        team = AgentFactory.create_team(
            {"executor": 1, "analyzer": 1, "learner": 1}
        )
        # create_team returns an AgentSystem with agents added
        # orchestrator + 3 agents = at least 3 agents in system
        assert len(team.agents) >= 3
