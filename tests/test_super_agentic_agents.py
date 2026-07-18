"""
Tests for src/agents/super_agentic_agents.py.

Covers:
- AgentSystem task creation / submission / assignment paths
- Task dependency enforcement and DependencyError behavior
- AgentSystem JSON snapshot save / load persistence
- ExperienceReplay add / sample / __len__ with thread-safety checks
- AgentLearningModel validation / decay / target-sync (selected paths)
- Structured metrics (SystemMetrics) counters and timers
- Edge cases and failure paths
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict

import numpy as np
import pytest

from src.agents.super_agentic_agents import (
    AgentCapability,
    AgentFactory,
    AgentRole,
    AgentStatus,
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
    """Minimal concrete agent for testing BaseAgent behavior."""

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
# 1. AgentSystem – task creation
# ---------------------------------------------------------------------------


class TestAgentSystemTaskCreation:
    def test_create_task_returns_task_with_correct_fields(self, system: AgentSystem) -> None:
        task = system.create_task("do something", {"key": "value"})
        assert isinstance(task, Task)
        assert task.description == "do something"
        assert task.parameters == {"key": "value"}
        assert task.status == TaskStatus.PENDING
        assert task.priority == TaskPriority.NORMAL

    def test_create_task_increments_total_tasks(self, system: AgentSystem) -> None:
        before = system.system_metrics["total_tasks"]
        system.create_task("t1", {})
        system.create_task("t2", {})
        assert system.system_metrics["total_tasks"] == before + 2

    def test_create_task_increments_structured_counter(self, system: AgentSystem) -> None:
        before = system.metrics.tasks_created.value
        system.create_task("t", {})
        assert system.metrics.tasks_created.value == before + 1

    def test_create_task_appends_to_queue(self, system: AgentSystem) -> None:
        before = len(system.global_task_queue)
        task = system.create_task("q-task", {})
        assert len(system.global_task_queue) == before + 1
        assert system.global_task_queue[-1].id == task.id

    def test_create_task_with_dependencies(self, system: AgentSystem) -> None:
        dep = system.create_task("dep-task", {})
        child = system.create_task("child-task", {}, dependencies=[dep.id])
        assert dep.id in child.dependencies

    def test_create_task_with_priority(self, system: AgentSystem) -> None:
        task = system.create_task("high-priority", {}, priority=TaskPriority.HIGH)
        assert task.priority == TaskPriority.HIGH


# ---------------------------------------------------------------------------
# 2. AgentSystem – task submission / assignment
# ---------------------------------------------------------------------------


class TestAgentSystemTaskSubmission:
    def test_submit_task_to_specific_agent(self, system: AgentSystem, executor: ExecutorAgent) -> None:
        task = system.create_task("submit-test", {})
        result = system.submit_task(task, agent_id=executor.id)
        assert result is True
        assert task.status == TaskStatus.ASSIGNED
        assert task.assigned_to == executor.id

    def test_submit_task_to_unknown_agent_returns_false(self, system: AgentSystem) -> None:
        task = system.create_task("unknown-agent", {})
        result = system.submit_task(task, agent_id="nonexistent-uuid")
        assert result is False

    def test_submit_task_via_orchestrator(self, system: AgentSystem) -> None:
        agent = ExecutorAgent("Exec-orch")
        system.add_agent(agent)
        task = system.create_task("orch-submit", {})
        result = system.submit_task(task)
        # Orchestrator assigns to managed agent; task should be assigned
        assert result is True
        assert task.status == TaskStatus.ASSIGNED

    def test_submit_task_increments_counter(self, system: AgentSystem, executor: ExecutorAgent) -> None:
        before = system.metrics.tasks_submitted.value
        task = system.create_task("count-test", {})
        system.submit_task(task, agent_id=executor.id)
        assert system.metrics.tasks_submitted.value == before + 1

    def test_add_and_remove_agent(self, system: AgentSystem) -> None:
        agent = AnalyzerAgent("Temp")
        assert system.add_agent(agent) is True
        assert agent.id in system.agents
        assert system.remove_agent(agent.id) is True
        assert agent.id not in system.agents

    def test_remove_nonexistent_agent_returns_false(self, system: AgentSystem) -> None:
        assert system.remove_agent("no-such-id") is False


# ---------------------------------------------------------------------------
# 3. Task dependency enforcement
# ---------------------------------------------------------------------------


class TestDependencyEnforcement:
    def test_execute_task_with_met_dependencies_succeeds(self) -> None:
        agent = _SimpleAgent("dep-ok")
        dep_task = Task(id="dep-1", description="dep", status=TaskStatus.COMPLETED)
        child_task = Task(description="child", dependencies=["dep-1"])
        agent.assign_task(child_task)

        result = agent.execute_task(child_task, completed_task_ids={"dep-1"})
        assert result is not None
        assert child_task.status == TaskStatus.COMPLETED

    def test_execute_task_with_unmet_dependencies_raises_dependency_error(self) -> None:
        agent = _SimpleAgent("dep-fail")
        child_task = Task(description="child", dependencies=["missing-id"])
        agent.assign_task(child_task)

        with pytest.raises(DependencyError) as exc_info:
            agent.execute_task(child_task, completed_task_ids=set())

        err = exc_info.value
        assert "missing-id" in err.unmet_dependencies
        assert err.task_id == child_task.id

    def test_execute_task_dependency_blocked_status_set(self) -> None:
        agent = _SimpleAgent("status-check")
        child_task = Task(description="child", dependencies=["x", "y"])
        agent.assign_task(child_task)

        with pytest.raises(DependencyError):
            agent.execute_task(child_task, completed_task_ids={"x"})

        assert child_task.status == TaskStatus.DEPENDENCY_BLOCKED
        assert "y" in child_task.error

    def test_execute_task_without_completed_ids_skips_check(self) -> None:
        """Backward-compatible: when completed_task_ids is None, skip dependency check."""
        agent = _SimpleAgent("backward-compat")
        task = Task(description="task", dependencies=["would-be-unmet"])
        agent.assign_task(task)

        result = agent.execute_task(task, completed_task_ids=None)
        assert task.status == TaskStatus.COMPLETED

    def test_execute_task_empty_dependencies_always_succeeds(self) -> None:
        agent = _SimpleAgent("no-deps")
        task = Task(description="nodeps")
        agent.assign_task(task)

        result = agent.execute_task(task, completed_task_ids=set())
        assert task.status == TaskStatus.COMPLETED

    def test_dependency_error_message_contains_task_id(self) -> None:
        agent = _SimpleAgent("msg-check")
        task = Task(description="msg", dependencies=["absent"])
        agent.assign_task(task)

        with pytest.raises(DependencyError) as exc_info:
            agent.execute_task(task, completed_task_ids=set())

        assert task.id in str(exc_info.value)

    def test_get_completed_task_ids(self, system: AgentSystem) -> None:
        done = Task(id="done-1", status=TaskStatus.COMPLETED)
        pending = Task(id="pend-1", status=TaskStatus.PENDING)
        with system._queue_lock:
            system.completed_tasks.extend([done, pending])

        ids = system.get_completed_task_ids()
        assert "done-1" in ids
        assert "pend-1" not in ids


# ---------------------------------------------------------------------------
# 4. AgentSystem persistence – save / load
# ---------------------------------------------------------------------------


class TestAgentSystemPersistence:
    def test_save_creates_file(self, tmp_path, system: AgentSystem) -> None:
        filepath = str(tmp_path / "snapshot.json")
        system.save_snapshot(filepath)
        assert os.path.exists(filepath)

    def test_save_file_is_valid_json(self, tmp_path, system: AgentSystem) -> None:
        filepath = str(tmp_path / "snapshot.json")
        system.save_snapshot(filepath)
        with open(filepath) as fh:
            data = json.load(fh)
        assert "name" in data
        assert "system_metrics" in data

    def test_save_snapshot_includes_task_queue(self, tmp_path, system: AgentSystem) -> None:
        system.create_task("queued-task", {"x": 1})
        filepath = str(tmp_path / "snap.json")
        system.save_snapshot(filepath)

        with open(filepath) as fh:
            data = json.load(fh)

        assert len(data["global_task_queue"]) == 1
        assert data["global_task_queue"][0]["description"] == "queued-task"

    def test_save_snapshot_includes_completed_tasks(self, tmp_path, system: AgentSystem) -> None:
        done = Task(id="done-x", status=TaskStatus.COMPLETED, description="done")
        system.completed_tasks.append(done)
        filepath = str(tmp_path / "snap.json")
        system.save_snapshot(filepath)

        with open(filepath) as fh:
            data = json.load(fh)

        ids = [t["id"] for t in data["completed_tasks"]]
        assert "done-x" in ids

    def test_load_snapshot_restores_name_and_id(self, tmp_path, system: AgentSystem) -> None:
        filepath = str(tmp_path / "snap.json")
        original_id = system.id
        system.save_snapshot(filepath)

        loaded = AgentSystem.load_snapshot(filepath)
        assert loaded.name == system.name
        assert loaded.id == original_id

    def test_load_snapshot_restores_system_metrics(self, tmp_path, system: AgentSystem) -> None:
        system.create_task("m1", {})
        system.create_task("m2", {})
        filepath = str(tmp_path / "snap.json")
        system.save_snapshot(filepath)

        loaded = AgentSystem.load_snapshot(filepath)
        assert loaded.system_metrics["total_tasks"] == 2

    def test_load_snapshot_restores_completed_tasks(self, tmp_path, system: AgentSystem) -> None:
        done = Task(id="r-done", status=TaskStatus.COMPLETED, description="restored")
        system.completed_tasks.append(done)
        filepath = str(tmp_path / "snap.json")
        system.save_snapshot(filepath)

        loaded = AgentSystem.load_snapshot(filepath)
        ids = [t.id for t in loaded.completed_tasks]
        assert "r-done" in ids

    def test_load_snapshot_restores_queue(self, tmp_path, system: AgentSystem) -> None:
        system.create_task("queue-item", {"p": 1})
        filepath = str(tmp_path / "snap.json")
        system.save_snapshot(filepath)

        loaded = AgentSystem.load_snapshot(filepath)
        assert len(loaded.global_task_queue) == 1
        assert loaded.global_task_queue[0].description == "queue-item"

    def test_load_snapshot_missing_optional_fields(self, tmp_path) -> None:
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

    def test_load_snapshot_nonexistent_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            AgentSystem.load_snapshot(str(tmp_path / "does_not_exist_xyz.json"))

    def test_load_snapshot_invalid_filepath_raises(self) -> None:
        with pytest.raises(ValueError):
            AgentSystem.load_snapshot("")

    def test_snapshot_roundtrip_dependencies_preserved(self, tmp_path, system: AgentSystem) -> None:
        dep = system.create_task("dep", {})
        child = system.create_task("child", {}, dependencies=[dep.id])
        filepath = str(tmp_path / "snap.json")
        system.save_snapshot(filepath)

        loaded = AgentSystem.load_snapshot(filepath)
        child_restored = next(t for t in loaded.global_task_queue if t.description == "child")
        assert dep.id in child_restored.dependencies


# ---------------------------------------------------------------------------
# 5. ExperienceReplay – add / sample / __len__
# ---------------------------------------------------------------------------


class TestExperienceReplayBehavior:
    """Extended behavior and concurrency tests for ExperienceReplay."""

    def _make_replay(self, size: int = 10, seed: int = 0) -> ExperienceReplay:
        return ExperienceReplay(state_size=4, max_size=size, seed=seed)

    def _add_n(self, replay: ExperienceReplay, n: int) -> None:
        rng = np.random.default_rng(42)
        for _ in range(n):
            s = rng.standard_normal(4).astype(np.float32)
            ns = rng.standard_normal(4).astype(np.float32)
            replay.add(s, 0, 1.0, ns, False)

    def test_len_starts_at_zero(self) -> None:
        assert len(self._make_replay()) == 0

    def test_add_increments_len(self) -> None:
        replay = self._make_replay()
        self._add_n(replay, 3)
        assert len(replay) == 3

    def test_len_capped_at_max_size(self) -> None:
        replay = self._make_replay(size=5)
        self._add_n(replay, 20)
        assert len(replay) == 5

    def test_sample_returns_correct_shapes(self) -> None:
        replay = self._make_replay(size=20, seed=7)
        self._add_n(replay, 10)
        states, actions, rewards, next_states, dones = replay.sample(5)
        assert states.shape == (5, 4)
        assert actions.shape == (5,)
        assert rewards.shape == (5,)
        assert next_states.shape == (5, 4)
        assert dones.shape == (5,)

    def test_sample_clamps_to_buffer_size(self) -> None:
        replay = self._make_replay(size=20)
        self._add_n(replay, 3)
        states, actions, _, _, _ = replay.sample(100)
        assert states.shape[0] == 3

    def test_sample_empty_raises(self) -> None:
        replay = self._make_replay()
        with pytest.raises(ValueError, match="empty"):
            replay.sample(1)

    def test_add_wrong_state_shape_raises(self) -> None:
        replay = self._make_replay()
        good = np.zeros(4, dtype=np.float32)
        bad = np.zeros(5, dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            replay.add(bad, 0, 0.0, good, False)

    def test_add_nan_state_raises(self) -> None:
        replay = self._make_replay()
        bad = np.array([np.nan, 0.0, 0.0, 0.0], dtype=np.float32)
        good = np.zeros(4, dtype=np.float32)
        with pytest.raises(ValueError, match="NaN"):
            replay.add(bad, 0, 0.0, good, False)

    def test_add_non_integer_action_raises(self) -> None:
        replay = self._make_replay()
        s = np.zeros(4, dtype=np.float32)
        with pytest.raises(TypeError, match="integer"):
            replay.add(s, "0", 0.0, s, False)  # type: ignore[arg-type]

    def test_add_infinite_reward_raises(self) -> None:
        replay = self._make_replay()
        s = np.zeros(4, dtype=np.float32)
        with pytest.raises(ValueError, match="finite"):
            replay.add(s, 0, np.inf, s, False)

    def test_overwrite_oldest_on_overflow(self) -> None:
        """Ring-buffer behavior: oldest experience is overwritten first."""
        replay = ExperienceReplay(state_size=1, max_size=2, seed=0)
        s_a = np.array([1.0], dtype=np.float32)
        s_b = np.array([2.0], dtype=np.float32)
        s_c = np.array([3.0], dtype=np.float32)
        replay.add(s_a, 0, 1.0, s_b, False)
        replay.add(s_b, 0, 2.0, s_c, False)
        replay.add(s_c, 0, 3.0, s_a, False)
        assert len(replay) == 2
        # Buffer contains s_c and s_b (position 0 was overwritten by s_c)
        buffer_rewards = {exp[2] for exp in replay.buffer}
        assert 1.0 not in buffer_rewards  # s_a overwritten

    def test_thread_safe_concurrent_add(self) -> None:
        """Concurrent adds from multiple threads must not corrupt buffer."""
        replay = ExperienceReplay(state_size=4, max_size=1000, seed=0)
        errors: list[Exception] = []

        def worker(n: int) -> None:
            rng = np.random.default_rng(n)
            for _ in range(50):
                s = rng.standard_normal(4).astype(np.float32)
                try:
                    replay.add(s, 0, 1.0, s, False)
                except Exception as exc:
                    errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Unexpected errors: {errors}"
        assert 0 < len(replay) <= 1000

    def test_thread_safe_concurrent_sample(self) -> None:
        """Sample while concurrent adds are happening must not raise."""
        replay = ExperienceReplay(state_size=4, max_size=500, seed=1)
        # Pre-fill so sample doesn't hit empty buffer immediately
        rng = np.random.default_rng(99)
        for _ in range(100):
            s = rng.standard_normal(4).astype(np.float32)
            replay.add(s, 0, 0.0, s, False)

        sample_errors: list[Exception] = []

        def sampler() -> None:
            for _ in range(20):
                try:
                    replay.sample(10)
                except Exception as exc:
                    sample_errors.append(exc)

        def adder() -> None:
            r = np.random.default_rng(77)
            for _ in range(50):
                s = r.standard_normal(4).astype(np.float32)
                try:
                    replay.add(s, 0, 1.0, s, False)
                except Exception as exc:
                    sample_errors.append(exc)

        threads = [threading.Thread(target=sampler) for _ in range(3)]
        threads += [threading.Thread(target=adder) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not sample_errors, f"Concurrency errors: {sample_errors}"


# ---------------------------------------------------------------------------
# 6. AgentLearningModel – validation, epsilon decay, target-sync
# ---------------------------------------------------------------------------


class TestAgentLearningModelSelected:
    """Targeted tests for paths not already covered in tests/models/."""

    def test_target_sync_on_interval(self) -> None:
        """After target_update_interval steps, target syncs automatically."""
        model = AgentLearningModel(
            state_size=4, action_size=2, target_update_interval=2, seed=5
        )
        batch_size = 8
        states = np.random.randn(batch_size, 4).astype(np.float32)
        actions = np.zeros(batch_size, dtype=np.int32)
        rewards = np.ones(batch_size, dtype=np.float32)
        next_states = np.random.randn(batch_size, 4).astype(np.float32)
        dones = np.zeros(batch_size, dtype=np.float32)

        model.train_step(states, actions, rewards, next_states, dones)
        model.train_step(states, actions, rewards, next_states, dones)
        # After 2 steps (== target_update_interval), target must match online
        for w_online, w_target in zip(
            model.network.get_weights(), model.target_network.get_weights()
        ):
            np.testing.assert_allclose(w_online, w_target, rtol=1e-5, atol=1e-5)

    def test_epsilon_decay_is_deterministic_with_seed(self) -> None:
        model = AgentLearningModel(
            state_size=4, action_size=2, epsilon=1.0, epsilon_decay=0.9, epsilon_min=0.01
        )
        model.decay_epsilon()
        assert model.epsilon == pytest.approx(0.9)

    def test_epsilon_never_drops_below_min(self) -> None:
        model = AgentLearningModel(
            state_size=4, action_size=2, epsilon=0.015, epsilon_decay=0.5, epsilon_min=0.01
        )
        for _ in range(10):
            model.decay_epsilon()
        assert model.epsilon >= model.epsilon_min

    def test_model_type_dqn_only(self) -> None:
        with pytest.raises(ValueError, match="model_type"):
            AgentLearningModel(state_size=4, action_size=2, model_type="actor_critic")

    def test_invalid_device_raises(self) -> None:
        with pytest.raises(ValueError, match="device"):
            AgentLearningModel(state_size=4, action_size=2, device="tpu")

    def test_train_step_increments_train_steps(self) -> None:
        model = AgentLearningModel(state_size=4, action_size=2, seed=0)
        states = np.random.randn(8, 4).astype(np.float32)
        actions = np.zeros(8, dtype=np.int32)
        rewards = np.zeros(8, dtype=np.float32)
        next_states = np.random.randn(8, 4).astype(np.float32)
        dones = np.zeros(8, dtype=np.float32)
        model.train_step(states, actions, rewards, next_states, dones)
        assert model.train_steps == 1


# ---------------------------------------------------------------------------
# 7. SystemMetrics
# ---------------------------------------------------------------------------


class TestSystemMetrics:
    def test_counter_starts_at_zero(self) -> None:
        m = SystemMetrics()
        assert m.tasks_created.value == 0
        assert m.tasks_submitted.value == 0
        assert m.tasks_completed.value == 0
        assert m.tasks_failed.value == 0
        assert m.tasks_dependency_blocked.value == 0

    def test_counter_increments(self) -> None:
        m = SystemMetrics()
        m.tasks_created.increment()
        m.tasks_created.increment(3)
        assert m.tasks_created.value == 4

    def test_timer_avg_zero_on_no_samples(self) -> None:
        m = SystemMetrics()
        assert m.task_duration.avg == 0.0
        assert m.task_duration.total == 0.0
        assert m.task_duration.count == 0

    def test_timer_records_samples(self) -> None:
        m = SystemMetrics()
        m.task_duration.record(1.0)
        m.task_duration.record(3.0)
        assert m.task_duration.count == 2
        assert m.task_duration.total == pytest.approx(4.0)
        assert m.task_duration.avg == pytest.approx(2.0)

    def test_to_dict_contains_all_keys(self) -> None:
        m = SystemMetrics()
        d = m.to_dict()
        expected_keys = {
            "tasks_created",
            "tasks_submitted",
            "tasks_completed",
            "tasks_failed",
            "tasks_dependency_blocked",
            "task_duration_avg_s",
            "task_duration_total_s",
        }
        assert expected_keys.issubset(d.keys())

    def test_counter_thread_safety(self) -> None:
        from src.agents.super_agentic_agents import _Counter

        counter = _Counter()
        errors: list[Exception] = []

        def inc() -> None:
            for _ in range(1000):
                try:
                    counter.increment()
                except Exception as exc:
                    errors.append(exc)

        threads = [threading.Thread(target=inc) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert counter.value == 10_000


# ---------------------------------------------------------------------------
# 8. _task_from_dict helper
# ---------------------------------------------------------------------------


class TestTaskFromDict:
    def test_roundtrip_via_to_dict(self) -> None:
        task = Task(
            description="roundtrip",
            priority=TaskPriority.HIGH,
            status=TaskStatus.COMPLETED,
            dependencies=["abc"],
            parameters={"x": 1},
            metadata={"m": 2},
        )
        restored = _task_from_dict(task.to_dict())
        assert restored.id == task.id
        assert restored.description == task.description
        assert restored.priority == TaskPriority.HIGH
        assert restored.status == TaskStatus.COMPLETED
        assert restored.dependencies == ["abc"]
        assert restored.parameters == {"x": 1}
        assert restored.metadata == {"m": 2}

    def test_unknown_priority_defaults_to_normal(self) -> None:
        data = {"priority": "ULTRA", "status": "pending"}
        task = _task_from_dict(data)
        assert task.priority == TaskPriority.NORMAL

    def test_unknown_status_defaults_to_pending(self) -> None:
        data = {"priority": "NORMAL", "status": "vaporware"}
        task = _task_from_dict(data)
        assert task.status == TaskStatus.PENDING

    def test_missing_fields_get_defaults(self) -> None:
        task = _task_from_dict({})
        assert task.description == ""
        assert task.dependencies == []
        assert task.parameters == {}

    def test_completed_at_parsed(self) -> None:
        from datetime import datetime

        now = datetime.now()
        data = {"completed_at": now.isoformat()}
        task = _task_from_dict(data)
        assert task.completed_at is not None


# ---------------------------------------------------------------------------
# 9. AgentFactory
# ---------------------------------------------------------------------------


class TestAgentFactory:
    def test_create_executor(self) -> None:
        agent = AgentFactory.create_agent("executor", "E1")
        assert isinstance(agent, ExecutorAgent)

    def test_create_analyzer(self) -> None:
        agent = AgentFactory.create_agent("analyzer", "A1")
        assert isinstance(agent, AnalyzerAgent)

    def test_create_learner(self) -> None:
        agent = AgentFactory.create_agent("learner", "L1")
        assert isinstance(agent, LearnerAgent)

    def test_create_unknown_returns_none(self) -> None:
        agent = AgentFactory.create_agent("unknown_type", "X")
        assert agent is None

    def test_create_team(self) -> None:
        system = AgentFactory.create_team({"executor": 2, "analyzer": 1})
        # orchestrator + 2 executors + 1 analyzer
        assert len(system.agents) == 4


# ---------------------------------------------------------------------------
# 10. BaseAgent – execute_task failure path
# ---------------------------------------------------------------------------


class TestBaseAgentExecuteFailure:
    def test_failed_task_sets_error_and_status(self) -> None:
        agent = _SimpleAgent("fail-agent", fail=True)
        task = Task(description="will-fail")
        agent.assign_task(task)

        with pytest.raises(RuntimeError, match="deliberate"):
            agent.execute_task(task, completed_task_ids=None)

        assert task.status == TaskStatus.FAILED
        assert "deliberate failure" in task.error

    def test_failed_task_sets_agent_error_status(self) -> None:
        agent = _SimpleAgent("err-status", fail=True)
        task = Task(description="err")
        agent.assign_task(task)

        with pytest.raises(RuntimeError):
            agent.execute_task(task, completed_task_ids=None)

        assert agent.status == AgentStatus.ERROR

    def test_successful_task_agent_returns_to_idle(self) -> None:
        agent = _SimpleAgent("idle-check")
        task = Task(description="ok")
        agent.assign_task(task)
        agent.execute_task(task, completed_task_ids=None)
        assert agent.status == AgentStatus.IDLE

    def test_task_removed_from_active_after_execution(self) -> None:
        agent = _SimpleAgent("cleanup")
        task = Task(description="cleanup")
        agent.assign_task(task)
        assert task.id in agent.active_tasks
        agent.execute_task(task, completed_task_ids=None)
        assert task.id not in agent.active_tasks

    def test_performance_metrics_updated_on_success(self) -> None:
        agent = _SimpleAgent("metrics-ok")
        task = Task(description="ok")
        agent.assign_task(task)
        agent.execute_task(task, completed_task_ids=None)
        assert agent.performance_metrics["tasks_completed"] == 1
        assert agent.performance_metrics["tasks_failed"] == 0
        assert agent.performance_metrics["success_rate"] == pytest.approx(1.0)

    def test_performance_metrics_updated_on_failure(self) -> None:
        agent = _SimpleAgent("metrics-fail", fail=True)
        task = Task(description="fail")
        agent.assign_task(task)

        with pytest.raises(RuntimeError):
            agent.execute_task(task, completed_task_ids=None)

        assert agent.performance_metrics["tasks_completed"] == 0
        assert agent.performance_metrics["tasks_failed"] == 1
        assert agent.performance_metrics["success_rate"] == pytest.approx(0.0)
