"""
Tests for the framework layer: plugin, engine, and orchestration.
"""

import pytest

# ---------------------------------------------------------------------------
# Plugin tests
# ---------------------------------------------------------------------------

from src.framework.plugin import (
    FunctionPlugin,
    Plugin,
    PluginRegistry,
    PluginResult,
    plugin,
)


class EchoPlugin(Plugin):
    name = "echo"
    version = "1.0.0"
    description = "Returns its kwargs"

    def run(self, **kwargs):
        return kwargs


class FailingPlugin(Plugin):
    name = "failing"
    version = "1.0.0"
    description = "Always raises"

    def run(self, **kwargs):
        raise ValueError("intentional failure")


class SetupTracker(Plugin):
    name = "tracker"
    version = "1.0.0"
    setup_called = False
    teardown_called = False

    def setup(self):
        SetupTracker.setup_called = True

    def teardown(self):
        SetupTracker.teardown_called = True

    def run(self, **kwargs):
        return "ok"


# --- Plugin registration ---


def test_register_plugin_calls_setup():
    SetupTracker.setup_called = False
    SetupTracker.teardown_called = False
    registry = PluginRegistry()
    registry.register(SetupTracker())
    assert SetupTracker.setup_called is True


def test_unregister_plugin_calls_teardown():
    SetupTracker.setup_called = False
    SetupTracker.teardown_called = False
    registry = PluginRegistry()
    registry.register(SetupTracker())
    result = registry.unregister("tracker")
    assert result is True
    assert SetupTracker.teardown_called is True


def test_register_unknown_type_raises():
    registry = PluginRegistry()
    with pytest.raises(TypeError):
        registry.register("not-a-plugin")  # type: ignore[arg-type]


def test_plugin_name_required():
    """A plugin with an empty name must be rejected."""
    class NoName(Plugin):
        name = ""

        def run(self, **kwargs):
            return {}

    registry = PluginRegistry()
    with pytest.raises(ValueError):
        registry.register(NoName())


# --- Plugin execution ---


def test_run_echo_plugin():
    registry = PluginRegistry()
    registry.register(EchoPlugin())
    result = registry.run("echo", x=1, y=2)
    assert isinstance(result, PluginResult)
    assert result.success is True
    assert result.output == {"x": 1, "y": 2}
    assert result.plugin_name == "echo"


def test_run_missing_plugin_returns_failure():
    registry = PluginRegistry()
    result = registry.run("nonexistent")
    assert result.success is False
    assert "not found" in result.error


def test_run_failing_plugin_captures_error():
    registry = PluginRegistry()
    registry.register(FailingPlugin())
    result = registry.run("failing")
    assert result.success is False
    assert "intentional failure" in result.error


def test_run_all_collects_results():
    registry = PluginRegistry()
    registry.register(EchoPlugin())
    registry.register(FailingPlugin())
    results = registry.run_all(x=42)
    assert len(results) == 2
    names = {r.plugin_name for r in results}
    assert names == {"echo", "failing"}


# --- Enable / disable ---


def test_disable_prevents_run():
    registry = PluginRegistry()
    registry.register(EchoPlugin())
    registry.disable("echo")
    result = registry.run("echo")
    assert result.success is False
    assert "disabled" in result.error


def test_enable_after_disable_allows_run():
    registry = PluginRegistry()
    registry.register(EchoPlugin())
    registry.disable("echo")
    registry.enable("echo")
    result = registry.run("echo", a=1)
    assert result.success is True


def test_run_all_skips_disabled():
    registry = PluginRegistry()
    registry.register(EchoPlugin())
    registry.register(FailingPlugin())
    registry.disable("failing")
    results = registry.run_all()
    assert len(results) == 1
    assert results[0].plugin_name == "echo"


# --- FunctionPlugin / decorator ---


def test_function_plugin_wraps_callable():
    fp = FunctionPlugin(lambda x=0: x * 2, name="double")
    assert fp.run(x=5) == 10


def test_plugin_decorator():
    @plugin(name="greet", description="say hello")
    def greet(name: str = "world") -> dict:
        return {"greeting": f"Hello, {name}!"}

    assert isinstance(greet, FunctionPlugin)
    registry = PluginRegistry()
    registry.register(greet)
    result = registry.run("greet", name="Alice")
    assert result.success is True
    assert result.output == {"greeting": "Hello, Alice!"}


def test_function_plugin_requires_callable():
    with pytest.raises(TypeError):
        FunctionPlugin("not-callable", name="bad")  # type: ignore[arg-type]


def test_function_plugin_requires_nonempty_name():
    with pytest.raises(ValueError):
        FunctionPlugin(lambda: None, name="")


# --- Stats ---


def test_registry_stats():
    registry = PluginRegistry()
    registry.register(EchoPlugin())
    registry.register(FailingPlugin())
    registry.run("echo")
    registry.run("echo")
    registry.run("failing")
    stats = registry.stats()
    assert stats["registered"] == 2
    assert stats["total_runs"] == 3
    assert stats["total_errors"] == 1


def test_list_plugins():
    registry = PluginRegistry()
    registry.register(EchoPlugin())
    registry.disable("echo")
    listing = registry.list_plugins()
    assert len(listing) == 1
    assert listing[0]["name"] == "echo"
    assert listing[0]["enabled"] is False


# ---------------------------------------------------------------------------
# Engine tests
# ---------------------------------------------------------------------------

from src.framework.engine import (
    EngineConfig,
    EngineHook,
    EngineState,
    ExecutionEngine,
    TaskRecord,
)
from src.agents.super_agentic_agents import AgentSystem, ExecutorAgent, TaskStatus


def _make_system_with_executor():
    system = AgentSystem("test-engine-system")
    agent = ExecutorAgent("worker")
    system.add_agent(agent)
    return system, agent


def test_engine_initial_state_is_idle():
    system, _ = _make_system_with_executor()
    engine = ExecutionEngine(system)
    assert engine.state == EngineState.IDLE


def test_engine_start_changes_state():
    system, _ = _make_system_with_executor()
    engine = ExecutionEngine(system)
    engine.start()
    assert engine.state == EngineState.RUNNING


def test_engine_pause_and_resume():
    system, _ = _make_system_with_executor()
    engine = ExecutionEngine(system)
    engine.start()
    engine.pause()
    assert engine.state == EngineState.PAUSED
    engine.resume()
    assert engine.state == EngineState.RUNNING


def test_engine_stop():
    system, _ = _make_system_with_executor()
    engine = ExecutionEngine(system)
    engine.start()
    engine.stop()
    assert engine.state == EngineState.STOPPED


def test_stopped_engine_cannot_restart():
    system, _ = _make_system_with_executor()
    engine = ExecutionEngine(system)
    engine.start()
    engine.stop()
    with pytest.raises(RuntimeError, match="stopped"):
        engine.start()


def test_submit_requires_running_engine():
    system, agent = _make_system_with_executor()
    engine = ExecutionEngine(system, config=EngineConfig(default_agent_id=agent.id))
    with pytest.raises(RuntimeError, match="start"):
        engine.submit("task without starting engine")


def test_submit_returns_task_id():
    system, agent = _make_system_with_executor()
    engine = ExecutionEngine(system, config=EngineConfig(default_agent_id=agent.id))
    engine.start()
    task_id = engine.submit("do something")
    assert isinstance(task_id, str) and len(task_id) > 0


def test_submit_and_run_task():
    system, agent = _make_system_with_executor()
    engine = ExecutionEngine(system, config=EngineConfig(default_agent_id=agent.id))
    engine.start()
    task_id = engine.submit("test task", parameters={"x": 42})
    record = engine.run_task(task_id)
    assert isinstance(record, TaskRecord)
    assert record.task_id == task_id
    assert record.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)


def test_engine_hooks_called():
    submitted_ids = []
    completed_ids = []

    hook = EngineHook(
        on_submitted=lambda r: submitted_ids.append(r.task_id),
        on_completed=lambda r: completed_ids.append(r.task_id),
    )

    system, agent = _make_system_with_executor()
    engine = ExecutionEngine(
        system,
        config=EngineConfig(default_agent_id=agent.id),
        hooks=[hook],
    )
    engine.start()
    task_id = engine.submit("hook test task")
    engine.run_task(task_id)

    assert task_id in submitted_ids
    assert task_id in completed_ids


def test_engine_metrics_update():
    system, agent = _make_system_with_executor()
    engine = ExecutionEngine(system, config=EngineConfig(default_agent_id=agent.id))
    engine.start()
    task_id = engine.submit("metrics task")
    engine.run_task(task_id)

    m = engine.metrics()
    assert m["submitted"] == 1
    assert m["completed"] + m["failed"] == 1


def test_get_record_returns_none_for_unknown():
    system, _ = _make_system_with_executor()
    engine = ExecutionEngine(system)
    assert engine.get_record("unknown-id") is None


def test_engine_requires_agent_system():
    with pytest.raises(TypeError, match="AgentSystem"):
        ExecutionEngine("not-a-system")  # type: ignore[arg-type]


def test_engine_add_hook_at_runtime():
    called = []
    system, agent = _make_system_with_executor()
    engine = ExecutionEngine(system, config=EngineConfig(default_agent_id=agent.id))
    engine.start()
    engine.add_hook(EngineHook(on_submitted=lambda r: called.append(r.task_id)))
    task_id = engine.submit("runtime hook")
    assert task_id in called


def test_engine_paused_prevents_submit():
    system, agent = _make_system_with_executor()
    engine = ExecutionEngine(system, config=EngineConfig(default_agent_id=agent.id))
    engine.start()
    engine.pause()
    with pytest.raises(RuntimeError):
        engine.submit("should fail")


# ---------------------------------------------------------------------------
# Orchestration tests
# ---------------------------------------------------------------------------

from src.framework.orchestration import (
    Coordinator,
    CoordinatorResult,
    Pipeline,
    PipelineResult,
    PipelineStep,
    Router,
    RouterResult,
)


# --- Pipeline ---


def test_pipeline_runs_steps_in_order():
    order = []
    pipeline = Pipeline("test")

    def step_a(ctx):
        order.append("a")
        return {**ctx, "a": 1}

    def step_b(ctx):
        order.append("b")
        return {**ctx, "b": ctx.get("a", 0) + 1}

    pipeline.add_step(PipelineStep("step_a", step_a))
    pipeline.add_step(PipelineStep("step_b", step_b))

    result = pipeline.run({})
    assert result.success is True
    assert order == ["a", "b"]
    assert result.final_context["a"] == 1
    assert result.final_context["b"] == 2


def test_pipeline_aborts_on_error_by_default():
    ran = []

    def fail(ctx):
        raise RuntimeError("boom")

    def after(ctx):
        ran.append("after")
        return ctx

    pipeline = Pipeline("abort-test")
    pipeline.add_step(PipelineStep("fail_step", fail, stop_on_error=True))
    pipeline.add_step(PipelineStep("after_step", after))

    result = pipeline.run()
    assert result.success is False
    assert "after" not in ran


def test_pipeline_continues_on_non_stop_error():
    pipeline = Pipeline("continue-test")

    def fail(ctx):
        raise ValueError("non-stop error")

    def after(ctx):
        return {**ctx, "after": True}

    pipeline.add_step(PipelineStep("fail", fail, stop_on_error=False))
    pipeline.add_step(PipelineStep("after", after))

    result = pipeline.run()
    # overall success is False because one step failed
    assert result.success is False
    # but the second step still ran
    assert result.final_context.get("after") is True


def test_pipeline_empty_context_default():
    pipeline = Pipeline("empty")
    pipeline.add_step(PipelineStep("noop", lambda ctx: ctx))
    result = pipeline.run()
    assert result.success is True
    assert result.final_context == {}


def test_pipeline_remove_step():
    pipeline = Pipeline("removable")
    pipeline.add_step(PipelineStep("s1", lambda ctx: ctx))
    pipeline.add_step(PipelineStep("s2", lambda ctx: ctx))
    removed = pipeline.remove_step("s1")
    assert removed is True
    assert len(pipeline.steps) == 1
    assert pipeline.steps[0].name == "s2"


def test_pipeline_name_required():
    with pytest.raises(ValueError):
        Pipeline("")


def test_pipeline_step_requires_callable():
    with pytest.raises(TypeError):
        PipelineStep("bad", "not-callable")  # type: ignore[arg-type]


def test_pipeline_step_name_required():
    with pytest.raises(ValueError):
        PipelineStep("", lambda ctx: ctx)


def test_pipeline_result_to_dict():
    pipeline = Pipeline("dict-test")
    pipeline.add_step(PipelineStep("s", lambda ctx: {**ctx, "x": 1}))
    result = pipeline.run({"initial": True})
    d = result.to_dict()
    assert d["pipeline_name"] == "dict-test"
    assert d["success"] is True
    assert "step_results" in d


# --- Router ---


def test_router_dispatch_correct_handler():
    router = Router()
    router.register("double", lambda p: p["v"] * 2)
    result = router.dispatch("double", {"v": 5})
    assert result.success is True
    assert result.output == 10


def test_router_dispatch_unknown_key_returns_failure():
    router = Router()
    result = router.dispatch("nonexistent")
    assert result.success is False
    assert "No handler" in result.error


def test_router_default_handler():
    router = Router(default_handler=lambda p: "default")
    result = router.dispatch("anything")
    assert result.success is True
    assert result.output == "default"


def test_router_handler_exception_captured():
    router = Router()
    router.register("boom", lambda p: (_ for _ in ()).throw(RuntimeError("kaboom")))
    result = router.dispatch("boom", {})
    assert result.success is False
    assert "kaboom" in result.error


def test_router_unregister():
    router = Router()
    router.register("key", lambda p: "value")
    removed = router.unregister("key")
    assert removed is True
    result = router.dispatch("key")
    assert result.success is False


def test_router_list_routes():
    router = Router()
    router.register("a", lambda p: None)
    router.register("b", lambda p: None)
    routes = router.list_routes()
    assert set(routes) == {"a", "b"}


def test_router_stats():
    router = Router()
    router.register("x", lambda p: p)
    router.dispatch("x", 1)
    router.dispatch("x", 2)
    stats = router.stats()
    assert stats.get("x") == 2


def test_router_routing_key_cannot_be_empty():
    router = Router()
    with pytest.raises(ValueError):
        router.register("", lambda p: None)


# --- Coordinator ---


def test_coordinator_run_all():
    coord = Coordinator("test-coord")
    coord.add_component("a", lambda ctx: {"a": 1})
    coord.add_component("b", lambda ctx: {"b": 2})
    result = coord.run_all()
    assert result.success is True
    assert "a" in result.results
    assert "b" in result.results


def test_coordinator_run_all_captures_errors():
    coord = Coordinator("err-coord")
    coord.add_component("good", lambda ctx: "ok")
    coord.add_component("bad", lambda ctx: (_ for _ in ()).throw(ValueError("oops")))
    result = coord.run_all()
    assert result.success is False
    assert "bad" in result.errors
    assert "good" in result.results


def test_coordinator_run_sequential_passes_context():
    coord = Coordinator("seq-coord")
    coord.add_component("step1", lambda ctx: {**ctx, "x": 10})
    coord.add_component("step2", lambda ctx: {**ctx, "y": ctx.get("x", 0) + 5})
    result = coord.run_sequential({"initial": True})
    assert result.success is True
    # step2 should have received x=10 from step1's output
    assert result.results["step2"]["y"] == 15


def test_coordinator_remove_component():
    coord = Coordinator("remove-test")
    coord.add_component("c1", lambda ctx: 1)
    removed = coord.remove_component("c1")
    assert removed is True
    assert "c1" not in coord.list_components()


def test_coordinator_name_required():
    with pytest.raises(ValueError):
        Coordinator("")


def test_coordinator_component_must_be_callable():
    coord = Coordinator("type-check")
    with pytest.raises(TypeError):
        coord.add_component("bad", "not-callable")  # type: ignore[arg-type]


def test_coordinator_result_to_dict():
    coord = Coordinator("dict-coord")
    coord.add_component("c", lambda ctx: "val")
    result = coord.run_all()
    d = result.to_dict()
    assert d["coordinator_name"] == "dict-coord"
    assert d["success"] is True
    assert "c" in d["results"]


# ---------------------------------------------------------------------------
# Integration: plugin registry → coordinator
# ---------------------------------------------------------------------------


def test_plugin_as_coordinator_component():
    """A PluginRegistry can be wired into a Coordinator as a component."""
    registry = PluginRegistry()
    registry.register(EchoPlugin())

    def plugin_component(ctx: dict) -> dict:
        result = registry.run("echo", **ctx)
        return result.output or {}

    coord = Coordinator("integration")
    coord.add_component("echo_plugin", plugin_component)

    out = coord.run_all({"msg": "hello"})
    assert out.success is True
    assert out.results["echo_plugin"] == {"msg": "hello"}
