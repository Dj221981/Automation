"""Extended unit tests for the orchestration Pipeline."""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from src.framework import orchestration as orchestration_module
from src.framework.orchestration import Pipeline, PipelineStep


@pytest.fixture
def sample_context() -> dict[str, Any]:
    """Provide a baseline context dictionary for pipeline tests."""
    return {"value": 1, "items": [1, 2, 3]}


@pytest.fixture
def noop_step() -> PipelineStep:
    """Provide a reusable step that returns the current context unchanged."""
    return PipelineStep("noop", lambda ctx: ctx)


def test_pipeline_basic_execution_transforms_context(sample_context: dict[str, Any]) -> None:
    """A single pipeline step should transform the context and succeed."""
    pipeline = Pipeline("basic")
    pipeline.add_step(PipelineStep("double", lambda ctx: {**ctx, "value": ctx["value"] * 2}))

    result = pipeline.run(sample_context)

    assert result.success is True
    assert result.final_context == {"value": 2, "items": [1, 2, 3]}
    assert [step.step_name for step in result.step_results] == ["double"]


def test_pipeline_multi_step_execution_flows_context() -> None:
    """Multiple steps should receive the updated context from prior steps."""
    pipeline = Pipeline("multi-step")
    pipeline.add_step(PipelineStep("start", lambda ctx: {**ctx, "count": 2}))
    pipeline.add_step(PipelineStep("multiply", lambda ctx: {**ctx, "count": ctx["count"] * 3}))
    pipeline.add_step(PipelineStep("finalize", lambda ctx: {**ctx, "done": ctx["count"] == 6}))

    result = pipeline.run({})

    assert result.success is True
    assert result.final_context == {"count": 6, "done": True}


def test_pipeline_context_mutation_is_preserved(sample_context: dict[str, Any]) -> None:
    """In-place mutations should remain visible when a step returns the same context object."""
    pipeline = Pipeline("mutation")

    def mutate(ctx: dict[str, Any]) -> dict[str, Any]:
        ctx["items"].append(4)
        ctx["mutated"] = True
        return ctx

    pipeline.add_step(PipelineStep("mutate", mutate))

    result = pipeline.run(sample_context)

    assert result.success is True
    assert result.final_context["items"] == [1, 2, 3, 4]
    assert result.final_context["mutated"] is True


def test_pipeline_handles_large_context() -> None:
    """Pipelines should accept large nested context structures without truncation."""
    large_context = {
        "numbers": list(range(1000)),
        "nested": {f"key_{index}": {"value": index} for index in range(200)},
    }
    pipeline = Pipeline("large-context")
    pipeline.add_step(
        PipelineStep(
            "summarize",
            lambda ctx: {**ctx, "total": sum(ctx["numbers"]), "nested_count": len(ctx["nested"])},
        )
    )

    result = pipeline.run(large_context)

    assert result.success is True
    assert result.final_context["total"] == sum(range(1000))
    assert result.final_context["nested_count"] == 200


def test_pipeline_scales_to_many_steps() -> None:
    """A pipeline with many steps should execute each step in insertion order."""
    pipeline = Pipeline("many-steps")
    total_steps = 60

    for index in range(total_steps):
        pipeline.add_step(
            PipelineStep(
                f"step_{index}",
                lambda ctx, idx=index: {**ctx, "count": ctx.get("count", 0) + idx + 1},
            )
        )

    result = pipeline.run({})

    assert result.success is True
    assert len(result.step_results) == total_steps
    assert result.final_context["count"] == sum(range(1, total_steps + 1))


def test_pipeline_stops_on_error_when_configured() -> None:
    """A failing step with stop_on_error=True should abort the remaining steps."""
    pipeline = Pipeline("stop-on-error")
    executed: list[str] = []

    def fail(ctx: dict[str, Any]) -> dict[str, Any]:
        executed.append("fail")
        raise RuntimeError("boom")

    def after(ctx: dict[str, Any]) -> dict[str, Any]:
        executed.append("after")
        return {**ctx, "after": True}

    pipeline.add_step(PipelineStep("fail", fail, stop_on_error=True))
    pipeline.add_step(PipelineStep("after", after))

    result = pipeline.run({})

    assert result.success is False
    assert result.error == "Pipeline aborted at step 'fail': RuntimeError: boom"
    assert executed == ["fail"]
    assert [step.success for step in result.step_results] == [False]


def test_pipeline_continues_on_error_when_allowed() -> None:
    """A failing step with stop_on_error=False should keep running later steps."""
    pipeline = Pipeline("continue-on-error")
    pipeline.add_step(
        PipelineStep(
            "fail", lambda ctx: (_ for _ in ()).throw(ValueError("skip")), stop_on_error=False
        )
    )
    pipeline.add_step(PipelineStep("recover", lambda ctx: {**ctx, "recovered": True}))

    result = pipeline.run({"original": 1})

    assert result.success is False
    assert result.error is None
    assert result.final_context == {"original": 1, "recovered": True}
    assert [step.success for step in result.step_results] == [False, True]


def test_pipeline_supports_mixed_error_modes() -> None:
    """Mixed stop_on_error modes should continue through soft failures and stop on hard failures."""
    pipeline = Pipeline("mixed-errors")
    executed: list[str] = []

    def soft_fail(ctx: dict[str, Any]) -> dict[str, Any]:
        executed.append("soft")
        raise ValueError("soft failure")

    def mutate(ctx: dict[str, Any]) -> dict[str, Any]:
        executed.append("mutate")
        return {**ctx, "value": "kept-going"}

    def hard_fail(ctx: dict[str, Any]) -> dict[str, Any]:
        executed.append("hard")
        raise RuntimeError("hard failure")

    def skipped(ctx: dict[str, Any]) -> dict[str, Any]:
        executed.append("skipped")
        return ctx

    pipeline.add_step(PipelineStep("soft_fail", soft_fail, stop_on_error=False))
    pipeline.add_step(PipelineStep("mutate", mutate))
    pipeline.add_step(PipelineStep("hard_fail", hard_fail, stop_on_error=True))
    pipeline.add_step(PipelineStep("skipped", skipped))

    result = pipeline.run({})

    assert executed == ["soft", "mutate", "hard"]
    assert result.final_context == {"value": "kept-going"}
    assert [step.step_name for step in result.step_results] == ["soft_fail", "mutate", "hard_fail"]
    assert result.error == "Pipeline aborted at step 'hard_fail': RuntimeError: hard failure"


@pytest.mark.parametrize(
    ("failing_index", "expected_names"),
    [
        (0, ["step_0"]),
        (1, ["step_0", "step_1"]),
        (2, ["step_0", "step_1", "step_2"]),
    ],
)
def test_pipeline_records_errors_at_varied_positions(
    failing_index: int,
    expected_names: list[str],
) -> None:
    """Errors should be reported consistently whether they occur early or late in the pipeline."""
    pipeline = Pipeline(f"error-position-{failing_index}")

    for index in range(3):
        if index == failing_index:
            pipeline.add_step(
                PipelineStep(
                    f"step_{index}",
                    lambda ctx, idx=index: (_ for _ in ()).throw(RuntimeError(f"failure-{idx}")),
                )
            )
        else:
            pipeline.add_step(
                PipelineStep(f"step_{index}", lambda ctx, idx=index: {**ctx, f"s{idx}": True})
            )

    result = pipeline.run({})

    assert result.success is False
    assert [step.step_name for step in result.step_results] == expected_names
    assert result.step_results[-1].error == f"RuntimeError: failure-{failing_index}"


@pytest.mark.parametrize("return_value", [[], "text", 99])
def test_pipeline_keeps_context_when_handler_returns_non_dict(return_value: Any) -> None:
    """Non-dict handler outputs should be recorded without replacing the current context."""
    pipeline = Pipeline("non-dict-output")
    pipeline.add_step(PipelineStep("emit", lambda ctx, value=return_value: value))
    pipeline.add_step(PipelineStep("confirm", lambda ctx: {**ctx, "still_here": ctx["value"]}))

    result = pipeline.run({"value": 5})

    assert result.success is True
    assert result.step_results[0].output == return_value
    assert result.final_context == {"value": 5, "still_here": 5}


def test_pipeline_keeps_context_when_handler_returns_none() -> None:
    """A None return value should leave the existing context unchanged for later steps."""
    pipeline = Pipeline("none-output")
    pipeline.add_step(PipelineStep("none", lambda ctx: None))
    pipeline.add_step(PipelineStep("next", lambda ctx: {**ctx, "after_none": True}))

    result = pipeline.run({"value": 2})

    assert result.success is True
    assert result.step_results[0].output is None
    assert result.final_context == {"value": 2, "after_none": True}


@pytest.mark.parametrize("name", ["", "   "])
def test_pipeline_rejects_blank_names(name: str) -> None:
    """Pipeline names must contain non-whitespace characters."""
    with pytest.raises(ValueError, match="Pipeline name cannot be empty"):
        Pipeline(name)


@pytest.mark.parametrize("invalid_step", ["bad-step", object()])
def test_pipeline_rejects_invalid_step_types(invalid_step: Any) -> None:
    """Only PipelineStep instances should be accepted by add_step."""
    pipeline = Pipeline("invalid-step")

    with pytest.raises(TypeError, match="PipelineStep instance"):
        pipeline.add_step(invalid_step)  # type: ignore[arg-type]


def test_pipeline_allows_duplicate_step_names() -> None:
    """Pipelines should preserve duplicate names and execute both registered steps."""
    pipeline = Pipeline("duplicates")
    pipeline.add_step(PipelineStep("duplicate", lambda ctx: {**ctx, "count": 1}))
    pipeline.add_step(PipelineStep("duplicate", lambda ctx: {**ctx, "count": ctx["count"] + 1}))

    result = pipeline.run({})

    assert result.success is True
    assert [step.step_name for step in result.step_results] == ["duplicate", "duplicate"]
    assert result.final_context["count"] == 2


@pytest.mark.parametrize("name", ["", "   "])
def test_pipeline_step_rejects_blank_names(name: str) -> None:
    """Pipeline steps must have non-empty names after trimming whitespace."""
    with pytest.raises(ValueError, match="PipelineStep name cannot be empty"):
        PipelineStep(name, lambda ctx: ctx)


@pytest.mark.parametrize("handler", [None, 42, "not callable"])
def test_pipeline_step_rejects_non_callable_handlers(handler: Any) -> None:
    """Pipeline steps must be created with callable handlers."""
    with pytest.raises(TypeError, match="PipelineStep handler must be callable"):
        PipelineStep("bad", handler)  # type: ignore[arg-type]


def test_pipeline_can_add_then_remove_step(noop_step: PipelineStep) -> None:
    """Removing an existing step should report success and update the pipeline."""
    pipeline = Pipeline("add-remove")
    pipeline.add_step(noop_step)

    removed = pipeline.remove_step("noop")

    assert removed is True
    assert pipeline.steps == []


def test_pipeline_remove_nonexistent_step_returns_false() -> None:
    """Removing a step that was never added should return False."""
    pipeline = Pipeline("missing-step")

    assert pipeline.remove_step("unknown") is False


def test_pipeline_steps_property_returns_independent_list(noop_step: PipelineStep) -> None:
    """The steps property should return a copy that callers can mutate safely."""
    pipeline = Pipeline("steps-copy")
    pipeline.add_step(noop_step)

    exposed_steps = pipeline.steps
    exposed_steps.append(PipelineStep("extra", lambda ctx: ctx))

    assert [step.name for step in pipeline.steps] == ["noop"]
    assert [step.name for step in exposed_steps] == ["noop", "extra"]


def test_pipeline_add_step_supports_chaining() -> None:
    """add_step should return the pipeline instance to support fluent chaining."""
    pipeline = Pipeline("chain")

    chained = pipeline.add_step(PipelineStep("one", lambda ctx: ctx)).add_step(
        PipelineStep("two", lambda ctx: ctx)
    )

    assert chained is pipeline
    assert [step.name for step in pipeline.steps] == ["one", "two"]


def test_pipeline_can_remove_middle_step() -> None:
    """Removing the middle step should preserve the surrounding step order."""
    pipeline = Pipeline("remove-middle")
    pipeline.add_step(PipelineStep("first", lambda ctx: {**ctx, "first": True}))
    pipeline.add_step(PipelineStep("middle", lambda ctx: {**ctx, "middle": True}))
    pipeline.add_step(PipelineStep("last", lambda ctx: {**ctx, "last": True}))

    removed = pipeline.remove_step("middle")
    result = pipeline.run({})

    assert removed is True
    assert [step.step_name for step in result.step_results] == ["first", "last"]
    assert "middle" not in result.final_context


def test_pipeline_run_treats_none_initial_context_as_empty() -> None:
    """Running with None should behave the same as running with an empty context dict."""
    pipeline = Pipeline("none-context")
    pipeline.add_step(PipelineStep("seed", lambda ctx: {**ctx, "seeded": True}))

    result = pipeline.run(None)

    assert result.success is True
    assert result.final_context == {"seeded": True}


def test_pipeline_run_with_no_steps_returns_success() -> None:
    """An empty pipeline should succeed and return the original context."""
    pipeline = Pipeline("empty-run")

    result = pipeline.run({"original": True})

    assert result.success is True
    assert result.step_results == []
    assert result.final_context == {"original": True}


def test_pipeline_supports_very_long_step_names() -> None:
    """Very long step names should be stored and reported without truncation."""
    long_name = "step-" + ("x" * 500)
    pipeline = Pipeline("long-step-name")
    pipeline.add_step(PipelineStep(long_name, lambda ctx: {**ctx, "ok": True}))

    result = pipeline.run({})

    assert result.step_results[0].step_name == long_name
    assert result.final_context["ok"] is True


@pytest.mark.parametrize("name", ["step_1", "step-2", "step99"])
def test_pipeline_supports_special_characters_in_step_names(name: str) -> None:
    """Underscores, hyphens, and numbers should be valid step names."""
    pipeline = Pipeline("special-names")
    pipeline.add_step(PipelineStep(name, lambda ctx: {**ctx, "name": name}))

    result = pipeline.run({})

    assert result.success is True
    assert result.step_results[0].step_name == name
    assert result.final_context["name"] == name


def test_pipeline_preserves_unicode_context_values() -> None:
    """Unicode content should pass through pipeline transformations unchanged."""
    pipeline = Pipeline("unicode")
    pipeline.add_step(
        PipelineStep(
            "greet",
            lambda ctx: {**ctx, "message": f"{ctx['greeting']} 🌍 — {ctx['language']}"},
        )
    )

    result = pipeline.run({"greeting": "こんにちは", "language": "Español"})

    assert result.success is True
    assert result.final_context["message"] == "こんにちは 🌍 — Español"


def test_pipeline_handles_circular_references_in_nested_context() -> None:
    """Circular nested objects should be passed through without causing recursion failures."""
    child: dict[str, Any] = {"name": "child"}
    child["self"] = child
    pipeline = Pipeline("circular")
    pipeline.add_step(PipelineStep("noop", lambda ctx: ctx))

    result = pipeline.run({"child": child})

    assert result.success is True
    assert result.final_context["child"]["self"] is child


def test_pipeline_add_step_is_thread_safe() -> None:
    """Concurrent add_step calls should retain every inserted step exactly once."""
    pipeline = Pipeline("thread-add")
    barrier = threading.Barrier(9)
    total_threads = 8
    steps_per_thread = 10
    failures: list[BaseException] = []

    def worker(worker_index: int) -> None:
        try:
            barrier.wait()
            for step_index in range(steps_per_thread):
                pipeline.add_step(
                    PipelineStep(
                        f"worker_{worker_index}_step_{step_index}",
                        lambda ctx, w=worker_index, s=step_index: {**ctx, f"{w}-{s}": True},
                    )
                )
        except BaseException as exc:  # pragma: no cover - propagated below
            failures.append(exc)
            raise

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(total_threads)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert failures == []
    assert len(pipeline.steps) == total_threads * steps_per_thread


def test_pipeline_can_run_while_steps_are_modified() -> None:
    """A running pipeline should operate on a stable step snapshot even if modified concurrently."""
    pipeline = Pipeline("run-and-modify")
    started = threading.Event()
    release = threading.Event()
    result_holder: dict[str, Any] = {}

    def blocking(ctx: dict[str, Any]) -> dict[str, Any]:
        started.set()
        release.wait(timeout=2)
        return {**ctx, "blocking": True}

    pipeline.add_step(PipelineStep("blocking", blocking))
    pipeline.add_step(PipelineStep("after", lambda ctx: {**ctx, "after": True}))

    def runner() -> None:
        result_holder["result"] = pipeline.run({})

    thread = threading.Thread(target=runner)
    thread.start()
    assert started.wait(timeout=2)

    pipeline.add_step(PipelineStep("added_late", lambda ctx: {**ctx, "late": True}))
    removed = pipeline.remove_step("after")
    release.set()
    thread.join(timeout=2)

    result = result_holder["result"]
    assert removed is True
    assert [step.step_name for step in result.step_results] == ["blocking", "after"]
    assert "late" not in result.final_context
    assert [step.name for step in pipeline.steps] == ["blocking", "added_late"]


def test_pipeline_steps_property_is_safe_under_concurrent_access() -> None:
    """Concurrent reads of steps should always return valid list snapshots."""
    pipeline = Pipeline("steps-threadsafe")
    pipeline.add_step(PipelineStep("initial", lambda ctx: ctx))
    stop = threading.Event()
    snapshots: list[list[str]] = []

    def reader() -> None:
        while not stop.is_set():
            snapshots.append([step.name for step in pipeline.steps])

    readers = [threading.Thread(target=reader) for _ in range(3)]
    for thread in readers:
        thread.start()

    for index in range(5):
        pipeline.add_step(PipelineStep(f"step_{index}", lambda ctx: ctx))

    stop.set()
    for thread in readers:
        thread.join(timeout=2)

    assert snapshots
    assert all(snapshot[0] == "initial" for snapshot in snapshots if snapshot)
    assert all(isinstance(snapshot, list) for snapshot in snapshots)


def test_pipeline_records_accurate_step_timing_with_monotonic_mock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Step and pipeline durations should be derived from monotonic time measurements."""
    clock = iter([1.0, 1.1, 1.35, 1.5])
    monkeypatch.setattr(orchestration_module.time, "monotonic", lambda: next(clock))
    pipeline = Pipeline("timed")
    pipeline.add_step(PipelineStep("step", lambda ctx: {**ctx, "done": True}))

    result = pipeline.run({})

    assert result.step_results[0].duration_seconds == pytest.approx(0.25)
    assert result.duration_seconds == pytest.approx(0.5)


def test_pipeline_records_non_negative_duration_for_fast_steps() -> None:
    """Very fast handlers should still record a valid non-negative duration."""
    pipeline = Pipeline("fast")
    pipeline.add_step(PipelineStep("fast-step", lambda ctx: ctx))

    result = pipeline.run({})

    assert result.success is True
    assert result.step_results[0].duration_seconds >= 0.0
    assert result.duration_seconds >= result.step_results[0].duration_seconds


def test_pipeline_records_measurable_duration_for_slow_steps() -> None:
    """Slow handlers should contribute a measurable execution duration."""
    pipeline = Pipeline("slow")

    def slow(ctx: dict[str, Any]) -> dict[str, Any]:
        time.sleep(0.02)
        return {**ctx, "slow": True}

    pipeline.add_step(PipelineStep("slow-step", slow))

    result = pipeline.run({})

    assert result.success is True
    assert result.step_results[0].duration_seconds >= 0.015
    assert result.duration_seconds >= result.step_results[0].duration_seconds


def test_pipeline_total_duration_tracks_sum_of_step_times(monkeypatch: pytest.MonkeyPatch) -> None:
    """Total duration should reflect the overall pipeline window around all executed steps."""
    clock = iter([10.0, 10.1, 10.4, 10.5, 10.9, 11.0])
    monkeypatch.setattr(orchestration_module.time, "monotonic", lambda: next(clock))
    pipeline = Pipeline("timing-summary")
    pipeline.add_step(PipelineStep("first", lambda ctx: {**ctx, "first": True}))
    pipeline.add_step(PipelineStep("second", lambda ctx: {**ctx, "second": True}))

    result = pipeline.run({})

    step_total = sum(step.duration_seconds for step in result.step_results)
    assert step_total == pytest.approx(0.7)
    assert result.duration_seconds == pytest.approx(1.0)
    assert result.duration_seconds >= step_total


def test_pipeline_result_to_dict_serializes_expected_fields() -> None:
    """PipelineResult.to_dict should expose the pipeline summary in plain Python types."""
    pipeline = Pipeline("serialize")
    pipeline.add_step(PipelineStep("step", lambda ctx: {**ctx, "value": 7}))

    payload = pipeline.run({}).to_dict()

    assert set(payload) == {
        "pipeline_name",
        "success",
        "final_context",
        "step_results",
        "error",
        "duration_seconds",
    }
    assert isinstance(payload["pipeline_name"], str)
    assert isinstance(payload["success"], bool)
    assert isinstance(payload["final_context"], dict)
    assert isinstance(payload["step_results"], list)
    assert isinstance(payload["duration_seconds"], float)


@pytest.mark.parametrize(
    ("steps", "expected_success", "expected_error"),
    [
        ([PipelineStep("ok", lambda ctx: ctx)], True, None),
        (
            [
                PipelineStep(
                    "soft",
                    lambda ctx: (_ for _ in ()).throw(ValueError("soft")),
                    stop_on_error=False,
                )
            ],
            False,
            None,
        ),
        (
            [
                PipelineStep(
                    "hard",
                    lambda ctx: (_ for _ in ()).throw(RuntimeError("hard")),
                    stop_on_error=True,
                )
            ],
            False,
            "Pipeline aborted at step 'hard': RuntimeError: hard",
        ),
    ],
)
def test_pipeline_result_success_flag_reflects_step_outcomes(
    steps: list[PipelineStep],
    expected_success: bool,
    expected_error: str | None,
) -> None:
    """Pipeline success should mirror whether every step completed and whether execution aborted."""
    pipeline = Pipeline("success-flag")
    for step in steps:
        pipeline.add_step(step)

    result = pipeline.run({})

    assert result.success is expected_success
    assert result.error == expected_error


def test_pipeline_step_results_preserve_execution_order() -> None:
    """Step results should be returned in the same order as the executed steps."""
    pipeline = Pipeline("ordering")
    for name in ["first", "second", "third"]:
        pipeline.add_step(PipelineStep(name, lambda ctx, step_name=name: {**ctx, step_name: True}))

    result = pipeline.run({})

    assert [step.step_name for step in result.step_results] == ["first", "second", "third"]


def test_pipeline_preserves_error_messages_from_exceptions() -> None:
    """Captured step errors should retain the exception type and message text."""
    pipeline = Pipeline("error-message")
    pipeline.add_step(
        PipelineStep(
            "fail", lambda ctx: (_ for _ in ()).throw(RuntimeError("failure: ünicode details"))
        )
    )

    result = pipeline.run({})

    assert result.step_results[0].error == "RuntimeError: failure: ünicode details"
    assert result.error == "Pipeline aborted at step 'fail': RuntimeError: failure: ünicode details"


def test_pipeline_supports_complex_workflow_transformations() -> None:
    """A multi-stage workflow should validate, enrich, and summarize shared context."""
    pipeline = Pipeline("complex-workflow")
    pipeline.add_step(
        PipelineStep(
            "normalize",
            lambda ctx: {
                **ctx,
                "email": ctx["email"].strip().lower(),
                "roles": sorted(ctx["roles"]),
            },
        )
    )
    pipeline.add_step(
        PipelineStep(
            "validate",
            lambda ctx: {**ctx, "valid": "@" in ctx["email"] and bool(ctx["roles"])},
        )
    )
    pipeline.add_step(
        PipelineStep(
            "enrich",
            lambda ctx: {
                **ctx,
                "profile": {"email": ctx["email"], "primary_role": ctx["roles"][0]},
            },
        )
    )

    result = pipeline.run({"email": " User@Example.COM ", "roles": ["editor", "admin"]})

    assert result.success is True
    assert result.final_context["email"] == "user@example.com"
    assert result.final_context["valid"] is True
    assert result.final_context["profile"] == {"email": "user@example.com", "primary_role": "admin"}


def test_pipeline_steps_can_depend_on_previous_output() -> None:
    """A downstream step should be able to rely on values produced upstream."""
    pipeline = Pipeline("dependency-chain")
    pipeline.add_step(PipelineStep("base", lambda ctx: {**ctx, "base": 4}))
    pipeline.add_step(PipelineStep("derived", lambda ctx: {**ctx, "derived": ctx["base"] ** 2}))

    result = pipeline.run({})

    assert result.success is True
    assert result.final_context["derived"] == 16


def test_pipeline_can_simulate_external_service_calls() -> None:
    """Handlers that simulate I/O should still integrate cleanly into the pipeline."""
    pipeline = Pipeline("external-service")
    calls: list[str] = []

    def fetch(ctx: dict[str, Any]) -> dict[str, Any]:
        calls.append(ctx["resource"])
        time.sleep(0.01)
        return {**ctx, "response": {"resource": ctx["resource"], "status": 200}}

    pipeline.add_step(PipelineStep("fetch", fetch))

    result = pipeline.run({"resource": "health"})

    assert result.success is True
    assert calls == ["health"]
    assert result.final_context["response"] == {"resource": "health", "status": 200}


@pytest.mark.parametrize(
    ("approved", "expected_status"),
    [(True, "ready"), (False, "pending-review")],
)
def test_pipeline_supports_conditional_logic(approved: bool, expected_status: str) -> None:
    """Step handlers should be able to branch on the current context state."""
    pipeline = Pipeline("conditional")
    pipeline.add_step(
        PipelineStep(
            "status",
            lambda ctx: {**ctx, "status": "ready" if ctx["approved"] else "pending-review"},
        )
    )

    result = pipeline.run({"approved": approved})

    assert result.success is True
    assert result.final_context["status"] == expected_status
