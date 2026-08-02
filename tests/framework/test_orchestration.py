from concurrent.futures import ThreadPoolExecutor

import pytest

from src.framework.orchestration import Coordinator, Pipeline, PipelineStep, Router


class TestPipeline:
    def test_run_flows_context_through_linear_steps(self):
        pipeline = Pipeline("flow")
        pipeline.add_step(PipelineStep("load", lambda ctx: {**ctx, "count": 1}))
        pipeline.add_step(PipelineStep("increment", lambda ctx: {**ctx, "count": ctx["count"] + 1}))

        result = pipeline.run({"start": True})

        assert result.success is True
        assert result.final_context == {"start": True, "count": 2}
        assert [step.step_name for step in result.step_results] == ["load", "increment"]

    def test_run_supports_in_place_context_updates(self):
        pipeline = Pipeline("mutate")

        def mutate(ctx):
            ctx["value"] = ctx.get("value", 0) + 3
            return ctx

        result = pipeline.add_step(PipelineStep("mutate", mutate)).run({"value": 2})

        assert result.success is True
        assert result.final_context == {"value": 5}
        assert result.step_results[0].output == {"value": 5}

    def test_run_aborts_when_stop_on_error_is_true(self):
        calls = []

        def fail(ctx):
            calls.append("fail")
            raise RuntimeError("boom")

        def never(ctx):
            calls.append("never")
            return ctx

        pipeline = Pipeline("abort")
        pipeline.add_step(PipelineStep("fail", fail, stop_on_error=True))
        pipeline.add_step(PipelineStep("never", never))

        result = pipeline.run({"seed": 1})

        assert result.success is False
        assert result.error == "Pipeline aborted at step 'fail': RuntimeError: boom"
        assert calls == ["fail"]
        assert result.final_context == {"seed": 1}

    def test_run_continues_when_stop_on_error_is_false(self):
        pipeline = Pipeline("continue")
        pipeline.add_step(PipelineStep("fail", lambda ctx: (_ for _ in ()).throw(ValueError("bad")), stop_on_error=False))
        pipeline.add_step(PipelineStep("after", lambda ctx: {**ctx, "after": True}))

        result = pipeline.run({"seed": 1})

        assert result.success is False
        assert result.error is None
        assert result.final_context == {"seed": 1, "after": True}
        assert [step.success for step in result.step_results] == [False, True]

    def test_empty_pipeline_succeeds_and_copies_initial_context(self):
        initial = {"value": 1}

        result = Pipeline("empty").run(initial)

        initial["value"] = 99
        assert result.success is True
        assert result.step_results == []
        assert result.final_context == {"value": 1}

    @pytest.mark.parametrize(
        ("factory", "error_type", "message"),
        [
            (lambda: Pipeline("   "), ValueError, "Pipeline name cannot be empty"),
            (lambda: PipelineStep("", lambda ctx: ctx), ValueError, "PipelineStep name cannot be empty"),
            (lambda: PipelineStep(123, lambda ctx: ctx), TypeError, "PipelineStep name must be a non-empty string"),
        ],
    )
    def test_validation_rejects_invalid_names(self, factory, error_type, message):
        with pytest.raises(error_type, match=message):
            factory()

    def test_add_step_rejects_non_pipeline_step(self):
        with pytest.raises(TypeError, match="PipelineStep"):
            Pipeline("invalid-step").add_step("not-a-step")  # type: ignore[arg-type]

    def test_run_rejects_non_mapping_initial_context(self):
        with pytest.raises(TypeError, match="initial_context must be a mapping or None"):
            Pipeline("bad-context").run(["not", "a", "mapping"])  # type: ignore[arg-type]

    def test_run_records_error_when_step_returns_non_mapping(self):
        pipeline = Pipeline("bad-output")
        pipeline.add_step(PipelineStep("bad", lambda ctx: "nope", stop_on_error=False))  # type: ignore[return-value]
        pipeline.add_step(PipelineStep("after", lambda ctx: {**ctx, "after": True}))

        result = pipeline.run({"seed": 1})

        assert result.success is False
        assert "must return a mapping" in result.step_results[0].error
        assert result.final_context == {"seed": 1, "after": True}

    def test_steps_property_returns_copy(self):
        pipeline = Pipeline("copy")
        pipeline.add_step(PipelineStep("only", lambda ctx: ctx))

        steps = pipeline.steps
        steps.clear()

        assert [step.name for step in pipeline.steps] == ["only"]


class TestRouter:
    def test_dispatches_to_registered_handler(self):
        router = Router().register("double", lambda payload: payload["value"] * 2)

        result = router.dispatch("double", {"value": 4})

        assert result.success is True
        assert result.output == 8
        assert result.routing_key == "double"

    def test_dispatch_uses_default_handler_fallback(self):
        router = Router(default_handler=lambda payload: f"default:{payload}")

        result = router.dispatch("  missing  ", "payload")

        assert result.success is True
        assert result.output == "default:payload"
        assert result.routing_key == "missing"
        assert router.stats()["missing"] == 1

    def test_dispatch_returns_failure_when_no_handler_exists(self):
        result = Router().dispatch("unknown")

        assert result.success is False
        assert result.error == "No handler registered for routing key 'unknown'"

    def test_dispatch_captures_handler_exceptions(self):
        router = Router().register("boom", lambda payload: (_ for _ in ()).throw(RuntimeError("kaboom")))

        result = router.dispatch("boom", {})

        assert result.success is False
        assert result.error == "RuntimeError: kaboom"

    def test_register_list_and_unregister_normalize_route_keys(self):
        router = Router()
        router.register("  alpha  ", lambda payload: payload)
        router.register("beta", lambda payload: payload)

        assert set(router.list_routes()) == {"alpha", "beta"}
        assert router.unregister(" alpha ") is True
        assert router.unregister("alpha") is False
        assert router.list_routes() == ["beta"]

    @pytest.mark.parametrize(
        ("factory", "error_type", "message"),
        [
            (lambda: Router(default_handler="bad"), TypeError, "default_handler must be callable or None"),
            (lambda: Router().register("", lambda payload: payload), ValueError, "routing_key cannot be empty"),
            (lambda: Router().dispatch(None), TypeError, "routing_key must be a non-empty string"),
        ],
    )
    def test_validation_rejects_invalid_router_inputs(self, factory, error_type, message):
        with pytest.raises(error_type, match=message):
            factory()

    def test_dispatch_stats_are_thread_safe(self):
        router = Router().register("work", lambda payload: payload)

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(router.dispatch, "work", i) for i in range(40)]
            results = [future.result() for future in futures]

        assert all(result.success for result in results)
        assert router.stats()["work"] == 40


class TestCoordinator:
    def test_run_all_fans_out_to_all_components(self):
        coordinator = Coordinator("fan-out")
        coordinator.add_component("first", lambda ctx: {"seen": ctx["value"]})
        coordinator.add_component("second", lambda ctx: ctx["value"] * 2)

        result = coordinator.run_all({"value": 3})

        assert result.success is True
        assert result.results == {"first": {"seen": 3}, "second": 6}
        assert result.errors == {}

    def test_run_all_isolates_component_context_mutations(self):
        seen = {}
        coordinator = Coordinator("isolated")

        def mutating_component(ctx):
            ctx["mutated"] = True
            return ctx

        def observer(ctx):
            seen.update(ctx)
            return ctx.get("mutated", False)

        coordinator.add_component("mutator", mutating_component)
        coordinator.add_component("observer", observer)

        result = coordinator.run_all({"seed": 1})

        assert result.success is True
        assert result.results["observer"] is False
        assert seen == {"seed": 1}

    def test_run_sequential_chains_context_and_keeps_scalar_outputs(self):
        coordinator = Coordinator("sequential")
        coordinator.add_component("step1", lambda ctx: {"count": ctx.get("count", 0) + 1})
        coordinator.add_component("step2", lambda ctx: ctx["count"] * 10)
        coordinator.add_component("step3", lambda ctx: {"summary": ctx["step2"] + ctx["count"]})

        result = coordinator.run_sequential({"count": 1})

        assert result.success is True
        assert result.results == {"step1": {"count": 2}, "step2": 20, "step3": {"summary": 22}}

    @pytest.mark.parametrize("method_name", ["run_all", "run_sequential"])
    def test_component_errors_are_captured_in_both_modes(self, method_name):
        coordinator = Coordinator("errors")
        coordinator.add_component("good", lambda ctx: {"ok": True})
        coordinator.add_component("bad", lambda ctx: (_ for _ in ()).throw(ValueError("oops")))

        result = getattr(coordinator, method_name)({"seed": 1})

        assert result.success is False
        assert result.results["good"] == {"ok": True}
        assert result.errors["bad"] == "ValueError: oops"

    def test_component_lifecycle_and_listing(self):
        coordinator = Coordinator("lifecycle")
        coordinator.add_component("first", lambda ctx: ctx)
        coordinator.add_component("second", lambda ctx: ctx)

        listed = coordinator.list_components()
        listed.clear()

        assert coordinator.remove_component(" first ") is True
        assert coordinator.remove_component("first") is False
        assert coordinator.list_components() == ["second"]

    @pytest.mark.parametrize(
        ("factory", "error_type", "message"),
        [
            (lambda: Coordinator(""), ValueError, "Coordinator name cannot be empty"),
            (lambda: Coordinator("bad").add_component("", lambda ctx: ctx), ValueError, "Component name cannot be empty"),
            (lambda: Coordinator("bad").add_component("x", "not-callable"), TypeError, "component must be callable"),
            (lambda: Coordinator("bad").run_all("oops"), TypeError, "context must be a mapping or None"),
            (lambda: Coordinator("bad").run_sequential(123), TypeError, "context must be a mapping or None"),
        ],
    )
    def test_validation_rejects_invalid_coordinator_inputs(self, factory, error_type, message):
        with pytest.raises(error_type, match=message):
            factory()
