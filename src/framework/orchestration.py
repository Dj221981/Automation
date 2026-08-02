"""
Orchestration Framework
=======================

Coordination primitives that let you compose components – plugins, callables,
or agent tasks – into structured execution patterns without coupling the logic
to specific agent implementations.

Three main abstractions are provided:

**Pipeline**
    Executes a linear sequence of :class:`PipelineStep` objects.  The output
    of each step is forwarded as the ``context`` to the next step so data
    flows naturally through the chain.

**Router**
    Dispatches a payload to a named handler based on a *routing key*.  Useful
    when the same orchestrator needs to handle multiple action types.

**Coordinator**
    Aggregates a :class:`~src.framework.plugin.PluginRegistry`,
    an :class:`~src.framework.engine.ExecutionEngine`, and user-defined
    components into a single entry-point.  Provides broadcast (fan-out) and
    sequential execution across all registered components.

Usage::

    from src.framework.orchestration import Pipeline, PipelineStep, Router, Coordinator

    # --- Pipeline ---
    def step_a(context):
        context["a"] = 1
        return context

    def step_b(context):
        context["b"] = context["a"] + 1
        return context

    pipeline = Pipeline("my-pipeline")
    pipeline.add_step(PipelineStep("step_a", step_a))
    pipeline.add_step(PipelineStep("step_b", step_b))
    result = pipeline.run({"initial": True})

    # --- Router ---
    router = Router()
    router.register("greet", lambda payload: f"Hello, {payload['name']}!")
    outcome = router.dispatch("greet", {"name": "Alice"})

    # --- Coordinator ---
    coordinator = Coordinator("main")
    coordinator.add_component("greeter", lambda ctx: {**ctx, "greeted": True})
    summary = coordinator.run_all({"x": 1})
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional

__all__ = [
    "PipelineStepResult",
    "PipelineStep",
    "PipelineResult",
    "Pipeline",
    "RouterResult",
    "Router",
    "CoordinatorResult",
    "Coordinator",
]


def _normalize_name(value: str, field_name: str) -> str:
    """Return a normalized non-empty string value for a public name field."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a non-empty string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty")
    return normalized


def _coerce_context(
    context: Optional[Mapping[str, Any]],
    field_name: str,
) -> Dict[str, Any]:
    """Copy a context mapping into a plain dict.

    ``None`` is treated as an empty context. Non-mapping inputs raise a clear
    :class:`TypeError` instead of leaking lower-level conversion errors.
    """
    if context is None:
        return {}
    if not isinstance(context, Mapping):
        raise TypeError(f"{field_name} must be a mapping or None")
    return dict(context)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass
class PipelineStepResult:
    """Records the outcome of a single pipeline step."""

    step_name: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_name": self.step_name,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "duration_seconds": round(self.duration_seconds, 6),
        }


@dataclass
class PipelineStep:
    """A single step in a :class:`Pipeline`.

    *handler* receives the current context dict and must return the (possibly
    modified) context dict to pass to the next step.  If it raises, the
    pipeline records the failure and – unless ``stop_on_error`` is set –
    continues with the unchanged context.
    """

    name: str
    handler: Callable[[Dict[str, Any]], Dict[str, Any]]
    description: str = ""
    #: When *True*, a failure in this step aborts the whole pipeline.
    stop_on_error: bool = True

    def __post_init__(self) -> None:
        self.name = _normalize_name(self.name, "PipelineStep name")
        if not callable(self.handler):
            raise TypeError("PipelineStep handler must be callable")


@dataclass
class PipelineResult:
    """Aggregate result of a full pipeline run."""

    pipeline_name: str
    success: bool
    final_context: Dict[str, Any] = field(default_factory=dict)
    step_results: List[PipelineStepResult] = field(default_factory=list)
    error: Optional[str] = None
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_name": self.pipeline_name,
            "success": self.success,
            "final_context": self.final_context,
            "step_results": [s.to_dict() for s in self.step_results],
            "error": self.error,
            "duration_seconds": round(self.duration_seconds, 6),
        }


class Pipeline:
    """Executes an ordered list of :class:`PipelineStep` objects.

    The context (a plain ``dict``) is passed through each step in order.
    Steps may mutate or replace the context by returning a new dict.
    """

    def __init__(self, name: str) -> None:
        self.name = _normalize_name(name, "Pipeline name")
        self._steps: List[PipelineStep] = []
        self._lock = threading.RLock()

    def add_step(self, step: PipelineStep) -> "Pipeline":
        """Append a step to the pipeline.  Returns *self* for chaining."""
        if not isinstance(step, PipelineStep):
            raise TypeError("step must be a PipelineStep instance")
        with self._lock:
            self._steps.append(step)
        return self

    def remove_step(self, name: str) -> bool:
        """Remove the first step with the given name.  Returns whether it was found."""
        normalized_name = _normalize_name(name, "step name")
        with self._lock:
            for i, s in enumerate(self._steps):
                if s.name == normalized_name:
                    self._steps.pop(i)
                    return True
        return False

    @property
    def steps(self) -> List[PipelineStep]:
        with self._lock:
            return list(self._steps)

    def run(self, initial_context: Optional[Mapping[str, Any]] = None) -> PipelineResult:
        """Execute all steps sequentially and return a :class:`PipelineResult`.

        ``initial_context`` may be ``None`` or any mapping. Each successful
        step must return a mapping; otherwise the step is recorded as failed.
        """
        context = _coerce_context(initial_context, "initial_context")
        step_results: List[PipelineStepResult] = []
        pipeline_start = time.monotonic()
        aborted = False
        abort_error: Optional[str] = None

        with self._lock:
            steps = list(self._steps)

        for step in steps:
            start = time.monotonic()
            try:
                output = step.handler(context)
                if not isinstance(output, Mapping):
                    raise TypeError(
                        f"Pipeline step {step.name!r} must return a mapping, "
                        f"got {type(output).__name__}"
                    )
                elapsed = time.monotonic() - start
                context = dict(output)
                step_results.append(
                    PipelineStepResult(
                        step_name=step.name,
                        success=True,
                        output=output,
                        duration_seconds=elapsed,
                    )
                )
            except Exception as exc:
                elapsed = time.monotonic() - start
                err_msg = f"{type(exc).__name__}: {exc}"
                step_results.append(
                    PipelineStepResult(
                        step_name=step.name,
                        success=False,
                        error=err_msg,
                        duration_seconds=elapsed,
                    )
                )
                if step.stop_on_error:
                    aborted = True
                    abort_error = f"Pipeline aborted at step {step.name!r}: {err_msg}"
                    break

        total_elapsed = time.monotonic() - pipeline_start
        overall_success = not aborted and all(r.success for r in step_results)
        return PipelineResult(
            pipeline_name=self.name,
            success=overall_success,
            final_context=context,
            step_results=step_results,
            error=abort_error,
            duration_seconds=total_elapsed,
        )

    def __repr__(self) -> str:
        return f"<Pipeline: {self.name!r} ({len(self._steps)} steps)>"


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


@dataclass
class RouterResult:
    """Outcome of a :class:`Router` dispatch call."""

    routing_key: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "routing_key": self.routing_key,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "duration_seconds": round(self.duration_seconds, 6),
        }


class Router:
    """Routes a payload to a named handler based on a *routing key*.

    Handlers are plain callables that accept a single ``payload`` argument
    and return any value.
    """

    def __init__(self, default_handler: Optional[Callable[[Any], Any]] = None) -> None:
        if default_handler is not None and not callable(default_handler):
            raise TypeError("default_handler must be callable or None")
        self._handlers: Dict[str, Callable[[Any], Any]] = {}
        self._default_handler = default_handler
        self._lock = threading.RLock()
        self._dispatch_counts: Dict[str, int] = {}

    def register(self, routing_key: str, handler: Callable[[Any], Any]) -> "Router":
        """Register a handler for *routing_key*.  Returns *self* for chaining."""
        normalized_key = _normalize_name(routing_key, "routing_key")
        if not callable(handler):
            raise TypeError("handler must be callable")
        with self._lock:
            self._handlers[normalized_key] = handler
            self._dispatch_counts.setdefault(normalized_key, 0)
        return self

    def unregister(self, routing_key: str) -> bool:
        """Remove the handler for *routing_key*."""
        normalized_key = _normalize_name(routing_key, "routing_key")
        with self._lock:
            if normalized_key in self._handlers:
                del self._handlers[normalized_key]
                return True
            return False

    def dispatch(self, routing_key: str, payload: Any = None) -> RouterResult:
        """Dispatch *payload* to the handler registered for *routing_key*.

        Falls back to the *default_handler* when no specific handler is found.
        Returns a :class:`RouterResult` whether or not dispatch succeeded.
        """
        normalized_key = _normalize_name(routing_key, "routing_key")
        with self._lock:
            handler = self._handlers.get(normalized_key) or self._default_handler

        if handler is None:
            return RouterResult(
                routing_key=normalized_key,
                success=False,
                error=f"No handler registered for routing key {normalized_key!r}",
            )

        start = time.monotonic()
        try:
            output = handler(payload)
            elapsed = time.monotonic() - start
            with self._lock:
                self._dispatch_counts[normalized_key] = self._dispatch_counts.get(normalized_key, 0) + 1
            return RouterResult(
                routing_key=normalized_key,
                success=True,
                output=output,
                duration_seconds=elapsed,
            )
        except Exception as exc:
            elapsed = time.monotonic() - start
            return RouterResult(
                routing_key=normalized_key,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                duration_seconds=elapsed,
            )

    def list_routes(self) -> List[str]:
        """Return all registered routing keys."""
        with self._lock:
            return list(self._handlers.keys())

    def stats(self) -> Dict[str, int]:
        """Return per-route dispatch counts."""
        with self._lock:
            return dict(self._dispatch_counts)

    def __repr__(self) -> str:
        return f"<Router routes={self.list_routes()}>"


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


@dataclass
class CoordinatorResult:
    """Aggregate result of a :class:`Coordinator` broadcast or sequential run."""

    coordinator_name: str
    success: bool
    results: Dict[str, Any] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coordinator_name": self.coordinator_name,
            "success": self.success,
            "results": self.results,
            "errors": self.errors,
            "duration_seconds": round(self.duration_seconds, 6),
        }


class Coordinator:
    """Aggregates named components and provides fan-out execution.

    A *component* is any callable that accepts a context dict and returns a
    value (dict, any scalar, etc.).  Components are identified by a string
    name and can be added or removed at runtime.

    :meth:`run_all` calls every registered component in insertion order and
    collects their outputs into a :class:`CoordinatorResult`.
    """

    def __init__(self, name: str) -> None:
        self.name = _normalize_name(name, "Coordinator name")
        self._components: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
        self._lock = threading.RLock()

    def add_component(
        self,
        name: str,
        component: Callable[[Dict[str, Any]], Any],
    ) -> "Coordinator":
        """Register a component.  Returns *self* for chaining."""
        normalized_name = _normalize_name(name, "Component name")
        if not callable(component):
            raise TypeError("component must be callable")
        with self._lock:
            self._components[normalized_name] = component
        return self

    def remove_component(self, name: str) -> bool:
        """Remove a component by name."""
        normalized_name = _normalize_name(name, "component name")
        with self._lock:
            if normalized_name in self._components:
                del self._components[normalized_name]
                return True
            return False

    def list_components(self) -> List[str]:
        """Return names of all registered components."""
        with self._lock:
            return list(self._components.keys())

    def run_all(self, context: Optional[Mapping[str, Any]] = None) -> CoordinatorResult:
        """Call every component with *context* and aggregate results.

        Failures in individual components are recorded in
        :attr:`CoordinatorResult.errors` and do not abort other components.
        Each component receives its own shallow copy of the input context so
        one component cannot accidentally mutate another component's view.
        """
        ctx = _coerce_context(context, "context")
        results: Dict[str, Any] = {}
        errors: Dict[str, str] = {}
        start = time.monotonic()

        with self._lock:
            components = list(self._components.items())

        for comp_name, comp_fn in components:
            try:
                results[comp_name] = comp_fn(dict(ctx))
            except Exception as exc:
                errors[comp_name] = f"{type(exc).__name__}: {exc}"

        elapsed = time.monotonic() - start
        return CoordinatorResult(
            coordinator_name=self.name,
            success=len(errors) == 0,
            results=results,
            errors=errors,
            duration_seconds=elapsed,
        )

    def run_sequential(self, context: Optional[Mapping[str, Any]] = None) -> CoordinatorResult:
        """Call components sequentially, passing each component's output as the
        context for the next component (dict outputs are merged; non-dict
        outputs are stored under the component name and the original context
        continues). ``context`` may be ``None`` or any mapping.
        """
        ctx = _coerce_context(context, "context")
        results: Dict[str, Any] = {}
        errors: Dict[str, str] = {}
        start = time.monotonic()

        with self._lock:
            components = list(self._components.items())

        for comp_name, comp_fn in components:
            try:
                output = comp_fn(ctx)
                results[comp_name] = output
                if isinstance(output, dict):
                    ctx.update(output)
                else:
                    ctx[comp_name] = output
            except Exception as exc:
                errors[comp_name] = f"{type(exc).__name__}: {exc}"

        elapsed = time.monotonic() - start
        return CoordinatorResult(
            coordinator_name=self.name,
            success=len(errors) == 0,
            results=results,
            errors=errors,
            duration_seconds=elapsed,
        )

    def __repr__(self) -> str:
        return f"<Coordinator: {self.name!r} ({len(self._components)} components)>"
