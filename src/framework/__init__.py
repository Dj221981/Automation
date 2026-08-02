"""
Framework Layer
===============

A reusable framework for registering extensible components (plugins, modules,
tools), coordinating agent execution, and orchestrating multi-step workflows.

Public API
----------

Plugin system::

    from src.framework import Plugin, PluginRegistry, plugin, FunctionPlugin

Execution engine::

    from src.framework import ExecutionEngine, EngineConfig, EngineHook, EngineState

Orchestration primitives::

    from src.framework import (
        Pipeline, PipelineStep,
        Router,
        Coordinator,
    )
"""

from src.framework.plugin import (
    FunctionPlugin,
    Plugin,
    PluginRegistry,
    PluginResult,
    plugin,
)
from src.framework.engine import (
    EngineConfig,
    EngineHook,
    EngineState,
    ExecutionEngine,
    TaskRecord,
)
from src.framework.orchestration import (
    Coordinator,
    CoordinatorResult,
    Pipeline,
    PipelineResult,
    PipelineStep,
    PipelineStepResult,
    Router,
    RouterResult,
)

__all__ = [
    # plugin
    "Plugin",
    "FunctionPlugin",
    "PluginResult",
    "PluginRegistry",
    "plugin",
    # engine
    "EngineState",
    "EngineConfig",
    "EngineHook",
    "TaskRecord",
    "ExecutionEngine",
    # orchestration
    "PipelineStepResult",
    "PipelineStep",
    "PipelineResult",
    "Pipeline",
    "RouterResult",
    "Router",
    "CoordinatorResult",
    "Coordinator",
]
