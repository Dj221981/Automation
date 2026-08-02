"""
Plugin Framework
================

A lightweight, registry-based plugin system that allows registering and running
extensible components (plugins, modules, tools) without coupling them to the
agent core.

Usage::

    from src.framework.plugin import Plugin, PluginRegistry, plugin

    # Option 1 – subclass Plugin
    class MyPlugin(Plugin):
        name = "my_plugin"
        version = "1.0.0"
        description = "Does something useful"

        def run(self, **kwargs):
            return {"status": "ok", "data": kwargs}

    registry = PluginRegistry()
    registry.register(MyPlugin())

    # Option 2 – decorate a callable
    @plugin(name="greet", description="Greet the user")
    def greet_fn(name: str = "world") -> dict:
        return {"greeting": f"Hello, {name}!"}

    registry.register(greet_fn)

    result = registry.run("greet", name="Alice")
"""

from __future__ import annotations

import inspect
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


__all__ = [
    "Plugin",
    "FunctionPlugin",
    "PluginResult",
    "PluginRegistry",
    "plugin",
]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class PluginResult:
    """Captures the outcome of a single plugin invocation."""

    plugin_name: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plugin_name": self.plugin_name,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "duration_seconds": round(self.duration_seconds, 6),
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Plugin ABC
# ---------------------------------------------------------------------------


class Plugin(ABC):
    """Abstract base class for all plugins.

    Subclasses must set ``name`` and ``version`` class-level attributes and
    implement :meth:`run`.
    """

    #: Unique identifier for the plugin. Must be set by the subclass.
    name: str = ""
    #: Semantic version string.
    version: str = "1.0.0"
    #: Human-readable description.
    description: str = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

    def setup(self) -> None:
        """Called once when the plugin is registered. Override for initialization."""

    def teardown(self) -> None:
        """Called when the plugin is unregistered. Override for cleanup."""

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """Execute the plugin and return a result."""

    def __repr__(self) -> str:
        return f"<Plugin: {self.name} v{self.version}>"


# ---------------------------------------------------------------------------
# Function-backed plugin (created via @plugin decorator)
# ---------------------------------------------------------------------------


class FunctionPlugin(Plugin):
    """Wraps a callable as a plugin."""

    def __init__(
        self,
        func: Callable[..., Any],
        name: str,
        version: str = "1.0.0",
        description: str = "",
    ) -> None:
        if not callable(func):
            raise TypeError("func must be callable")
        if not name.strip():
            raise ValueError("Plugin name cannot be empty")
        self._func = func
        self.name = name.strip()
        self.version = version
        self.description = description or (inspect.getdoc(func) or "")

    def run(self, **kwargs: Any) -> Any:
        return self._func(**kwargs)


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def plugin(
    name: str,
    *,
    version: str = "1.0.0",
    description: str = "",
) -> Callable[[Callable[..., Any]], "FunctionPlugin"]:
    """Decorate a function to turn it into a :class:`FunctionPlugin`.

    The decorated object is replaced by the :class:`FunctionPlugin` instance
    so it can be passed directly to :meth:`PluginRegistry.register`.
    """

    def decorator(func: Callable[..., Any]) -> FunctionPlugin:
        return FunctionPlugin(func, name=name, version=version, description=description)

    return decorator


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class PluginRegistry:
    """Thread-safe registry for discovering, registering, and running plugins.

    Plugins are keyed by their :attr:`Plugin.name`.  Re-registering a plugin
    under the same name replaces the previous entry.
    """

    def __init__(self) -> None:
        self._plugins: Dict[str, Plugin] = {}
        self._disabled: set[str] = set()
        self._lock = threading.RLock()
        self._run_counts: Dict[str, int] = {}
        self._error_counts: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, plugin_instance: Plugin) -> "PluginRegistry":
        """Register a plugin, calling its :meth:`Plugin.setup` hook.

        Returns *self* to allow chaining.
        """
        if not isinstance(plugin_instance, Plugin):
            raise TypeError(f"Expected a Plugin instance, got {type(plugin_instance).__name__!r}")
        if not plugin_instance.name.strip():
            raise ValueError("Plugin must have a non-empty name")

        with self._lock:
            name = plugin_instance.name
            if name in self._plugins:
                # tear down the old one before replacing
                try:
                    self._plugins[name].teardown()
                except Exception:
                    pass
            plugin_instance.setup()
            self._plugins[name] = plugin_instance
            self._disabled.discard(name)
            self._run_counts.setdefault(name, 0)
            self._error_counts.setdefault(name, 0)
        return self

    def unregister(self, name: str) -> bool:
        """Remove a plugin by name, calling its :meth:`Plugin.teardown` hook."""
        with self._lock:
            instance = self._plugins.pop(name, None)
            if instance is None:
                return False
            self._disabled.discard(name)
            try:
                instance.teardown()
            except Exception:
                pass
            return True

    def enable(self, name: str) -> bool:
        """Re-enable a previously disabled plugin."""
        with self._lock:
            if name not in self._plugins:
                return False
            self._disabled.discard(name)
            return True

    def disable(self, name: str) -> bool:
        """Disable a plugin so :meth:`run` will refuse to call it."""
        with self._lock:
            if name not in self._plugins:
                return False
            self._disabled.add(name)
            return True

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_plugins(self, *, include_disabled: bool = True) -> List[Dict[str, Any]]:
        """Return a list of plugin metadata dicts."""
        with self._lock:
            result = []
            for name, p in self._plugins.items():
                result.append({
                    "name": name,
                    "version": p.version,
                    "description": p.description,
                    "enabled": name not in self._disabled,
                    "run_count": self._run_counts.get(name, 0),
                    "error_count": self._error_counts.get(name, 0),
                })
            if not include_disabled:
                result = [r for r in result if r["enabled"]]
            return result

    def get(self, name: str) -> Optional[Plugin]:
        """Return the plugin instance by name, or *None* if not found."""
        with self._lock:
            return self._plugins.get(name)

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._plugins

    def __len__(self) -> int:
        with self._lock:
            return len(self._plugins)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, plugin_name: str, **kwargs: Any) -> PluginResult:
        """Run a registered plugin by name.

        Returns a :class:`PluginResult` regardless of whether the plugin
        succeeded or raised an exception.
        """
        with self._lock:
            instance = self._plugins.get(plugin_name)
            disabled = plugin_name in self._disabled

        if instance is None:
            return PluginResult(
                plugin_name=plugin_name,
                success=False,
                error=f"Plugin not found: {plugin_name!r}",
            )
        if disabled:
            return PluginResult(
                plugin_name=plugin_name,
                success=False,
                error=f"Plugin is disabled: {plugin_name!r}",
            )

        start = time.monotonic()
        try:
            output = instance.run(**kwargs)
            elapsed = time.monotonic() - start
            with self._lock:
                self._run_counts[plugin_name] = self._run_counts.get(plugin_name, 0) + 1
            return PluginResult(
                plugin_name=plugin_name,
                success=True,
                output=output,
                duration_seconds=elapsed,
            )
        except Exception as exc:
            elapsed = time.monotonic() - start
            with self._lock:
                self._run_counts[plugin_name] = self._run_counts.get(plugin_name, 0) + 1
                self._error_counts[plugin_name] = self._error_counts.get(plugin_name, 0) + 1
            return PluginResult(
                plugin_name=plugin_name,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                duration_seconds=elapsed,
            )

    def run_all(self, **kwargs: Any) -> List[PluginResult]:
        """Run every enabled plugin and return the collected results."""
        with self._lock:
            names = [n for n in self._plugins if n not in self._disabled]
        return [self.run(plugin_name=n, **kwargs) for n in names]

    def stats(self) -> Dict[str, Any]:
        """Return aggregate statistics for the registry."""
        with self._lock:
            total_runs = sum(self._run_counts.values())
            total_errors = sum(self._error_counts.values())
            return {
                "registered": len(self._plugins),
                "disabled": len(self._disabled),
                "enabled": len(self._plugins) - len(self._disabled),
                "total_runs": total_runs,
                "total_errors": total_errors,
                "error_rate": (total_errors / total_runs) if total_runs else 0.0,
            }
