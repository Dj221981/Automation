"""Distributed tracing helpers with graceful OpenTelemetry fallback."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Mapping, Optional

try:
    from opentelemetry import propagate, trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.zipkin.json import ZipkinExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    OTEL_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    OTEL_AVAILABLE = False
    propagate = None
    trace = None


@dataclass(frozen=True)
class TracingConfig:
    service_name: str = "automation"
    exporter: str = "none"  # none|jaeger|zipkin
    endpoint: Optional[str] = None


class TracingManager:
    """Central tracing entrypoint used across agents and training."""

    def __init__(self, config: Optional[TracingConfig] = None):
        self.config = config or TracingConfig()
        self._enabled = bool(OTEL_AVAILABLE)
        self._tracer = None
        if self._enabled:
            self._configure_provider()

    @property
    def enabled(self) -> bool:
        return self._enabled and self._tracer is not None

    def _configure_provider(self) -> None:
        provider = TracerProvider(resource=Resource.create({"service.name": self.config.service_name}))
        exporter_name = self.config.exporter.lower().strip()
        endpoint = self.config.endpoint

        if exporter_name == "jaeger" and endpoint:
            exporter = JaegerExporter(collector_endpoint=endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))
        elif exporter_name == "zipkin" and endpoint:
            exporter = ZipkinExporter(endpoint=endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))

        trace.set_tracer_provider(provider)
        self._tracer = trace.get_tracer(self.config.service_name)

    @contextmanager
    def start_span(self, name: str, attributes: Optional[Mapping[str, Any]] = None) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        with self._tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            yield

    def inject_context(self, carrier: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        payload = carrier or {}
        if self.enabled:
            propagate.inject(payload)
        return payload

    def extract_context(self, carrier: Mapping[str, str]) -> None:
        if self.enabled:
            propagate.extract(carrier)


_GLOBAL_TRACING_MANAGER = TracingManager()


def get_tracing_manager() -> TracingManager:
    return _GLOBAL_TRACING_MANAGER
