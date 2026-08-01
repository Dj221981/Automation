"""Structured logging with JSON/plaintext format and trace correlation context."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    from opentelemetry import trace

    OTEL_TRACE_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    OTEL_TRACE_AVAILABLE = False
    trace = None


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    output_format: str = "json"  # json|plaintext


class _ContextFilter(logging.Filter):
    def __init__(self, correlation_id: Optional[str] = None):
        super().__init__()
        self._correlation_id = correlation_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = self._correlation_id or getattr(record, "correlation_id", None) or "-"
        trace_id = "-"
        span_id = "-"
        if OTEL_TRACE_AVAILABLE:
            span = trace.get_current_span()
            span_context = span.get_span_context() if span else None
            if span_context and span_context.is_valid:
                trace_id = f"{span_context.trace_id:032x}"
                span_id = f"{span_context.span_id:016x}"
        record.trace_id = getattr(record, "trace_id", trace_id)
        record.span_id = getattr(record, "span_id", span_id)
        return True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", "-"),
            "trace_id": getattr(record, "trace_id", "-"),
            "span_id": getattr(record, "span_id", "-"),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(config: Optional[LoggingConfig] = None, correlation_id: Optional[str] = None) -> None:
    cfg = config or LoggingConfig()
    root = logging.getLogger()
    root.setLevel(getattr(logging, cfg.level.upper(), logging.INFO))

    if root.handlers:
        for handler in list(root.handlers):
            root.removeHandler(handler)

    handler = logging.StreamHandler()
    if cfg.output_format.lower() == "plaintext":
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s correlation_id=%(correlation_id)s "
                "trace_id=%(trace_id)s span_id=%(span_id)s %(message)s"
            )
        )
    else:
        handler.setFormatter(_JsonFormatter())

    handler.addFilter(_ContextFilter(correlation_id=correlation_id))
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
