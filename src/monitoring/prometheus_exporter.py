"""Prometheus exporter endpoint utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    from prometheus_client import start_http_server

    PROM_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    PROM_AVAILABLE = False
    start_http_server = None


@dataclass(frozen=True)
class PrometheusExporterConfig:
    host: str = "0.0.0.0"
    port: int = 9464


class PrometheusExporter:
    def __init__(self, config: Optional[PrometheusExporterConfig] = None):
        self.config = config or PrometheusExporterConfig()
        self._started = False

    @property
    def metrics_path(self) -> str:
        return "/metrics"

    def start(self) -> bool:
        if not PROM_AVAILABLE:
            return False
        if not self._started:
            start_http_server(addr=self.config.host, port=self.config.port)
            self._started = True
        return True
