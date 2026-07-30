# Troubleshooting

## CI fails with `manage.py` not found
Use pytest-based workflow (`python -m pytest tests/ -v`) instead of Django test runner.

## Metrics endpoint unavailable
Install `prometheus-client` and start exporter from `src/monitoring/prometheus_exporter.py`.

## Missing traces
Install OpenTelemetry dependencies and set valid Jaeger/Zipkin endpoint.
