# Deployment

## Docker
- Build: `docker build -t automation .`
- Run: `docker run -p 9464:9464 automation`

## Kubernetes
- Expose `/metrics` and health probes.
- Configure `config/observability.yaml` with cluster endpoints.

## Cloud
- Route traces to managed Jaeger/Zipkin.
- Scrape Prometheus metrics and import Grafana dashboard templates.
