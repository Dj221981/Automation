# Monitoring

## Prometheus
- Start exporter on port `9464`
- Scrape `/metrics`

## Grafana
Use dashboard definitions from `src/monitoring/grafana_dashboards.py`:
- Agent performance
- Training metrics
- System health

## Alerting
Threshold logic is implemented in `src/monitoring/thresholds.py`.
