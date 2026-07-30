# API

## Agent System
- `AgentSystem.create_task(description, parameters, priority)`
- `AgentSystem.submit_task(task, agent_id=None)`
- `AgentSystem.execute_task(task_id, agent_id)`
- `AgentSystem.get_observability_snapshot()`

## Observability
- `src/observability/tracing.py`
- `src/observability/structured_logging.py`
- `src/observability/metrics.py`

## Monitoring/Resilience
- `src/monitoring/*`
- `src/resilience/*`
