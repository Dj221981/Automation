from src.agents.super_agentic_agents import AgentSystem, ExecutorAgent
from src.monitoring.health_checks import HealthChecker, HealthCheckResult
from src.monitoring.thresholds import ThresholdMonitor
from src.observability.metrics import AutomationMetrics
from src.observability.tracing import get_tracing_manager
from src.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerOpenError


def test_tracing_context_injection_returns_mapping():
    tracing = get_tracing_manager()
    carrier = tracing.inject_context({})
    assert isinstance(carrier, dict)


def test_metrics_snapshot_updates_after_agent_task_recording():
    metrics = AutomationMetrics()
    metrics.record_agent_task(success=True, duration_seconds=0.2)
    metrics.record_agent_task(success=False, duration_seconds=0.4)
    snap = metrics.snapshot()

    assert snap["tasks_completed"] == 1.0
    assert snap["tasks_failed"] == 1.0
    assert 0.0 <= snap["success_rate"] <= 1.0


def test_threshold_monitor_detects_queue_alert():
    monitor = ThresholdMonitor()
    alerts = monitor.evaluate(metrics={"success_rate": 1.0, "queue_depth": 2000, "avg_duration": 0.1})
    assert any(alert["name"] == "queue_depth" for alert in alerts)


def test_circuit_breaker_opens_after_failures():
    breaker = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=2, recovery_timeout_seconds=60.0))

    def fail() -> None:
        raise RuntimeError("boom")

    for _ in range(2):
        try:
            breaker.call(fail)
        except RuntimeError:
            pass

    try:
        breaker.call(lambda: "ok")
    except CircuitBreakerOpenError:
        assert True
    else:
        raise AssertionError("circuit breaker should be open")


def test_health_checker_runs_registered_checks():
    checker = HealthChecker()
    checker.register("ok", lambda: HealthCheckResult("ok", True, "fine"))
    report = checker.run()
    assert report["healthy"] is True
    assert report["checks"][0]["name"] == "ok"


def test_agent_system_observability_snapshot_exposes_new_fields():
    system = AgentSystem("obs2")
    agent = ExecutorAgent("worker")
    assert system.add_agent(agent)

    task = system.create_task("obs", {})
    assert system.submit_task(task, agent.id)
    system.execute_task(task.id, agent.id)

    snap = system.get_observability_snapshot()
    assert "health" in snap
    assert "performance" in snap
    assert "prometheus_metrics" in snap
