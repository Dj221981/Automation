"""Threshold and anomaly evaluation for alerting."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Dict, List


@dataclass(frozen=True)
class ThresholdConfig:
    task_failure_rate_max: float = 0.2
    queue_depth_warning: int = 1000
    memory_usage_bytes_max: float = 2_000_000_000
    cpu_usage_percent_max: float = 90.0
    response_time_slo_seconds: float = 2.0
    training_loss_zscore_threshold: float = 3.0


class ThresholdMonitor:
    def __init__(self, config: ThresholdConfig | None = None):
        self.config = config or ThresholdConfig()

    def evaluate(self, *, metrics: Dict[str, float], recent_losses: List[float] | None = None) -> List[Dict[str, str]]:
        alerts: List[Dict[str, str]] = []

        if metrics.get("success_rate", 1.0) < (1.0 - self.config.task_failure_rate_max):
            alerts.append({"name": "task_failure_rate", "severity": "high", "message": "Task failure rate exceeded"})
        if metrics.get("queue_depth", 0.0) > self.config.queue_depth_warning:
            alerts.append({"name": "queue_depth", "severity": "medium", "message": "Queue depth warning"})
        if metrics.get("memory_usage", 0.0) > self.config.memory_usage_bytes_max:
            alerts.append({"name": "memory_usage", "severity": "high", "message": "Memory threshold exceeded"})
        if metrics.get("cpu_usage", 0.0) > self.config.cpu_usage_percent_max:
            alerts.append({"name": "cpu_usage", "severity": "high", "message": "CPU threshold exceeded"})
        if metrics.get("avg_duration", 0.0) > self.config.response_time_slo_seconds:
            alerts.append({"name": "response_time_slo", "severity": "medium", "message": "Response-time SLO violated"})

        if recent_losses and len(recent_losses) >= 5:
            baseline = recent_losses[:-1]
            latest = recent_losses[-1]
            std = pstdev(baseline)
            if std > 0:
                z_score = abs((latest - mean(baseline)) / std)
                if z_score >= self.config.training_loss_zscore_threshold:
                    alerts.append({"name": "training_loss_anomaly", "severity": "medium", "message": "Training loss anomaly"})

        return alerts
