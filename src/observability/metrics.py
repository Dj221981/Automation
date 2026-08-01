"""Prometheus metrics facade with no-op fallback."""

from __future__ import annotations

import threading
from typing import Any, Dict

try:
    from prometheus_client import Counter, Gauge, Histogram

    PROM_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    PROM_AVAILABLE = False
    Counter = Gauge = Histogram = None


class AutomationMetrics:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._snapshot: Dict[str, float] = {
            "tasks_completed": 0.0,
            "tasks_failed": 0.0,
            "avg_duration": 0.0,
            "success_rate": 0.0,
            "train_steps": 0.0,
            "loss": 0.0,
            "epsilon": 0.0,
            "learning_rate": 0.0,
            "queue_depth": 0.0,
            "processing_time": 0.0,
            "throughput": 0.0,
            "memory_usage": 0.0,
            "cpu_usage": 0.0,
            "model_size": 0.0,
        }

        if PROM_AVAILABLE:
            self.agent_tasks_completed = Counter("agent_tasks_completed_total", "Completed tasks")
            self.agent_tasks_failed = Counter("agent_tasks_failed_total", "Failed tasks")
            self.agent_task_duration = Histogram("agent_task_duration_seconds", "Task execution duration")
            self.agent_success_rate = Gauge("agent_success_rate", "Agent success rate")

            self.nn_train_steps = Counter("nn_train_steps_total", "Neural network train steps")
            self.nn_loss = Gauge("nn_loss", "Latest training loss")
            self.nn_epsilon = Gauge("nn_epsilon", "Current epsilon")
            self.nn_learning_rate = Gauge("nn_learning_rate", "Learning rate")

            self.task_queue_depth = Gauge("task_queue_depth", "Task queue depth")
            self.task_processing_time = Histogram("task_processing_time_seconds", "Task processing time")
            self.task_throughput = Gauge("task_throughput", "Tasks per second")

            self.resource_memory_usage = Gauge("resource_memory_usage_bytes", "Resident memory usage")
            self.resource_cpu_usage = Gauge("resource_cpu_usage_percent", "CPU usage")
            self.resource_model_size = Gauge("resource_model_size_bytes", "Model size in bytes")

    def record_agent_task(self, success: bool, duration_seconds: float) -> None:
        with self._lock:
            if success:
                self._snapshot["tasks_completed"] += 1
                if PROM_AVAILABLE:
                    self.agent_tasks_completed.inc()
            else:
                self._snapshot["tasks_failed"] += 1
                if PROM_AVAILABLE:
                    self.agent_tasks_failed.inc()

            completed = self._snapshot["tasks_completed"]
            failed = self._snapshot["tasks_failed"]
            total = completed + failed
            self._snapshot["success_rate"] = (completed / total) if total else 0.0
            prev = self._snapshot["avg_duration"]
            self._snapshot["avg_duration"] = prev + ((duration_seconds - prev) / total) if total else duration_seconds

            if PROM_AVAILABLE:
                self.agent_task_duration.observe(max(0.0, duration_seconds))
                self.agent_success_rate.set(self._snapshot["success_rate"])

    def record_training_step(self, loss: float, epsilon: float, learning_rate: float) -> None:
        with self._lock:
            self._snapshot["train_steps"] += 1
            self._snapshot["loss"] = float(loss)
            self._snapshot["epsilon"] = float(epsilon)
            self._snapshot["learning_rate"] = float(learning_rate)
            if PROM_AVAILABLE:
                self.nn_train_steps.inc()
                self.nn_loss.set(loss)
                self.nn_epsilon.set(epsilon)
                self.nn_learning_rate.set(learning_rate)

    def record_task_system(self, queue_depth: int, processing_time: float, throughput: float) -> None:
        with self._lock:
            self._snapshot["queue_depth"] = float(max(0, queue_depth))
            self._snapshot["processing_time"] = float(max(0.0, processing_time))
            self._snapshot["throughput"] = float(max(0.0, throughput))
            if PROM_AVAILABLE:
                self.task_queue_depth.set(queue_depth)
                self.task_processing_time.observe(max(0.0, processing_time))
                self.task_throughput.set(max(0.0, throughput))

    def record_resources(self, memory_usage: float, cpu_usage: float, model_size: float) -> None:
        with self._lock:
            self._snapshot["memory_usage"] = float(max(0.0, memory_usage))
            self._snapshot["cpu_usage"] = float(max(0.0, cpu_usage))
            self._snapshot["model_size"] = float(max(0.0, model_size))
            if PROM_AVAILABLE:
                self.resource_memory_usage.set(memory_usage)
                self.resource_cpu_usage.set(cpu_usage)
                self.resource_model_size.set(model_size)

    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._snapshot)


_GLOBAL_METRICS = AutomationMetrics()


def get_metrics_registry() -> AutomationMetrics:
    return _GLOBAL_METRICS
