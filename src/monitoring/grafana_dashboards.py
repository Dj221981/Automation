"""Grafana dashboard definitions for Automation metrics."""

from __future__ import annotations

from typing import Any, Dict, List


def agent_performance_dashboard() -> Dict[str, Any]:
    return {
        "title": "Agent Performance",
        "panels": [
            {"type": "graph", "title": "Tasks Completed", "query": "agent_tasks_completed_total"},
            {"type": "graph", "title": "Tasks Failed", "query": "agent_tasks_failed_total"},
            {"type": "graph", "title": "Success Rate", "query": "agent_success_rate"},
        ],
    }


def training_metrics_dashboard() -> Dict[str, Any]:
    return {
        "title": "Training Metrics",
        "panels": [
            {"type": "graph", "title": "Train Steps", "query": "nn_train_steps_total"},
            {"type": "graph", "title": "Training Loss", "query": "nn_loss"},
            {"type": "graph", "title": "Epsilon", "query": "nn_epsilon"},
        ],
    }


def system_health_dashboard() -> Dict[str, Any]:
    return {
        "title": "System Health",
        "panels": [
            {"type": "graph", "title": "Queue Depth", "query": "task_queue_depth"},
            {"type": "graph", "title": "Memory Usage", "query": "resource_memory_usage_bytes"},
            {"type": "graph", "title": "CPU Usage", "query": "resource_cpu_usage_percent"},
        ],
    }


def alerts_configuration() -> List[Dict[str, Any]]:
    return [
        {"name": "HighTaskFailureRate", "expr": "(agent_tasks_failed_total / (agent_tasks_completed_total + agent_tasks_failed_total)) > 0.2"},
        {"name": "QueueDepthWarning", "expr": "task_queue_depth > 1000"},
        {"name": "HighCPU", "expr": "resource_cpu_usage_percent > 90"},
    ]
