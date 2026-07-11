"""
Ai-morphasis 2.0 – REST API
============================

Lightweight FastAPI application that exposes health, status, and task
submission endpoints for the AgentSystem.

Run with::

    uvicorn src.agents.api:app --reload

Endpoints:
    GET  /health  – System health check
    GET  /status  – Full system status
    POST /tasks   – Create and submit a new task
"""

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel as PydanticBaseModel, Field

from src.agents.super_agentic_agents import (
    AgentSystem,
    TaskConfig,
    TaskPriority,
)

app = FastAPI(
    title="Ai-morphasis Agent System API",
    description="Production-ready multi-agent framework REST interface",
    version="2.0.0",
)

# ---------------------------------------------------------------------------
# Shared system instance
# ---------------------------------------------------------------------------
system = AgentSystem("Ai-morphasis-2.0")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class TaskRequest(PydanticBaseModel):
    """Payload for POST /tasks."""
    description: str = Field(..., min_length=1)
    priority: str = Field(default="NORMAL")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    dependencies: list = Field(default_factory=list)
    agent_id: Optional[str] = Field(default=None, description="Target agent ID (optional)")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", summary="Health check")
async def health() -> Dict[str, Any]:
    """Return system health status."""
    return system.get_health()


@app.get("/status", summary="Full system status")
async def status() -> Dict[str, Any]:
    """Return full system status including all agents and metrics."""
    return system.get_system_status()


@app.post("/tasks", summary="Create and submit a task", status_code=202)
async def create_task(body: TaskRequest) -> Dict[str, Any]:
    """Validate and submit a new task to the agent system."""
    try:
        priority = TaskPriority[body.priority.upper()]
    except KeyError:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid priority '{body.priority}'. "
                   f"Valid values: {[p.name for p in TaskPriority]}",
        )

    try:
        task = system.create_task(
            description=body.description,
            parameters=body.parameters,
            priority=priority,
            dependencies=body.dependencies,
        )
    except (ValueError, Exception) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    submitted = system.submit_task(task, agent_id=body.agent_id)
    return {
        "task_id": task.id,
        "status": task.status,
        "submitted": submitted,
    }
