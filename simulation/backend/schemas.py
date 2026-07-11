"""
Pydantic schemas for request/response validation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Step schemas
# ---------------------------------------------------------------------------

class StepSchema(BaseModel):
    prompt: str = Field(..., min_length=1, description="The question or prompt for this step.")
    expected_outcome: str = Field(..., min_length=1, description="Ideal answer / expected outcome.")
    keywords: list[str] = Field(default_factory=list, description="Key terms that should appear in a correct response.")
    weight: float = Field(default=1.0, ge=0.1, description="Relative weight of this step in the total score.")


# ---------------------------------------------------------------------------
# Scenario schemas
# ---------------------------------------------------------------------------

class ScenarioCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(default="")
    difficulty: str = Field(default="medium", pattern="^(easy|medium|hard)$")
    tags: list[str] = Field(default_factory=list)
    steps: list[StepSchema] = Field(default_factory=list)


class ScenarioUpdate(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=200)
    description: Optional[str] = None
    difficulty: Optional[str] = Field(default=None, pattern="^(easy|medium|hard)$")
    tags: Optional[list[str]] = None
    steps: Optional[list[StepSchema]] = None


class ScenarioOut(BaseModel):
    id: int
    title: str
    description: str
    difficulty: str
    tags: list[str]
    steps: list[dict[str, Any]]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Session schemas
# ---------------------------------------------------------------------------

class SessionStart(BaseModel):
    participant_name: Optional[str] = Field(default="Anonymous", max_length=100)


class ResponseSubmit(BaseModel):
    step_index: int = Field(..., ge=0, description="Zero-based index of the step being answered.")
    response: str = Field(..., min_length=1, description="Participant's free-text response.")


class SessionOut(BaseModel):
    id: int
    scenario_id: int
    participant_name: str
    started_at: datetime
    completed_at: Optional[datetime]
    responses: list[dict[str, Any]]
    score: Optional[float]
    feedback: str
    status: str

    model_config = {"from_attributes": True}
