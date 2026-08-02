"""
SQLAlchemy ORM models for scenarios and simulation sessions.
"""

import json
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from database import Base


class Scenario(Base):
    """A training scenario containing steps and expected outcomes."""

    __tablename__ = "scenarios"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), nullable=False)
    description = Column(Text, default="")
    difficulty = Column(String(20), default="medium")  # easy | medium | hard
    tags = Column(Text, default="[]")        # JSON list of strings
    steps = Column(Text, default="[]")       # JSON list of step dicts
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    sessions = relationship("SimulationSession", back_populates="scenario")

    # ------------------------------------------------------------------
    # JSON helpers
    # ------------------------------------------------------------------
    def get_tags(self) -> list:
        return json.loads(self.tags or "[]")

    def set_tags(self, value: list) -> None:
        self.tags = json.dumps(value)

    def get_steps(self) -> list:
        return json.loads(self.steps or "[]")

    def set_steps(self, value: list) -> None:
        self.steps = json.dumps(value)


class SimulationSession(Base):
    """
    A participant's run-through of a scenario.

    Tracks per-step responses, scoring, and overall feedback.
    """

    __tablename__ = "simulation_sessions"

    id = Column(Integer, primary_key=True, index=True)
    scenario_id = Column(Integer, ForeignKey("scenarios.id"), nullable=False)
    participant_name = Column(String(100), default="Anonymous")
    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)
    responses = Column(Text, default="[]")   # JSON list of response dicts
    score = Column(Float, nullable=True)
    feedback = Column(Text, default="")
    status = Column(String(20), default="active")  # active | completed

    scenario = relationship("Scenario", back_populates="sessions")

    # ------------------------------------------------------------------
    # JSON helpers
    # ------------------------------------------------------------------
    def get_responses(self) -> list:
        return json.loads(self.responses or "[]")

    def set_responses(self, value: list) -> None:
        self.responses = json.dumps(value)
