"""
FastAPI application – Simulation Training & Builder Platform.

Run with:
    cd simulation/backend
    uvicorn app:app --reload --port 8000
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from database import Base, engine, get_db
from models import Scenario, SimulationSession
from schemas import (
    ResponseSubmit,
    ScenarioCreate,
    ScenarioOut,
    ScenarioUpdate,
    SessionOut,
    SessionStart,
)
from scoring import score_session
from seed_data import seed_database

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app):
    """Create tables and seed demo data on startup."""
    Base.metadata.create_all(bind=engine)
    db = next(get_db())
    try:
        seed_database(db)
    finally:
        db.close()
    yield


app = FastAPI(
    title="Simulation Training Platform",
    description="Build and run simulation training scenarios.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Scenario endpoints
# ---------------------------------------------------------------------------

def _scenario_to_out(scenario: Scenario) -> dict:
    """Serialise an ORM Scenario to the response schema dict."""
    return {
        "id": scenario.id,
        "title": scenario.title,
        "description": scenario.description,
        "difficulty": scenario.difficulty,
        "tags": scenario.get_tags(),
        "steps": scenario.get_steps(),
        "created_at": scenario.created_at,
        "updated_at": scenario.updated_at,
    }


@app.get("/scenarios", response_model=List[ScenarioOut], tags=["scenarios"])
def list_scenarios(db: Session = Depends(get_db)) -> list:
    """Return all scenarios."""
    return [_scenario_to_out(s) for s in db.query(Scenario).all()]


@app.post(
    "/scenarios",
    response_model=ScenarioOut,
    status_code=status.HTTP_201_CREATED,
    tags=["scenarios"],
)
def create_scenario(payload: ScenarioCreate, db: Session = Depends(get_db)) -> dict:
    """Create a new scenario."""
    scenario = Scenario(
        title=payload.title,
        description=payload.description,
        difficulty=payload.difficulty,
    )
    scenario.set_tags(payload.tags)
    scenario.set_steps([s.model_dump() for s in payload.steps])
    db.add(scenario)
    db.commit()
    db.refresh(scenario)
    return _scenario_to_out(scenario)


@app.get("/scenarios/{scenario_id}", response_model=ScenarioOut, tags=["scenarios"])
def get_scenario(scenario_id: int, db: Session = Depends(get_db)) -> dict:
    """Return a single scenario."""
    scenario = db.get(Scenario, scenario_id)
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")
    return _scenario_to_out(scenario)


@app.put("/scenarios/{scenario_id}", response_model=ScenarioOut, tags=["scenarios"])
def update_scenario(
    scenario_id: int,
    payload: ScenarioUpdate,
    db: Session = Depends(get_db),
) -> dict:
    """Update a scenario (partial update supported)."""
    scenario = db.get(Scenario, scenario_id)
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")

    if payload.title is not None:
        scenario.title = payload.title
    if payload.description is not None:
        scenario.description = payload.description
    if payload.difficulty is not None:
        scenario.difficulty = payload.difficulty
    if payload.tags is not None:
        scenario.set_tags(payload.tags)
    if payload.steps is not None:
        scenario.set_steps([s.model_dump() for s in payload.steps])

    scenario.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(scenario)
    return _scenario_to_out(scenario)


@app.delete("/scenarios/{scenario_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["scenarios"])
def delete_scenario(scenario_id: int, db: Session = Depends(get_db)) -> None:
    """Delete a scenario and its associated sessions."""
    scenario = db.get(Scenario, scenario_id)
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")
    db.delete(scenario)
    db.commit()


# ---------------------------------------------------------------------------
# Session endpoints
# ---------------------------------------------------------------------------

def _session_to_out(session: SimulationSession) -> dict:
    return {
        "id": session.id,
        "scenario_id": session.scenario_id,
        "participant_name": session.participant_name,
        "started_at": session.started_at,
        "completed_at": session.completed_at,
        "responses": session.get_responses(),
        "score": session.score,
        "feedback": session.feedback,
        "status": session.status,
    }


@app.post(
    "/scenarios/{scenario_id}/sessions",
    response_model=SessionOut,
    status_code=status.HTTP_201_CREATED,
    tags=["sessions"],
)
def start_session(
    scenario_id: int,
    payload: SessionStart,
    db: Session = Depends(get_db),
) -> dict:
    """Start a new simulation session for a scenario."""
    scenario = db.get(Scenario, scenario_id)
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")

    session = SimulationSession(
        scenario_id=scenario_id,
        participant_name=payload.participant_name or "Anonymous",
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return _session_to_out(session)


@app.post(
    "/sessions/{session_id}/responses",
    response_model=SessionOut,
    tags=["sessions"],
)
def submit_response(
    session_id: int,
    payload: ResponseSubmit,
    db: Session = Depends(get_db),
) -> dict:
    """Submit a response for a single step in an active session."""
    session = db.get(SimulationSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.status != "active":
        raise HTTPException(status_code=400, detail="Session is not active")

    scenario = db.get(Scenario, session.scenario_id)
    steps = scenario.get_steps() if scenario else []

    if payload.step_index < 0 or (steps and payload.step_index >= len(steps)):
        raise HTTPException(status_code=400, detail="Invalid step_index")

    responses = session.get_responses()
    # Replace an existing response for the same step if re-submitted
    responses = [r for r in responses if r.get("step_index") != payload.step_index]
    responses.append({"step_index": payload.step_index, "response": payload.response})
    session.set_responses(responses)

    db.commit()
    db.refresh(session)
    return _session_to_out(session)


@app.post(
    "/sessions/{session_id}/complete",
    response_model=SessionOut,
    tags=["sessions"],
)
def complete_session(session_id: int, db: Session = Depends(get_db)) -> dict:
    """Complete a session: compute score and feedback."""
    session = db.get(SimulationSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.status != "active":
        raise HTTPException(status_code=400, detail="Session is already completed")

    scenario = db.get(Scenario, session.scenario_id)
    steps = scenario.get_steps() if scenario else []
    responses = session.get_responses()

    result = score_session(steps, responses)

    session.score = result["percentage"]
    session.feedback = result["feedback"]
    session.status = "completed"
    session.completed_at = datetime.now(timezone.utc)

    # Persist per-step scores back into responses
    step_scores = {ps["step_index"]: ps for ps in result["per_step"]}
    enriched = []
    for resp in responses:
        idx = resp.get("step_index", -1)
        entry = dict(resp)
        if idx in step_scores:
            entry["step_score"] = step_scores[idx]["score"]
            entry["max_score"] = step_scores[idx]["max_score"]
        enriched.append(entry)
    session.set_responses(enriched)

    db.commit()
    db.refresh(session)
    return _session_to_out(session)


@app.get("/sessions", response_model=List[SessionOut], tags=["sessions"])
def list_sessions(db: Session = Depends(get_db)) -> list:
    """List all simulation sessions."""
    return [_session_to_out(s) for s in db.query(SimulationSession).all()]


@app.get("/sessions/{session_id}", response_model=SessionOut, tags=["sessions"])
def get_session(session_id: int, db: Session = Depends(get_db)) -> dict:
    """Get a single simulation session."""
    session = db.get(SimulationSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return _session_to_out(session)


@app.get(
    "/scenarios/{scenario_id}/sessions",
    response_model=List[SessionOut],
    tags=["sessions"],
)
def list_scenario_sessions(scenario_id: int, db: Session = Depends(get_db)) -> list:
    """List all sessions for a specific scenario."""
    scenario = db.get(Scenario, scenario_id)
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")
    sessions = (
        db.query(SimulationSession)
        .filter(SimulationSession.scenario_id == scenario_id)
        .all()
    )
    return [_session_to_out(s) for s in sessions]
