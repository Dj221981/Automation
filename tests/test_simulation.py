"""
Tests for the simulation backend: scoring engine and session/scenario flow.

Run with:
    cd /home/runner/work/Automation/Automation
    python -m pytest tests/test_simulation.py -v
"""

import sys
import os
import json
import pytest

# Make simulation/backend importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simulation', 'backend'))

from scoring import score_response, score_session, _generate_feedback


# ===========================================================================
# Tests: score_response
# ===========================================================================

class TestScoreResponse:
    """Unit tests for per-step response scoring."""

    def test_perfect_keyword_match(self):
        score = score_response(
            "I pulled the pin, aimed at the base, squeezed the handle and swept side to side sweep",
            "Pull pin, aim, squeeze, sweep",
            keywords=["pull", "aim", "squeeze", "sweep"],
        )
        assert score == 1.0

    def test_partial_keyword_match(self):
        score = score_response(
            "I pulled the pin and aimed",
            "Pull pin, aim, squeeze, sweep",
            keywords=["pull", "aim", "squeeze", "sweep"],
        )
        assert 0.0 < score < 1.0

    def test_no_keyword_match(self):
        score = score_response(
            "I just called someone",
            "Pull pin, aim, squeeze, sweep",
            keywords=["pull", "aim", "squeeze", "sweep"],
        )
        assert score == 0.0

    def test_empty_response_returns_zero(self):
        assert score_response("", "some expected outcome") == 0.0
        assert score_response("   ", "some expected outcome", keywords=["a", "b"]) == 0.0

    def test_jaccard_fallback_identical(self):
        text = "evacuate the building using the nearest exit"
        score = score_response(text, text)
        assert score == 1.0

    def test_jaccard_fallback_partial_overlap(self):
        score = score_response(
            "use the nearest exit",
            "evacuate the building using the nearest marked emergency exit immediately",
        )
        assert 0.0 < score < 1.0

    def test_jaccard_fallback_no_overlap(self):
        score = score_response("completely unrelated answer", "fire alarm evacuation exit")
        assert score == 0.0

    def test_score_range(self):
        for response in ["hello world", "fire alarm", "evacuate the building via the nearest exit please"]:
            score = score_response(response, "fire alarm evacuation exit")
            assert 0.0 <= score <= 1.0

    def test_empty_keywords_list_falls_back_to_jaccard(self):
        score = score_response(
            "I pulled the pin and aimed at the fire",
            "pull the pin aim at the fire",
            keywords=[],
        )
        assert score > 0.0


# ===========================================================================
# Tests: score_session
# ===========================================================================

class TestScoreSession:
    """Unit tests for full-session scoring."""

    @pytest.fixture
    def sample_steps(self):
        return [
            {
                "prompt": "What do you do first?",
                "expected_outcome": "activate the fire alarm immediately",
                "keywords": ["alarm", "activate"],
                "weight": 1.0,
            },
            {
                "prompt": "How do you evacuate?",
                "expected_outcome": "use the nearest exit and go to the assembly point",
                "keywords": ["exit", "assembly"],
                "weight": 1.0,
            },
            {
                "prompt": "What does PASS stand for?",
                "expected_outcome": "Pull Aim Squeeze Sweep",
                "keywords": ["pull", "aim", "squeeze", "sweep"],
                "weight": 2.0,
            },
        ]

    @pytest.fixture
    def perfect_responses(self):
        return [
            {"step_index": 0, "response": "I activate the fire alarm"},
            {"step_index": 1, "response": "I use the nearest exit and go to the assembly point"},
            {"step_index": 2, "response": "Pull the pin, aim at the base, squeeze handle, sweep side to side"},
        ]

    def test_empty_steps_returns_zero(self):
        result = score_session([], [{"step_index": 0, "response": "anything"}])
        assert result["total_score"] == 0.0
        assert result["percentage"] == 0.0

    def test_perfect_score(self, sample_steps, perfect_responses):
        result = score_session(sample_steps, perfect_responses)
        assert result["percentage"] == 100.0
        assert result["total_score"] == result["max_possible"]

    def test_zero_score_empty_responses(self, sample_steps):
        result = score_session(sample_steps, [])
        assert result["total_score"] == 0.0
        assert result["percentage"] == 0.0

    def test_per_step_count_matches_steps(self, sample_steps, perfect_responses):
        result = score_session(sample_steps, perfect_responses)
        assert len(result["per_step"]) == len(sample_steps)

    def test_max_possible_reflects_weights(self, sample_steps, perfect_responses):
        result = score_session(sample_steps, perfect_responses)
        expected_max = sum(s["weight"] for s in sample_steps)
        assert result["max_possible"] == pytest.approx(expected_max)

    def test_partial_responses_partial_score(self, sample_steps):
        responses = [
            {"step_index": 0, "response": "I activate the fire alarm"},
            # steps 1 and 2 unanswered
        ]
        result = score_session(sample_steps, responses)
        assert 0.0 < result["percentage"] < 100.0

    def test_feedback_included(self, sample_steps, perfect_responses):
        result = score_session(sample_steps, perfect_responses)
        assert isinstance(result["feedback"], str)
        assert len(result["feedback"]) > 0

    def test_weighted_step_contributes_more(self, sample_steps):
        # Respond only to step 2 (weight=2), and steps 0,1 (weight=1 each) unanswered
        responses = [
            {"step_index": 2, "response": "pull aim squeeze sweep"},
        ]
        result = score_session(sample_steps, responses)
        # Step 2 alone should contribute more than each of step 0 or 1
        step2_score = result["per_step"][2]["score"]
        step0_score = result["per_step"][0]["score"]
        assert step2_score >= step0_score

    def test_score_percentage_range(self, sample_steps):
        import random
        for _ in range(10):
            responses = [
                {"step_index": i, "response": "random text " * random.randint(0, 5)}
                for i in range(len(sample_steps))
            ]
            result = score_session(sample_steps, responses)
            assert 0.0 <= result["percentage"] <= 100.0


# ===========================================================================
# Tests: feedback generator
# ===========================================================================

class TestFeedback:
    """Unit tests for feedback text generation."""

    def _make_per_step(self, raw_scores, weights=None):
        weights = weights or [1.0] * len(raw_scores)
        return [
            {
                "step_index": i,
                "prompt": f"Prompt {i}",
                "raw_score": rs,
                "score": rs * w,
                "max_score": w,
            }
            for i, (rs, w) in enumerate(zip(raw_scores, weights))
        ]

    def test_excellent_feedback(self):
        per_step = self._make_per_step([1.0, 0.9, 0.8])
        fb = _generate_feedback(per_step, 90.0)
        assert "Excellent" in fb

    def test_good_feedback(self):
        per_step = self._make_per_step([0.7, 0.6, 0.65])
        fb = _generate_feedback(per_step, 65.0)
        assert "Good" in fb

    def test_poor_feedback(self):
        per_step = self._make_per_step([0.2, 0.1, 0.0])
        fb = _generate_feedback(per_step, 10.0)
        assert "practising" in fb or "Keep" in fb

    def test_strengths_mentioned(self):
        per_step = self._make_per_step([1.0, 0.0, 0.0])
        fb = _generate_feedback(per_step, 33.3)
        assert "Strength" in fb

    def test_improvements_mentioned(self):
        per_step = self._make_per_step([0.0, 0.0, 1.0])
        fb = _generate_feedback(per_step, 33.3)
        assert "improv" in fb.lower()

    def test_no_steps_returns_message(self):
        fb = _generate_feedback([], 0.0)
        assert "No responses" in fb


# ===========================================================================
# FastAPI integration tests
# ===========================================================================

@pytest.fixture(scope="module")
def client():
    """Create a test client with a shared in-memory SQLite database."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from fastapi.testclient import TestClient

    # StaticPool reuses a single connection so all sessions share the same
    # in-memory SQLite database.
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    import database as db_module
    import app as app_module
    from database import Base

    # Patch module-level globals so app's startup uses test DB
    original_engine = db_module.engine
    original_session = db_module.SessionLocal
    db_module.engine = test_engine
    db_module.SessionLocal = TestingSessionLocal

    # Create all tables on the test engine before startup fires
    Base.metadata.create_all(bind=test_engine)

    # Override the get_db FastAPI dependency
    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app_module.app.dependency_overrides[db_module.get_db] = override_get_db

    with TestClient(app_module.app, raise_server_exceptions=True) as c:
        yield c

    # Restore originals
    db_module.engine = original_engine
    db_module.SessionLocal = original_session
    app_module.app.dependency_overrides.clear()


class TestScenarioEndpoints:
    """Integration tests for /scenarios CRUD."""

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_list_scenarios_initially_empty_or_seeded(self, client):
        r = client.get("/scenarios")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_create_scenario(self, client):
        payload = {
            "title": "Test Scenario",
            "description": "A test",
            "difficulty": "easy",
            "tags": ["test"],
            "steps": [
                {
                    "prompt": "What is 2+2?",
                    "expected_outcome": "4",
                    "keywords": ["4"],
                    "weight": 1.0,
                }
            ],
        }
        r = client.post("/scenarios", json=payload)
        assert r.status_code == 201
        data = r.json()
        assert data["title"] == "Test Scenario"
        assert len(data["steps"]) == 1

    def test_get_scenario(self, client):
        # Create first
        r = client.post("/scenarios", json={
            "title": "Get Test", "difficulty": "medium", "steps": []
        })
        sid = r.json()["id"]
        r2 = client.get(f"/scenarios/{sid}")
        assert r2.status_code == 200
        assert r2.json()["id"] == sid

    def test_update_scenario(self, client):
        r = client.post("/scenarios", json={"title": "Old Title", "steps": []})
        sid = r.json()["id"]
        r2 = client.put(f"/scenarios/{sid}", json={"title": "New Title"})
        assert r2.status_code == 200
        assert r2.json()["title"] == "New Title"

    def test_delete_scenario(self, client):
        r = client.post("/scenarios", json={"title": "To Delete", "steps": []})
        sid = r.json()["id"]
        r2 = client.delete(f"/scenarios/{sid}")
        assert r2.status_code == 204
        r3 = client.get(f"/scenarios/{sid}")
        assert r3.status_code == 404

    def test_get_nonexistent_scenario(self, client):
        r = client.get("/scenarios/99999")
        assert r.status_code == 404

    def test_invalid_difficulty_rejected(self, client):
        r = client.post("/scenarios", json={"title": "Bad", "difficulty": "extreme", "steps": []})
        assert r.status_code == 422


class TestSessionEndpoints:
    """Integration tests for session flow."""

    @pytest.fixture
    def scenario_with_steps(self, client):
        r = client.post("/scenarios", json={
            "title": "Session Test Scenario",
            "difficulty": "easy",
            "steps": [
                {"prompt": "Q1?", "expected_outcome": "answer one", "keywords": ["one"], "weight": 1.0},
                {"prompt": "Q2?", "expected_outcome": "answer two", "keywords": ["two"], "weight": 1.0},
            ],
        })
        return r.json()

    def test_start_session(self, client, scenario_with_steps):
        r = client.post(f"/scenarios/{scenario_with_steps['id']}/sessions",
                        json={"participant_name": "Alice"})
        assert r.status_code == 201
        data = r.json()
        assert data["status"] == "active"
        assert data["participant_name"] == "Alice"

    def test_submit_response(self, client, scenario_with_steps):
        r = client.post(f"/scenarios/{scenario_with_steps['id']}/sessions",
                        json={"participant_name": "Bob"})
        sess_id = r.json()["id"]
        r2 = client.post(f"/sessions/{sess_id}/responses",
                         json={"step_index": 0, "response": "answer one here"})
        assert r2.status_code == 200
        responses = r2.json()["responses"]
        assert any(resp["step_index"] == 0 for resp in responses)

    def test_complete_session(self, client, scenario_with_steps):
        r = client.post(f"/scenarios/{scenario_with_steps['id']}/sessions",
                        json={"participant_name": "Carol"})
        sess_id = r.json()["id"]

        # Submit answers
        client.post(f"/sessions/{sess_id}/responses",
                    json={"step_index": 0, "response": "one"})
        client.post(f"/sessions/{sess_id}/responses",
                    json={"step_index": 1, "response": "two"})

        # Complete
        r2 = client.post(f"/sessions/{sess_id}/complete")
        assert r2.status_code == 200
        data = r2.json()
        assert data["status"] == "completed"
        assert data["score"] is not None
        assert data["feedback"] != ""

    def test_cannot_submit_to_completed_session(self, client, scenario_with_steps):
        r = client.post(f"/scenarios/{scenario_with_steps['id']}/sessions",
                        json={"participant_name": "Dave"})
        sess_id = r.json()["id"]
        client.post(f"/sessions/{sess_id}/complete")

        r2 = client.post(f"/sessions/{sess_id}/responses",
                         json={"step_index": 0, "response": "late answer"})
        assert r2.status_code == 400

    def test_cannot_complete_session_twice(self, client, scenario_with_steps):
        r = client.post(f"/scenarios/{scenario_with_steps['id']}/sessions",
                        json={"participant_name": "Eve"})
        sess_id = r.json()["id"]
        client.post(f"/sessions/{sess_id}/complete")
        r2 = client.post(f"/sessions/{sess_id}/complete")
        assert r2.status_code == 400

    def test_list_sessions(self, client):
        r = client.get("/sessions")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_get_session(self, client, scenario_with_steps):
        r = client.post(f"/scenarios/{scenario_with_steps['id']}/sessions",
                        json={"participant_name": "Frank"})
        sess_id = r.json()["id"]
        r2 = client.get(f"/sessions/{sess_id}")
        assert r2.status_code == 200
        assert r2.json()["id"] == sess_id

    def test_list_scenario_sessions(self, client, scenario_with_steps):
        scen_id = scenario_with_steps["id"]
        client.post(f"/scenarios/{scen_id}/sessions", json={"participant_name": "Grace"})
        r = client.get(f"/scenarios/{scen_id}/sessions")
        assert r.status_code == 200
        assert len(r.json()) >= 1

    def test_invalid_step_index_rejected(self, client, scenario_with_steps):
        r = client.post(f"/scenarios/{scenario_with_steps['id']}/sessions", json={})
        sess_id = r.json()["id"]
        r2 = client.post(f"/sessions/{sess_id}/responses",
                         json={"step_index": 999, "response": "something"})
        assert r2.status_code == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
