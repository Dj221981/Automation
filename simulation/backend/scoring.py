"""
Scoring engine for simulation sessions.

Each step in a scenario has an ``expected_outcome`` string.
The engine compares the participant's free-text response against the
expected outcome using a simple keyword-overlap metric (no external ML
dependencies required).

The design is intentionally modular so that more sophisticated strategies
(e.g. semantic similarity, rubric-based scoring) can be plugged in later.
"""

from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    """Lower-case and split text into a set of non-empty word tokens."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_response(
    user_response: str,
    expected_outcome: str,
    *,
    keywords: list[str] | None = None,
) -> float:
    """
    Score a single step response in the range [0.0, 1.0].

    Strategy:
    1. If explicit ``keywords`` are provided for the step, score by how many
       keywords the response contains (keyword recall).
    2. Otherwise fall back to word-overlap (Jaccard) between the response and
       the expected outcome.

    Parameters
    ----------
    user_response:
        The free-text answer submitted by the participant.
    expected_outcome:
        The ideal answer text stored on the scenario step.
    keywords:
        Optional list of must-have keywords for this step.

    Returns
    -------
    float in [0.0, 1.0]
    """
    if not user_response.strip():
        return 0.0

    if keywords:
        kw_tokens = {kw.lower().strip() for kw in keywords if kw.strip()}
        if not kw_tokens:
            return 0.0
        matched = sum(1 for kw in kw_tokens if kw in user_response.lower())
        return round(matched / len(kw_tokens), 3)

    # Jaccard-based fallback
    response_tokens = _tokenize(user_response)
    expected_tokens = _tokenize(expected_outcome)

    if not expected_tokens:
        return 0.0

    intersection = response_tokens & expected_tokens
    union = response_tokens | expected_tokens

    return round(len(intersection) / len(union), 3) if union else 0.0


def score_session(
    steps: list[dict[str, Any]],
    responses: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compute per-step scores and an overall session score.

    Parameters
    ----------
    steps:
        List of step dicts from the scenario.  Each dict should have at least:
        ``{"prompt": str, "expected_outcome": str}``.
        Optional keys: ``"keywords": [str]``, ``"weight": float``.
    responses:
        List of response dicts submitted during the session.  Each dict should
        have at least: ``{"step_index": int, "response": str}``.

    Returns
    -------
    dict with keys:
        - ``per_step``: list of ``{step_index, score, max_score}``
        - ``total_score``: float (0–100)
        - ``max_possible``: float
        - ``percentage``: float (0–100)
        - ``feedback``: str
    """
    if not steps:
        return {
            "per_step": [],
            "total_score": 0.0,
            "max_possible": 0.0,
            "percentage": 0.0,
            "feedback": "No steps to evaluate.",
        }

    # Build a lookup: step_index -> response text
    response_map: dict[int, str] = {
        r.get("step_index", -1): r.get("response", "")
        for r in responses
    }

    per_step: list[dict[str, Any]] = []
    total_score = 0.0
    max_possible = 0.0

    for idx, step in enumerate(steps):
        weight = float(step.get("weight", 1.0))
        user_resp = response_map.get(idx, "")
        expected = step.get("expected_outcome", "")
        keywords = step.get("keywords") or []

        raw_score = score_response(user_resp, expected, keywords=keywords)
        step_score = round(raw_score * weight, 3)

        per_step.append(
            {
                "step_index": idx,
                "prompt": step.get("prompt", ""),
                "score": step_score,
                "max_score": weight,
                "raw_score": raw_score,
            }
        )
        total_score += step_score
        max_possible += weight

    percentage = round((total_score / max_possible) * 100, 1) if max_possible else 0.0
    feedback = _generate_feedback(per_step, percentage)

    return {
        "per_step": per_step,
        "total_score": round(total_score, 3),
        "max_possible": round(max_possible, 3),
        "percentage": percentage,
        "feedback": feedback,
    }


# ---------------------------------------------------------------------------
# Feedback generator
# ---------------------------------------------------------------------------

def _generate_feedback(
    per_step: list[dict[str, Any]],
    percentage: float,
) -> str:
    """Generate a short human-readable feedback summary."""
    if not per_step:
        return "No responses recorded."

    strengths = [
        p for p in per_step if p["raw_score"] >= 0.7
    ]
    improvements = [
        p for p in per_step if p["raw_score"] < 0.4
    ]

    lines: list[str] = []

    if percentage >= 80:
        lines.append("Excellent work! You demonstrated strong understanding of the material.")
    elif percentage >= 60:
        lines.append("Good effort! You covered most key concepts.")
    elif percentage >= 40:
        lines.append("Decent attempt. There is room for improvement.")
    else:
        lines.append("Keep practising — review the material and try again.")

    if strengths:
        step_nums = ", ".join(str(s["step_index"] + 1) for s in strengths)
        lines.append(f"Strengths: step(s) {step_nums} answered well.")

    if improvements:
        step_nums = ", ".join(str(s["step_index"] + 1) for s in improvements)
        lines.append(f"Focus on improving: step(s) {step_nums}.")

    return " ".join(lines)
