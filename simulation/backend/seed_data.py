"""
Seed data: two sample scenarios loaded on first startup if the DB is empty.
"""

from __future__ import annotations

import json

SEED_SCENARIOS = [
    {
        "title": "Fire Safety in the Workplace",
        "description": (
            "A scenario designed to test employees' knowledge of fire safety "
            "procedures, evacuation routes, and the use of fire extinguishers."
        ),
        "difficulty": "easy",
        "tags": ["safety", "fire", "workplace"],
        "steps": [
            {
                "prompt": "You notice smoke coming from a storage room. What is the first action you should take?",
                "expected_outcome": (
                    "Alert others, activate the fire alarm, and call emergency services (911). "
                    "Do not enter the room or attempt to fight the fire alone."
                ),
                "keywords": ["alarm", "alert", "call", "evacuate", "emergency"],
                "weight": 1.0,
            },
            {
                "prompt": "You must evacuate the building. Describe the steps you would follow.",
                "expected_outcome": (
                    "Use the nearest marked exit, do not use elevators, assist others who need help, "
                    "proceed to the assembly point, and report to your supervisor."
                ),
                "keywords": ["exit", "staircase", "assembly", "elevator", "supervisor"],
                "weight": 1.0,
            },
            {
                "prompt": "A small wastebasket fire breaks out. When is it appropriate to use a fire extinguisher?",
                "expected_outcome": (
                    "Use an extinguisher only if: the fire is small and contained, you have a clear "
                    "escape route, you have been trained, and others have already been evacuated."
                ),
                "keywords": ["small", "contained", "trained", "escape", "evacuation"],
                "weight": 1.0,
            },
            {
                "prompt": "What does the acronym PASS stand for when using a fire extinguisher?",
                "expected_outcome": (
                    "Pull the pin, Aim the nozzle at the base of the fire, Squeeze the handle, "
                    "Sweep from side to side."
                ),
                "keywords": ["pull", "aim", "squeeze", "sweep"],
                "weight": 1.0,
            },
        ],
    },
    {
        "title": "Customer Conflict Resolution",
        "description": (
            "Practice handling difficult customer interactions with empathy and "
            "professionalism to reach a satisfactory resolution."
        ),
        "difficulty": "medium",
        "tags": ["customer service", "communication", "conflict"],
        "steps": [
            {
                "prompt": (
                    "A customer calls in angry because their order arrived damaged. "
                    "How do you open the conversation?"
                ),
                "expected_outcome": (
                    "Greet the customer calmly, apologise for the inconvenience, listen actively "
                    "without interrupting, and acknowledge their frustration."
                ),
                "keywords": ["apologise", "listen", "acknowledge", "empathy", "understand"],
                "weight": 1.0,
            },
            {
                "prompt": "The customer demands an immediate full refund but your policy allows only an exchange. What do you do?",
                "expected_outcome": (
                    "Explain the policy clearly but empathetically, offer alternatives such as an "
                    "exchange or store credit, and escalate to a manager if the customer is not satisfied."
                ),
                "keywords": ["policy", "exchange", "alternative", "escalate", "manager"],
                "weight": 1.0,
            },
            {
                "prompt": "The customer starts using abusive language. How do you handle this?",
                "expected_outcome": (
                    "Remain calm and professional. Politely but firmly state that abusive language is "
                    "not acceptable. Offer to continue the conversation when they are ready, or escalate "
                    "to a supervisor."
                ),
                "keywords": ["calm", "firm", "abusive", "supervisor", "professional"],
                "weight": 1.5,
            },
            {
                "prompt": "After resolving the issue, how do you close the interaction?",
                "expected_outcome": (
                    "Summarise the agreed resolution, confirm the next steps, thank the customer for "
                    "their patience, and document the case notes."
                ),
                "keywords": ["summarise", "confirm", "thank", "document", "follow-up"],
                "weight": 0.5,
            },
        ],
    },
]


def seed_database(db_session) -> None:
    """Insert seed scenarios into the database if no scenarios exist."""
    from models import Scenario

    if db_session.query(Scenario).count() > 0:
        return  # Already seeded

    for data in SEED_SCENARIOS:
        scenario = Scenario(
            title=data["title"],
            description=data["description"],
            difficulty=data["difficulty"],
            tags=json.dumps(data["tags"]),
            steps=json.dumps(data["steps"]),
        )
        db_session.add(scenario)

    db_session.commit()
