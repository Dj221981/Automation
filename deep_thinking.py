#!/usr/bin/env python3
"""Production-ready deep thinking CLI utility."""

from __future__ import annotations

import argparse
import logging
import random
import sys
from datetime import datetime, timezone
from typing import List, Optional, Sequence

DEFAULT_TOPIC = "deep ai thinking"
FALLBACK_TOPIC = "general intelligence"
MAX_TOPIC_LENGTH = 256
MAX_DEPTH = 100

logger = logging.getLogger(__name__)

__all__ = [
    "DeepThinker",
    "MAX_DEPTH",
    "MAX_TOPIC_LENGTH",
    "sanitize_topic",
    "depth_type",
    "configure_logging",
    "parse_args",
    "run",
    "main",
]


class DeepThinker:
    """Simulates layered AI-style thinking for a given topic."""

    def __init__(self, topic: str, depth: int = 5, seed: Optional[int] = None) -> None:
        self.topic = sanitize_topic(topic)
        self.depth = depth
        self._rng = random.Random(seed)

    def _question_templates(self) -> List[str]:
        return [
            "What is the core meaning of '{topic}'?",
            "Which assumptions am I making about '{topic}'?",
            "What are possible risks or limitations in '{topic}'?",
            "How can '{topic}' be improved step by step?",
            "What would an expert challenge about '{topic}'?",
            "How does '{topic}' connect to real-world impact?",
            "What data or evidence is missing for '{topic}'?",
            "What is a simpler way to explain '{topic}'?",
        ]

    def _insight_templates(self) -> List[str]:
        return [
            "Insight: clarity increases when goals are measurable.",
            "Insight: uncertainty can be reduced through iteration.",
            "Insight: strong results require both logic and feedback.",
            "Insight: every system has trade-offs that must be managed.",
            "Insight: better questions produce better decisions.",
            "Insight: progress is faster with small testable steps.",
        ]

    def think(self) -> List[str]:
        """Generate layered thinking steps and concise insights."""
        thoughts: List[str] = []
        q_templates = self._question_templates()
        i_templates = self._insight_templates()

        for step in range(1, self.depth + 1):
            question = self._rng.choice(q_templates).format(topic=self.topic)
            insight = self._rng.choice(i_templates)
            thoughts.append(f"Step {step}: {question}")
            thoughts.append(f"         {insight}")

        return thoughts

    def summarize(self, thoughts: Optional[List[str]] = None) -> str:
        """Create a compact summary after thinking."""
        insight_count = 0
        if thoughts:
            insight_count = sum(1 for line in thoughts if line.strip().startswith("Insight:"))

        return (
            f"Summary: After {self.depth} deep-thinking steps on '{self.topic}', "
            f"{insight_count or self.depth} insights were generated. "
            "The best path is to define clear goals, test assumptions, and iterate quickly."
        )


def depth_type(value: str) -> int:
    """Validate CLI depth input with explicit bounds."""
    try:
        depth = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("depth must be an integer") from exc

    if depth < 1:
        raise argparse.ArgumentTypeError("depth must be at least 1")
    if depth > MAX_DEPTH:
        raise argparse.ArgumentTypeError(f"depth must be <= {MAX_DEPTH}")
    return depth


def sanitize_topic(topic: str) -> str:
    """Normalize topic and enforce maximum allowed length."""
    normalized = topic.strip() if topic else ""
    if not normalized:
        return FALLBACK_TOPIC
    if len(normalized) > MAX_TOPIC_LENGTH:
        logger.warning(
            "Topic too long (%s chars). Truncating to %s.",
            len(normalized),
            MAX_TOPIC_LENGTH,
        )
        return normalized[:MAX_TOPIC_LENGTH]
    return normalized


def configure_logging(verbose: bool, quiet: bool) -> None:
    """Set logging level based on CLI verbosity flags."""
    if verbose and quiet:
        raise ValueError("--verbose and --quiet cannot be used together")

    level = logging.INFO
    if verbose:
        level = logging.DEBUG
    if quiet:
        level = logging.ERROR

    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(description="Deep AI thinking simulator")
    parser.add_argument(
        "topic",
        nargs="?",
        default=DEFAULT_TOPIC,
        help=f"Topic to think about (default: '{DEFAULT_TOPIC}')",
    )
    parser.add_argument(
        "--depth",
        type=depth_type,
        default=5,
        help=f"How many thinking layers to run (1-{MAX_DEPTH}, default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible output",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only show errors",
    )
    return parser.parse_args(argv)


def run(argv: Optional[Sequence[str]] = None) -> int:
    """Run the CLI and return a process exit code."""
    try:
        args = parse_args(argv)
        configure_logging(verbose=args.verbose, quiet=args.quiet)

        thinker = DeepThinker(topic=args.topic, depth=args.depth, seed=args.seed)
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

        print("ai deepthinking")
        print("-" * 40)
        print(f"Started at (UTC): {timestamp}")
        print(f"Topic: {thinker.topic}")
        print(f"Depth: {thinker.depth}")
        print("-" * 40)

        thoughts = thinker.think()
        for line in thoughts:
            print(line)

        print("-" * 40)
        print(thinker.summarize(thoughts))
        return 0
    except SystemExit:
        raise
    except Exception:
        logger.exception("Deep thinking run failed")
        return 1


def main() -> None:
    """CLI entrypoint with explicit process exit behavior."""
    sys.exit(run())


if __name__ == "__main__":
    main()
