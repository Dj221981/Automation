import argparse

import pytest

from deep_thinking import (
    DeepThinker,
    MAX_DEPTH,
    MAX_TOPIC_LENGTH,
    depth_type,
    parse_args,
    sanitize_topic,
)


def test_think_generates_expected_line_count():
    thinker = DeepThinker(topic="automation", depth=3, seed=11)
    thoughts = thinker.think()
    assert len(thoughts) == 6
    assert thoughts[0].startswith("Step 1:")
    assert thoughts[1].startswith("         Insight:")


def test_seeded_output_is_reproducible():
    first = DeepThinker(topic="automation", depth=4, seed=123).think()
    second = DeepThinker(topic="automation", depth=4, seed=123).think()
    assert first == second


def test_summary_contains_depth_and_topic():
    thinker = DeepThinker(topic="ops", depth=2, seed=1)
    summary = thinker.summarize(thinker.think())
    assert "After 2 deep-thinking steps on 'ops'" in summary


def test_depth_type_rejects_out_of_range_values():
    with pytest.raises(argparse.ArgumentTypeError):
        depth_type("0")
    with pytest.raises(argparse.ArgumentTypeError):
        depth_type(str(MAX_DEPTH + 1))


def test_sanitize_topic_handles_empty_and_long_values():
    assert sanitize_topic("   ") == "general intelligence"
    long_topic = "x" * (MAX_TOPIC_LENGTH + 10)
    assert len(sanitize_topic(long_topic)) == MAX_TOPIC_LENGTH


def test_parse_args_default_and_flags():
    args = parse_args(["--depth", "2", "--verbose"])
    assert args.depth == 2
    assert args.verbose is True
    assert args.quiet is False
    assert args.topic == "deep ai thinking"
