"""Tests for the package module entry point."""

import importlib
import sys
import types


def test_main_initializes_training_service() -> None:
    """The module entrypoint should build training config and service with defaults."""
    captured: dict[str, object] = {}

    class FakeConfig:
        def __init__(self, state_size: int, action_size: int) -> None:
            captured["config"] = (state_size, action_size)

    class FakeService:
        def __init__(self, config: FakeConfig) -> None:
            captured["service_config"] = config

    fake_module = types.ModuleType("src.training.dqn_service")
    fake_module.DQNTrainingConfig = FakeConfig
    fake_module.DQNTrainingService = FakeService

    sys.modules["src.training.dqn_service"] = fake_module
    main_module = importlib.import_module("src.__main__")

    main_module.main()

    assert captured["config"] == (64, 8)
    assert isinstance(captured["service_config"], FakeConfig)
