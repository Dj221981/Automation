"""Entry point for running the Automation package as a module."""

from src.training.dqn_service import DQNTrainingConfig, DQNTrainingService


def main() -> None:
    """Initialize the training service using baseline production defaults."""
    DQNTrainingService(DQNTrainingConfig(state_size=64, action_size=8))


if __name__ == "__main__":
    main()
