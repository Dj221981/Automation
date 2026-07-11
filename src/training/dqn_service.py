"""Production DQN training wrapper/service."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.models.neural_network import AgentLearningModel, ExperienceReplay

logger = logging.getLogger(__name__)


@dataclass
class DQNTrainingConfig:
    """Configuration for production DQN training workflows."""

    state_size: int
    action_size: int
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    hidden_layers: List[int] = field(default_factory=lambda: [128, 64])
    dropout_rate: float = 0.0
    use_batch_norm: bool = False
    gradient_clip_norm: float = 10.0
    replay_buffer_size: int = 100000
    batch_size: int = 32
    warmup_steps: int = 1000
    target_update_interval: int = 100
    checkpoint_dir: str = "storage/checkpoints"
    history_path: str = "storage/training/dqn_history.json"
    metadata_path: str = "storage/training/dqn_checkpoint_metadata.json"
    seed: Optional[int] = None
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.state_size <= 0:
            raise ValueError("state_size must be greater than 0")
        if self.action_size <= 0:
            raise ValueError("action_size must be greater than 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        if self.replay_buffer_size <= 0:
            raise ValueError("replay_buffer_size must be greater than 0")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative")
        if self.target_update_interval <= 0:
            raise ValueError("target_update_interval must be greater than 0")


@dataclass
class TrainingCheckpointMetadata:
    """Serialized metadata associated with a saved training checkpoint."""

    episode: int
    timestamp_utc: str
    checkpoint_path: str
    mean_reward: float
    latest_loss: Optional[float]
    epsilon: float
    replay_buffer_size: int
    total_training_steps: int
    config: Dict[str, Any]


class DQNTrainingService:
    """Service wrapper that coordinates replay, training, checkpoints, and history."""

    def __init__(self, config: DQNTrainingConfig):
        self.config = config
        self.model = AgentLearningModel(
            state_size=config.state_size,
            action_size=config.action_size,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            epsilon=config.epsilon,
            epsilon_decay=config.epsilon_decay,
            epsilon_min=config.epsilon_min,
            model_type="dqn",
            device=config.device,
            hidden_layers=config.hidden_layers,
            dropout_rate=config.dropout_rate,
            use_batch_norm=config.use_batch_norm,
            gradient_clip_norm=config.gradient_clip_norm,
            seed=config.seed,
        )
        self.replay_buffer = ExperienceReplay(
            state_size=config.state_size,
            max_size=config.replay_buffer_size,
            seed=config.seed,
        )
        self.training_steps = 0
        self.training_history: Dict[str, List[float]] = {
            "step": [],
            "loss": [],
            "epsilon": [],
            "replay_buffer_size": [],
        }

        self._ensure_parent_directory(config.checkpoint_dir, treat_as_directory=True)
        self._ensure_parent_directory(config.history_path)
        self._ensure_parent_directory(config.metadata_path)

    @staticmethod
    def _ensure_parent_directory(path_value: str, treat_as_directory: bool = False) -> None:
        path = Path(path_value)
        directory = path if treat_as_directory else path.parent
        if str(directory):
            directory.mkdir(parents=True, exist_ok=True)

    def record_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay_buffer.add(state, action, reward, next_state, done)

    def train_on_replay_batch(self) -> Optional[float]:
        if len(self.replay_buffer) < max(self.config.warmup_steps, self.config.batch_size):
            logger.debug(
                "Skipping train step: replay buffer size %s below warmup threshold %s",
                len(self.replay_buffer),
                max(self.config.warmup_steps, self.config.batch_size),
            )
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)
        loss = self.model.train_step(states, actions, rewards, next_states, dones)
        self.training_steps += 1
        self.model.decay_epsilon()

        if self.training_steps % self.config.target_update_interval == 0:
            self.model.update_target_network()

        self.training_history["step"].append(self.training_steps)
        self.training_history["loss"].append(float(loss))
        self.training_history["epsilon"].append(float(self.model.epsilon))
        self.training_history["replay_buffer_size"].append(float(len(self.replay_buffer)))

        return float(loss)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        return self.model.select_action(state, training=training)

    def save_checkpoint(
        self,
        episode: int,
        mean_reward: float,
        latest_loss: Optional[float],
    ) -> TrainingCheckpointMetadata:
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f"dqn_episode_{episode}.weights.h5")
        self.model.save_model(checkpoint_path)

        metadata = TrainingCheckpointMetadata(
            episode=episode,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            checkpoint_path=checkpoint_path,
            mean_reward=float(mean_reward),
            latest_loss=None if latest_loss is None else float(latest_loss),
            epsilon=float(self.model.epsilon),
            replay_buffer_size=len(self.replay_buffer),
            total_training_steps=self.training_steps,
            config=self.model.get_config(),
        )

        Path(self.config.metadata_path).write_text(
            json.dumps(asdict(metadata), indent=2),
            encoding="utf-8",
        )
        return metadata

    def save_history(self) -> None:
        Path(self.config.history_path).write_text(
            json.dumps(self.training_history, indent=2),
            encoding="utf-8",
        )

    def load_checkpoint(self, checkpoint_path: str) -> None:
        self.model.load_model(checkpoint_path)

    def get_metrics_snapshot(self) -> Dict[str, Any]:
        latest_loss = self.training_history["loss"][-1] if self.training_history["loss"] else None
        return {
            "training_steps": self.training_steps,
            "epsilon": self.model.epsilon,
            "replay_buffer_size": len(self.replay_buffer),
            "latest_loss": latest_loss,
        }
