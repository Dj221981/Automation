"""
Production-ready Deep Q-Network (DQN) model for agent learning.

This module provides a hardened DQN implementation built with TensorFlow/Keras.
It includes:
- explicit model initialization/build steps for subclassed Keras models
- strict configuration and batch validation
- stable target network synchronization
- gradient clipping and finite-loss checks
- resource and memory safeguards
- structured observability helpers
- thread-safe persistence and replay buffer behavior
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import io
import json
import logging
import os
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam

logger = logging.getLogger(__name__)

_MAX_ARRAY_BYTES = 1 << 30
_LARGE_ALLOCATION_BYTES = 64 << 20
_MEMORY_WARNING_BYTES = 768 << 20
_MAX_MODEL_FILE_BYTES = 256 << 20
_MAX_STATE_SIZE = 1_000_000
_MAX_ACTION_SIZE = 100_000
_MAX_BATCH_SIZE = 100_000
_LOSS_EXPLOSION_THRESHOLD = 1e6
_GRADIENT_WARNING_THRESHOLD = 100.0
_WEIGHT_MAGNITUDE_THRESHOLD = 1e4
_Q_VALUE_MAGNITUDE_THRESHOLD = 1e6
_METADATA_FORMAT_VERSION = 2
_RECOVERY_METADATA_FORMATS = {1, 2}
_DANGEROUS_PATH_TOKENS = ("..", "\x00", "|", ";", "&", "$(`", "$(", "`")


@dataclass(frozen=True)
class MemoryStats:
    """Simple memory statistics snapshot for monitored arrays."""

    bytes_used: int
    threshold_bytes: int
    exceeded: bool


class FileLock(contextlib.AbstractContextManager["FileLock"]):
    """Best-effort filesystem lock based on sidecar lock files."""

    def __init__(self, target_path: str, timeout_seconds: float = 5.0, poll_interval: float = 0.05):
        self.target_path = target_path
        self.timeout_seconds = timeout_seconds
        self.poll_interval = poll_interval
        self.lock_path = f"{target_path}.lock"
        self._fd: Optional[int] = None

    def __enter__(self) -> "FileLock":
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                self._fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.write(self._fd, str(os.getpid()).encode("utf-8"))
                return self
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out acquiring model file lock for {self.target_path}")
                time.sleep(self.poll_interval)

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        with contextlib.suppress(FileNotFoundError):
            os.remove(self.lock_path)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_error_message(message: str) -> str:
    sanitized = message.replace("\n", " ").replace("\r", " ")
    return sanitized[:500]


def _structured_log(level: int, event: str, **context: Any) -> None:
    payload = {
        "timestamp": _utc_timestamp(),
        "event": event,
        **context,
    }
    logger.log(level, json.dumps(payload, sort_keys=True, default=str))


def _gc_hint_if_large(allocation_bytes: int) -> None:
    if allocation_bytes >= _LARGE_ALLOCATION_BYTES:
        gc.collect()


def _array_bytes(array: np.ndarray) -> int:
    return int(np.asarray(array).nbytes)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DQNNetwork(Model):
    """Deep Q-Network (DQN) for agent decision-making and learning."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_layers: Optional[List[int]] = None,
        activation: str = "relu",
        dropout_rate: float = 0.0,
        use_batch_norm: bool = False,
        name: str = "dqn_network",
    ) -> None:
        super().__init__(name=name)

        if state_size <= 0:
            raise ValueError("state_size must be a positive integer")
        if action_size <= 0:
            raise ValueError("action_size must be a positive integer")
        if hidden_layers is None:
            hidden_layers = [128, 64]
        if not hidden_layers:
            raise ValueError("hidden_layers must contain at least one layer size")
        if any(units <= 0 for units in hidden_layers):
            raise ValueError("all hidden layer sizes must be positive")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in the range [0.0, 1.0)")

        self.state_size = state_size
        self.action_size = action_size
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.hidden_layers = list(hidden_layers)

        self.hidden_stack: List[layers.Layer] = []
        for units in hidden_layers:
            self.hidden_stack.append(layers.Dense(units, activation=activation))
            if use_batch_norm:
                self.hidden_stack.append(layers.BatchNormalization())
            if dropout_rate > 0.0:
                self.hidden_stack.append(layers.Dropout(dropout_rate))
        self.dense_layers = self.hidden_stack
        self.output_layer = layers.Dense(action_size, activation=None)

        logger.info(
            "DQNNetwork initialized: state_size=%s action_size=%s hidden_layers=%s",
            state_size,
            action_size,
            hidden_layers,
        )

    def call(self, states: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Run a forward pass and return Q-values."""
        x = states
        for layer in self.hidden_stack:
            if isinstance(layer, (layers.Dropout, layers.BatchNormalization)):
                x = layer(x, training=training)
            else:
                x = layer(x)
        outputs = self.output_layer(x)
        if not tf.reduce_all(tf.math.is_finite(outputs)):
            raise ValueError("network produced non-finite Q-values")
        return outputs


class AgentLearningModel:
    """Production-hardened DQN learning model with safe operational defaults."""

    SUPPORTED_MODEL_TYPES = {"dqn"}
    SUPPORTED_DEVICES = {"cpu", "gpu"}
    CONFIG_PRESETS: Mapping[str, Mapping[str, float]] = {
        "conservative": {"learning_rate": 0.0005, "epsilon_decay": 0.997, "gradient_clip_norm": 5.0},
        "balanced": {"learning_rate": 0.001, "epsilon_decay": 0.995, "gradient_clip_norm": 10.0},
        "aggressive": {"learning_rate": 0.002, "epsilon_decay": 0.99, "gradient_clip_norm": 20.0},
    }
    IMMUTABLE_CONFIG_FIELDS = {
        "state_size",
        "action_size",
        "model_type",
        "device",
        "hidden_layers",
        "dropout_rate",
        "use_batch_norm",
        "gradient_clip_norm",
        "target_update_interval",
        "checkpoint_interval",
        "checkpoint_dir",
        "stable_training",
    }

    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        model_type: str = "dqn",
        device: str = "cpu",
        hidden_layers: Optional[List[int]] = None,
        dropout_rate: float = 0.0,
        use_batch_norm: bool = False,
        gradient_clip_norm: float = 10.0,
        seed: Optional[int] = None,
        target_update_interval: int = 1000,
        checkpoint_interval: int = 0,
        checkpoint_dir: Optional[str] = None,
        stable_training: bool = False,
        debug_mode: bool = False,
    ) -> None:
        object.__setattr__(self, "_config_locked", False)
        self._validate_configuration(
            state_size=state_size,
            action_size=action_size,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            model_type=model_type,
            device=device,
            gradient_clip_norm=gradient_clip_norm,
            target_update_interval=target_update_interval,
            checkpoint_interval=checkpoint_interval,
        )

        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.model_type = model_type
        self.device = device
        self.hidden_layers = hidden_layers or [128, 64]
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.gradient_clip_norm = gradient_clip_norm
        self.seed = seed
        self.target_update_interval = target_update_interval
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = checkpoint_dir
        self.stable_training = stable_training
        self.debug_mode = debug_mode
        self.train_steps = 0
        self.training_paused = False
        self.last_loss: Optional[float] = None
        self.loss_ema: Optional[float] = None
        self.loss_ema_momentum = 0.95
        self.gradient_warning_threshold = _GRADIENT_WARNING_THRESHOLD
        self.loss_explosion_threshold = _LOSS_EXPLOSION_THRESHOLD
        self.weight_magnitude_threshold = _WEIGHT_MAGNITUDE_THRESHOLD
        self.max_array_bytes = _MAX_ARRAY_BYTES
        self.memory_warning_bytes = _MEMORY_WARNING_BYTES
        self.max_model_file_bytes = _MAX_MODEL_FILE_BYTES
        self.last_checkpoint_path: Optional[str] = None
        self.last_checkpoint_checksum: Optional[str] = None
        self.last_correlation_id: Optional[str] = None
        self.metrics: Dict[str, Any] = {
            "last_gradient_max": 0.0,
            "last_weight_max": 0.0,
            "last_q_value_max": 0.0,
            "last_reward_mean": 0.0,
            "last_reward_std": 0.0,
            "last_memory_bytes": 0,
            "instability_events": 0,
            "training_retries": 0,
        }
        self.health_status: Dict[str, Any] = {
            "healthy": True,
            "status": "initialized",
            "last_error": None,
            "last_update": _utc_timestamp(),
            "circuit_open": False,
        }
        self.reward_statistics: Dict[str, float] = {"count": 0.0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        self.action_frequencies: Dict[int, int] = {}
        self._lock = threading.RLock()
        self._active_operations: set[str] = set()
        self._circuit_failures = 0
        self._circuit_open_until = 0.0
        self._circuit_threshold = 3
        self._circuit_cooldown_seconds = 5.0

        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)

        self.device_name = self._resolve_device_name(device)

        self.network = DQNNetwork(
            state_size=state_size,
            action_size=action_size,
            hidden_layers=self.hidden_layers,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            name="online_dqn_network",
        )
        self.target_network = DQNNetwork(
            state_size=state_size,
            action_size=action_size,
            hidden_layers=self.hidden_layers,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            name="target_dqn_network",
        )

        self._build_networks()
        self.update_target_network()

        self.optimizer = Adam(learning_rate=learning_rate, clipnorm=gradient_clip_norm)
        self.loss_fn = Huber()
        self.train_loss = keras.metrics.Mean(name="train_loss")
        self._base_learning_rate = float(learning_rate)
        self._freeze_config()

        _structured_log(
            logging.INFO,
            "agent_learning_model_initialized",
            model_type=model_type,
            learning_rate=learning_rate,
            device=self.device_name,
            checkpoint_interval=checkpoint_interval,
            stable_training=stable_training,
        )

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_config_locked", False) and name in self.IMMUTABLE_CONFIG_FIELDS:
            current_value = getattr(self, name, None)
            if current_value != value:
                raise AttributeError(f"{name} is immutable after initialization")
        object.__setattr__(self, name, value)

    @classmethod
    def create_with_preset(
        cls,
        preset: str,
        *,
        state_size: int,
        action_size: int,
        **overrides: Any,
    ) -> "AgentLearningModel":
        """Create a model using a validated preset with optional overrides."""
        if preset not in cls.CONFIG_PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Supported presets: {sorted(cls.CONFIG_PRESETS)}")
        config = dict(cls.CONFIG_PRESETS[preset])
        config.update(overrides)
        return cls(state_size=state_size, action_size=action_size, **config)

    @classmethod
    def _validate_configuration(
        cls,
        *,
        state_size: int,
        action_size: int,
        learning_rate: float,
        gamma: float,
        epsilon: float,
        epsilon_decay: float,
        epsilon_min: float,
        model_type: str,
        device: str,
        gradient_clip_norm: float,
        target_update_interval: int,
        checkpoint_interval: int,
    ) -> None:
        if state_size <= 0:
            raise ValueError("state_size must be a positive integer")
        if state_size > _MAX_STATE_SIZE:
            raise ValueError(f"state_size exceeds safe limit of {_MAX_STATE_SIZE}")
        if action_size <= 0:
            raise ValueError("action_size must be a positive integer")
        if action_size > _MAX_ACTION_SIZE:
            raise ValueError(f"action_size exceeds safe limit of {_MAX_ACTION_SIZE}")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0")
        if learning_rate > 1.0:
            raise ValueError("learning_rate must be less than or equal to 1.0")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in the range [0.0, 1.0]")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in the range [0.0, 1.0]")
        if not 0.0 < epsilon_decay <= 1.0:
            raise ValueError("epsilon_decay must be in the range (0.0, 1.0]")
        if not 0.0 <= epsilon_min <= 1.0:
            raise ValueError("epsilon_min must be in the range [0.0, 1.0]")
        if epsilon_min > epsilon:
            raise ValueError("epsilon_min cannot be greater than epsilon")
        if model_type not in cls.SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. Supported values: {sorted(cls.SUPPORTED_MODEL_TYPES)}"
            )
        if device not in cls.SUPPORTED_DEVICES:
            raise ValueError(
                f"Unsupported device '{device}'. Supported values: {sorted(cls.SUPPORTED_DEVICES)}"
            )
        if gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be greater than 0")
        if gradient_clip_norm > 1_000:
            raise ValueError("gradient_clip_norm exceeds safe upper bound")
        if target_update_interval <= 0:
            raise ValueError("target_update_interval must be greater than 0")
        if checkpoint_interval < 0:
            raise ValueError("checkpoint_interval cannot be negative")
        if checkpoint_interval and checkpoint_interval < 10:
            raise ValueError("checkpoint_interval must be 0 or at least 10 to avoid excessive I/O")

    def _freeze_config(self) -> None:
        object.__setattr__(self, "_config_locked", True)

    def _resolve_device_name(self, device: str) -> str:
        if device == "gpu":
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                logger.info("Using GPU for training")
                return "/GPU:0"
            logger.warning("GPU requested but no GPU was detected. Falling back to CPU.")
        logger.info("Using CPU for training")
        return "/CPU:0"

    def _build_networks(self) -> None:
        dummy_input = tf.zeros((1, self.state_size), dtype=tf.float32)
        self.network(dummy_input, training=False)
        self.target_network(dummy_input, training=False)

    @contextlib.contextmanager
    def _operation_guard(self, operation: str) -> Iterator[str]:
        correlation_id = uuid.uuid4().hex
        with self._lock:
            if operation in self._active_operations:
                raise RuntimeError(f"Operation '{operation}' is already in progress")
            self._active_operations.add(operation)
            self.last_correlation_id = correlation_id
        try:
            yield correlation_id
        finally:
            with self._lock:
                self._active_operations.discard(operation)

    @contextlib.contextmanager
    def _device_context(self) -> Iterator[None]:
        with tf.device(self.device_name):
            try:
                yield
            finally:
                if self.device_name.startswith("/GPU"):
                    with contextlib.suppress(Exception):
                        tf.experimental.async_clear_error()

    def _assert_circuit_closed(self) -> None:
        if time.monotonic() < self._circuit_open_until:
            self.health_status.update(
                {
                    "healthy": False,
                    "status": "circuit_open",
                    "last_update": _utc_timestamp(),
                    "circuit_open": True,
                }
            )
            raise RuntimeError("Training circuit is open due to recent TensorFlow failures. Retry after cooldown.")
        self.health_status["circuit_open"] = False

    def _record_failure(self, error: BaseException, correlation_id: str, operation: str) -> None:
        if isinstance(error, (tf.errors.ResourceExhaustedError, tf.errors.InternalError)):
            self._circuit_failures += 1
            if self._circuit_failures >= self._circuit_threshold:
                self._circuit_open_until = time.monotonic() + self._circuit_cooldown_seconds
        sanitized_error = _sanitize_error_message(str(error))
        self.health_status.update(
            {
                "healthy": False,
                "status": f"{operation}_failed",
                "last_error": sanitized_error,
                "last_update": _utc_timestamp(),
                "circuit_open": time.monotonic() < self._circuit_open_until,
            }
        )
        _structured_log(
            logging.ERROR,
            "agent_learning_model_failure",
            correlation_id=correlation_id,
            operation=operation,
            error=sanitized_error,
            traceback=traceback.format_exc(limit=5),
            retries=self.metrics["training_retries"],
        )

    def _record_success(self, operation: str, correlation_id: str) -> None:
        self._circuit_failures = 0
        self.health_status.update(
            {
                "healthy": True,
                "status": operation,
                "last_error": None,
                "last_update": _utc_timestamp(),
                "circuit_open": False,
            }
        )
        if self.debug_mode:
            _structured_log(logging.INFO, f"{operation}_completed", correlation_id=correlation_id)

    def _validate_path(self, filepath: str, *, for_write: bool) -> str:
        if not filepath or not isinstance(filepath, str):
            raise ValueError("filepath must be a non-empty string")
        normalized = os.path.normpath(filepath)
        if any(token in filepath for token in _DANGEROUS_PATH_TOKENS):
            raise ValueError("filepath contains dangerous path characters")
        if normalized.startswith(".."):
            raise ValueError("path traversal is not allowed")
        abs_path = os.path.abspath(normalized)
        directory = os.path.dirname(abs_path) or os.getcwd()
        if for_write:
            os.makedirs(directory, exist_ok=True)
            if not os.access(directory, os.W_OK):
                raise PermissionError("target directory is not writable")
        else:
            if not os.path.exists(abs_path):
                raise FileNotFoundError("model weights file not found")
            if not os.access(abs_path, os.R_OK):
                raise PermissionError("model weights file is not readable")
            if os.path.getsize(abs_path) > self.max_model_file_bytes:
                raise ValueError("model weights file exceeds maximum supported size")
        return abs_path

    def _validate_memory_stats(self, *arrays: np.ndarray) -> MemoryStats:
        total_bytes = sum(_array_bytes(np.asarray(array)) for array in arrays)
        _gc_hint_if_large(total_bytes)
        exceeded = total_bytes > self.max_array_bytes
        stats = MemoryStats(bytes_used=total_bytes, threshold_bytes=self.max_array_bytes, exceeded=exceeded)
        self.metrics["last_memory_bytes"] = total_bytes
        if exceeded:
            raise ValueError("input batch exceeds 1GB safety limit")
        if total_bytes > self.memory_warning_bytes:
            _structured_log(
                logging.WARNING,
                "memory_usage_warning",
                bytes_used=total_bytes,
                threshold=self.memory_warning_bytes,
            )
        return stats

    def _validate_state_vector(self, state: np.ndarray) -> np.ndarray:
        state_array = np.asarray(state, dtype=np.float32)
        self._validate_memory_stats(state_array)
        if state_array.shape != (self.state_size,):
            raise ValueError(f"state must have shape ({self.state_size},), received {state_array.shape}")
        if not np.all(np.isfinite(state_array)):
            raise ValueError("state contains NaN or infinite values")
        if np.max(np.abs(state_array), initial=0.0) > _Q_VALUE_MAGNITUDE_THRESHOLD:
            raise ValueError("state contains values outside the supported numerical range")
        return state_array

    def _validate_training_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        states = np.asarray(states, dtype=np.float32)
        next_states = np.asarray(next_states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int32)
        rewards = np.asarray(rewards, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)
        self._validate_memory_stats(states, next_states, actions, rewards, dones)

        if states.ndim != 2 or states.shape[1] != self.state_size:
            raise ValueError(f"states must have shape [batch_size, {self.state_size}], received {states.shape}")
        if next_states.ndim != 2 or next_states.shape[1] != self.state_size:
            raise ValueError(
                f"next_states must have shape [batch_size, {self.state_size}], received {next_states.shape}"
            )
        batch_size = states.shape[0]
        if batch_size == 0:
            raise ValueError("training batch must contain at least one sample")
        if batch_size > _MAX_BATCH_SIZE:
            raise ValueError(f"training batch exceeds safe batch size limit of {_MAX_BATCH_SIZE}")
        if actions.shape != (batch_size,):
            raise ValueError(f"actions must have shape ({batch_size},), received {actions.shape}")
        if rewards.shape != (batch_size,):
            raise ValueError(f"rewards must have shape ({batch_size},), received {rewards.shape}")
        if dones.shape != (batch_size,):
            raise ValueError(f"dones must have shape ({batch_size},), received {dones.shape}")
        if next_states.shape[0] != batch_size:
            raise ValueError(
                "next_states batch size must match states batch size; "
                f"received states={states.shape}, next_states={next_states.shape}"
            )
        if np.any(actions < 0) or np.any(actions >= self.action_size):
            raise ValueError("actions contain values outside the valid action range")
        for name, array in {"states": states, "next_states": next_states, "rewards": rewards, "dones": dones}.items():
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} contains NaN or infinite values")

        reward_mean = float(np.mean(rewards))
        reward_std = float(np.std(rewards))
        clipped_rewards = np.clip(rewards, -1.0, 1.0)
        outlier_threshold = reward_mean + 10.0 * max(reward_std, 1e-6)
        if np.any(np.abs(rewards) > max(abs(outlier_threshold), 1_000.0)):
            _structured_log(logging.WARNING, "reward_outlier_detected", reward_mean=reward_mean, reward_std=reward_std)
        self.reward_statistics = {
            "count": float(batch_size),
            "mean": float(np.mean(clipped_rewards)),
            "std": float(np.std(clipped_rewards)),
            "min": float(np.min(clipped_rewards)),
            "max": float(np.max(clipped_rewards)),
        }
        self.metrics["last_reward_mean"] = self.reward_statistics["mean"]
        self.metrics["last_reward_std"] = self.reward_statistics["std"]
        return states, actions, clipped_rewards.astype(np.float32), next_states, dones

    def _snapshot_runtime_state(self) -> Dict[str, Any]:
        return {
            "weights": [weight.copy() for weight in self.network.get_weights()],
            "target_weights": [weight.copy() for weight in self.target_network.get_weights()],
            "train_steps": self.train_steps,
            "epsilon": self.epsilon,
            "optimizer_lr": self._get_learning_rate(),
            "last_loss": self.last_loss,
            "loss_ema": self.loss_ema,
            "metrics": dict(self.metrics),
            "health_status": dict(self.health_status),
        }

    def _restore_runtime_state(self, snapshot: Mapping[str, Any]) -> None:
        self.network.set_weights(snapshot["weights"])
        self.target_network.set_weights(snapshot["target_weights"])
        self.train_steps = int(snapshot["train_steps"])
        self.epsilon = float(snapshot["epsilon"])
        self._set_learning_rate(float(snapshot["optimizer_lr"]))
        self.last_loss = snapshot["last_loss"]
        self.loss_ema = snapshot["loss_ema"]
        self.metrics = dict(snapshot["metrics"])
        self.health_status = dict(snapshot["health_status"])

    def _set_learning_rate(self, learning_rate: float) -> None:
        optimizer_learning_rate = self.optimizer.learning_rate
        if hasattr(optimizer_learning_rate, "assign"):
            optimizer_learning_rate.assign(learning_rate)
            return
        self.optimizer.learning_rate = learning_rate

    def _get_learning_rate(self) -> float:
        optimizer_learning_rate = self.optimizer.learning_rate
        if hasattr(optimizer_learning_rate, "numpy"):
            return float(optimizer_learning_rate.numpy())
        return float(optimizer_learning_rate)

    def _update_loss_ema(self, loss_value: float) -> None:
        self.last_loss = loss_value
        if self.loss_ema is None:
            self.loss_ema = loss_value
        else:
            self.loss_ema = self.loss_ema_momentum * self.loss_ema + (1.0 - self.loss_ema_momentum) * loss_value

    def _track_action_frequencies(self, actions: np.ndarray) -> None:
        unique_actions, counts = np.unique(actions, return_counts=True)
        for action, count in zip(unique_actions.tolist(), counts.tolist()):
            self.action_frequencies[int(action)] = self.action_frequencies.get(int(action), 0) + int(count)
        total = sum(self.action_frequencies.values())
        if total >= 100:
            dominant_frequency = max(self.action_frequencies.values()) / total
            if dominant_frequency > 0.98:
                _structured_log(logging.WARNING, "action_distribution_warning", dominant_frequency=dominant_frequency)

    def _validate_weight_magnitudes(self) -> None:
        max_weight = max(float(np.max(np.abs(weight))) for weight in self.network.get_weights())
        self.metrics["last_weight_max"] = max_weight
        if max_weight > self.weight_magnitude_threshold:
            raise ValueError("weight magnitude exceeded safe operating threshold")

    def _adapt_learning_rate_for_instability(self, reason: str) -> None:
        current_lr = self._get_learning_rate()
        new_lr = max(self._base_learning_rate * 0.1, current_lr * 0.5)
        if new_lr < current_lr:
            self._set_learning_rate(new_lr)
            self.metrics["instability_events"] += 1
            _structured_log(logging.WARNING, "learning_rate_reduced", reason=reason, old_lr=current_lr, new_lr=new_lr)

    def _maybe_checkpoint(self) -> None:
        if not self.checkpoint_interval or not self.checkpoint_dir:
            return
        if self.train_steps % self.checkpoint_interval != 0:
            return
        checkpoint_path = os.path.join(self.checkpoint_dir, f"train_step_{self.train_steps}.weights.h5")
        self.save_model(checkpoint_path)
        self.last_checkpoint_path = checkpoint_path

    def pause_training(self) -> None:
        """Pause training while continuing to allow safe reads and inference."""
        with self._lock:
            self.training_paused = True
            self.health_status.update({"status": "paused", "last_update": _utc_timestamp()})

    def resume_training(self) -> None:
        """Resume training after a previous pause."""
        with self._lock:
            self.training_paused = False
            self.health_status.update({"status": "resumed", "last_update": _utc_timestamp()})

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select an action using epsilon-greedy exploration."""
        with self._operation_guard("select_action") as correlation_id:
            try:
                self._assert_circuit_closed()
                state_array = self._validate_state_vector(state)
                with self._lock:
                    epsilon = float(self.epsilon)
                if training and np.random.random() < epsilon:
                    action = int(np.random.randint(0, self.action_size))
                else:
                    state_tensor = tf.convert_to_tensor(state_array[None, :], dtype=tf.float32)
                    with self._device_context():
                        q_values = self.network(state_tensor, training=False)
                    q_value_max = float(tf.reduce_max(tf.abs(q_values)).numpy())
                    self.metrics["last_q_value_max"] = q_value_max
                    if q_value_max > _Q_VALUE_MAGNITUDE_THRESHOLD:
                        raise ValueError("Q-values exceeded safe numerical range")
                    action = int(tf.argmax(q_values[0]).numpy())
                self._record_success("select_action", correlation_id)
                return action
            except Exception as error:
                self._record_failure(error, correlation_id, "select_action")
                raise

    def train_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> float:
        """Perform one validated DQN training step on a batch of experiences."""
        with self._operation_guard("train_step") as correlation_id:
            with self._lock:
                if self.training_paused:
                    _structured_log(logging.INFO, "train_step_skipped_paused", correlation_id=correlation_id)
                    return float(self.loss_ema or self.last_loss or 0.0)
                self._assert_circuit_closed()
                snapshot = self._snapshot_runtime_state()
            try:
                validated = self._validate_training_batch(states, actions, rewards, next_states, dones)
                return self._train_step_with_recovery(*validated, correlation_id=correlation_id, snapshot=snapshot)
            except Exception as error:
                with self._lock:
                    self._restore_runtime_state(snapshot)
                self._record_failure(error, correlation_id, "train_step")
                raise

    def _train_step_with_recovery(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        *,
        correlation_id: str,
        snapshot: Mapping[str, Any],
    ) -> float:
        attempts = 2
        backoff_seconds = 0.05
        last_error: Optional[BaseException] = None

        for attempt in range(attempts):
            try:
                with self._device_context():
                    return self._execute_train_step(
                        states,
                        actions,
                        rewards,
                        next_states,
                        dones,
                        correlation_id=correlation_id,
                    )
            except (tf.errors.ResourceExhaustedError, tf.errors.InternalError) as error:
                last_error = error
                self.metrics["training_retries"] += 1
                if self.device_name.startswith("/GPU"):
                    self.device_name = "/CPU:0"
                    _structured_log(logging.WARNING, "gpu_fallback_enabled", correlation_id=correlation_id)
                if attempt + 1 >= attempts:
                    break
                time.sleep(backoff_seconds * (2 ** attempt))
            except Exception as error:
                last_error = error
                break

        assert last_error is not None
        with self._lock:
            self._restore_runtime_state(snapshot)
        raise RuntimeError(
            "Training step failed after retries. Recovery actions applied; inspect logs and retry with a smaller batch."
        ) from last_error

    def _execute_train_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        *,
        correlation_id: str,
    ) -> float:
        start_time = time.perf_counter()
        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        actions_tensor = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards_tensor = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states_tensor = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones_tensor = tf.convert_to_tensor(dones, dtype=tf.float32)

        with tf.GradientTape() as tape:
            q_values = self.network(states_tensor, training=True)
            batch_indices = tf.range(tf.shape(q_values)[0], dtype=tf.int32)
            action_indices = tf.stack([batch_indices, actions_tensor], axis=1)
            current_q = tf.gather_nd(q_values, action_indices)

            next_q_values = self.target_network(next_states_tensor, training=False)
            if not tf.reduce_all(tf.math.is_finite(next_q_values)):
                raise ValueError("target network produced non-finite Q-values")
            max_next_q = tf.reduce_max(next_q_values, axis=1)
            target_q = rewards_tensor + self.gamma * max_next_q * (1.0 - dones_tensor)
            if not tf.reduce_all(tf.math.is_finite(target_q)):
                raise ValueError("target Q-values overflowed or became non-finite")
            target_q = tf.stop_gradient(target_q)
            loss = self.loss_fn(target_q, current_q)

        if not tf.math.is_finite(loss):
            raise ValueError("training produced a non-finite loss value")

        loss_value = float(loss.numpy())
        self._update_loss_ema(loss_value)
        if loss_value > self.loss_explosion_threshold:
            self._adapt_learning_rate_for_instability("loss_explosion")
            raise ValueError("loss explosion detected above critical threshold")

        gradients = tape.gradient(loss, self.network.trainable_weights)
        gradients_and_weights = [
            (gradient, weight)
            for gradient, weight in zip(gradients, self.network.trainable_weights)
            if gradient is not None
        ]
        if not gradients_and_weights:
            raise RuntimeError("no gradients were produced during the training step")

        gradient_max = max(float(tf.reduce_max(tf.abs(gradient)).numpy()) for gradient, _ in gradients_and_weights)
        self.metrics["last_gradient_max"] = gradient_max
        if gradient_max > self.gradient_warning_threshold:
            _structured_log(logging.WARNING, "gradient_warning", gradient_max=gradient_max, correlation_id=correlation_id)
            self._adapt_learning_rate_for_instability("gradient_warning")
            if self.stable_training and gradient_max > self.gradient_warning_threshold * 10:
                raise ValueError("stable training mode blocked an excessive gradient update")

        self.optimizer.apply_gradients(gradients_and_weights)
        self._validate_weight_magnitudes()
        self.train_loss.update_state(loss)
        self._track_action_frequencies(actions)

        with self._lock:
            self.train_steps += 1
            if self.train_steps % self.target_update_interval == 0:
                self.update_target_network()
            self._maybe_checkpoint()
        duration_ms = round((time.perf_counter() - start_time) * 1000.0, 3)
        self._record_success("train_step", correlation_id)
        _structured_log(
            logging.INFO,
            "train_step_completed",
            correlation_id=correlation_id,
            train_steps=self.train_steps,
            loss=loss_value,
            loss_ema=self.loss_ema,
            gradient_max=gradient_max,
            duration_ms=duration_ms,
        )
        return loss_value

    def update_target_network(self) -> None:
        """Synchronize target network weights from the online network."""
        with self._lock:
            self.target_network.set_weights(self.network.get_weights())

    def decay_epsilon(self) -> None:
        """Decay epsilon while respecting the configured minimum value."""
        with self._lock:
            decayed_epsilon = self.epsilon * self.epsilon_decay
            if self.epsilon <= self.epsilon_min * 2:
                self.epsilon = self.epsilon_min
            else:
                self.epsilon = max(self.epsilon_min, decayed_epsilon)

    def get_model_summary(self) -> str:
        """Return the model summary as a string."""
        buffer = io.StringIO()
        self.network.summary(print_fn=lambda line: buffer.write(line + os.linesep))
        return buffer.getvalue().strip()

    def get_config(self) -> Dict[str, float | int | str | List[int] | bool | None]:
        """Return serializable model configuration metadata."""
        with self._lock:
            return {
                "state_size": self.state_size,
                "action_size": self.action_size,
                "learning_rate": self.learning_rate,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_decay": self.epsilon_decay,
                "epsilon_min": self.epsilon_min,
                "model_type": self.model_type,
                "device": self.device,
                "hidden_layers": list(self.hidden_layers),
                "dropout_rate": self.dropout_rate,
                "use_batch_norm": self.use_batch_norm,
                "gradient_clip_norm": self.gradient_clip_norm,
                "seed": self.seed,
                "target_update_interval": self.target_update_interval,
                "checkpoint_interval": self.checkpoint_interval,
                "checkpoint_dir": self.checkpoint_dir,
                "stable_training": self.stable_training,
                "debug_mode": self.debug_mode,
            }

    def get_health_status(self) -> Dict[str, Any]:
        """Return a thread-safe health snapshot for observability consumers."""
        with self._lock:
            return dict(self.health_status)

    def get_state_snapshot(self) -> Dict[str, Any]:
        """Return a safe concurrent runtime snapshot for debugging and monitoring."""
        with self._lock:
            return {
                "config": self.get_config(),
                "train_steps": self.train_steps,
                "last_loss": self.last_loss,
                "loss_ema": self.loss_ema,
                "metrics": dict(self.metrics),
                "health_status": dict(self.health_status),
                "reward_statistics": dict(self.reward_statistics),
                "action_frequencies": dict(self.action_frequencies),
                "training_paused": self.training_paused,
                "last_checkpoint_path": self.last_checkpoint_path,
                "last_checkpoint_checksum": self.last_checkpoint_checksum,
            }

    def save_model(self, filepath: str) -> None:
        """Save online network weights and hardened metadata to disk."""
        with self._operation_guard("save_model") as correlation_id:
            try:
                safe_path = self._validate_path(filepath, for_write=True)
                metadata_path = f"{safe_path}.meta.npz"
                with FileLock(safe_path), self._lock:
                    np.savez(
                        metadata_path,
                        format_version=np.asarray(_METADATA_FORMAT_VERSION, dtype=np.int64),
                        train_steps=np.asarray(self.train_steps, dtype=np.int64),
                        epsilon=np.asarray(self.epsilon, dtype=np.float32),
                        loss_ema=np.asarray(self.loss_ema if self.loss_ema is not None else np.nan, dtype=np.float32),
                    )
                    self.network.save_weights(safe_path)
                    checksum = _sha256_file(safe_path)
                    checksum_meta = f"{safe_path}.sha256"
                    with open(checksum_meta, "w", encoding="utf-8") as handle:
                        handle.write(checksum)
                    self.last_checkpoint_checksum = checksum
                    self.last_checkpoint_path = safe_path
                self._record_success("save_model", correlation_id)
                _structured_log(logging.INFO, "model_saved", correlation_id=correlation_id, filepath=safe_path)
            except Exception as error:
                self._record_failure(error, correlation_id, "save_model")
                raise

    def load_model(self, filepath: str) -> None:
        """Load network weights and metadata with checksum validation and recovery."""
        with self._operation_guard("load_model") as correlation_id:
            safe_path = self._validate_path(filepath, for_write=False)
            metadata_path = f"{safe_path}.meta.npz"
            checksum_path = f"{safe_path}.sha256"
            try:
                with FileLock(safe_path), self._lock:
                    if os.path.exists(checksum_path):
                        expected_checksum = open(checksum_path, "r", encoding="utf-8").read().strip()
                        actual_checksum = _sha256_file(safe_path)
                        if expected_checksum != actual_checksum:
                            raise ValueError("model checksum verification failed")
                        self.last_checkpoint_checksum = actual_checksum

                    self.network.load_weights(safe_path)
                    self.train_steps = 0
                    if os.path.exists(metadata_path):
                        try:
                            metadata = np.load(metadata_path)
                            format_version = int(metadata["format_version"]) if "format_version" in metadata else 1
                            if format_version not in _RECOVERY_METADATA_FORMATS:
                                raise ValueError("unsupported metadata format version")
                            self.train_steps = int(metadata["train_steps"])
                            if "epsilon" in metadata:
                                self.epsilon = float(metadata["epsilon"])
                            if "loss_ema" in metadata and np.isfinite(metadata["loss_ema"]):
                                self.loss_ema = float(metadata["loss_ema"])
                        except Exception:
                            self.train_steps = 0
                            _structured_log(
                                logging.WARNING,
                                "metadata_recovery_used",
                                correlation_id=correlation_id,
                                filepath=safe_path,
                                recovery_instructions="Weights were restored. Rebuild metadata by saving the model again.",
                            )
                    self.update_target_network()
                    self.last_checkpoint_path = safe_path
                self._record_success("load_model", correlation_id)
                _structured_log(logging.INFO, "model_loaded", correlation_id=correlation_id, filepath=safe_path)
            except Exception as error:
                self._record_failure(error, correlation_id, "load_model")
                raise


class ExperienceReplay:
    """Experience replay buffer for storing and sampling DQN experiences."""

    def __init__(self, state_size: int, max_size: int = 100000, seed: Optional[int] = None) -> None:
        if state_size <= 0:
            raise ValueError("state_size must be a positive integer")
        if state_size > _MAX_STATE_SIZE:
            raise ValueError(f"state_size exceeds safe limit of {_MAX_STATE_SIZE}")
        if max_size <= 0:
            raise ValueError("max_size must be a positive integer")
        estimated_bytes = max_size * state_size * np.dtype(np.float32).itemsize * 2
        if estimated_bytes > _MAX_ARRAY_BYTES:
            raise ValueError("replay buffer configuration exceeds 1GB safety limit")

        self.state_size = state_size
        self.max_size = max_size
        self.buffer: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self.position = 0
        self.rng = np.random.default_rng(seed)
        self._lock = threading.RLock()
        self.memory_stats = MemoryStats(bytes_used=0, threshold_bytes=_MAX_ARRAY_BYTES, exceeded=False)

    def _update_memory_stats(self) -> None:
        approximate_bytes = len(self.buffer) * self.state_size * np.dtype(np.float32).itemsize * 2
        self.memory_stats = MemoryStats(
            bytes_used=int(approximate_bytes),
            threshold_bytes=_MAX_ARRAY_BYTES,
            exceeded=approximate_bytes > _MAX_ARRAY_BYTES,
        )
        if approximate_bytes > _MEMORY_WARNING_BYTES:
            _structured_log(logging.WARNING, "replay_memory_warning", bytes_used=approximate_bytes)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a validated experience to the replay buffer."""
        state_array = np.asarray(state, dtype=np.float32)
        next_state_array = np.asarray(next_state, dtype=np.float32)
        _gc_hint_if_large(_array_bytes(state_array) + _array_bytes(next_state_array))

        if state_array.shape != (self.state_size,):
            raise ValueError(f"state must have shape ({self.state_size},), received {state_array.shape}")
        if next_state_array.shape != (self.state_size,):
            raise ValueError(f"next_state must have shape ({self.state_size},), received {next_state_array.shape}")
        if not np.all(np.isfinite(state_array)):
            raise ValueError("state contains NaN or infinite values")
        if not np.all(np.isfinite(next_state_array)):
            raise ValueError("next_state contains NaN or infinite values")
        if not isinstance(action, (int, np.integer)):
            raise TypeError("action must be an integer")
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")

        experience = (
            state_array.copy(),
            int(action),
            float(np.clip(reward, -1.0, 1.0)),
            next_state_array.copy(),
            bool(done),
        )

        with self._lock:
            if len(self.buffer) < self.max_size:
                self.buffer.append(experience)
            else:
                self.buffer[self.position] = experience
            self.position = (self.position + 1) % self.max_size
            self._update_memory_stats()

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of experiences from the replay buffer."""
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        with self._lock:
            if not self.buffer:
                raise ValueError("cannot sample from an empty replay buffer")
            actual_batch_size = min(batch_size, len(self.buffer))
            indices = self.rng.choice(len(self.buffer), actual_batch_size, replace=False)
            experiences = [self.buffer[index] for index in indices]

        states = np.stack([experience[0] for experience in experiences]).astype(np.float32)
        actions = np.asarray([experience[1] for experience in experiences], dtype=np.int32)
        rewards = np.asarray([experience[2] for experience in experiences], dtype=np.float32)
        next_states = np.stack([experience[3] for experience in experiences]).astype(np.float32)
        dones = np.asarray([experience[4] for experience in experiences], dtype=np.float32)
        if states.nbytes + next_states.nbytes > _MAX_ARRAY_BYTES:
            raise ValueError("sampled batch exceeds 1GB safety limit")
        return states, actions, rewards, next_states, dones

    def clear(self) -> None:
        """Clear replay state to release references and encourage memory reclamation."""
        with self._lock:
            self.buffer.clear()
            self.position = 0
            self._update_memory_stats()
        gc.collect()

    def snapshot(self) -> Dict[str, Any]:
        """Return a lightweight concurrent replay buffer snapshot."""
        with self._lock:
            return {
                "size": len(self.buffer),
                "max_size": self.max_size,
                "position": self.position,
                "memory_bytes": self.memory_stats.bytes_used,
            }

    def __len__(self) -> int:
        """Return the current replay buffer size."""
        with self._lock:
            return len(self.buffer)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Creating example DQN agent learning model...")

    model = AgentLearningModel(
        state_size=64,
        action_size=10,
        learning_rate=0.001,
        model_type="dqn",
        device="cpu",
        seed=42,
        checkpoint_interval=0,
    )

    replay = ExperienceReplay(state_size=64, max_size=10000, seed=42)

    for _ in range(100):
        state = np.random.randn(64).astype(np.float32)
        action = model.select_action(state)
        reward = float(np.random.randn())
        next_state = np.random.randn(64).astype(np.float32)
        done = bool(np.random.random() > 0.9)
        replay.add(state, action, reward, next_state, done)

    if len(replay) >= 32:
        states, actions, rewards, next_states, dones = replay.sample(32)
        loss = model.train_step(states, actions, rewards, next_states, dones)
        model.decay_epsilon()
        logger.info("Training loss: %s", loss)
        logger.info("Current epsilon: %s", model.epsilon)

    logger.info("Example training completed successfully")
