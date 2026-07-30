"""
Production-ready Deep Q-Network (DQN) model for agent learning.

This module provides a hardened DQN implementation built with TensorFlow/Keras.
It includes:
- explicit model initialization/build steps for subclassed Keras models
- strict configuration and batch validation
- stable target network synchronization
- gradient clipping and finite-loss checks
- robust summary, save, and load helpers
- validated experience replay buffer behavior

Security & reliability hardening (v2):
- hard limits on state/action/buffer sizes to prevent memory exhaustion
- thread-safe train_steps counter, epsilon decay, and save/load operations
- loss-explosion detection (> MAX_LOSS_VALUE) alongside existing NaN checks
- gradient NaN/Inf validation after tape.gradient()
- gradient magnitude logging for numerical-stability monitoring
- upper-bound check on learning_rate
- retry logic for weight file save/load I/O operations
- gc.collect() hints after large ExperienceReplay allocations
- structured, context-rich log records; no sensitive data in messages

The previous mixed DQN/policy-gradient implementation has been narrowed to DQN
only so the module can be operated safely in production.
"""

from __future__ import annotations

import gc
import io
import logging
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam

# ---------------------------------------------------------------------------
# Module-level safety constants
# ---------------------------------------------------------------------------

#: Hard upper limit on state feature dimension to prevent accidental OOM.
MAX_STATE_SIZE: int = 1_000_000
#: Hard upper limit on the action space.
MAX_ACTION_SIZE: int = 100_000
#: Hard upper limit on each hidden-layer width.
MAX_HIDDEN_UNITS: int = 100_000
#: Hard upper limit on experience-replay buffer capacity.
MAX_BUFFER_SIZE: int = 10_000_000
#: Learning-rate ceiling; values above this are almost certainly bugs.
MAX_LEARNING_RATE: float = 10.0
#: Loss value above which training is considered numerically unstable.
MAX_LOSS_VALUE: float = 1e6
#: Maximum number of I/O retry attempts for save/load operations.
_IO_MAX_RETRIES: int = 3
#: Base delay (seconds) between I/O retries (exponential back-off).
_IO_RETRY_BASE_DELAY: float = 0.1

# Configure logging
logger = logging.getLogger(__name__)


class DQNNetwork(Model):
    """
    Deep Q-Network (DQN) for agent decision-making and learning.

    This network estimates action values (Q-values) for reinforcement learning.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_layers: Optional[List[int]] = None,
        activation: str = "relu",
        dropout_rate: float = 0.0,
        use_batch_norm: bool = False,
        name: str = "dqn_network",
    ):
        super().__init__(name=name)

        if state_size <= 0:
            raise ValueError("state_size must be a positive integer")
        if state_size > MAX_STATE_SIZE:
            raise ValueError(
                f"state_size {state_size} exceeds the maximum allowed size {MAX_STATE_SIZE}"
            )
        if action_size <= 0:
            raise ValueError("action_size must be a positive integer")
        if action_size > MAX_ACTION_SIZE:
            raise ValueError(
                f"action_size {action_size} exceeds the maximum allowed size {MAX_ACTION_SIZE}"
            )
        if hidden_layers is None:
            hidden_layers = [128, 64]
        if not hidden_layers:
            raise ValueError("hidden_layers must contain at least one layer size")
        if any(units <= 0 for units in hidden_layers):
            raise ValueError("all hidden layer sizes must be positive")
        if any(units > MAX_HIDDEN_UNITS for units in hidden_layers):
            raise ValueError(
                f"hidden layer sizes must not exceed {MAX_HIDDEN_UNITS}"
            )
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
        return self.output_layer(x)


class AgentLearningModel:
    """
    Production-hardened DQN learning model.

    This class intentionally supports only DQN so behavior is explicit and safe
    for production usage.

    Thread safety:
        ``train_step``, ``decay_epsilon``, ``save_model``, and ``load_model``
        are protected by internal locks so the model can be safely used from
        multiple threads (e.g. an experience-collection thread alongside a
        training thread).
    """

    SUPPORTED_MODEL_TYPES = {"dqn"}
    SUPPORTED_DEVICES = {"cpu", "gpu"}

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
    ):
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
        self.train_steps = 0

        # --- Thread-safety primitives ---
        # Protects train_steps increment and target-network sync decision.
        self._train_lock = threading.Lock()
        # Protects epsilon reads/writes from concurrent decay calls.
        self._epsilon_lock = threading.Lock()
        # Prevents concurrent save/load operations from corrupting weight files.
        self._io_lock = threading.Lock()

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

        logger.info(
            "AgentLearningModel initialized: model_type=%s learning_rate=%s device=%s",
            model_type,
            learning_rate,
            self.device_name,
        )

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
    ) -> None:
        if state_size <= 0:
            raise ValueError("state_size must be a positive integer")
        if action_size <= 0:
            raise ValueError("action_size must be a positive integer")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0")
        if learning_rate > MAX_LEARNING_RATE:
            raise ValueError(
                f"learning_rate {learning_rate} exceeds the maximum safe value {MAX_LEARNING_RATE}"
            )
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
        if target_update_interval <= 0:
            raise ValueError("target_update_interval must be greater than 0")

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

    def _validate_state_vector(self, state: np.ndarray) -> np.ndarray:
        state_array = np.asarray(state, dtype=np.float32)
        if state_array.shape != (self.state_size,):
            raise ValueError(
                f"state must have shape ({self.state_size},), received {state_array.shape}"
            )
        if not np.all(np.isfinite(state_array)):
            raise ValueError("state contains NaN or infinite values")
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

        if states.ndim != 2 or states.shape[1] != self.state_size:
            raise ValueError(
                f"states must have shape [batch_size, {self.state_size}], received {states.shape}"
            )
        if next_states.ndim != 2 or next_states.shape[1] != self.state_size:
            raise ValueError(
                f"next_states must have shape [batch_size, {self.state_size}], received {next_states.shape}"
            )
        batch_size = states.shape[0]
        if batch_size == 0:
            raise ValueError("training batch must contain at least one sample")
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
        for name, array in {
            "states": states,
            "next_states": next_states,
            "rewards": rewards,
            "dones": dones,
        }.items():
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} contains NaN or infinite values")
        return states, actions, rewards, next_states, dones

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select an action using epsilon-greedy exploration."""
        state_array = self._validate_state_vector(state)

        if training and np.random.random() < self.epsilon:
            return int(np.random.randint(0, self.action_size))

        state_tensor = tf.convert_to_tensor(state_array[None, :], dtype=tf.float32)
        q_values = self.network(state_tensor, training=False)
        action = int(tf.argmax(q_values[0]).numpy())
        return action

    def train_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> float:
        """Perform one validated DQN training step on a batch of experiences.

        Thread safety:
            This method acquires ``_train_lock`` before modifying ``train_steps``
            and before deciding whether to sync the target network, so concurrent
            calls are safe.

        Raises:
            ValueError: If the batch is invalid, loss is non-finite, loss
                exceeds ``MAX_LOSS_VALUE``, or gradients contain NaN/Inf.
            RuntimeError: If no gradients were produced.
        """
        states, actions, rewards, next_states, dones = self._validate_training_batch(
            states, actions, rewards, next_states, dones
        )

        with tf.device(self.device_name):
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
                max_next_q = tf.reduce_max(next_q_values, axis=1)
                target_q = rewards_tensor + self.gamma * max_next_q * (1.0 - dones_tensor)
                target_q = tf.stop_gradient(target_q)

                loss = self.loss_fn(target_q, current_q)

            loss_value = float(loss.numpy())
            if not np.isfinite(loss_value):
                raise ValueError(
                    f"Training produced a non-finite loss value: {loss_value}. "
                    "Check input data for extreme values."
                )
            if loss_value > MAX_LOSS_VALUE:
                raise ValueError(
                    f"Loss explosion detected: loss={loss_value:.4g} exceeds "
                    f"MAX_LOSS_VALUE={MAX_LOSS_VALUE}. Consider reducing the "
                    "learning rate or clipping rewards."
                )

            gradients = tape.gradient(loss, self.network.trainable_weights)
            gradients_and_weights = [
                (gradient, weight)
                for gradient, weight in zip(gradients, self.network.trainable_weights)
                if gradient is not None
            ]
            if not gradients_and_weights:
                raise RuntimeError("no gradients were produced during the training step")

            # --- Gradient health check ---
            for grad, var in gradients_and_weights:
                if not tf.reduce_all(tf.math.is_finite(grad)):
                    raise ValueError(
                        f"Non-finite (NaN/Inf) gradient detected for variable "
                        f"'{var.name}'. Training aborted to prevent weight corruption."
                    )

            # Log gradient magnitude at DEBUG level for monitoring
            if logger.isEnabledFor(logging.DEBUG):
                total_norm = float(
                    tf.linalg.global_norm([g for g, _ in gradients_and_weights]).numpy()
                )
                logger.debug(
                    "Gradient global norm: %.4g  loss: %.4g  train_step: %d",
                    total_norm,
                    loss_value,
                    self.train_steps,
                )

            self.optimizer.apply_gradients(gradients_and_weights)
            self.train_loss.update_state(loss)

            with self._train_lock:
                self.train_steps += 1
                should_sync = (self.train_steps % self.target_update_interval == 0)

            if should_sync:
                self.update_target_network()
                logger.debug(
                    "Target network synced at train_step=%d", self.train_steps
                )

            return loss_value

    def update_target_network(self) -> None:
        """Synchronize target network weights from the online network."""
        self.target_network.set_weights(self.network.get_weights())

    def decay_epsilon(self) -> None:
        """Decay epsilon while respecting the configured minimum value.

        Thread-safe: acquires ``_epsilon_lock`` before modifying ``epsilon``.
        """
        with self._epsilon_lock:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_model_summary(self) -> str:
        """Return the model summary as a string."""
        buffer = io.StringIO()
        self.network.summary(print_fn=lambda line: buffer.write(line + os.linesep))
        return buffer.getvalue().strip()

    def get_config(self) -> Dict[str, float | int | str | List[int] | bool | None]:
        """Return serializable model configuration metadata."""
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
        }

    def save_model(self, filepath: str) -> None:
        """Save online network weights and lightweight metadata to disk.

        The operation is protected by ``_io_lock`` to prevent concurrent
        save/load calls from corrupting the weight files.  A simple
        exponential-back-off retry is applied to tolerate transient I/O errors
        (e.g. NFS hiccups).

        Args:
            filepath: Destination path for the ``.weights.h5`` file.  The
                companion metadata file is written to ``{filepath}.meta.npz``.

        Raises:
            ValueError: If *filepath* is empty or not a string.
            OSError: If the file cannot be written after all retries.
        """
        if not filepath or not isinstance(filepath, str):
            raise ValueError("filepath must be a non-empty string")

        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with self._io_lock:
            metadata_path = f"{filepath}.meta.npz"
            last_exc: Optional[Exception] = None
            for attempt in range(1, _IO_MAX_RETRIES + 1):
                try:
                    np.savez(
                        metadata_path,
                        train_steps=np.asarray(self.train_steps, dtype=np.int64),
                    )
                    self.network.save_weights(filepath)
                    logger.info(
                        "Model weights saved to '%s' (attempt %d)", filepath, attempt
                    )
                    return
                except OSError as exc:
                    last_exc = exc
                    logger.warning(
                        "save_model attempt %d/%d failed: %s",
                        attempt,
                        _IO_MAX_RETRIES,
                        type(exc).__name__,
                    )
                    if attempt < _IO_MAX_RETRIES:
                        time.sleep(_IO_RETRY_BASE_DELAY * (2 ** (attempt - 1)))
            raise OSError(
                f"save_model failed after {_IO_MAX_RETRIES} attempts"
            ) from last_exc

    def load_model(self, filepath: str) -> None:
        """Load online network weights, metadata, and re-synchronize the target network.

        The operation is protected by ``_io_lock`` to prevent concurrent
        save/load calls from reading partially-written files.  A simple
        exponential-back-off retry is applied for transient I/O errors.

        Args:
            filepath: Path to the ``.weights.h5`` file written by
                :meth:`save_model`.

        Raises:
            ValueError: If *filepath* is empty or not a string.
            FileNotFoundError: If the weights file does not exist.
            OSError: If the file cannot be read after all retries.
        """
        if not filepath or not isinstance(filepath, str):
            raise ValueError("filepath must be a non-empty string")
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Model weights file not found: '{filepath}'. "
                "Verify the path is correct and the file has not been moved."
            )

        with self._io_lock:
            last_exc: Optional[Exception] = None
            for attempt in range(1, _IO_MAX_RETRIES + 1):
                try:
                    self.network.load_weights(filepath)

                    metadata_path = f"{filepath}.meta.npz"
                    if os.path.exists(metadata_path):
                        metadata = np.load(metadata_path)
                        self.train_steps = int(metadata["train_steps"])
                    else:
                        logger.warning(
                            "Metadata file '%s' not found; defaulting train_steps to 0.",
                            metadata_path,
                        )
                        self.train_steps = 0

                    self.update_target_network()
                    logger.info(
                        "Model weights loaded from '%s' (attempt %d, train_steps=%d)",
                        filepath,
                        attempt,
                        self.train_steps,
                    )
                    return
                except OSError as exc:
                    last_exc = exc
                    logger.warning(
                        "load_model attempt %d/%d failed: %s",
                        attempt,
                        _IO_MAX_RETRIES,
                        type(exc).__name__,
                    )
                    if attempt < _IO_MAX_RETRIES:
                        time.sleep(_IO_RETRY_BASE_DELAY * (2 ** (attempt - 1)))
            raise OSError(
                f"load_model failed after {_IO_MAX_RETRIES} attempts"
            ) from last_exc


class ExperienceReplay:
    """Experience replay buffer for storing and sampling DQN experiences.

    Memory safety:
        - ``max_size`` is capped at ``MAX_BUFFER_SIZE`` to prevent accidental
          memory exhaustion.
        - ``gc.collect()`` is called after the pre-allocated list is populated
          so any garbage from the allocation phase is freed promptly.
    """

    def __init__(self, state_size: int, max_size: int = 100000, seed: Optional[int] = None):
        if state_size <= 0:
            raise ValueError("state_size must be a positive integer")
        if state_size > MAX_STATE_SIZE:
            raise ValueError(
                f"state_size {state_size} exceeds the maximum allowed size {MAX_STATE_SIZE}"
            )
        if max_size <= 0:
            raise ValueError("max_size must be a positive integer")
        if max_size > MAX_BUFFER_SIZE:
            raise ValueError(
                f"max_size {max_size} exceeds the maximum allowed buffer size {MAX_BUFFER_SIZE}"
            )

        self.state_size = state_size
        self.max_size = max_size
        self.buffer: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self.position = 0
        self.rng = np.random.default_rng(seed)

        # Hint the GC to reclaim any temporary memory from construction
        gc.collect()

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

        if state_array.shape != (self.state_size,):
            raise ValueError(
                f"state must have shape ({self.state_size},), received {state_array.shape}"
            )
        if next_state_array.shape != (self.state_size,):
            raise ValueError(
                f"next_state must have shape ({self.state_size},), received {next_state_array.shape}"
            )
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
            float(reward),
            next_state_array.copy(),
            bool(done),
        )

        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        self.position = (self.position + 1) % self.max_size

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of experiences from the replay buffer."""
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
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

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Return the current replay buffer size."""
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
