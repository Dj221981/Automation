"""
Production-ready data processing utilities for AI-morphasis neural network models.

Handles data loading, preprocessing, normalization, and augmentation
for agent training and inference. This module includes:
- strict input validation and shape checking
- robust file I/O with error handling
- finite-value checks (NaN/inf detection)
- configuration validation
- comprehensive logging
- reproducible data operations
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class NormalizationType(Enum):
    """Types of normalization."""
    NONE = "none"
    MINMAX = "minmax"
    ZSCORE = "zscore"
    ROBUST = "robust"


@dataclass
class NormalizationStats:
    """Statistics for normalization."""
    min_val: np.ndarray
    max_val: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    median: np.ndarray
    q25: np.ndarray
    q75: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "min": self.min_val.tolist(),
            "max": self.max_val.tolist(),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "median": self.median.tolist(),
            "q25": self.q25.tolist(),
            "q75": self.q75.tolist()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NormalizationStats:
        """Create from dictionary."""
        required_keys = {"min", "max", "mean", "std", "median", "q25", "q75"}
        if not required_keys.issubset(data.keys()):
            missing = required_keys - set(data.keys())
            raise ValueError(f"Missing required keys in normalization stats: {missing}")
        
        return cls(
            min_val=np.array(data["min"], dtype=np.float32),
            max_val=np.array(data["max"], dtype=np.float32),
            mean=np.array(data["mean"], dtype=np.float32),
            std=np.array(data["std"], dtype=np.float32),
            median=np.array(data["median"], dtype=np.float32),
            q25=np.array(data["q25"], dtype=np.float32),
            q75=np.array(data["q75"], dtype=np.float32)
        )


class StateNormalizer:
    """Normalizes state data for neural network input."""

    def __init__(
        self,
        normalization_type: NormalizationType = NormalizationType.MINMAX,
        epsilon: float = 1e-8,
        feature_size: Optional[int] = None
    ):
        """
        Initialize state normalizer.

        Args:
            normalization_type: Type of normalization to apply
            epsilon: Small value to prevent division by zero (must be > 0)
            feature_size: Expected feature dimension (optional validation hint)

        Raises:
            ValueError: If epsilon is not positive
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, received {epsilon}")
        
        self.normalization_type = normalization_type
        self.epsilon = epsilon
        self.feature_size = feature_size
        self.stats: Optional[NormalizationStats] = None
        self.fitted = False
        logger.debug(f"StateNormalizer initialized with type={normalization_type.value}, epsilon={epsilon}")

    def fit(self, data: np.ndarray) -> None:
        """
        Fit normalizer on data with strict validation.

        Args:
            data: Data to compute statistics from [samples, features]

        Raises:
            ValueError: If data is invalid (empty, non-2D, contains NaN/inf)
        """
        data = np.asarray(data, dtype=np.float32)
        
        if data.ndim != 2:
            raise ValueError(f"data must be 2-dimensional [samples, features], got shape {data.shape}")
        
        if data.shape[0] == 0:
            raise ValueError("data must contain at least one sample")
        
        if data.shape[1] == 0:
            raise ValueError("data must contain at least one feature")
        
        if not np.all(np.isfinite(data)):
            nan_count = np.isnan(data).sum()
            inf_count = np.isinf(data).sum()
            raise ValueError(
                f"data contains NaN or infinite values (NaN: {nan_count}, Inf: {inf_count})"
            )
        
        if self.feature_size is not None and data.shape[1] != self.feature_size:
            raise ValueError(
                f"data feature dimension {data.shape[1]} does not match expected {self.feature_size}"
            )
        
        self.feature_size = data.shape[1]
        
        self.stats = NormalizationStats(
            min_val=np.min(data, axis=0),
            max_val=np.max(data, axis=0),
            mean=np.mean(data, axis=0),
            std=np.std(data, axis=0),
            median=np.median(data, axis=0),
            q25=np.percentile(data, 25, axis=0),
            q75=np.percentile(data, 75, axis=0)
        )
        
        self.fitted = True
        logger.info(f"StateNormalizer fitted with {data.shape[0]} samples, {data.shape[1]} features")

    def _validate_input_shape(self, data: np.ndarray, operation: str = "normalize") -> np.ndarray:
        """Validate and convert input data shape."""
        data = np.asarray(data, dtype=np.float32)
        
        if self.feature_size is None:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        
        if data.ndim == 1:
            if data.shape[0] != self.feature_size:
                raise ValueError(
                    f"1D input must match feature size {self.feature_size}, got {data.shape[0]}"
                )
        elif data.ndim == 2:
            if data.shape[1] != self.feature_size:
                raise ValueError(
                    f"2D input must have {self.feature_size} features, got {data.shape[1]}"
                )
        else:
            raise ValueError(f"input must be 1D or 2D, got shape {data.shape}")
        
        if not np.all(np.isfinite(data)):
            nan_count = np.isnan(data).sum()
            inf_count = np.isinf(data).sum()
            raise ValueError(
                f"input data for {operation} contains NaN or infinite values (NaN: {nan_count}, Inf: {inf_count})"
            )
        
        return data

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data with strict validation.

        Args:
            data: Data to normalize [samples, features] or [features,]

        Returns:
            Normalized data

        Raises:
            ValueError: If normalizer not fitted or input is invalid
        """
        if not self.fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        data = self._validate_input_shape(data, "normalize")
        
        if self.normalization_type == NormalizationType.NONE:
            return data
        
        elif self.normalization_type == NormalizationType.MINMAX:
            range_val = self.stats.max_val - self.stats.min_val
            range_val = np.where(range_val == 0, self.epsilon, range_val)
            normalized = (data - self.stats.min_val) / range_val
        
        elif self.normalization_type == NormalizationType.ZSCORE:
            normalized = (data - self.stats.mean) / (self.stats.std + self.epsilon)
        
        elif self.normalization_type == NormalizationType.ROBUST:
            iqr = self.stats.q75 - self.stats.q25
            iqr = np.where(iqr == 0, self.epsilon, iqr)
            normalized = (data - self.stats.median) / iqr
        
        else:
            raise ValueError(f"Unknown normalization type: {self.normalization_type}")
        
        if not np.all(np.isfinite(normalized)):
            nan_count = np.isnan(normalized).sum()
            inf_count = np.isinf(normalized).sum()
            raise ValueError(
                f"normalization produced NaN or infinite values (NaN: {nan_count}, Inf: {inf_count})"
            )
        
        return normalized

    def denormalize(self, data: np.ndarray) -> np.ndarray:
        """
        Reverse normalization with strict validation.

        Args:
            data: Normalized data [samples, features] or [features,]

        Returns:
            Original scale data

        Raises:
            ValueError: If normalizer not fitted or input is invalid
        """
        if not self.fitted:
            raise ValueError("Normalizer not fitted.")
        
        data = self._validate_input_shape(data, "denormalize")
        
        if self.normalization_type == NormalizationType.NONE:
            return data
        
        elif self.normalization_type == NormalizationType.MINMAX:
            range_val = self.stats.max_val - self.stats.min_val
            denormalized = data * range_val + self.stats.min_val
        
        elif self.normalization_type == NormalizationType.ZSCORE:
            denormalized = data * self.stats.std + self.stats.mean
        
        elif self.normalization_type == NormalizationType.ROBUST:
            iqr = self.stats.q75 - self.stats.q25
            denormalized = data * iqr + self.stats.median
        
        else:
            raise ValueError(f"Unknown normalization type: {self.normalization_type}")
        
        if not np.all(np.isfinite(denormalized)):
            nan_count = np.isnan(denormalized).sum()
            inf_count = np.isinf(denormalized).sum()
            raise ValueError(
                f"denormalization produced NaN or infinite values (NaN: {nan_count}, Inf: {inf_count})"
            )
        
        return denormalized

    def save(self, filepath: str) -> None:
        """
        Save normalizer statistics to disk.

        Args:
            filepath: Path to save normalizer

        Raises:
            ValueError: If normalizer not fitted or filepath is invalid
            OSError: If file I/O fails
        """
        if not self.fitted:
            raise ValueError("Normalizer not fitted. Cannot save unfitted normalizer.")
        
        if not filepath or not isinstance(filepath, str):
            raise ValueError("filepath must be a non-empty string")
        
        try:
            filepath_obj = Path(filepath)
            filepath_obj.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "type": self.normalization_type.value,
                "feature_size": self.feature_size,
                "epsilon": float(self.epsilon),
                "stats": self.stats.to_dict()
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Normalizer saved to {filepath}")
        
        except (OSError, IOError) as e:
            logger.error(f"Failed to save normalizer to {filepath}: {e}")
            raise

    def load(self, filepath: str) -> None:
        """
        Load normalizer statistics from disk.

        Args:
            filepath: Path to normalizer file

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file format is invalid
            OSError: If file I/O fails
        """
        if not filepath or not isinstance(filepath, str):
            raise ValueError("filepath must be a non-empty string")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Normalizer file not found: {filepath}")
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            if "type" not in data or "stats" not in data:
                raise ValueError("Invalid normalizer file format: missing 'type' or 'stats'")
            
            self.normalization_type = NormalizationType(data["type"])
            self.feature_size = data.get("feature_size")
            self.epsilon = data.get("epsilon", 1e-8)
            self.stats = NormalizationStats.from_dict(data["stats"])
            self.fitted = True
            
            logger.info(f"Normalizer loaded from {filepath} (type={self.normalization_type.value})")
        
        except (OSError, IOError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load normalizer from {filepath}: {e}")
            raise


class DataAugmentation:
    """Production-ready data augmentation utilities for training."""

    @staticmethod
    def add_gaussian_noise(
        data: np.ndarray,
        noise_std: float = 0.1,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Add Gaussian noise to data.

        Args:
            data: Input data
            noise_std: Standard deviation of noise (must be >= 0)
            seed: Random seed for reproducibility

        Returns:
            Data with added noise

        Raises:
            ValueError: If noise_std is negative or data contains NaN/inf
        """
        if noise_std < 0:
            raise ValueError(f"noise_std must be non-negative, got {noise_std}")
        
        data = np.asarray(data, dtype=np.float32)
        
        if not np.all(np.isfinite(data)):
            raise ValueError("input data contains NaN or infinite values")
        
        if seed is not None:
            np.random.seed(seed)
        
        noise = np.random.normal(0, noise_std, data.shape)
        augmented = data + noise
        
        if not np.all(np.isfinite(augmented)):
            raise ValueError("augmentation produced NaN or infinite values")
        
        return augmented

    @staticmethod
    def mixup(
        x1: np.ndarray,
        x2: np.ndarray,
        y1: np.ndarray,
        y2: np.ndarray,
        alpha: float = 0.2,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mixup data augmentation with validation.

        Args:
            x1: First batch of features
            x2: Second batch of features
            y1: First batch of labels/rewards
            y2: Second batch of labels/rewards
            alpha: Beta parameter for mixup (must be > 0)
            seed: Random seed for reproducibility

        Returns:
            Tuple of (mixed_x, mixed_y)

        Raises:
            ValueError: If inputs are invalid or alpha is not positive
        """
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        
        x1 = np.asarray(x1, dtype=np.float32)
        x2 = np.asarray(x2, dtype=np.float32)
        y1 = np.asarray(y1, dtype=np.float32)
        y2 = np.asarray(y2, dtype=np.float32)
        
        if x1.shape != x2.shape:
            raise ValueError(f"x1 and x2 must have same shape, got {x1.shape} vs {x2.shape}")
        if y1.shape != y2.shape:
            raise ValueError(f"y1 and y2 must have same shape, got {y1.shape} vs {y2.shape}")
        
        for name, arr in {"x1": x1, "x2": x2, "y1": y1, "y2": y2}.items():
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} contains NaN or infinite values")
        
        if seed is not None:
            np.random.seed(seed)
        
        lam = np.random.beta(alpha, alpha)
        x_mixed = lam * x1 + (1 - lam) * x2
        y_mixed = lam * y1 + (1 - lam) * y2
        
        for name, arr in {"x_mixed": x_mixed, "y_mixed": y_mixed}.items():
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} contains NaN or infinite values")
        
        return x_mixed, y_mixed

    @staticmethod
    def random_crop(
        data: np.ndarray,
        crop_fraction: float = 0.9,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Random feature dropout (crop).

        Args:
            data: Input data [batch, features]
            crop_fraction: Fraction of features to keep (must be in (0, 1])
            seed: Random seed for reproducibility

        Returns:
            Data with random features zeroed

        Raises:
            ValueError: If crop_fraction is invalid
        """
        if not 0 < crop_fraction <= 1.0:
            raise ValueError(f"crop_fraction must be in (0, 1], got {crop_fraction}")
        
        data = np.asarray(data, dtype=np.float32)
        
        if not np.all(np.isfinite(data)):
            raise ValueError("input data contains NaN or infinite values")
        
        if seed is not None:
            np.random.seed(seed)
        
        augmented = data.copy()
        num_features = data.shape[-1]
        num_drop = int(num_features * (1 - crop_fraction))
        
        if num_drop > 0:
            drop_idx = np.random.choice(num_features, num_drop, replace=False)
            augmented[..., drop_idx] = 0
        
        return augmented

    @staticmethod
    def temporal_shift(
        sequences: np.ndarray,
        shift_range: int = 2,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Temporal shift for sequence data.

        Args:
            sequences: Sequence data [batch, time, features]
            shift_range: Maximum shift amount (must be non-negative)
            seed: Random seed for reproducibility

        Returns:
            Shifted sequences

        Raises:
            ValueError: If shift_range is negative or sequences invalid
        """
        if shift_range < 0:
            raise ValueError(f"shift_range must be non-negative, got {shift_range}")
        
        sequences = np.asarray(sequences, dtype=np.float32)
        
        if sequences.ndim != 3:
            raise ValueError(f"sequences must be 3-dimensional [batch, time, features], got {sequences.shape}")
        
        if not np.all(np.isfinite(sequences)):
            raise ValueError("input sequences contain NaN or infinite values")
        
        if seed is not None:
            np.random.seed(seed)
        
        shift = np.random.randint(-shift_range, shift_range + 1)
        shifted = np.roll(sequences, shift, axis=1)
        
        return shifted


class ExperiencePreprocessor:
    """Production-ready experience preprocessor for training."""

    def __init__(
        self,
        state_normalizer: Optional[StateNormalizer] = None,
        reward_normalizer: Optional[StateNormalizer] = None
    ):
        """
        Initialize preprocessor.

        Args:
            state_normalizer: Normalizer for states
            reward_normalizer: Normalizer for rewards

        Raises:
            TypeError: If normalizers are not StateNormalizer instances
        """
        if state_normalizer is not None and not isinstance(state_normalizer, StateNormalizer):
            raise TypeError("state_normalizer must be a StateNormalizer instance")
        if reward_normalizer is not None and not isinstance(reward_normalizer, StateNormalizer):
            raise TypeError("reward_normalizer must be a StateNormalizer instance")
        
        self.state_normalizer = state_normalizer
        self.reward_normalizer = reward_normalizer
        logger.debug("ExperiencePreprocessor initialized")

    def process_state(self, state: np.ndarray) -> np.ndarray:
        """
        Process state with validation.

        Args:
            state: Raw state

        Returns:
            Processed state

        Raises:
            ValueError: If state is invalid
        """
        state = np.asarray(state, dtype=np.float32)
        
        if not np.all(np.isfinite(state)):
            raise ValueError("state contains NaN or infinite values")
        
        if self.state_normalizer and self.state_normalizer.fitted:
            state = self.state_normalizer.normalize(state)
        
        return state

    def process_reward(self, reward: float) -> float:
        """
        Process reward with validation.

        Args:
            reward: Raw reward

        Returns:
            Processed reward

        Raises:
            ValueError: If reward is invalid
        """
        reward_array = np.asarray([reward], dtype=np.float32)
        
        if not np.isfinite(reward_array[0]):
            raise ValueError(f"reward contains NaN or infinite value: {reward}")
        
        if self.reward_normalizer and self.reward_normalizer.fitted:
            reward_array = self.reward_normalizer.normalize(reward_array)
        
        return float(reward_array[0])

    def process_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Process batch of experience with comprehensive validation.

        Args:
            states: State batch
            actions: Action batch
            rewards: Reward batch
            next_states: Next state batch
            dones: Done flags

        Returns:
            Processed batch tuple

        Raises:
            ValueError: If batch is invalid
        """
        states = np.asarray(states, dtype=np.float32)
        next_states = np.asarray(next_states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int32)
        rewards = np.asarray(rewards, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)
        
        # Validate shapes
        if states.ndim != 2:
            raise ValueError(f"states must be 2D, got shape {states.shape}")
        if next_states.ndim != 2:
            raise ValueError(f"next_states must be 2D, got shape {next_states.shape}")
        
        batch_size = states.shape[0]
        if batch_size == 0:
            raise ValueError("batch must contain at least one sample")
        
        if states.shape != next_states.shape:
            raise ValueError(
                f"states and next_states must have same shape, got {states.shape} vs {next_states.shape}"
            )
        if actions.shape != (batch_size,):
            raise ValueError(f"actions must have shape ({batch_size},), got {actions.shape}")
        if rewards.shape != (batch_size,):
            raise ValueError(f"rewards must have shape ({batch_size},), got {rewards.shape}")
        if dones.shape != (batch_size,):
            raise ValueError(f"dones must have shape ({batch_size},), got {dones.shape}")
        
        # Validate finite values
        for name, arr in {"states": states, "next_states": next_states, "rewards": rewards}.items():
            if not np.all(np.isfinite(arr)):
                nan_count = np.isnan(arr).sum()
                inf_count = np.isinf(arr).sum()
                raise ValueError(
                    f"{name} contains NaN or infinite values (NaN: {nan_count}, Inf: {inf_count})"
                )
        
        # Apply normalization
        if self.state_normalizer and self.state_normalizer.fitted:
            states = self.state_normalizer.normalize(states)
            next_states = self.state_normalizer.normalize(next_states)
        
        if self.reward_normalizer and self.reward_normalizer.fitted:
            rewards = self.reward_normalizer.normalize(rewards.reshape(-1, 1)).flatten()
        
        return states, actions, rewards, next_states, dones


class BatchGenerator:
    """Production-ready batch generator with validation."""

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize batch generator.

        Args:
            states: State array
            actions: Action array
            rewards: Reward array
            next_states: Next state array
            dones: Done flags
            batch_size: Batch size (must be positive)
            shuffle: Whether to shuffle data
            seed: Random seed for reproducibility

        Raises:
            ValueError: If inputs are invalid
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int32)
        rewards = np.asarray(rewards, dtype=np.float32)
        next_states = np.asarray(next_states, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)
        
        num_samples = len(states)
        if num_samples == 0:
            raise ValueError("arrays must contain at least one sample")
        
        if not all(len(arr) == num_samples for arr in [actions, rewards, next_states, dones]):
            raise ValueError("all arrays must have the same length")
        
        if not np.all(np.isfinite(states)) or not np.all(np.isfinite(next_states)) or \
           not np.all(np.isfinite(rewards)):
            raise ValueError("input arrays contain NaN or infinite values")
        
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.dones = dones
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = num_samples
        self.indices = np.arange(self.num_samples)
        
        if seed is not None:
            np.random.seed(seed)
        
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        self.current_idx = 0
        logger.debug(f"BatchGenerator initialized: {num_samples} samples, batch_size={batch_size}")

    def __iter__(self):
        """Iterate through batches."""
        self.current_idx = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get next batch."""
        if self.current_idx >= self.num_samples:
            raise StopIteration
        
        end_idx = min(self.current_idx + self.batch_size, self.num_samples)
        batch_indices = self.indices[self.current_idx:end_idx]
        
        batch = (
            self.states[batch_indices],
            self.actions[batch_indices],
            self.rewards[batch_indices],
            self.next_states[batch_indices],
            self.dones[batch_indices]
        )
        
        self.current_idx = end_idx
        return batch

    def __len__(self) -> int:
        """Number of batches."""
        return int(np.ceil(self.num_samples / self.batch_size))


def split_data(
    data: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    shuffle: bool = True,
    random_seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train/validation/test sets with validation.

    Args:
        data: Data to split
        train_ratio: Training set ratio (must be in (0, 1))
        val_ratio: Validation set ratio (must be in [0, 1))
        shuffle: Whether to shuffle data
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_data, val_data, test_data)

    Raises:
        ValueError: If ratios are invalid or data is empty
    """
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    if not 0 <= val_ratio < 1:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    if train_ratio + val_ratio >= 1:
        raise ValueError(
            f"train_ratio + val_ratio must be < 1, got {train_ratio + val_ratio}"
        )
    
    data = np.asarray(data, dtype=np.float32)
    
    if len(data) == 0:
        raise ValueError("data must not be empty")
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    num_samples = len(data)
    indices = np.arange(num_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    train_size = int(num_samples * train_ratio)
    val_size = int(num_samples * val_ratio)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    logger.info(
        f"Data split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
    )
    
    return data[train_idx], data[val_idx], data[test_idx]


def create_sliding_window(
    data: np.ndarray,
    window_size: int,
    step: int = 1
) -> np.ndarray:
    """
    Create sliding windows from sequence data.

    Args:
        data: Sequence data [time, features]
        window_size: Window size (must be positive)
        step: Step size (must be positive)

    Returns:
        Windowed data [num_windows, window_size, features]

    Raises:
        ValueError: If window_size or step is invalid, or data is insufficient
    """
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")
    
    data = np.asarray(data, dtype=np.float32)
    
    if data.ndim != 2:
        raise ValueError(f"data must be 2D [time, features], got shape {data.shape}")
    
    if data.shape[0] < window_size:
        raise ValueError(
            f"data has {data.shape[0]} time steps but window_size is {window_size}"
        )
    
    if not np.all(np.isfinite(data)):
        raise ValueError("data contains NaN or infinite values")
    
    num_windows = (data.shape[0] - window_size) // step + 1
    if num_windows <= 0:
        raise ValueError(
            f"cannot create windows: {data.shape[0]} time steps, window_size={window_size}, step={step}"
        )
    
    windows = np.zeros((num_windows, window_size, data.shape[1]), dtype=np.float32)
    
    for i in range(num_windows):
        start = i * step
        end = start + window_size
        windows[i] = data[start:end]
    
    logger.debug(f"Created {num_windows} sliding windows")
    return windows


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing production-ready data preprocessing utilities...")
    
    # Test state normalizer
    logger.info("Testing StateNormalizer...")
    data = np.random.randn(1000, 32).astype(np.float32)
    normalizer = StateNormalizer(normalization_type=NormalizationType.MINMAX)
    normalizer.fit(data)
    
    normalized = normalizer.normalize(data[:10])
    denormalized = normalizer.denormalize(normalized)
    
    logger.info(f"Original shape: {data[:10].shape}, Normalized: {normalized.shape}")
    logger.info(f"Denormalized match: {np.allclose(data[:10], denormalized, atol=1e-5)}")
    
    # Test data augmentation
    logger.info("Testing DataAugmentation...")
    augmented = DataAugmentation.add_gaussian_noise(data[:10], noise_std=0.1, seed=42)
    logger.info(f"Augmented shape: {augmented.shape}")
    
    # Test batch generator
    logger.info("Testing BatchGenerator...")
    batch_gen = BatchGenerator(
        states=data[:100],
        actions=np.random.randint(0, 8, 100),
        rewards=np.random.randn(100).astype(np.float32),
        next_states=data[:100],
        dones=np.zeros(100, dtype=np.float32),
        batch_size=32,
        seed=42
    )
    
    batch_count = 0
    for batch in batch_gen:
        batch_count += 1
    
    logger.info(f"Generated {batch_count} batches")
    
    # Test data splitting
    logger.info("Testing split_data...")
    train, val, test = split_data(data, train_ratio=0.8, val_ratio=0.1, random_seed=42)
    logger.info(f"Split sizes: train={len(train)}, val={len(val)}, test={len(test)}")
    
    logger.info("All tests completed successfully!")
