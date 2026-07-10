"""
Data processing utilities for AI-morphasis neural network models.

Handles data loading, preprocessing, normalization, and augmentation
for agent training and inference.
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, List, Dict, Any, Union
from pathlib import Path
import json
import logging
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
    def from_dict(cls, data: Dict[str, Any]) -> "NormalizationStats":
        """Create from dictionary."""
        return cls(
            min_val=np.array(data["min"]),
            max_val=np.array(data["max"]),
            mean=np.array(data["mean"]),
            std=np.array(data["std"]),
            median=np.array(data["median"]),
            q25=np.array(data["q25"]),
            q75=np.array(data["q75"])
        )


class StateNormalizer:
    """Normalizes state data for neural network input."""

    def __init__(
        self,
        normalization_type: NormalizationType = NormalizationType.MINMAX,
        epsilon: float = 1e-8
    ):
        """
        Initialize state normalizer.

        Args:
            normalization_type: Type of normalization to apply
            epsilon: Small value to prevent division by zero
        """
        self.normalization_type = normalization_type
        self.epsilon = epsilon
        self.stats: Optional[NormalizationStats] = None
        self.fitted = False

    def fit(self, data: np.ndarray) -> None:
        """
        Fit normalizer on data.

        Args:
            data: Data to compute statistics from [samples, features]
        """
        data = np.asarray(data)
        
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
        logger.info(f"Normalizer fitted with {data.shape[0]} samples")

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data.

        Args:
            data: Data to normalize

        Returns:
            Normalized data
        """
        if not self.fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        data = np.asarray(data, dtype=np.float32)
        
        if self.normalization_type == NormalizationType.NONE:
            return data
        
        elif self.normalization_type == NormalizationType.MINMAX:
            range_val = self.stats.max_val - self.stats.min_val
            range_val = np.where(range_val == 0, self.epsilon, range_val)
            return (data - self.stats.min_val) / range_val
        
        elif self.normalization_type == NormalizationType.ZSCORE:
            return (data - self.stats.mean) / (self.stats.std + self.epsilon)
        
        elif self.normalization_type == NormalizationType.ROBUST:
            iqr = self.stats.q75 - self.stats.q25
            iqr = np.where(iqr == 0, self.epsilon, iqr)
            return (data - self.stats.median) / iqr
        
        else:
            raise ValueError(f"Unknown normalization type: {self.normalization_type}")

    def denormalize(self, data: np.ndarray) -> np.ndarray:
        """
        Reverse normalization.

        Args:
            data: Normalized data

        Returns:
            Original scale data
        """
        if not self.fitted:
            raise ValueError("Normalizer not fitted.")
        
        data = np.asarray(data, dtype=np.float32)
        
        if self.normalization_type == NormalizationType.NONE:
            return data
        
        elif self.normalization_type == NormalizationType.MINMAX:
            range_val = self.stats.max_val - self.stats.min_val
            return data * range_val + self.stats.min_val
        
        elif self.normalization_type == NormalizationType.ZSCORE:
            return data * self.stats.std + self.stats.mean
        
        elif self.normalization_type == NormalizationType.ROBUST:
            iqr = self.stats.q75 - self.stats.q25
            return data * iqr + self.stats.median
        
        else:
            raise ValueError(f"Unknown normalization type: {self.normalization_type}")

    def save(self, filepath: str) -> None:
        """Save normalizer statistics."""
        if not self.fitted:
            raise ValueError("Normalizer not fitted.")
        
        data = {
            "type": self.normalization_type.value,
            "stats": self.stats.to_dict()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Normalizer saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Load normalizer statistics."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.normalization_type = NormalizationType(data["type"])
        self.stats = NormalizationStats.from_dict(data["stats"])
        self.fitted = True
        logger.info(f"Normalizer loaded from {filepath}")


class DataAugmentation:
    """Data augmentation utilities for training."""

    @staticmethod
    def add_gaussian_noise(
        data: np.ndarray,
        noise_std: float = 0.1
    ) -> np.ndarray:
        """
        Add Gaussian noise to data.

        Args:
            data: Input data
            noise_std: Standard deviation of noise

        Returns:
            Data with added noise
        """
        noise = np.random.normal(0, noise_std, data.shape)
        return data + noise

    @staticmethod
    def mixup(
        x1: np.ndarray,
        x2: np.ndarray,
        y1: np.ndarray,
        y2: np.ndarray,
        alpha: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mixup data augmentation.

        Args:
            x1: First batch of features
            x2: Second batch of features
            y1: First batch of labels/rewards
            y2: Second batch of labels/rewards
            alpha: Beta parameter for mixup

        Returns:
            Tuple of (mixed_x, mixed_y)
        """
        lam = np.random.beta(alpha, alpha)
        x_mixed = lam * x1 + (1 - lam) * x2
        y_mixed = lam * y1 + (1 - lam) * y2
        return x_mixed, y_mixed

    @staticmethod
    def random_crop(
        data: np.ndarray,
        crop_fraction: float = 0.9
    ) -> np.ndarray:
        """
        Random feature dropout (crop).

        Args:
            data: Input data [batch, features]
            crop_fraction: Fraction of features to keep

        Returns:
            Data with random features zeroed
        """
        augmented = data.copy()
        num_features = data.shape[-1]
        num_drop = int(num_features * (1 - crop_fraction))
        
        drop_idx = np.random.choice(num_features, num_drop, replace=False)
        augmented[..., drop_idx] = 0
        
        return augmented

    @staticmethod
    def temporal_shift(
        sequences: np.ndarray,
        shift_range: int = 2
    ) -> np.ndarray:
        """
        Temporal shift for sequence data.

        Args:
            sequences: Sequence data [batch, time, features]
            shift_range: Maximum shift amount

        Returns:
            Shifted sequences
        """
        shift = np.random.randint(-shift_range, shift_range + 1)
        return np.roll(sequences, shift, axis=1)


class ExperiencePreprocessor:
    """Preprocesses experience data for training."""

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
        """
        self.state_normalizer = state_normalizer
        self.reward_normalizer = reward_normalizer

    def process_state(self, state: np.ndarray) -> np.ndarray:
        """
        Process state.

        Args:
            state: Raw state

        Returns:
            Processed state
        """
        state = np.asarray(state, dtype=np.float32)
        
        if self.state_normalizer and self.state_normalizer.fitted:
            state = self.state_normalizer.normalize(state)
        
        return state

    def process_reward(self, reward: float) -> float:
        """
        Process reward.

        Args:
            reward: Raw reward

        Returns:
            Processed reward
        """
        reward = np.asarray([reward], dtype=np.float32)
        
        if self.reward_normalizer and self.reward_normalizer.fitted:
            reward = self.reward_normalizer.normalize(reward)
        
        return float(reward[0])

    def process_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Process batch of experience.

        Args:
            states: State batch
            actions: Action batch
            rewards: Reward batch
            next_states: Next state batch
            dones: Done flags

        Returns:
            Processed batch tuple
        """
        states = np.asarray(states, dtype=np.float32)
        next_states = np.asarray(next_states, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)
        
        if self.state_normalizer and self.state_normalizer.fitted:
            states = self.state_normalizer.normalize(states)
            next_states = self.state_normalizer.normalize(next_states)
        
        if self.reward_normalizer and self.reward_normalizer.fitted:
            rewards = self.reward_normalizer.normalize(rewards.reshape(-1, 1)).flatten()
        
        return states, actions, rewards, next_states, dones


class BatchGenerator:
    """Generates batches from experience data."""

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True
    ):
        """
        Initialize batch generator.

        Args:
            states: State array
            actions: Action array
            rewards: Reward array
            next_states: Next state array
            dones: Done flags
            batch_size: Batch size
            shuffle: Whether to shuffle data
        """
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.dones = dones
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(states)
        self.indices = np.arange(self.num_samples)
        
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        self.current_idx = 0

    def __iter__(self):
        """Iterate through batches."""
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get next batch."""
        if self.current_idx >= self.num_samples:
            if self.shuffle:
                np.random.shuffle(self.indices)
            self.current_idx = 0
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
    Split data into train/validation/test sets.

    Args:
        data: Data to split
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        shuffle: Whether to shuffle data
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
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
        window_size: Window size
        step: Step size

    Returns:
        Windowed data [num_windows, window_size, features]
    """
    num_windows = (data.shape[0] - window_size) // step + 1
    windows = np.zeros((num_windows, window_size, data.shape[1]))
    
    for i in range(num_windows):
        start = i * step
        end = start + window_size
        windows[i] = data[start:end]
    
    return windows


if __name__ == "__main__":
    logger.info("Testing data preprocessing utilities...")
    
    # Test state normalizer
    logger.info("Testing StateNormalizer...")
    data = np.random.randn(1000, 32)
    normalizer = StateNormalizer(normalization_type=NormalizationType.MINMAX)
    normalizer.fit(data)
    
    normalized = normalizer.normalize(data[:10])
    denormalized = normalizer.denormalize(normalized)
    
    logger.info(f"Original shape: {data[:10].shape}, Normalized: {normalized.shape}")
    logger.info(f"Denormalized match: {np.allclose(data[:10], denormalized)}")
    
    # Test data augmentation
    logger.info("Testing DataAugmentation...")
    augmented = DataAugmentation.add_gaussian_noise(data[:10], noise_std=0.1)
    logger.info(f"Augmented shape: {augmented.shape}")
    
    # Test batch generator
    logger.info("Testing BatchGenerator...")
    batch_gen = BatchGenerator(
        states=data[:100],
        actions=np.random.randint(0, 8, 100),
        rewards=np.random.randn(100),
        next_states=data[:100],
        dones=np.zeros(100),
        batch_size=32
    )
    
    batch_count = 0
    for batch in batch_gen:
        batch_count += 1
    
    logger.info(f"Generated {batch_count} batches")
    logger.info("Testing completed!")
