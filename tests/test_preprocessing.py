"""
Unit tests for data preprocessing module.

Tests cover:
- Input validation and error handling
- Normalization correctness (all types)
- Data augmentation operations
- Batch generation and data splitting
- File I/O operations
- Edge cases and production scenarios
"""

import unittest
import numpy as np
import tempfile
import os
from pathlib import Path

from src.data.preprocessing import (
    NormalizationType,
    NormalizationStats,
    StateNormalizer,
    DataAugmentation,
    ExperiencePreprocessor,
    BatchGenerator,
    split_data,
    create_sliding_window,
)


class TestNormalizationStats(unittest.TestCase):
    """Test NormalizationStats dataclass."""

    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        stats = NormalizationStats(
            min_val=np.array([0.0, -1.0]),
            max_val=np.array([1.0, 1.0]),
            mean=np.array([0.5, 0.0]),
            std=np.array([0.3, 0.5]),
            median=np.array([0.5, 0.0]),
            q25=np.array([0.25, -0.5]),
            q75=np.array([0.75, 0.5])
        )
        
        dict_data = stats.to_dict()
        recovered_stats = NormalizationStats.from_dict(dict_data)
        
        for key in ["min_val", "max_val", "mean", "std", "median", "q25", "q75"]:
            np.testing.assert_array_almost_equal(
                getattr(stats, key), getattr(recovered_stats, key)
            )

    def test_from_dict_missing_keys(self):
        """Test error handling for missing keys."""
        incomplete_dict = {"min": [0], "max": [1]}
        with self.assertRaises(ValueError):
            NormalizationStats.from_dict(incomplete_dict)


class TestStateNormalizer(unittest.TestCase):
    """Test StateNormalizer class."""

    def setUp(self):
        """Set up test data."""
        self.data = np.random.randn(1000, 10).astype(np.float32)

    def test_initialization(self):
        """Test normalizer initialization."""
        normalizer = StateNormalizer(normalization_type=NormalizationType.MINMAX)
        self.assertFalse(normalizer.fitted)
        self.assertEqual(normalizer.normalization_type, NormalizationType.MINMAX)

    def test_invalid_epsilon(self):
        """Test error on invalid epsilon."""
        with self.assertRaises(ValueError):
            StateNormalizer(epsilon=0)
        with self.assertRaises(ValueError):
            StateNormalizer(epsilon=-0.1)

    def test_fit_validation(self):
        """Test fit() input validation."""
        normalizer = StateNormalizer()
        
        # Empty data
        with self.assertRaises(ValueError):
            normalizer.fit(np.array([]))
        
        # 1D data
        with self.assertRaises(ValueError):
            normalizer.fit(np.array([1, 2, 3]))
        
        # Data with NaN
        bad_data = self.data.copy()
        bad_data[0, 0] = np.nan
        with self.assertRaises(ValueError):
            normalizer.fit(bad_data)
        
        # Data with inf
        bad_data = self.data.copy()
        bad_data[0, 0] = np.inf
        with self.assertRaises(ValueError):
            normalizer.fit(bad_data)

    def test_fit_success(self):
        """Test successful fit."""
        normalizer = StateNormalizer()
        normalizer.fit(self.data)
        self.assertTrue(normalizer.fitted)
        self.assertEqual(normalizer.feature_size, 10)

    def test_normalize_minmax(self):
        """Test MINMAX normalization."""
        normalizer = StateNormalizer(normalization_type=NormalizationType.MINMAX)
        normalizer.fit(self.data)
        
        normalized = normalizer.normalize(self.data)
        # Check that values are roughly in [0, 1] range (with some tolerance)
        self.assertTrue(np.all(normalized >= -0.1))
        self.assertTrue(np.all(normalized <= 1.1))

    def test_normalize_zscore(self):
        """Test Z-score normalization."""
        normalizer = StateNormalizer(normalization_type=NormalizationType.ZSCORE)
        normalizer.fit(self.data)
        
        normalized = normalizer.normalize(self.data)
        # Check that mean is near 0 and std is near 1
        self.assertAlmostEqual(np.mean(normalized), 0, places=1)
        self.assertAlmostEqual(np.std(normalized), 1, places=1)

    def test_normalize_robust(self):
        """Test robust normalization."""
        normalizer = StateNormalizer(normalization_type=NormalizationType.ROBUST)
        normalizer.fit(self.data)
        
        normalized = normalizer.normalize(self.data)
        self.assertTrue(np.all(np.isfinite(normalized)))

    def test_normalize_none(self):
        """Test NONE normalization (identity)."""
        normalizer = StateNormalizer(normalization_type=NormalizationType.NONE)
        normalizer.fit(self.data)
        
        normalized = normalizer.normalize(self.data)
        np.testing.assert_array_almost_equal(normalized, self.data)

    def test_denormalize(self):
        """Test denormalization."""
        for norm_type in [NormalizationType.MINMAX, NormalizationType.ZSCORE, NormalizationType.ROBUST]:
            normalizer = StateNormalizer(normalization_type=norm_type)
            normalizer.fit(self.data)
            
            normalized = normalizer.normalize(self.data)
            denormalized = normalizer.denormalize(normalized)
            
            np.testing.assert_array_almost_equal(denormalized, self.data, decimal=5)

    def test_normalize_without_fit(self):
        """Test error when normalizing without fit."""
        normalizer = StateNormalizer()
        with self.assertRaises(ValueError):
            normalizer.normalize(self.data)

    def test_normalize_shape_mismatch(self):
        """Test error on shape mismatch."""
        normalizer = StateNormalizer()
        normalizer.fit(self.data)
        
        wrong_shape_data = np.random.randn(5, 20).astype(np.float32)
        with self.assertRaises(ValueError):
            normalizer.normalize(wrong_shape_data)

    def test_normalize_with_nans(self):
        """Test error on NaN in input."""
        normalizer = StateNormalizer()
        normalizer.fit(self.data)
        
        bad_data = self.data.copy()
        bad_data[0, 0] = np.nan
        with self.assertRaises(ValueError):
            normalizer.normalize(bad_data)

    def test_save_and_load(self):
        """Test save and load operations."""
        normalizer = StateNormalizer(normalization_type=NormalizationType.MINMAX)
        normalizer.fit(self.data)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "normalizer.json")
            normalizer.save(filepath)
            self.assertTrue(os.path.exists(filepath))
            
            # Load into new normalizer
            normalizer2 = StateNormalizer()
            normalizer2.load(filepath)
            
            # Test that loaded normalizer works the same
            norm1 = normalizer.normalize(self.data[:10])
            norm2 = normalizer2.normalize(self.data[:10])
            np.testing.assert_array_almost_equal(norm1, norm2)

    def test_save_unfitted_raises_error(self):
        """Test error when saving unfitted normalizer."""
        normalizer = StateNormalizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "normalizer.json")
            with self.assertRaises(ValueError):
                normalizer.save(filepath)

    def test_load_nonexistent_file(self):
        """Test error on loading nonexistent file."""
        normalizer = StateNormalizer()
        with self.assertRaises(FileNotFoundError):
            normalizer.load("/nonexistent/path/to/file.json")


class TestDataAugmentation(unittest.TestCase):
    """Test DataAugmentation class."""

    def setUp(self):
        """Set up test data."""
        self.data = np.random.randn(100, 10).astype(np.float32)

    def test_gaussian_noise_valid(self):
        """Test Gaussian noise augmentation."""
        augmented = DataAugmentation.add_gaussian_noise(self.data, noise_std=0.1, seed=42)
        self.assertEqual(augmented.shape, self.data.shape)
        self.assertTrue(np.all(np.isfinite(augmented)))
        # Data should be different
        self.assertFalse(np.allclose(augmented, self.data))

    def test_gaussian_noise_zero_std(self):
        """Test Gaussian noise with zero std."""
        augmented = DataAugmentation.add_gaussian_noise(self.data, noise_std=0.0, seed=42)
        np.testing.assert_array_almost_equal(augmented, self.data)

    def test_gaussian_noise_invalid_std(self):
        """Test error on negative noise_std."""
        with self.assertRaises(ValueError):
            DataAugmentation.add_gaussian_noise(self.data, noise_std=-0.1)

    def test_mixup(self):
        """Test mixup augmentation."""
        x1 = np.random.randn(10, 5).astype(np.float32)
        x2 = np.random.randn(10, 5).astype(np.float32)
        y1 = np.random.randn(10, 2).astype(np.float32)
        y2 = np.random.randn(10, 2).astype(np.float32)
        
        x_mixed, y_mixed = DataAugmentation.mixup(x1, x2, y1, y2, alpha=0.2, seed=42)
        
        self.assertEqual(x_mixed.shape, x1.shape)
        self.assertEqual(y_mixed.shape, y1.shape)
        self.assertTrue(np.all(np.isfinite(x_mixed)))
        self.assertTrue(np.all(np.isfinite(y_mixed)))

    def test_mixup_shape_mismatch(self):
        """Test error on shape mismatch in mixup."""
        x1 = np.random.randn(10, 5).astype(np.float32)
        x2 = np.random.randn(10, 6).astype(np.float32)
        y1 = np.random.randn(10, 2).astype(np.float32)
        y2 = np.random.randn(10, 2).astype(np.float32)
        
        with self.assertRaises(ValueError):
            DataAugmentation.mixup(x1, x2, y1, y2)

    def test_mixup_invalid_alpha(self):
        """Test error on invalid alpha."""
        x1 = np.random.randn(10, 5).astype(np.float32)
        x2 = np.random.randn(10, 5).astype(np.float32)
        y1 = np.random.randn(10, 2).astype(np.float32)
        y2 = np.random.randn(10, 2).astype(np.float32)
        
        with self.assertRaises(ValueError):
            DataAugmentation.mixup(x1, x2, y1, y2, alpha=0)

    def test_random_crop(self):
        """Test random crop augmentation."""
        cropped = DataAugmentation.random_crop(self.data, crop_fraction=0.8, seed=42)
        self.assertEqual(cropped.shape, self.data.shape)
        # Some features should be zeroed
        self.assertTrue(np.any(cropped == 0))

    def test_random_crop_invalid_fraction(self):
        """Test error on invalid crop_fraction."""
        with self.assertRaises(ValueError):
            DataAugmentation.random_crop(self.data, crop_fraction=0)
        with self.assertRaises(ValueError):
            DataAugmentation.random_crop(self.data, crop_fraction=1.5)

    def test_temporal_shift(self):
        """Test temporal shift augmentation."""
        sequences = np.random.randn(10, 20, 5).astype(np.float32)
        shifted = DataAugmentation.temporal_shift(sequences, shift_range=2, seed=42)
        self.assertEqual(shifted.shape, sequences.shape)
        self.assertTrue(np.all(np.isfinite(shifted)))

    def test_temporal_shift_invalid_range(self):
        """Test error on invalid shift_range."""
        sequences = np.random.randn(10, 20, 5).astype(np.float32)
        with self.assertRaises(ValueError):
            DataAugmentation.temporal_shift(sequences, shift_range=-1)


class TestExperiencePreprocessor(unittest.TestCase):
    """Test ExperiencePreprocessor class."""

    def setUp(self):
        """Set up test data."""
        self.data = np.random.randn(100, 10).astype(np.float32)
        self.normalizer = StateNormalizer()
        self.normalizer.fit(self.data)

    def test_process_state(self):
        """Test state processing."""
        preprocessor = ExperiencePreprocessor(state_normalizer=self.normalizer)
        state = self.data[0]
        processed = preprocessor.process_state(state)
        self.assertEqual(processed.shape, state.shape)
        self.assertTrue(np.all(np.isfinite(processed)))

    def test_process_reward(self):
        """Test reward processing."""
        preprocessor = ExperiencePreprocessor()
        reward = 1.5
        processed = preprocessor.process_reward(reward)
        self.assertIsInstance(processed, float)

    def test_process_batch(self):
        """Test batch processing."""
        preprocessor = ExperiencePreprocessor(state_normalizer=self.normalizer)
        states = self.data[:10]
        actions = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32)
        rewards = np.random.randn(10).astype(np.float32)
        next_states = self.data[:10]
        dones = np.zeros(10, dtype=np.float32)
        
        result = preprocessor.process_batch(states, actions, rewards, next_states, dones)
        processed_states, processed_actions, processed_rewards, processed_next_states, processed_dones = result
        
        self.assertEqual(processed_states.shape, states.shape)
        self.assertTrue(np.all(np.isfinite(processed_states)))

    def test_process_batch_shape_mismatch(self):
        """Test error on shape mismatch in process_batch."""
        preprocessor = ExperiencePreprocessor()
        states = np.random.randn(10, 5).astype(np.float32)
        actions = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)  # Wrong size
        rewards = np.random.randn(10).astype(np.float32)
        next_states = np.random.randn(10, 5).astype(np.float32)
        dones = np.zeros(10, dtype=np.float32)
        
        with self.assertRaises(ValueError):
            preprocessor.process_batch(states, actions, rewards, next_states, dones)


class TestBatchGenerator(unittest.TestCase):
    """Test BatchGenerator class."""

    def setUp(self):
        """Set up test data."""
        self.states = np.random.randn(100, 10).astype(np.float32)
        self.actions = np.random.randint(0, 8, 100)
        self.rewards = np.random.randn(100).astype(np.float32)
        self.next_states = np.random.randn(100, 10).astype(np.float32)
        self.dones = np.zeros(100, dtype=np.float32)

    def test_initialization(self):
        """Test batch generator initialization."""
        gen = BatchGenerator(
            self.states, self.actions, self.rewards, self.next_states, self.dones,
            batch_size=32
        )
        self.assertEqual(len(gen), int(np.ceil(100 / 32)))

    def test_iteration(self):
        """Test batch generation."""
        gen = BatchGenerator(
            self.states, self.actions, self.rewards, self.next_states, self.dones,
            batch_size=32, shuffle=False
        )
        
        batch_count = 0
        total_samples = 0
        for states, actions, rewards, next_states, dones in gen:
            batch_count += 1
            total_samples += len(states)
            self.assertTrue(np.all(np.isfinite(states)))
        
        self.assertEqual(total_samples, 100)
        self.assertEqual(batch_count, len(gen))

    def test_invalid_batch_size(self):
        """Test error on invalid batch_size."""
        with self.assertRaises(ValueError):
            BatchGenerator(
                self.states, self.actions, self.rewards, self.next_states, self.dones,
                batch_size=0
            )

    def test_empty_data(self):
        """Test error on empty data."""
        empty_states = np.array([], dtype=np.float32).reshape(0, 10)
        empty_actions = np.array([], dtype=np.int32)
        empty_rewards = np.array([], dtype=np.float32)
        empty_next_states = np.array([], dtype=np.float32).reshape(0, 10)
        empty_dones = np.array([], dtype=np.float32)
        
        with self.assertRaises(ValueError):
            BatchGenerator(
                empty_states, empty_actions, empty_rewards, empty_next_states, empty_dones
            )


class TestSplitData(unittest.TestCase):
    """Test split_data function."""

    def setUp(self):
        """Set up test data."""
        self.data = np.random.randn(1000, 10).astype(np.float32)

    def test_split_valid(self):
        """Test valid data split."""
        train, val, test = split_data(self.data, train_ratio=0.8, val_ratio=0.1, random_seed=42)
        
        total = len(train) + len(val) + len(test)
        self.assertEqual(total, 1000)
        self.assertEqual(len(train), 800)
        self.assertEqual(len(val), 100)
        self.assertEqual(len(test), 100)

    def test_split_no_test_set(self):
        """Test split with no test set."""
        train, val, test = split_data(self.data, train_ratio=0.9, val_ratio=0.1)
        
        total = len(train) + len(val) + len(test)
        self.assertEqual(total, 1000)
        self.assertEqual(len(test), 0)

    def test_split_invalid_ratios(self):
        """Test error on invalid ratios."""
        with self.assertRaises(ValueError):
            split_data(self.data, train_ratio=0)  # Must be > 0
        
        with self.assertRaises(ValueError):
            split_data(self.data, train_ratio=1.0)  # Must be < 1
        
        with self.assertRaises(ValueError):
            split_data(self.data, train_ratio=0.6, val_ratio=0.5)  # Sum must be < 1

    def test_split_empty_data(self):
        """Test error on empty data."""
        empty = np.array([], dtype=np.float32).reshape(0, 10)
        with self.assertRaises(ValueError):
            split_data(empty)


class TestSlidingWindow(unittest.TestCase):
    """Test create_sliding_window function."""

    def setUp(self):
        """Set up test data."""
        self.data = np.random.randn(100, 5).astype(np.float32)

    def test_sliding_window_valid(self):
        """Test valid sliding window creation."""
        windows = create_sliding_window(self.data, window_size=10, step=5)
        
        expected_num = (100 - 10) // 5 + 1
        self.assertEqual(windows.shape[0], expected_num)
        self.assertEqual(windows.shape[1], 10)
        self.assertEqual(windows.shape[2], 5)

    def test_sliding_window_step_1(self):
        """Test sliding window with step 1."""
        windows = create_sliding_window(self.data, window_size=10, step=1)
        
        expected_num = 100 - 10 + 1
        self.assertEqual(windows.shape[0], expected_num)

    def test_sliding_window_invalid_window_size(self):
        """Test error on invalid window_size."""
        with self.assertRaises(ValueError):
            create_sliding_window(self.data, window_size=0)
        
        with self.assertRaises(ValueError):
            create_sliding_window(self.data, window_size=150)  # Larger than data

    def test_sliding_window_invalid_step(self):
        """Test error on invalid step."""
        with self.assertRaises(ValueError):
            create_sliding_window(self.data, window_size=10, step=0)
        
        with self.assertRaises(ValueError):
            create_sliding_window(self.data, window_size=10, step=-1)


if __name__ == "__main__":
    unittest.main()
