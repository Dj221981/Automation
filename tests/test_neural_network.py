"""
Unit tests for neural network module.

Tests cover:
- DQNNetwork model initialization and forward passes
- AgentLearningModel configuration validation
- Training step correctness
- Target network synchronization
- Epsilon decay behavior
- Model persistence (save/load)
- Experience replay buffer operations
- Error handling and edge cases
"""

import unittest
import numpy as np
import tempfile
import os
import tensorflow as tf

from src.models.neural_network import (
    DQNNetwork,
    AgentLearningModel,
    ExperienceReplay,
)


class TestDQNNetwork(unittest.TestCase):
    """Test DQNNetwork class."""

    def setUp(self):
        """Set up test fixtures."""
        self.state_size = 32
        self.action_size = 8
        tf.random.set_seed(42)
        np.random.seed(42)

    def test_initialization(self):
        """Test DQNNetwork initialization."""
        network = DQNNetwork(
            state_size=self.state_size,
            action_size=self.action_size,
            hidden_layers=[128, 64]
        )
        self.assertEqual(network.state_size, self.state_size)
        self.assertEqual(network.action_size, self.action_size)

    def test_invalid_state_size(self):
        """Test error on invalid state_size."""
        with self.assertRaises(ValueError):
            DQNNetwork(state_size=0, action_size=self.action_size)
        
        with self.assertRaises(ValueError):
            DQNNetwork(state_size=-1, action_size=self.action_size)

    def test_invalid_action_size(self):
        """Test error on invalid action_size."""
        with self.assertRaises(ValueError):
            DQNNetwork(state_size=self.state_size, action_size=0)
        
        with self.assertRaises(ValueError):
            DQNNetwork(state_size=self.state_size, action_size=-5)

    def test_invalid_hidden_layers(self):
        """Test error on invalid hidden_layers."""
        with self.assertRaises(ValueError):
            DQNNetwork(
                state_size=self.state_size,
                action_size=self.action_size,
                hidden_layers=[]
            )
        
        with self.assertRaises(ValueError):
            DQNNetwork(
                state_size=self.state_size,
                action_size=self.action_size,
                hidden_layers=[128, 0, 64]
            )

    def test_forward_pass(self):
        """Test forward pass through network."""
        network = DQNNetwork(
            state_size=self.state_size,
            action_size=self.action_size
        )
        
        states = tf.random.normal((16, self.state_size))
        q_values = network(states, training=False)
        
        self.assertEqual(q_values.shape, (16, self.action_size))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(q_values)))

    def test_batch_norm(self):
        """Test network with batch normalization."""
        network = DQNNetwork(
            state_size=self.state_size,
            action_size=self.action_size,
            use_batch_norm=True
        )
        
        states = tf.random.normal((16, self.state_size))
        q_values = network(states, training=True)
        
        self.assertEqual(q_values.shape, (16, self.action_size))

    def test_dropout(self):
        """Test network with dropout."""
        network = DQNNetwork(
            state_size=self.state_size,
            action_size=self.action_size,
            dropout_rate=0.3
        )
        
        states = tf.random.normal((16, self.state_size))
        q_values = network(states, training=True)
        
        self.assertEqual(q_values.shape, (16, self.action_size))


class TestAgentLearningModel(unittest.TestCase):
    """Test AgentLearningModel class."""

    def setUp(self):
        """Set up test fixtures."""
        self.state_size = 32
        self.action_size = 8
        tf.random.set_seed(42)
        np.random.seed(42)

    def test_initialization(self):
        """Test model initialization."""
        model = AgentLearningModel(
            state_size=self.state_size,
            action_size=self.action_size,
            seed=42
        )
        self.assertEqual(model.state_size, self.state_size)
        self.assertEqual(model.action_size, self.action_size)
        self.assertEqual(model.epsilon, 1.0)

    def test_invalid_state_size(self):
        """Test error on invalid state_size."""
        with self.assertRaises(ValueError):
            AgentLearningModel(state_size=0, action_size=self.action_size)

    def test_invalid_action_size(self):
        """Test error on invalid action_size."""
        with self.assertRaises(ValueError):
            AgentLearningModel(state_size=self.state_size, action_size=0)

    def test_invalid_learning_rate(self):
        """Test error on invalid learning_rate."""
        with self.assertRaises(ValueError):
            AgentLearningModel(
                state_size=self.state_size,
                action_size=self.action_size,
                learning_rate=0
            )

    def test_invalid_gamma(self):
        """Test error on invalid gamma."""
        with self.assertRaises(ValueError):
            AgentLearningModel(
                state_size=self.state_size,
                action_size=self.action_size,
                gamma=-0.1
            )
        
        with self.assertRaises(ValueError):
            AgentLearningModel(
                state_size=self.state_size,
                action_size=self.action_size,
                gamma=1.5
            )

    def test_invalid_epsilon(self):
        """Test error on invalid epsilon."""
        with self.assertRaises(ValueError):
            AgentLearningModel(
                state_size=self.state_size,
                action_size=self.action_size,
                epsilon=-0.1
            )

    def test_invalid_epsilon_decay(self):
        """Test error on invalid epsilon_decay."""
        with self.assertRaises(ValueError):
            AgentLearningModel(
                state_size=self.state_size,
                action_size=self.action_size,
                epsilon_decay=0
            )

    def test_invalid_epsilon_min(self):
        """Test error on epsilon_min > epsilon."""
        with self.assertRaises(ValueError):
            AgentLearningModel(
                state_size=self.state_size,
                action_size=self.action_size,
                epsilon=0.1,
                epsilon_min=0.5
            )

    def test_unsupported_model_type(self):
        """Test error on unsupported model_type."""
        with self.assertRaises(ValueError):
            AgentLearningModel(
                state_size=self.state_size,
                action_size=self.action_size,
                model_type="policy_gradient"
            )

    def test_unsupported_device(self):
        """Test error on unsupported device."""
        with self.assertRaises(ValueError):
            AgentLearningModel(
                state_size=self.state_size,
                action_size=self.action_size,
                device="tpu"
            )

    def test_select_action_training(self):
        """Test action selection during training."""
        model = AgentLearningModel(
            state_size=self.state_size,
            action_size=self.action_size,
            epsilon=0.5,
            seed=42
        )
        
        state = np.random.randn(self.state_size).astype(np.float32)
        action = model.select_action(state, training=True)
        
        self.assertIsInstance(action, (int, np.integer))
        self.assertTrue(0 <= action < self.action_size)

    def test_select_action_inference(self):
        """Test action selection during inference."""
        model = AgentLearningModel(
            state_size=self.state_size,
            action_size=self.action_size,
            seed=42
        )
        
        state = np.random.randn(self.state_size).astype(np.float32)
        action = model.select_action(state, training=False)
        
        self.assertIsInstance(action, (int, np.integer))
        self.assertTrue(0 <= action < self.action_size)

    def test_select_action_invalid_state(self):
        """Test error on invalid state."""
        model = AgentLearningModel(
            state_size=self.state_size,
            action_size=self.action_size
        )
        
        # Wrong shape
        bad_state = np.random.randn(self.state_size + 1).astype(np.float32)
        with self.assertRaises(ValueError):
            model.select_action(bad_state)
        
        # NaN values
        bad_state = np.random.randn(self.state_size).astype(np.float32)
        bad_state[0] = np.nan
        with self.assertRaises(ValueError):
            model.select_action(bad_state)

    def test_train_step_basic(self):
        """Test basic training step."""
        model = AgentLearningModel(
            state_size=self.state_size,
            action_size=self.action_size,
            seed=42
        )
        
        batch_size = 32
        states = np.random.randn(batch_size, self.state_size).astype(np.float32)
        actions = np.random.randint(0, self.action_size, batch_size)
        rewards = np.random.randn(batch_size).astype(np.float32)
        next_states = np.random.randn(batch_size, self.state_size).astype(np.float32)
        dones = np.zeros(batch_size, dtype=np.float32)
        
        loss = model.train_step(states, actions, rewards, next_states, dones)
        
        self.assertIsInstance(loss, float)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(loss > 0)

    def test_train_step_invalid_batch(self):
        """Test error on invalid training batch."""
        model = AgentLearningModel(
            state_size=self.state_size,
            action_size=self.action_size
        )
        
        # Empty batch
        with self.assertRaises(ValueError):
            model.train_step(
                np.array([], dtype=np.float32).reshape(0, self.state_size),
                np.array([], dtype=np.int32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32).reshape(0, self.state_size),
                np.array([], dtype=np.float32)
            )
        
        # Wrong state shape
        with self.assertRaises(ValueError):
            model.train_step(
                np.random.randn(32, self.state_size + 1).astype(np.float32),
                np.random.randint(0, self.action_size, 32),
                np.random.randn(32).astype(np.float32),
                np.random.randn(32, self.state_size).astype(np.float32),
                np.zeros(32, dtype=np.float32)
            )

    def test_update_target_network(self):
        """Test target network synchronization."""
        model = AgentLearningModel(
            state_size=self.state_size,
            action_size=self.action_size,
            seed=42
        )
        
        # Get initial weights
        online_weights = [w.numpy().copy() for w in model.network.trainable_weights]
        target_weights = [w.numpy().copy() for w in model.target_network.trainable_weights]
        
        # Initially they should be the same
        for ow, tw in zip(online_weights, target_weights):
            np.testing.assert_array_almost_equal(ow, tw)

    def test_decay_epsilon(self):
        """Test epsilon decay."""
        model = AgentLearningModel(
            state_size=self.state_size,
            action_size=self.action_size,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01
        )
        
        initial_epsilon = model.epsilon
        model.decay_epsilon()
        
        self.assertLess(model.epsilon, initial_epsilon)
        self.assertEqual(model.epsilon, 1.0 * 0.995)

    def test_decay_epsilon_respects_minimum(self):
        """Test that epsilon decay respects minimum value."""
        model = AgentLearningModel(
            state_size=self.state_size,
            action_size=self.action_size,
            epsilon=0.02,
            epsilon_decay=0.95,
            epsilon_min=0.01
        )
        
        model.decay_epsilon()
        self.assertEqual(model.epsilon, 0.01)

    def test_get_config(self):
        """Test model configuration retrieval."""
        model = AgentLearningModel(
            state_size=self.state_size,
            action_size=self.action_size,
            learning_rate=0.001,
            gamma=0.99
        )
        
        config = model.get_config()
        
        self.assertEqual(config["state_size"], self.state_size)
        self.assertEqual(config["action_size"], self.action_size)
        self.assertEqual(config["learning_rate"], 0.001)
        self.assertEqual(config["gamma"], 0.99)

    def test_get_model_summary(self):
        """Test model summary generation."""
        model = AgentLearningModel(
            state_size=self.state_size,
            action_size=self.action_size
        )
        
        summary = model.get_model_summary()
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)

    def test_save_and_load_model(self):
        """Test model saving and loading."""
        model = AgentLearningModel(
            state_size=self.state_size,
            action_size=self.action_size,
            seed=42
        )
        
        # Train a bit
        states = np.random.randn(32, self.state_size).astype(np.float32)
        actions = np.random.randint(0, self.action_size, 32)
        rewards = np.random.randn(32).astype(np.float32)
        next_states = np.random.randn(32, self.state_size).astype(np.float32)
        dones = np.zeros(32, dtype=np.float32)
        model.train_step(states, actions, rewards, next_states, dones)
        
        original_weights = [w.numpy().copy() for w in model.network.trainable_weights]
        original_train_steps = model.train_steps
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "model.weights.h5")
            model.save_model(filepath)
            self.assertTrue(os.path.exists(filepath))
            
            # Load into new model
            model2 = AgentLearningModel(
                state_size=self.state_size,
                action_size=self.action_size
            )
            model2.load_model(filepath)
            
            # Check weights match
            loaded_weights = [w.numpy() for w in model2.network.trainable_weights]
            for ow, lw in zip(original_weights, loaded_weights):
                np.testing.assert_array_almost_equal(ow, lw, decimal=5)
            
            # Check train_steps metadata
            self.assertEqual(model2.train_steps, original_train_steps)

    def test_save_invalid_filepath(self):
        """Test error on invalid filepath."""
        model = AgentLearningModel(
            state_size=self.state_size,
            action_size=self.action_size
        )
        
        with self.assertRaises(ValueError):
            model.save_model("")


class TestExperienceReplay(unittest.TestCase):
    """Test ExperienceReplay class."""

    def setUp(self):
        """Set up test fixtures."""
        self.state_size = 32
        self.max_size = 1000

    def test_initialization(self):
        """Test buffer initialization."""
        buffer = ExperienceReplay(state_size=self.state_size, max_size=self.max_size)
        self.assertEqual(len(buffer), 0)

    def test_invalid_state_size(self):
        """Test error on invalid state_size."""
        with self.assertRaises(ValueError):
            ExperienceReplay(state_size=0)
        
        with self.assertRaises(ValueError):
            ExperienceReplay(state_size=-1)

    def test_invalid_max_size(self):
        """Test error on invalid max_size."""
        with self.assertRaises(ValueError):
            ExperienceReplay(state_size=self.state_size, max_size=0)

    def test_add_experience(self):
        """Test adding experience to buffer."""
        buffer = ExperienceReplay(state_size=self.state_size)
        
        state = np.random.randn(self.state_size).astype(np.float32)
        action = 3
        reward = 1.0
        next_state = np.random.randn(self.state_size).astype(np.float32)
        done = False
        
        buffer.add(state, action, reward, next_state, done)
        self.assertEqual(len(buffer), 1)

    def test_add_invalid_state(self):
        """Test error on invalid state."""
        buffer = ExperienceReplay(state_size=self.state_size)
        
        # Wrong shape
        bad_state = np.random.randn(self.state_size + 1).astype(np.float32)
        next_state = np.random.randn(self.state_size).astype(np.float32)
        
        with self.assertRaises(ValueError):
            buffer.add(bad_state, 0, 1.0, next_state, False)

    def test_add_invalid_reward(self):
        """Test error on invalid reward."""
        buffer = ExperienceReplay(state_size=self.state_size)
        
        state = np.random.randn(self.state_size).astype(np.float32)
        next_state = np.random.randn(self.state_size).astype(np.float32)
        
        with self.assertRaises(ValueError):
            buffer.add(state, 0, np.nan, next_state, False)

    def test_buffer_overflow(self):
        """Test buffer overflow behavior."""
        buffer = ExperienceReplay(state_size=self.state_size, max_size=10)
        
        for i in range(20):
            state = np.random.randn(self.state_size).astype(np.float32)
            next_state = np.random.randn(self.state_size).astype(np.float32)
            buffer.add(state, i % 8, float(i), next_state, i % 2 == 0)
        
        # Buffer should not exceed max_size
        self.assertEqual(len(buffer), 10)

    def test_sample_basic(self):
        """Test sampling from buffer."""
        buffer = ExperienceReplay(state_size=self.state_size)
        
        # Add experiences
        for i in range(50):
            state = np.random.randn(self.state_size).astype(np.float32)
            next_state = np.random.randn(self.state_size).astype(np.float32)
            buffer.add(state, i % 8, float(i), next_state, i % 2 == 0)
        
        states, actions, rewards, next_states, dones = buffer.sample(32)
        
        self.assertEqual(states.shape, (32, self.state_size))
        self.assertEqual(actions.shape, (32,))
        self.assertEqual(rewards.shape, (32,))
        self.assertEqual(next_states.shape, (32, self.state_size))
        self.assertEqual(dones.shape, (32,))

    def test_sample_invalid_batch_size(self):
        """Test error on invalid batch_size."""
        buffer = ExperienceReplay(state_size=self.state_size)
        
        with self.assertRaises(ValueError):
            buffer.sample(0)

    def test_sample_empty_buffer(self):
        """Test error on sampling from empty buffer."""
        buffer = ExperienceReplay(state_size=self.state_size)
        
        with self.assertRaises(ValueError):
            buffer.sample(32)

    def test_sample_smaller_than_buffer(self):
        """Test sampling when batch_size < buffer size."""
        buffer = ExperienceReplay(state_size=self.state_size)
        
        for i in range(100):
            state = np.random.randn(self.state_size).astype(np.float32)
            next_state = np.random.randn(self.state_size).astype(np.float32)
            buffer.add(state, i % 8, float(i), next_state, False)
        
        # Sample smaller batch
        states, actions, rewards, next_states, dones = buffer.sample(10)
        
        self.assertEqual(states.shape[0], 10)


if __name__ == "__main__":
    unittest.main()
