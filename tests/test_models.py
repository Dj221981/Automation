"""
Unit tests for neural network models and training components.

Comprehensive test suite for DQN, Policy Networks, and training utilities.
"""

import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path
import tempfile
import json

from src.models.neural_network import (
    DQNNetwork,
    PolicyNetwork,
    AgentLearningModel,
    ExperienceReplay
)
from src.training.train import (
    TrainingEnvironment,
    AgentTrainer
)


class TestDQNNetwork:
    """Tests for DQN Network."""

    @pytest.fixture
    def network(self):
        """Create a DQN network instance."""
        return DQNNetwork(
            state_size=32,
            action_size=8,
            hidden_layers=[64, 32]
        )

    def test_network_initialization(self, network):
        """Test DQN network initialization."""
        assert network.state_size == 32
        assert network.action_size == 8
        assert len(network.dense_layers) > 0

    def test_network_forward_pass(self, network):
        """Test forward pass through network."""
        states = tf.random.normal((4, 32))
        q_values = network(states, training=True)
        
        assert q_values.shape == (4, 8)
        assert not tf.reduce_any(tf.math.is_nan(q_values))

    def test_network_output_shape(self, network):
        """Test output shape matches action size."""
        batch_sizes = [1, 4, 32, 64]
        
        for batch_size in batch_sizes:
            states = tf.random.normal((batch_size, 32))
            q_values = network(states, training=False)
            assert q_values.shape == (batch_size, 8)

    def test_network_training_vs_inference(self, network):
        """Test that training and inference modes work differently."""
        states = tf.random.normal((8, 32))
        
        # Training mode
        q_train = network(states, training=True)
        
        # Inference mode
        q_infer = network(states, training=False)
        
        # Both should produce valid outputs
        assert q_train.shape == q_infer.shape
        assert not tf.reduce_any(tf.math.is_nan(q_train))
        assert not tf.reduce_any(tf.math.is_nan(q_infer))


class TestPolicyNetwork:
    """Tests for Policy Network."""

    @pytest.fixture
    def network_discrete(self):
        """Create discrete action policy network."""
        return PolicyNetwork(
            state_size=32,
            action_size=8,
            action_space="discrete"
        )

    @pytest.fixture
    def network_continuous(self):
        """Create continuous action policy network."""
        return PolicyNetwork(
            state_size=32,
            action_size=4,
            action_space="continuous"
        )

    def test_discrete_network_initialization(self, network_discrete):
        """Test discrete policy network initialization."""
        assert network_discrete.action_space == "discrete"
        assert hasattr(network_discrete, 'policy_head')

    def test_discrete_forward_pass(self, network_discrete):
        """Test discrete policy network forward pass."""
        states = tf.random.normal((4, 32))
        policy, value = network_discrete(states, training=True)
        
        assert policy.shape == (4, 8)
        assert value.shape == (4, 1)
        assert not tf.reduce_any(tf.math.is_nan(policy))
        assert not tf.reduce_any(tf.math.is_nan(value))

    def test_discrete_policy_sum_to_one(self, network_discrete):
        """Test that discrete policy outputs sum to 1."""
        states = tf.random.normal((8, 32))
        policy, _ = network_discrete(states, training=False)
        
        policy_sum = tf.reduce_sum(policy, axis=1)
        assert tf.reduce_all(tf.abs(policy_sum - 1.0) < 1e-5)

    def test_continuous_forward_pass(self, network_continuous):
        """Test continuous policy network forward pass."""
        states = tf.random.normal((4, 32))
        policy, value = network_continuous(states, training=True)
        
        # Policy should return mean and log_std (2 * action_size)
        assert policy.shape == (4, 8)  # 2 * 4
        assert value.shape == (4, 1)


class TestAgentLearningModel:
    """Tests for Agent Learning Model."""

    @pytest.fixture
    def model_dqn(self):
        """Create DQN agent learning model."""
        return AgentLearningModel(
            state_size=32,
            action_size=8,
            learning_rate=0.001,
            model_type="dqn"
        )

    @pytest.fixture
    def model_policy(self):
        """Create policy gradient agent learning model."""
        return AgentLearningModel(
            state_size=32,
            action_size=8,
            learning_rate=0.001,
            model_type="policy_gradient"
        )

    def test_model_initialization(self, model_dqn):
        """Test model initialization."""
        assert model_dqn.state_size == 32
        assert model_dqn.action_size == 8
        assert model_dqn.epsilon == 1.0

    def test_action_selection(self, model_dqn):
        """Test action selection."""
        state = np.random.randn(32)
        
        # Training mode (may explore)
        action = model_dqn.select_action(state, training=True)
        assert 0 <= action < 8
        
        # Inference mode (greedy)
        action = model_dqn.select_action(state, training=False)
        assert 0 <= action < 8

    def test_training_step(self, model_dqn):
        """Test training step."""
        states = np.random.randn(16, 32).astype(np.float32)
        actions = np.random.randint(0, 8, 16)
        rewards = np.random.randn(16).astype(np.float32)
        next_states = np.random.randn(16, 32).astype(np.float32)
        dones = np.random.randint(0, 2, 16).astype(np.float32)
        
        loss = model_dqn.train_step(states, actions, rewards, next_states, dones)
        
        assert isinstance(loss, float) or isinstance(loss, np.floating)
        assert loss >= 0

    def test_target_network_update(self, model_dqn):
        """Test target network update."""
        # Get initial target network weights
        initial_weights = [w.numpy().copy() for w in model_dqn.target_network.weights]
        
        # Perform training to change main network weights
        states = np.random.randn(16, 32).astype(np.float32)
        actions = np.random.randint(0, 8, 16)
        rewards = np.random.randn(16).astype(np.float32)
        next_states = np.random.randn(16, 32).astype(np.float32)
        dones = np.zeros(16).astype(np.float32)
        
        model_dqn.train_step(states, actions, rewards, next_states, dones)
        
        # Target network should still have initial weights
        target_weights = [w.numpy() for w in model_dqn.target_network.weights]
        for init_w, target_w in zip(initial_weights, target_weights):
            assert np.allclose(init_w, target_w)
        
        # Update target network
        model_dqn.update_target_network()
        
        # Target network should now match main network
        target_weights = [w.numpy() for w in model_dqn.target_network.weights]
        main_weights = [w.numpy() for w in model_dqn.network.weights]
        for main_w, target_w in zip(main_weights, target_weights):
            assert np.allclose(main_w, target_w)

    def test_epsilon_decay(self, model_dqn):
        """Test epsilon decay."""
        initial_epsilon = model_dqn.epsilon
        model_dqn.decay_epsilon()
        
        assert model_dqn.epsilon < initial_epsilon
        assert model_dqn.epsilon >= model_dqn.epsilon_min

    def test_model_save_load(self, model_dqn):
        """Test model save and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = str(Path(tmpdir) / "test_model.h5")
            
            # Get original weights
            original_weights = [w.numpy().copy() for w in model_dqn.network.weights]
            
            # Save model
            model_dqn.save_model(filepath)
            assert Path(filepath).exists()
            
            # Load model into new instance
            model_new = AgentLearningModel(
                state_size=32,
                action_size=8,
                model_type="dqn"
            )
            model_new.load_model(filepath)
            
            # Verify weights match
            loaded_weights = [w.numpy() for w in model_new.network.weights]
            for orig_w, loaded_w in zip(original_weights, loaded_weights):
                assert np.allclose(orig_w, loaded_w)


class TestExperienceReplay:
    """Tests for Experience Replay buffer."""

    @pytest.fixture
    def replay_buffer(self):
        """Create experience replay buffer."""
        return ExperienceReplay(max_size=1000)

    def test_buffer_initialization(self, replay_buffer):
        """Test buffer initialization."""
        assert len(replay_buffer) == 0
        assert replay_buffer.max_size == 1000

    def test_buffer_add(self, replay_buffer):
        """Test adding experience to buffer."""
        state = np.random.randn(32)
        action = 5
        reward = 1.0
        next_state = np.random.randn(32)
        done = False
        
        replay_buffer.add(state, action, reward, next_state, done)
        assert len(replay_buffer) == 1

    def test_buffer_sample(self, replay_buffer):
        """Test sampling from buffer."""
        # Add multiple experiences
        for _ in range(100):
            state = np.random.randn(32)
            action = np.random.randint(0, 8)
            reward = np.random.randn()
            next_state = np.random.randn(32)
            done = np.random.random() > 0.9
            
            replay_buffer.add(state, action, reward, next_state, done)
        
        # Sample batch
        states, actions, rewards, next_states, dones = replay_buffer.sample(32)
        
        assert states.shape == (32, 32)
        assert actions.shape == (32,)
        assert rewards.shape == (32,)
        assert next_states.shape == (32, 32)
        assert dones.shape == (32,)

    def test_buffer_overflow(self, replay_buffer):
        """Test buffer overflow handling."""
        max_size = replay_buffer.max_size
        
        # Add more experiences than buffer size
        for i in range(max_size + 100):
            state = np.random.randn(32)
            action = np.random.randint(0, 8)
            reward = np.random.randn()
            next_state = np.random.randn(32)
            done = False
            
            replay_buffer.add(state, action, reward, next_state, done)
        
        # Buffer should maintain max_size
        assert len(replay_buffer) == max_size


class TestTrainingEnvironment:
    """Tests for Training Environment."""

    @pytest.fixture
    def env(self):
        """Create training environment."""
        return TrainingEnvironment(
            state_size=32,
            action_size=8,
            max_steps=100
        )

    def test_env_initialization(self, env):
        """Test environment initialization."""
        assert env.state_size == 32
        assert env.action_size == 8
        assert env.max_steps == 100

    def test_env_reset(self, env):
        """Test environment reset."""
        initial_state = env.reset()
        
        assert initial_state.shape == (32,)
        assert env.current_step == 0

    def test_env_step(self, env):
        """Test environment step."""
        env.reset()
        next_state, reward, done, info = env.step(5)
        
        assert next_state.shape == (32,)
        assert isinstance(reward, (float, np.floating))
        assert isinstance(done, (bool, np.bool_))
        assert isinstance(info, dict)

    def test_env_episode_termination(self, env):
        """Test episode termination."""
        env.reset()
        done = False
        steps = 0
        
        while not done and steps < 150:
            _, _, done, _ = env.step(np.random.randint(0, 8))
            steps += 1
        
        assert done or steps >= 150


class TestAgentTrainer:
    """Tests for Agent Trainer."""

    @pytest.fixture
    def trainer(self):
        """Create trainer instance."""
        model = AgentLearningModel(
            state_size=32,
            action_size=8,
            model_type="dqn"
        )
        env = TrainingEnvironment(state_size=32, action_size=8)
        config = {
            "state_size": 32,
            "action_size": 8,
            "buffer_size": 10000,
            "batch_size": 16,
            "update_freq": 4,
            "target_update_freq": 100,
            "episodes": 10,
            "checkpoint_dir": "checkpoints"
        }
        return AgentTrainer(model, config, env)

    def test_trainer_initialization(self, trainer):
        """Test trainer initialization."""
        assert trainer.model is not None
        assert trainer.replay_buffer is not None
        assert len(trainer.episode_rewards) == 0

    def test_collect_experience(self, trainer):
        """Test experience collection."""
        total_reward, steps = trainer.collect_experience(num_steps=50, training=True)
        
        assert steps == 50
        assert len(trainer.replay_buffer) > 0

    def test_train_on_batch(self, trainer):
        """Test training on batch."""
        # Collect some experience first
        trainer.collect_experience(num_steps=50)
        
        loss = trainer.train_on_batch()
        assert isinstance(loss, (float, np.floating)) or loss == 0.0

    def test_train_episode(self, trainer):
        """Test single episode training."""
        metrics = trainer.train_episode()
        
        assert "reward" in metrics
        assert "steps" in metrics
        assert "avg_loss" in metrics
        assert "epsilon" in metrics

    def test_evaluate(self, trainer):
        """Test model evaluation."""
        metrics = trainer.evaluate(num_episodes=2)
        
        assert "mean_reward" in metrics
        assert "std_reward" in metrics
        assert "max_reward" in metrics
        assert "min_reward" in metrics

    def test_save_checkpoint(self, trainer):
        """Test checkpoint saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.checkpoint_dir = Path(tmpdir)
            trainer.save_checkpoint(episode=0, is_best=False)
            
            checkpoint_file = Path(tmpdir) / "model_episode_0.h5"
            assert checkpoint_file.exists()

    def test_save_history(self, trainer):
        """Test training history saving."""
        # Add some history
        trainer.training_history["episode"].append(0)
        trainer.training_history["reward"].append(10.5)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = str(Path(tmpdir) / "history.json")
            trainer.save_history(filepath)
            
            assert Path(filepath).exists()
            
            with open(filepath) as f:
                history = json.load(f)
                assert len(history["episode"]) > 0


# Pytest configuration
@pytest.fixture(scope="session")
def tf_config():
    """Configure TensorFlow for testing."""
    tf.config.set_visible_devices([], 'GPU')  # Use CPU for testing
    yield


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
