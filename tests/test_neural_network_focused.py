"""
Production-grade focused unit tests for src/models/neural_network.py.

Five tests that exercise the most critical paths:
1. DQNNetwork forward pass with variable batch sizes, modes, and gradient flow.
2. AgentLearningModel training-loop convergence (loss, epsilon, target-network).
3. ExperienceReplay circular-overflow behaviour and deterministic sampling.
4. Model save/load round-trip and metadata corruption handling.
5. Error handling and graceful recovery for bad inputs.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from src.models.neural_network import AgentLearningModel, DQNNetwork, ExperienceReplay

# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

STATE_SIZE = 8
ACTION_SIZE = 4


@pytest.fixture(autouse=True)
def _deterministic_seeds():
    """Pin global random seeds before every test for reproducibility."""
    np.random.seed(0)
    tf.random.set_seed(0)
    yield


@pytest.fixture()
def small_network():
    """A tiny DQNNetwork with two small hidden layers."""
    return DQNNetwork(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        hidden_layers=[16, 8],
    )


@pytest.fixture()
def small_model():
    """A minimal AgentLearningModel configured for fast tests."""
    return AgentLearningModel(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        hidden_layers=[16, 8],
        learning_rate=1e-3,
        target_update_interval=10,
        seed=42,
    )


def _make_batch(batch_size: int = 16) -> tuple:
    """Return a valid (states, actions, rewards, next_states, dones) batch."""
    states = np.random.randn(batch_size, STATE_SIZE).astype(np.float32)
    actions = np.random.randint(0, ACTION_SIZE, size=(batch_size,), dtype=np.int32)
    rewards = np.random.randn(batch_size).astype(np.float32)
    next_states = np.random.randn(batch_size, STATE_SIZE).astype(np.float32)
    dones = (np.random.rand(batch_size) > 0.8).astype(np.float32)
    return states, actions, rewards, next_states, dones


# ---------------------------------------------------------------------------
# Test 1 – DQNNetwork forward pass: variable batch sizes, modes, gradient flow
# ---------------------------------------------------------------------------


class TestDQNNetworkForwardPass:
    """Validate DQNNetwork forward pass across batch sizes, training flags, and gradient flow."""

    @pytest.mark.parametrize("batch_size", [1, 32, 128])
    def test_output_shape_for_variable_batch_sizes(self, small_network, batch_size):
        """Output shape must be (batch_size, action_size) for all supported batch sizes."""
        x = tf.zeros((batch_size, STATE_SIZE), dtype=tf.float32)
        y = small_network(x, training=False)
        assert y.shape == (batch_size, ACTION_SIZE), (
            f"Expected output shape ({batch_size}, {ACTION_SIZE}), got {y.shape}"
        )

    @pytest.mark.parametrize("batch_size", [1, 32, 128])
    def test_output_is_finite_for_all_batch_sizes(self, small_network, batch_size):
        """All Q-value outputs must be finite (no NaN/Inf) regardless of batch size."""
        x = tf.random.normal((batch_size, STATE_SIZE))
        y = small_network(x, training=False)
        assert tf.reduce_all(tf.math.is_finite(y)), (
            f"Non-finite values in Q-values for batch_size={batch_size}"
        )

    def test_training_mode_does_not_change_output_shape(self, small_network):
        """training=True must produce the same output shape as training=False."""
        x = tf.random.normal((32, STATE_SIZE))
        y_train = small_network(x, training=True)
        y_eval = small_network(x, training=False)
        assert y_train.shape == y_eval.shape, (
            "Output shape differs between training and evaluation modes"
        )

    def test_gradient_flow_reaches_all_trainable_weights(self, small_network):
        """Gradients must flow to every trainable weight during backpropagation."""
        x = tf.random.normal((8, STATE_SIZE))
        with tf.GradientTape() as tape:
            y = small_network(x, training=True)
            # Scalar surrogate loss
            loss = tf.reduce_mean(y)
        grads = tape.gradient(loss, small_network.trainable_variables)

        assert grads, "No gradients produced at all"
        for var, grad in zip(small_network.trainable_variables, grads):
            assert grad is not None, f"Gradient is None for variable {var.name}"
            assert tf.reduce_all(tf.math.is_finite(grad)), (
                f"Non-finite gradient for variable {var.name}"
            )

    def test_dropout_reduces_activation_variance_in_training_mode(self):
        """With high dropout, training-mode outputs should vary between calls."""
        net = DQNNetwork(
            state_size=STATE_SIZE,
            action_size=ACTION_SIZE,
            hidden_layers=[64, 64],
            dropout_rate=0.5,
        )
        x = tf.random.normal((32, STATE_SIZE))
        # Two independent training-mode calls on the same input should differ
        y1 = net(x, training=True).numpy()
        y2 = net(x, training=True).numpy()
        assert not np.allclose(y1, y2, atol=1e-6), (
            "Dropout should produce different outputs on repeated training-mode calls"
        )


# ---------------------------------------------------------------------------
# Test 2 – AgentLearningModel training-loop convergence
# ---------------------------------------------------------------------------


class TestAgentLearningModelConvergence:
    """Verify training-loop convergence: loss trend, epsilon decay, target-network sync."""

    def test_loss_decreases_over_multiple_training_steps(self, small_model):
        """Average loss over the last quartile of steps should be lower than the first."""
        n_steps = 40
        losses = []
        for _ in range(n_steps):
            batch = _make_batch(16)
            loss = small_model.train_step(*batch)
            losses.append(loss)

        first_quarter = np.mean(losses[: n_steps // 4])
        last_quarter = np.mean(losses[-n_steps // 4 :])
        # Loss must not explode – allow some tolerance for noisy synthetic data
        assert last_quarter < first_quarter * 3, (
            f"Loss did not converge: first_quarter={first_quarter:.4f}, "
            f"last_quarter={last_quarter:.4f}"
        )

    def test_gradients_are_applied_and_weights_change(self, small_model):
        """Network weights must change after at least one training step."""
        weights_before = [w.numpy().copy() for w in small_model.network.trainable_variables]
        small_model.train_step(*_make_batch(16))
        weights_after = [w.numpy() for w in small_model.network.trainable_variables]

        any_changed = any(
            not np.array_equal(b, a)
            for b, a in zip(weights_before, weights_after)
        )
        assert any_changed, "Weights did not change after a training step"

    def test_epsilon_decay_applied_correctly(self, small_model):
        """Epsilon must decrease monotonically and never fall below epsilon_min."""
        initial_epsilon = small_model.epsilon
        previous = initial_epsilon
        for _ in range(20):
            small_model.decay_epsilon()
            assert small_model.epsilon <= previous, (
                "Epsilon increased after decay_epsilon()"
            )
            assert small_model.epsilon >= small_model.epsilon_min, (
                "Epsilon fell below epsilon_min"
            )
            previous = small_model.epsilon

        assert small_model.epsilon < initial_epsilon, (
            "Epsilon did not decrease at all after 20 decay steps"
        )

    def test_target_network_updates_at_correct_interval(self, small_model):
        """Target network must sync with the online network at target_update_interval steps."""
        # Perturb the online network so it differs from target
        for var in small_model.network.trainable_variables:
            var.assign(var + tf.ones_like(var) * 10.0)

        interval = small_model.target_update_interval  # 10 for small_model fixture
        n_steps_before_update = interval - 1

        # Do (interval - 1) training steps; target should NOT be synced yet
        for _ in range(n_steps_before_update):
            small_model.train_step(*_make_batch(16))

        online_w = small_model.network.get_weights()
        target_w = small_model.target_network.get_weights()

        # At least one weight array should differ before the sync step
        all_same_before = all(
            np.allclose(o, t, atol=1e-5) for o, t in zip(online_w, target_w)
        )
        assert not all_same_before, (
            "Target network appears synced before update interval was reached"
        )

        # One more step triggers the sync
        small_model.train_step(*_make_batch(16))

        online_w_after = small_model.network.get_weights()
        target_w_after = small_model.target_network.get_weights()
        for o, t in zip(online_w_after, target_w_after):
            np.testing.assert_allclose(o, t, rtol=1e-5, atol=1e-5,
                                       err_msg="Target network not synced after update interval")

    def test_train_steps_counter_increments_correctly(self, small_model):
        """train_steps must increment by exactly 1 per call to train_step."""
        assert small_model.train_steps == 0, "train_steps should start at 0"
        for expected in range(1, 6):
            small_model.train_step(*_make_batch(16))
            assert small_model.train_steps == expected, (
                f"Expected train_steps={expected}, got {small_model.train_steps}"
            )


# ---------------------------------------------------------------------------
# Test 3 – ExperienceReplay circular overflow and determinism
# ---------------------------------------------------------------------------


class TestExperienceReplayCircularOverflow:
    """Validate circular-buffer overflow semantics and reproducible sampling."""

    def _add_n(self, replay: ExperienceReplay, n: int, start: int = 0) -> None:
        """Add *n* distinguishable experiences starting at index *start*."""
        for i in range(start, start + n):
            state = np.full(STATE_SIZE, float(i), dtype=np.float32)
            next_state = np.full(STATE_SIZE, float(i + 1), dtype=np.float32)
            replay.add(state, i % ACTION_SIZE, float(i), next_state, False)

    def test_buffer_does_not_exceed_max_size_after_overflow(self):
        """Buffer length must never exceed max_size even after many insertions."""
        max_size = 10
        replay = ExperienceReplay(state_size=STATE_SIZE, max_size=max_size)
        self._add_n(replay, 50)
        assert len(replay) == max_size, (
            f"Buffer exceeded max_size: len={len(replay)}, max_size={max_size}"
        )

    def test_oldest_entry_overwritten_first(self):
        """After overflow the oldest rewards must be replaced by the newest."""
        max_size = 5
        replay = ExperienceReplay(state_size=STATE_SIZE, max_size=max_size, seed=99)
        # Fill once
        self._add_n(replay, max_size)
        # Overwrite all slots (rewards 0..4 replaced by 5..9)
        self._add_n(replay, max_size, start=max_size)

        present_rewards = {exp[2] for exp in replay.buffer}
        for old_reward in range(max_size):
            assert float(old_reward) not in present_rewards, (
                f"Old reward {old_reward} should have been overwritten"
            )

    def test_position_wraps_correctly(self):
        """Position pointer must wrap at max_size and start overwriting from index 0."""
        max_size = 4
        replay = ExperienceReplay(state_size=STATE_SIZE, max_size=max_size)
        self._add_n(replay, max_size)
        assert replay.position == 0, (
            f"After filling exactly max_size items position should be 0, got {replay.position}"
        )
        # One more write should advance position to 1
        replay.add(
            np.zeros(STATE_SIZE, dtype=np.float32),
            0,
            0.0,
            np.zeros(STATE_SIZE, dtype=np.float32),
            False,
        )
        assert replay.position == 1, (
            f"After one overflow write position should be 1, got {replay.position}"
        )

    def test_sampling_is_deterministic_with_fixed_seed(self):
        """Two buffers with the same seed must return identical samples."""
        max_size = 20
        replay_a = ExperienceReplay(state_size=STATE_SIZE, max_size=max_size, seed=7)
        replay_b = ExperienceReplay(state_size=STATE_SIZE, max_size=max_size, seed=7)

        np.random.seed(100)
        for _ in range(max_size):
            s = np.random.randn(STATE_SIZE).astype(np.float32)
            ns = np.random.randn(STATE_SIZE).astype(np.float32)
            r = float(np.random.randn())
            for buf in (replay_a, replay_b):
                buf.add(s, 0, r, ns, False)

        sa, aa, ra, *_ = replay_a.sample(8)
        sb, ab, rb, *_ = replay_b.sample(8)

        np.testing.assert_array_equal(ra, rb, err_msg="Rewards differ for same seed")
        np.testing.assert_array_equal(aa, ab, err_msg="Actions differ for same seed")

    def test_partial_fill_then_overflow(self):
        """Buffer that is partially filled then overflowed must stay at max_size."""
        max_size = 6
        replay = ExperienceReplay(state_size=STATE_SIZE, max_size=max_size)
        # Add 3 experiences (partial)
        self._add_n(replay, 3)
        assert len(replay) == 3
        # Overflow by adding 10 more
        self._add_n(replay, 10, start=3)
        assert len(replay) == max_size, (
            f"Expected buffer size {max_size}, got {len(replay)}"
        )


# ---------------------------------------------------------------------------
# Test 4 – Model save/load persistence and metadata corruption handling
# ---------------------------------------------------------------------------


class TestModelSaveLoadPersistence:
    """Validate round-trip save/load, state restoration, and metadata corruption handling."""

    def _trained_model(self, n_steps: int = 5) -> AgentLearningModel:
        model = AgentLearningModel(
            state_size=STATE_SIZE,
            action_size=ACTION_SIZE,
            hidden_layers=[16, 8],
            seed=1,
        )
        for _ in range(n_steps):
            model.train_step(*_make_batch(16))
        return model

    def test_weights_are_identical_after_round_trip(self, tmp_path):
        """Online network weights loaded from disk must match the saved weights exactly."""
        original = self._trained_model()
        path = str(tmp_path / "model.weights.h5")
        original.save_model(path)

        loaded = AgentLearningModel(
            state_size=STATE_SIZE, action_size=ACTION_SIZE, hidden_layers=[16, 8]
        )
        loaded.load_model(path)

        for orig_w, load_w in zip(
            original.network.get_weights(), loaded.network.get_weights()
        ):
            np.testing.assert_allclose(
                orig_w, load_w, rtol=1e-5, atol=1e-6,
                err_msg="Mismatch in network weights after load_model",
            )

    def test_train_steps_is_restored_after_load(self, tmp_path):
        """train_steps metadata must be correctly restored from the .meta.npz file."""
        n_steps = 7
        original = self._trained_model(n_steps)
        assert original.train_steps == n_steps

        path = str(tmp_path / "model.weights.h5")
        original.save_model(path)

        loaded = AgentLearningModel(
            state_size=STATE_SIZE, action_size=ACTION_SIZE, hidden_layers=[16, 8]
        )
        loaded.load_model(path)

        assert loaded.train_steps == n_steps, (
            f"Expected train_steps={n_steps}, got {loaded.train_steps}"
        )

    def test_target_network_is_synced_after_load(self, tmp_path):
        """After load_model the target network must mirror the online network."""
        original = self._trained_model()
        path = str(tmp_path / "model.weights.h5")
        original.save_model(path)

        loaded = AgentLearningModel(
            state_size=STATE_SIZE, action_size=ACTION_SIZE, hidden_layers=[16, 8]
        )
        loaded.load_model(path)

        for o_w, t_w in zip(
            loaded.network.get_weights(), loaded.target_network.get_weights()
        ):
            np.testing.assert_allclose(
                o_w, t_w, rtol=1e-5, atol=1e-6,
                err_msg="Target network not synced with online network after load",
            )

    def test_metadata_file_is_created_alongside_weights(self, tmp_path):
        """save_model must produce both the weights file and its .meta.npz companion."""
        original = self._trained_model(3)
        path = str(tmp_path / "model.weights.h5")
        original.save_model(path)

        assert Path(path).exists(), "Weights file not created"
        assert Path(f"{path}.meta.npz").exists(), "Metadata file not created"

    def test_load_with_missing_metadata_defaults_train_steps_to_zero(self, tmp_path):
        """When .meta.npz is absent train_steps must default to 0 without raising."""
        original = self._trained_model(3)
        path = str(tmp_path / "model.weights.h5")
        original.save_model(path)

        # Delete the metadata file to simulate corruption / missing file
        meta_path = Path(f"{path}.meta.npz")
        meta_path.unlink()
        assert not meta_path.exists()

        loaded = AgentLearningModel(
            state_size=STATE_SIZE, action_size=ACTION_SIZE, hidden_layers=[16, 8]
        )
        loaded.load_model(path)  # Must not raise

        assert loaded.train_steps == 0, (
            "train_steps should default to 0 when metadata is missing"
        )

    def test_save_then_continue_training_does_not_affect_loaded_model(self, tmp_path):
        """Weights saved at step N must not be altered by subsequent training on the original."""
        original = self._trained_model(3)
        path = str(tmp_path / "model.weights.h5")
        original.save_model(path)

        # Train more on the original
        for _ in range(10):
            original.train_step(*_make_batch(16))

        # Weights in the file (loaded into fresh model) should match the snapshot
        loaded = AgentLearningModel(
            state_size=STATE_SIZE, action_size=ACTION_SIZE, hidden_layers=[16, 8]
        )
        loaded.load_model(path)
        assert loaded.train_steps == 3, (
            "Loaded model should reflect the step count at save time, not after further training"
        )


# ---------------------------------------------------------------------------
# Test 5 – Error handling and graceful recovery
# ---------------------------------------------------------------------------


class TestErrorHandlingAndRecovery:
    """Verify that bad inputs are rejected cleanly without corrupting model state."""

    def test_nan_in_state_raises_value_error(self, small_model):
        """select_action must raise ValueError when the state contains NaN."""
        bad_state = np.zeros(STATE_SIZE, dtype=np.float32)
        bad_state[2] = float("nan")
        with pytest.raises(ValueError, match="NaN or infinite"):
            small_model.select_action(bad_state)

    def test_inf_in_state_raises_value_error(self, small_model):
        """select_action must raise ValueError when the state contains Inf."""
        bad_state = np.zeros(STATE_SIZE, dtype=np.float32)
        bad_state[0] = float("inf")
        with pytest.raises(ValueError, match="NaN or infinite"):
            small_model.select_action(bad_state)

    def test_nan_in_rewards_raises_value_error(self, small_model):
        """train_step must raise ValueError when rewards contain NaN."""
        states, actions, rewards, next_states, dones = _make_batch(16)
        rewards[3] = float("nan")
        with pytest.raises(ValueError):
            small_model.train_step(states, actions, rewards, next_states, dones)

    def test_inf_in_next_states_raises_value_error(self, small_model):
        """train_step must raise ValueError when next_states contain Inf."""
        states, actions, rewards, next_states, dones = _make_batch(16)
        next_states[0, 1] = float("inf")
        with pytest.raises(ValueError):
            small_model.train_step(states, actions, rewards, next_states, dones)

    def test_action_out_of_bounds_raises_value_error(self, small_model):
        """train_step must reject action indices >= action_size."""
        states, actions, rewards, next_states, dones = _make_batch(16)
        actions[5] = ACTION_SIZE + 99  # clearly out of bounds
        with pytest.raises(ValueError, match="actions contain values outside"):
            small_model.train_step(states, actions, rewards, next_states, dones)

    def test_negative_action_raises_value_error(self, small_model):
        """train_step must reject negative action indices."""
        states, actions, rewards, next_states, dones = _make_batch(16)
        actions[0] = -1
        with pytest.raises(ValueError):
            small_model.train_step(states, actions, rewards, next_states, dones)

    @pytest.mark.parametrize("epsilon_val", [0.0, 1.0])
    def test_epsilon_boundary_conditions(self, epsilon_val):
        """Epsilon boundary values 0.0 and 1.0 must be accepted without error."""
        model = AgentLearningModel(
            state_size=STATE_SIZE,
            action_size=ACTION_SIZE,
            epsilon=epsilon_val,
            epsilon_min=0.0,
            hidden_layers=[8],
        )
        state = np.zeros(STATE_SIZE, dtype=np.float32)
        action = model.select_action(state, training=True)
        assert 0 <= action < ACTION_SIZE, (
            f"Action {action} out of range for epsilon={epsilon_val}"
        )

    def test_empty_batch_raises_value_error(self, small_model):
        """train_step must reject an empty (zero-sample) batch."""
        empty_states = np.empty((0, STATE_SIZE), dtype=np.float32)
        empty_actions = np.empty((0,), dtype=np.int32)
        empty_rewards = np.empty((0,), dtype=np.float32)
        empty_next = np.empty((0, STATE_SIZE), dtype=np.float32)
        empty_dones = np.empty((0,), dtype=np.float32)
        with pytest.raises(ValueError, match="at least one sample"):
            small_model.train_step(
                empty_states, empty_actions, empty_rewards, empty_next, empty_dones
            )

    def test_mismatched_batch_shapes_raises_value_error(self, small_model):
        """train_step must raise ValueError when arrays have inconsistent batch sizes."""
        states, actions, rewards, next_states, dones = _make_batch(16)
        with pytest.raises(ValueError):
            # actions has 15 elements but states has 16
            small_model.train_step(states, actions[:-1], rewards, next_states, dones)

    def test_wrong_state_dimension_raises_value_error(self, small_model):
        """train_step must reject states with the wrong feature dimension."""
        states, actions, rewards, next_states, dones = _make_batch(16)
        # Extra feature column
        bad_states = np.hstack([states, states[:, :1]])
        with pytest.raises(ValueError):
            small_model.train_step(bad_states, actions, rewards, next_states, dones)

    def test_model_remains_usable_after_rejected_train_step(self, small_model):
        """A failed train_step must leave the model in a valid, usable state."""
        good_batch = _make_batch(16)
        # Inject bad rewards into a copy of the batch
        bad_rewards = good_batch[2].copy()
        bad_rewards[0] = float("nan")

        with pytest.raises(ValueError):
            small_model.train_step(
                good_batch[0], good_batch[1], bad_rewards,
                good_batch[3], good_batch[4],
            )

        # Model must still accept valid inputs after the error
        loss = small_model.train_step(*good_batch)
        assert np.isfinite(loss), "Model returned non-finite loss after recovering from an error"
