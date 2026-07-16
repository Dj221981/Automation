import numpy as np
import pytest
import tensorflow as tf

from src.models.neural_network import DQNNetwork, AgentLearningModel, ExperienceReplay


def test_dqn_network_init_validation():
    with pytest.raises(ValueError):
        DQNNetwork(state_size=0, action_size=2)
    with pytest.raises(ValueError):
        DQNNetwork(state_size=4, action_size=0)
    with pytest.raises(ValueError):
        DQNNetwork(state_size=4, action_size=2, hidden_layers=[])
    with pytest.raises(ValueError):
        DQNNetwork(state_size=4, action_size=2, hidden_layers=[64, 0])
    with pytest.raises(ValueError):
        DQNNetwork(state_size=4, action_size=2, dropout_rate=1.2)


def test_dqn_network_forward_shape():
    model = DQNNetwork(state_size=4, action_size=3, hidden_layers=[16, 8])
    x = tf.zeros((5, 4), dtype=tf.float32)
    y = model(x, training=False)
    assert tuple(y.shape) == (5, 3)


def test_agent_learning_model_config_validation_errors():
    with pytest.raises(ValueError):
        AgentLearningModel(state_size=0, action_size=2)
    with pytest.raises(ValueError):
        AgentLearningModel(state_size=4, action_size=0)
    with pytest.raises(ValueError):
        AgentLearningModel(state_size=4, action_size=2, learning_rate=0)
    with pytest.raises(ValueError):
        AgentLearningModel(state_size=4, action_size=2, gamma=1.5)
    with pytest.raises(ValueError):
        AgentLearningModel(state_size=4, action_size=2, epsilon=-0.1)
    with pytest.raises(ValueError):
        AgentLearningModel(state_size=4, action_size=2, epsilon_decay=0)
    with pytest.raises(ValueError):
        AgentLearningModel(state_size=4, action_size=2, epsilon=0.1, epsilon_min=0.2)
    with pytest.raises(ValueError):
        AgentLearningModel(state_size=4, action_size=2, model_type="pg")
    with pytest.raises(ValueError):
        AgentLearningModel(state_size=4, action_size=2, device="tpu")
    with pytest.raises(ValueError):
        AgentLearningModel(state_size=4, action_size=2, gradient_clip_norm=0)


def test_select_action_output_range_training_and_eval():
    model = AgentLearningModel(state_size=4, action_size=3, seed=123)
    state = np.zeros(4, dtype=np.float32)

    # Evaluation path (greedy)
    action_eval = model.select_action(state, training=False)
    assert 0 <= action_eval < 3

    # Training path (epsilon-greedy)
    model.epsilon = 1.0
    action_train = model.select_action(state, training=True)
    assert 0 <= action_train < 3


def test_select_action_rejects_bad_state_shape_and_values():
    model = AgentLearningModel(state_size=4, action_size=2)

    with pytest.raises(ValueError):
        model.select_action(np.zeros((4, 1), dtype=np.float32))

    bad = np.array([0.0, 1.0, np.nan, 2.0], dtype=np.float32)
    with pytest.raises(ValueError):
        model.select_action(bad)


def _valid_batch(state_size=4, batch_size=8, action_size=3):
    states = np.random.randn(batch_size, state_size).astype(np.float32)
    actions = np.random.randint(0, action_size, size=(batch_size,), dtype=np.int32)
    rewards = np.random.randn(batch_size).astype(np.float32)
    next_states = np.random.randn(batch_size, state_size).astype(np.float32)
    dones = (np.random.rand(batch_size) > 0.5).astype(np.float32)
    return states, actions, rewards, next_states, dones


def test_train_step_returns_finite_loss_and_updates_metric():
    model = AgentLearningModel(state_size=4, action_size=3, seed=7)
    batch = _valid_batch(state_size=4, batch_size=16, action_size=3)
    loss = model.train_step(*batch)
    assert np.isfinite(loss)
    assert model.train_loss.result().numpy() > 0 or model.train_loss.result().numpy() == 0


def test_train_step_validation_failures():
    model = AgentLearningModel(state_size=4, action_size=3)
    states, actions, rewards, next_states, dones = _valid_batch(state_size=4, batch_size=8, action_size=3)

    with pytest.raises(ValueError):
        model.train_step(states.reshape(4, 2, 4), actions, rewards, next_states, dones)

    with pytest.raises(ValueError):
        model.train_step(states, actions[:-1], rewards, next_states, dones)

    bad_actions = actions.copy()
    bad_actions[0] = 999
    with pytest.raises(ValueError):
        model.train_step(states, bad_actions, rewards, next_states, dones)

    bad_states = states.copy()
    bad_states[0, 0] = np.nan
    with pytest.raises(ValueError):
        model.train_step(bad_states, actions, rewards, next_states, dones)


def test_update_target_network_synchronizes_weights():
    model = AgentLearningModel(state_size=4, action_size=3, seed=1)

    for var in model.network.trainable_variables:
        var.assign_add(tf.ones_like(var))

    model.update_target_network()

    for w_online, w_target in zip(model.network.get_weights(), model.target_network.get_weights()):
        np.testing.assert_allclose(w_online, w_target, rtol=1e-6, atol=1e-6)


def test_decay_epsilon_respects_minimum():
    model = AgentLearningModel(
        state_size=4,
        action_size=2,
        epsilon=0.5,
        epsilon_decay=0.5,
        epsilon_min=0.2,
    )
    model.decay_epsilon()
    assert model.epsilon == pytest.approx(0.25)

    model.decay_epsilon()
    assert model.epsilon == pytest.approx(0.2)

    model.decay_epsilon()
    assert model.epsilon == pytest.approx(0.2)


def test_save_and_load_model(tmp_path):
    model = AgentLearningModel(state_size=4, action_size=3, seed=9)
    path = tmp_path / "weights.weights.h5"
    batch = _valid_batch(state_size=4, batch_size=8, action_size=3)

    model.train_step(*batch)
    assert model.train_steps == 1

    model.save_model(str(path))
    assert path.exists()
    assert (tmp_path / "weights.weights.h5.meta.npz").exists()

    other = AgentLearningModel(state_size=4, action_size=3, seed=99)
    other.load_model(str(path))

    for a, b in zip(model.network.get_weights(), other.network.get_weights()):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)

    assert other.train_steps == model.train_steps


def test_experience_replay_init_validation():
    with pytest.raises(ValueError):
        ExperienceReplay(state_size=0)
    with pytest.raises(ValueError):
        ExperienceReplay(state_size=4, max_size=0)


def test_experience_replay_add_and_len_and_overwrite():
    replay = ExperienceReplay(state_size=3, max_size=2, seed=42)

    s1 = np.array([1, 2, 3], dtype=np.float32)
    s2 = np.array([4, 5, 6], dtype=np.float32)
    s3 = np.array([7, 8, 9], dtype=np.float32)

    replay.add(s1, 0, 1.0, s2, False)
    replay.add(s2, 1, 0.5, s3, True)
    assert len(replay) == 2

    replay.add(s3, 2, -1.0, s1, False)
    assert len(replay) == 2


def test_experience_replay_add_validation_errors():
    replay = ExperienceReplay(state_size=3, max_size=10)
    good = np.array([1, 2, 3], dtype=np.float32)

    with pytest.raises(ValueError):
        replay.add(np.array([1, 2], dtype=np.float32), 0, 1.0, good, False)

    with pytest.raises(ValueError):
        replay.add(good, 0, 1.0, np.array([1, 2], dtype=np.float32), False)

    with pytest.raises(TypeError):
        replay.add(good, "0", 1.0, good, False)

    with pytest.raises(ValueError):
        replay.add(good, 0, np.inf, good, False)


def test_experience_replay_sample_shapes_and_errors():
    replay = ExperienceReplay(state_size=4, max_size=10, seed=123)

    with pytest.raises(ValueError):
        replay.sample(1)

    for _ in range(5):
        s = np.random.randn(4).astype(np.float32)
        ns = np.random.randn(4).astype(np.float32)
        replay.add(s, 1, 0.1, ns, False)

    states, actions, rewards, next_states, dones = replay.sample(3)
    assert states.shape == (3, 4)
    assert actions.shape == (3,)
    assert rewards.shape == (3,)
    assert next_states.shape == (3, 4)
    assert dones.shape == (3,)

    # Requesting bigger batch than available should clamp to len(buffer)
    states2, actions2, rewards2, next_states2, dones2 = replay.sample(99)
    assert states2.shape[0] == 5
    assert actions2.shape[0] == 5
    assert rewards2.shape[0] == 5
    assert next_states2.shape[0] == 5
    assert dones2.shape[0] == 5
