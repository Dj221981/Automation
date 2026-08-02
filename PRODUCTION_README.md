# Production-Ready DQN Agent - Documentation

## Overview

This repository contains a **production-hardened Deep Q-Network (DQN)** implementation for reinforcement learning agent training and inference. The codebase prioritizes **safety, reliability, and correctness** over feature breadth.

## Architecture

### Core Components

```
src/
├── models/
│   └── neural_network.py       # DQNNetwork, AgentLearningModel, ExperienceReplay
├── data/
│   └── preprocessing.py        # StateNormalizer, DataAugmentation, BatchGenerator
├── config.py                   # Configuration management
└── __init__.py

tests/
├── test_neural_network.py      # 50+ unit tests for models
└── test_preprocessing.py       # 40+ unit tests for data pipeline

config/
└── production.json             # Production configuration defaults
```

## Installation

### Requirements
- Python 3.8+
- TensorFlow 2.10+
- NumPy 1.21+

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests to verify installation
pytest tests/ -v

# Check imports
python -c "from src.models.neural_network import AgentLearningModel"
python -c "from src.data.preprocessing import StateNormalizer"
```

## Quick Start

### Basic Training Loop

```python
import numpy as np
from src.models.neural_network import AgentLearningModel, ExperienceReplay
from src.data.preprocessing import StateNormalizer, NormalizationType

# Initialize components
state_normalizer = StateNormalizer(normalization_type=NormalizationType.MINMAX)
model = AgentLearningModel(
    state_size=64,
    action_size=8,
    learning_rate=0.001,
    device="cpu",
    seed=42
)
replay = ExperienceReplay(state_size=64, max_size=100000, seed=42)

# Fit normalizer
training_data = np.random.randn(10000, 64).astype(np.float32)
state_normalizer.fit(training_data)

# Training loop
for episode in range(100):
    state = np.random.randn(64).astype(np.float32)
    
    for step in range(100):
        # Normalize state
        state_norm = state_normalizer.normalize(state)
        
        # Select action
        action = model.select_action(state_norm, training=True)
        
        # Simulate environment
        reward = np.random.randn()
        next_state = np.random.randn(64).astype(np.float32)
        done = np.random.random() > 0.9
        
        # Normalize next state
        next_state_norm = state_normalizer.normalize(next_state)
        
        # Store experience
        replay.add(state_norm, action, reward, next_state_norm, done)
        
        # Train if enough experiences
        if len(replay) >= 1000:
            states, actions, rewards, next_states, dones = replay.sample(32)
            loss = model.train_step(states, actions, rewards, next_states, dones)
        
        state = next_state
    
    # Decay exploration
    model.decay_epsilon()
```

## Configuration Management

### Configuration File (`config/production.json`)

```json
{
  "model": {
    "state_size": 64,
    "action_size": 8,
    "learning_rate": 0.001,
    "gamma": 0.99,
    "epsilon": 1.0,
    "device": "cpu"
  },
  "training": {
    "batch_size": 32,
    "replay_buffer_size": 100000
  }
}
```

### Environment Variable Overrides

Configuration values can be overridden via environment variables with the pattern `AUTOMATION_SECTION_KEY`:

```bash
# Override model state size
export AUTOMATION_MODEL_STATE_SIZE=128

# Override learning rate
export AUTOMATION_MODEL_LEARNING_RATE=0.0005

# Override device
export AUTOMATION_MODEL_DEVICE=gpu
```

### Programmatic Configuration

```python
from src.config import ConfigManager

# Load configuration
config = ConfigManager("config/production.json")
config.apply_env_overrides()

# Access values
state_size = config.get("model", "state_size", default=64)
learning_rate = config.get("model", "learning_rate", default=0.001)

# Get entire section
model_config = config.get_section("model")
```

## Production Features

### Input Validation

All public methods validate inputs strictly:

```python
# These will raise ValueError with clear messages
model = AgentLearningModel(state_size=0)          # ❌ state_size must be positive
model = AgentLearningModel(gamma=1.5)             # ❌ gamma must be in [0, 1]
normalizer = StateNormalizer(epsilon=-0.1)        # ❌ epsilon must be positive

# These will raise descriptive errors
model.train_step(
    states=np.array([np.nan] * 64),               # ❌ NaN values detected
    actions=np.array([0]),
    rewards=np.array([1.0]),
    next_states=np.array([1.0] * 64),
    dones=np.array([0])
)
```

### Error Handling

- **Shape validation**: Strict dimension checking on all arrays
- **Finite value checks**: Detection and reporting of NaN/Inf values
- **Batch size validation**: Minimum and maximum size enforcement
- **Type conversion**: Safe conversion with proper error messages
- **File I/O**: Proper exception handling for load/save operations

### Numerical Stability

- **Gradient clipping**: Configurable gradient norm clipping (default: 10.0)
- **Finite loss checks**: Training stops on NaN/Inf loss
- **Target network sync**: Stable weight transfer between networks
- **Epsilon smoothing**: Gradual exploration decay respecting minimum

## Model Checkpointing

```python
# Save model
model.save_model("checkpoints/model.weights.h5")

# Load model (restores weights and metadata)
model.load_model("checkpoints/model.weights.h5")

# Check training progress
print(f"Trained for {model.train_steps} steps")
```

## Data Preprocessing

### Normalization

```python
from src.data.preprocessing import StateNormalizer, NormalizationType

# Fit normalizer
normalizer = StateNormalizer(normalization_type=NormalizationType.MINMAX)
normalizer.fit(training_data)

# Normalize new data
normalized = normalizer.normalize(new_data)

# Reverse normalization
original = normalizer.denormalize(normalized)

# Persist normalizer
normalizer.save("normalizer.json")
normalizer.load("normalizer.json")
```

### Data Augmentation

```python
from src.data.preprocessing import DataAugmentation

# Add Gaussian noise
augmented = DataAugmentation.add_gaussian_noise(data, noise_std=0.1, seed=42)

# Mixup
x_mixed, y_mixed = DataAugmentation.mixup(x1, x2, y1, y2, alpha=0.2, seed=42)

# Random crop (feature dropout)
cropped = DataAugmentation.random_crop(data, crop_fraction=0.8, seed=42)

# Temporal shift
shifted = DataAugmentation.temporal_shift(sequences, shift_range=5, seed=42)
```

### Batch Generation

```python
from src.data.preprocessing import BatchGenerator

batch_gen = BatchGenerator(
    states=training_states,
    actions=training_actions,
    rewards=training_rewards,
    next_states=training_next_states,
    dones=training_dones,
    batch_size=32,
    shuffle=True,
    seed=42
)

for states, actions, rewards, next_states, dones in batch_gen:
    loss = model.train_step(states, actions, rewards, next_states, dones)
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_neural_network.py -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_neural_network.py::TestAgentLearningModel::test_train_step_basic -v
```

### Test Coverage

- **Neural Network Module**: 50+ tests covering initialization, training, validation, persistence
- **Preprocessing Module**: 40+ tests covering normalization, augmentation, batch generation
- **Integration Tests**: End-to-end training pipeline tests

All tests validate:
- ✅ Input validation and error handling
- ✅ Numerical correctness
- ✅ State management and synchronization
- ✅ File I/O operations
- ✅ Edge cases and boundary conditions

## CI/CD Pipeline

The repository includes comprehensive GitHub Actions workflows:

- **Unit Tests**: Multi-version Python testing (3.8-3.11) with coverage reporting
- **Code Quality**: Linting (flake8), formatting (black), type checking (mypy)
- **Security**: Vulnerability scanning (bandit, safety)
- **Integration Tests**: Full pipeline validation
- **Documentation**: Docstring and README checks

View workflows in `tests/workflows.yaml`

## Logging

All modules use Python's standard logging:

```python
import logging

logging.basicConfig(level=logging.INFO)

# Enable debug logging
logging.getLogger('src').setLevel(logging.DEBUG)
```

Example output:
```
INFO:src.models.neural_network:DQNNetwork initialized: state_size=64 action_size=8
INFO:src.models.neural_network:AgentLearningModel initialized: model_type=dqn device=/CPU:0
INFO:src.data.preprocessing:StateNormalizer fitted with 1000 samples, 64 features
```

## Performance Considerations

### Memory

- Experience replay buffer: ~800MB for 100k transitions (32D state)
- Model weights: ~1-2MB depending on architecture
- Use `max_size` parameter to control replay buffer memory

### Computation

- Training step: ~50ms per batch (32 samples, CPU)
- Action selection: ~5ms (inference only)
- Batch generation: Minimal overhead, uses NumPy views

### Optimization

```python
# GPU acceleration
model = AgentLearningModel(device="gpu")

# Larger batches for efficiency
loss = model.train_step(states, actions, rewards, next_states, dones)  # Batch size 32+

# Vectorized preprocessing
states = state_normalizer.normalize(batch_states)  # Entire batch at once
```

## Troubleshooting

### Issue: NaN/Inf Loss

**Cause**: Unstable gradients or invalid data

**Solution**:
```python
# Check data validity
assert np.all(np.isfinite(states)), "States contain NaN/Inf"
assert np.all(np.isfinite(rewards)), "Rewards contain NaN/Inf"

# Reduce learning rate
model = AgentLearningModel(learning_rate=0.0001)

# Enable gradient clipping
model = AgentLearningModel(gradient_clip_norm=1.0)
```

### Issue: Poor Learning Performance

**Cause**: Suboptimal hyperparameters or insufficient data

**Solution**:
```python
# Adjust exploration
model.epsilon = 0.1  # Lower exploration
model.epsilon_decay = 0.99  # Slower decay

# Tune learning rate
model = AgentLearningModel(learning_rate=0.001)

# Increase replay buffer
replay = ExperienceReplay(max_size=500000)
```

### Issue: Out of Memory

**Cause**: Replay buffer too large or batch size too high

**Solution**:
```python
# Reduce replay buffer
replay = ExperienceReplay(max_size=50000)

# Reduce batch size
loss = model.train_step(states[:16], ...)  # Use smaller batches

# Use smaller model
model = AgentLearningModel(hidden_layers=[64, 32])
```

## Best Practices

1. **Always normalize state inputs** before feeding to model
2. **Validate all data** before training (check for NaN/Inf)
3. **Monitor training metrics** (loss, epsilon, exploration)
4. **Checkpoint regularly** to avoid losing training progress
5. **Use seeds** for reproducibility in experiments
6. **Test on small data** before scaling to production
7. **Profile code** to identify bottlenecks
8. **Use appropriate batch sizes** (32-128 typically)

## API Reference

See inline docstrings in source files for detailed API documentation:

- `src/models/neural_network.py`: DQN and training components
- `src/data/preprocessing.py`: Data processing and augmentation
- `src/config.py`: Configuration management
- `tests/`: Comprehensive usage examples

## License

This code is provided as-is for educational and research purposes.

## Support

For issues, questions, or contributions:

1. Check existing documentation and tests
2. Review error messages (they're descriptive!)
3. Enable debug logging for detailed traces
4. Run unit tests to isolate problems
