# API Reference

## Core Modules

### `ai.agents.BaseAgent`

Base class for all adaptive agents in the system.

#### Methods

**`__init__(name: str, agent_type: str = "base")`**
- Initialize a new agent
- Parameters:
  - `name`: Unique identifier for the agent
  - `agent_type`: Type of agent (default: "base")

**`execute_action(action: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]`**
- Execute an action in the environment
- Parameters:
  - `action`: Name of the action to execute
  - `params`: Optional parameters for the action
- Returns: Dictionary with action result

**`learn(experience: Dict[str, Any]) -> None`**
- Process experience and update knowledge
- Parameters:
  - `experience`: Dictionary containing experience data

**`get_state() -> Dict[str, Any]`**
- Get current agent state
- Returns: State dictionary

**`reset() -> None`**
- Reset agent to initial state

#### Properties

- `name`: Get agent name
- `agent_id`: Get unique agent ID

#### Example

```python
from ai.agents import BaseAgent

# Create an agent
agent = BaseAgent(name="MyAgent")

# Execute an action
result = agent.execute_action("move", {"direction": "north", "distance": 10})

# Learn from experience
agent.learn({"observation": "wall_detected", "reward": -1})

# Get current state
state = agent.get_state()

# Reset agent
agent.reset()
```

## Configuration

### `config.Settings`

Application-wide settings using Pydantic.

#### Configuration Options

```python
# App Info
app_name: str = "Ai-morphasis"
version: str = "2.0.2"
debug: bool = False

# Agent Configuration
max_agents: int = 100
agent_memory_size: int = 10000

# Game Configuration
game_width: int = 1280
game_height: int = 720
target_fps: int = 60

# Model Configuration
model_device: str = "cpu"  # cpu or cuda
batch_size: int = 32
learning_rate: float = 0.001

# Logging
log_level: str = "INFO"
log_file: Optional[str] = "logs/ai_morphasis.log"
```

#### Usage

```python
from config import Settings

config = Settings()
print(f"Running {config.app_name} v{config.version}")
print(f"Using device: {config.model_device}")
```

## Testing

### Fixtures

All pytest fixtures are defined in `tests/conftest.py`:

- `sample_agent`: Pre-configured test agent
- `test_config`: Test configuration

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_base_agent.py -v

# Run specific test
pytest tests/test_base_agent.py::TestBaseAgentInitialization::test_agent_creation -v

# Run with markers
pytest tests/ -m "unit"
```

## Logging

The application uses `loguru` for logging:

```python
from loguru import logger

logger.info("Information message")
logger.debug("Debug message")
logger.warning("Warning message")
logger.error("Error message")
logger.critical("Critical message")
```

Logs are written to:
- Console (stdout)
- File: `logs/ai_morphasis.log` (with rotation)

---

**For more information, see:**
- [Architecture Guide](ARCHITECTURE.md)
- [Contributing Guide](CONTRIBUTING.md)
- [README](README.md)
