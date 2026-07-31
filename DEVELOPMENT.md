# Development Guide

## For Contributors

This guide covers setting up a development environment, running tests, and contributing to the project.

## Development Environment Setup

### Prerequisites

- Python 3.8 or higher
- Git
- pip/conda for package management

### Installation for Development

```bash
# Clone the repository
git clone https://github.com/Dj221981/Automation.git
cd Automation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development tools
pip install pytest pytest-cov black flake8 mypy isort

# Install in editable mode
pip install -e .
```

## Running Tests

### Quick Test Run

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_neural_network.py -v

# Run specific test class
pytest tests/test_neural_network.py::TestAgentLearningModel -v

# Run specific test
pytest tests/test_neural_network.py::TestAgentLearningModel::test_train_step_basic -v
```

### Continuous Testing

```bash
# Watch for changes and auto-run tests
pytest-watch tests/

# Run tests on multiple Python versions
tox
```

## Code Quality

### Code Formatting

```bash
# Format code with black
black src/ tests/ --line-length=100

# Sort imports with isort
isort src/ tests/

# Check formatting without changing
black --check src/ tests/
```

### Linting

```bash
# Run flake8
flake8 src/ tests/ --max-line-length=100

# Run pylint (more detailed)
pylint src/ tests/

# Type checking with mypy
mypy src/ --ignore-missing-imports
```

### Pre-commit Hook

```bash
# Create .git/hooks/pre-commit
#!/bin/bash
set -e

echo "Running code quality checks..."
black --check src/ tests/
isort --check-only src/ tests/
flake8 src/ tests/
mypy src/ --ignore-missing-imports

echo "Running tests..."
pytest tests/ -q

echo "✅ All checks passed!"
```

## Project Structure

```
Automation/
├── src/
│   ├── __init__.py
│   ├── config.py                      # Configuration management
│   ├── models/
│   │   ├── __init__.py
│   │   └── neural_network.py          # DQN implementation
│   └── data/
│       ├── __init__.py
│       └── preprocessing.py           # Data processing
├── tests/
│   ├── __init__.py
│   ├── test_neural_network.py        # Neural network tests
│   └── test_preprocessing.py         # Data preprocessing tests
├── config/
│   └── production.json               # Default configuration
├── requirements.txt                  # Production dependencies
├── PRODUCTION_README.md              # Production documentation
├── DEVELOPMENT.md                    # This file
└── README.md                         # Project overview
```

## Adding New Features

### Code Standards

1. **Validation First**: All inputs must be validated before use
2. **Error Messages**: Provide clear, actionable error messages
3. **Docstrings**: All public methods must have comprehensive docstrings
4. **Type Hints**: Use type hints for better IDE support
5. **Logging**: Use logging instead of print statements
6. **Tests**: Write tests before or alongside new code

### Example: Adding a New Function

```python
def new_feature(input_data: np.ndarray, param: float = 0.5) -> np.ndarray:
    """
    Brief description of what this does.
    
    Longer description explaining the algorithm, edge cases,
    and any important considerations.
    
    Args:
        input_data: Description of input [shape info]
        param: Description of parameter (must be in [0, 1])
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: If param is outside valid range
        TypeError: If input_data is not ndarray
    
    Examples:
        >>> result = new_feature(np.array([1, 2, 3]))
    """
    # Input validation
    if not isinstance(input_data, np.ndarray):
        raise TypeError("input_data must be numpy array")
    
    if not 0 <= param <= 1:
        raise ValueError(f"param must be in [0, 1], got {param}")
    
    # Implementation
    logger.debug(f"Processing {input_data.shape}")
    
    # Validation of results
    result = input_data * param
    
    if not np.all(np.isfinite(result)):
        raise ValueError("Operation produced NaN or Inf values")
    
    return result
```

### Writing Tests

```python
class TestNewFeature(unittest.TestCase):
    """Test new_feature function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.valid_input = np.array([1, 2, 3]).astype(np.float32)
    
    def test_basic_operation(self):
        """Test basic operation."""
        result = new_feature(self.valid_input, param=0.5)
        expected = self.valid_input * 0.5
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_invalid_param(self):
        """Test error on invalid parameter."""
        with self.assertRaises(ValueError):
            new_feature(self.valid_input, param=1.5)
    
    def test_invalid_input_type(self):
        """Test error on invalid input type."""
        with self.assertRaises(TypeError):
            new_feature([1, 2, 3], param=0.5)
```

## Debugging

### Enable Debug Logging

```python
import logging

# Set to DEBUG level
logging.basicConfig(level=logging.DEBUG)

# Or for specific module
logging.getLogger('src').setLevel(logging.DEBUG)
```

### Using pdb

```python
import pdb

# Set breakpoint
pdb.set_trace()

# Commands:
# n (next line)
# s (step into)
# c (continue)
# l (list code)
# p variable (print variable)
```

### Inspecting Data

```python
import numpy as np

# Check shape
print(f"Shape: {array.shape}")

# Check data type
print(f"Dtype: {array.dtype}")

# Check for NaN/Inf
print(f"Has NaN: {np.isnan(array).any()}")
print(f"Has Inf: {np.isinf(array).any()}")

# Basic statistics
print(f"Min: {np.min(array)}, Max: {np.max(array)}")
print(f"Mean: {np.mean(array)}, Std: {np.std(array)}")
```

## Performance Profiling

### Using cProfile

```python
import cProfile
import pstats

# Profile a function
profiler = cProfile.Profile()
profiler.enable()

# Code to profile
for i in range(1000):
    model.train_step(states, actions, rewards, next_states, dones)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Print top 20
```

### Using line_profiler

```bash
# Install
pip install line_profiler

# Decorate function with @profile
@profile
def expensive_function():
    pass

# Run
kernprof -l -v script.py
```

## Common Issues

### Issue: Import Errors

**Problem**: `ModuleNotFoundError: No module named 'src'`

**Solution**:
```bash
# Make sure you're in project root
cd Automation

# Install in editable mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: Test Failures

**Problem**: Tests fail locally but pass in CI

**Solution**:
```bash
# Check Python version
python --version  # Should be 3.8+

# Check dependencies
pip list | grep tensorflow

# Run tests with verbose output
pytest -vv tests/

# Check for random seed issues
# Ensure seed is set in test setup
```

### Issue: TensorFlow Not Found

**Problem**: `ModuleNotFoundError: No module named 'tensorflow'`

**Solution**:
```bash
# Install TensorFlow
pip install tensorflow>=2.10.0

# Verify installation
python -c "import tensorflow; print(tensorflow.__version__)"
```

## Release Process

### Before Release

1. **Update version** in `src/__init__.py`
2. **Update CHANGELOG.md**
3. **Run full test suite**: `pytest tests/ -v`
4. **Check code quality**: `flake8 src/ && mypy src/`
5. **Build locally**: `python -m build`

### Create Release

```bash
# Tag version
git tag v1.0.0

# Push tag
git push origin v1.0.0

# GitHub Actions will build and publish
```

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def example_function(param1: str, param2: int = 10) -> bool:
    """
    One-line summary of what the function does.
    
    More detailed explanation if needed. Explain the algorithm,
    any assumptions, and edge cases.
    
    Args:
        param1: Description of param1
        param2: Description of param2 (default: 10)
    
    Returns:
        Description of return value and type
    
    Raises:
        ValueError: When this happens
        TypeError: When that happens
    
    Examples:
        >>> result = example_function("test")
        >>> print(result)
        True
    """
```

### Updating Documentation

1. **Docstrings**: Update in source files
2. **README.md**: High-level overview and quick start
3. **PRODUCTION_README.md**: Detailed usage and API
4. **DEVELOPMENT.md**: Development process (this file)

## Getting Help

1. **Check existing tests**: Look at `tests/` for usage examples
2. **Read docstrings**: All public APIs are documented
3. **Search issues**: Check GitHub issues for similar problems
4. **Enable logging**: Debug with `logging.DEBUG` level
5. **Ask questions**: Open an issue or discussion

## Code Review Checklist

When reviewing code:

- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Code follows style guide (run black, flake8)
- [ ] No hardcoded values (use config or parameters)
- [ ] Proper error handling with meaningful messages
- [ ] All inputs validated
- [ ] No NaN/Inf values in outputs
- [ ] Performance considered for production use
- [ ] Logging used appropriately
- [ ] Type hints present

## Performance Benchmarks

### Expected Performance (CPU, i7-9700K, 16GB RAM)

| Operation | Time | Notes |
|-----------|------|-------|
| Forward pass (batch 32) | ~20ms | 64D state, 2x128 hidden |
| Training step | ~50ms | Includes gradient computation |
| Inference | ~5ms | Single action selection |
| Normalize batch | ~5ms | 32x64 state normalization |
| Sample from buffer | ~2ms | 100k capacity buffer |

### Optimization Tips

1. **Vectorize operations**: Use NumPy/TensorFlow operations
2. **Batch processing**: Process multiple samples together
3. **Avoid Python loops**: Use NumPy/TensorFlow where possible
4. **Profile first**: Use cProfile to find bottlenecks
5. **Use GPU**: Set `device="gpu"` for large-scale training

## Resources

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [NumPy Documentation](https://numpy.org/doc/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [pytest Documentation](https://docs.pytest.org/)
