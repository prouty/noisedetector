# Test Suite

This directory contains automated tests for the noise detector system.

## Setup

Install test dependencies:
```bash
pip install -r requirements.txt
```

This will install `pytest` along with other dependencies.

## Running Tests

### Run all tests:
```bash
make test
```

Or directly:
```bash
pytest tests/ -v
```

### Run specific test file:
```bash
pytest tests/test_capture_ml.py -v
```

Or using make:
```bash
make test-capture-ml
```

### Run specific test:
```bash
pytest tests/test_capture_ml.py::TestCaptureMLValidation::test_validate_ml_capture_setup -v
```

## Test Structure

- `conftest.py`: Shared fixtures and configuration
- `test_capture_ml.py`: Tests for ML-based capture decision system

## Writing New Tests

1. Create a new file `test_*.py` in the `tests/` directory
2. Import pytest: `import pytest`
3. Use fixtures from `conftest.py` (e.g., `config`, `data_dir`)
4. Follow naming convention: `test_*` for functions, `Test*` for classes

Example:
```python
def test_something(config):
    """Test description."""
    assert something == expected_value
```

## Test Markers

Tests can be marked for selective running:

```bash
# Run only unit tests
pytest -m unit

# Skip slow tests
pytest -m "not slow"
```

## Current Tests

### `test_capture_ml.py`

Tests for ML-based capture decision:

- **TestCaptureMLModel**: Model loading and basic functionality
  - `test_model_files_exist`: Verifies model files exist
  - `test_model_loads_with_config`: Tests model loading from config
  - `test_feature_extraction`: Tests feature extraction
  - `test_should_capture_with_model`: Tests prediction with model
  - `test_should_capture_fallback`: Tests threshold fallback
  - `test_model_metadata_valid`: Validates model metadata

- **TestCaptureMLIntegration**: Integration with monitor.py
  - `test_monitor_imports_capture_ml`: Verifies imports work
  - `test_config_has_ml_capture_option`: Checks config options
  - `test_model_paths_in_config`: Validates config paths

- **TestCaptureMLValidation**: Comprehensive validation
  - `test_validate_ml_capture_setup`: Main validation test

## Notes

- Tests that require model files will skip if models aren't trained yet
- Use `pytest.skip()` for conditional tests
- Fixtures are automatically available to all tests

