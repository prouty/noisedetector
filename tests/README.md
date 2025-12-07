# Test Suite

This directory contains automated tests for the noise detector system.

**All tests run locally on your development machine, not on the Raspberry Pi.**

## Setup

Install test dependencies:
```bash
make init        # Create virtual environment
make shell       # Activate venv
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
pytest tests/test_features.py -v
pytest tests/test_email.py -v
pytest tests/test_reporting.py -v
```

Or using make:
```bash
make test-capture-ml    # ML capture tests
make test-features      # Feature extraction tests
make test-email        # Email functionality tests
make test-reporting    # Report generation tests
make test-core         # All core module tests (features, email, reporting)
```

### Run specific test:
```bash
pytest tests/test_capture_ml.py::TestCaptureMLValidation::test_validate_ml_capture_setup -v
```

## Test Structure

- `conftest.py`: Shared fixtures and configuration
- `test_capture_ml.py`: Tests for ML-based capture decision system
- `test_features.py`: Tests for `core.features` module (feature extraction)
- `test_email.py`: Tests for `core.email` module (email functionality)
- `test_reporting.py`: Tests for `core.reporting` module (report generation)

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

### `test_features.py`

Tests for `core.features` module (audio feature extraction):

- **TestLoadMonoWav**: WAV file loading
  - `test_load_mono_wav`: Load existing WAV file
  - `test_load_mono_wav_creates_temp_file`: Load temporary WAV file
  - `test_load_nonexistent_file`: Error handling

- **TestComputeAvgSpectrum**: Average spectrum computation
  - `test_compute_avg_spectrum`: Basic spectrum computation
  - `test_compute_avg_spectrum_short_audio`: Short audio handling
  - `test_compute_avg_spectrum_empty`: Empty input handling

- **TestMelFilterbank**: Mel filterbank creation
  - `test_create_mel_filterbank`: Basic filterbank creation
  - `test_create_mel_filterbank_different_sizes`: Various parameters

- **TestDCT**: DCT computation
  - `test_dct`: Basic DCT computation
  - `test_dct_identity`: Single coefficient case

- **TestMFCCFeatures**: MFCC feature extraction
  - `test_extract_mfcc_features`: Basic MFCC extraction
  - `test_extract_mfcc_features_different_n`: Different coefficient counts

- **TestAdditionalFeatures**: Additional feature extraction
  - `test_extract_additional_features`: Spectral/temporal features

- **TestSpectralFeatures**: Spectral feature computation
  - `test_compute_spectral_features`: Spectral analysis
  - `test_compute_spectral_features_empty`: Empty input handling

- **TestTemporalFeatures**: Temporal feature computation
  - `test_compute_temporal_features`: Temporal analysis
  - `test_compute_temporal_features_empty`: Empty input handling

### `test_email.py`

Tests for `core.email` module (email functionality):

- **TestGetEmailConfig**: Email configuration loading
  - `test_get_email_config_from_config`: Load from config.json
  - `test_get_email_config_from_env`: Load from environment variables
  - `test_get_email_config_env_overrides_config`: Environment precedence
  - `test_get_email_config_defaults`: Default values

- **TestSendEmail**: Email sending functionality
  - `test_send_email_success`: Successful email sending
  - `test_send_email_no_tls`: Sending without TLS
  - `test_send_email_missing_config`: Missing configuration handling
  - `test_send_email_smtp_error`: SMTP error handling
  - `test_send_email_cleanup_on_error`: Cleanup on errors

### `test_reporting.py`

Tests for `core.reporting` module (report generation):

- **TestLoadEvents**: Event loading from CSV
  - `test_load_events_existing_file`: Load existing events.csv
  - `test_load_events_nonexistent_file`: Non-existent file handling
  - `test_load_events_creates_temp_file`: Temporary file loading
  - `test_load_events_strips_column_names`: Column name whitespace handling

- **TestFilterRecentEvents**: Event filtering by time
  - `test_filter_recent_events`: Basic time filtering
  - `test_filter_recent_events_empty`: Empty DataFrame handling
  - `test_filter_recent_events_no_timestamp_column`: Missing column handling

- **TestGenerateEmailReport**: Email report generation
  - `test_generate_email_report_empty`: Empty DataFrame report
  - `test_generate_email_report_with_data`: Report with event data
  - `test_generate_email_report_with_chirps`: Report with chirp events

- **TestAddDateColumn**: Date column addition
  - `test_add_date_column`: Basic date column addition
  - `test_add_date_column_preserves_original`: Original columns preserved

- **TestChooseLatestDate**: Latest date selection
  - `test_choose_latest_date`: Basic date selection
  - `test_choose_latest_date_empty`: Empty DataFrame handling
  - `test_choose_latest_date_no_date_column`: Missing column handling

- **TestGenerateChirpReport**: Chirp report generation
  - `test_generate_chirp_report`: Basic markdown report generation
  - `test_generate_chirp_report_no_chirps`: Report with no chirps

## Test Coverage

The test suite covers:

- **Feature Extraction** (`test_features.py`): 17 tests covering WAV loading, MFCC extraction, spectral/temporal features
- **Email Functionality** (`test_email.py`): 9 tests covering configuration loading and SMTP sending
- **Report Generation** (`test_reporting.py`): 17 tests covering event loading, filtering, and report generation
- **ML Capture** (`test_capture_ml.py`): Tests for ML-based capture decision system

## Running Tests on Pi

While tests are designed to run locally, you can also run them on the Pi:

```bash
ssh prouty@raspberrypi.local "cd ~/projects/noisedetector && python3 -m pytest tests/ -v"
```

However, this is not necessary for development - all tests use mocking and don't require hardware.

## Notes

- Tests that require model files will skip if models aren't trained yet
- Use `pytest.skip()` for conditional tests
- Fixtures are automatically available to all tests
- Tests use mocking for external dependencies (SMTP, file I/O)
- Some tests create temporary files that are cleaned up automatically

