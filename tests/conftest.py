"""
Pytest configuration and shared fixtures.

This module provides:
- Common fixtures for test configuration
- Helper functions for test data creation
- Constants used across tests
"""
import sys
import tempfile
import wave
from pathlib import Path
from typing import Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import config_loader
import pytest
import numpy as np

# Test constants
TEST_SAMPLE_RATE = 16000
TEST_FREQUENCY = 440  # Hz
TEST_DURATION = 1.0  # seconds
INT16_FULL_SCALE = 32768.0


@pytest.fixture
def project_root_path():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def config(project_root_path):
    """Load configuration for testing."""
    config_path = project_root_path / "config.json"
    return config_loader.load_config(config_path if config_path.exists() else None)


@pytest.fixture
def data_dir(project_root_path):
    """Return the data directory path."""
    return project_root_path / "data"


# Helper functions for test data creation

def create_test_audio_samples(
    sample_rate: int = TEST_SAMPLE_RATE,
    duration: float = TEST_DURATION,
    frequency: float = TEST_FREQUENCY,
    amplitude: float = 0.5
) -> np.ndarray:
    """
    Create test audio samples (sine wave).
    
    Args:
        sample_rate: Sample rate in Hz
        duration: Duration in seconds
        frequency: Frequency in Hz
        amplitude: Amplitude (0.0 to 1.0)
        
    Returns:
        int16 array of audio samples
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    samples = (np.sin(2 * np.pi * frequency * t) * amplitude * INT16_FULL_SCALE).astype(np.int16)
    return samples


def create_test_wav_file(
    sample_rate: int = TEST_SAMPLE_RATE,
    duration: float = TEST_DURATION,
    frequency: float = TEST_FREQUENCY,
    amplitude: float = 0.5
) -> Tuple[Path, int]:
    """
    Create a temporary WAV file with test audio.
    
    Args:
        sample_rate: Sample rate in Hz
        duration: Duration in seconds
        frequency: Frequency in Hz
        amplitude: Amplitude (0.0 to 1.0)
        
    Returns:
        Tuple of (file_path, sample_rate)
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    
    samples = create_test_audio_samples(sample_rate, duration, frequency, amplitude)
    
    with wave.open(str(tmp_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    
    return tmp_path, sample_rate


def create_test_csv_file(rows: list, headers: list = None) -> Path:
    """
    Create a temporary CSV file with test data.
    
    Args:
        rows: List of rows (each row is a list of values)
        headers: Optional list of header names
        
    Returns:
        Path to temporary CSV file
    """
    import csv
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        tmp_path = Path(tmp.name)
        writer = csv.writer(tmp)
        
        if headers:
            writer.writerow(headers)
        
        for row in rows:
            writer.writerow(row)
    
    return tmp_path

