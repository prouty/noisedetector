"""
Tests for audio I/O operations.

Tests WAV file loading functionality from core.features module.
"""
import pytest
import numpy as np
from pathlib import Path

from core.features import load_mono_wav, INT16_FULL_SCALE

from tests.conftest import (
    create_test_wav_file,
    create_test_audio_samples,
    TEST_SAMPLE_RATE,
    TEST_FREQUENCY,
    TEST_DURATION,
)


class TestLoadMonoWav:
    """Test WAV file loading."""
    
    def test_load_mono_wav(self):
        """Test loading a mono WAV file."""
        tmp_path, sr = create_test_wav_file(duration=0.1)
        
        try:
            samples, loaded_sr = load_mono_wav(tmp_path)
            
            assert isinstance(samples, np.ndarray)
            assert samples.dtype == np.float32
            assert len(samples) > 0
            assert loaded_sr == sr
            assert np.all(samples >= -1.0)
            assert np.all(samples < 1.0)
        finally:
            tmp_path.unlink()
    
    def test_load_mono_wav_roundtrip(self):
        """Test that loaded samples match original audio data."""
        tmp_path, sr = create_test_wav_file()
        samples_int16 = create_test_audio_samples(sr, TEST_DURATION, TEST_FREQUENCY)
        
        try:
            loaded_samples, loaded_sr = load_mono_wav(tmp_path)
            
            assert loaded_sr == sr
            assert len(loaded_samples) == len(samples_int16)
            expected = samples_int16.astype(np.float32) / INT16_FULL_SCALE
            assert np.allclose(loaded_samples, expected, atol=0.01)
        finally:
            tmp_path.unlink()
    
    def test_load_nonexistent_file(self):
        """Test loading a non-existent file raises error."""
        fake_path = Path("/nonexistent/file.wav")
        with pytest.raises(Exception):  # FileNotFoundError or similar
            load_mono_wav(fake_path)

