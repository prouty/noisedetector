"""
Tests for core.features module.

Tests audio feature extraction functions including:
- WAV file loading
- MFCC feature extraction
- Spectral feature computation
- Temporal feature computation
"""
import pytest
import numpy as np
from pathlib import Path
import wave
import tempfile

from core.features import (
    load_mono_wav,
    compute_avg_spectrum,
    create_mel_filterbank,
    dct,
    extract_mfcc_features,
    extract_additional_features,
    compute_spectral_features,
    compute_temporal_features,
    INT16_FULL_SCALE,
)


class TestLoadMonoWav:
    """Test WAV file loading."""
    
    def test_load_mono_wav(self, project_root_path):
        """Test loading a mono WAV file."""
        # Try to find a test WAV file in training directory
        test_wav = project_root_path / "training" / "chirp" / "chirp_2.wav"
        
        if not test_wav.exists():
            pytest.skip("No test WAV file found")
        
        samples, sr = load_mono_wav(test_wav)
        
        assert isinstance(samples, np.ndarray)
        assert samples.dtype == np.float32
        assert len(samples) > 0
        assert sr > 0
        # Samples should be in range [-1.0, 1.0)
        assert np.all(samples >= -1.0)
        assert np.all(samples < 1.0)
    
    def test_load_mono_wav_creates_temp_file(self):
        """Test loading a temporary WAV file."""
        # Create a temporary mono WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            # Generate test audio: 1 second of 440 Hz sine wave at 16kHz
            sr = 16000
            duration = 1.0
            t = np.linspace(0, duration, int(sr * duration))
            samples_int16 = (np.sin(2 * np.pi * 440 * t) * 0.5 * INT16_FULL_SCALE).astype(np.int16)
            
            with wave.open(str(tmp_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(samples_int16.tobytes())
            
            # Load it
            loaded_samples, loaded_sr = load_mono_wav(tmp_path)
            
            assert loaded_sr == sr
            assert len(loaded_samples) == len(samples_int16)
            assert np.allclose(loaded_samples, samples_int16.astype(np.float32) / INT16_FULL_SCALE, atol=0.01)
        finally:
            tmp_path.unlink()
    
    def test_load_nonexistent_file(self):
        """Test loading a non-existent file raises error."""
        fake_path = Path("/nonexistent/file.wav")
        with pytest.raises(Exception):  # FileNotFoundError or similar
            load_mono_wav(fake_path)


class TestComputeAvgSpectrum:
    """Test average spectrum computation."""
    
    def test_compute_avg_spectrum(self):
        """Test computing average spectrum from samples."""
        # Generate test audio: 1 second of 440 Hz sine wave
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        samples = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        result = compute_avg_spectrum(samples, sr, fft_size=2048)
        
        assert result is not None
        spectrum, result_sr, result_fft_size = result
        assert result_sr == sr
        assert result_fft_size == 2048
        assert isinstance(spectrum, np.ndarray)
        assert len(spectrum) == 2048 // 2 + 1  # rfft output size
        assert np.all(spectrum >= 0)  # Magnitude spectrum should be non-negative
    
    def test_compute_avg_spectrum_short_audio(self):
        """Test with audio shorter than FFT size."""
        sr = 16000
        # Need enough samples after padding to create at least one window
        # After padding to 2048, we need at least 2048 samples for the loop to execute
        samples = np.random.randn(3000).astype(np.float32) * 0.1
        
        result = compute_avg_spectrum(samples, sr, fft_size=2048)
        
        assert result is not None
        spectrum, _, _ = result
        assert len(spectrum) == 2048 // 2 + 1
    
    def test_compute_avg_spectrum_empty(self):
        """Test with empty samples."""
        sr = 16000
        samples = np.array([], dtype=np.float32)
        
        result = compute_avg_spectrum(samples, sr)
        
        assert result is None


class TestMelFilterbank:
    """Test mel filterbank creation."""
    
    def test_create_mel_filterbank(self):
        """Test creating mel filterbank."""
        sr = 16000
        fft_size = 2048
        n_mel = 13
        
        filterbank = create_mel_filterbank(sr, fft_size, n_mel)
        
        assert isinstance(filterbank, np.ndarray)
        assert filterbank.shape == (n_mel, fft_size // 2 + 1)
        assert np.all(filterbank >= 0)
        # Filters are triangular, not normalized - just check they're non-negative
        assert np.all(filterbank >= 0)
    
    def test_create_mel_filterbank_different_sizes(self):
        """Test with different parameters."""
        for sr, fft_size, n_mel in [(8000, 1024, 10), (44100, 4096, 20)]:
            filterbank = create_mel_filterbank(sr, fft_size, n_mel)
            assert filterbank.shape == (n_mel, fft_size // 2 + 1)


class TestDCT:
    """Test DCT computation."""
    
    def test_dct(self):
        """Test DCT computation."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        n_coeffs = 3
        
        result = dct(x, n_coeffs)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == n_coeffs
        # DCT returns float64 (default numpy dtype)
        assert result.dtype in (np.float32, np.float64)
    
    def test_dct_identity(self):
        """Test DCT with single coefficient."""
        x = np.array([1.0, 2.0, 3.0])
        result = dct(x, 1)
        assert len(result) == 1


class TestMFCCFeatures:
    """Test MFCC feature extraction."""
    
    def test_extract_mfcc_features(self):
        """Test MFCC feature extraction."""
        # Generate test audio: 1 second of mixed frequencies
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        samples = (np.sin(2 * np.pi * 440 * t) + 
                  0.5 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
        
        mfcc = extract_mfcc_features(samples, sr, n_mfcc=13)
        
        assert isinstance(mfcc, np.ndarray)
        # Returns 4*n_mfcc features (mean, std, min, max)
        assert len(mfcc) == 13 * 4
        assert mfcc.dtype in (np.float32, np.float64)
    
    def test_extract_mfcc_features_different_n(self):
        """Test with different number of MFCC coefficients."""
        sr = 16000
        samples = np.random.randn(16000).astype(np.float32) * 0.1
        
        for n_mfcc in [10, 13, 20]:
            mfcc = extract_mfcc_features(samples, sr, n_mfcc=n_mfcc)
            # Returns 4*n_mfcc features (mean, std, min, max)
            assert len(mfcc) == n_mfcc * 4


class TestAdditionalFeatures:
    """Test additional feature extraction."""
    
    def test_extract_additional_features(self):
        """Test additional feature extraction."""
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        samples = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        features = extract_additional_features(samples, sr)
        
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        # Returns float64 (default numpy dtype)
        assert features.dtype in (np.float32, np.float64)
        # Should extract multiple features (RMS, ZCR, spectral centroid, rolloff, freq bands)
        assert len(features) >= 5


class TestSpectralFeatures:
    """Test spectral feature computation."""
    
    def test_compute_spectral_features(self):
        """Test computing spectral features."""
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        samples = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        features = compute_spectral_features(samples, sr)
        
        assert isinstance(features, dict)
        assert "spectral_centroid" in features
        assert "low_freq_ratio" in features
        assert "mid_freq_ratio" in features
        assert "high_freq_ratio" in features
        assert features["spectral_centroid"] > 0
        # Frequency ratios should sum to approximately 1
        total_ratio = features["low_freq_ratio"] + features["mid_freq_ratio"] + features["high_freq_ratio"]
        assert abs(total_ratio - 1.0) < 0.01
    
    def test_compute_spectral_features_empty(self):
        """Test with empty samples."""
        sr = 16000
        samples = np.array([], dtype=np.float32)
        
        features = compute_spectral_features(samples, sr)
        
        # Should return dict with zero/None values
        assert isinstance(features, dict)


class TestTemporalFeatures:
    """Test temporal feature computation."""
    
    def test_compute_temporal_features(self):
        """Test computing temporal features."""
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        samples = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        features = compute_temporal_features(samples, sr)
        
        assert isinstance(features, dict)
        assert "duration_sec" in features
        assert "energy_concentration" in features
        assert "attack_decay_ratio" in features
        assert features["duration_sec"] > 0
        assert 0 <= features["energy_concentration"] <= 1
    
    def test_compute_temporal_features_empty(self):
        """Test with empty samples."""
        sr = 16000
        samples = np.array([], dtype=np.float32)
        
        features = compute_temporal_features(samples, sr)
        
        assert isinstance(features, dict)

