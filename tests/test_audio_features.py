"""
Tests for audio feature extraction.

Tests feature extraction functions from core.features module including:
- MFCC feature extraction
- Spectral feature computation
- Temporal feature computation
"""
import pytest
import numpy as np

from core.features import (
    compute_avg_spectrum,
    create_mel_filterbank,
    dct,
    extract_mfcc_features,
    extract_additional_features,
    compute_spectral_features,
    compute_temporal_features,
    INT16_FULL_SCALE,
)

from tests.conftest import (
    create_test_audio_samples,
    TEST_SAMPLE_RATE,
    TEST_FREQUENCY,
    TEST_DURATION,
)


class TestComputeAvgSpectrum:
    """Test average spectrum computation."""
    
    def test_compute_avg_spectrum(self):
        """Test computing average spectrum from samples."""
        samples_int16 = create_test_audio_samples()
        samples = samples_int16.astype(np.float32) / INT16_FULL_SCALE
        
        result = compute_avg_spectrum(samples, TEST_SAMPLE_RATE, fft_size=2048)
        
        assert result is not None
        spectrum, result_sr, result_fft_size = result
        assert result_sr == TEST_SAMPLE_RATE
        assert result_fft_size == 2048
        assert isinstance(spectrum, np.ndarray)
        assert len(spectrum) == 2048 // 2 + 1  # rfft output size
        assert np.all(spectrum >= 0)  # Magnitude spectrum should be non-negative
    
    def test_compute_avg_spectrum_short_audio(self):
        """Test with audio shorter than FFT size."""
        # Need enough samples after padding to create at least one window
        samples = np.random.randn(3000).astype(np.float32) * 0.1
        
        result = compute_avg_spectrum(samples, TEST_SAMPLE_RATE, fft_size=2048)
        
        assert result is not None
        spectrum, _, _ = result
        assert len(spectrum) == 2048 // 2 + 1
    
    def test_compute_avg_spectrum_empty(self):
        """Test with empty samples."""
        samples = np.array([], dtype=np.float32)
        
        result = compute_avg_spectrum(samples, TEST_SAMPLE_RATE)
        
        assert result is None


class TestMelFilterbank:
    """Test mel filterbank creation."""
    
    def test_create_mel_filterbank(self):
        """Test creating mel filterbank."""
        fft_size = 2048
        n_mel = 13
        
        filterbank = create_mel_filterbank(TEST_SAMPLE_RATE, fft_size, n_mel)
        
        assert isinstance(filterbank, np.ndarray)
        assert filterbank.shape == (n_mel, fft_size // 2 + 1)
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
        # Generate test audio: mixed frequencies
        t = np.linspace(0, TEST_DURATION, int(TEST_SAMPLE_RATE * TEST_DURATION))
        samples = (np.sin(2 * np.pi * TEST_FREQUENCY * t) + 
                  0.5 * np.sin(2 * np.pi * TEST_FREQUENCY * 2 * t)).astype(np.float32)
        
        mfcc = extract_mfcc_features(samples, TEST_SAMPLE_RATE, n_mfcc=13)
        
        assert isinstance(mfcc, np.ndarray)
        # Returns 4*n_mfcc features (mean, std, min, max)
        assert len(mfcc) == 13 * 4
        assert mfcc.dtype in (np.float32, np.float64)
    
    @pytest.mark.parametrize("n_mfcc", [10, 13, 20])
    def test_extract_mfcc_features_different_n(self, n_mfcc):
        """Test with different number of MFCC coefficients."""
        samples = np.random.randn(TEST_SAMPLE_RATE).astype(np.float32) * 0.1
        
        mfcc = extract_mfcc_features(samples, TEST_SAMPLE_RATE, n_mfcc=n_mfcc)
        # Returns 4*n_mfcc features (mean, std, min, max)
        assert len(mfcc) == n_mfcc * 4


class TestAdditionalFeatures:
    """Test additional feature extraction."""
    
    def test_extract_additional_features(self):
        """Test additional feature extraction."""
        samples_int16 = create_test_audio_samples()
        samples = samples_int16.astype(np.float32) / INT16_FULL_SCALE
        
        features = extract_additional_features(samples, TEST_SAMPLE_RATE)
        
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        assert features.dtype in (np.float32, np.float64)
        # Should extract multiple features (RMS, ZCR, spectral centroid, rolloff, freq bands)
        assert len(features) >= 5


class TestSpectralFeatures:
    """Test spectral feature computation."""
    
    def test_compute_spectral_features(self):
        """Test computing spectral features."""
        samples_int16 = create_test_audio_samples()
        samples = samples_int16.astype(np.float32) / INT16_FULL_SCALE
        
        features = compute_spectral_features(samples, TEST_SAMPLE_RATE)
        
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
        samples = np.array([], dtype=np.float32)
        
        features = compute_spectral_features(samples, TEST_SAMPLE_RATE)
        
        assert isinstance(features, dict)


class TestTemporalFeatures:
    """Test temporal feature computation."""
    
    def test_compute_temporal_features(self):
        """Test computing temporal features."""
        samples_int16 = create_test_audio_samples()
        samples = samples_int16.astype(np.float32) / INT16_FULL_SCALE
        
        features = compute_temporal_features(samples, TEST_SAMPLE_RATE)
        
        assert isinstance(features, dict)
        assert "duration_sec" in features
        assert "energy_concentration" in features
        assert "attack_decay_ratio" in features
        assert features["duration_sec"] > 0
        assert 0 <= features["energy_concentration"] <= 1
    
    def test_compute_temporal_features_empty(self):
        """Test with empty samples."""
        samples = np.array([], dtype=np.float32)
        
        features = compute_temporal_features(samples, TEST_SAMPLE_RATE)
        
        assert isinstance(features, dict)

