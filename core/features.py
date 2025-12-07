"""
Feature extraction for audio analysis.

This module provides functions for extracting features from audio samples,
including MFCC features, spectral features, and temporal features.

Single Responsibility: Audio feature extraction.
"""
import wave
from pathlib import Path
from typing import Tuple, Dict, Optional
import numpy as np

# Constant for int16 to float32 conversion (2^15)
INT16_FULL_SCALE = 32768.0


def load_mono_wav(path: Path) -> Tuple[np.ndarray, int]:
    """
    Load WAV file as mono float32 array.
    
    Args:
        path: Path to WAV file
        
    Returns:
        Tuple of (samples, sample_rate)
        - samples: float32 array in range [-1.0, 1.0)
        - sample_rate: Sample rate in Hz
    """
    with wave.open(str(path), "rb") as wf:
        nch = wf.getnchannels()
        sr = wf.getframerate()
        nframes = wf.getnframes()
        frames = wf.readframes(nframes)
    
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / INT16_FULL_SCALE
    
    if nch > 1:
        samples = samples.reshape(-1, nch).mean(axis=1)
    
    return samples, sr


def compute_avg_spectrum(samples: np.ndarray, sr: int, fft_size: int = 2048) -> Optional[Tuple[np.ndarray, int, int]]:
    """
    Compute average spectrum from audio samples.
    
    Args:
        samples: Audio samples (float32, -1 to 1)
        sr: Sample rate
        fft_size: FFT size for analysis
        
    Returns:
        Tuple of (average_spectrum, sample_rate, fft_size) or None if invalid
    """
    if samples.shape[0] < fft_size:
        pad = fft_size - samples.shape[0]
        samples = np.pad(samples, (0, pad))
    
    hop = fft_size // 2
    window = np.hanning(fft_size)
    specs = []
    
    for start in range(0, len(samples) - fft_size, hop):
        chunk = samples[start:start + fft_size] * window
        spec = np.abs(np.fft.rfft(chunk))
        specs.append(spec)
    
    if not specs:
        return None
    
    avg_spec = np.mean(specs, axis=0)
    norm = np.linalg.norm(avg_spec) + 1e-9
    avg_spec = avg_spec / norm
    return avg_spec, sr, fft_size


def create_mel_filterbank(sr: int, fft_size: int, n_mel: int) -> np.ndarray:
    """
    Create mel-scale filterbank (simplified version).
    
    Args:
        sr: Sample rate
        fft_size: FFT size
        n_mel: Number of mel filters
        
    Returns:
        Filterbank matrix of shape (n_mel, n_fft_bins)
    """
    n_fft_bins = fft_size // 2 + 1
    freq_bins = np.arange(n_fft_bins) * sr / fft_size
    
    # Mel scale: m = 2595 * log10(1 + f/700)
    mel_max = 2595 * np.log10(1 + sr / 2 / 700)
    mel_points = np.linspace(0, mel_max, n_mel + 2)
    
    # Convert back to Hz
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    
    # Create triangular filters
    filters = np.zeros((n_mel, n_fft_bins))
    
    for i in range(n_mel):
        start = hz_points[i]
        center = hz_points[i + 1]
        end = hz_points[i + 2]
        
        for j, freq in enumerate(freq_bins):
            if start <= freq < center:
                filters[i, j] = (freq - start) / (center - start)
            elif center <= freq < end:
                filters[i, j] = (end - freq) / (end - center)
    
    return filters


def dct(x: np.ndarray, n_coeffs: int) -> np.ndarray:
    """
    Discrete Cosine Transform (Type II).
    
    Args:
        x: Input array
        n_coeffs: Number of DCT coefficients to compute
        
    Returns:
        DCT coefficients
    """
    N = x.shape[-1]
    coeffs = np.arange(n_coeffs)
    n = np.arange(N)
    
    # DCT-II: X[k] = sum(x[n] * cos(pi * k * (2n + 1) / (2N)))
    dct_matrix = np.cos(np.pi * np.outer(coeffs, 2 * n + 1) / (2 * N))
    
    # Apply scaling
    dct_matrix[0] *= 1 / np.sqrt(2)
    dct_matrix *= np.sqrt(2 / N)
    
    return np.dot(x, dct_matrix.T)


def extract_mfcc_features(samples: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    """
    Extract MFCC features from audio samples.
    
    MFCCs (Mel-frequency Cepstral Coefficients) are more robust than raw spectrum
    for audio classification. They capture perceptual characteristics better.
    
    Args:
        samples: Audio samples (float32, -1 to 1)
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients (default 13)
        
    Returns:
        Feature vector (mean, std, min, max of MFCCs across time)
    """
    # Simple MFCC-like features using FFT and mel-scale approximation
    # For Pi compatibility, we'll use a lightweight implementation
    
    fft_size = 2048
    hop_size = fft_size // 4
    
    if len(samples) < fft_size:
        # Pad short clips
        samples = np.pad(samples, (0, fft_size - len(samples)))
    
    # Compute STFT
    window = np.hanning(fft_size)
    frames = []
    
    for start in range(0, len(samples) - fft_size, hop_size):
        frame = samples[start:start + fft_size] * window
        fft = np.fft.rfft(frame)
        magnitude = np.abs(fft)
        frames.append(magnitude)
    
    if not frames:
        # Fallback for very short clips
        frame = np.pad(samples, (0, fft_size - len(samples))) * window
        fft = np.fft.rfft(frame)
        magnitude = np.abs(fft)
        frames = [magnitude]
    
    frames = np.array(frames)
    
    # Apply mel-scale filterbank (simplified)
    n_mel = 26
    mel_filters = create_mel_filterbank(sr, fft_size, n_mel)
    
    # Apply filters
    mel_spectrum = np.dot(frames, mel_filters.T)
    
    # Log scale
    mel_spectrum = np.log(mel_spectrum + 1e-10)
    
    # DCT to get MFCCs
    mfccs = dct(mel_spectrum, n_mfcc)
    
    # Aggregate across time: mean, std, min, max
    features = np.concatenate([
        np.mean(mfccs, axis=0),      # Mean MFCCs
        np.std(mfccs, axis=0),       # Std MFCCs
        np.min(mfccs, axis=0),       # Min MFCCs
        np.max(mfccs, axis=0),       # Max MFCCs
    ])
    
    return features


def extract_additional_features(samples: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract additional temporal and spectral features.
    
    Args:
        samples: Audio samples (float32, -1 to 1)
        sr: Sample rate
        
    Returns:
        Feature vector with temporal and spectral features
    """
    features = []
    
    # Temporal features
    rms = np.sqrt(np.mean(samples ** 2))
    features.append(rms)
    
    zero_crossing_rate = np.mean(np.abs(np.diff(np.sign(samples)))) / 2
    features.append(zero_crossing_rate)
    
    # Spectral features
    fft = np.fft.rfft(samples[:2048] if len(samples) >= 2048 else np.pad(samples, (0, 2048 - len(samples))))
    magnitude = np.abs(fft)
    
    # Spectral centroid
    freqs = np.arange(len(magnitude)) * sr / (2 * len(magnitude))
    spectral_centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-10)
    features.append(spectral_centroid / 4000.0)  # Normalize
    
    # Spectral rolloff (frequency below which 85% of energy is contained)
    cumsum = np.cumsum(magnitude)
    rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
    spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
    features.append(spectral_rolloff / 4000.0)  # Normalize
    
    # Energy in frequency bands
    freq_resolution = sr / (2 * len(magnitude))
    low_band = int(500 / freq_resolution)
    mid_band = int(2000 / freq_resolution)
    
    total_energy = np.sum(magnitude)
    if total_energy > 0:
        low_energy = np.sum(magnitude[:low_band]) / total_energy
        mid_energy = np.sum(magnitude[low_band:mid_band]) / total_energy
        high_energy = np.sum(magnitude[mid_band:]) / total_energy
        features.extend([low_energy, mid_energy, high_energy])
    else:
        features.extend([0, 0, 0])
    
    return np.array(features)


def compute_spectral_features(samples: np.ndarray, sample_rate: int, fft_size: int = 2048) -> Dict:
    """
    Compute spectral features from audio samples.
    
    Args:
        samples: Audio samples (float32, -1 to 1)
        sample_rate: Sample rate
        fft_size: FFT size for analysis
        
    Returns:
        Dictionary with spectral features:
        - spectral_centroid: Spectral centroid in Hz
        - low_freq_ratio: Ratio of energy in low frequencies (< 500 Hz)
        - mid_freq_ratio: Ratio of energy in mid frequencies (500-1000 Hz)
        - high_freq_ratio: Ratio of energy in high frequencies (> 1000 Hz)
    """
    if len(samples) < fft_size:
        pad = fft_size - len(samples)
        samples = np.pad(samples, (0, pad))
    
    hop = fft_size // 2
    window = np.hanning(fft_size)
    specs = []
    
    for start in range(0, len(samples) - fft_size, hop):
        chunk = samples[start:start + fft_size] * window
        spec = np.abs(np.fft.rfft(chunk))
        specs.append(spec)
    
    if not specs:
        return {}
    
    avg_spec = np.mean(specs, axis=0)
    avg_spec = avg_spec / (np.linalg.norm(avg_spec) + 1e-9)
    
    # Calculate features
    freq_resolution = sample_rate / fft_size
    frequencies = np.arange(len(avg_spec)) * freq_resolution
    
    # Spectral centroid
    magnitude = np.abs(avg_spec)
    total_magnitude = np.sum(magnitude)
    spectral_centroid = np.sum(frequencies * magnitude) / total_magnitude if total_magnitude > 0 else 0
    
    # Energy in frequency bands
    fan_noise_max_bin = int(500 / freq_resolution)
    chirp_min_bin = int(1000 / freq_resolution)
    
    low_freq_energy = np.sum(avg_spec[:fan_noise_max_bin])
    mid_freq_energy = np.sum(avg_spec[fan_noise_max_bin:chirp_min_bin])
    high_freq_energy = np.sum(avg_spec[chirp_min_bin:])
    total_energy = np.sum(avg_spec)
    
    return {
        "spectral_centroid": float(spectral_centroid),
        "low_freq_ratio": float(low_freq_energy / total_energy) if total_energy > 0 else 0,
        "mid_freq_ratio": float(mid_freq_energy / total_energy) if total_energy > 0 else 0,
        "high_freq_ratio": float(high_freq_energy / total_energy) if total_energy > 0 else 0,
    }


def compute_temporal_features(samples: np.ndarray, sample_rate: int, chunk_duration: float = 0.5) -> Dict:
    """
    Compute temporal features from audio samples.
    
    Args:
        samples: Audio samples (float32, -1 to 1)
        sample_rate: Sample rate
        chunk_duration: Duration of chunks for analysis (seconds)
        
    Returns:
        Dictionary with temporal features:
        - duration_sec: Total duration
        - energy_concentration: Ratio of energy in first half vs second half
        - attack_decay_ratio: Ratio of attack time to decay time (None if not calculable)
    """
    chunk_samples = int(sample_rate * chunk_duration)
    chunk_rms_values = []
    
    for i in range(0, len(samples), chunk_samples):
        chunk = samples[i:i + chunk_samples]
        if len(chunk) > 0:
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            chunk_rms_values.append(rms)
    
    if len(chunk_rms_values) < 2:
        return {
            "duration_sec": len(samples) / sample_rate,
            "energy_concentration": 0.5,
            "attack_decay_ratio": None,
        }
    
    # Energy concentration
    mid_point = len(chunk_rms_values) // 2
    first_half_energy = sum(r**2 for r in chunk_rms_values[:mid_point])
    second_half_energy = sum(r**2 for r in chunk_rms_values[mid_point:])
    total_energy = first_half_energy + second_half_energy
    energy_concentration = first_half_energy / total_energy if total_energy > 0 else 0.5
    
    # Attack/decay
    peak_idx = chunk_rms_values.index(max(chunk_rms_values))
    attack_decay_ratio = None
    if peak_idx > 0 and peak_idx < len(chunk_rms_values) - 1:
        peak_value = chunk_rms_values[peak_idx]
        attack_threshold = peak_value * 0.9
        decay_threshold = peak_value * 0.1
        
        attack_time = peak_idx
        for i in range(peak_idx):
            if chunk_rms_values[i] >= attack_threshold:
                attack_time = peak_idx - i
                break
        
        decay_time = len(chunk_rms_values) - peak_idx - 1
        for i in range(peak_idx + 1, len(chunk_rms_values)):
            if chunk_rms_values[i] <= decay_threshold:
                decay_time = i - peak_idx
                break
        
        if decay_time > 0:
            attack_decay_ratio = attack_time / decay_time
    
    return {
        "duration_sec": len(samples) / sample_rate,
        "energy_concentration": float(energy_concentration),
        "attack_decay_ratio": float(attack_decay_ratio) if attack_decay_ratio else None,
    }

