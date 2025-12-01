#!/usr/bin/env python3
"""
ML-based capture decision system.

This module provides lightweight ML-based decision making for whether to capture
an event, replacing the simple threshold-based approach.

Features:
- Fast inference on single 0.5s chunks
- Lightweight features (no full MFCC)
- Trained on actual event outcomes
"""
import joblib
import json
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

INT16_FULL_SCALE = 32768.0


def extract_capture_features(samples: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract lightweight features from a single audio chunk for capture decision.
    
    This is optimized for speed - runs on every 0.5s chunk in real-time.
    Uses simpler features than full classification to keep inference fast.
    
    Args:
        samples: Audio samples (float32, -1 to 1)
        sr: Sample rate
        
    Returns:
        Feature vector (15-20 features)
    """
    features = []
    
    # Temporal features (fast)
    rms = np.sqrt(np.mean(samples ** 2))
    features.append(rms)
    
    # Peak level
    peak = np.max(np.abs(samples))
    features.append(peak)
    
    # Zero crossing rate
    zero_crossings = np.sum(np.abs(np.diff(np.sign(samples)))) / (2 * len(samples))
    features.append(zero_crossings)
    
    # Spectral features (single FFT, fast)
    fft_size = min(2048, len(samples))
    if len(samples) < fft_size:
        padded = np.pad(samples, (0, fft_size - len(samples)))
    else:
        padded = samples[:fft_size]
    
    window = np.hanning(len(padded))
    fft = np.fft.rfft(padded * window)
    magnitude = np.abs(fft)
    
    # Frequency resolution
    freqs = np.arange(len(magnitude)) * sr / (2 * len(magnitude))
    
    # Spectral centroid (brightness)
    total_energy = np.sum(magnitude)
    if total_energy > 1e-10:
        spectral_centroid = np.sum(freqs * magnitude) / total_energy
        features.append(spectral_centroid / 4000.0)  # Normalize
    else:
        features.append(0.0)
    
    # Spectral rolloff (85% energy)
    cumsum = np.cumsum(magnitude)
    if cumsum[-1] > 1e-10:
        rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
        spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
        features.append(spectral_rolloff / 4000.0)  # Normalize
    else:
        features.append(0.0)
    
    # Energy in frequency bands (chirps are typically mid-high freq)
    freq_resolution = sr / (2 * len(magnitude))
    low_band = int(500 / freq_resolution) if freq_resolution > 0 else 0
    mid_band = int(2000 / freq_resolution) if freq_resolution > 0 else 0
    high_band = int(4000 / freq_resolution) if freq_resolution > 0 else len(magnitude)
    
    if total_energy > 1e-10:
        low_energy = np.sum(magnitude[:low_band]) / total_energy
        mid_energy = np.sum(magnitude[low_band:mid_band]) / total_energy
        high_energy = np.sum(magnitude[mid_band:high_band]) / total_energy
        very_high_energy = np.sum(magnitude[high_band:]) / total_energy
        features.extend([low_energy, mid_energy, high_energy, very_high_energy])
    else:
        features.extend([0, 0, 0, 0])
    
    # Spectral flatness (noise vs tone)
    if len(magnitude) > 0 and np.mean(magnitude) > 1e-10:
        geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-10)))
        arithmetic_mean = np.mean(magnitude)
        spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)
        features.append(spectral_flatness)
    else:
        features.append(0.0)
    
    # Spectral spread (variance of spectrum)
    if total_energy > 1e-10:
        spectral_spread = np.sqrt(np.sum(((freqs - features[3] * 4000.0) ** 2) * magnitude) / total_energy) / 4000.0
        features.append(spectral_spread)
    else:
        features.append(0.0)
    
    # Attack detection (energy increase rate) - compare first vs second half
    if len(samples) > 1:
        first_half = samples[:len(samples)//2]
        second_half = samples[len(samples)//2:]
        first_energy = np.mean(first_half ** 2)
        second_energy = np.mean(second_half ** 2)
        if first_energy > 1e-10:
            attack_ratio = second_energy / (first_energy + 1e-10)
            features.append(attack_ratio)
        else:
            features.append(0.0)
    else:
        features.append(0.0)
    
    return np.array(features)


def load_capture_model(config: dict) -> Optional[Tuple]:
    """
    Load capture decision ML model.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (model, scaler, metadata) or None if not found
    """
    event_detection = config.get("event_detection", {})
    model_file = Path(event_detection.get("capture_ml_model_file", "data/capture_ml_model.joblib"))
    scaler_file = Path(event_detection.get("capture_ml_scaler_file", "data/capture_ml_scaler.joblib"))
    metadata_file = Path(event_detection.get("capture_ml_metadata_file", "data/capture_ml_metadata.json"))
    
    if not model_file.exists() or not scaler_file.exists() or not metadata_file.exists():
        return None
    
    try:
        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
        
        with metadata_file.open() as f:
            metadata = json.load(f)
        
        return model, scaler, metadata
    except Exception as e:
        print(f"[WARN] Failed to load capture ML model: {e}")
        return None


def should_capture_chunk(samples: np.ndarray, sr: int, model_info: Optional[Tuple], 
                        baseline_rms_db: Optional[float] = None) -> Tuple[bool, float]:
    """
    Decide if we should start capturing an event based on ML model.
    
    Args:
        samples: Audio samples from current chunk (float32, -1 to 1)
        sr: Sample rate
        model_info: Tuple from load_capture_model() or None
        baseline_rms_db: Current baseline level (for fallback)
        
    Returns:
        Tuple of (should_capture, confidence)
    """
    if model_info is None:
        # Fallback to threshold-based if model not available
        if baseline_rms_db is not None:
            rms_db = 20 * np.log10(np.sqrt(np.mean(samples ** 2)) + 1e-10)
            threshold_db = baseline_rms_db + 10.0  # Default threshold
            # Ensure we return Python bool, not numpy bool
            return bool(rms_db > threshold_db), 0.5
        return False, 0.0
    
    model, scaler, metadata = model_info
    
    try:
        # Extract features
        features = extract_capture_features(samples, sr)
        
        # Reshape for single sample
        features = features.reshape(1, -1)
        
        # Scale
        features_scaled = scaler.transform(features)
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        # Ensure we return Python bool, not numpy bool
        should_capture = bool(int(prediction == 1))
        confidence = float(probability[1] if should_capture else probability[0])
        
        return should_capture, confidence
        
    except Exception as e:
        print(f"[WARN] Capture ML prediction failed: {e}")
        # Fallback to threshold
        if baseline_rms_db is not None:
            rms_db = 20 * np.log10(np.sqrt(np.mean(samples ** 2)) + 1e-10)
            threshold_db = baseline_rms_db + 10.0
            # Ensure we return Python bool, not numpy bool
            return bool(rms_db > threshold_db), 0.5
        return False, 0.0

