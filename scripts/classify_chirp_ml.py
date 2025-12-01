#!/usr/bin/env python3
"""
ML-based chirp classification using trained model.

This module provides classification functions that use the trained ML model
instead of simple spectral fingerprinting.
"""
import joblib
import json
import wave
from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np

# Import feature extraction functions
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from train_chirp_ml import (
    load_mono_wav,
    extract_mfcc_features,
    extract_additional_features,
    INT16_FULL_SCALE
)


def load_ml_model(config: Dict) -> Optional[Tuple]:
    """
    Load ML model, scaler, and metadata.
    
    Returns:
        Tuple of (model, scaler, metadata) or None if not found
    """
    metadata_file = Path(config["chirp_classification"].get("ml_metadata_file", "data/chirp_model_metadata.json"))
    
    if not metadata_file.exists():
        return None
    
    try:
        with metadata_file.open() as f:
            metadata = json.load(f)
        
        model_file = metadata_file.parent / metadata["model_file"]
        scaler_file = metadata_file.parent / metadata["scaler_file"]
        
        if not model_file.exists() or not scaler_file.exists():
            return None
        
        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
        
        return model, scaler, metadata
    except Exception as e:
        print(f"[WARN] Failed to load ML model: {e}")
        return None


def classify_clip_ml(clip_path: Path, model_info: Tuple) -> Tuple[bool, float, Optional[str]]:
    """
    Classify a clip using ML model.
    
    Args:
        clip_path: Path to WAV file
        model_info: Tuple from load_ml_model()
        
    Returns:
        Tuple of (is_chirp, confidence, error_message)
    """
    model, scaler, metadata = model_info
    
    try:
        # Load audio
        samples, sr = load_mono_wav(clip_path)
        
        # Extract features
        mfcc_features = extract_mfcc_features(samples, sr)
        additional_features = extract_additional_features(samples, sr)
        features = np.concatenate([mfcc_features, additional_features])
        
        # Reshape for single sample
        features = features.reshape(1, -1)
        
        # Scale
        features_scaled = scaler.transform(features)
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        is_chirp = bool(prediction == 1)
        confidence = float(probability[1] if is_chirp else probability[0])
        
        return is_chirp, confidence, None
        
    except Exception as e:
        return False, 0.0, str(e)


def classify_event_chunks_ml(event_chunks: list, sr: int, model_info: Tuple) -> Tuple[bool, float, Optional[str]]:
    """
    Classify event from raw PCM chunks using ML model.
    
    Args:
        event_chunks: List of raw PCM byte chunks
        sr: Sample rate
        model_info: Tuple from load_ml_model()
        
    Returns:
        Tuple of (is_chirp, confidence, error_message)
    """
    model, scaler, metadata = model_info
    
    try:
        # Convert chunks to samples
        samples_list = []
        for chunk in event_chunks:
            chunk_samples = np.frombuffer(chunk, dtype="<i2").astype(np.float32) / INT16_FULL_SCALE
            samples_list.append(chunk_samples)
        
        if not samples_list:
            return False, 0.0, "No audio data"
        
        # Concatenate all chunks
        samples = np.concatenate(samples_list)
        
        # Extract features
        mfcc_features = extract_mfcc_features(samples, sr)
        additional_features = extract_additional_features(samples, sr)
        features = np.concatenate([mfcc_features, additional_features])
        
        # Reshape for single sample
        features = features.reshape(1, -1)
        
        # Scale
        features_scaled = scaler.transform(features)
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        is_chirp = bool(prediction == 1)
        confidence = float(probability[1] if is_chirp else probability[0])
        
        return is_chirp, confidence, None
        
    except Exception as e:
        return False, 0.0, str(e)

