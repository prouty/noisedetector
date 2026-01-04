#!/usr/bin/env python3
"""
Train ML model for capture decision.

This trains a lightweight model to decide whether to capture an event,
based on confirmed training data.

Training data:
- Positive examples: Audio files in training/chirp/ (confirmed chirps)
- Negative examples: Audio files in training/not_chirp/ (confirmed non-chirps)

Usage:
    python3 scripts/train_capture_ml.py
"""
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config_loader

# Import feature extraction
from capture_ml import extract_capture_features, INT16_FULL_SCALE

OUTPUT_DIR = Path("data")
MODEL_FILE = OUTPUT_DIR / "capture_ml_model.joblib"
SCALER_FILE = OUTPUT_DIR / "capture_ml_scaler.joblib"
METADATA_FILE = OUTPUT_DIR / "capture_ml_metadata.json"


def load_training_audio(chirp_dir: Path, not_chirp_dir: Path, config: dict) -> List[Tuple[np.ndarray, int, bool]]:
    """
    Load audio from training directories.
    
    Args:
        chirp_dir: Directory containing confirmed chirp audio files
        not_chirp_dir: Directory containing confirmed non-chirp audio files
        config: Configuration dictionary
    
    Returns:
        List of (samples, sample_rate, is_chirp) tuples
    """
    results = []
    sample_rate = config["audio"]["sample_rate"]
    
    # Load positive examples (chirps)
    chirp_files = list(chirp_dir.glob("*.wav")) if chirp_dir.exists() else []
    print(f"Found {len(chirp_files)} chirp files in {chirp_dir}")
    
    # Load negative examples (not chirps)
    not_chirp_files = list(not_chirp_dir.glob("*.wav")) if not_chirp_dir.exists() else []
    print(f"Found {len(not_chirp_files)} non-chirp files in {not_chirp_dir}")
    
    total_files = len(chirp_files) + len(not_chirp_files)
    if total_files == 0:
        print("No training files found!")
        return []
    
    print(f"Loading audio from {total_files} training files...")
    
    # Process chirp files (positive examples)
    for clip_path in chirp_files:
        try:
            samples, sr = _load_audio_chunk(clip_path, sample_rate)
            if samples is not None:
                results.append((samples, sr, True))
        except Exception as e:
            print(f"  Warning: Failed to load {clip_path.name}: {e}")
            continue
    
    # Process non-chirp files (negative examples)
    for clip_path in not_chirp_files:
        try:
            samples, sr = _load_audio_chunk(clip_path, sample_rate)
            if samples is not None:
                results.append((samples, sr, False))
        except Exception as e:
            print(f"  Warning: Failed to load {clip_path.name}: {e}")
            continue
    
    chirp_count = sum(1 for _, _, is_chirp in results if is_chirp)
    not_chirp_count = len(results) - chirp_count
    print(f"Loaded {len(results)} audio samples ({chirp_count} chirps, {not_chirp_count} non-chirps)")
    return results


def _load_audio_chunk(clip_path: Path, target_sample_rate: int, chunk_duration: float = 0.5) -> Optional[Tuple[np.ndarray, int]]:
    """
    Load first chunk of audio from a WAV file.
    
    Args:
        clip_path: Path to WAV file
        target_sample_rate: Target sample rate
        chunk_duration: Duration of chunk to read in seconds (default: 0.5)
    
    Returns:
        Tuple of (samples, sample_rate) or (None, None) if failed
    """
    import wave
    
    with wave.open(str(clip_path), "rb") as wf:
        sr = wf.getframerate()
        # Read first chunk (0.5 seconds by default)
        frames_to_read = int(sr * chunk_duration)
        frames = wf.readframes(frames_to_read)
        
        if len(frames) == 0:
            return None, None
        
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / INT16_FULL_SCALE
        
        # Convert to mono if needed
        if wf.getnchannels() > 1:
            samples = samples.reshape(-1, wf.getnchannels()).mean(axis=1)
        
        # Resample if needed (simple linear interpolation)
        if sr != target_sample_rate:
            from scipy import signal
            num_samples = int(len(samples) * target_sample_rate / sr)
            samples = signal.resample(samples, num_samples)
        
        return samples, target_sample_rate


def train_capture_model(chirp_dir: Path, not_chirp_dir: Path, config: dict, 
                        model_type: str = "rf") -> Tuple:
    """
    Train capture decision model.
    
    Args:
        chirp_dir: Directory containing confirmed chirp audio files
        not_chirp_dir: Directory containing confirmed non-chirp audio files
        config: Configuration dictionary
        model_type: "rf" for Random Forest or "svm" for SVM
        
    Returns:
        Tuple of (model, scaler, metadata)
    """
    # Load training data
    audio_data = load_training_audio(chirp_dir, not_chirp_dir, config)
    
    if len(audio_data) < 10:
        raise ValueError(f"Need at least 10 training samples, got {len(audio_data)}")
    
    # Extract features
    print("Extracting features...")
    X = []
    y = []
    
    for samples, sr, is_chirp in audio_data:
        try:
            features = extract_capture_features(samples, sr)
            X.append(features)
            y.append(1 if is_chirp else 0)
        except Exception as e:
            print(f"  Warning: Feature extraction failed: {e}")
            continue
    
    if len(X) < 10:
        raise ValueError(f"Need at least 10 valid feature vectors, got {len(X)}")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Training on {len(X)} samples ({np.sum(y)} positive, {len(y) - np.sum(y)} negative)")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    if model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
    else:
        from sklearn.svm import SVC
        model = SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        )
    
    print(f"Training {model_type.upper()} model...")
    model.fit(X_scaled, y)
    
    # Evaluate
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")
    print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Predictions for metrics
    predictions = model.predict(X_scaled)
    precision = np.sum((predictions == 1) & (y == 1)) / (np.sum(predictions == 1) + 1e-10)
    recall = np.sum((predictions == 1) & (y == 1)) / (np.sum(y == 1) + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    print(f"Training metrics:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1: {f1:.3f}")
    
    # Metadata
    metadata = {
        "model_type": model_type,
        "n_features": X.shape[1],
        "n_samples": len(X),
        "n_positive": int(np.sum(y)),
        "n_negative": int(len(y) - np.sum(y)),
        "cv_accuracy": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "model_file": MODEL_FILE.name,
        "scaler_file": SCALER_FILE.name
    }
    
    return model, scaler, metadata


def main():
    parser = argparse.ArgumentParser(description="Train ML model for capture decision")
    parser.add_argument("--model-type", choices=["rf", "svm"], default="rf",
                       help="Model type: rf (Random Forest) or svm (SVM)")
    parser.add_argument("--config", type=Path, help="Path to config.json")
    parser.add_argument("--chirp-dir", type=Path, default=Path("training/chirp"),
                       help="Directory containing confirmed chirp audio files (default: training/chirp)")
    parser.add_argument("--not-chirp-dir", type=Path, default=Path("training/not_chirp"),
                       help="Directory containing confirmed non-chirp audio files (default: training/not_chirp)")
    
    args = parser.parse_args()
    
    # Load config
    config = config_loader.load_config(args.config)
    
    chirp_dir = args.chirp_dir
    not_chirp_dir = args.not_chirp_dir
    
    if not chirp_dir.exists():
        print(f"Error: Chirp directory not found: {chirp_dir}")
        sys.exit(1)
    
    if not not_chirp_dir.exists():
        print(f"Error: Non-chirp directory not found: {not_chirp_dir}")
        sys.exit(1)
    
    # Train model
    try:
        model, scaler, metadata = train_capture_model(
            chirp_dir, not_chirp_dir, config, args.model_type
        )
    except Exception as e:
        print(f"Error training model: {e}")
        sys.exit(1)
    
    # Save model
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    
    with METADATA_FILE.open("w") as f:
        json.dump(metadata, f, indent=2)
    
    print()
    print("=" * 60)
    print("Model saved successfully!")
    print("=" * 60)
    print(f"Model: {MODEL_FILE}")
    print(f"Scaler: {SCALER_FILE}")
    print(f"Metadata: {METADATA_FILE}")
    print()
    print("To use ML capture decision, set in config.json:")
    print('  "event_detection": {')
    print('    "use_ml_capture": true')
    print('  }')


if __name__ == "__main__":
    main()

