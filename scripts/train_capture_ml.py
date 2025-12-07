#!/usr/bin/env python3
"""
Train ML model for capture decision.

This trains a lightweight model to decide whether to capture an event,
based on whether events turned out to be chirps or noise.

Training data:
- Positive examples: Events that were classified as chirps
- Negative examples: Events that were classified as noise

Usage:
    python3 scripts/train_capture_ml.py
"""
import json
import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config_loader
import monitor

# Import feature extraction
from capture_ml import extract_capture_features, INT16_FULL_SCALE

# Import event loading
from core.reporting import load_events

OUTPUT_DIR = Path("data")
MODEL_FILE = OUTPUT_DIR / "capture_ml_model.joblib"
SCALER_FILE = OUTPUT_DIR / "capture_ml_scaler.joblib"
METADATA_FILE = OUTPUT_DIR / "capture_ml_metadata.json"


def load_event_audio(events_file: Path, clips_dir: Path, config: dict) -> List[Tuple[np.ndarray, int, bool]]:
    """
    Load audio from events and extract features.
    
    Returns:
        List of (samples, sample_rate, is_chirp) tuples
    """
    df = load_events(events_file)
    
    if df.empty or "clip_file" not in df.columns:
        print("No events found in CSV")
        return []
    
    # Filter to events with clips
    events_with_clips = df[
        (df["clip_file"].notna()) &
        (df["clip_file"] != "") &
        (df["is_chirp"].notna())
    ]
    
    if events_with_clips.empty:
        print("No events with clips found")
        return []
    
    results = []
    sample_rate = config["audio"]["sample_rate"]
    
    print(f"Loading audio from {len(events_with_clips)} events...")
    
    for idx, row in events_with_clips.iterrows():
        clip_file = row["clip_file"]
        is_chirp = str(row["is_chirp"]).upper() in ["TRUE", "True", "1"]
        
        # Resolve clip path
        clip_path = clips_dir / Path(clip_file).name
        if not clip_path.exists():
            clip_path = Path(clip_file)
        
        if not clip_path.exists():
            continue
        
        try:
            # Load audio - just get first chunk (0.5s) for capture decision
            # This simulates what we see when deciding to capture
            import wave
            with wave.open(str(clip_path), "rb") as wf:
                sr = wf.getframerate()
                # Read first 0.5 seconds
                frames_to_read = int(sr * 0.5)
                frames = wf.readframes(frames_to_read)
            
            if len(frames) == 0:
                continue
            
            samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / INT16_FULL_SCALE
            
            # Convert to mono if needed
            if wf.getnchannels() > 1:
                samples = samples.reshape(-1, wf.getnchannels()).mean(axis=1)
            
            # Resample if needed (simple linear interpolation)
            if sr != sample_rate:
                from scipy import signal
                num_samples = int(len(samples) * sample_rate / sr)
                samples = signal.resample(samples, num_samples)
            
            results.append((samples, sample_rate, is_chirp))
            
        except Exception as e:
            print(f"  Warning: Failed to load {clip_path.name}: {e}")
            continue
    
    print(f"Loaded {len(results)} audio samples")
    return results


def train_capture_model(events_file: Path, clips_dir: Path, config: dict, 
                        model_type: str = "rf") -> Tuple:
    """
    Train capture decision model.
    
    Args:
        events_file: Path to events.csv
        clips_dir: Directory containing clip files
        config: Configuration dictionary
        model_type: "rf" for Random Forest or "svm" for SVM
        
    Returns:
        Tuple of (model, scaler, metadata)
    """
    # Load training data
    audio_data = load_event_audio(events_file, clips_dir, config)
    
    if len(audio_data) < 10:
        raise ValueError(f"Need at least 10 events for training, got {len(audio_data)}")
    
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
    
    args = parser.parse_args()
    
    # Load config
    config = config_loader.load_config(args.config)
    
    events_file = Path(config["event_detection"]["events_file"])
    clips_dir = Path(config["event_clips"]["clips_dir"])
    
    if not events_file.exists():
        print(f"Error: Events file not found: {events_file}")
        sys.exit(1)
    
    # Train model
    try:
        model, scaler, metadata = train_capture_model(
            events_file, clips_dir, config, args.model_type
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

