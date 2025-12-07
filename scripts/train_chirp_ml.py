#!/usr/bin/env python3
"""
Train a lightweight ML model for chirp classification using MFCC features.

This replaces the simple spectral fingerprinting with a proper ML approach:
- Extracts MFCC features (more robust than raw spectrum)
- Trains a Random Forest classifier (fast inference on Pi)
- Supports incremental learning (can retrain with new examples)
- Exports to a format that runs efficiently on Raspberry Pi

Usage:
    python3 scripts/train_chirp_ml.py
    python3 scripts/train_chirp_ml.py --model-type svm  # Use SVM instead
"""
import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.features import (
    load_mono_wav,
    extract_mfcc_features,
    extract_additional_features,
)

CHIRP_TRAIN_DIR = Path("training/chirp")
NON_CHIRP_TRAIN_DIR = Path("training/not_chirp")
OUTPUT_DIR = Path("data")
MODEL_FILE = OUTPUT_DIR / "chirp_model.pkl"
SCALER_FILE = OUTPUT_DIR / "chirp_scaler.pkl"
METADATA_FILE = OUTPUT_DIR / "chirp_model_metadata.json"


# Feature extraction functions are now in core.features


def load_training_data(chirp_dir: Path, non_chirp_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load and extract features from training data."""
    X = []
    y = []
    
    # Load chirps (positive examples)
    chirp_files = sorted(chirp_dir.glob("chirp_*.wav"))
    print(f"Loading {len(chirp_files)} chirp examples...")
    
    for wav_path in chirp_files:
        try:
            samples, sr = load_mono_wav(wav_path)
            mfcc_features = extract_mfcc_features(samples, sr)
            additional_features = extract_additional_features(samples, sr)
            features = np.concatenate([mfcc_features, additional_features])
            X.append(features)
            y.append(1)  # Chirp = 1
        except Exception as e:
            print(f"Warning: Failed to process {wav_path.name}: {e}")
            continue
    
    # Load non-chirps (negative examples)
    non_chirp_files = sorted(non_chirp_dir.glob("not_chirp_*.wav"))
    print(f"Loading {len(non_chirp_files)} non-chirp examples...")
    
    for wav_path in non_chirp_files:
        try:
            samples, sr = load_mono_wav(wav_path)
            mfcc_features = extract_mfcc_features(samples, sr)
            additional_features = extract_additional_features(samples, sr)
            features = np.concatenate([mfcc_features, additional_features])
            X.append(features)
            y.append(0)  # Non-chirp = 0
        except Exception as e:
            print(f"Warning: Failed to process {wav_path.name}: {e}")
            continue
    
    if not X:
        raise ValueError("No training data loaded!")
    
    X_array = np.array(X)
    y_array = np.array(y)
    
    # Check minimum requirements
    n_samples = len(X_array)
    n_chirps = int(np.sum(y_array == 1))
    n_non_chirps = int(np.sum(y_array == 0))
    
    if n_samples < 2:
        raise ValueError(
            f"Insufficient training data: Only {n_samples} sample(s) available.\n"
            f"  Need at least 2 samples (ideally 10+ for good results).\n"
            f"  Current: {n_chirps} chirp(s), {n_non_chirps} non-chirp(s)\n"
            f"  Add more examples to:\n"
            f"    - training/chirp/chirp_*.wav (for chirp examples)\n"
            f"    - training/not_chirp/not_chirp_*.wav (for non-chirp examples)"
        )
    
    if n_chirps == 0:
        raise ValueError(
            "No chirp examples found! Need at least 1 chirp example in training/chirp/chirp_*.wav"
        )
    
    if n_non_chirps == 0:
        print(f"\n⚠️  WARNING: No non-chirp examples found in training/not_chirp/")
        print(f"  Model will only learn what chirps look like, not what to reject")
        print(f"  Recommendation: Add non-chirp examples for better accuracy")
    
    return X_array, y_array


def train_model(X: np.ndarray, y: np.ndarray, model_type: str = "random_forest") -> Tuple:
    """Train ML model and return model, scaler, and metrics."""
    print(f"\nTraining {model_type} model...")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
    elif model_type == "svm":
        model = SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Check if we have enough samples for cross-validation
    n_samples = len(X)
    min_samples_for_cv = 5
    
    if n_samples < min_samples_for_cv:
        print(f"\n⚠️  WARNING: Only {n_samples} training sample(s) available")
        print(f"  Cross-validation requires at least {min_samples_for_cv} samples")
        print(f"  Model will be trained but accuracy metrics will be limited")
        print(f"  Recommendation: Add more training examples to training/chirp/ and training/not_chirp/")
        print(f"    - Need at least {min_samples_for_cv - n_samples} more example(s)")
        if n_samples == 1:
            print(f"    - With only 1 sample, the model will always predict that class")
            print(f"    - This is not useful for classification - please add more training data")
    
    # Train
    model.fit(X_scaled, y)
    
    # Cross-validation score (only if we have enough samples)
    train_accuracy = model.score(X_scaled, y)
    
    if n_samples >= min_samples_for_cv:
        cv_scores = cross_val_score(model, X_scaled, y, cv=min(5, n_samples), scoring="accuracy")
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std() * 2
        print(f"\n✓ Model trained")
        print(f"  Training accuracy: {train_accuracy:.3f}")
        print(f"  Cross-validation accuracy: {cv_mean:.3f} (+/- {cv_std:.3f})")
    else:
        print(f"\n✓ Model trained (with limited validation)")
        print(f"  Training accuracy: {train_accuracy:.3f}")
        print(f"  Cross-validation: Skipped (need {min_samples_for_cv} samples, have {n_samples})")
        cv_mean = train_accuracy
        cv_std = 0.0
    
    # Feature importance (for Random Forest)
    if model_type == "random_forest":
        importances = model.feature_importances_
        top_features = np.argsort(importances)[-10:][::-1]
        print(f"\n  Top 10 most important features:")
        for idx in top_features:
            print(f"    Feature {idx}: {importances[idx]:.4f}")
    
    # Build metrics dictionary
    metrics = {
        "train_accuracy": float(train_accuracy),
        "n_features": int(X.shape[1]),
        "n_samples": int(n_samples),
        "n_chirps": int(np.sum(y == 1)),
        "n_non_chirps": int(np.sum(y == 0)),
    }
    
    if n_samples >= min_samples_for_cv:
        metrics["cv_accuracy"] = float(cv_mean)
        metrics["cv_std"] = float(cv_std)
    else:
        metrics["cv_accuracy"] = None
        metrics["cv_std"] = None
        metrics["warning"] = f"Insufficient samples for cross-validation (need {min_samples_for_cv}, have {n_samples})"
    
    return model, scaler, metrics


def main():
    parser = argparse.ArgumentParser(description="Train ML model for chirp classification")
    parser.add_argument("--model-type", choices=["random_forest", "svm"], default="random_forest",
                       help="Model type to train (default: random_forest)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                       help="Output directory for model files")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CHIRP ML MODEL TRAINING")
    print("=" * 60)
    print()
    
    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load training data
    try:
        X, y = load_training_data(CHIRP_TRAIN_DIR, NON_CHIRP_TRAIN_DIR)
        print(f"\n✓ Loaded {len(X)} training examples ({np.sum(y==1)} chirps, {np.sum(y==0)} non-chirps)")
        print(f"  Feature vector size: {X.shape[1]}")
    except ValueError as e:
        print(f"ERROR: {e}")
        return
    
    # Train model
    try:
        model, scaler, metrics = train_model(X, y, args.model_type)
    except Exception as e:
        print(f"ERROR: Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Save model and scaler
    model_path = args.output_dir / MODEL_FILE.name
    scaler_path = args.output_dir / SCALER_FILE.name
    metadata_path = args.output_dir / METADATA_FILE.name
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    # Save metadata
    metadata = {
        "model_type": args.model_type,
        "model_file": str(model_path.name),
        "scaler_file": str(scaler_path.name),
        "sample_rate": 16000,  # Assumed from config
        "metrics": metrics,
        "feature_extraction": "mfcc_plus_temporal",
    }
    
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)
    
    print()
    print("=" * 60)
    print(f"✓ Model saved to {model_path}")
    print(f"✓ Scaler saved to {scaler_path}")
    print(f"✓ Metadata saved to {metadata_path}")
    print()
    print("Next steps:")
    print("  1. Test the model: python3 scripts/test_chirp_ml.py --training")
    print("  2. Compare with fingerprint: make compare-classifiers")
    print("  3. Deploy to Pi: make deploy-ml-restart")
    print("  4. Enable ML in config.json: set 'use_ml_classifier': true")
    print("=" * 60)


if __name__ == "__main__":
    main()

