#!/usr/bin/env python3
"""
Test ML model on individual clips or training data.

Usage:
    python3 scripts/test_chirp_ml.py training/chirp/chirp_1.wav
    python3 scripts/test_chirp_ml.py clips/clip_2025-01-01_12-00-00.wav
"""
import sys
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config_loader
from core.classifier import load_chirp_ml_model, classify_clip_ml


def test_clip(clip_path: Path, config_path: Path = None):
    """Test ML model on a single clip."""
    config = config_loader.load_config(config_path)
    ml_model_info = load_chirp_ml_model(config)
    
    if ml_model_info is None:
        print("ERROR: ML model not found. Run 'make train-ml' first.")
        return
    
    if not clip_path.exists():
        print(f"ERROR: Clip file not found: {clip_path}")
        return
    
    print(f"Testing clip: {clip_path}")
    print()
    
    is_chirp, confidence, error = classify_clip_ml(clip_path, ml_model_info)
    
    if error:
        print(f"ERROR: {error}")
        return
    
    result = "CHIRP" if is_chirp else "NOT CHIRP"
    print(f"Classification: {result}")
    print(f"Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
    print()
    
    if is_chirp:
        print("✓ Classified as chirp")
    else:
        print("✗ Classified as not chirp")


def test_training_data(config_path: Path = None):
    """Test ML model on all training data."""
    config = config_loader.load_config(config_path)
    ml_model_info = load_chirp_ml_model(config)
    
    if ml_model_info is None:
        print("ERROR: ML model not found. Run 'make train-ml' first.")
        return
    
    chirp_dir = Path("training/chirp")
    non_chirp_dir = Path("training/not_chirp")
    
    print("Testing ML model on training data...")
    print()
    
    results = {"chirp": {"correct": 0, "total": 0}, "non_chirp": {"correct": 0, "total": 0}}
    
    # Test chirps
    chirp_files = sorted(chirp_dir.glob("chirp_*.wav"))
    print(f"Testing {len(chirp_files)} chirp examples:")
    for clip_path in chirp_files:
        is_chirp, confidence, error = classify_clip_ml(clip_path, ml_model_info)
        results["chirp"]["total"] += 1
        if is_chirp and not error:
            results["chirp"]["correct"] += 1
            status = "✓"
        else:
            status = "✗"
        print(f"  {status} {clip_path.name}: {confidence:.3f} ({'chirp' if is_chirp else 'not_chirp'})")
    
    print()
    
    # Test non-chirps
    non_chirp_files = sorted(non_chirp_dir.glob("not_chirp_*.wav"))
    print(f"Testing {len(non_chirp_files)} non-chirp examples:")
    for clip_path in non_chirp_files:
        is_chirp, confidence, error = classify_clip_ml(clip_path, ml_model_info)
        results["non_chirp"]["total"] += 1
        if not is_chirp and not error:
            results["non_chirp"]["correct"] += 1
            status = "✓"
        else:
            status = "✗"
        print(f"  {status} {clip_path.name}: {confidence:.3f} ({'chirp' if is_chirp else 'not_chirp'})")
    
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    chirp_acc = results["chirp"]["correct"] / results["chirp"]["total"] if results["chirp"]["total"] > 0 else 0
    non_chirp_acc = results["non_chirp"]["correct"] / results["non_chirp"]["total"] if results["non_chirp"]["total"] > 0 else 0
    total_correct = results["chirp"]["correct"] + results["non_chirp"]["correct"]
    total = results["chirp"]["total"] + results["non_chirp"]["total"]
    overall_acc = total_correct / total if total > 0 else 0
    
    print(f"Chirp accuracy:     {chirp_acc:.1%} ({results['chirp']['correct']}/{results['chirp']['total']})")
    print(f"Non-chirp accuracy: {non_chirp_acc:.1%} ({results['non_chirp']['correct']}/{results['non_chirp']['total']})")
    print(f"Overall accuracy:   {overall_acc:.1%} ({total_correct}/{total})")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ML model on clips")
    parser.add_argument("clip", nargs="?", type=Path, help="Path to clip file to test (optional)")
    parser.add_argument("--config", type=Path, help="Path to config.json")
    parser.add_argument("--training", action="store_true", help="Test on all training data")
    
    args = parser.parse_args()
    
    if args.training:
        test_training_data(args.config)
    elif args.clip:
        test_clip(args.clip, args.config)
    else:
        print("Usage:")
        print("  Test single clip: python3 scripts/test_chirp_ml.py <clip_path>")
        print("  Test training data: python3 scripts/test_chirp_ml.py --training")
        sys.exit(1)

