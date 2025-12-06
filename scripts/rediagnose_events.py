#!/usr/bin/env python3
"""Re-classify files in training/chirp/ and training/not_chirp/ directories.

This script scans the training directories, re-classifies each file with the current
algorithm, and outputs a list of files that may be in the wrong directory for review.
"""
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
import config_loader
import monitor
import validate_classification
from core.classifier import load_chirp_fingerprint

# Try to import ML classifier
try:
    import scripts.classify_chirp_ml as ml_classifier
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


def classify_file(clip_path: Path, config: Dict, fingerprint_info: Optional[Dict], 
                  ml_model_info: Optional[Tuple]) -> Tuple[bool, Optional[float], Optional[float], Optional[str]]:
    """Classify a single file using the configured classifier."""
    use_ml = config.get("chirp_classification", {}).get("use_ml_classifier", False)
    
    if use_ml and ml_model_info is not None and ML_AVAILABLE:
        # Use ML classifier
        is_chirp, confidence, error = ml_classifier.classify_clip_ml(clip_path, ml_model_info)
        if error:
            # Fallback to fingerprint
            return validate_classification.classify_clip(clip_path, config, fingerprint_info)
        return is_chirp, None, confidence, None
    else:
        # Use fingerprint classifier
        return validate_classification.classify_clip(clip_path, config, fingerprint_info)


def rediagnose_training_files(
    training_dir: Path = Path("training"),
    config_path: Optional[Path] = None
) -> Dict[str, List[Tuple[Path, bool, Optional[float], Optional[float]]]]:
    """
    Re-classify files in training/chirp/ and training/not_chirp/ directories.
    
    Returns:
        Dictionary with keys:
        - 'chirp_misplaced': Files in training/chirp/ classified as NOT chirp
        - 'not_chirp_misplaced': Files in training/not_chirp/ classified as chirp
        - 'chirp_correct': Files in training/chirp/ correctly classified (optional)
        - 'not_chirp_correct': Files in training/not_chirp/ correctly classified (optional)
    """
    config = config_loader.load_config(config_path)
    fingerprint_info = load_chirp_fingerprint(config)
    
    # Load ML model if configured
    ml_model_info = None
    use_ml = config.get("chirp_classification", {}).get("use_ml_classifier", False)
    if use_ml and ML_AVAILABLE:
        ml_model_info = ml_classifier.load_ml_model(config)
        if ml_model_info:
            print(f"Using ML classifier for re-classification")
        else:
            print(f"ML classifier configured but model not found, using fingerprint")
    elif use_ml and not ML_AVAILABLE:
        print(f"ML classifier configured but module not available, using fingerprint")
    
    chirp_dir = training_dir / "chirp"
    not_chirp_dir = training_dir / "not_chirp"
    
    results = {
        'chirp_misplaced': [],  # In chirp/ but classified as NOT chirp
        'not_chirp_misplaced': [],  # In not_chirp/ but classified as chirp
        'chirp_correct': [],  # In chirp/ and classified as chirp
        'not_chirp_correct': []  # In not_chirp/ and classified as NOT chirp
    }
    
    # Process files in training/chirp/
    if chirp_dir.exists():
        chirp_files = sorted(chirp_dir.glob("*.wav"))
        print(f"Scanning {len(chirp_files)} files in {chirp_dir}...")
        
        for clip_path in chirp_files:
            is_chirp, similarity, confidence, rejection_reason = classify_file(
                clip_path, config, fingerprint_info, ml_model_info
            )
            
            if is_chirp:
                results['chirp_correct'].append((clip_path, is_chirp, similarity, confidence))
            else:
                results['chirp_misplaced'].append((clip_path, is_chirp, similarity, confidence))
    
    # Process files in training/not_chirp/
    if not_chirp_dir.exists():
        not_chirp_files = sorted(not_chirp_dir.glob("*.wav"))
        print(f"Scanning {len(not_chirp_files)} files in {not_chirp_dir}...")
        
        for clip_path in not_chirp_files:
            is_chirp, similarity, confidence, rejection_reason = classify_file(
                clip_path, config, fingerprint_info, ml_model_info
            )
            
            if is_chirp:
                results['not_chirp_misplaced'].append((clip_path, is_chirp, similarity, confidence))
            else:
                results['not_chirp_correct'].append((clip_path, is_chirp, similarity, confidence))
    
    return results


def print_review_list(results: Dict[str, List[Tuple[Path, bool, Optional[float], Optional[float]]]]):
    """Print a review list of files that may be in the wrong directory."""
    print()
    print("=" * 80)
    print("CLASSIFICATION REVIEW LIST")
    print("=" * 80)
    print()
    
    # Files in training/chirp/ that are classified as NOT chirp
    if results['chirp_misplaced']:
        print(f"⚠️  FILES IN training/chirp/ CLASSIFIED AS NOT CHIRP ({len(results['chirp_misplaced'])} files):")
        print("-" * 80)
        for clip_path, is_chirp, similarity, confidence in results['chirp_misplaced']:
            sim_str = f"similarity={similarity:.3f}" if similarity is not None else "similarity=N/A"
            conf_str = f"confidence={confidence:.3f}" if confidence is not None else "confidence=N/A"
            print(f"  {clip_path.name}")
            print(f"    Path: {clip_path}")
            print(f"    Classification: NOT CHIRP ({sim_str}, {conf_str})")
            print()
    else:
        print("✓ All files in training/chirp/ are correctly classified as chirps")
        print()
    
    # Files in training/not_chirp/ that are classified as chirp
    if results['not_chirp_misplaced']:
        print(f"⚠️  FILES IN training/not_chirp/ CLASSIFIED AS CHIRP ({len(results['not_chirp_misplaced'])} files):")
        print("-" * 80)
        for clip_path, is_chirp, similarity, confidence in results['not_chirp_misplaced']:
            sim_str = f"similarity={similarity:.3f}" if similarity is not None else "similarity=N/A"
            conf_str = f"confidence={confidence:.3f}" if confidence is not None else "confidence=N/A"
            print(f"  {clip_path.name}")
            print(f"    Path: {clip_path}")
            print(f"    Classification: CHIRP ({sim_str}, {conf_str})")
            print()
    else:
        print("✓ All files in training/not_chirp/ are correctly classified as not chirps")
        print()
    
    # Summary
    total_misplaced = len(results['chirp_misplaced']) + len(results['not_chirp_misplaced'])
    total_correct = len(results['chirp_correct']) + len(results['not_chirp_correct'])
    total = total_misplaced + total_correct
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files scanned: {total}")
    print(f"  Correctly placed: {total_correct}")
    print(f"  Potentially misplaced: {total_misplaced}")
    print()
    print(f"  training/chirp/: {len(results['chirp_correct'])} correct, {len(results['chirp_misplaced'])} misplaced")
    print(f"  training/not_chirp/: {len(results['not_chirp_correct'])} correct, {len(results['not_chirp_misplaced'])} misplaced")
    print()
    
    if total_misplaced > 0:
        print("💡 Review the files listed above. You may want to:")
        print("   - Move misplaced files to the correct directory")
        print("   - Re-train the classifier after moving files")
        print("   - Or verify the classification is correct (false positive/negative)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Re-classify files in training/chirp/ and training/not_chirp/ directories",
        epilog="""
This script scans the training directories and re-classifies each file with the current
algorithm. It outputs a list of files that may be in the wrong directory for review.

Example:
  python3 scripts/rediagnose_events.py
  python3 scripts/rediagnose_events.py --training-dir custom_training
        """
    )
    parser.add_argument("--config", type=Path, help="Path to config.json")
    parser.add_argument("--training-dir", type=Path, default=Path("training"),
                       help="Path to training directory (default: training)")
    
    args = parser.parse_args()
    
    if not args.training_dir.exists():
        print(f"Error: Training directory not found: {args.training_dir}")
        sys.exit(1)
    
    results = rediagnose_training_files(args.training_dir, args.config)
    print_review_list(results)

