#!/usr/bin/env python3
"""
Compare ML model vs fingerprint classification on reviewed clips.

This script runs both classifiers on your reviewed events and shows:
- Accuracy comparison
- Which clips are classified differently
- Detailed metrics for each method
"""
import json
import wave
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config_loader
import monitor
from core.classifier import classify_event_is_chirp, load_chirp_fingerprint

# Import ML classification functions from core
from core.classifier import load_chirp_ml_model, classify_clip_ml

# Import event loading from core
from core.reporting import load_events


# load_events is now in core.reporting - imported above


def classify_clip_fingerprint(clip_path: Path, config: Dict, fingerprint_info: Optional[Dict]) -> Tuple[bool, Optional[float], Optional[float], Optional[str]]:
    """Classify a clip using fingerprint method."""
    if not clip_path.exists():
        return False, None, None, "file_not_found"
    
    # Load audio from clip
    with wave.open(str(clip_path), "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        n_frames = wf.getnframes()
        audio_data = wf.readframes(n_frames)
    
    # Convert to numpy array
    samples = np.frombuffer(audio_data, dtype="<i2").astype(np.float32) / monitor.INT16_FULL_SCALE
    
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)
    
    # Convert to chunks (matching monitor.py chunk size)
    audio_cfg = config["audio"]
    chunk_samples = int(audio_cfg["sample_rate"] * audio_cfg["chunk_duration"])
    chunk_bytes = chunk_samples * monitor.BYTES_PER_SAMPLE * audio_cfg["channels"]
    
    # Split into chunks
    chunks = []
    for i in range(0, len(samples), chunk_samples):
        chunk_samples_actual = min(chunk_samples, len(samples) - i)
        chunk_data = samples[i:i + chunk_samples_actual]
        # Convert back to bytes
        chunk_int16 = (chunk_data * monitor.INT16_FULL_SCALE).astype("<i2")
        chunks.append(chunk_int16.tobytes())
    
    if not chunks:
        return False, None, None, "no_audio_data"
    
    # Calculate duration
    duration_sec = len(samples) / audio_cfg["sample_rate"]
    
    # Classify
    return classify_event_is_chirp(chunks, fingerprint_info, duration_sec, config)


def compare_classifiers(
    events_file: Path = Path("data/events.csv"),
    config_path: Optional[Path] = None
):
    """Compare ML vs fingerprint classification on reviewed events."""
    config = config_loader.load_config(config_path)
    
    # Load both classifiers
    fingerprint_info = load_chirp_fingerprint(config)
    ml_model_info = load_chirp_ml_model(config)
    
    if fingerprint_info is None:
        print("ERROR: Fingerprint not found. Run 'make train' first.")
        return
    
    if ml_model_info is None:
        print("ERROR: ML model not found. Run 'make train-ml' first.")
        return
    
    # Load events
    df = load_events(events_file)
    if df.empty:
        print("No events found in CSV file")
        return
    
    # Filter to reviewed events
    if "reviewed" not in df.columns:
        print("Warning: No 'reviewed' column found.")
        print("Please review events and mark them in the 'reviewed' column.")
        print("Use 'chirp', 'not_chirp', 'false_positive', etc.")
        return
    
    reviewed_df = df[df["reviewed"].notna() & (df["reviewed"] != "")]
    if reviewed_df.empty:
        print("No reviewed events found. Please review some events first.")
        return
    
    # Determine ground truth
    reviewed_df = reviewed_df.copy()
    reviewed_df["ground_truth"] = reviewed_df["reviewed"].str.lower().str.contains("chirp|true|yes|positive")
    
    print(f"Comparing classifiers on {len(reviewed_df)} reviewed events...")
    print()
    
    # Classify each event with both methods
    results = []
    clips_dir = Path(config["event_clips"]["clips_dir"])
    
    for idx, row in reviewed_df.iterrows():
        clip_file = row.get("clip_file", "")
        if not clip_file:
            continue
        
        clip_path = clips_dir / Path(clip_file).name
        if not clip_path.exists():
            clip_path = Path(clip_file)
        
        if not clip_path.exists():
            continue
        
        # Classify with fingerprint
        fp_is_chirp, fp_sim, fp_conf, fp_reason = classify_clip_fingerprint(
            clip_path, config, fingerprint_info
        )
        
        # Classify with ML
        ml_is_chirp, ml_conf, ml_error = classify_clip_ml(clip_path, ml_model_info)
        
        actual = row["ground_truth"]
        
        results.append({
            "timestamp": row.get("start_timestamp", ""),
            "clip_file": clip_file,
            "ground_truth": actual,
            "fingerprint_pred": fp_is_chirp,
            "fingerprint_correct": fp_is_chirp == actual,
            "fingerprint_similarity": fp_sim,
            "fingerprint_confidence": fp_conf,
            "ml_pred": ml_is_chirp,
            "ml_correct": ml_is_chirp == actual,
            "ml_confidence": ml_conf,
            "agree": fp_is_chirp == ml_is_chirp,
            "reviewed_note": row.get("reviewed", "")
        })
    
    if not results:
        print("No valid clips found for comparison")
        return
    
    results_df = pd.DataFrame(results)
    
    # Calculate metrics for each method
    def calc_metrics(pred_col, actual_col):
        total = len(results_df)
        correct = (results_df[pred_col] == results_df[actual_col]).sum()
        accuracy = correct / total if total > 0 else 0
        
        tp = ((results_df[actual_col] == True) & (results_df[pred_col] == True)).sum()
        fp = ((results_df[actual_col] == False) & (results_df[pred_col] == True)).sum()
        fn = ((results_df[actual_col] == True) & (results_df[pred_col] == False)).sum()
        tn = ((results_df[actual_col] == False) & (results_df[pred_col] == False)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
    
    fp_metrics = calc_metrics("fingerprint_pred", "ground_truth")
    ml_metrics = calc_metrics("ml_pred", "ground_truth")
    
    # Print comparison
    print("=" * 80)
    print("CLASSIFIER COMPARISON RESULTS")
    print("=" * 80)
    print()
    
    print(f"{'Metric':<20} {'Fingerprint':<20} {'ML Model':<20} {'Winner':<20}")
    print("-" * 80)
    print(f"{'Accuracy':<20} {fp_metrics['accuracy']:>18.1%} {ml_metrics['accuracy']:>18.1%} ", end="")
    if ml_metrics['accuracy'] > fp_metrics['accuracy']:
        print("ML Model")
    elif fp_metrics['accuracy'] > ml_metrics['accuracy']:
        print("Fingerprint")
    else:
        print("Tie")
    
    print(f"{'Precision':<20} {fp_metrics['precision']:>18.1%} {ml_metrics['precision']:>18.1%} ", end="")
    if ml_metrics['precision'] > fp_metrics['precision']:
        print("ML Model")
    elif fp_metrics['precision'] > ml_metrics['precision']:
        print("Fingerprint")
    else:
        print("Tie")
    
    print(f"{'Recall':<20} {fp_metrics['recall']:>18.1%} {ml_metrics['recall']:>18.1%} ", end="")
    if ml_metrics['recall'] > fp_metrics['recall']:
        print("ML Model")
    elif fp_metrics['recall'] > ml_metrics['recall']:
        print("Fingerprint")
    else:
        print("Tie")
    
    print(f"{'F1 Score':<20} {fp_metrics['f1']:>18.1%} {ml_metrics['f1']:>18.1%} ", end="")
    if ml_metrics['f1'] > fp_metrics['f1']:
        print("ML Model")
    elif fp_metrics['f1'] > ml_metrics['f1']:
        print("Fingerprint")
    else:
        print("Tie")
    
    print()
    print("Confusion Matrices:")
    print()
    print("Fingerprint:")
    print(f"  TP: {fp_metrics['tp']:3d}  FP: {fp_metrics['fp']:3d}")
    print(f"  FN: {fp_metrics['fn']:3d}  TN: {fp_metrics['tn']:3d}")
    print()
    print("ML Model:")
    print(f"  TP: {ml_metrics['tp']:3d}  FP: {ml_metrics['fp']:3d}")
    print(f"  FN: {ml_metrics['fn']:3d}  TN: {ml_metrics['tn']:3d}")
    print()
    
    # Show disagreements
    disagreements = results_df[results_df["agree"] == False]
    if len(disagreements) > 0:
        print(f"Disagreements ({len(disagreements)} clips where methods differ):")
        print()
        for _, row in disagreements.iterrows():
            truth = "chirp" if row["ground_truth"] else "not_chirp"
            fp_pred = "chirp" if row["fingerprint_pred"] else "not_chirp"
            ml_pred = "chirp" if row["ml_pred"] else "not_chirp"
            
            fp_correct = "✓" if row["fingerprint_correct"] else "✗"
            ml_correct = "✓" if row["ml_correct"] else "✗"
            
            print(f"  {row['timestamp']}:")
            print(f"    Truth: {truth}")
            print(f"    Fingerprint: {fp_pred} {fp_correct} (sim={row['fingerprint_similarity']:.3f})")
            print(f"    ML Model:     {ml_pred} {ml_correct} (conf={row['ml_confidence']:.3f})")
            print(f"    Clip: {row['clip_file']}")
            print()
    else:
        print("✓ Both methods agree on all clips!")
        print()
    
    # Save detailed results
    output_file = Path("classifier_comparison.csv")
    results_df.to_csv(output_file, index=False)
    print(f"Detailed results saved to {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare ML vs fingerprint classification")
    parser.add_argument("--config", type=Path, help="Path to config.json")
    parser.add_argument("--events", type=Path, default=Path("data/events.csv"), help="Path to events.csv")
    
    args = parser.parse_args()
    compare_classifiers(args.events, args.config)

