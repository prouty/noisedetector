#!/usr/bin/env python3
"""Validate chirp classification accuracy against reviewed events."""
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config_loader
import monitor


def load_events(events_file: Path) -> pd.DataFrame:
    """Load events from CSV file."""
    if not events_file.exists():
        print(f"Error: {events_file} not found")
        return pd.DataFrame()
    
    df = pd.read_csv(events_file)
    return df


def classify_clip(clip_path: Path, config: Dict, fingerprint_info: Optional[Dict]) -> Tuple[bool, Optional[float], Optional[float], Optional[str]]:
    """Classify a single clip file."""
    import wave
    
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
    return monitor.classify_event_is_chirp(chunks, fingerprint_info, duration_sec, config)


def validate_classification(
    events_file: Path = Path("events.csv"),
    config_path: Optional[Path] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None
):
    """Validate classification accuracy against reviewed events."""
    config = config_loader.load_config(config_path)
    fingerprint_info = monitor.load_chirp_fingerprint(config)
    
    df = load_events(events_file)
    if df.empty:
        print("No events found in CSV file")
        return
    
    # Filter by date if specified
    if date_start or date_end:
        if "start_timestamp" in df.columns:
            df["start_timestamp"] = pd.to_datetime(df["start_timestamp"])
            if date_start:
                df = df[df["start_timestamp"] >= pd.to_datetime(date_start)]
            if date_end:
                df = df[df["start_timestamp"] <= pd.to_datetime(date_end)]
    
    # Filter to reviewed events
    if "reviewed" not in df.columns:
        print("Warning: No 'reviewed' column found. Cannot validate accuracy.")
        print("Please review events and mark them in the 'reviewed' column.")
        return
    
    reviewed_df = df[df["reviewed"].notna() & (df["reviewed"] != "")]
    if reviewed_df.empty:
        print("No reviewed events found. Please review some events first.")
        return
    
    print(f"Validating {len(reviewed_df)} reviewed events...")
    print()
    
    # Determine ground truth from reviewed column
    # Assume format like "chirp", "not_chirp", "false_positive", etc.
    reviewed_df = reviewed_df.copy()
    reviewed_df["ground_truth"] = reviewed_df["reviewed"].str.lower().str.contains("chirp|true|yes|positive")
    
    # Re-classify each event
    results = []
    clips_dir = Path(config["event_clips"]["clips_dir"])
    
    for idx, row in reviewed_df.iterrows():
        clip_file = row.get("clip_file", "")
        if not clip_file:
            continue
        
        clip_path = clips_dir / Path(clip_file).name
        if not clip_path.exists():
            # Try absolute path
            clip_path = Path(clip_file)
        
        is_chirp, similarity, confidence, rejection_reason = classify_clip(clip_path, config, fingerprint_info)
        
        predicted = is_chirp
        actual = row["ground_truth"]
        
        results.append({
            "timestamp": row.get("start_timestamp", ""),
            "clip_file": clip_file,
            "actual": actual,
            "predicted": predicted,
            "correct": predicted == actual,
            "similarity": similarity,
            "confidence": confidence,
            "rejection_reason": rejection_reason,
            "reviewed_note": row.get("reviewed", "")
        })
    
    # Calculate metrics
    results_df = pd.DataFrame(results)
    if results_df.empty:
        print("No valid clips found for validation")
        return
    
    total = len(results_df)
    correct = results_df["correct"].sum()
    accuracy = correct / total if total > 0 else 0
    
    true_positives = ((results_df["actual"] == True) & (results_df["predicted"] == True)).sum()
    false_positives = ((results_df["actual"] == False) & (results_df["predicted"] == True)).sum()
    false_negatives = ((results_df["actual"] == True) & (results_df["predicted"] == False)).sum()
    true_negatives = ((results_df["actual"] == False) & (results_df["predicted"] == False)).sum()
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Print results
    print("=" * 60)
    print("CLASSIFICATION VALIDATION RESULTS")
    print("=" * 60)
    print(f"Total reviewed events: {total}")
    print(f"Accuracy: {accuracy:.1%} ({correct}/{total})")
    print()
    print("Confusion Matrix:")
    print(f"  True Positives:  {true_positives:3d}")
    print(f"  False Positives: {false_positives:3d}")
    print(f"  False Negatives: {false_negatives:3d}")
    print(f"  True Negatives:  {true_negatives:3d}")
    print()
    print("Metrics:")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall:    {recall:.1%}")
    print(f"  F1 Score:  {f1:.1%}")
    print()
    
    # Show errors
    if false_positives > 0:
        print("False Positives (predicted chirp, but not):")
        fp_df = results_df[(results_df["actual"] == False) & (results_df["predicted"] == True)]
        for _, row in fp_df.iterrows():
            print(f"  {row['timestamp']}: similarity={row['similarity']:.3f}, confidence={row['confidence']:.3f if row['confidence'] else 'N/A'}, reason={row['rejection_reason'] or 'N/A'}")
        print()
    
    if false_negatives > 0:
        print("False Negatives (missed chirps):")
        fn_df = results_df[(results_df["actual"] == True) & (results_df["predicted"] == False)]
        for _, row in fn_df.iterrows():
            print(f"  {row['timestamp']}: similarity={row['similarity']:.3f}, reason={row['rejection_reason'] or 'N/A'}")
        print()
    
    # Tuning recommendations
    print("Tuning Recommendations:")
    if false_positives > false_negatives:
        print("  - Too many false positives: Consider increasing similarity_threshold or")
        print("    adjusting frequency/temporal filtering thresholds")
    elif false_negatives > false_positives:
        print("  - Too many false negatives: Consider decreasing similarity_threshold or")
        print("    relaxing frequency/temporal filtering")
    else:
        print("  - Balance looks good, but review individual errors for patterns")
    
    # Save detailed results
    output_file = Path("validation_results.csv")
    results_df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate chirp classification accuracy")
    parser.add_argument("--config", type=Path, help="Path to config.json")
    parser.add_argument("--events", type=Path, default=Path("events.csv"), help="Path to events.csv")
    parser.add_argument("--date-start", help="Start date filter (YYYY-MM-DD)")
    parser.add_argument("--date-end", help="End date filter (YYYY-MM-DD)")
    
    args = parser.parse_args()
    validate_classification(args.events, args.config, args.date_start, args.date_end)

