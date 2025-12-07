#!/usr/bin/env python3
"""Analyze audio characteristics of clips to compare chirp vs non-chirp."""
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import wave

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config_loader
from core.features import (
    load_mono_wav,
    compute_spectral_features,
    compute_temporal_features,
)
from core.reporting import load_events


# Feature extraction functions are now in core.features


def analyze_clips(
    events_file: Path = Path("events.csv"),
    config_path: Optional[Path] = None,
    output_file: Path = Path("clip_analysis.csv")
):
    """Analyze all clips and export features to CSV."""
    config = config_loader.load_config(config_path)
    
    df = load_events(events_file)
    if df.empty:
        print(f"No events found in {events_file}")
        return
    
    clips_dir = Path(config["event_clips"]["clips_dir"])
    audio_cfg = config["audio"]
    
    print(f"Analyzing clips from {events_file}...")
    print()
    
    results = []
    for idx, row in df.iterrows():
        clip_file = row.get("clip_file", "")
        if not clip_file:
            continue
        
        clip_path = clips_dir / Path(clip_file).name
        if not clip_path.exists():
            clip_path = Path(clip_file)
        
        if not clip_path.exists():
            continue
        
        try:
            samples, sr = load_mono_wav(clip_path)
            
            spectral = compute_spectral_features(samples, sr)
            temporal = compute_temporal_features(samples, sr, audio_cfg["chunk_duration"])
            
            result = {
                "timestamp": row.get("start_timestamp", ""),
                "clip_file": clip_file,
                "is_chirp": row.get("is_chirp", "FALSE"),
                "chirp_similarity": row.get("chirp_similarity", ""),
                "confidence": row.get("confidence", ""),
                "reviewed": row.get("reviewed", ""),
                "duration_sec": temporal["duration_sec"],
                "spectral_centroid": spectral["spectral_centroid"],
                "low_freq_ratio": spectral["low_freq_ratio"],
                "mid_freq_ratio": spectral["mid_freq_ratio"],
                "high_freq_ratio": spectral["high_freq_ratio"],
                "energy_concentration": temporal["energy_concentration"],
                "attack_decay_ratio": temporal["attack_decay_ratio"],
            }
            results.append(result)
        except Exception as e:
            print(f"Error analyzing {clip_path}: {e}")
            continue
    
    if not results:
        print("No clips could be analyzed")
        return
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    
    print(f"Analyzed {len(results_df)} clips")
    print(f"Results saved to {output_file}")
    print()
    
    # Summary statistics
    if "is_chirp" in results_df.columns:
        chirp_df = results_df[results_df["is_chirp"].astype(str).str.upper() == "TRUE"]
        non_chirp_df = results_df[results_df["is_chirp"].astype(str).str.upper() == "FALSE"]
        
        if len(chirp_df) > 0 and len(non_chirp_df) > 0:
            print("Summary Statistics:")
            print()
            print("Chirps:")
            print(f"  Count: {len(chirp_df)}")
            print(f"  Avg Duration: {chirp_df['duration_sec'].mean():.2f}s")
            print(f"  Avg Spectral Centroid: {chirp_df['spectral_centroid'].mean():.0f} Hz")
            print(f"  Avg High Freq Ratio: {chirp_df['high_freq_ratio'].mean():.2f}")
            print(f"  Avg Energy Concentration: {chirp_df['energy_concentration'].mean():.2f}")
            print()
            print("Non-Chirps:")
            print(f"  Count: {len(non_chirp_df)}")
            print(f"  Avg Duration: {non_chirp_df['duration_sec'].mean():.2f}s")
            print(f"  Avg Spectral Centroid: {non_chirp_df['spectral_centroid'].mean():.0f} Hz")
            print(f"  Avg High Freq Ratio: {non_chirp_df['high_freq_ratio'].mean():.2f}")
            print(f"  Avg Energy Concentration: {non_chirp_df['energy_concentration'].mean():.2f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze audio characteristics of clips")
    parser.add_argument("--config", type=Path, help="Path to config.json")
    parser.add_argument("--events", type=Path, default=Path("events.csv"), help="Path to events.csv")
    parser.add_argument("--output", type=Path, default=Path("clip_analysis.csv"), help="Output CSV file")
    
    args = parser.parse_args()
    analyze_clips(args.events, args.config, args.output)

