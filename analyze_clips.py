#!/usr/bin/env python3
"""Analyze audio characteristics of clips to compare chirp vs non-chirp."""
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import wave

import config_loader
import monitor


def load_mono_wav(path: Path) -> tuple[np.ndarray, int]:
    """Load mono WAV file, converting to mono if needed."""
    with wave.open(str(path), "rb") as wf:
        nch = wf.getnchannels()
        sr = wf.getframerate()
        nframes = wf.getnframes()
        frames = wf.readframes(nframes)
    
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / monitor.INT16_FULL_SCALE
    
    if nch > 1:
        samples = samples.reshape(-1, nch).mean(axis=1)
    
    return samples, sr


def compute_spectral_features(samples: np.ndarray, sample_rate: int, fft_size: int = 2048) -> Dict:
    """Compute spectral features from audio samples."""
    if len(samples) < fft_size:
        pad = fft_size - len(samples)
        samples = np.pad(samples, (0, pad))
    
    hop = fft_size // 2
    window = np.hanning(fft_size)
    specs = []
    
    for start in range(0, len(samples) - fft_size, hop):
        chunk = samples[start:start + fft_size] * window
        spec = np.abs(np.fft.rfft(chunk))
        specs.append(spec)
    
    if not specs:
        return {}
    
    avg_spec = np.mean(specs, axis=0)
    avg_spec = avg_spec / (np.linalg.norm(avg_spec) + 1e-9)
    
    # Calculate features
    freq_resolution = sample_rate / fft_size
    frequencies = np.arange(len(avg_spec)) * freq_resolution
    
    # Spectral centroid
    magnitude = np.abs(avg_spec)
    total_magnitude = np.sum(magnitude)
    spectral_centroid = np.sum(frequencies * magnitude) / total_magnitude if total_magnitude > 0 else 0
    
    # Energy in frequency bands
    fan_noise_max_bin = int(500 / freq_resolution)
    chirp_min_bin = int(1000 / freq_resolution)
    
    low_freq_energy = np.sum(avg_spec[:fan_noise_max_bin])
    mid_freq_energy = np.sum(avg_spec[fan_noise_max_bin:chirp_min_bin])
    high_freq_energy = np.sum(avg_spec[chirp_min_bin:])
    total_energy = np.sum(avg_spec)
    
    return {
        "spectral_centroid": float(spectral_centroid),
        "low_freq_ratio": float(low_freq_energy / total_energy) if total_energy > 0 else 0,
        "mid_freq_ratio": float(mid_freq_energy / total_energy) if total_energy > 0 else 0,
        "high_freq_ratio": float(high_freq_energy / total_energy) if total_energy > 0 else 0,
    }


def compute_temporal_features(samples: np.ndarray, sample_rate: int, chunk_duration: float = 0.5) -> Dict:
    """Compute temporal features from audio samples."""
    chunk_samples = int(sample_rate * chunk_duration)
    chunk_rms_values = []
    
    for i in range(0, len(samples), chunk_samples):
        chunk = samples[i:i + chunk_samples]
        if len(chunk) > 0:
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            chunk_rms_values.append(rms)
    
    if len(chunk_rms_values) < 2:
        return {
            "duration_sec": len(samples) / sample_rate,
            "energy_concentration": 0.5,
            "attack_decay_ratio": None,
        }
    
    # Energy concentration
    mid_point = len(chunk_rms_values) // 2
    first_half_energy = sum(r**2 for r in chunk_rms_values[:mid_point])
    second_half_energy = sum(r**2 for r in chunk_rms_values[mid_point:])
    total_energy = first_half_energy + second_half_energy
    energy_concentration = first_half_energy / total_energy if total_energy > 0 else 0.5
    
    # Attack/decay
    peak_idx = chunk_rms_values.index(max(chunk_rms_values))
    attack_decay_ratio = None
    if peak_idx > 0 and peak_idx < len(chunk_rms_values) - 1:
        peak_value = chunk_rms_values[peak_idx]
        attack_threshold = peak_value * 0.9
        decay_threshold = peak_value * 0.1
        
        attack_time = peak_idx
        for i in range(peak_idx):
            if chunk_rms_values[i] >= attack_threshold:
                attack_time = peak_idx - i
                break
        
        decay_time = len(chunk_rms_values) - peak_idx - 1
        for i in range(peak_idx + 1, len(chunk_rms_values)):
            if chunk_rms_values[i] <= decay_threshold:
                decay_time = i - peak_idx
                break
        
        if decay_time > 0:
            attack_decay_ratio = attack_time / decay_time
    
    return {
        "duration_sec": len(samples) / sample_rate,
        "energy_concentration": float(energy_concentration),
        "attack_decay_ratio": float(attack_decay_ratio) if attack_decay_ratio else None,
    }


def analyze_clips(
    events_file: Path = Path("events.csv"),
    config_path: Optional[Path] = None,
    output_file: Path = Path("clip_analysis.csv")
):
    """Analyze all clips and export features to CSV."""
    config = config_loader.load_config(config_path)
    
    df = pd.read_csv(events_file) if events_file.exists() else pd.DataFrame()
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

