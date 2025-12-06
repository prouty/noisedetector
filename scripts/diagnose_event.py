#!/usr/bin/env python3
"""Diagnose why a specific event was or wasn't classified as a chirp."""
import argparse
import json
from pathlib import Path
from typing import Optional
import numpy as np
import wave

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config_loader
import monitor
from core.classifier import (
    classify_event_is_chirp,
    load_chirp_fingerprint,
    find_best_chirp_segment,
    compute_event_spectrum_from_chunks,
    compute_spectral_centroid
)


def diagnose_clip(clip_path: Path, config_path: Optional[Path] = None):
    """Analyze a single clip and show detailed classification results."""
    config = config_loader.load_config(config_path)
    fingerprint_info = load_chirp_fingerprint(config)
    
    if not clip_path.exists():
        print(f"Error: {clip_path} not found")
        return
    
    # Load audio
    with wave.open(str(clip_path), "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        n_frames = wf.getnframes()
        audio_data = wf.readframes(n_frames)
    
    samples = np.frombuffer(audio_data, dtype="<i2").astype(np.float32) / monitor.INT16_FULL_SCALE
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)
    
    duration_sec = len(samples) / sample_rate
    
    # Convert to chunks
    audio_cfg = config["audio"]
    chunk_samples = int(audio_cfg["sample_rate"] * audio_cfg["chunk_duration"])
    chunks = []
    for i in range(0, len(samples), chunk_samples):
        chunk_samples_actual = min(chunk_samples, len(samples) - i)
        chunk_data = samples[i:i + chunk_samples_actual]
        chunk_int16 = (chunk_data * monitor.INT16_FULL_SCALE).astype("<i2")
        chunks.append(chunk_int16.tobytes())
    
    if not chunks:
        print("Error: No audio data in clip")
        return
    
    # Get classification result (this uses sliding window internally)
    is_chirp, similarity, confidence, rejection_reason = classify_event_is_chirp(
        chunks, fingerprint_info, duration_sec, config
    )
    
    # Find the best segment that was used for classification
    best_chunks, best_similarity, _ = find_best_chirp_segment(
        chunks, fingerprint_info, config
    )
    
    # Use best_chunks for analysis if available, otherwise use all chunks
    analysis_chunks = best_chunks if best_chunks is not None else chunks
    best_segment_duration = len(analysis_chunks) * audio_cfg["chunk_duration"] if best_chunks else duration_sec
    
    # Calculate detailed features
    chirp_cfg = config["chirp_classification"]
    freq_cfg = chirp_cfg["frequency_filtering"]
    temp_cfg = chirp_cfg["temporal_filtering"]
    
    # Compute spectrum from the best segment (what was actually used for classification)
    if fingerprint_info:
        event_spec = compute_event_spectrum_from_chunks(
            analysis_chunks, fingerprint_info["sample_rate"], fingerprint_info["fft_size"]
        )
        
        if event_spec is not None:
            freq_resolution = fingerprint_info["sample_rate"] / fingerprint_info["fft_size"]
            fan_noise_max_bin = int(freq_cfg["fan_noise_max_freq_hz"] / freq_resolution)
            chirp_min_bin = int(freq_cfg["chirp_min_freq_hz"] / freq_resolution)
            
            total_energy = np.sum(event_spec)
            low_freq_energy = np.sum(event_spec[:fan_noise_max_bin])
            high_freq_energy = np.sum(event_spec[chirp_min_bin:])
            low_freq_ratio = low_freq_energy / total_energy if total_energy > 0 else 0
            high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
            
            spectral_centroid = compute_spectral_centroid(
                event_spec, fingerprint_info["sample_rate"], fingerprint_info["fft_size"]
            )
        else:
            event_spec = None
            low_freq_ratio = high_freq_ratio = spectral_centroid = 0
    else:
        event_spec = None
        low_freq_ratio = high_freq_ratio = spectral_centroid = 0
    
    # Compute temporal features from the best segment
    chunk_rms_values = []
    for chunk in analysis_chunks:
        chunk_samples = np.frombuffer(chunk, dtype="<i2").astype(np.float32) / monitor.INT16_FULL_SCALE
        if len(chunk_samples) > 0:
            rms = float(np.sqrt(np.mean(chunk_samples ** 2)))
            chunk_rms_values.append(rms)
    
    energy_concentration = 0.5
    if len(chunk_rms_values) > 1:
        mid_point = len(chunk_rms_values) // 2
        first_half_energy = sum(r**2 for r in chunk_rms_values[:mid_point])
        second_half_energy = sum(r**2 for r in chunk_rms_values[mid_point:])
        total_chunk_energy = first_half_energy + second_half_energy
        if total_chunk_energy > 0:
            energy_concentration = first_half_energy / total_chunk_energy
    
    attack_decay_ratio = monitor.compute_attack_decay_ratio(chunk_rms_values)
    
    # Print diagnosis
    print("=" * 60)
    print(f"EVENT DIAGNOSIS: {clip_path.name}")
    print("=" * 60)
    print()
    print(f"Full Event Duration: {duration_sec:.2f} seconds")
    if best_chunks is not None and len(best_chunks) < len(chunks):
        print(f"Best Segment Duration: {best_segment_duration:.2f} seconds ({len(best_chunks)}/{len(chunks)} chunks)")
        print(f"  (Using sliding window - focusing on best matching segment)")
    print(f"Classification: {'CHIRP' if is_chirp else 'NOT CHIRP'}")
    print()
    
    if fingerprint_info:
        print("Spectral Analysis:")
        if similarity is not None:
            print(f"  Similarity to fingerprint: {similarity:.3f} (threshold: {chirp_cfg['similarity_threshold']:.3f})")
            if similarity < chirp_cfg['similarity_threshold']:
                print(f"  ❌ FAILED: Similarity too low")
            else:
                print(f"  ✓ PASSED: Similarity above threshold")
        else:
            print(f"  Similarity: N/A (classification failed before similarity check)")
        print()
        
        if event_spec is not None:
            print("Frequency Analysis:")
            print(f"  Spectral Centroid: {spectral_centroid:.0f} Hz")
            print(f"  Low-freq ratio (<{freq_cfg['fan_noise_max_freq_hz']} Hz): {low_freq_ratio:.3f} (threshold: {freq_cfg['low_freq_energy_threshold']:.3f})")
            if low_freq_ratio > freq_cfg['low_freq_energy_threshold']:
                print(f"  ❌ FAILED: Too much low-frequency energy (fan noise)")
            else:
                print(f"  ✓ PASSED: Low-frequency energy acceptable")
            
            min_high_freq = freq_cfg.get("high_freq_energy_min_ratio", 0.1)
            print(f"  High-freq ratio (>{freq_cfg['chirp_min_freq_hz']} Hz): {high_freq_ratio:.3f} (min: {min_high_freq:.3f})")
            if high_freq_ratio < min_high_freq:
                print(f"  ❌ FAILED: Insufficient high-frequency energy")
            else:
                print(f"  ✓ PASSED: Sufficient high-frequency energy")
            print()
    
    print("Temporal Analysis:")
    print(f"  Best Segment Duration: {best_segment_duration:.2f}s (max: {temp_cfg['max_duration_sec']:.2f}s)")
    if best_segment_duration > temp_cfg['max_duration_sec']:
        print(f"  ❌ FAILED: Duration too long (sustained sound)")
    else:
        print(f"  ✓ PASSED: Duration acceptable")
    
    print(f"  Energy Concentration: {energy_concentration:.3f} (threshold: {temp_cfg['energy_concentration_threshold']:.3f})")
    if energy_concentration < temp_cfg['energy_concentration_threshold']:
        print(f"  ❌ FAILED: Energy too spread out (sustained sound)")
    else:
        print(f"  ✓ PASSED: Energy concentrated in first half")
    
    if attack_decay_ratio is not None:
        print(f"  Attack/Decay Ratio: {attack_decay_ratio:.2f} (higher = sharper attack)")
    else:
        print(f"  Attack/Decay Ratio: N/A (insufficient data)")
    print()
    
    if confidence is not None:
        print(f"Confidence Score: {confidence:.3f}")
        print()
    
    if rejection_reason:
        print(f"Rejection Reason: {rejection_reason}")
        print()
    
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose why an event was/wasn't classified as chirp")
    parser.add_argument("clip", type=Path, help="Path to clip file")
    parser.add_argument("--config", type=Path, help="Path to config.json")
    
    args = parser.parse_args()
    diagnose_clip(args.clip, args.config)

