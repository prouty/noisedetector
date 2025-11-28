#!/usr/bin/env python3
"""
Train chirp fingerprint from positive and negative examples.

Positive examples (chirps) go in training/chirp/chirp_*.wav
Negative examples (non-chirps) go in training/not_chirp/not_chirp_*.wav

The fingerprint file will contain both:
- chirp_fingerprint: Spectral template of what chirps look like
- non_chirp_fingerprint: Spectral template of what non-chirps look like (optional)

Classification uses both: high similarity to chirp AND low similarity to non-chirp.
"""
import json
import wave
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

CHIRP_TRAIN_DIR = Path("training/chirp")
NON_CHIRP_TRAIN_DIR = Path("training/not_chirp")
OUTPUT_FILE = Path("chirp_fingerprint.json")

INT16_FULL_SCALE = 32768.0


def load_mono_wav(path: Path):
    with wave.open(str(path), "rb") as wf:
        nch = wf.getnchannels()
        sr = wf.getframerate()
        nframes = wf.getnframes()
        frames = wf.readframes(nframes)

    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / INT16_FULL_SCALE

    if nch > 1:
        samples = samples.reshape(-1, nch).mean(axis=1)

    return samples, sr


def compute_avg_spectrum(samples: np.ndarray, sr: int, fft_size: int = 2048):
    if samples.shape[0] < fft_size:
        pad = fft_size - samples.shape[0]
        samples = np.pad(samples, (0, pad))

    hop = fft_size // 2
    window = np.hanning(fft_size)
    specs = []

    for start in range(0, len(samples) - fft_size, hop):
        chunk = samples[start:start + fft_size] * window
        spec = np.abs(np.fft.rfft(chunk))
        specs.append(spec)

    if not specs:
        return None

    avg_spec = np.mean(specs, axis=0)
    norm = np.linalg.norm(avg_spec) + 1e-9
    avg_spec = avg_spec / norm
    return avg_spec, sr, fft_size


def train_fingerprint(train_dir: Path, pattern: str, name: str) -> Optional[Tuple[np.ndarray, int, int]]:
    """
    Train a fingerprint from a directory of audio files.
    
    Args:
        train_dir: Directory containing training files
        pattern: Glob pattern for files (e.g., "chirp_*.wav")
        name: Name for logging (e.g., "chirp" or "non-chirp")
        
    Returns:
        Tuple of (fingerprint, sample_rate, fft_size) or None if no valid files
    """
    wavs = sorted(train_dir.glob(pattern))
    if not wavs:
        print(f"No {pattern} files found in {train_dir}")
        return None

    spectra = []
    sr_used = None
    fft_used = 2048

    print(f"Processing {len(wavs)} {name} training files...")
    for wav_path in wavs:
        try:
            samples, sr = load_mono_wav(wav_path)
            if sr_used is None:
                sr_used = sr
            elif sr != sr_used:
                print(f"Warning: sample rate mismatch in {wav_path.name} ({sr} Hz vs {sr_used} Hz), skipping.")
                continue

            result = compute_avg_spectrum(samples, sr, fft_size=fft_used)
            if result is not None:
                spec, sr, fft_size = result
                spectra.append(spec)
        except Exception as e:
            print(f"Warning: Failed to process {wav_path.name}: {e}")
            continue

    if not spectra:
        print(f"No valid spectra computed for {name}.")
        return None

    mean_spec = np.mean(np.stack(spectra, axis=0), axis=0)
    mean_spec = mean_spec / (np.linalg.norm(mean_spec) + 1e-9)

    print(f"✓ Created {name} fingerprint from {len(spectra)} files")
    return mean_spec, sr_used, fft_used


def main():
    print("=" * 60)
    print("CHIRP FINGERPRINT TRAINING")
    print("=" * 60)
    print()
    
    # Train chirp fingerprint (required)
    chirp_result = train_fingerprint(CHIRP_TRAIN_DIR, "chirp_*.wav", "chirp")
    if chirp_result is None:
        print("ERROR: No chirp training files found. At least one chirp example is required.")
        return
    
    chirp_fp, sr_used, fft_used = chirp_result
    
    # Train non-chirp fingerprint (optional)
    non_chirp_result = train_fingerprint(NON_CHIRP_TRAIN_DIR, "not_chirp_*.wav", "non-chirp")
    non_chirp_fp = None
    if non_chirp_result is not None:
        non_chirp_fp, non_chirp_sr, non_chirp_fft = non_chirp_result
        # Validate sample rate and FFT size match
        if non_chirp_sr != sr_used:
            print(f"WARNING: Non-chirp sample rate ({non_chirp_sr} Hz) doesn't match chirp ({sr_used} Hz)")
            print("  Non-chirp fingerprint will be skipped")
            non_chirp_fp = None
        elif non_chirp_fft != fft_used:
            print(f"WARNING: Non-chirp FFT size ({non_chirp_fft}) doesn't match chirp ({fft_used})")
            print("  Non-chirp fingerprint will be skipped")
            non_chirp_fp = None
    
    # Build output data
    data = {
        "sample_rate": sr_used,
        "fft_size": fft_used,
        "fingerprint": chirp_fp.tolist(),
    }
    
    if non_chirp_fp is not None:
        data["non_chirp_fingerprint"] = non_chirp_fp.tolist()
        print()
        print("✓ Training complete with both chirp and non-chirp fingerprints")
        print("  Classification will use: high similarity to chirp AND low similarity to non-chirp")
    else:
        print()
        print("✓ Training complete with chirp fingerprint only")
        print("  Classification will use: similarity to chirp (no negative training)")
        print("  Tip: Add non-chirp examples to training/not_chirp/ for better accuracy")
    
    # Save fingerprint file
    with OUTPUT_FILE.open("w") as f:
        json.dump(data, f, indent=2)

    print()
    print(f"Saved fingerprint to {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
