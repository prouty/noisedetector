#!/usr/bin/env python3
import json
import wave
from pathlib import Path

import numpy as np

TRAIN_DIR = Path("training/chirp")
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


def main():
    wavs = sorted(TRAIN_DIR.glob("chirp_*.wav"))
    if not wavs:
        print(f"No chirp_*.wav files found in {TRAIN_DIR}")
        return

    spectra = []
    sr_used = None
    fft_used = 2048

    for wav_path in wavs:
        samples, sr = load_mono_wav(wav_path)
        if sr_used is None:
            sr_used = sr
        elif sr != sr_used:
            print(f"Warning: sample rate mismatch in {wav_path}, skipping.")
            continue

        result = compute_avg_spectrum(samples, sr, fft_size=fft_used)
        if result is not None:
            spec, sr, fft_size = result
            spectra.append(spec)

    if not spectra:
        print("No valid spectra computed.")
        return

    mean_spec = np.mean(np.stack(spectra, axis=0), axis=0)
    mean_spec = mean_spec / (np.linalg.norm(mean_spec) + 1e-9)

    data = {
        "sample_rate": sr_used,
        "fft_size": fft_used,
        "fingerprint": mean_spec.tolist(),
    }

    with OUTPUT_FILE.open("w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved fingerprint to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
