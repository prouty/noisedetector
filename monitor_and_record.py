#!/usr/bin/env python3
import subprocess
import datetime
import numpy as np
import sys
import wave
from pathlib import Path

# ------- Configuration -------
DEVICE = "plughw:CARD=Device,DEV=0"   # From your working arecord command
SAMPLE_RATE = 16000                   # Hz
CHANNELS = 1                          # mono
SAMPLE_FORMAT = "S16_LE"              # 16-bit little endian
CHUNK_DURATION = 0.5                  # seconds

# Recording segments
OUTPUT_DIR = Path("recordings")       # directory to store WAV files
SEGMENT_DURATION_SEC = 300            # 5 minutes per file
# -----------------------------

# Derived constants
BYTES_PER_SAMPLE = 2                  # 16-bit = 2 bytes
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
CHUNK_BYTES = CHUNK_SAMPLES * BYTES_PER_SAMPLE * CHANNELS
SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_DURATION_SEC)

# Normalization constant for int16 PCM
INT16_FULL_SCALE = 32768.0            # 2^15, maps int16 to roughly [-1.0, 1.0)


def dbfs(value: float, eps: float = 1e-12) -> float:
    """Convert linear amplitude (0.0–1.0) to dBFS."""
    return 20.0 * np.log10(value + eps)


def start_arecord():
    """Start an arecord subprocess that outputs raw PCM to stdout."""
    cmd = [
        "arecord",
        "-D", DEVICE,
        "-f", SAMPLE_FORMAT,
        "-r", str(SAMPLE_RATE),
        "-c", str(CHANNELS),
        "-q",           # quiet (no ALSA banner)
        "-t", "raw"     # raw PCM data to stdout
    ]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )


def open_new_wav_segment(start_time: datetime.datetime):
    """Open a new WAV file for a recording segment."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fname = start_time.strftime("%Y-%m-%d_%H-%M-%S") + ".wav"
    fpath = OUTPUT_DIR / fname

    wf = wave.open(str(fpath), "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(BYTES_PER_SAMPLE)   # 2 bytes for S16_LE
    wf.setframerate(SAMPLE_RATE)

    print(f"[INFO] Started new segment: {fpath}")
    return wf, fpath


def main():
    print("Starting audio monitor + recorder (arecord-based)...")
    print(f"Device: {DEVICE}")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print("Press Ctrl+C to stop.\n")

    proc = start_arecord()

    if proc.stdout is None:
        print("[ERROR] Failed to open arecord stdout.")
        sys.exit(1)

    wav_file = None
    current_path = None
    samples_written_in_segment = 0

    try:
        # Setup initial segment
        segment_start_time = datetime.datetime.now()
        wav_file, current_path = open_new_wav_segment(segment_start_time)
        samples_written_in_segment = 0

        while True:
            # Read one chunk worth of audio
            data = proc.stdout.read(CHUNK_BYTES)

            if not data or len(data) < CHUNK_BYTES:
                # arecord stopped or stream interrupted
                if proc.stderr:
                    err = proc.stderr.read().decode(errors="ignore").strip()
                    if err:
                        print(f"\n[arecord stderr] {err}")
                print("\n[INFO] arecord stream ended.")
                break

            # --- Monitoring / analysis part ---

            # Convert raw bytes to numpy array of int16
            samples = np.frombuffer(data, dtype="<i2")  # little-endian int16

            if samples.size == 0:
                continue

            # Normalize to roughly -1.0..1.0 float
            float_samples = samples.astype(np.float32) / INT16_FULL_SCALE

            # Compute peak and RMS
            peak = float(np.max(np.abs(float_samples)))
            rms = float(np.sqrt(np.mean(float_samples ** 2)))

            peak_db = dbfs(peak)
            rms_db = dbfs(rms)

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"{timestamp} | peak: {peak_db:6.1f} dBFS | "
                f"rms: {rms_db:6.1f} dBFS",
                flush=True
            )

            # --- Recording part ---

            # Write raw PCM bytes directly to WAV file
            wav_file.writeframesraw(data)
            samples_written_in_segment += CHUNK_SAMPLES

            # Check if we need to roll over to a new segment
            if samples_written_in_segment >= SEGMENT_SAMPLES:
                wav_file.close()
                print(f"[INFO] Closed segment: {current_path}")

                segment_start_time = datetime.datetime.now()
                wav_file, current_path = open_new_wav_segment(segment_start_time)
                samples_written_in_segment = 0

    except KeyboardInterrupt:
        print("\nStopping monitor (Ctrl+C)...")

    finally:
        # Clean up WAV file
        if wav_file is not None:
            try:
                wav_file.close()
                print(f"[INFO] Closed segment: {current_path}")
            except Exception:
                pass

        # Clean up arecord process
        if proc and proc.poll() is None:
            proc.terminate()
        print("Monitor stopped.")


if __name__ == "__main__":
    main()

