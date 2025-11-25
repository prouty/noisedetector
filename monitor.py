#!/usr/bin/env python3
import subprocess
import datetime
import json
import csv
from pathlib import Path
from collections import deque

import numpy as np
import wave

# ------- Configuration -------
DEVICE = "plughw:CARD=Device,DEV=0"   # Your working arecord device
SAMPLE_RATE = 16000                   # Hz
CHANNELS = 1                          # mono
SAMPLE_FORMAT = "S16_LE"              # 16-bit little endian
CHUNK_DURATION = 0.5                  # seconds

# Recording segments
OUTPUT_DIR = Path("clips")       # directory to store WAV files
SEGMENT_DURATION_SEC = 300            # 5 minutes per file

# Event detection
BASELINE_FILE = Path("baseline.json")
EVENTS_FILE = Path("events.csv")
THRESHOLD_ABOVE_BASELINE_DB = 10.0    # trigger if RMS > baseline + this
MIN_EVENT_DURATION_SEC = 0.5          # ignore super-short blips
BASELINE_WINDOW_CHUNKS = 120          # for rolling baseline (~60s)

# Event clips
CLIPS_DIR = Path("clips")
PRE_ROLL_SEC = 2.0                    # seconds of audio before event
PRE_ROLL_CHUNKS = int(PRE_ROLL_SEC / CHUNK_DURATION)

# -----------------------------

# Derived constants
BYTES_PER_SAMPLE = 2                  # 16-bit = 2 bytes
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
CHUNK_BYTES = CHUNK_SAMPLES * BYTES_PER_SAMPLE * CHANNELS
SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_DURATION_SEC)

# Normalization constant for int16 PCM
INT16_FULL_SCALE = 32768.0            # 2^15, maps int16 to roughly [-1.0, 1.0)

# Training data
FINGERPRINT_FILE = Path("chirp_fingerprint.json")
CHIRP_SIMILARITY_THRESHOLD = 0.8  # start here, we can tune later


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


def load_initial_baseline():
    """Try to load baseline.rms_db from baseline.json, else return None."""
    if BASELINE_FILE.exists():
        try:
            data = json.load(BASELINE_FILE.open())
            rms_db = float(data.get("rms_db"))
            print(f"[INFO] Loaded baseline RMS from {BASELINE_FILE}: {rms_db:.1f} dBFS")
            return rms_db
        except Exception as e:
            print(f"[WARN] Failed to load baseline.json: {e}")
    print("[INFO] No baseline.json found; using rolling baseline only.")
    return None


def ensure_events_header():
    """Ensure EVENTS_FILE has a header row."""
    if not EVENTS_FILE.exists():
        with EVENTS_FILE.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "start_timestamp",
                "end_timestamp",
                "duration_sec",
                "max_peak_db",
                "max_rms_db",
                "baseline_rms_db",
                "segment_file",
                "segment_offset_sec",
                "clip_file",
                "is_chirp",
                "chirp_similarity",
            ])


def log_event(event):
    """Append a single event dict to EVENTS_FILE."""
    ensure_events_header()
    with EVENTS_FILE.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            event["start_timestamp"],
            event["end_timestamp"],
            f"{event['duration_sec']:.2f}",
            f"{event['max_peak_db']:.2f}",
            f"{event['max_rms_db']:.2f}",
            f"{event['baseline_rms_db']:.2f}"
                if event["baseline_rms_db"] is not None else "",
            str(event["segment_file"]),
            f"{event['segment_offset_sec']:.2f}",
            str(event.get("clip_file", "")),
            "TRUE" if event.get("is_chirp") else "FALSE",
            f"{event.get('chirp_similarity', 0.0):.3f}" if event.get("chirp_similarity") is not None else "",
        ])
    print(
        f"[EVENT] Logged event: {event['start_timestamp']} – {event['end_timestamp']} "
        f"({event['duration_sec']:.2f}s, max_rms {event['max_rms_db']:.1f} dBFS)"
    )


def save_event_clip(event_start_time: datetime.datetime, event_chunks):
    """Save a short WAV clip for an event and return its path."""
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    fname = event_start_time.strftime("clip_%Y-%m-%d_%H-%M-%S.wav")
    fpath = CLIPS_DIR / fname

    with wave.open(str(fpath), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(BYTES_PER_SAMPLE)
        wf.setframerate(SAMPLE_RATE)
        for chunk in event_chunks:
            wf.writeframes(chunk)

    print(f"[CLIP] Saved event clip: {fpath}")
    return fpath


def run_monitor():
    print("Starting audio monitor + recorder (arecord-based)...")
    print(f"Device: {DEVICE}")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print(f"Events log: {EVENTS_FILE.resolve()}")
    print(f"Clips directory: {CLIPS_DIR.resolve()}")
    print("Press Ctrl+C to stop.\n")

    fingerprint_info = load_chirp_fingerprint()

    proc = start_arecord()

    if proc.stdout is None:
        print("[ERROR] Failed to open arecord stdout.")
        return

    wav_file = None
    current_path = None
    samples_written_in_segment = 0

    # Event detection state
    baseline_rms_db = load_initial_baseline()
    baseline_window = []  # rolling list of recent rms_db values (when not in event)
    in_event = False
    event_start_time = None
    event_end_time = None
    event_max_peak_db = None
    event_max_rms_db = None
    event_baseline_at_start = None
    event_start_offset_samples = None  # in current segment
    event_segment_file = None  # segment file path when event started

    # Clip state
    pre_roll_buffer = deque(maxlen=PRE_ROLL_CHUNKS)  # raw bytes
    event_chunks = None  # list of raw bytes that will become the clip
    event_actual_start_idx = None  # index in event_chunks where actual event starts (after pre-roll)

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

            timestamp = datetime.datetime.now()
            timestamp_str = timestamp.strftime("%Y-%m-%dT%H:%M:%S")

            # Rolling baseline update (only when not in event)
            if not in_event:
                baseline_window.append(rms_db)
                if len(baseline_window) > BASELINE_WINDOW_CHUNKS:
                    baseline_window.pop(0)

                if baseline_window:
                    # Use a low percentile to represent "typical quiet"
                    baseline_rms_db = float(np.percentile(baseline_window, 20))

            # Determine threshold
            effective_baseline = baseline_rms_db if baseline_rms_db is not None else rms_db
            threshold_db = effective_baseline + THRESHOLD_ABOVE_BASELINE_DB

            # Maintain pre-roll buffer when not in an event
            if not in_event:
                pre_roll_buffer.append(data)

            # Event state machine
            if not in_event:
                # Check if we should start an event
                if rms_db > threshold_db:
                    in_event = True
                    event_start_time = timestamp
                    event_end_time = timestamp
                    event_max_peak_db = peak_db
                    event_max_rms_db = rms_db
                    event_baseline_at_start = baseline_rms_db
                    event_start_offset_samples = samples_written_in_segment
                    # Capture the segment file path when event starts
                    event_segment_file = str(current_path) if current_path else None

                    # Build initial clip buffer: pre-roll + current chunk
                    # Note: pre-roll is kept for clip saving (context), but chirp classification
                    # will use only the actual event chunks (see classify_event_is_chirp)
                    if pre_roll_buffer:
                        event_chunks = list(pre_roll_buffer)
                    else:
                        event_chunks = []
                    event_chunks.append(data)
                    
                    # Track where actual event starts (after pre-roll) for chirp classification
                    event_actual_start_idx = len(event_chunks) - 1  # Index of first actual event chunk
            else:
                # Already in event - update stats
                event_end_time = timestamp
                if peak_db > event_max_peak_db:
                    event_max_peak_db = peak_db
                if rms_db > event_max_rms_db:
                    event_max_rms_db = rms_db

                # Check if event ended (back below baseline+threshold)
                if rms_db <= threshold_db:
                    # Compute duration
                    duration_sec = (event_end_time - event_start_time).total_seconds()
                    if duration_sec >= MIN_EVENT_DURATION_SEC:
                        # Save clip (includes pre-roll for context)
                        clip_path = save_event_clip(event_start_time, event_chunks or [])

                        is_chirp = False
                        similarity = None

                        # For chirp classification, use only actual event chunks (exclude pre-roll)
                        # Note: The ending chunk (below threshold) is never appended to event_chunks,
                        # so the last chunk in event_chunks is the last valid event chunk.
                        if event_chunks is not None and event_actual_start_idx is not None:
                            # Use all chunks from actual event start to the end
                            # (the ending chunk below threshold was never added, so no need to exclude it)
                            actual_event_chunks = event_chunks[event_actual_start_idx:]
                            if actual_event_chunks:
                                is_chirp, similarity = classify_event_is_chirp(actual_event_chunks, fingerprint_info)

                        # Compute offset within segment
                        offset_sec = (event_start_offset_samples or 0) / float(SAMPLE_RATE)
                        event = {
                            "start_timestamp": event_start_time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "end_timestamp": event_end_time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "duration_sec": duration_sec,
                            "max_peak_db": event_max_peak_db,
                            "max_rms_db": event_max_rms_db,
                            "baseline_rms_db": event_baseline_at_start,
                            "segment_file": event_segment_file,
                            "segment_offset_sec": offset_sec,
                            "is_chirp": is_chirp,
                            "chirp_similarity": similarity,
                            "clip_file": clip_path,
                        }
                        log_event(event)

                    # Reset event/clip state
                    in_event = False
                    event_start_time = None
                    event_end_time = None
                    event_max_peak_db = None
                    event_max_rms_db = None
                    event_baseline_at_start = None
                    event_start_offset_samples = None
                    event_segment_file = None
                    event_chunks = None
                    event_actual_start_idx = None
                else:
                    # Still in event - append this chunk to the clip buffer
                    if event_chunks is not None:
                        event_chunks.append(data)

            # Print live metrics (for your sanity while testing)
            baseline_str = f"{baseline_rms_db:6.1f}" if baseline_rms_db is not None else "  N/A "
            print(
                f"{timestamp_str} | peak: {peak_db:6.1f} dBFS | "
                f"rms: {rms_db:6.1f} dBFS | "
                f"baseline: {baseline_str} dBFS",
                flush=True
            )

            # --- Recording part ---

            # Write raw PCM bytes directly to WAV file
            wav_file.writeframes(data)
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

def load_chirp_fingerprint():
    if not FINGERPRINT_FILE.exists():
        print("[INFO] No chirp_fingerprint.json found; chirp classification disabled.")
        return None

    try:
        data = json.load(FINGERPRINT_FILE.open())
        fp = np.array(data["fingerprint"], dtype=np.float32)
        fp = fp / (np.linalg.norm(fp) + 1e-9)
        sr = data["sample_rate"]
        fft_size = data["fft_size"]
        
        # Validate sample rate matches current configuration
        if sr != SAMPLE_RATE:
            print(f"[WARN] Fingerprint sample rate ({sr} Hz) doesn't match SAMPLE_RATE ({SAMPLE_RATE} Hz). Chirp classification may be inaccurate.")
        
        print(f"[INFO] Loaded chirp fingerprint from {FINGERPRINT_FILE} (sr={sr}Hz, fft_size={fft_size})")
        return {"fingerprint": fp, "sample_rate": sr, "fft_size": fft_size}
    except Exception as e:
        print(f"[WARN] Failed to load chirp fingerprint: {e}")
        return None


def compute_event_spectrum_from_chunks(chunks, sample_rate, fft_size):
    if not chunks:
        return None

    raw = b"".join(chunks)
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / INT16_FULL_SCALE

    if samples.shape[0] < fft_size:
        pad = fft_size - samples.shape[0]
        samples = np.pad(samples, (0, pad))

    hop = fft_size // 2
    window = np.hanning(fft_size)
    specs = []

    # Fix edge case: if samples exactly equals fft_size, ensure at least one window
    max_start = max(0, len(samples) - fft_size)
    if max_start == 0 and len(samples) >= fft_size:
        # Single window case
        chunk = samples[0:fft_size] * window
        spec = np.abs(np.fft.rfft(chunk))
        specs.append(spec)
    else:
        for start in range(0, max_start, hop):
            chunk = samples[start:start + fft_size] * window
            spec = np.abs(np.fft.rfft(chunk))
            specs.append(spec)

    if not specs:
        return None

    avg_spec = np.mean(specs, axis=0)
    avg_spec = avg_spec / (np.linalg.norm(avg_spec) + 1e-9)
    return avg_spec


def classify_event_is_chirp(event_chunks, fingerprint_info):
    if fingerprint_info is None:
        return False, None

    fp = fingerprint_info["fingerprint"]
    fft_size = fingerprint_info["fft_size"]
    sr = fingerprint_info["sample_rate"]

    event_spec = compute_event_spectrum_from_chunks(event_chunks, sr, fft_size)
    if event_spec is None:
        return False, None

    sim = float(np.dot(fp, event_spec))  # cosine similarity (because both normalized)
    is_chirp = sim >= CHIRP_SIMILARITY_THRESHOLD
    return is_chirp, sim

