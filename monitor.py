#!/usr/bin/env python3
import subprocess
import datetime
import json
import csv
import sys
import argparse
from pathlib import Path
from collections import deque
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import wave

import config_loader

# Normalization constant for int16 PCM
INT16_FULL_SCALE = 32768.0            # 2^15, maps int16 to roughly [-1.0, 1.0)
BYTES_PER_SAMPLE = 2                  # 16-bit = 2 bytes

# Global config (loaded at module level or via run_monitor)
_config: Optional[Dict[str, Any]] = None


def dbfs(value: float, eps: float = 1e-12) -> float:
    """Convert linear amplitude (0.0–1.0) to dBFS."""
    return 20.0 * np.log10(value + eps)


def get_config() -> Dict[str, Any]:
    """Get current configuration, loading defaults if not set."""
    global _config
    if _config is None:
        _config = config_loader.load_config()
    return _config


def start_arecord(config: Dict[str, Any]):
    """Start an arecord subprocess that outputs raw PCM to stdout."""
    audio = config["audio"]
    cmd = [
        "arecord",
        "-D", audio["device"],
        "-f", audio["sample_format"],
        "-r", str(audio["sample_rate"]),
        "-c", str(audio["channels"]),
        "-q",           # quiet (no ALSA banner)
        "-t", "raw"     # raw PCM data to stdout
    ]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )


def open_new_wav_segment(start_time: datetime.datetime, config: Dict[str, Any]):
    """Open a new WAV file for a recording segment."""
    output_dir = Path(config["recording"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = start_time.strftime("%Y-%m-%d_%H-%M-%S") + ".wav"
    fpath = output_dir / fname

    audio = config["audio"]
    wf = wave.open(str(fpath), "wb")
    wf.setnchannels(audio["channels"])
    wf.setsampwidth(BYTES_PER_SAMPLE)   # 2 bytes for S16_LE
    wf.setframerate(audio["sample_rate"])

    print(f"[INFO] Started new segment: {fpath}")
    return wf, fpath


def load_initial_baseline(config: Dict[str, Any]):
    """Try to load baseline.rms_db from baseline.json, else return None."""
    baseline_file = Path(config["event_detection"]["baseline_file"])
    if baseline_file.exists():
        try:
            data = json.load(baseline_file.open())
            rms_db = float(data.get("rms_db"))
            print(f"[INFO] Loaded baseline RMS from {baseline_file}: {rms_db:.1f} dBFS")
            return rms_db
        except Exception as e:
            print(f"[WARN] Failed to load baseline.json: {e}")
    print("[INFO] No baseline.json found; using rolling baseline only.")
    return None


def ensure_events_header(config: Dict[str, Any]):
    """Ensure EVENTS_FILE has a header row."""
    events_file = Path(config["event_detection"]["events_file"])
    if not events_file.exists():
        with events_file.open("w", newline="") as f:
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
                "confidence",
                "rejection_reason",
                "reviewed",
            ])


def log_event(event, config: Dict[str, Any]):
    """Append a single event dict to EVENTS_FILE."""
    ensure_events_header(config)
    events_file = Path(config["event_detection"]["events_file"])
    with events_file.open("a", newline="") as f:
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
            f"{event.get('confidence', 0.0):.3f}" if event.get("confidence") is not None else "",
            event.get("rejection_reason", ""),
            event.get("reviewed", ""),  # User can fill this in manually
        ])
    chirp_status = "CHIRP" if event.get("is_chirp") else "noise"
    conf_str = f", confidence={event.get('confidence', 0.0):.2f}" if event.get("confidence") is not None else ""
    print(
        f"[EVENT] Logged {chirp_status}: {event['start_timestamp']} – {event['end_timestamp']} "
        f"({event['duration_sec']:.2f}s, max_rms {event['max_rms_db']:.1f} dBFS{conf_str})"
    )


def save_event_clip(event_start_time: datetime.datetime, event_chunks, config: Dict[str, Any]):
    """Save a short WAV clip for an event and return its path."""
    clips_dir = Path(config["event_clips"]["clips_dir"])
    clips_dir.mkdir(parents=True, exist_ok=True)
    fname = event_start_time.strftime("clip_%Y-%m-%d_%H-%M-%S.wav")
    fpath = clips_dir / fname

    audio = config["audio"]
    with wave.open(str(fpath), "wb") as wf:
        wf.setnchannels(audio["channels"])
        wf.setsampwidth(BYTES_PER_SAMPLE)
        wf.setframerate(audio["sample_rate"])
        for chunk in event_chunks:
            wf.writeframes(chunk)

    print(f"[CLIP] Saved event clip: {fpath}")
    return fpath


def run_monitor(config_path: Optional[Path] = None):
    """Run the audio monitor with optional config file."""
    global _config
    _config = config_loader.load_config(config_path)
    config = _config
    
    # Calculate derived constants from config
    audio = config["audio"]
    recording = config["recording"]
    event_detection = config["event_detection"]
    event_clips = config["event_clips"]
    
    chunk_duration = audio["chunk_duration"]
    sample_rate = audio["sample_rate"]
    channels = audio["channels"]
    chunk_samples = int(sample_rate * chunk_duration)
    chunk_bytes = chunk_samples * BYTES_PER_SAMPLE * channels
    segment_samples = int(sample_rate * recording["segment_duration_sec"])
    pre_roll_chunks = int(event_clips["pre_roll_sec"] / chunk_duration)
    baseline_window_chunks = event_detection["baseline_window_chunks"]
    
    print("Starting audio monitor + recorder (arecord-based)...")
    print(f"Device: {audio['device']}")
    print(f"Output directory: {Path(recording['output_dir']).resolve()}")
    print(f"Events log: {Path(event_detection['events_file']).resolve()}")
    print(f"Clips directory: {Path(event_clips['clips_dir']).resolve()}")
    print("Press Ctrl+C to stop.\n")

    fingerprint_info = load_chirp_fingerprint(config)

    proc = start_arecord(config)

    if proc.stdout is None:
        print("[ERROR] Failed to open arecord stdout.")
        return

    wav_file = None
    current_path = None
    samples_written_in_segment = 0

    # Event detection state
    baseline_rms_db = load_initial_baseline(config)
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
    pre_roll_buffer = deque(maxlen=pre_roll_chunks)  # raw bytes
    event_chunks = None  # list of raw bytes that will become the clip
    event_actual_start_idx = None  # index in event_chunks where actual event starts (after pre-roll)

    try:
        # Setup initial segment
        segment_start_time = datetime.datetime.now()
        wav_file, current_path = open_new_wav_segment(segment_start_time, config)
        samples_written_in_segment = 0

        while True:
            # Read one chunk worth of audio
            data = proc.stdout.read(chunk_bytes)

            if not data or len(data) < chunk_bytes:
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
            
            # Remove DC offset (common in Pi audio capture, especially USB mics)
            # Use exponential moving average for DC tracking to handle slow drift
            if not hasattr(run_monitor, '_dc_offset_ema'):
                run_monitor._dc_offset_ema = 0.0
            alpha = 0.001  # Slow adaptation for DC offset
            run_monitor._dc_offset_ema = alpha * float(np.mean(float_samples)) + (1 - alpha) * run_monitor._dc_offset_ema
            float_samples = float_samples - run_monitor._dc_offset_ema

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
                if len(baseline_window) > baseline_window_chunks:
                    baseline_window.pop(0)

                if baseline_window:
                    # Use a low percentile to represent "typical quiet"
                    # 20th percentile is good - it ignores occasional spikes while capturing true baseline
                    # Also filter out any anomalously quiet values (likely dropouts or processing errors)
                    valid_baseline = [v for v in baseline_window if np.isfinite(v) and v > -100]
                    if valid_baseline:
                        baseline_rms_db = float(np.percentile(valid_baseline, 20))
                    else:
                        baseline_rms_db = None

            # Determine threshold
            effective_baseline = baseline_rms_db if baseline_rms_db is not None else rms_db
            threshold_db = effective_baseline + event_detection["threshold_above_baseline_db"]

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
                    if duration_sec >= event_detection["min_event_duration_sec"]:
                        # Save clip (includes pre-roll for context)
                        clip_path = save_event_clip(event_start_time, event_chunks or [], config)

                        is_chirp = False
                        similarity = None
                        confidence = None
                        rejection_reason = None

                        # For chirp classification, use only actual event chunks (exclude pre-roll)
                        # Note: The ending chunk (below threshold) is never appended to event_chunks,
                        # so the last chunk in event_chunks is the last valid event chunk.
                        if event_chunks is not None and event_actual_start_idx is not None:
                            # Use all chunks from actual event start to the end
                            # (the ending chunk below threshold was never added, so no need to exclude it)
                            actual_event_chunks = event_chunks[event_actual_start_idx:]
                            if actual_event_chunks:
                                is_chirp, similarity, confidence, rejection_reason = classify_event_is_chirp(
                                    actual_event_chunks, fingerprint_info, duration_sec, config
                                )

                        # Compute offset within segment
                        offset_sec = (event_start_offset_samples or 0) / float(sample_rate)
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
                            "confidence": confidence,
                            "rejection_reason": rejection_reason,
                            "clip_file": clip_path,
                        }
                        log_event(event, config)

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
            samples_written_in_segment += chunk_samples

            # Check if we need to roll over to a new segment
            if samples_written_in_segment >= segment_samples:
                wav_file.close()
                print(f"[INFO] Closed segment: {current_path}")

                segment_start_time = datetime.datetime.now()
                wav_file, current_path = open_new_wav_segment(segment_start_time, config)
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

def load_chirp_fingerprint(config: Dict[str, Any]):
    fingerprint_file = Path(config["chirp_classification"]["fingerprint_file"])
    if not fingerprint_file.exists():
        print("[INFO] No chirp_fingerprint.json found; chirp classification disabled.")
        return None

    try:
        data = json.load(fingerprint_file.open())
        fp = np.array(data["fingerprint"], dtype=np.float32)
        fp = fp / (np.linalg.norm(fp) + 1e-9)
        sr = data["sample_rate"]
        fft_size = data["fft_size"]
        
        # Validate sample rate matches current configuration
        expected_sr = config["audio"]["sample_rate"]
        if sr != expected_sr:
            print(f"[WARN] Fingerprint sample rate ({sr} Hz) doesn't match config sample_rate ({expected_sr} Hz). Chirp classification may be inaccurate.")
        
        print(f"[INFO] Loaded chirp fingerprint from {fingerprint_file} (sr={sr}Hz, fft_size={fft_size})")
        return {"fingerprint": fp, "sample_rate": sr, "fft_size": fft_size}
    except Exception as e:
        print(f"[WARN] Failed to load chirp fingerprint: {e}")
        return None


def compute_event_spectrum_from_chunks(chunks, sample_rate, fft_size):
    if not chunks:
        return None

    raw = b"".join(chunks)
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / INT16_FULL_SCALE
    
    # Remove DC offset before spectral analysis
    # DC component can skew frequency analysis, especially on Pi hardware
    dc_offset = np.mean(samples)
    samples = samples - dc_offset

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
    
    # Apply high-pass filter in frequency domain to remove very low-frequency noise
    # This helps with Pi hardware artifacts and room rumble
    if specs:
        freq_resolution = sample_rate / fft_size
        # Remove DC and very low frequencies (< 20 Hz typically)
        # This is done after FFT to avoid phase issues with time-domain filtering
        for i, spec in enumerate(specs):
            # Zero out DC and very low frequencies
            cutoff_bin = max(1, int(20.0 / freq_resolution))  # 20 Hz high-pass
            spec[:cutoff_bin] = 0.0
            specs[i] = spec

    if not specs:
        return None

    avg_spec = np.mean(specs, axis=0)
    avg_spec = avg_spec / (np.linalg.norm(avg_spec) + 1e-9)
    return avg_spec


def compute_attack_decay_ratio(chunk_rms_values: list) -> Optional[float]:
    """
    Calculate attack/decay ratio. Higher values indicate sharp attack (chirp-like).
    Returns ratio of attack time to decay time, or None if insufficient data.
    """
    if len(chunk_rms_values) < 3:
        return None
    
    # Find peak
    peak_idx = chunk_rms_values.index(max(chunk_rms_values))
    if peak_idx == 0 or peak_idx == len(chunk_rms_values) - 1:
        return None  # Peak at edge, can't calculate properly
    
    # Attack: time to reach 90% of peak from start
    peak_value = chunk_rms_values[peak_idx]
    attack_threshold = peak_value * 0.9
    attack_time = peak_idx
    for i in range(peak_idx):
        if chunk_rms_values[i] >= attack_threshold:
            attack_time = peak_idx - i
            break
    
    # Decay: time to drop to 10% of peak after peak
    decay_threshold = peak_value * 0.1
    decay_time = len(chunk_rms_values) - peak_idx - 1
    for i in range(peak_idx + 1, len(chunk_rms_values)):
        if chunk_rms_values[i] <= decay_threshold:
            decay_time = i - peak_idx
            break
    
    if decay_time == 0:
        return None
    
    return attack_time / decay_time


def compute_spectral_centroid(spectrum: np.ndarray, sample_rate: int, fft_size: int) -> float:
    """
    Calculate spectral centroid (weighted frequency center of mass).
    Higher values indicate more high-frequency content (chirp-like).
    """
    freq_resolution = sample_rate / fft_size
    frequencies = np.arange(len(spectrum)) * freq_resolution
    
    magnitude = np.abs(spectrum)
    total_magnitude = np.sum(magnitude)
    
    if total_magnitude == 0:
        return 0.0
    
    centroid = np.sum(frequencies * magnitude) / total_magnitude
    return float(centroid)


def find_best_chirp_segment(
    event_chunks,
    fingerprint_info,
    config: Dict[str, Any]
) -> Tuple[Optional[List[bytes]], Optional[float], Optional[str]]:
    """
    Find the best segment within event_chunks that matches the chirp fingerprint.
    Uses sliding windows to find the segment with highest similarity and best frequency characteristics.
    
    Returns:
        (best_chunks, best_similarity, rejection_reason) or (None, None, reason) if no good segment found
    """
    if fingerprint_info is None or not event_chunks:
        return None, None, "no_fingerprint_or_chunks"
    
    chirp_cfg = config["chirp_classification"]
    freq_cfg = chirp_cfg["frequency_filtering"]
    fp = fingerprint_info["fingerprint"]
    fft_size = fingerprint_info["fft_size"]
    sr = fingerprint_info["sample_rate"]
    
    # Try different window sizes and positions
    # Strategy: Try the last portion of the event (where chirp likely is), then try other segments
    num_chunks = len(event_chunks)
    if num_chunks < 1:
        return None, None, "insufficient_chunks"
    
    # Window sizes to try (as fractions of total event)
    # Start with smaller windows to focus on the end where chirp likely is
    # Try very small windows first to isolate the chirp from initial noise
    window_sizes = [0.14, 0.25, 0.33, 0.5, 0.75, 1.0]  # Last 1 chunk (~14%), 25%, 33%, 50%, 75%, full event
    best_chunks = None
    best_similarity = -1.0
    best_score = -1.0  # Combined score considering similarity and frequency characteristics
    
    freq_resolution = sr / fft_size
    fan_noise_max_bin = int(freq_cfg["fan_noise_max_freq_hz"] / freq_resolution)
    chirp_min_bin = int(freq_cfg["chirp_min_freq_hz"] / freq_resolution)
    
    for window_size in window_sizes:
        # Start from the end (most recent chunks, where chirp likely is)
        num_chunks_in_window = max(1, int(num_chunks * window_size))
        start_idx = num_chunks - num_chunks_in_window
        window_chunks = event_chunks[start_idx:]
        
        if not window_chunks:
            continue
        
        # Compute spectrum for this window
        window_spec = compute_event_spectrum_from_chunks(window_chunks, sr, fft_size)
        if window_spec is None:
            continue
        
        # Calculate similarity
        sim = float(np.dot(fp, window_spec))
        
        # Calculate frequency characteristics
        total_energy = np.sum(window_spec)
        if total_energy == 0:
            continue
        
        low_freq_energy = np.sum(window_spec[:fan_noise_max_bin])
        high_freq_energy = np.sum(window_spec[chirp_min_bin:])
        low_freq_ratio = low_freq_energy / total_energy
        high_freq_ratio = high_freq_energy / total_energy
        
        # Score: combine similarity with frequency quality
        # Prefer segments with high similarity AND good frequency characteristics
        # Give bonus for segments that pass frequency filters
        freq_score = 1.0
        passes_low_freq = low_freq_ratio <= freq_cfg["low_freq_energy_threshold"]
        min_high_freq = freq_cfg.get("high_freq_energy_min_ratio", 0.1)
        passes_high_freq = high_freq_ratio >= min_high_freq
        
        if passes_low_freq and passes_high_freq:
            # Bonus for segments that pass both frequency filters
            freq_score = 1.2
        elif passes_low_freq:
            # Slight bonus for passing low-freq filter
            freq_score = 1.1
        elif passes_high_freq:
            # Slight bonus for passing high-freq filter
            freq_score = 1.05
        else:
            # Penalize segments that fail both
            if low_freq_ratio > freq_cfg["low_freq_energy_threshold"]:
                freq_score *= (1.0 - (low_freq_ratio - freq_cfg["low_freq_energy_threshold"]))
            if high_freq_ratio < min_high_freq:
                freq_score *= (high_freq_ratio / min_high_freq)
        
        # Combined score: similarity weighted by frequency quality
        # Also give slight preference to smaller windows (more focused on chirp)
        window_size_factor = 1.0 + (1.0 - window_size) * 0.1  # Up to 10% bonus for smaller windows
        combined_score = sim * max(0.0, freq_score) * window_size_factor
        
        if combined_score > best_score:
            best_score = combined_score
            best_similarity = sim
            best_chunks = window_chunks
    
    if best_chunks is None or best_similarity < 0:
        return None, None, "no_valid_segment"
    
    return best_chunks, best_similarity, None


def classify_event_is_chirp(
    event_chunks, 
    fingerprint_info, 
    duration_sec: float,
    config: Dict[str, Any]
) -> Tuple[bool, Optional[float], Optional[float], Optional[str]]:
    """
    Classify if event is a chirp.
    Uses sliding window to find the best matching segment (handles cases where
    noise at beginning masks chirp at end).
    
    Returns:
        (is_chirp, similarity, confidence, rejection_reason)
    """
    if fingerprint_info is None:
        return False, None, None, "no_fingerprint"
    
    chirp_cfg = config["chirp_classification"]
    freq_cfg = chirp_cfg["frequency_filtering"]
    temp_cfg = chirp_cfg["temporal_filtering"]
    conf_cfg = chirp_cfg.get("confidence", {})
    
    fp = fingerprint_info["fingerprint"]
    fft_size = fingerprint_info["fft_size"]
    sr = fingerprint_info["sample_rate"]
    
    # Find the best segment within the event (handles noise at beginning, chirp at end)
    # We do this BEFORE duration check because the best segment might be shorter than the full event
    best_chunks, best_similarity, segment_reason = find_best_chirp_segment(
        event_chunks, fingerprint_info, config
    )
    
    if best_chunks is None:
        return False, None, None, f"no_valid_segment_{segment_reason}"
    
    # Calculate duration of the best segment (not the full event)
    audio_cfg = config["audio"]
    chunk_duration = audio_cfg["chunk_duration"]
    best_segment_duration = len(best_chunks) * chunk_duration
    
    # Temporal filtering: reject if the best segment is too long (door sounds are drawn out)
    if best_segment_duration > temp_cfg["max_duration_sec"]:
        return False, best_similarity, None, f"duration_too_long_{best_segment_duration:.1f}s"
    
    # Use the best segment for classification
    event_spec = compute_event_spectrum_from_chunks(best_chunks, sr, fft_size)
    if event_spec is None:
        return False, None, None, "spectrum_computation_failed"
    
    # Frequency-domain filtering on the best segment
    freq_resolution = sr / fft_size  # Hz per bin
    fan_noise_max_bin = int(freq_cfg["fan_noise_max_freq_hz"] / freq_resolution)
    chirp_min_bin = int(freq_cfg["chirp_min_freq_hz"] / freq_resolution)
    
    # Calculate energy in different frequency ranges
    total_energy = np.sum(event_spec)
    if total_energy > 0:
        low_freq_energy = np.sum(event_spec[:fan_noise_max_bin])
        high_freq_energy = np.sum(event_spec[chirp_min_bin:])
        low_freq_ratio = low_freq_energy / total_energy
        high_freq_ratio = high_freq_energy / total_energy
        
        # Reject if too much low-frequency energy (fan noise)
        # Allow small tolerance if similarity is above threshold (chirp is likely present)
        # This helps when there's residual noise (like a thud) mixed with the chirp
        low_freq_threshold = freq_cfg["low_freq_energy_threshold"]
        similarity_threshold = chirp_cfg["similarity_threshold"]
        if best_similarity >= similarity_threshold:
            # Good similarity - allow slightly more low-freq to handle residual noise from thuds/etc
            # Allow up to 0.32 (instead of 0.30) when similarity is good
            low_freq_threshold = min(0.32, low_freq_threshold + 0.02)
        
        if low_freq_ratio > low_freq_threshold:
            return False, best_similarity, None, f"too_much_low_freq_{low_freq_ratio:.2f}"
        
        # Reject if insufficient high-frequency energy (chirp)
        min_high_freq = freq_cfg.get("high_freq_energy_min_ratio", 0.1)
        if high_freq_ratio < min_high_freq:
            return False, best_similarity, None, f"insufficient_high_freq_{high_freq_ratio:.2f}"
    
    # Temporal envelope analysis: use best_chunks (the segment we're classifying)
    chunk_rms_values = []
    for chunk in best_chunks:
        samples = np.frombuffer(chunk, dtype="<i2").astype(np.float32) / INT16_FULL_SCALE
        if len(samples) > 0:
            rms = float(np.sqrt(np.mean(samples ** 2)))
            chunk_rms_values.append(rms)
    
    energy_concentration_score = 0.5  # Default neutral
    attack_decay_ratio = None
    spectral_centroid = None
    
    if len(chunk_rms_values) > 1:
        # Calculate energy concentration in first half vs second half
        mid_point = len(chunk_rms_values) // 2
        first_half_energy = sum(r**2 for r in chunk_rms_values[:mid_point])
        second_half_energy = sum(r**2 for r in chunk_rms_values[mid_point:])
        total_chunk_energy = first_half_energy + second_half_energy
        
        if total_chunk_energy > 0:
            energy_concentration = first_half_energy / total_chunk_energy
            energy_concentration_score = energy_concentration
            
            # Reject if energy is too evenly distributed (sustained sound like door)
            # But be more lenient for very short segments (2-3 chunks) where this metric is less meaningful
            energy_threshold = temp_cfg["energy_concentration_threshold"]
            if len(chunk_rms_values) <= 3:
                # For very short segments, relax the threshold slightly
                energy_threshold = max(0.3, energy_threshold - 0.2)
            
            if energy_concentration < energy_threshold:
                return False, best_similarity, None, f"energy_too_spread_{energy_concentration:.2f}"
        
        # Calculate attack/decay ratio
        attack_decay_ratio = compute_attack_decay_ratio(chunk_rms_values)
    
    # Calculate spectral centroid
    spectral_centroid = compute_spectral_centroid(event_spec, sr, fft_size)
    
    # Spectral similarity
    sim = float(np.dot(fp, event_spec))  # cosine similarity (because both normalized)
    
    # Check minimum similarity threshold
    similarity_threshold = chirp_cfg["similarity_threshold"]
    if sim < similarity_threshold:
        return False, sim, None, f"similarity_too_low_{sim:.3f}"
    
    # Calculate confidence score if enabled
    confidence = None
    if conf_cfg.get("enabled", True):
        # Normalize scores to 0-1 range
        sim_score = sim  # Already 0-1
        freq_score = min(1.0, spectral_centroid / 4000.0) if spectral_centroid else 0.5  # Normalize to 4kHz max
        temp_score = energy_concentration_score  # Already 0-1
        
        # Weighted combination
        weights = [
            conf_cfg.get("similarity_weight", 0.6),
            conf_cfg.get("frequency_weight", 0.2),
            conf_cfg.get("temporal_weight", 0.2)
        ]
        confidence = (
            weights[0] * sim_score +
            weights[1] * freq_score +
            weights[2] * temp_score
        )
    
    is_chirp = True
    return is_chirp, sim, confidence, None

