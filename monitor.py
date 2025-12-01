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


def start_arecord(config: Dict[str, Any]) -> subprocess.Popen:
    """
    Start an arecord subprocess that outputs raw PCM to stdout.
    
    This function launches arecord (ALSA recording utility) as a subprocess.
    The process runs continuously, outputting raw PCM audio data to stdout.
    
    Args:
        config: Configuration dictionary containing audio settings
        
    Returns:
        Popen process object with stdout/stderr pipes
        
    Raises:
        FileNotFoundError: If arecord command is not found
        ValueError: If audio device configuration is invalid
        
    Common failures:
        - "Device or resource busy": Another process is using the audio device
          Solution: Stop noise-monitor service or other audio processes
        - "No such file or directory": Audio device doesn't exist
          Solution: Check device name with 'arecord -l', update config.json
        - "Permission denied": User doesn't have access to audio device
          Solution: Add user to 'audio' group: sudo usermod -a -G audio $USER
    """
    audio = config["audio"]
    device = audio["device"]
    
    # Validate device string format (basic check)
    if not device or not isinstance(device, str):
        raise ValueError(f"Invalid audio device configuration: {device}. Expected string like 'plughw:CARD=Device,DEV=0'")
    
    cmd = [
        "arecord",
        "-D", device,
        "-f", audio["sample_format"],
        "-r", str(audio["sample_rate"]),
        "-c", str(audio["channels"]),
        "-q",           # quiet (no ALSA banner)
        "-t", "raw"     # raw PCM data to stdout
    ]
    
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            "arecord command not found. Install alsa-utils: sudo apt-get install alsa-utils"
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to start arecord process. Command: {' '.join(cmd)}. Error: {e}"
        )
    
    # Give process a moment to initialize and check if it failed immediately
    import time
    time.sleep(0.1)
    if proc.poll() is not None:
        # Process exited immediately - read stderr for error message
        stderr_msg = ""
        if proc.stderr:
            stderr_msg = proc.stderr.read().decode(errors="ignore").strip()
        
        error_hints = {
            "Device or resource busy": "Audio device is in use. Stop noise-monitor service: 'make stop'",
            "No such file or directory": f"Audio device '{device}' not found. Check with 'arecord -l'",
            "Permission denied": "No permission to access audio device. Add user to audio group: 'sudo usermod -a -G audio $USER'",
            "Invalid argument": f"Invalid audio device or format. Device: {device}, Format: {audio['sample_format']}"
        }
        
        hint = ""
        for key, msg in error_hints.items():
            if key in stderr_msg:
                hint = f" Hint: {msg}"
                break
        
        raise RuntimeError(
            f"arecord failed to start. Device: {device}. Error: {stderr_msg}.{hint}"
        )
    
    return proc


def open_new_wav_segment(start_time: datetime.datetime, config: Dict[str, Any]) -> Tuple[wave.Wave_write, Path]:
    """
    Open a new WAV file for a recording segment.
    
    Segments are 5-minute continuous recordings that serve as backup/archive.
    Event clips are extracted from these segments.
    
    Args:
        start_time: Timestamp for segment start (used in filename)
        config: Configuration dictionary
        
    Returns:
        Tuple of (wave file object, file path)
        
    Raises:
        OSError: If directory can't be created or file can't be opened
        wave.Error: If WAV format is invalid
        
    File naming:
        Format: YYYY-MM-DD_HH-MM-SS.wav
        Location: output_dir from config (default: clips/)
    """
    output_dir = Path(config["recording"]["output_dir"])
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        raise PermissionError(
            f"Cannot create output directory {output_dir}. Check permissions."
        )
    except Exception as e:
        raise OSError(f"Failed to create output directory {output_dir}: {e}")
    
    fname = start_time.strftime("%Y-%m-%d_%H-%M-%S") + ".wav"
    fpath = output_dir / fname

    audio = config["audio"]
    
    try:
        wf = wave.open(str(fpath), "wb")
        wf.setnchannels(audio["channels"])
        wf.setsampwidth(BYTES_PER_SAMPLE)   # 2 bytes for S16_LE
        wf.setframerate(audio["sample_rate"])
    except OSError as e:
        raise OSError(
            f"Failed to open segment file {fpath}: {e}. "
            f"Check disk space (df -h) and permissions."
        )
    except wave.Error as e:
        raise wave.Error(f"Invalid WAV parameters for {fpath}: {e}")

    print(f"[INFO] Started new segment: {fpath}")
    return wf, fpath


def load_initial_baseline(config: Dict[str, Any]) -> Optional[float]:
    """
    Load initial baseline from named baseline system or fallback to old format.
    
    Supports:
    1. Named baseline system (new): Uses baseline_name from config or active baseline
    2. Old baseline.json file (backward compatibility)
    
    If no baseline file exists, returns None and system uses rolling baseline only.
    Rolling baseline is calculated from recent audio chunks (20th percentile).
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Baseline RMS in dBFS (float) or None if file doesn't exist or is invalid
        
    Common issues:
        - File not found: Normal - system will use rolling baseline
        - Invalid JSON: File corrupted, regenerate with 'python3 baseline.py create <name>'
        - Missing rms_db: Old format, regenerate baseline
    """
    import baseline as baseline_module
    
    # Try named baseline system first
    baseline_name = config.get("event_detection", {}).get("baseline_name")
    
    # Auto-migrate old baseline if needed
    baseline_module.migrate_old_baseline()
    
    # Get baseline name (from config, active, or default)
    index = baseline_module.get_baselines_index()
    if baseline_name is None:
        baseline_name = index.get("active", "default")
    
    # Try to load named baseline
    if baseline_name in index.get("baselines", {}):
        baseline_file = baseline_module.get_baseline_file(baseline_name)
        if baseline_file.exists():
            try:
                history = baseline_module.load_baseline_history(baseline_file)
                if history:
                    latest = history[-1]
                    rms_db = float(latest.get("rms_db"))
                    
                    # Validate reasonable range
                    if not (-100 < rms_db < 0):
                        print(f"[WARN] Baseline RMS value seems unusual: {rms_db:.1f} dBFS")
                        print("  Expected range: -100 to 0 dBFS")
                        print(f"  Regenerate with: python3 baseline.py create {baseline_name}")
                    
                    print(f"[INFO] Loaded baseline '{baseline_name}' RMS: {rms_db:.1f} dBFS")
                    return rms_db
            except Exception as e:
                print(f"[WARN] Failed to load named baseline '{baseline_name}': {e}")
    
    # Fallback to old baseline.json file (backward compatibility)
    baseline_file = Path(config["event_detection"]["baseline_file"])
    
    if not baseline_file.exists():
        print("[INFO] No baseline found; using rolling baseline only.")
        print("  To set initial baseline: python3 baseline.py create <name>")
        return None
    
    try:
        with baseline_file.open() as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[WARN] Invalid JSON in baseline file {baseline_file}: {e}")
        print("  File may be corrupted. Regenerate with: python3 baseline.py create <name>")
        return None
    except Exception as e:
        print(f"[WARN] Failed to read baseline file {baseline_file}: {e}")
        return None
    
    # Handle both single baseline and history array formats
    if isinstance(data, list):
        if len(data) == 0:
            print("[WARN] Baseline history is empty")
            return None
        latest = data[-1]  # Use most recent baseline
    else:
        latest = data
    
    try:
        rms_db = float(latest.get("rms_db"))
        
        # Validate reasonable range
        if not (-100 < rms_db < 0):
            print(f"[WARN] Baseline RMS value seems unusual: {rms_db:.1f} dBFS")
            print("  Expected range: -100 to 0 dBFS")
            print("  Regenerate with: python3 baseline.py create <name>")
        
        print(f"[INFO] Loaded baseline RMS from {baseline_file}: {rms_db:.1f} dBFS")
        return rms_db
    
    except (ValueError, TypeError, KeyError) as e:
        print(f"[WARN] Invalid baseline data in {baseline_file}: {e}")
        print("  Missing or invalid 'rms_db' field. Regenerate with: python3 baseline.py create <name>")
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


def log_event(event: Dict[str, Any], config: Dict[str, Any]):
    """
    Append a single event to the events CSV file.
    
    This function writes event data to events.csv in a thread-safe manner
    (single process, append-only). The CSV format allows easy analysis with
    spreadsheet software or pandas.
    
    Args:
        event: Dictionary containing event data (see required fields below)
        config: Configuration dictionary
        
    Required event fields:
        - start_timestamp: ISO format timestamp string
        - end_timestamp: ISO format timestamp string
        - duration_sec: Event duration in seconds (float)
        - max_peak_db: Maximum peak level in dBFS (float)
        - max_rms_db: Maximum RMS level in dBFS (float)
        - baseline_rms_db: Baseline level at event start (float or None)
        - segment_file: Path to segment WAV file (string)
        - segment_offset_sec: Offset within segment in seconds (float)
        - clip_file: Path to event clip (string, optional)
        - is_chirp: Boolean classification result
        - chirp_similarity: Similarity score 0-1 (float or None)
        - confidence: Confidence score 0-1 (float or None)
        - rejection_reason: Why non-chirp was rejected (string, optional)
        - reviewed: User review status (string, optional)
        
    CSV columns (in order):
        start_timestamp, end_timestamp, duration_sec, max_peak_db, max_rms_db,
        baseline_rms_db, segment_file, segment_offset_sec, clip_file, is_chirp,
        chirp_similarity, confidence, rejection_reason, reviewed
        
    Raises:
        PermissionError: If events file is not writable
        OSError: If directory doesn't exist and can't be created
    """
    ensure_events_header(config)
    events_file = Path(config["event_detection"]["events_file"])
    
    try:
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
    except PermissionError:
        print(f"[ERROR] No permission to write events file: {events_file}")
        print("  Check file permissions and directory access")
        raise
    except OSError as e:
        print(f"[ERROR] Failed to write events file {events_file}: {e}")
        raise
    
    # Print summary to console for real-time monitoring
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


def run_monitor(config_path: Optional[Path] = None, debug: bool = False):
    """
    Run the audio monitor - main event loop.
    
    This is the core monitoring function that:
    1. Captures continuous audio from arecord
    2. Detects events above baseline threshold
    3. Classifies events as chirps or noise
    4. Saves clips and logs events to CSV
    
    Args:
        config_path: Optional path to config.json (defaults to ./config.json)
        debug: If True, enable verbose debug output
        
    State machine:
        IDLE → (RMS > threshold) → IN_EVENT → (RMS <= threshold) → IDLE
    
    Critical state variables (for debugging):
        - baseline_rms_db: Current baseline level (None if not initialized)
        - baseline_window: Rolling window of RMS values for baseline calculation
        - in_event: Boolean flag indicating if we're currently in an event
        - event_chunks: List of raw audio chunks for current event
        - event_actual_start_idx: Index where real event starts (after pre-roll)
        - pre_roll_buffer: Audio buffer before event (for context in clips)
    
    Error recovery:
        - arecord failures: Process is monitored, errors logged, system exits gracefully
        - File I/O errors: Caught in try/except, logged, system continues
        - Invalid audio data: Skipped chunks, baseline calculation handles missing data
    
    Performance notes:
        - Processes 0.5s chunks (2 Hz update rate)
        - Baseline window: 120 chunks (~60s history)
        - FFT operations only during event classification (not every chunk)
    """
    global _config
    
    try:
        _config = config_loader.load_config(config_path)
    except Exception as e:
        print(f"[ERROR] Failed to load configuration: {e}")
        print("  Check that config.json exists and is valid JSON")
        print("  Or create from config.example.json")
        raise
    
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
    
    # Load classifier (ML or fingerprint)
    chirp_cfg = config["chirp_classification"]
    use_ml = chirp_cfg.get("use_ml_classifier", False)
    
    fingerprint_info = None
    ml_model_info = None
    classifier_info = None
    
    if use_ml:
        ml_model_info = load_chirp_ml_model(config)
        if ml_model_info:
            classifier_info = ml_model_info
        else:
            print("[WARN] ML model not found, falling back to fingerprint")
            fingerprint_info = load_chirp_fingerprint(config)
            classifier_info = fingerprint_info
    else:
        fingerprint_info = load_chirp_fingerprint(config)
        classifier_info = fingerprint_info
    
    print("=" * 60)
    print("NOISE DETECTOR - Starting Monitor")
    print("=" * 60)
    print(f"Audio Device: {audio['device']}")
    print(f"Sample Rate: {audio['sample_rate']} Hz")
    print(f"Chunk Duration: {audio['chunk_duration']}s")
    print(f"Output Directory: {Path(recording['output_dir']).resolve()}")
    print(f"Events Log: {Path(event_detection['events_file']).resolve()}")
    print(f"Clips Directory: {Path(event_clips['clips_dir']).resolve()}")
    print(f"Baseline Threshold: +{event_detection['threshold_above_baseline_db']:.1f} dB")
    print(f"Min Event Duration: {event_detection['min_event_duration_sec']:.1f}s")
    if classifier_info:
        if use_ml and ml_model_info:
            print("Chirp Classification: ENABLED (ML Model)")
        elif fingerprint_info:
            print(f"Chirp Classification: ENABLED (Fingerprint, threshold: {chirp_cfg['similarity_threshold']:.2f})")
        else:
            print("Chirp Classification: DISABLED")
    else:
        print("Chirp Classification: DISABLED (no classifier)")
    print("=" * 60)
    print("Press Ctrl+C to stop.\n")
    if classifier_info is None:
        print("[WARN] Chirp classification disabled - no classifier found")
        if use_ml:
            print("  Expected ML model files in data/")
            print("  Run 'make train-ml' to create ML model")
        else:
            print(f"  Expected: {chirp_cfg['fingerprint_file']}")
            print("  Run 'make train' to create fingerprint from training samples")

    try:
        proc = start_arecord(config)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"[ERROR] Failed to start audio capture: {e}")
        print("\nTroubleshooting:")
        print("  1. Check audio device: arecord -l")
        print("  2. Verify device in config.json matches hardware")
        print("  3. Check permissions: groups (should include 'audio')")
        print("  4. Stop other processes using audio: pkill arecord")
        raise

    if proc.stdout is None:
        print("[ERROR] Failed to open arecord stdout pipe.")
        print("  This should not happen - arecord process started but stdout is None")
        print("  Possible subprocess.Popen issue")
        proc.terminate()
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
                # This can happen if:
                # - Device disconnected (USB mic unplugged)
                # - arecord process crashed
                # - System audio subsystem restarted
                # - Permission revoked
                error_msg = ""
                if proc.stderr:
                    err = proc.stderr.read().decode(errors="ignore").strip()
                    if err:
                        error_msg = f"arecord stderr: {err}"
                
                exit_code = proc.poll()
                if exit_code is not None:
                    print(f"\n[ERROR] arecord process exited with code {exit_code}")
                    if error_msg:
                        print(f"  {error_msg}")
                    print("  Possible causes:")
                    print("    - Audio device disconnected")
                    print("    - Device permissions changed")
                    print("    - System audio subsystem issue")
                    print("    - Hardware failure")
                else:
                    print(f"\n[WARN] arecord stream ended unexpectedly (process still running)")
                    if error_msg:
                        print(f"  {error_msg}")
                    print("  This may indicate a partial read or buffer issue")
                
                print("\n[INFO] Stopping monitor due to audio stream interruption")
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
                        # CRITICAL: The ending chunk (below threshold) is never appended to event_chunks,
                        # so the last chunk in event_chunks is the last valid event chunk.
                        # We don't need to exclude anything - event_chunks already contains only valid event audio.
                        if event_chunks is not None and event_actual_start_idx is not None:
                            # Use all chunks from actual event start to the end
                            # event_actual_start_idx marks where the real event begins (after pre-roll)
                            actual_event_chunks = event_chunks[event_actual_start_idx:]
                            
                            if actual_event_chunks:
                                try:
                                    is_chirp, similarity, confidence, rejection_reason = classify_event_is_chirp(
                                        actual_event_chunks, classifier_info, duration_sec, config,
                                        use_ml=use_ml, ml_model_info=ml_model_info
                                    )
                                except Exception as e:
                                    # Classification failed - log error but continue
                                    # This prevents classification bugs from stopping event logging
                                    print(f"[ERROR] Classification failed for event {event_start_time}: {e}")
                                    if debug:
                                        import traceback
                                        traceback.print_exc()
                                    # Continue with is_chirp=False (default)

                        # Compute offset within segment
                        # This tells us where in the 5-minute segment file the event occurred
                        offset_sec = (event_start_offset_samples or 0) / float(sample_rate)
                        
                        # Build event dictionary for logging
                        # All fields must be present for CSV consistency
                        event = {
                            "start_timestamp": event_start_time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "end_timestamp": event_end_time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "duration_sec": duration_sec,
                            "max_peak_db": event_max_peak_db or 0.0,
                            "max_rms_db": event_max_rms_db or 0.0,
                            "baseline_rms_db": event_baseline_at_start,  # Can be None
                            "segment_file": event_segment_file or "",
                            "segment_offset_sec": offset_sec,
                            "is_chirp": is_chirp,
                            "chirp_similarity": similarity,  # Can be None
                            "confidence": confidence,  # Can be None
                            "rejection_reason": rejection_reason or "",
                            "clip_file": str(clip_path) if clip_path else "",
                        }
                        
                        try:
                            log_event(event, config)
                        except (PermissionError, OSError) as e:
                            # Event logging failed - this is critical
                            # Print error but don't crash - we want to keep monitoring
                            print(f"[ERROR] Failed to log event to CSV: {e}")
                            print(f"  Event data: {event_start_time} to {event_end_time}, duration {duration_sec:.2f}s")
                            if debug:
                                import traceback
                                traceback.print_exc()

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

            # Print live metrics (for real-time monitoring)
            # Format: timestamp | peak | rms | baseline | threshold | status
            baseline_str = f"{baseline_rms_db:6.1f}" if baseline_rms_db is not None else "  N/A "
            threshold_str = f"{threshold_db:6.1f}" if baseline_rms_db is not None else "  N/A "
            status = "EVENT" if in_event else "IDLE"
            
            print(
                f"{timestamp_str} | peak: {peak_db:6.1f} dBFS | "
                f"rms: {rms_db:6.1f} dBFS | "
                f"baseline: {baseline_str} dBFS | "
                f"threshold: {threshold_str} dBFS | "
                f"{status}",
                flush=True
            )
            
            # Debug mode: additional verbose output
            if debug and in_event:
                if event_chunks:
                    chunk_count = len(event_chunks)
                    actual_chunks = chunk_count - (event_actual_start_idx or 0)
                    print(f"  [DEBUG] Event chunks: {chunk_count} total, {actual_chunks} actual")

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
        print("\n[INFO] Stopping monitor (Ctrl+C received)...")
    except Exception as e:
        # Catch-all for unexpected errors - log with full context
        print(f"\n[ERROR] Unexpected error in monitor loop: {e}")
        print(f"  Error type: {type(e).__name__}")
        import traceback
        if debug:
            print("  Full traceback:")
            traceback.print_exc()
        else:
            print("  Run with --debug flag for full traceback")
        raise  # Re-raise so caller knows something went wrong

    finally:
        # Clean up WAV file
        # CRITICAL: Always close WAV file properly or it will be corrupted
        if wav_file is not None:
            try:
                wav_file.close()
                if current_path:
                    print(f"[INFO] Closed segment: {current_path}")
            except Exception as e:
                print(f"[ERROR] Failed to close WAV file {current_path}: {e}")
                # Don't raise - we're in cleanup, continue with other cleanup

        # Clean up arecord process
        # CRITICAL: Terminate arecord or it will keep running and block device
        if proc:
            if proc.poll() is None:
                # Process still running - terminate it
                try:
                    proc.terminate()
                    # Give it a moment to terminate gracefully
                    import time
                    time.sleep(0.1)
                    if proc.poll() is None:
                        # Still running - force kill
                        proc.kill()
                except Exception as e:
                    print(f"[WARN] Error terminating arecord process: {e}")
            else:
                # Process already exited - check for errors
                if proc.returncode != 0 and proc.stderr:
                    err = proc.stderr.read().decode(errors="ignore").strip()
                    if err:
                        print(f"[INFO] arecord exit code {proc.returncode}: {err}")
        
        print("[INFO] Monitor stopped and cleaned up.")

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


def load_chirp_ml_model(config: Dict[str, Any]):
    """Load ML model for chirp classification."""
    try:
        import joblib
        from scripts.classify_chirp_ml import load_ml_model
        
        ml_model_info = load_ml_model(config)
        if ml_model_info is None:
            return None
        
        model, scaler, metadata = ml_model_info
        print(f"[INFO] Loaded ML model: {metadata.get('model_type', 'unknown')} "
              f"(train_acc={metadata.get('metrics', {}).get('train_accuracy', 0):.3f})")
        return ml_model_info
    except ImportError:
        print("[WARN] scikit-learn not available - ML classification disabled")
        return None
    except Exception as e:
        print(f"[WARN] Failed to load ML model: {e}")
        return None


def compute_event_spectrum_from_chunks(
    chunks: List[bytes], 
    sample_rate: int, 
    fft_size: int
) -> Optional[np.ndarray]:
    """
    Compute averaged magnitude spectrum from event audio chunks.
    
    This function:
    1. Concatenates all chunks into single audio stream
    2. Removes DC offset (hardware artifact)
    3. Applies Hanning window and FFT with 50% overlap
    4. Averages multiple FFT windows for stability
    5. Normalizes result (L2 norm) for cosine similarity
    
    Args:
        chunks: List of raw PCM byte chunks (int16 little-endian)
        sample_rate: Audio sample rate in Hz
        fft_size: FFT window size (must match fingerprint fft_size)
        
    Returns:
        Normalized magnitude spectrum (1D numpy array) or None if computation fails
        
    Frequency resolution:
        With sample_rate=16000 and fft_size=2048: ~7.8 Hz per bin
        Bin 0 = DC, Bin 1 = 7.8 Hz, Bin 2 = 15.6 Hz, etc.
        Nyquist = sample_rate/2 = 8000 Hz (bin 1024)
    
    Edge cases handled:
        - Empty chunks: Returns None
        - Very short audio: Zero-pads to fft_size
        - Single window: Ensures at least one FFT is computed
    """
    if not chunks:
        return None

    try:
        raw = b"".join(chunks)
        if len(raw) == 0:
            return None
        
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / INT16_FULL_SCALE
        
        # Remove DC offset before spectral analysis
        # DC component can skew frequency analysis, especially on Pi hardware
        # DC appears as energy at 0 Hz, which we don't want in our frequency analysis
        dc_offset = np.mean(samples)
        samples = samples - dc_offset

        # Zero-pad if audio is shorter than FFT window
        # This allows FFT of very short events (though with reduced frequency resolution)
        if samples.shape[0] < fft_size:
            pad = fft_size - samples.shape[0]
            samples = np.pad(samples, (0, pad), mode='constant')
    except Exception as e:
        # If we can't process the audio, return None rather than crashing
        # This can happen with corrupted chunks or invalid data
        return None

    # FFT parameters
    hop = fft_size // 2  # 50% overlap for better time resolution
    window = np.hanning(fft_size)  # Window function to reduce spectral leakage
    specs = []

    # Compute FFT windows with overlap
    # Edge case: if samples exactly equals fft_size, ensure at least one window
    max_start = max(0, len(samples) - fft_size)
    
    if max_start == 0 and len(samples) >= fft_size:
        # Single window case (audio is exactly fft_size samples)
        chunk = samples[0:fft_size] * window
        spec = np.abs(np.fft.rfft(chunk))  # rfft = real FFT (only positive frequencies)
        specs.append(spec)
    else:
        # Multiple windows with 50% overlap
        # Overlap helps capture transients that might fall between windows
        for start in range(0, max_start, hop):
            chunk = samples[start:start + fft_size] * window
            spec = np.abs(np.fft.rfft(chunk))
            specs.append(spec)
    
    if not specs:
        # No valid FFT windows computed - this shouldn't happen but handle it
        return None
    
    # Apply high-pass filter in frequency domain to remove very low-frequency noise
    # This helps with Pi hardware artifacts and room rumble
    # Done after FFT to avoid phase issues with time-domain filtering
    freq_resolution = sample_rate / fft_size  # Hz per bin
    cutoff_bin = max(1, int(20.0 / freq_resolution))  # 20 Hz high-pass
    
    for i, spec in enumerate(specs):
        # Zero out DC (bin 0) and very low frequencies (< 20 Hz)
        # These are typically hardware artifacts, not actual audio content
        spec[:cutoff_bin] = 0.0
        specs[i] = spec

    # Average all FFT windows for stability
    # Averaging reduces variance from individual window artifacts
    avg_spec = np.mean(specs, axis=0)
    
    # Normalize to unit length (L2 norm)
    # This enables cosine similarity calculation (dot product of normalized vectors)
    # The 1e-9 epsilon prevents division by zero for silent audio
    norm = np.linalg.norm(avg_spec)
    if norm < 1e-9:
        # Silent or near-silent audio - return None rather than all-zeros
        return None
    
    avg_spec = avg_spec / norm
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
        
        # Calculate similarity to chirp fingerprint
        sim = float(np.dot(fp, window_spec))
        
        # If non-chirp fingerprint is available, also calculate similarity to it
        # We want HIGH similarity to chirp AND LOW similarity to non-chirp
        non_chirp_sim = None
        if "non_chirp_fingerprint" in fingerprint_info:
            non_chirp_fp = fingerprint_info["non_chirp_fingerprint"]
            non_chirp_sim = float(np.dot(non_chirp_fp, window_spec))
        
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
        
        # If non-chirp fingerprint is available, penalize high similarity to non-chirp
        # This helps select segments that are clearly chirps, not ambiguous
        non_chirp_penalty = 1.0
        if non_chirp_sim is not None:
            # Penalize if similarity to non-chirp is high (sounds like a non-chirp)
            # Scale: 1.0 if non_chirp_sim is 0, decreasing as non_chirp_sim increases
            non_chirp_penalty = max(0.5, 1.0 - non_chirp_sim * 0.5)
        
        combined_score = sim * max(0.0, freq_score) * window_size_factor * non_chirp_penalty
        
        if combined_score > best_score:
            best_score = combined_score
            best_similarity = sim
            best_chunks = window_chunks
    
    if best_chunks is None or best_similarity < 0:
        return None, None, "no_valid_segment"
    
    return best_chunks, best_similarity, None


def classify_event_is_chirp_ml(
    event_chunks: List[bytes],
    ml_model_info: Tuple,
    duration_sec: float,
    config: Dict[str, Any]
) -> Tuple[bool, Optional[float], Optional[float], Optional[str]]:
    """
    Classify event using ML model.
    
    Args:
        event_chunks: List of raw PCM byte chunks
        ml_model_info: Tuple from load_chirp_ml_model()
        duration_sec: Event duration
        config: Configuration dictionary
        
    Returns:
        Tuple of (is_chirp, confidence, confidence, None)
    """
    try:
        from scripts.classify_chirp_ml import classify_event_chunks_ml
        
        sr = config["audio"]["sample_rate"]
        is_chirp, confidence, error = classify_event_chunks_ml(event_chunks, sr, ml_model_info)
        
        if error:
            return False, None, None, f"ml_error_{error}"
        
        # For ML, we use confidence as both similarity and confidence
        # (ML gives probability, not similarity)
        return is_chirp, confidence, confidence, None
        
    except Exception as e:
        return False, None, None, f"ml_exception_{str(e)}"


def classify_event_is_chirp(
    event_chunks: List[bytes], 
    classifier_info: Optional[Dict[str, Any]], 
    duration_sec: float,
    config: Dict[str, Any],
    use_ml: bool = False,
    ml_model_info: Optional[Tuple] = None
) -> Tuple[bool, Optional[float], Optional[float], Optional[str]]:
    """
    Classify if event is a chirp using either ML model or fingerprint method.
    
    This is the core classification function. It can use either:
    - ML model: Trained classifier (Random Forest or SVM)
    - Fingerprint: Multi-stage filtering with spectral similarity
    
    Args:
        event_chunks: List of raw PCM byte chunks (int16 little-endian)
        classifier_info: Dictionary with fingerprint info (for fingerprint method) or None
        duration_sec: Total event duration in seconds
        config: Configuration dictionary
        use_ml: If True, use ML model instead of fingerprint
        ml_model_info: ML model info tuple (if use_ml=True)
        
    Returns:
        Tuple of:
            - is_chirp (bool): True if classified as chirp
            - similarity (float or None): Similarity/confidence score (0-1)
            - confidence (float or None): Confidence score (0-1)
            - rejection_reason (str or None): Why it was rejected (if not chirp)
    """
    # Use ML model if requested and available
    if use_ml and ml_model_info is not None:
        return classify_event_is_chirp_ml(event_chunks, ml_model_info, duration_sec, config)
    
    # Fall back to fingerprint method
    fingerprint_info = classifier_info
    if fingerprint_info is None:
        return False, None, None, "no_classifier"
    
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

