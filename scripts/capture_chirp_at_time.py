#!/usr/bin/env python3
"""
Retroactively capture and analyze chirps from a specific timestamp.

This script allows you to manually capture a chirp from a specific timestamp (in USA East Coast timezone).
It extracts 3 minutes of audio centered on the specified time and creates a clip, bypassing event detection
(since you manually confirmed the chirp was there).

Usage:
    python3 scripts/capture_chirp_at_time.py "2025-01-15 14:30"
    python3 scripts/capture_chirp_at_time.py "2025-01-15T14:30:00"
    
Note: The --force-chirp flag is now the default behavior (always marks as chirp for manual captures).
"""
import sys
import wave
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
import numpy as np
import pytz

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config_loader
from core.repository import EventRepository
import monitor  # For dbfs function


# USA East Coast timezone (handles EST/EDT automatically)
EASTERN = pytz.timezone('US/Eastern')


def parse_timestamp(time_str: str) -> datetime:
    """
    Parse timestamp string in USA East Coast timezone.
    
    Accepts formats like:
    - "2025-01-15 14:30"
    - "2025-01-15T14:30:00"
    - "2025-01-15 14:30:00"
    """
    # Try different formats
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M",
    ]
    
    dt = None
    for fmt in formats:
        try:
            dt = datetime.strptime(time_str, fmt)
            break
        except ValueError:
            continue
    
    if dt is None:
        raise ValueError(f"Could not parse timestamp: {time_str}. Expected format like '2025-01-15 14:30'")
    
    # Localize to Eastern timezone
    dt = EASTERN.localize(dt)
    
    return dt


def find_segment_file(target_time: datetime, segments_dir: Path) -> Optional[Path]:
    """
    Find the segment file that contains the target timestamp.
    
    Segments are 300 seconds (5 minutes) long and named like "2025-01-15_14-00-00.wav"
    """
    if not segments_dir.exists():
        return None
    
    # Segments are 300 seconds, so we need to check which segment contains our target
    # Segment files are named with their start time
    segment_duration = 300  # seconds
    
    # Find all segment files
    segment_files = sorted(segments_dir.glob("*.wav"))
    
    for seg_file in segment_files:
        # Parse segment start time from filename
        # Format: "2025-01-15_14-00-00.wav"
        try:
            name = seg_file.stem  # "2025-01-15_14-00-00"
            seg_start = datetime.strptime(name, "%Y-%m-%d_%H-%M-%S")
            # Assume segment times are in local time (same as target_time)
            seg_start = EASTERN.localize(seg_start)
            seg_end = seg_start + timedelta(seconds=segment_duration)
            
            # Check if target_time falls within this segment
            if seg_start <= target_time < seg_end:
                return seg_file
        except ValueError:
            # Skip files that don't match the naming pattern
            continue
    
    return None


def extract_window_from_segment(
    segment_path: Path,
    target_time: datetime,
    config: dict,
    window_duration_sec: float = 180.0
) -> Optional[Tuple[np.ndarray, int, datetime]]:
    """
    Extract audio window (default 3 minutes) centered on target_time from segment file.
    
    Args:
        segment_path: Path to segment WAV file
        target_time: Target timestamp (center of window)
        config: Configuration dictionary
        window_duration_sec: Total duration to extract in seconds (default 180 = 3 minutes)
    
    Returns:
        (samples, sample_rate, actual_start_time) or None if extraction fails
    """
    try:
        with wave.open(str(segment_path), "rb") as wf:
            sr = wf.getframerate()
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            
            # Parse segment start time from filename
            seg_name = segment_path.stem
            seg_start = datetime.strptime(seg_name, "%Y-%m-%d_%H-%M-%S")
            seg_start = EASTERN.localize(seg_start)
            seg_end = seg_start + timedelta(seconds=300)  # Segments are 300 seconds
            
            # Calculate window: centered on target_time
            half_window = window_duration_sec / 2.0
            window_start = target_time - timedelta(seconds=half_window)
            window_end = target_time + timedelta(seconds=half_window)
            
            # Clamp to segment boundaries
            actual_start = max(window_start, seg_start)
            actual_end = min(window_end, seg_end)
            
            # Calculate offsets within segment
            start_offset_sec = (actual_start - seg_start).total_seconds()
            end_offset_sec = (actual_end - seg_start).total_seconds()
            actual_duration_sec = (actual_end - actual_start).total_seconds()
            
            if start_offset_sec < 0 or start_offset_sec >= 300:
                print(f"[ERROR] Target time {target_time} is outside segment range")
                return None
            
            if actual_duration_sec <= 0:
                print(f"[ERROR] Invalid duration: {actual_duration_sec} seconds")
                return None
            
            # Seek to start position
            wf.setpos(int(start_offset_sec * sr))
            
            # Read the window
            num_samples = int(sr * actual_duration_sec)
            frames = wf.readframes(num_samples)
            
            if len(frames) == 0:
                print(f"[ERROR] No audio data read from segment")
                return None
            
            # Convert to numpy array
            if sample_width == 2:  # int16
                samples = np.frombuffer(frames, dtype="<i2")
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")
            
            # Convert to mono if needed
            if channels > 1:
                samples = samples.reshape(-1, channels).mean(axis=1)
            
            # Normalize to float32 [-1.0, 1.0]
            samples = samples.astype(np.float32) / 32768.0
            
            # Remove DC offset
            samples = samples - np.mean(samples)
            
            print(f"[INFO] Extracted {actual_duration_sec:.1f} seconds: {actual_start.strftime('%H:%M:%S')} to {actual_end.strftime('%H:%M:%S')}")
            
            return samples, sr, actual_start
            
    except Exception as e:
        print(f"[ERROR] Failed to extract audio from segment: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_clip_from_audio(
    samples: np.ndarray,
    sample_rate: int,
    start_time: datetime,
    config: dict,
    target_time: datetime
) -> Optional[dict]:
    """
    Create a clip directly from audio samples, bypassing event detection.
    
    Since the user manually specified this time, we trust them and create the clip
    regardless of whether the detector would have triggered.
    
    Returns:
        Event record dictionary ready to save to events.csv
    """
    import wave
    import monitor
    
    # Calculate audio statistics for the record
    rms = float(np.sqrt(np.mean(samples ** 2)))
    peak = float(np.max(np.abs(samples)))
    rms_db = monitor.dbfs(rms)
    peak_db = monitor.dbfs(peak)
    
    # Convert samples back to int16 bytes for saving
    samples_int16 = (samples * 32768.0).astype(np.int16)
    audio_bytes = samples_int16.tobytes()
    
    # Calculate duration
    duration_sec = len(samples) / float(sample_rate)
    end_time = start_time + timedelta(seconds=duration_sec)
    
    # Save clip
    clips_dir = Path(config["event_clips"]["clips_dir"])
    clips_dir.mkdir(parents=True, exist_ok=True)
    
    # Use target_time for filename (the time user specified, not actual start)
    clip_filename = target_time.strftime("clip_%Y-%m-%d_%H-%M-%S.wav")
    clip_path = clips_dir / clip_filename
    
    # Write WAV file
    channels = config["audio"]["channels"]
    with wave.open(str(clip_path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)
    
    print(f"[CLIP] Saved clip: {clip_path.name} ({duration_sec:.1f}s)")
    
    # Build event record
    # Use target_time as the event timestamp (what user specified)
    event_record = {
        "start_timestamp": target_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "end_timestamp": (target_time + timedelta(seconds=duration_sec)).strftime("%Y-%m-%dT%H:%M:%S"),
        "duration_sec": duration_sec,
        "max_peak_db": peak_db,
        "max_rms_db": rms_db,
        "baseline_rms_db": None,  # Not applicable for manual capture
        "segment_file": "",  # Not from live segment
        "segment_offset_sec": 0.0,
        "is_chirp": True,  # Always mark as chirp for manual captures
        "chirp_similarity": None,
        "confidence": 1.0,  # High confidence - user confirmed it
        "rejection_reason": "",
        "clip_file": str(clip_path),
    }
    
    return event_record


def capture_chirp_at_time(
    time_str: str,
    config_path: Optional[Path] = None,
    force_chirp: bool = False
) -> bool:
    """
    Main function to capture chirp from a specific timestamp.
    
    Args:
        time_str: Timestamp string in USA East Coast timezone
        config_path: Path to config.json (optional)
        force_chirp: If True, mark all events as chirps regardless of classification
    
    Returns:
        True if successful, False otherwise
    """
    # Load config
    config = config_loader.load_config(config_path)
    
    # Parse timestamp
    try:
        target_time = parse_timestamp(time_str)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return False
    
    print(f"[INFO] Target time: {target_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    # Find segment file
    segments_dir = Path(config["recording"]["output_dir"])
    if not segments_dir.is_absolute():
        # Try to find project root
        if config_path:
            project_root = config_path.parent
        else:
            project_root = Path.cwd()
        segments_dir = project_root / segments_dir
    
    segment_file = find_segment_file(target_time, segments_dir)
    
    if segment_file is None:
        print(f"[ERROR] Could not find segment file containing {target_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Searched in: {segments_dir}")
        print(f"  Make sure segments are being recorded and check the directory")
        return False
    
    print(f"[INFO] Found segment: {segment_file.name}")
    
    # Extract 3 minutes (180 seconds) centered on target time
    result = extract_window_from_segment(segment_file, target_time, config, window_duration_sec=180.0)
    if result is None:
        return False
    
    samples, sample_rate, actual_start = result
    
    # Create clip directly (bypass detection - user manually specified this time)
    event_record = create_clip_from_audio(
        samples,
        sample_rate,
        actual_start,
        config,
        target_time  # Use target_time for event timestamp
    )
    
    if event_record is None:
        print(f"[ERROR] Failed to create clip")
        return False
    
    # Save to events.csv
    event_repo = EventRepository(config)
    try:
        event_repo.save(event_record)
        print(f"[EVENT] Saved event: {event_record['start_timestamp']} - CHIRP (manual capture)")
    except Exception as e:
        print(f"[ERROR] Failed to save event: {e}")
        return False
    
    print(f"\n[SUCCESS] Created clip and saved event")
    print(f"  Clip: {event_record['clip_file']}")
    print(f"  Duration: {event_record['duration_sec']:.1f} seconds")
    print(f"  Events saved to: {config['event_detection']['events_file']}")
    print(f"  Clips saved to: {config['event_clips']['clips_dir']}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Retroactively capture chirps from a specific timestamp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Capture chirp from a specific minute (USA East Coast time)
  python3 scripts/capture_chirp_at_time.py "2025-01-15 14:30"
  
  # Force mark as chirp even if classifier doesn't detect it
  python3 scripts/capture_chirp_at_time.py "2025-01-15 14:30" --force-chirp
  
  # With explicit config file
  python3 scripts/capture_chirp_at_time.py "2025-01-15 14:30" --config config.json
        """
    )
    parser.add_argument("timestamp", help="Timestamp in USA East Coast timezone (e.g., '2025-01-15 14:30')")
    parser.add_argument("--config", type=Path, help="Path to config.json")
    # Note: force_chirp is now always True for manual captures, but keep flag for backward compatibility
    parser.add_argument("--force-chirp", action="store_true", default=True,
                       help="Always mark as chirp for manual captures (default: True)")
    
    args = parser.parse_args()
    
    success = capture_chirp_at_time(
        args.timestamp,
        config_path=args.config,
        force_chirp=True  # Always True for manual captures
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

