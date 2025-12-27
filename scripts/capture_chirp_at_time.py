#!/usr/bin/env python3
"""
Retroactively capture and analyze chirps from a specific timestamp.

This script allows you to mark a specific minute (in USA East Coast timezone) as containing
a chirp, then automatically extracts that minute from the recorded segments, analyzes it,
and creates events/clips as if it had been detected in real-time.

Usage:
    python3 scripts/capture_chirp_at_time.py "2025-01-15 14:30"
    python3 scripts/capture_chirp_at_time.py "2025-01-15T14:30:00"
    python3 scripts/capture_chirp_at_time.py "2025-01-15 14:30" --force-chirp  # Force mark as chirp even if classifier says no
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
from core.repository import EventRepository, SegmentRepository
from core.detector import EventDetector, Event
from core.baseline import BaselineTracker
from core.classifier import create_classifier, Classifier
from core.audio import AudioChunk
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


def extract_minute_from_segment(
    segment_path: Path,
    target_time: datetime,
    config: dict
) -> Optional[Tuple[np.ndarray, int, datetime]]:
    """
    Extract 60 seconds of audio from segment file starting at target_time.
    
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
            
            # Calculate offset within segment
            offset_sec = (target_time - seg_start).total_seconds()
            if offset_sec < 0 or offset_sec >= 300:  # Segment is 300 seconds
                print(f"[ERROR] Target time {target_time} is outside segment range")
                return None
            
            # Calculate byte position
            bytes_per_sample = sample_width * channels
            offset_bytes = int(offset_sec * sr * bytes_per_sample)
            
            # Seek to position
            wf.setpos(int(offset_sec * sr))
            
            # Read 60 seconds
            duration_sec = 60.0
            num_samples = int(sr * duration_sec)
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
            
            return samples, sr, target_time
            
    except Exception as e:
        print(f"[ERROR] Failed to extract audio from segment: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_audio_minute(
    samples: np.ndarray,
    sample_rate: int,
    start_time: datetime,
    config: dict,
    classifier: Optional[Classifier],
    force_chirp: bool = False
) -> List[dict]:
    """
    Process 60 seconds of audio through event detection and classification.
    
    Returns:
        List of event records (ready to save to events.csv)
    """
    chunk_duration = config["audio"]["chunk_duration"]
    chunk_samples = int(sample_rate * chunk_duration)
    bytes_per_sample = 2  # int16
    chunk_bytes = chunk_samples * bytes_per_sample
    
    # Initialize components
    baseline_tracker = BaselineTracker(config)
    event_detector = EventDetector(config, baseline_tracker)
    event_repo = EventRepository(config)
    segment_repo = SegmentRepository(config)
    
    events_found = []
    current_time = start_time
    
    # Process audio in chunks
    num_chunks = len(samples) // chunk_samples
    
    print(f"[INFO] Processing {num_chunks} chunks from {start_time.strftime('%Y-%m-%d %H:%M:%S')} EST")
    
    for i in range(num_chunks):
        chunk_start = i * chunk_samples
        chunk_end = chunk_start + chunk_samples
        chunk_samples_data = samples[chunk_start:chunk_end]
        
        # Convert to int16 bytes for processing
        chunk_int16 = (chunk_samples_data * 32768.0).astype(np.int16)
        chunk_bytes_data = chunk_int16.tobytes()
        
        # Create AudioChunk
        chunk_timestamp = (start_time + timedelta(seconds=i * chunk_duration)).timestamp()
        chunk = AudioChunk(
            samples=chunk_samples_data,
            raw_bytes=chunk_bytes_data,
            sample_rate=sample_rate,
            timestamp=chunk_timestamp
        )
        
        # Process through event detector
        event = event_detector.process_chunk(chunk, chunk_bytes_data)
        
        if event:
            # Process event (classify and save)
            clip_path = segment_repo.save_clip(event.start_time, event.chunks)
            
            # Classify
            is_chirp = force_chirp  # If force_chirp, skip classification
            similarity = None
            confidence = None
            rejection_reason = None
            
            if classifier and not force_chirp:
                actual_chunks = event.chunks[event.actual_start_idx:]
                duration_sec = (event.end_time - event.start_time).total_seconds()
                
                if actual_chunks:
                    try:
                        is_chirp, similarity, confidence, rejection_reason = classifier.classify(
                            actual_chunks,
                            duration_sec,
                            config
                        )
                    except Exception as e:
                        print(f"[WARN] Classification failed: {e}")
            
            if force_chirp:
                is_chirp = True
                confidence = 1.0  # High confidence when forced
            
            # Build event record
            event_record = {
                "start_timestamp": event.start_time.strftime("%Y-%m-%dT%H:%M:%S"),
                "end_timestamp": event.end_time.strftime("%Y-%m-%dT%H:%M:%S"),
                "duration_sec": (event.end_time - event.start_time).total_seconds(),
                "max_peak_db": event.max_peak_db,
                "max_rms_db": event.max_rms_db,
                "baseline_rms_db": event.baseline_rms_db,
                "segment_file": "",  # Not from live segment
                "segment_offset_sec": 0.0,
                "is_chirp": is_chirp,
                "chirp_similarity": similarity,
                "confidence": confidence,
                "rejection_reason": rejection_reason or "",
                "clip_file": str(clip_path),
            }
            
            events_found.append(event_record)
            
            # Save to events.csv
            try:
                event_repo.save(event_record)
                print(f"[EVENT] Saved event: {event_record['start_timestamp']} - {'CHIRP' if is_chirp else 'noise'}")
            except Exception as e:
                print(f"[ERROR] Failed to save event: {e}")
    
    return events_found


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
    
    # Extract minute of audio
    result = extract_minute_from_segment(segment_file, target_time, config)
    if result is None:
        return False
    
    samples, sample_rate, actual_start = result
    print(f"[INFO] Extracted {len(samples)/sample_rate:.1f} seconds of audio")
    
    # Load classifier (factory function returns appropriate type)
    classifier = create_classifier(config)
    
    # Process audio
    events = process_audio_minute(
        samples,
        sample_rate,
        actual_start,
        config,
        classifier,
        force_chirp=force_chirp
    )
    
    if not events:
        print(f"[WARN] No events detected in the specified minute")
        if force_chirp:
            print(f"[INFO] Use --force-chirp to create a manual event even if nothing is detected")
        return False
    
    chirps = [e for e in events if e.get("is_chirp")]
    print(f"\n[SUCCESS] Processed {len(events)} event(s), {len(chirps)} chirp(s)")
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
    parser.add_argument("--force-chirp", action="store_true", 
                       help="Force mark all detected events as chirps (skip classification)")
    
    args = parser.parse_args()
    
    success = capture_chirp_at_time(
        args.timestamp,
        config_path=args.config,
        force_chirp=args.force_chirp
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

