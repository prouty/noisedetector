#!/usr/bin/env python3
"""
Refactored monitor using SOLID principles.

This is a cleaner implementation that will eventually replace monitor.py.
"""
import datetime
from pathlib import Path
from typing import Optional

import config_loader
from core import (
    AudioCapture,
    BaselineTracker,
    EventDetector,
    EventRepository,
    SegmentRepository,
    create_classifier
)
import monitor as monitor_legacy  # For dbfs and classification functions


def run_monitor(config_path: Optional[Path] = None, debug: bool = False) -> None:
    """
    Run the audio monitor - main event loop (refactored version).
    
    This version uses dependency injection and follows SOLID principles:
    - Single Responsibility: Each class has one clear purpose
    - Dependency Inversion: Depends on abstractions (interfaces)
    - Open/Closed: Can extend with new classifiers without modifying this code
    
    Args:
        config_path: Optional path to config.json (defaults to ./config.json)
        debug: If True, enable verbose debug output
    """
    # Load configuration
    try:
        config = config_loader.load_config(config_path)
    except Exception as e:
        print(f"[ERROR] Failed to load configuration: {e}")
        print("  Check that config.json exists and is valid JSON")
        print("  Or create from config.example.json")
        raise
    
    # Initialize components (Dependency Injection)
    audio_capture = AudioCapture(config)
    baseline_tracker = BaselineTracker(config)
    event_detector = EventDetector(config, baseline_tracker)
    event_repo = EventRepository(config)
    segment_repo = SegmentRepository(config)
    classifier = create_classifier(config)
    
    # Print startup information
    _print_startup_info(config, classifier)
    
    # Start audio capture
    try:
        audio_capture.start()
    except Exception as e:
        print(f"[ERROR] Failed to start audio capture: {e}")
        print("\nTroubleshooting:")
        print("  1. Check audio device: arecord -l")
        print("  2. Verify device in config.json matches hardware")
        print("  3. Check permissions: groups (should include 'audio')")
        print("  4. Stop other processes using audio: pkill arecord")
        raise
    
    # Start segment recording
    segment_repo.start_segment(datetime.datetime.now())
    
    try:
        # Main event loop
        while audio_capture.is_running():
            chunk = audio_capture.read_chunk()
            
            if chunk is None:
                # Stream ended
                print("\n[INFO] Audio stream ended")
                break
            
            # Use raw bytes from chunk (preserves original PCM data)
            chunk_bytes = chunk.raw_bytes
            
            # Write to segment
            segment_repo.write_chunk(
                chunk_bytes,
                len(chunk.samples)
            )
            
            # Detect events
            event = event_detector.process_chunk(chunk, chunk_bytes)
            
            if event:
                # Event detected - classify and save
                _process_event(
                    event,
                    classifier,
                    config,
                    event_repo,
                    segment_repo,
                    debug
                )
            
            # Print live metrics
            _print_metrics(
                chunk,
                baseline_tracker,
                event_detector.in_event
            )
    
    except KeyboardInterrupt:
        print("\n[INFO] Stopping monitor (Ctrl+C received)...")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error in monitor loop: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        raise
    finally:
        # Cleanup
        audio_capture.stop()
        segment_repo.close_segment()
        print("[INFO] Monitor stopped and cleaned up.")


def _print_startup_info(config: dict, classifier) -> None:
    """Print startup information."""
    audio = config["audio"]
    recording = config["recording"]
    event_detection = config["event_detection"]
    event_clips = config["event_clips"]
    
    print("=" * 60)
    print("NOISE DETECTOR - Starting Monitor (Refactored)")
    print("=" * 60)
    print(f"Audio Device: {audio['device']}")
    print(f"Sample Rate: {audio['sample_rate']} Hz")
    print(f"Chunk Duration: {audio['chunk_duration']}s")
    print(f"Output Directory: {Path(recording['output_dir']).resolve()}")
    print(f"Events Log: {Path(event_detection['events_file']).resolve()}")
    print(f"Clips Directory: {Path(event_clips['clips_dir']).resolve()}")
    print(f"Baseline Threshold: +{event_detection['threshold_above_baseline_db']:.1f} dB")
    print(f"Min Event Duration: {event_detection['min_event_duration_sec']:.1f}s")
    
    if classifier:
        classifier_type = "ML Model" if config["chirp_classification"].get("use_ml_classifier") else "Fingerprint"
        print(f"Chirp Classification: ENABLED ({classifier_type})")
    else:
        print("Chirp Classification: DISABLED (no classifier)")
    
    print("=" * 60)
    print("Press Ctrl+C to stop.\n")


def _process_event(
    event,
    classifier,
    config: dict,
    event_repo: EventRepository,
    segment_repo: SegmentRepository,
    debug: bool
) -> None:
    """Process a detected event: classify, save clip, log."""
    # Save clip
    clip_path = segment_repo.save_clip(event.start_time, event.chunks)
    
    # Classify event
    is_chirp = False
    similarity = None
    confidence = None
    rejection_reason = None
    
    if classifier:
        # Use actual event chunks (exclude pre-roll)
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
                print(f"[ERROR] Classification failed for event {event.start_time}: {e}")
                if debug:
                    import traceback
                    traceback.print_exc()
    
    # Get segment info
    segment_file = segment_repo.get_current_segment_path()
    segment_offset = segment_repo.get_current_segment_offset()
    
    # Build event record
    event_record = {
        "start_timestamp": event.start_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "end_timestamp": event.end_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "duration_sec": (event.end_time - event.start_time).total_seconds(),
        "max_peak_db": event.max_peak_db,
        "max_rms_db": event.max_rms_db,
        "baseline_rms_db": event.baseline_rms_db,
        "segment_file": str(segment_file) if segment_file else "",
        "segment_offset_sec": segment_offset,
        "is_chirp": is_chirp,
        "chirp_similarity": similarity,
        "confidence": confidence,
        "rejection_reason": rejection_reason or "",
        "clip_file": str(clip_path),
    }
    
    # Save to repository
    try:
        event_repo.save(event_record)
    except Exception as e:
        print(f"[ERROR] Failed to log event: {e}")


def _print_metrics(chunk, baseline_tracker: BaselineTracker, in_event: bool) -> None:
    """Print live monitoring metrics."""
    peak_db = monitor_legacy.dbfs(chunk.peak)
    rms_db = monitor_legacy.dbfs(chunk.rms)
    baseline_rms_db = baseline_tracker.baseline_rms_db
    threshold_db = baseline_tracker.get_threshold_db()
    
    baseline_str = f"{baseline_rms_db:6.1f}" if baseline_rms_db is not None else "  N/A "
    threshold_str = f"{threshold_db:6.1f}" if baseline_rms_db is not None else "  N/A "
    status = "EVENT" if in_event else "IDLE"
    
    timestamp_str = datetime.datetime.fromtimestamp(chunk.timestamp).strftime("%Y-%m-%dT%H:%M:%S")
    
    print(
        f"{timestamp_str} | peak: {peak_db:6.1f} dBFS | "
        f"rms: {rms_db:6.1f} dBFS | "
        f"baseline: {baseline_str} dBFS | "
        f"threshold: {threshold_str} dBFS | "
        f"{status}",
        flush=True
    )

