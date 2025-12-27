#!/usr/bin/env python3
"""
Extract a specific segment from an existing clip file.

After manually capturing a 3-minute clip, use this script to extract just the
chirp portion (with padding before/after) into a shorter, shareable clip.

Usage:
    # Extract seconds 45-55 from a clip (with 2s padding = 43-57)
    python3 scripts/extract_chirp_segment.py clips/clip_2025-12-27_14-30-00.wav 45 55
    
    # Custom padding (5 seconds before/after)
    python3 scripts/extract_chirp_segment.py clips/clip_2025-12-27_14-30-00.wav 45 55 --padding 5
    
    # Update events.csv to point to the new clip
    python3 scripts/extract_chirp_segment.py clips/clip_2025-12-27_14-30-00.wav 45 55 --update-events
"""
import sys
import wave
import argparse
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config_loader


def extract_segment_from_clip(
    clip_path: Path,
    start_sec: float,
    end_sec: float,
    padding_sec: float = 2.0,
    output_path: Optional[Path] = None
) -> Optional[Path]:
    """
    Extract a segment from a clip file.
    
    Args:
        clip_path: Path to source clip file
        start_sec: Start time in seconds (from beginning of clip)
        end_sec: End time in seconds (from beginning of clip)
        padding_sec: Seconds to add before start and after end
        output_path: Optional output path (default: adds _segment suffix)
    
    Returns:
        Path to new clip file, or None if extraction fails
    """
    if not clip_path.exists():
        print(f"[ERROR] Clip file not found: {clip_path}")
        return None
    
    if start_sec < 0 or end_sec <= start_sec:
        print(f"[ERROR] Invalid time range: {start_sec} to {end_sec}")
        return None
    
    try:
        with wave.open(str(clip_path), "rb") as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            total_frames = wf.getnframes()
            total_duration = total_frames / float(sample_rate)
            
            # Validate time range
            if end_sec > total_duration:
                print(f"[WARN] End time {end_sec}s exceeds clip duration {total_duration:.1f}s, using clip end")
                end_sec = total_duration
            
            # Apply padding
            padded_start = max(0.0, start_sec - padding_sec)
            padded_end = min(total_duration, end_sec + padding_sec)
            
            # Calculate frame positions
            start_frame = int(padded_start * sample_rate)
            end_frame = int(padded_end * sample_rate)
            num_frames = end_frame - start_frame
            
            # Seek to start position
            wf.setpos(start_frame)
            
            # Read frames
            frames = wf.readframes(num_frames)
            
            if len(frames) == 0:
                print(f"[ERROR] No audio data read")
                return None
            
            # Determine output path
            if output_path is None:
                # Create new filename: clip_2025-12-27_14-30-00.wav -> clip_2025-12-27_14-30-00_segment.wav
                stem = clip_path.stem
                output_path = clip_path.parent / f"{stem}_segment.wav"
            
            # Write new clip
            with wave.open(str(output_path), "wb") as out_wf:
                out_wf.setnchannels(channels)
                out_wf.setsampwidth(sample_width)
                out_wf.setframerate(sample_rate)
                out_wf.writeframes(frames)
            
            actual_duration = num_frames / float(sample_rate)
            print(f"[SUCCESS] Extracted segment: {padded_start:.1f}s - {padded_end:.1f}s ({actual_duration:.1f}s)")
            print(f"  Source: {clip_path.name}")
            print(f"  Output: {output_path.name}")
            
            return output_path
            
    except Exception as e:
        print(f"[ERROR] Failed to extract segment: {e}")
        import traceback
        traceback.print_exc()
        return None


def update_events_csv(
    clip_path: Path,
    new_clip_path: Path,
    config_path: Optional[Path] = None
) -> bool:
    """
    Update events.csv to point to the new segment clip instead of the original.
    
    Args:
        clip_path: Original clip path (to find in events.csv)
        new_clip_path: New segment clip path
        config_path: Path to config.json
    
    Returns:
        True if successful, False otherwise
    """
    try:
        import pandas as pd
        
        config = config_loader.load_config(config_path)
        events_file = Path(config["event_detection"]["events_file"])
        
        if not events_file.exists():
            print(f"[WARN] Events file not found: {events_file}")
            return False
        
        # Load events
        df = pd.read_csv(events_file)
        
        if "clip_file" not in df.columns:
            print(f"[WARN] No 'clip_file' column in events.csv")
            return False
        
        # Find rows matching the original clip
        clip_filename = clip_path.name
        mask = df["clip_file"].astype(str).str.endswith(clip_filename, na=False)
        
        if not mask.any():
            print(f"[WARN] No events found for clip: {clip_filename}")
            return False
        
        # Update clip_file column
        num_updated = mask.sum()
        df.loc[mask, "clip_file"] = str(new_clip_path)
        
        # Save updated CSV
        df.to_csv(events_file, index=False)
        
        print(f"[SUCCESS] Updated {num_updated} event(s) in events.csv")
        print(f"  Changed clip_file to: {new_clip_path.name}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to update events.csv: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Extract a specific segment from a clip file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract seconds 45-55 with default 2s padding (result: 43-57s)
  python3 scripts/extract_chirp_segment.py clips/clip_2025-12-27_14-30-00.wav 45 55
  
  # Custom padding (5 seconds)
  python3 scripts/extract_chirp_segment.py clips/clip_2025-12-27_14-30-00.wav 45 55 --padding 5
  
  # Update events.csv to point to new clip
  python3 scripts/extract_chirp_segment.py clips/clip_2025-12-27_14-30-00.wav 45 55 --update-events
  
  # Specify custom output filename
  python3 scripts/extract_chirp_segment.py clips/clip_2025-12-27_14-30-00.wav 45 55 \\
    --output clips/chirp_clean.wav
        """
    )
    parser.add_argument("clip", type=Path, help="Path to source clip file")
    parser.add_argument("start_sec", type=float, help="Start time in seconds (from beginning of clip)")
    parser.add_argument("end_sec", type=float, help="End time in seconds (from beginning of clip)")
    parser.add_argument("--padding", type=float, default=2.0,
                       help="Seconds to add before start and after end (default: 2.0)")
    parser.add_argument("--output", type=Path, help="Output clip path (default: adds _segment suffix)")
    parser.add_argument("--update-events", action="store_true",
                       help="Update events.csv to point to new clip")
    parser.add_argument("--config", type=Path, help="Path to config.json")
    
    args = parser.parse_args()
    
    # Extract segment
    new_clip_path = extract_segment_from_clip(
        args.clip,
        args.start_sec,
        args.end_sec,
        padding_sec=args.padding,
        output_path=args.output
    )
    
    if new_clip_path is None:
        sys.exit(1)
    
    # Update events.csv if requested
    if args.update_events:
        update_events_csv(args.clip, new_clip_path, args.config)
    
    print(f"\n[SUCCESS] Segment extracted: {new_clip_path}")
    sys.exit(0)


if __name__ == "__main__":
    main()

