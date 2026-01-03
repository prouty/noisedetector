#!/usr/bin/env python3
"""
Extract a specific segment from an existing clip file.

After manually capturing a 3-minute clip, use this script to extract just the
chirp portion (with padding before/after) into a shorter, shareable clip.

Usage:
    # Extract segment at 50% of clip duration (5s before/after that point)
    python3 scripts/extract_chirp_segment.py clips/clip_2025-12-27_14-30-00.wav --percent 50
    
    # Prompt for percentage if not provided
    python3 scripts/extract_chirp_segment.py clips/clip_2025-12-27_14-30-00.wav
    
    # Update events.csv to point to the new clip
    python3 scripts/extract_chirp_segment.py clips/clip_2025-12-27_14-30-00.wav --percent 50 --update-events
"""
import sys
import wave
import argparse
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config_loader


def get_clip_duration(clip_path: Path) -> Optional[float]:
    """
    Get the duration of a clip file in seconds.
    
    Args:
        clip_path: Path to clip file
    
    Returns:
        Duration in seconds, or None if error
    """
    if not clip_path.exists():
        print(f"[ERROR] Clip file not found: {clip_path}")
        return None
    
    try:
        with wave.open(str(clip_path), "rb") as wf:
            sample_rate = wf.getframerate()
            total_frames = wf.getnframes()
            duration = total_frames / float(sample_rate)
            return duration
    except Exception as e:
        print(f"[ERROR] Failed to read clip: {e}")
        return None


def extract_segment_from_clip(
    clip_path: Path,
    center_sec: float,
    padding_sec: float = 5.0,
    output_path: Optional[Path] = None
) -> Optional[Path]:
    """
    Extract a segment from a clip file centered at a specific point.
    
    Args:
        clip_path: Path to source clip file
        center_sec: Center point in seconds (from beginning of clip)
        padding_sec: Seconds to extract before and after center point (default: 5.0)
        output_path: Optional output path (default: adds _segment suffix)
    
    Returns:
        Path to new clip file, or None if extraction fails
    """
    if not clip_path.exists():
        print(f"[ERROR] Clip file not found: {clip_path}")
        return None
    
    if center_sec < 0:
        print(f"[ERROR] Invalid center point: {center_sec}")
        return None
    
    try:
        with wave.open(str(clip_path), "rb") as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            total_frames = wf.getnframes()
            total_duration = total_frames / float(sample_rate)
            
            # Calculate start and end times
            start_sec = max(0.0, center_sec - padding_sec)
            end_sec = min(total_duration, center_sec + padding_sec)
            
            # Calculate frame positions
            start_frame = int(start_sec * sample_rate)
            end_frame = int(end_sec * sample_rate)
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
            print(f"[SUCCESS] Extracted segment: {start_sec:.1f}s - {end_sec:.1f}s ({actual_duration:.1f}s)")
            print(f"  Center point: {center_sec:.1f}s ({center_sec/total_duration*100:.1f}% of clip)")
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
        description="Extract a specific segment from a clip file based on percentage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract segment at 50% of clip duration (5s before/after that point)
  python3 scripts/extract_chirp_segment.py clips/clip_2025-12-27_14-30-00.wav --percent 50
  
  # Prompt for percentage if not provided
  python3 scripts/extract_chirp_segment.py clips/clip_2025-12-27_14-30-00.wav
  
  # Update events.csv to point to new clip
  python3 scripts/extract_chirp_segment.py clips/clip_2025-12-27_14-30-00.wav --percent 50 --update-events
  
  # Specify custom output filename
  python3 scripts/extract_chirp_segment.py clips/clip_2025-12-27_14-30-00.wav --percent 50 \\
    --output clips/chirp_clean.wav
        """
    )
    parser.add_argument("clip", type=Path, help="Path to source clip file")
    parser.add_argument("--percent", type=float, help="Percentage (1-100) of clip duration to extract segment at")
    parser.add_argument("--output", type=Path, help="Output clip path (default: adds _segment suffix)")
    parser.add_argument("--update-events", action="store_true",
                       help="Update events.csv to point to new clip")
    parser.add_argument("--config", type=Path, help="Path to config.json")
    
    args = parser.parse_args()
    
    # Get clip duration
    duration = get_clip_duration(args.clip)
    if duration is None:
        sys.exit(1)
    
    print(f"Clip duration: {duration:.1f} seconds")
    
    # Get percentage (prompt if not provided)
    percent = args.percent
    if percent is None:
        while True:
            try:
                percent_input = input("Enter percentage (1-100) of clip to extract segment at: ").strip()
                percent = float(percent_input)
                if 1 <= percent <= 100:
                    break
                else:
                    print("Error: Percentage must be between 1 and 100")
            except ValueError:
                print("Error: Please enter a valid number")
            except (EOFError, KeyboardInterrupt):
                print("\nCancelled")
                sys.exit(1)
    
    # Validate percentage
    if not (1 <= percent <= 100):
        print(f"[ERROR] Percentage must be between 1 and 100, got: {percent}")
        sys.exit(1)
    
    # Calculate center point
    center_sec = (percent / 100.0) * duration
    print(f"Extracting segment at {percent:.1f}% ({center_sec:.1f}s) with 5s padding before/after")
    
    # Extract segment (always 5 seconds before/after)
    new_clip_path = extract_segment_from_clip(
        args.clip,
        center_sec,
        padding_sec=5.0,
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

