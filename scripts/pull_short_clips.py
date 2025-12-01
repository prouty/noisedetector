#!/usr/bin/env python3
"""
Extract clip filenames from events.csv that are <= 10 seconds long.

Outputs filenames one per line for use with rsync --files-from.
"""
import sys
import pandas as pd
from pathlib import Path


def get_short_clips(events_file: Path, max_duration_sec: float = 10.0) -> list:
    """Get list of clip filenames that are <= max_duration_sec."""
    if not events_file.exists():
        print(f"Error: {events_file} not found", file=sys.stderr)
        return []
    
    try:
        df = pd.read_csv(events_file)
    except Exception as e:
        print(f"Error reading {events_file}: {e}", file=sys.stderr)
        return []
    
    if df.empty:
        return []
    
    # Filter by duration
    if "duration_sec" not in df.columns:
        print("Warning: No duration_sec column found", file=sys.stderr)
        return []
    
    if "clip_file" not in df.columns:
        return []
    
    # Filter clips <= max_duration_sec
    df["duration_sec"] = pd.to_numeric(df["duration_sec"], errors="coerce")
    short_clips = df[
        (df["duration_sec"] <= max_duration_sec) &
        (df["clip_file"].notna()) &
        (df["clip_file"] != "")
    ]
    
    clip_files = []
    for _, row in short_clips.iterrows():
        clip_file = row["clip_file"]
        if clip_file:
            # Extract just the filename
            clip_path = Path(clip_file)
            clip_files.append(clip_path.name)
    
    return clip_files


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Get short clips from events.csv")
    parser.add_argument("events_file", nargs="?", type=Path, default=Path("data/events.csv"),
                       help="Path to events.csv")
    parser.add_argument("--max-duration", type=float, default=10.0,
                       help="Maximum duration in seconds (default: 10.0)")
    
    args = parser.parse_args()
    
    clips = get_short_clips(args.events_file, args.max_duration)
    
    # Output filenames, one per line (for use with rsync --files-from)
    for clip in clips:
        print(clip)

