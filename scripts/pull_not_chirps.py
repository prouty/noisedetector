#!/usr/bin/env python3
"""Extract non-chirp clip filenames from events.csv and prepare for rsync."""
import csv
import sys
from pathlib import Path
import pandas as pd


def get_not_chirp_clips(events_file: Path, max_duration_sec: float = 10.0) -> list:
    """Get list of clip filenames that are classified as not chirps and <= max_duration_sec."""
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
    
    if "is_chirp" not in df.columns or "clip_file" not in df.columns:
        return []
    
    # Filter out reviewed clips (skip if reviewed is YES or TRUE)
    if "reviewed" in df.columns:
        df = df[
            (df["reviewed"].astype(str).str.upper() != "YES") &
            (df["reviewed"].astype(str).str.upper() != "TRUE")
        ]
    
    # Filter to non-chirps (FALSE or empty)
    not_chirps = df[
        (df["is_chirp"].astype(str).str.upper() == "FALSE") |
        (df["is_chirp"].astype(str).str.strip() == "")
    ]
    
    # Filter by duration if duration_sec column exists
    if "duration_sec" in not_chirps.columns:
        not_chirps = not_chirps.copy()
        not_chirps["duration_sec"] = pd.to_numeric(not_chirps["duration_sec"], errors="coerce")
        not_chirps = not_chirps[
            (not_chirps["duration_sec"] <= max_duration_sec) &
            (not_chirps["clip_file"].notna()) &
            (not_chirps["clip_file"] != "")
        ]
    
    # Extract clip filenames
    clip_files = []
    for _, row in not_chirps.iterrows():
        clip_file = row.get("clip_file", "")
        if clip_file:
            # Extract just the filename (in case path is included)
            clip_path = Path(clip_file)
            clip_files.append(clip_path.name)
    
    return clip_files


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Get non-chirp clips from events.csv")
    parser.add_argument("events_file", nargs="?", type=Path, default=Path("events.csv"),
                       help="Path to events.csv")
    parser.add_argument("--max-duration", type=float, default=10.0,
                       help="Maximum duration in seconds (default: 10.0)")
    
    args = parser.parse_args()
    
    clips = get_not_chirp_clips(args.events_file, args.max_duration)
    
    # Output filenames, one per line (for use with rsync --files-from)
    for clip in clips:
        print(clip)

