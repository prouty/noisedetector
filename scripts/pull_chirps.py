#!/usr/bin/env python3
"""Extract chirp clip filenames from events.csv and prepare for rsync."""
import csv
import sys
from pathlib import Path
import pandas as pd


def get_chirp_clips(events_file: Path) -> list:
    """Get list of clip filenames that are classified as chirps."""
    if not events_file.exists():
        print(f"Error: {events_file} not found", file=sys.stderr)
        return []
    
    df = pd.read_csv(events_file)
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
    
    # Filter to chirps
    chirps = df[df["is_chirp"].astype(str).str.upper() == "TRUE"]
    
    # Extract clip filenames
    clip_files = []
    for _, row in chirps.iterrows():
        clip_file = row.get("clip_file", "")
        if clip_file:
            # Extract just the filename (in case path is included)
            clip_path = Path(clip_file)
            clip_files.append(clip_path.name)
    
    return clip_files


if __name__ == "__main__":
    events_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("events.csv")
    clips = get_chirp_clips(events_file)
    
    # Output filenames, one per line (for use with rsync --files-from)
    for clip in clips:
        print(clip)

