#!/usr/bin/env python3
"""
Extract manually captured chirp clip filenames from events.csv.

Manually captured clips are identified by the manual_capture flag (TRUE).
Outputs filenames one per line for use with rsync --files-from.
"""
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.reporting import load_events


def get_manual_chirp_clips(events_file: Path) -> list:
    """
    Get list of clip filenames that are manually captured chirps.
    
    Manually captured clips are identified by the manual_capture flag (TRUE).
    """
    df = load_events(events_file)
    
    if df.empty:
        return []
    
    if "clip_file" not in df.columns:
        return []
    
    # Filter out reviewed clips (skip if reviewed is YES or TRUE)
    if "reviewed" in df.columns:
        df = df[
            (df["reviewed"].astype(str).str.upper() != "YES") &
            (df["reviewed"].astype(str).str.upper() != "TRUE")
        ]
    
    # Filter to manually captured clips
    if "manual_capture" in df.columns:
        manual_clips = df[
            (df["manual_capture"].astype(str).str.upper() == "TRUE") &
            (df["clip_file"].notna()) &
            (df["clip_file"] != "")
        ]
    else:
        # Fallback: if manual_capture column doesn't exist, return empty list
        # (old events.csv files won't have this column)
        return []
    
    # Extract clip filenames
    clip_files = []
    for _, row in manual_clips.iterrows():
        clip_file = row.get("clip_file", "")
        if clip_file:
            # Extract just the filename (in case path is included)
            clip_path = Path(clip_file)
            clip_files.append(clip_path.name)
    
    return clip_files


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Get manually captured chirp clips from events.csv")
    parser.add_argument("events_file", nargs="?", type=Path, default=Path("data/events.csv"),
                       help="Path to events.csv")
    
    args = parser.parse_args()
    
    clips = get_manual_chirp_clips(args.events_file)
    
    # Output filenames, one per line (for use with rsync --files-from)
    for clip in clips:
        print(clip)

