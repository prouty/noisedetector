#!/usr/bin/env python3
"""
Extract manually captured chirp clip filenames from events.csv.

Manually captured clips are chirps with duration >= 180 seconds (3 minutes).
Outputs filenames one per line for use with rsync --files-from.
"""
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.reporting import load_events


def get_manual_chirp_clips(events_file: Path, min_duration_sec: float = 180.0) -> list:
    """
    Get list of clip filenames that are manually captured chirps.
    
    Manually captured clips are:
    - Marked as chirps (is_chirp=TRUE)
    - Have duration >= min_duration_sec (default 180s = 3 minutes)
    """
    df = load_events(events_file)
    
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
    
    # Filter by duration (manually captured clips are 3 minutes = 180 seconds)
    if "duration_sec" not in chirps.columns:
        return []
    
    chirps = chirps.copy()
    chirps["duration_sec"] = pd.to_numeric(chirps["duration_sec"], errors="coerce")
    manual_chirps = chirps[
        (chirps["duration_sec"] >= min_duration_sec) &
        (chirps["clip_file"].notna()) &
        (chirps["clip_file"] != "")
    ]
    
    # Extract clip filenames
    clip_files = []
    for _, row in manual_chirps.iterrows():
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
    parser.add_argument("--min-duration", type=float, default=180.0,
                       help="Minimum duration in seconds for manual clips (default: 180.0 = 3 minutes)")
    
    args = parser.parse_args()
    
    clips = get_manual_chirp_clips(args.events_file, args.min_duration)
    
    # Output filenames, one per line (for use with rsync --files-from)
    for clip in clips:
        print(clip)

