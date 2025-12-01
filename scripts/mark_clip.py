#!/usr/bin/env python3
"""
Mark a clip as a valid chirp or not a chirp.

This script:
1. Moves the clip to the appropriate training directory
2. Updates events.csv to mark it as reviewed

Usage:
    python3 scripts/mark_clip.py --chirp clips/clip_2025-01-01_12-00-00.wav
    python3 scripts/mark_clip.py --not-chirp clips/clip_2025-01-01_12-00-00.wav
    python3 scripts/mark_clip.py --chirp --from-events  # Mark latest unreviewed event
"""
import sys
import shutil
from pathlib import Path
from typing import Optional
import pandas as pd
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config_loader


def find_next_number(directory: Path, pattern_prefix: str) -> int:
    """Find the next available number for a file pattern."""
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
        return 1
    
    existing_files = sorted(directory.glob(f"{pattern_prefix}_*.wav"))
    if not existing_files:
        return 1
    
    # Extract numbers from filenames
    numbers = []
    for f in existing_files:
        try:
            # Extract number from filename like "chirp_5.wav" -> 5
            name = f.stem  # "chirp_5"
            num_str = name.split("_")[-1]
            numbers.append(int(num_str))
        except (ValueError, IndexError):
            continue
    
    if not numbers:
        return 1
    
    return max(numbers) + 1


def find_event_by_clip(clip_path: Path, events_file: Path, clips_dir: Path) -> Optional[pd.Series]:
    """Find the event row in events.csv that matches this clip."""
    if not events_file.exists():
        return None
    
    df = pd.read_csv(events_file)
    if df.empty or "clip_file" not in df.columns:
        return None
    
    # Try to match by filename
    clip_filename = clip_path.name
    
    # Check if clip_file column contains this filename
    matches = df[df["clip_file"].astype(str).str.contains(clip_filename, na=False)]
    
    if len(matches) > 0:
        # Return the most recent match (last row)
        return matches.iloc[-1]
    
    # Also try matching by full path
    clip_str = str(clip_path)
    matches = df[df["clip_file"].astype(str) == clip_str]
    
    if len(matches) > 0:
        return matches.iloc[-1]
    
    return None


def update_events_csv(events_file: Path, clip_path: Path, is_chirp: bool, clips_dir: Path):
    """Update events.csv to mark the clip as reviewed."""
    if not events_file.exists():
        print(f"[WARN] Events file not found: {events_file}")
        return False
    
    df = pd.read_csv(events_file)
    if df.empty:
        print("[WARN] Events file is empty")
        return False
    
    # Find the matching event
    event_row = find_event_by_clip(clip_path, events_file, clips_dir)
    
    if event_row is None:
        print(f"[WARN] Could not find event in events.csv for clip: {clip_path.name}")
        print("  The clip will be moved to training, but events.csv won't be updated")
        return False
    
    # Find the index of this row
    idx = event_row.name
    
    # Update the row
    df.at[idx, "reviewed"] = "YES"
    df.at[idx, "is_chirp"] = "TRUE" if is_chirp else "FALSE"
    
    # Save updated CSV
    df.to_csv(events_file, index=False)
    
    print(f"✓ Updated events.csv: marked as {'chirp' if is_chirp else 'not_chirp'}")
    return True


def mark_clip(
    clip_path: Path,
    is_chirp: bool,
    config_path: Optional[Path] = None,
    events_file: Optional[Path] = None
):
    """Mark a clip as chirp or not-chirp and move to training directory."""
    config = config_loader.load_config(config_path)
    
    if events_file is None:
        events_file = Path(config["event_detection"]["events_file"])
    
    # Handle relative paths
    if not events_file.is_absolute():
        events_file = Path.cwd() / events_file
    
    clips_dir = Path(config["event_clips"]["clips_dir"])
    
    # Resolve clip path
    if not clip_path.is_absolute():
        # Try relative to clips directory first
        if (clips_dir / clip_path.name).exists():
            clip_path = clips_dir / clip_path.name
        elif clip_path.exists():
            clip_path = clip_path.resolve()
        else:
            print(f"[ERROR] Clip file not found: {clip_path}")
            return False
    
    if not clip_path.exists():
        print(f"[ERROR] Clip file not found: {clip_path}")
        return False
    
    # Determine target directory and pattern
    if is_chirp:
        target_dir = Path("training/chirp")
        pattern_prefix = "chirp"
    else:
        target_dir = Path("training/not_chirp")
        pattern_prefix = "not_chirp"
    
    # Find next available number
    next_num = find_next_number(target_dir, pattern_prefix)
    target_filename = f"{pattern_prefix}_{next_num}.wav"
    target_path = target_dir / target_filename
    
    # Move file
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(clip_path), str(target_path))
        print(f"✓ Moved {clip_path.name} → {target_path}")
    except Exception as e:
        print(f"[ERROR] Failed to move file: {e}")
        return False
    
    # Update events.csv
    update_events_csv(events_file, clip_path, is_chirp, clips_dir)
    
    return True


def mark_latest_unreviewed(is_chirp: bool, config_path: Optional[Path] = None):
    """Mark the latest unreviewed event from events.csv."""
    config = config_loader.load_config(config_path)
    events_file = Path(config["event_detection"]["events_file"])
    clips_dir = Path(config["event_clips"]["clips_dir"])
    
    # Handle relative paths
    if not events_file.is_absolute():
        events_file = Path.cwd() / events_file
    
    if not events_file.exists():
        print(f"[ERROR] Events file not found: {events_file}")
        return False
    
    df = pd.read_csv(events_file)
    if df.empty:
        print("[ERROR] Events file is empty")
        return False
    
    # Find unreviewed events with clips
    if "reviewed" not in df.columns:
        df["reviewed"] = ""
    
    unreviewed = df[
        (df["reviewed"].isna() | (df["reviewed"] == "")) &
        (df["clip_file"].notna()) &
        (df["clip_file"] != "")
    ]
    
    if unreviewed.empty:
        print("No unreviewed events with clips found")
        return False
    
    # Get the most recent one (last row)
    latest = unreviewed.iloc[-1]
    clip_file = latest["clip_file"]
    
    # Resolve clip path
    clip_path = clips_dir / Path(clip_file).name
    if not clip_path.exists():
        clip_path = Path(clip_file)
    
    if not clip_path.exists():
        print(f"[ERROR] Clip file not found: {clip_file}")
        return False
    
    print(f"Marking latest unreviewed event: {clip_path.name}")
    return mark_clip(clip_path, is_chirp, config_path, events_file)


def main():
    parser = argparse.ArgumentParser(
        description="Mark a clip as valid chirp or not chirp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mark a specific clip as chirp
  python3 scripts/mark_clip.py --chirp clips/clip_2025-01-01_12-00-00.wav
  
  # Mark a specific clip as not chirp
  python3 scripts/mark_clip.py --not-chirp clips/clip_2025-01-01_12-00-00.wav
  
  # Mark latest unreviewed event as chirp
  python3 scripts/mark_clip.py --chirp --from-events
  
  # Mark latest unreviewed event as not chirp
  python3 scripts/mark_clip.py --not-chirp --from-events
        """
    )
    parser.add_argument("--chirp", action="store_true", help="Mark as valid chirp")
    parser.add_argument("--not-chirp", action="store_true", help="Mark as not chirp")
    parser.add_argument("--clip", type=Path, help="Path to clip file")
    parser.add_argument("--from-events", action="store_true", help="Use latest unreviewed event from events.csv")
    parser.add_argument("--config", type=Path, help="Path to config.json")
    parser.add_argument("--events", type=Path, help="Path to events.csv")
    
    args = parser.parse_args()
    
    if not args.chirp and not args.not_chirp:
        parser.error("Must specify either --chirp or --not-chirp")
    
    if args.chirp and args.not_chirp:
        parser.error("Cannot specify both --chirp and --not-chirp")
    
    is_chirp = args.chirp
    
    if args.from_events:
        success = mark_latest_unreviewed(is_chirp, args.config)
    elif args.clip:
        success = mark_clip(args.clip, is_chirp, args.config, args.events)
    else:
        parser.error("Must specify either --clip <path> or --from-events")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

