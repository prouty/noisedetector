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
from core.reporting import load_events


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


def find_event_by_clip(clip_path: Path, df: pd.DataFrame, clips_dir: Path) -> Optional[tuple]:
    """Find the event row in the dataframe that matches this clip.
    
    Args:
        clip_path: Path to the clip file
        df: DataFrame containing events (must have 'clip_file' column)
        clips_dir: Directory where clips are stored
    
    Returns:
        Tuple of (row Series, integer position in dataframe) if found, None otherwise.
    """
    if df.empty or "clip_file" not in df.columns:
        return None
    
    # Normalize the clip path to handle different formats
    clip_filename = clip_path.name
    clip_basename = clip_path.stem  # filename without extension
    
    # Try multiple matching strategies
    matches = None
    
    # Strategy 1: Exact filename match (most reliable)
    # Match if clip_file ends with the filename
    matches = df[df["clip_file"].astype(str).str.endswith(clip_filename, na=False)]
    
    if len(matches) == 0:
        # Strategy 2: Match by basename (without extension) in case extension differs
        matches = df[df["clip_file"].astype(str).str.contains(clip_basename, na=False, regex=False)]
    
    if len(matches) == 0:
        # Strategy 3: Try matching with normalized paths (handle different separators)
        clip_str = str(clip_path).replace("\\", "/")
        clip_relative = str(clip_path.relative_to(clips_dir)) if clips_dir in clip_path.parents else clip_filename
        clip_relative_norm = clip_relative.replace("\\", "/")
        
        # Try exact match with normalized path
        matches = df[df["clip_file"].astype(str).str.replace("\\", "/", regex=False) == clip_str]
        
        if len(matches) == 0:
            # Try relative path match
            matches = df[df["clip_file"].astype(str).str.replace("\\", "/", regex=False).str.endswith(clip_relative_norm, na=False)]
    
    if len(matches) > 0:
        # Return the most recent match (last row) - this handles duplicates
        # Since df has a reset index (0, 1, 2, ...), the position is just the index value
        matched_row = matches.iloc[-1]
        # The integer position is the index value (since index was reset)
        int_idx = matched_row.name
        # Return both the row and its position
        return (matched_row, int_idx)
    
    return None


def update_events_csv(events_file: Path, clip_path: Path, is_chirp: bool, clips_dir: Path):
    """Update events.csv to mark the clip as reviewed."""
    # Try to resolve the path if it doesn't exist
    if not events_file.exists():
        # Try common locations
        candidates = [
            events_file,
            events_file.resolve() if events_file.is_absolute() else None,
            Path("data/events.csv"),
            Path.cwd() / "data/events.csv",
        ]
        # Filter out None values
        candidates = [c for c in candidates if c is not None]
        
        for candidate in candidates:
            if candidate.exists():
                events_file = candidate
                break
        else:
            print(f"[WARN] Events file not found: {events_file}")
            print(f"  Tried locations:")
            for c in candidates:
                print(f"    - {c}")
            return False
    
    df = load_events(events_file)
    if df.empty:
        print("[WARN] Events file is empty")
        return False
    
    # Reset index to ensure it's a simple integer range (0, 1, 2, ...)
    # This prevents issues with updating rows
    df = df.reset_index(drop=True)
    
    # Find the matching event - pass the dataframe directly
    match_result = find_event_by_clip(clip_path, df, clips_dir)
    
    if match_result is None:
        print(f"[WARN] Could not find event in events.csv for clip: {clip_path.name}")
        print(f"  Looking for clip: {clip_path}")
        print(f"  In events file: {events_file}")
        # Show a few example clip_file values for debugging
        if "clip_file" in df.columns:
            sample_clips = df["clip_file"].dropna().head(3).tolist()
            print(f"  Sample clip_file values in CSV: {sample_clips}")
        print("  The clip will be moved to training, but events.csv won't be updated")
        return False
    
    # Unpack the result: (row Series, integer position)
    event_row, int_idx = match_result
    
    # Get the label index for this row (for display)
    label_idx = event_row.name
    
    # Get current values for debugging (using integer position)
    old_reviewed = str(df.iloc[int_idx]["reviewed"]) if "reviewed" in df.columns and pd.notna(df.iloc[int_idx]["reviewed"]) else ""
    old_is_chirp = str(df.iloc[int_idx]["is_chirp"]) if "is_chirp" in df.columns and pd.notna(df.iloc[int_idx]["is_chirp"]) else ""
    
    print(f"[DEBUG] Found event at row {int_idx} (label: {label_idx})")
    print(f"  Current values: reviewed={old_reviewed!r}, is_chirp={old_is_chirp!r}")
    print(f"  Updating to: reviewed=YES, is_chirp={'TRUE' if is_chirp else 'FALSE'}")
    print(f"  Total rows before update: {len(df)}")
    
    # Find the row by matching clip_file directly and update it
    # This is more reliable than using index positions
    clip_filename = clip_path.name
    clip_basename = clip_path.stem
    
    # Find matching rows using the same logic as find_event_by_clip
    mask = df["clip_file"].astype(str).str.endswith(clip_filename, na=False)
    if not mask.any():
        mask = df["clip_file"].astype(str).str.contains(clip_basename, na=False, regex=False)
    
    if mask.any():
        # Get the last matching row (most recent) - now using integer position since we reset index
        matching_positions = df[mask].index.tolist()
        target_pos = matching_positions[-1]  # Last match (integer position)
        
        # Update using integer position - this will definitely update existing row
        df.iloc[target_pos, df.columns.get_loc("reviewed")] = "YES"
        df.iloc[target_pos, df.columns.get_loc("is_chirp")] = "TRUE" if is_chirp else "FALSE"
        
        print(f"  Updated row at position: {target_pos}")
        verify_pos = target_pos
    else:
        # Fallback: use integer position from find_event_by_clip
        # Adjust int_idx if needed (since we reset index, it should be correct)
        print(f"  [WARN] Could not find row by mask, using integer position {int_idx}")
        df.iloc[int_idx, df.columns.get_loc("reviewed")] = "YES"
        df.iloc[int_idx, df.columns.get_loc("is_chirp")] = "TRUE" if is_chirp else "FALSE"
        verify_pos = int_idx
    
    # Verify the update worked
    new_reviewed = str(df.iloc[verify_pos]["reviewed"])
    new_is_chirp = str(df.iloc[verify_pos]["is_chirp"])
    
    print(f"  After update: reviewed={new_reviewed!r}, is_chirp={new_is_chirp!r}")
    print(f"  Total rows after update: {len(df)}")
    
    # Check for duplicate clip_file entries and remove the old one
    clip_filename = clip_path.name
    clip_basename = clip_path.stem
    
    # Find all rows with matching clip_file
    duplicate_mask = df["clip_file"].astype(str).str.endswith(clip_filename, na=False)
    if not duplicate_mask.any():
        duplicate_mask = df["clip_file"].astype(str).str.contains(clip_basename, na=False, regex=False)
    
    if duplicate_mask.sum() > 1:
        print(f"  [WARN] Found {duplicate_mask.sum()} rows with matching clip_file, removing duplicates...")
        duplicate_indices = df[duplicate_mask].index.tolist()
        
        # Keep the row with the updated values (FALSE for is_chirp if marking as not_chirp, or YES for reviewed)
        # Delete rows that still have the old values
        rows_to_keep = []
        rows_to_delete = []
        
        for dup_idx in duplicate_indices:
            row_reviewed = str(df.iloc[dup_idx]["reviewed"]) if pd.notna(df.iloc[dup_idx]["reviewed"]) else ""
            row_is_chirp = str(df.iloc[dup_idx]["is_chirp"]) if pd.notna(df.iloc[dup_idx]["is_chirp"]) else ""
            
            # Keep rows that have been updated (reviewed=YES) or match our target state
            if row_reviewed.upper() == "YES":
                rows_to_keep.append(dup_idx)
            elif is_chirp and row_is_chirp.upper() in ["TRUE", "TRUE"]:
                # If marking as chirp, keep rows that are TRUE
                rows_to_keep.append(dup_idx)
            elif not is_chirp and row_is_chirp.upper() in ["FALSE", "False"]:
                # If marking as not_chirp, keep rows that are FALSE
                rows_to_keep.append(dup_idx)
            else:
                # Delete rows that still have old values (TRUE when we want FALSE, or not reviewed)
                rows_to_delete.append(dup_idx)
        
        # If we have multiple rows to keep, keep the most recent one (last in list)
        if len(rows_to_keep) > 1:
            rows_to_delete.extend(rows_to_keep[:-1])  # Delete all but the last one
            rows_to_keep = [rows_to_keep[-1]]
        
        # Delete duplicate rows (in reverse order to maintain indices)
        if rows_to_delete:
            print(f"  Deleting {len(rows_to_delete)} duplicate row(s): {rows_to_delete}")
            df = df.drop(index=df.index[rows_to_delete]).reset_index(drop=True)
            print(f"  Kept row at position: {rows_to_keep[0] if rows_to_keep else 'none'}")
            print(f"  Total rows after deduplication: {len(df)}")
    
    # Save updated CSV
    try:
        df.to_csv(events_file, index=False)
        print(f"✓ Updated events.csv: marked as {'chirp' if is_chirp else 'not_chirp'}")
        if 'int_idx' in locals():
            print(f"  Row {int_idx}: reviewed: {old_reviewed!r} → {new_reviewed!r}, is_chirp: {old_is_chirp!r} → {new_is_chirp!r}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save events.csv: {e}")
        import traceback
        traceback.print_exc()
        return False


def mark_clip(
    clip_path: Path,
    is_chirp: bool,
    config_path: Optional[Path] = None,
    events_file: Optional[Path] = None
):
    """Mark a clip as chirp or not-chirp and move to training directory."""
    config = config_loader.load_config(config_path)
    
    # Determine project root (where config.json is located)
    if config_path is not None and config_path.exists():
        project_root = config_path.parent.resolve()
    else:
        # Try to find config.json in current directory or parent
        config_candidate = Path("config.json")
        if config_candidate.exists():
            project_root = config_candidate.parent.resolve()
        else:
            # Fall back to current working directory
            project_root = Path.cwd()
    
    if events_file is None:
        events_file = Path(config["event_detection"]["events_file"])
    
    # Handle relative paths - resolve relative to project root
    if not events_file.is_absolute():
        events_file = project_root / events_file
    
    # If the resolved path doesn't exist, try common alternative locations
    if not events_file.exists():
        # Try data/ subdirectory (common case where config has "events.csv" but file is in "data/events.csv")
        data_candidate = project_root / "data" / events_file.name
        if data_candidate.exists():
            events_file = data_candidate
        # Also try just "data/events.csv" as fallback
        elif (project_root / "data" / "events.csv").exists():
            events_file = project_root / "data" / "events.csv"
    
    clips_dir = Path(config["event_clips"]["clips_dir"])
    if not clips_dir.is_absolute():
        clips_dir = project_root / clips_dir
    
    # Resolve clip path - try multiple locations
    if not clip_path.is_absolute():
        # Strategy 1: If path contains directory (e.g., "clips/file.wav"), try relative to project root
        if "/" in str(clip_path) or "\\" in str(clip_path):
            candidate = project_root / clip_path
            if candidate.exists():
                clip_path = candidate
            elif clip_path.exists():
                clip_path = clip_path.resolve()
        # Strategy 2: Try relative to clips directory (just filename)
        elif (clips_dir / clip_path.name).exists():
            clip_path = clips_dir / clip_path.name
        # Strategy 3: Try as-is (relative to current directory)
        elif clip_path.exists():
            clip_path = clip_path.resolve()
        # Strategy 4: Try relative to project root
        else:
            candidate = project_root / clip_path
            if candidate.exists():
                clip_path = candidate
            else:
                print(f"[ERROR] Clip file not found: {clip_path}")
                print(f"  Tried: {project_root / clip_path}")
                print(f"  Tried: {clips_dir / clip_path.name}")
                return False
    
    if not clip_path.exists():
        print(f"[ERROR] Clip file not found: {clip_path}")
        return False
    
    # Validate that the path is a file, not a directory
    if not clip_path.is_file():
        print(f"[ERROR] Path is not a file: {clip_path}")
        if clip_path.is_dir():
            print(f"  This is a directory. Please specify a specific clip file, e.g.:")
            print(f"  make mark-chirp CLIP=clips/clip_2025-01-01_12-00-00.wav")
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
    
    # Update events.csv (use the resolved path)
    # Make sure events_file is resolved before passing to update_events_csv
    events_file_resolved = events_file.resolve() if events_file.exists() else events_file
    
    update_events_csv(events_file_resolved, clip_path, is_chirp, clips_dir)
    
    return True


def mark_latest_unreviewed(is_chirp: bool, config_path: Optional[Path] = None):
    """Mark the latest unreviewed event from events.csv."""
    config = config_loader.load_config(config_path)
    
    # Determine project root (where config.json is located)
    if config_path is not None and config_path.exists():
        project_root = config_path.parent.resolve()
    else:
        # Try to find config.json in current directory or parent
        config_candidate = Path("config.json")
        if config_candidate.exists():
            project_root = config_candidate.parent.resolve()
        else:
            # Fall back to current working directory
            project_root = Path.cwd()
    
    events_file = Path(config["event_detection"]["events_file"])
    clips_dir = Path(config["event_clips"]["clips_dir"])
    
    # Handle relative paths - resolve relative to project root
    if not events_file.is_absolute():
        events_file = project_root / events_file
    
    # If the resolved path doesn't exist, try common alternative locations
    if not events_file.exists():
        # Try data/ subdirectory (common case where config has "events.csv" but file is in "data/events.csv")
        data_candidate = project_root / "data" / events_file.name
        if data_candidate.exists():
            events_file = data_candidate
        # Also try just "data/events.csv" as fallback
        elif (project_root / "data" / "events.csv").exists():
            events_file = project_root / "data" / "events.csv"
    
    if not clips_dir.is_absolute():
        clips_dir = project_root / clips_dir
    
    if not events_file.exists():
        print(f"[ERROR] Events file not found: {events_file}")
        return False
    
    df = load_events(events_file)
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

