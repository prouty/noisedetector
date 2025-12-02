#!/usr/bin/env python3
"""
Filter out clip filenames that already exist in local directories.

Reads clip filenames from stdin (one per line) and outputs only those
that don't exist in any of these directories:
- clips/
- training/review/
- training/chirp/
- training/not_chirp/

Usage:
    python3 scripts/filter_existing_clips.py < /tmp/clips.txt > /tmp/new_clips.txt
"""
import sys
from pathlib import Path


def clip_exists_locally(clip_filename: str, project_root: Path) -> bool:
    """Check if clip exists in any local directory."""
    # Directories to check
    check_dirs = [
        project_root / "clips",
        project_root / "training" / "review",
        project_root / "training" / "chirp",
        project_root / "training" / "not_chirp",
    ]
    
    # Check each directory
    for check_dir in check_dirs:
        clip_path = check_dir / clip_filename
        if clip_path.exists():
            return True
    
    return False


def main():
    # Determine project root (where this script is located)
    script_dir = Path(__file__).parent.parent
    project_root = script_dir.resolve()
    
    # Read clip filenames from stdin
    existing_count = 0
    new_count = 0
    
    for line in sys.stdin:
        clip_filename = line.strip()
        if not clip_filename:
            continue
        
        # Check if clip already exists locally
        if clip_exists_locally(clip_filename, project_root):
            existing_count += 1
            # Don't output - skip this clip
        else:
            # Output clip filename (doesn't exist locally, should be transferred)
            print(clip_filename)
            new_count += 1
    
    # Print summary to stderr (so it doesn't interfere with stdout)
    total = existing_count + new_count
    if total > 0:
        if existing_count > 0:
            print(f"  Filtered out {existing_count} existing clip(s), {new_count} new clip(s) to transfer", file=sys.stderr)
        else:
            print(f"  All {new_count} clip(s) are new (none exist locally)", file=sys.stderr)


if __name__ == "__main__":
    main()

