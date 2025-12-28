#!/usr/bin/env python3
"""
Show events from events.csv.

Displays recent events with key information like timestamp, duration, classification, etc.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.reporting import load_events
import pandas as pd


def show_events(
    events_file: Path = Path("data/events.csv"),
    recent_hours: Optional[int] = None,
    limit: Optional[int] = None,
    show_all: bool = False
):
    """
    Show events from events.csv.
    
    Args:
        events_file: Path to events.csv
        recent_hours: Show only events from last N hours (None = all)
        limit: Maximum number of events to show (None = all)
        show_all: If True, show all columns (default: show key columns only)
    """
    df = load_events(events_file)
    
    if df.empty:
        print("No events found in events.csv")
        return
    
    # Filter by time if requested
    if recent_hours and "start_timestamp" in df.columns:
        try:
            df["timestamp_parsed"] = pd.to_datetime(df["start_timestamp"], errors="coerce")
            cutoff = datetime.now() - timedelta(hours=recent_hours)
            df = df[df["timestamp_parsed"] >= cutoff].copy()
            if df.empty:
                print(f"No events found in the last {recent_hours} hours")
                return
        except Exception as e:
            print(f"Warning: Could not filter by time: {e}")
    
    # Sort by timestamp (most recent first)
    if "start_timestamp" in df.columns:
        df = df.sort_values("start_timestamp", ascending=False)
    
    # Limit results
    if limit:
        df = df.head(limit)
    
    total_events = len(df)
    
    print("=" * 80)
    print("EVENTS LOG")
    print("=" * 80)
    
    if recent_hours:
        print(f"Showing events from last {recent_hours} hours")
    elif limit:
        print(f"Showing most recent {limit} events")
    else:
        print(f"Showing all events")
    
    print(f"Total events displayed: {total_events}")
    print()
    
    if show_all:
        # Show all columns in a table format
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 50)
        print(df.to_string(index=False))
    else:
        # Show key columns in a formatted way
        key_columns = [
            "start_timestamp",
            "duration_sec",
            "is_chirp",
            "max_rms_db",
            "clip_file",
            "reviewed",
            "manual_capture"
        ]
        
        # Only show columns that exist
        display_cols = [col for col in key_columns if col in df.columns]
        
        if not display_cols:
            print("No displayable columns found")
            return
        
        # Format the output
        for idx, row in df.iterrows():
            print("-" * 80)
            
            # Timestamp
            if "start_timestamp" in row:
                print(f"Time: {row['start_timestamp']}")
            
            # Duration
            if "duration_sec" in row:
                duration = row["duration_sec"]
                try:
                    duration_float = float(duration)
                    print(f"Duration: {duration_float:.1f}s")
                except:
                    print(f"Duration: {duration}")
            
            # Classification
            if "is_chirp" in row:
                is_chirp = str(row["is_chirp"]).upper()
                chirp_status = "✓ CHIRP" if is_chirp == "TRUE" else "✗ noise"
                print(f"Type: {chirp_status}")
            
            # Manual capture
            if "manual_capture" in row:
                manual = str(row["manual_capture"]).upper()
                if manual == "TRUE":
                    print("Source: Manual capture")
            
            # Audio levels
            if "max_rms_db" in row:
                try:
                    rms = float(row["max_rms_db"])
                    print(f"Max RMS: {rms:.1f} dBFS")
                except:
                    pass
            
            # Clip file
            if "clip_file" in row:
                clip = row["clip_file"]
                if pd.notna(clip) and str(clip).strip():
                    clip_path = Path(clip)
                    print(f"Clip: {clip_path.name}")
            
            # Reviewed status
            if "reviewed" in row:
                reviewed = str(row["reviewed"]).upper()
                if reviewed in ["YES", "TRUE"]:
                    print("Status: Reviewed")
            
            print()
    
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Show events from events.csv")
    parser.add_argument("--events", type=Path, default=Path("data/events.csv"),
                       help="Path to events.csv (default: data/events.csv)")
    parser.add_argument("--recent", type=int, metavar="HOURS",
                       help="Show only events from last N hours")
    parser.add_argument("--limit", type=int, metavar="N",
                       help="Show only the most recent N events")
    parser.add_argument("--all", action="store_true",
                       help="Show all columns (default: show key columns only)")
    
    args = parser.parse_args()
    
    show_events(
        events_file=args.events,
        recent_hours=args.recent,
        limit=args.limit,
        show_all=args.all
    )

