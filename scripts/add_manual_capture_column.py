#!/usr/bin/env python3
"""
Add manual_capture column to existing events.csv file if it doesn't exist.
"""
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config_loader

def main():
    config = config_loader.load_config()
    events_file = Path(config["event_detection"]["events_file"])
    
    if not events_file.exists():
        print(f"Events file not found: {events_file}")
        return
    
    df = pd.read_csv(events_file)
    
    if "manual_capture" in df.columns:
        print(f"Column 'manual_capture' already exists in {events_file}")
        return
    
    print(f"Adding 'manual_capture' column to {events_file}...")
    df["manual_capture"] = "FALSE"
    
    # Reorder columns to match expected order
    expected_columns = [
        "start_timestamp",
        "end_timestamp",
        "duration_sec",
        "max_peak_db",
        "max_rms_db",
        "baseline_rms_db",
        "segment_file",
        "segment_offset_sec",
        "clip_file",
        "is_chirp",
        "chirp_similarity",
        "confidence",
        "rejection_reason",
        "reviewed",
        "manual_capture",
    ]
    
    existing_cols = [col for col in expected_columns if col in df.columns]
    extra_cols = [col for col in df.columns if col not in expected_columns]
    df = df[existing_cols + extra_cols]
    
    # Backup original
    backup_file = events_file.with_suffix('.csv.backup')
    import shutil
    shutil.copy2(events_file, backup_file)
    print(f"Backup saved to {backup_file}")
    
    # Save updated file
    df.to_csv(events_file, index=False)
    print(f"✓ Updated {events_file} with 'manual_capture' column")

if __name__ == "__main__":
    main()

