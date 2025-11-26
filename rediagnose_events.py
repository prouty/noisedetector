#!/usr/bin/env python3
"""Re-classify all events in events.csv with the current algorithm and update the file."""
import csv
import json
from pathlib import Path
from typing import Optional, Dict
import pandas as pd

import config_loader
import monitor
import validate_classification


def rediagnose_events(
    events_file: Path = Path("events.csv"),
    config_path: Optional[Path] = None,
    backup: bool = True
):
    """Re-classify all events in events.csv and update the file."""
    config = config_loader.load_config(config_path)
    fingerprint_info = monitor.load_chirp_fingerprint(config)
    
    if not events_file.exists():
        print(f"Error: {events_file} not found")
        return
    
    # Backup original file
    if backup:
        backup_file = events_file.with_suffix('.csv.backup')
        import shutil
        shutil.copy2(events_file, backup_file)
        print(f"Backed up original to {backup_file}")
    
    # Load events
    df = pd.read_csv(events_file)
    if df.empty:
        print("No events found in CSV file")
        return
    
    # Ensure required columns exist
    required_cols = ["clip_file", "is_chirp", "chirp_similarity", "confidence", "rejection_reason"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""
    
    print(f"Re-classifying {len(df)} events...")
    print()
    
    clips_dir = Path(config["event_clips"]["clips_dir"])
    updated_count = 0
    
    for idx, row in df.iterrows():
        clip_file = row.get("clip_file", "")
        if not clip_file:
            continue
        
        clip_path = clips_dir / Path(clip_file).name
        if not clip_path.exists():
            clip_path = Path(clip_file)
        
        if not clip_path.exists():
            print(f"  Warning: Clip not found: {clip_file}")
            continue
        
        # Re-classify
        is_chirp, similarity, confidence, rejection_reason = validate_classification.classify_clip(
            clip_path, config, fingerprint_info
        )
        
        # Update row
        df.at[idx, "is_chirp"] = "TRUE" if is_chirp else "FALSE"
        if similarity is not None:
            df.at[idx, "chirp_similarity"] = f"{similarity:.3f}"
        else:
            df.at[idx, "chirp_similarity"] = ""
        
        if confidence is not None:
            df.at[idx, "confidence"] = f"{confidence:.3f}"
        else:
            df.at[idx, "confidence"] = ""
        
        if rejection_reason:
            df.at[idx, "rejection_reason"] = rejection_reason
        else:
            df.at[idx, "rejection_reason"] = ""
        
        updated_count += 1
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(df)} events...")
    
    # Save updated CSV
    df.to_csv(events_file, index=False)
    
    print()
    print(f"Updated {updated_count} events in {events_file}")
    print()
    
    # Show summary
    chirp_count = len(df[df["is_chirp"].astype(str).str.upper() == "TRUE"])
    non_chirp_count = len(df[df["is_chirp"].astype(str).str.upper() == "FALSE"])
    
    print("Classification Summary:")
    print(f"  Chirps: {chirp_count}")
    print(f"  Non-chirps: {non_chirp_count}")
    print(f"  Total: {len(df)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Re-classify all events in events.csv")
    parser.add_argument("--config", type=Path, help="Path to config.json")
    parser.add_argument("--events", type=Path, default=Path("events.csv"), help="Path to events.csv")
    parser.add_argument("--no-backup", action="store_true", help="Don't create backup file")
    
    args = parser.parse_args()
    rediagnose_events(args.events, args.config, backup=not args.no_backup)

