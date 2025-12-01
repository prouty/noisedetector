#!/usr/bin/env python3
"""
Merge two events.csv files, preserving local reviewed status and is_chirp values
while updating other columns from the remote file.

This script is used during 'make pull' to merge remote events.csv with local
reviewed status, preventing loss of review work.
"""
import sys
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional


def merge_events_csv(
    local_file: Path,
    remote_file: Path,
    output_file: Path,
    preserve_columns: list = None
) -> Dict[str, int]:
    """
    Merge local and remote events.csv files.
    
    Strategy:
    - Uses start_timestamp as the unique key to match rows
    - For matched rows: preserves local 'reviewed' and 'is_chirp' values,
      updates all other columns from remote
    - Adds new rows from remote that don't exist locally (new events from Pi)
    - Keeps local-only rows (edge case: manually added entries)
    
    Args:
        local_file: Path to local events.csv (with reviewed status)
        remote_file: Path to remote events.csv (from Pi)
        output_file: Path to save merged result
        preserve_columns: List of column names to preserve from local file
                         (default: ['reviewed', 'is_chirp'])
    
    Returns:
        Dictionary with merge statistics:
        - 'total_local': Total rows in local file
        - 'total_remote': Total rows in remote file
        - 'matched': Number of rows matched by start_timestamp
        - 'new_from_remote': Number of new rows added from remote
        - 'local_only': Number of local-only rows kept
        - 'preserved_reviews': Number of reviewed rows preserved
    """
    if preserve_columns is None:
        preserve_columns = ['reviewed', 'is_chirp']
    
    stats = {
        'total_local': 0,
        'total_remote': 0,
        'matched': 0,
        'new_from_remote': 0,
        'local_only': 0,
        'preserved_reviews': 0
    }
    
    # Load local file (optional - may not exist on first pull)
    local_df = None
    if local_file.exists():
        try:
            local_df = pd.read_csv(local_file)
            stats['total_local'] = len(local_df)
        except Exception as e:
            print(f"Warning: Error reading local file {local_file}: {e}", file=sys.stderr)
            local_df = None
    
    # Load remote file (required)
    if not remote_file.exists():
        print(f"Error: Remote file not found: {remote_file}", file=sys.stderr)
        return stats
    
    try:
        remote_df = pd.read_csv(remote_file)
        stats['total_remote'] = len(remote_df)
    except Exception as e:
        print(f"Error reading remote file {remote_file}: {e}", file=sys.stderr)
        return stats
    
    # If no local file, just copy remote to output
    if local_df is None or local_df.empty:
        remote_df.to_csv(output_file, index=False)
        stats['new_from_remote'] = len(remote_df)
        return stats
    
    # If remote is empty but local has data, keep local
    if remote_df.empty:
        local_df.to_csv(output_file, index=False)
        stats['local_only'] = len(local_df)
        return stats
    
    # Ensure start_timestamp column exists (required for merging)
    if 'start_timestamp' not in local_df.columns:
        print(f"Error: 'start_timestamp' column not found in local file", file=sys.stderr)
        return stats
    if 'start_timestamp' not in remote_df.columns:
        print(f"Error: 'start_timestamp' column not found in remote file", file=sys.stderr)
        return stats
    
    # Ensure reviewed column exists in both dataframes
    if 'reviewed' not in local_df.columns:
        local_df['reviewed'] = ''
    if 'reviewed' not in remote_df.columns:
        remote_df['reviewed'] = ''
    
    # Ensure is_chirp column exists
    if 'is_chirp' not in local_df.columns:
        local_df['is_chirp'] = ''
    if 'is_chirp' not in remote_df.columns:
        remote_df['is_chirp'] = ''
    
    # Use start_timestamp as the merge key
    # Set it as index for easier merging
    local_df = local_df.set_index('start_timestamp')
    remote_df = remote_df.set_index('start_timestamp')
    
    # Get all unique timestamps
    all_timestamps = set(local_df.index) | set(remote_df.index)
    
    # Build merged dataframe
    merged_rows = []
    
    for timestamp in sorted(all_timestamps):
        local_row = local_df.loc[timestamp] if timestamp in local_df.index else None
        remote_row = remote_df.loc[timestamp] if timestamp in remote_df.index else None
        
        if local_row is not None and remote_row is not None:
            # Matched row: preserve local reviewed/is_chirp, update other columns from remote
            stats['matched'] += 1
            
            # Start with remote row (has latest data)
            merged_row = remote_row.copy()
            
            # Preserve local values for specified columns if they exist and are not empty
            for col in preserve_columns:
                if col in local_row.index:
                    local_val = local_row[col]
                    # Preserve if local value is not empty/NaN
                    if pd.notna(local_val) and str(local_val).strip() != '':
                        merged_row[col] = local_val
                        if col == 'reviewed' and str(local_val).strip().upper() in ['YES', 'TRUE', '1']:
                            stats['preserved_reviews'] += 1
            
            merged_rows.append(merged_row)
            
        elif remote_row is not None:
            # New row from remote (new event from Pi)
            stats['new_from_remote'] += 1
            merged_rows.append(remote_row)
            
        elif local_row is not None:
            # Local-only row (manually added entry)
            stats['local_only'] += 1
            merged_rows.append(local_row)
    
    # Create merged dataframe
    if merged_rows:
        merged_df = pd.DataFrame(merged_rows)
        # Reset index to make start_timestamp a column again
        merged_df = merged_df.reset_index()
        
        # Ensure column order matches original (use remote as reference for order)
        if not remote_df.empty:
            # Get column order from remote (with start_timestamp first)
            original_columns = ['start_timestamp'] + [col for col in remote_df.columns if col != 'start_timestamp']
            # Only use columns that exist in merged_df
            column_order = [col for col in original_columns if col in merged_df.columns]
            # Add any extra columns from merged_df
            extra_cols = [col for col in merged_df.columns if col not in column_order]
            merged_df = merged_df[column_order + extra_cols]
        
        # Save merged result
        merged_df.to_csv(output_file, index=False)
    else:
        # Both files were empty or no rows matched
        if not remote_df.empty:
            remote_df.reset_index().to_csv(output_file, index=False)
        elif not local_df.empty:
            local_df.reset_index().to_csv(output_file, index=False)
    
    return stats


def main():
    """Command-line interface for merge_events_csv."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Merge local and remote events.csv files, preserving reviewed status"
    )
    parser.add_argument("local_file", type=Path, help="Path to local events.csv")
    parser.add_argument("remote_file", type=Path, help="Path to remote events.csv")
    parser.add_argument("output_file", type=Path, help="Path to save merged result")
    parser.add_argument(
        "--preserve",
        nargs="+",
        default=["reviewed", "is_chirp"],
        help="Column names to preserve from local file (default: reviewed is_chirp)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output (only print errors)"
    )
    
    args = parser.parse_args()
    
    # Perform merge
    stats = merge_events_csv(
        args.local_file,
        args.remote_file,
        args.output_file,
        preserve_columns=args.preserve
    )
    
    # Print statistics
    if not args.quiet:
        print(f"Merge complete: {args.output_file}")
        print(f"  Local events: {stats['total_local']}")
        print(f"  Remote events: {stats['total_remote']}")
        print(f"  Matched rows: {stats['matched']}")
        print(f"  New from remote: {stats['new_from_remote']}")
        print(f"  Local-only rows: {stats['local_only']}")
        print(f"  Preserved reviews: {stats['preserved_reviews']}")
    
    # Exit with error if merge failed
    if stats['total_local'] == 0 and stats['total_remote'] == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

