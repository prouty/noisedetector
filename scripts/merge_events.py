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
    
    # Remove any 'index' column that might exist (pandas artifact)
    if 'index' in local_df.columns:
        local_df = local_df.drop(columns=['index'])
    if 'index' in remote_df.columns:
        remote_df = remote_df.drop(columns=['index'])
    
    # Ensure start_timestamp column exists (required for merging)
    # If missing, try to use 'index' column or create from index
    if 'start_timestamp' not in local_df.columns:
        # Check if there's an unnamed index that should be start_timestamp
        if local_df.index.name is None and len(local_df) > 0:
            # Try to infer from first column if it looks like a timestamp
            first_col = local_df.columns[0]
            if 'timestamp' in first_col.lower() or 'time' in first_col.lower():
                local_df = local_df.rename(columns={first_col: 'start_timestamp'})
            else:
                print(f"Error: 'start_timestamp' column not found in local file", file=sys.stderr)
                return stats
        else:
            print(f"Error: 'start_timestamp' column not found in local file", file=sys.stderr)
            return stats
    
    if 'start_timestamp' not in remote_df.columns:
        # Check if there's an unnamed index that should be start_timestamp
        if remote_df.index.name is None and len(remote_df) > 0:
            # Try to infer from first column if it looks like a timestamp
            first_col = remote_df.columns[0]
            if 'timestamp' in first_col.lower() or 'time' in first_col.lower():
                remote_df = remote_df.rename(columns={first_col: 'start_timestamp'})
            else:
                print(f"Error: 'start_timestamp' column not found in remote file", file=sys.stderr)
                return stats
        else:
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
    
    # Convert start_timestamp to string and filter out invalid values
    # This ensures consistent types for sorting and prevents errors
    def clean_timestamp(ts):
        """Convert timestamp to string, handling NaN and empty values."""
        if pd.isna(ts):
            return None
        ts_str = str(ts).strip()
        return ts_str if ts_str else None
    
    local_df['start_timestamp'] = local_df['start_timestamp'].apply(clean_timestamp)
    remote_df['start_timestamp'] = remote_df['start_timestamp'].apply(clean_timestamp)
    
    # Filter out rows with invalid timestamps
    local_df = local_df[local_df['start_timestamp'].notna()]
    remote_df = remote_df[remote_df['start_timestamp'].notna()]
    
    # Use start_timestamp as the merge key
    # Set it as index for easier merging (this will name the index 'start_timestamp')
    local_df = local_df.set_index('start_timestamp', drop=True)
    remote_df = remote_df.set_index('start_timestamp', drop=True)
    
    # Ensure index name is set correctly (should be 'start_timestamp' after set_index)
    local_df.index.name = 'start_timestamp'
    remote_df.index.name = 'start_timestamp'
    
    # Get all unique timestamps (all should be strings now)
    all_timestamps = set(local_df.index) | set(remote_df.index)
    
    # Build merged dataframe
    merged_rows = []
    
    # Sort timestamps (all are strings now, so sorting works)
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
            
            # Convert Series to dict and add timestamp
            row_dict = merged_row.to_dict()
            row_dict['start_timestamp'] = timestamp
            merged_rows.append(row_dict)
            
        elif remote_row is not None:
            # New row from remote (new event from Pi)
            stats['new_from_remote'] += 1
            # Convert Series to dict and add timestamp
            row_dict = remote_row.to_dict()
            row_dict['start_timestamp'] = timestamp
            merged_rows.append(row_dict)
            
        elif local_row is not None:
            # Local-only row (manually added entry)
            stats['local_only'] += 1
            # Convert Series to dict and add timestamp
            row_dict = local_row.to_dict()
            row_dict['start_timestamp'] = timestamp
            merged_rows.append(row_dict)
    
    # Create merged dataframe
    if merged_rows:
        merged_df = pd.DataFrame(merged_rows)
        # start_timestamp should already be a column (we added it to each dict)
        # No need to reset_index since we're not using index for timestamps anymore
        
        # Remove any 'index' column that might have been created (shouldn't happen, but safety check)
        if 'index' in merged_df.columns:
            # If 'index' exists but 'start_timestamp' doesn't, rename it
            if 'start_timestamp' not in merged_df.columns:
                merged_df = merged_df.rename(columns={'index': 'start_timestamp'})
            else:
                # Both exist - drop the redundant 'index' column
                merged_df = merged_df.drop(columns=['index'])
        
        # Define the correct column order (matching monitor.py ensure_events_header)
        expected_columns = [
            'start_timestamp',
            'end_timestamp',
            'duration_sec',
            'max_peak_db',
            'max_rms_db',
            'baseline_rms_db',
            'segment_file',
            'segment_offset_sec',
            'clip_file',
            'is_chirp',
            'chirp_similarity',
            'confidence',
            'rejection_reason',
            'reviewed'
        ]
        
        # Build column order: expected columns first (in order), then any extras
        column_order = [col for col in expected_columns if col in merged_df.columns]
        extra_cols = [col for col in merged_df.columns if col not in column_order]
        merged_df = merged_df[column_order + extra_cols]
        
        # Save merged result
        merged_df.to_csv(output_file, index=False)
    else:
        # Both files were empty or no rows matched
        if not remote_df.empty:
            result_df = remote_df.reset_index()
            if 'index' in result_df.columns and 'start_timestamp' not in result_df.columns:
                result_df = result_df.rename(columns={'index': 'start_timestamp'})
            elif 'index' in result_df.columns:
                result_df = result_df.drop(columns=['index'])
            result_df.to_csv(output_file, index=False)
        elif not local_df.empty:
            result_df = local_df.reset_index()
            if 'index' in result_df.columns and 'start_timestamp' not in result_df.columns:
                result_df = result_df.rename(columns={'index': 'start_timestamp'})
            elif 'index' in result_df.columns:
                result_df = result_df.drop(columns=['index'])
            result_df.to_csv(output_file, index=False)
    
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

