"""
Report generation and event data loading.

This module provides functions for loading events data and generating reports.

Single Responsibility: Event data loading and report generation.
"""
import warnings
import sys
from io import StringIO
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

# Suppress NumPy 2.0 compatibility warnings from optional pandas dependencies
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*Bottleneck.*')

# Redirect stderr and stdout during pandas import to suppress optional dependency errors
_original_stderr = sys.stderr
_original_stdout = sys.stdout
try:
    with open(sys.devnull, 'w') as devnull:
        sys.stderr = devnull
        sys.stdout = devnull
        import pandas as pd
finally:
    sys.stderr = _original_stderr
    sys.stdout = _original_stdout


def load_events(events_file: Path) -> pd.DataFrame:
    """
    Load events CSV file.
    
    Args:
        events_file: Path to events.csv file
        
    Returns:
        DataFrame with events data, or empty DataFrame if file doesn't exist
    """
    if not events_file.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(events_file)
    df.columns = [c.strip() for c in df.columns]
    return df


def filter_recent_events(df: pd.DataFrame, hours: int = 2) -> pd.DataFrame:
    """
    Filter events to the last N hours.
    
    Args:
        df: Events DataFrame
        hours: Number of hours to look back (default: 2)
        
    Returns:
        Filtered DataFrame with only recent events
    """
    if df.empty or "start_timestamp" not in df.columns:
        return df
    
    try:
        df["timestamp_parsed"] = pd.to_datetime(df["start_timestamp"])
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = df[df["timestamp_parsed"] >= cutoff].copy()
        return recent
    except Exception as e:
        print(f"[WARN] Failed to filter by timestamp: {e}")
        return df


def generate_email_report(df: pd.DataFrame, hours: int = 2) -> str:
    """
    Generate email report text from events DataFrame.
    
    Args:
        df: Events DataFrame
        hours: Number of hours covered by report (for header)
        
    Returns:
        Formatted report text as string
    """
    lines = []
    
    # Header
    now = datetime.now()
    period_start = now - timedelta(hours=hours)
    lines.append(f"Noise Detector Report - Last {hours} Hours")
    lines.append("=" * 60)
    lines.append(f"Period: {period_start.strftime('%Y-%m-%d %H:%M:%S')} to {now.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    if df.empty:
        lines.append("No events recorded in this period.")
        return "\n".join(lines)
    
    # Summary statistics
    total_clips = len(df)
    lines.append(f"Total clips created: {total_clips}")
    lines.append("")
    
    # Chirp statistics
    if "is_chirp" in df.columns:
        chirps = df[df["is_chirp"].astype(str).str.upper().isin(["TRUE", "1", "YES"])]
        total_chirps = len(chirps)
        lines.append(f"Events identified as chirps: {total_chirps}")
        lines.append(f"Non-chirp events: {total_clips - total_chirps}")
        lines.append("")
        
        # Chirp details with confidence
        if total_chirps > 0:
            lines.append("Chirp Details:")
            lines.append("-" * 120)
            
            # Sort by confidence if available, otherwise by similarity
            if "confidence" in chirps.columns:
                chirps = chirps.copy()
                chirps["confidence_float"] = pd.to_numeric(chirps["confidence"], errors="coerce")
                chirps_sorted = chirps.sort_values("confidence_float", ascending=False, na_position="last")
            elif "chirp_similarity" in chirps.columns:
                chirps = chirps.copy()
                chirps["similarity_float"] = pd.to_numeric(chirps["chirp_similarity"], errors="coerce")
                chirps_sorted = chirps.sort_values("similarity_float", ascending=False, na_position="last")
            else:
                chirps_sorted = chirps.sort_values("start_timestamp", ascending=False)
            
            # Table header
            lines.append(f"{'Timestamp':<20} {'Duration':20} {'Confidence':<12} {'Similarity':<12} {'Clip File'}")
            lines.append("-" * 120)
            
            for _, row in chirps_sorted.iterrows():
                timestamp = row.get("start_timestamp", "N/A")
                duration = row.get("duration_sec", "")
                duration_str = f"{float(duration):.2f}s" if duration else "N/A"
                
                confidence = row.get("confidence", "")
                conf_str = f"{float(confidence):.3f}" if confidence and pd.notna(confidence) else "N/A"
                
                similarity = row.get("chirp_similarity", "")
                sim_str = f"{float(similarity):.3f}" if similarity and pd.notna(similarity) else "N/A"
                
                clip_file = row.get("clip_file", "")
                clip_name = Path(clip_file).name if clip_file else "N/A"
                
                lines.append(f"{timestamp:<20} {duration_str:<20} {conf_str:<12} {sim_str:<12} {clip_name}")
        else:
            lines.append("No chirps detected in this period.")
    else:
        lines.append("Chirp classification not available (no 'is_chirp' column).")
    
    lines.append("")
    lines.append("=" * 60)
    
    return "\n".join(lines)


def add_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add date column to events DataFrame.
    
    Assumes ISO-ish timestamps like 2025-11-23T12:43:55
    
    Args:
        df: Events DataFrame with 'start_timestamp' column
        
    Returns:
        DataFrame with added 'date' column
    """
    df = df.copy()
    df["date"] = df["start_timestamp"].str.slice(0, 10)
    return df


def choose_latest_date(df: pd.DataFrame) -> Optional[str]:
    """
    Choose the latest date from events DataFrame.
    
    Args:
        df: Events DataFrame with 'date' column
        
    Returns:
        Latest date as string (YYYY-MM-DD) or None if no dates
    """
    if "date" not in df.columns:
        return None
    dates = sorted(df["date"].unique())
    if not dates:
        return None
    return dates[-1]


def generate_chirp_report(df: pd.DataFrame, report_date: str) -> str:
    """
    Generate markdown chirp report for a specific date.
    
    Args:
        df: Events DataFrame with 'date' column
        report_date: Date to generate report for (YYYY-MM-DD)
        
    Returns:
        Markdown formatted report text
    """
    df_day = df[df["date"] == report_date].copy()
    
    total_events = len(df_day)
    
    has_chirp_cols = "is_chirp" in df_day.columns and "chirp_similarity" in df_day.columns
    
    chirp_events = None
    if has_chirp_cols:
        # Normalize boolean/text values
        mask = df_day["is_chirp"].astype(str).str.upper().isin(["TRUE", "1", "YES"])
        chirp_events = df_day[mask].copy()
        chirp_events = chirp_events.sort_values(
            by="chirp_similarity", ascending=False
        )
    else:
        # Fallback: no classifier yet, use top N loud events
        chirp_events = df_day.sort_values(
            by="max_rms_db", ascending=False
        ).head(10)
    
    lines = []
    lines.append(f"# Noise Chirp Report – {report_date}")
    lines.append("")
    lines.append(f"- Total events recorded: **{total_events}**")
    if has_chirp_cols:
        lines.append(f"- Events classified as chirp: **{len(chirp_events)}**")
    else:
        lines.append(
            "- No chirp classifier columns found; showing top loudest events as candidates."
        )
    lines.append("")
    
    if chirp_events.empty:
        lines.append("No chirp-like events found for this date.")
        return "\n".join(lines)
    
    lines.append("## Chirp Events")
    lines.append("")
    lines.append("| Start | End | Duration (s) | Max RMS (dB) | Similarity | Clip |")
    lines.append("|-------|-----|-------------:|-------------:|-----------:|------|")
    
    for _, row in chirp_events.iterrows():
        start = row.get("start_timestamp", "")
        end = row.get("end_timestamp", "")
        duration = row.get("duration_sec", "")
        max_rms = row.get("max_rms_db", "")
        clip = row.get("clip_file", "")
        
        if has_chirp_cols:
            sim = row.get("chirp_similarity", "")
            sim_str = f"{sim:.3f}" if pd.notna(sim) else ""
        else:
            sim_str = ""
        
        lines.append(
            f"| {start} | {end} | {duration:.2f} | {max_rms:.2f} | {sim_str} | {clip} |"
        )
    
    return "\n".join(lines)

