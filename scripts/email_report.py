#!/usr/bin/env python3
"""
Generate and email a summary report of clips and chirps from the last 2 hours.
"""
import os
import sys
import warnings
from io import StringIO

# Suppress NumPy 2.0 compatibility warnings and errors from optional dependencies
# These are from numexpr/bottleneck which are optional pandas dependencies
# Pandas works fine without them, but they generate noisy error messages
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*Bottleneck.*')

# Redirect stderr and stdout during pandas import to suppress optional dependency errors
# These errors are from numexpr/bottleneck (optional dependencies) and can be safely ignored
_original_stderr = sys.stderr
_original_stdout = sys.stdout
try:
    with open(os.devnull, 'w') as devnull:
        sys.stderr = devnull
        sys.stdout = devnull
        import pandas as pd
finally:
    sys.stderr = _original_stderr
    sys.stdout = _original_stdout

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import argparse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
import config_loader


def load_events(events_file: Path) -> pd.DataFrame:
    """Load events CSV file."""
    if not events_file.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(events_file)
    df.columns = [c.strip() for c in df.columns]
    return df


def filter_recent_events(df: pd.DataFrame, hours: int = 2) -> pd.DataFrame:
    """Filter events to the last N hours."""
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


def generate_report(df: pd.DataFrame, hours: int = 2) -> str:
    """Generate email report text."""
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


def get_email_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """Get email configuration from environment variables or config."""
    email_config = {}
    
    # Try config file first
    if config and "email" in config:
        email_config = config["email"].copy()
    
    # Environment variables override config
    email_config["smtp_server"] = os.getenv("EMAIL_SMTP_SERVER", email_config.get("smtp_server", ""))
    email_config["smtp_port"] = int(os.getenv("EMAIL_SMTP_PORT", email_config.get("smtp_port", 587)))
    email_config["smtp_username"] = os.getenv("EMAIL_SMTP_USERNAME", email_config.get("smtp_username", ""))
    email_config["smtp_password"] = os.getenv("EMAIL_SMTP_PASSWORD", email_config.get("smtp_password", ""))
    email_config["from_address"] = os.getenv("EMAIL_FROM", email_config.get("from_address", ""))
    email_config["to_address"] = os.getenv("EMAIL_TO", email_config.get("to_address", ""))
    email_config["use_tls"] = os.getenv("EMAIL_USE_TLS", str(email_config.get("use_tls", True))).lower() == "true"
    
    return email_config


def send_email(report_text: str, email_config: Dict[str, Any]) -> bool:
    """Send email report."""
    if not email_config.get("smtp_server") or not email_config.get("to_address"):
        print("[ERROR] Email configuration incomplete. Set EMAIL_SMTP_SERVER and EMAIL_TO environment variables.")
        return False
    
    server = None
    email_sent = False
    try:
        # Create message
        msg = MIMEMultipart()
        msg["From"] = email_config.get("from_address", email_config.get("smtp_username", "noisedetector@raspberrypi"))
        msg["To"] = email_config["to_address"]
        msg["Subject"] = f"Noise Detector Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Add body
        msg.attach(MIMEText(report_text, "plain"))
        
        # Send email
        server = smtplib.SMTP(email_config["smtp_server"], email_config["smtp_port"])
        if email_config.get("use_tls", True):
            server.starttls()
        
        if email_config.get("smtp_username") and email_config.get("smtp_password"):
            server.login(email_config["smtp_username"], email_config["smtp_password"])
        
        server.send_message(msg)
        email_sent = True  # Mark as sent before cleanup
        
        print(f"[INFO] Email sent successfully to {email_config['to_address']}")
        return True
        
    except Exception as e:
        if email_sent:
            # Email was sent but cleanup failed - this is OK
            print(f"[WARN] Email sent but cleanup error occurred: {e}")
            return True
        else:
            # Actual send failure
            print(f"[ERROR] Failed to send email: {e}")
            return False
    finally:
        # Clean up connection - don't fail if quit() raises an exception
        if server:
            try:
                server.quit()
            except Exception as cleanup_error:
                # Only log cleanup errors if email wasn't already marked as sent
                if not email_sent:
                    # This shouldn't happen, but just in case
                    pass
                # Otherwise silently ignore - email was sent successfully


def main():
    parser = argparse.ArgumentParser(description="Generate and email noise detector report")
    parser.add_argument("--events", type=Path, default=Path("data/events.csv"), help="Path to events.csv")
    parser.add_argument("--hours", type=int, default=2, help="Number of hours to include in report (default: 2)")
    parser.add_argument("--config", type=Path, help="Path to config.json")
    parser.add_argument("--email-only", action="store_true", help="Only send email (don't print to console)")
    parser.add_argument("--no-email", action="store_true", help="Only print report (don't send email)")
    
    args = parser.parse_args()
    
    # Load config
    try:
        config = config_loader.load_config(args.config)
    except Exception:
        config = None
    
    # Load events
    df = load_events(args.events)
    if df.empty:
        report = f"No events found in {args.events}"
    else:
        # Filter to recent events
        df_recent = filter_recent_events(df, args.hours)
        report = generate_report(df_recent, args.hours)
    
    # Print report
    if not args.email_only:
        print(report)
    
    # Send email
    if not args.no_email:
        email_config = get_email_config(config)
        if email_config.get("smtp_server") and email_config.get("to_address"):
            send_email(report, email_config)
        else:
            print("[WARN] Email not sent - configuration incomplete")
            print("  Set EMAIL_SMTP_SERVER, EMAIL_TO, and optionally EMAIL_SMTP_USERNAME/EMAIL_SMTP_PASSWORD")


if __name__ == "__main__":
    main()

