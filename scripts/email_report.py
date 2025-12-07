#!/usr/bin/env python3
"""
Generate and email a summary report of clips and chirps from the last 2 hours.

This script is a thin CLI wrapper around core.reporting and core.email modules.
"""
import sys
from pathlib import Path
import argparse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
import config_loader
from core.reporting import load_events, filter_recent_events, generate_email_report
from core.email import get_email_config, send_email


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
        report = generate_email_report(df_recent, args.hours)
    
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

