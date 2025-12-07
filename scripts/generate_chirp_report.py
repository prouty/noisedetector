#!/usr/bin/env python3
"""
Generate markdown chirp report for a specific date.

This script is a thin CLI wrapper around core.reporting module.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.reporting import load_events, add_date_column, choose_latest_date, generate_chirp_report

EVENTS_FILE = Path("data/events.csv")


# build_report is now in core.reporting as generate_chirp_report


def main():
    df = load_events(EVENTS_FILE)
    if df.empty:
        print(f"No {EVENTS_FILE} found.")
        return

    if "start_timestamp" not in df.columns:
        print("events.csv missing 'start_timestamp' column.")
        return

    df = add_date_column(df)
    report_date = choose_latest_date(df)
    if report_date is None:
        print("No dates found in events.csv.")
        return

    report_md = generate_chirp_report(df, report_date)

    # Write to reports/ directory
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    out_name = f"chirp_report_{report_date}.md"
    out_path = reports_dir / out_name
    out_path.write_text(report_md, encoding="utf-8")

    print(f"Wrote report to {out_path.resolve()}")


if __name__ == "__main__":
    main()
