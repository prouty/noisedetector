#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
from datetime import datetime

EVENTS_FILE = Path("events.csv")


def load_events():
    if not EVENTS_FILE.exists():
        print(f"No {EVENTS_FILE} found.")
        return None

    df = pd.read_csv(EVENTS_FILE)
    # Normalize column names just in case
    df.columns = [c.strip() for c in df.columns]
    return df


def add_date_column(df):
    # assume ISO-ish timestamps like 2025-11-23T12:43:55
    df["date"] = df["start_timestamp"].str.slice(0, 10)
    return df


def choose_latest_date(df):
    dates = sorted(df["date"].unique())
    if not dates:
        return None
    return dates[-1]


def build_report(df, report_date):
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


def main():
    df = load_events()
    if df is None:
        return

    if "start_timestamp" not in df.columns:
        print("events.csv missing 'start_timestamp' column.")
        return

    df = add_date_column(df)
    report_date = choose_latest_date(df)
    if report_date is None:
        print("No dates found in events.csv.")
        return

    report_md = build_report(df, report_date)

    out_name = f"chirp_report_{report_date}.md"
    out_path = Path(out_name)
    out_path.write_text(report_md, encoding="utf-8")

    print(f"Wrote report to {out_path.resolve()}")


if __name__ == "__main__":
    main()
