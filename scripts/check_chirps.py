#!/usr/bin/env python3
"""Quick check for detected chirps in events.csv."""
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.reporting import load_events


def check_chirps(events_file: Path = Path("data/events.csv"), recent_hours: Optional[int] = None):
    """Check for chirps in events.csv and show summary."""
    df = load_events(events_file)
    if df.empty:
        print("No events found in events.csv")
        return
    
    # Filter to chirps
    if "is_chirp" not in df.columns:
        print("No 'is_chirp' column found in events.csv")
        return
    
    chirps = df[df["is_chirp"].astype(str).str.upper() == "TRUE"]
    total_events = len(df)
    total_chirps = len(chirps)
    
    print("=" * 60)
    print("CHIRP DETECTION SUMMARY")
    print("=" * 60)
    print(f"Total events: {total_events}")
    print(f"Chirps detected: {total_chirps}")
    print(f"Non-chirps: {total_events - total_chirps}")
    
    if total_chirps > 0:
        chirp_rate = (total_chirps / total_events) * 100
        print(f"Chirp rate: {chirp_rate:.1f}%")
        print()
        
        # Show most recent chirps
        if "start_timestamp" in chirps.columns:
            chirps_sorted = chirps.sort_values("start_timestamp", ascending=False)
            
            if recent_hours:
                # Filter to recent
                now = datetime.now()
                recent_cutoff = now.timestamp() - (recent_hours * 3600)
                try:
                    chirps_sorted["timestamp_parsed"] = pd.to_datetime(chirps_sorted["start_timestamp"])
                    recent_chirps = chirps_sorted[
                        chirps_sorted["timestamp_parsed"].apply(lambda x: x.timestamp()) > recent_cutoff
                    ]
                    if len(recent_chirps) > 0:
                        print(f"Recent chirps (last {recent_hours} hours): {len(recent_chirps)}")
                        print()
                        show_count = min(5, len(recent_chirps))
                        print(f"Most recent {show_count} chirps:")
                        for idx, row in recent_chirps.head(show_count).iterrows():
                            timestamp = row.get("start_timestamp", "N/A")
                            similarity = row.get("chirp_similarity", "")
                            confidence = row.get("confidence", "")
                            clip = row.get("clip_file", "")
                            
                            sim_str = f"sim={similarity}" if similarity else ""
                            conf_str = f"conf={confidence}" if confidence else ""
                            info = " ".join(filter(None, [sim_str, conf_str]))
                            
                            print(f"  {timestamp}  {info}")
                            if clip:
                                print(f"    {clip}")
                    else:
                        print(f"No chirps in the last {recent_hours} hours")
                except Exception:
                    # Fallback if timestamp parsing fails
                    pass
            
            # Show top chirps by similarity
            if "chirp_similarity" in chirps_sorted.columns:
                try:
                    chirps_sorted["similarity_float"] = pd.to_numeric(
                        chirps_sorted["chirp_similarity"], errors="coerce"
                    )
                    top_chirps = chirps_sorted.nlargest(5, "similarity_float")
                    print()
                    print("Top 5 chirps by similarity:")
                    for idx, row in top_chirps.iterrows():
                        timestamp = row.get("start_timestamp", "N/A")
                        similarity = row.get("chirp_similarity", "")
                        confidence = row.get("confidence", "")
                        sim_str = f"sim={similarity}" if similarity else ""
                        conf_str = f"conf={confidence}" if confidence else ""
                        info = " ".join(filter(None, [sim_str, conf_str]))
                        print(f"  {timestamp}  {info}")
                except Exception:
                    pass
        print()
    else:
        print()
        print("No chirps detected yet.")
        print()
    
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick check for detected chirps")
    parser.add_argument("--events", type=Path, default=Path("data/events.csv"), help="Path to events.csv")
    parser.add_argument("--recent", type=int, help="Show chirps from last N hours")
    
    args = parser.parse_args()
    check_chirps(args.events, args.recent)

