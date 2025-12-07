#!/usr/bin/env python3
"""
Debug state dump utility - saves current system state for analysis.

Use this when the system is behaving unexpectedly. It captures:
- Configuration
- Baseline status
- Recent events
- File system state
- Audio device status

Run: python3 debug_state.py [output_file.json]
"""
import json
import sys
from pathlib import Path
from datetime import datetime
import subprocess

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
import config_loader


def dump_system_state(output_file: Path = Path("debug_state.json")):
    """Dump complete system state for debugging."""
    state = {
        "timestamp": datetime.now().isoformat(),
        "config": {},
        "baseline": {},
        "events_summary": {},
        "filesystem": {},
        "audio": {},
        "errors": []
    }
    
    # Load config
    try:
        config = config_loader.load_config()
        state["config"] = config
    except Exception as e:
        state["errors"].append(f"Config load failed: {e}")
        state["config"] = None
    
    # Check baseline
    baseline_file = Path("baseline.json")
    if baseline_file.exists():
        try:
            import json as json_lib
            with baseline_file.open() as f:
                baseline_data = json_lib.load(f)
            state["baseline"] = {
                "exists": True,
                "data": baseline_data if isinstance(baseline_data, dict) else {"history_count": len(baseline_data) if isinstance(baseline_data, list) else 0}
            }
        except Exception as e:
            state["baseline"] = {"exists": True, "error": str(e)}
    else:
        state["baseline"] = {"exists": False}
    
    # Check events.csv
    events_file = Path("events.csv")
    if events_file.exists():
        try:
            from core.reporting import load_events
            df = load_events(events_file)
            state["events_summary"] = {
                "total_events": len(df),
                "chirps": len(df[df["is_chirp"].astype(str).str.upper() == "TRUE"]) if "is_chirp" in df.columns else 0,
                "latest_event": df.iloc[-1].to_dict() if len(df) > 0 else None,
                "file_size_bytes": events_file.stat().st_size
            }
        except Exception as e:
            state["events_summary"] = {"error": str(e)}
    else:
        state["events_summary"] = {"exists": False}
    
    # Filesystem state
    try:
        import shutil
        stat = shutil.disk_usage(Path.cwd())
        state["filesystem"] = {
            "current_dir": str(Path.cwd()),
            "disk_total_gb": stat.total / (1024**3),
            "disk_used_gb": stat.used / (1024**3),
            "disk_free_gb": stat.free / (1024**3),
            "disk_free_percent": (stat.free / stat.total) * 100
        }
        
        # Check key directories
        clips_dir = Path("clips")
        state["filesystem"]["clips_dir"] = {
            "exists": clips_dir.exists(),
            "writable": clips_dir.exists() and clips_dir.is_dir() and (clips_dir / ".test").touch() and (clips_dir / ".test").unlink() or False,
            "file_count": len(list(clips_dir.glob("*.wav"))) if clips_dir.exists() else 0
        }
    except Exception as e:
        state["filesystem"] = {"error": str(e)}
    
    # Audio device check
    try:
        result = subprocess.run(
            ["arecord", "-l"],
            capture_output=True,
            text=True,
            timeout=5
        )
        state["audio"] = {
            "arecord_available": result.returncode == 0,
            "device_list": result.stdout if result.returncode == 0 else result.stderr
        }
    except FileNotFoundError:
        state["audio"] = {"arecord_available": False, "error": "arecord not found"}
    except Exception as e:
        state["audio"] = {"error": str(e)}
    
    # Save state
    try:
        with output_file.open("w") as f:
            json.dump(state, f, indent=2, default=str)
        print(f"System state saved to {output_file}")
        print(f"  Errors encountered: {len(state['errors'])}")
        if state["errors"]:
            print("  Errors:")
            for err in state["errors"]:
                print(f"    - {err}")
    except Exception as e:
        print(f"Failed to save state file: {e}")
        # Print to stdout as fallback
        print("\n" + "=" * 60)
        print("SYSTEM STATE (JSON)")
        print("=" * 60)
        print(json.dumps(state, indent=2, default=str))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Dump system state for debugging")
    parser.add_argument("output", nargs="?", type=Path, default=Path("debug_state.json"),
                       help="Output file path (default: debug_state.json)")
    
    args = parser.parse_args()
    dump_system_state(args.output)

