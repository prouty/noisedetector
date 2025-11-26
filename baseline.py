import json
import time
import datetime
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

import config_loader
import monitor


def load_baseline_history(baseline_file: Path) -> List[Dict]:
    """Load baseline history from file."""
    if not baseline_file.exists():
        return []
    
    try:
        with baseline_file.open() as f:
            data = json.load(f)
        # Support both single baseline and history array
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        return []
    except Exception:
        return []


def save_baseline_history(baseline_file: Path, history: List[Dict], max_history: int = 100):
    """Save baseline history, keeping only recent entries."""
    if len(history) > max_history:
        history = history[-max_history:]
    
    with baseline_file.open("w") as f:
        json.dump(history, f, indent=2)


def set_baseline(duration_sec=10, config_path: Optional[Path] = None):
    """Set baseline noise level."""
    config = config_loader.load_config(config_path)
    baseline_file = Path(config["event_detection"]["baseline_file"])
    
    print(f"\nCollecting {duration_sec} seconds for baseline...")
    rms_vals = []
    peak_vals = []

    # Use monitor's audio setup
    proc = monitor.start_arecord(config)
    if proc.stdout is None:
        print("[ERROR] Failed to open arecord stdout.")
        return

    audio_cfg = config["audio"]
    chunk_samples = int(audio_cfg["sample_rate"] * audio_cfg["chunk_duration"])
    chunk_bytes = chunk_samples * monitor.BYTES_PER_SAMPLE * audio_cfg["channels"]

    start = time.time()

    try:
        while True:
            data = proc.stdout.read(chunk_bytes)
            if not data or len(data) < chunk_bytes:
                break

            samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / monitor.INT16_FULL_SCALE
            if len(samples) == 0:
                continue

            peak = float(np.max(np.abs(samples)))
            rms = float(np.sqrt(np.mean(samples ** 2)))

            peak_vals.append(monitor.dbfs(peak))
            rms_vals.append(monitor.dbfs(rms))

            if time.time() - start >= duration_sec:
                break
    finally:
        if proc and proc.poll() is None:
            proc.terminate()

    if not rms_vals:
        print("[ERROR] No audio data collected")
        return

    baseline_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "rms_db": float(np.mean(rms_vals)),
        "peak_db": float(np.mean(peak_vals)),
        "rms_std": float(np.std(rms_vals)),
        "rms_min": float(np.min(rms_vals)),
        "rms_max": float(np.max(rms_vals)),
    }

    # Load existing history and append
    history = load_baseline_history(baseline_file)
    history.append(baseline_data)
    save_baseline_history(baseline_file, history)

    print("\nBaseline saved:")
    print(f"  RMS: {baseline_data['rms_db']:.1f} dBFS (std: {baseline_data['rms_std']:.2f})")
    print(f"  Peak: {baseline_data['peak_db']:.1f} dBFS")
    print(f"  Range: {baseline_data['rms_min']:.1f} to {baseline_data['rms_max']:.1f} dBFS")
    print()


def show_baseline(config_path: Optional[Path] = None):
    """Show current baseline."""
    config = config_loader.load_config(config_path)
    baseline_file = Path(config["event_detection"]["baseline_file"])
    
    history = load_baseline_history(baseline_file)
    if not history:
        print("No baseline set yet.\n")
        return
    
    latest = history[-1]
    print("\nLast Baseline:")
    print(f"  Timestamp: {latest.get('timestamp', 'N/A')}")
    print(f"  RMS: {latest['rms_db']:.1f} dBFS")
    if 'rms_std' in latest:
        print(f"  RMS Std Dev: {latest['rms_std']:.2f} dBFS")
    print(f"  Peak: {latest.get('peak_db', 'N/A')} dBFS")
    print()


def analyze_baseline(config_path: Optional[Path] = None):
    """Analyze baseline statistics and history."""
    config = config_loader.load_config(config_path)
    baseline_file = Path(config["event_detection"]["baseline_file"])
    
    history = load_baseline_history(baseline_file)
    if not history:
        print("No baseline history found.\n")
        return
    
    print(f"\nBaseline History ({len(history)} entries):")
    print()
    
    rms_values = [b["rms_db"] for b in history]
    print(f"RMS Statistics:")
    print(f"  Mean: {np.mean(rms_values):.1f} dBFS")
    print(f"  Std Dev: {np.std(rms_values):.2f} dBFS")
    print(f"  Min: {np.min(rms_values):.1f} dBFS")
    print(f"  Max: {np.max(rms_values):.1f} dBFS")
    print(f"  Range: {np.max(rms_values) - np.min(rms_values):.1f} dBFS")
    print()
    
    # Check for stability
    if len(history) > 1:
        recent_std = np.std(rms_values[-10:]) if len(rms_values) >= 10 else np.std(rms_values)
        if recent_std > 3.0:
            print("  WARNING: Baseline appears unstable (high variance)")
        elif recent_std > 1.5:
            print("  CAUTION: Baseline shows moderate variance")
        else:
            print("  Baseline appears stable")
    print()


def validate_baseline(config_path: Optional[Path] = None) -> bool:
    """Validate if current baseline is reliable."""
    config = config_loader.load_config(config_path)
    baseline_file = Path(config["event_detection"]["baseline_file"])
    
    history = load_baseline_history(baseline_file)
    if not history:
        print("No baseline found - not reliable")
        return False
    
    latest = history[-1]
    
    # Check if baseline has reasonable statistics
    if "rms_std" in latest:
        if latest["rms_std"] > 5.0:
            print(f"Baseline has high variance (std={latest['rms_std']:.2f}) - may not be reliable")
            return False
    
    # Check if baseline is too quiet or too loud (might indicate measurement error)
    if latest["rms_db"] < -80:
        print(f"Baseline very quiet ({latest['rms_db']:.1f} dBFS) - may indicate measurement issue")
        return False
    
    if latest["rms_db"] > -20:
        print(f"Baseline very loud ({latest['rms_db']:.1f} dBFS) - may indicate measurement issue")
        return False
    
    print("Baseline appears valid")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Baseline management")
    parser.add_argument("command", choices=["set", "show", "analyze", "validate"], help="Command to run")
    parser.add_argument("--config", type=Path, help="Path to config.json")
    parser.add_argument("--duration", type=int, default=10, help="Duration for set command (seconds)")
    
    args = parser.parse_args()
    
    if args.command == "set":
        set_baseline(args.duration, args.config)
    elif args.command == "show":
        show_baseline(args.config)
    elif args.command == "analyze":
        analyze_baseline(args.config)
    elif args.command == "validate":
        validate_baseline(args.config)
