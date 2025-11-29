import json
import time
import datetime
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import config_loader
import monitor


# New named baseline system
def get_baselines_dir(config_path: Optional[Path] = None) -> Path:
    """Get the baselines directory path."""
    config = config_loader.load_config(config_path)
    data_dir = Path(config["event_detection"]["baseline_file"]).parent
    baselines_dir = data_dir / "baselines"
    baselines_dir.mkdir(parents=True, exist_ok=True)
    return baselines_dir


def get_baselines_index(config_path: Optional[Path] = None) -> Dict:
    """Load or create the baselines index."""
    baselines_dir = get_baselines_dir(config_path)
    index_file = baselines_dir / "baselines_index.json"
    
    if index_file.exists():
        try:
            with index_file.open() as f:
                return json.load(f)
        except Exception:
            pass
    
    # Default index
    return {
        "active": "default",
        "baselines": {}
    }


def save_baselines_index(index: Dict, config_path: Optional[Path] = None):
    """Save the baselines index."""
    baselines_dir = get_baselines_dir(config_path)
    index_file = baselines_dir / "baselines_index.json"
    
    with index_file.open("w") as f:
        json.dump(index, f, indent=2)


def get_baseline_file(baseline_name: str, config_path: Optional[Path] = None) -> Path:
    """Get the file path for a named baseline."""
    baselines_dir = get_baselines_dir(config_path)
    # Sanitize name for filesystem
    safe_name = "".join(c for c in baseline_name if c.isalnum() or c in ('-', '_'))
    return baselines_dir / f"{safe_name}.json"


def migrate_old_baseline(config_path: Optional[Path] = None) -> bool:
    """Migrate old baseline.json to named baseline system. Returns True if migration occurred."""
    config = config_loader.load_config(config_path)
    old_baseline_file = Path(config["event_detection"]["baseline_file"])
    
    if not old_baseline_file.exists():
        return False
    
    index = get_baselines_index(config_path)
    
    # Check if "default" already exists
    if "default" in index["baselines"]:
        return False
    
    # Load old baseline
    try:
        with old_baseline_file.open() as f:
            data = json.load(f)
        
        # Convert to history format if needed
        if isinstance(data, dict):
            history = [data]
        elif isinstance(data, list):
            history = data
        else:
            return False
        
        if not history:
            return False
        
        # Save as "default" baseline
        baseline_file = get_baseline_file("default", config_path)
        with baseline_file.open("w") as f:
            json.dump(history, f, indent=2)
        
        # Update index
        latest = history[-1]
        index["baselines"]["default"] = {
            "name": "default",
            "created": latest.get("timestamp", datetime.datetime.now().isoformat()),
            "updated": latest.get("timestamp", datetime.datetime.now().isoformat()),
            "rms_db": latest.get("rms_db"),
            "description": "Migrated from old baseline.json"
        }
        index["active"] = "default"
        save_baselines_index(index, config_path)
        
        print(f"[INFO] Migrated old baseline.json to named baseline 'default'")
        return True
    except Exception as e:
        print(f"[WARN] Failed to migrate old baseline: {e}")
        return False


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
    
    baseline_file.parent.mkdir(parents=True, exist_ok=True)
    with baseline_file.open("w") as f:
        json.dump(history, f, indent=2)


def list_baselines(config_path: Optional[Path] = None):
    """List all available baselines."""
    migrate_old_baseline(config_path)  # Auto-migrate if needed
    
    index = get_baselines_index(config_path)
    active = index.get("active", "default")
    baselines = index.get("baselines", {})
    
    if not baselines:
        print("No baselines found. Create one with: python3 baseline.py create <name>")
        return
    
    print("\nAvailable Baselines:")
    print("=" * 60)
    for name, info in sorted(baselines.items()):
        marker = " *" if name == active else ""
        rms = info.get("rms_db")
        rms_str = f"{rms:.1f} dBFS" if isinstance(rms, (int, float)) and np.isfinite(rms) else "N/A"
        desc = info.get("description", "")
        updated = info.get("updated", "N/A")
        print(f"  {name}{marker} - {rms_str} - {desc}")
        print(f"    Updated: {updated}")
    
    print()
    print(f"Active baseline: {active}")
    print()


def create_baseline(name: str, duration_sec: int = 10, description: str = "", config_path: Optional[Path] = None):
    """Create a new named baseline."""
    migrate_old_baseline(config_path)  # Auto-migrate if needed
    
    if not name:
        print("[ERROR] Baseline name is required")
        return
    
    # Sanitize name
    safe_name = "".join(c for c in name if c.isalnum() or c in ('-', '_'))
    if safe_name != name:
        print(f"[WARN] Baseline name sanitized: '{name}' -> '{safe_name}'")
        name = safe_name
    
    config = config_loader.load_config(config_path)
    
    print(f"\nCollecting {duration_sec} seconds for baseline '{name}'...")
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
            
            # Fix: If all samples are zeros, skip this chunk (likely silent/invalid)
            if np.all(samples == 0):
                continue
            
            peak = float(np.max(np.abs(samples)))
            rms = float(np.sqrt(np.mean(samples ** 2)))
            
            peak_db = monitor.dbfs(peak)
            rms_db = monitor.dbfs(rms)
            
            # Don't allow NaN or inf values into arrays
            if not (np.isfinite(peak_db) and np.isfinite(rms_db)):
                continue
            
            peak_vals.append(peak_db)
            rms_vals.append(rms_db)
            
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
        "rms_db": float(np.mean(rms_vals)) if rms_vals else float('nan'),
        "peak_db": float(np.mean(peak_vals)) if peak_vals else float('nan'),
        "rms_std": float(np.std(rms_vals)) if rms_vals else float('nan'),
        "rms_min": float(np.min(rms_vals)) if rms_vals else float('nan'),
        "rms_max": float(np.max(rms_vals)) if rms_vals else float('nan'),
    }
    
    # Save baseline history
    baseline_file = get_baseline_file(name, config_path)
    history = load_baseline_history(baseline_file)
    history.append(baseline_data)
    save_baseline_history(baseline_file, history)
    
    # Update index
    index = get_baselines_index(config_path)
    now = datetime.datetime.now().isoformat()
    
    if name in index["baselines"]:
        index["baselines"][name]["updated"] = now
        index["baselines"][name]["rms_db"] = baseline_data["rms_db"]
        if description:
            index["baselines"][name]["description"] = description
    else:
        index["baselines"][name] = {
            "name": name,
            "created": now,
            "updated": now,
            "rms_db": baseline_data["rms_db"],
            "description": description or f"Baseline '{name}'"
        }
    
    save_baselines_index(index, config_path)
    
    print(f"\nBaseline '{name}' saved:")
    print(f"  RMS: {baseline_data['rms_db']:.1f} dBFS (std: {baseline_data['rms_std']:.2f})")
    print(f"  Peak: {baseline_data['peak_db']:.1f} dBFS")
    print(f"  Range: {baseline_data['rms_min']:.1f} to {baseline_data['rms_max']:.1f} dBFS")
    print()


def delete_baseline(name: str, config_path: Optional[Path] = None):
    """Delete a named baseline."""
    index = get_baselines_index(config_path)
    
    if name not in index["baselines"]:
        print(f"[ERROR] Baseline '{name}' not found")
        return
    
    if index.get("active") == name:
        print(f"[ERROR] Cannot delete active baseline '{name}'. Switch to another baseline first.")
        return
    
    # Delete file
    baseline_file = get_baseline_file(name, config_path)
    if baseline_file.exists():
        baseline_file.unlink()
    
    # Remove from index
    del index["baselines"][name]
    save_baselines_index(index, config_path)
    
    print(f"Baseline '{name}' deleted")


def switch_baseline(name: str, config_path: Optional[Path] = None):
    """Switch to a different baseline."""
    index = get_baselines_index(config_path)
    
    if name not in index["baselines"]:
        print(f"[ERROR] Baseline '{name}' not found")
        print(f"Available baselines: {', '.join(index['baselines'].keys())}")
        return
    
    index["active"] = name
    save_baselines_index(index, config_path)
    
    info = index["baselines"][name]
    rms = info.get("rms_db")
    rms_str = f"{rms:.1f} dBFS" if isinstance(rms, (int, float)) and np.isfinite(rms) else "N/A"
    
    print(f"Switched to baseline '{name}' ({rms_str})")
    print("Note: Restart the noise-monitor service for changes to take effect:")
    print("  make restart")


def show_baseline(baseline_name: Optional[str] = None, config_path: Optional[Path] = None):
    """Show baseline information."""
    migrate_old_baseline(config_path)  # Auto-migrate if needed
    
    index = get_baselines_index(config_path)
    
    # Use provided name, active baseline, or "default"
    if not baseline_name:
        baseline_name = index.get("active", "default")
    
    if baseline_name not in index.get("baselines", {}):
        print(f"[ERROR] Baseline '{baseline_name}' not found")
        return
    
    baseline_file = get_baseline_file(baseline_name, config_path)
    history = load_baseline_history(baseline_file)
    
    if not history:
        print(f"No baseline data found for '{baseline_name}'.\n")
        return
    
    latest = history[-1]
    info = index["baselines"][baseline_name]
    is_active = index.get("active") == baseline_name
    
    print(f"\nBaseline: {baseline_name}{' (ACTIVE)' if is_active else ''}")
    print(f"Description: {info.get('description', 'N/A')}")
    print(f"Created: {info.get('created', 'N/A')}")
    print(f"Updated: {info.get('updated', 'N/A')}")
    print(f"History entries: {len(history)}")
    print()
    print("Latest Measurement:")
    print(f"  Timestamp: {latest.get('timestamp', 'N/A')}")
    rms_db = latest.get('rms_db')
    print(f"  RMS: {rms_db:.1f} dBFS" if isinstance(rms_db, (int, float)) and np.isfinite(rms_db) else "  RMS: N/A")
    if 'rms_std' in latest and isinstance(latest['rms_std'], (int, float)) and np.isfinite(latest['rms_std']):
        print(f"  RMS Std Dev: {latest['rms_std']:.2f} dBFS")
    peak_db = latest.get('peak_db')
    print(f"  Peak: {peak_db:.1f} dBFS" if isinstance(peak_db, (int, float)) and np.isfinite(peak_db) else f"  Peak: {peak_db if peak_db is not None else 'N/A'} dBFS")
    print()


def analyze_baseline(baseline_name: Optional[str] = None, config_path: Optional[Path] = None):
    """Analyze baseline statistics and history."""
    migrate_old_baseline(config_path)  # Auto-migrate if needed
    
    index = get_baselines_index(config_path)
    
    if not baseline_name:
        baseline_name = index.get("active", "default")
    
    if baseline_name not in index.get("baselines", {}):
        print(f"[ERROR] Baseline '{baseline_name}' not found")
        return
    
    baseline_file = get_baseline_file(baseline_name, config_path)
    history = load_baseline_history(baseline_file)
    
    if not history:
        print(f"No baseline history found for '{baseline_name}'.\n")
        return
    
    print(f"\nBaseline '{baseline_name}' History ({len(history)} entries):")
    print()
    
    # Only include valid (finite) values in the statistics
    rms_values = [b["rms_db"] for b in history if "rms_db" in b and np.isfinite(b["rms_db"])]
    if not rms_values:
        print("No valid RMS values found in baseline history.")
        print()
        return
    
    print(f"RMS Statistics:")
    print(f"  Mean: {np.mean(rms_values):.1f} dBFS")
    print(f"  Std Dev: {np.std(rms_values):.2f} dBFS")
    print(f"  Min: {np.min(rms_values):.1f} dBFS")
    print(f"  Max: {np.max(rms_values):.1f} dBFS")
    print(f"  Range: {np.max(rms_values) - np.min(rms_values):.1f} dBFS")
    print()
    
    # Check for stability
    if len(rms_values) > 1:
        recent_std = np.std(rms_values[-10:]) if len(rms_values) >= 10 else np.std(rms_values)
        if recent_std > 3.0:
            print("  WARNING: Baseline appears unstable (high variance)")
        elif recent_std > 1.5:
            print("  CAUTION: Baseline shows moderate variance")
        else:
            print("  Baseline appears stable")
    print()


def validate_baseline(baseline_name: Optional[str] = None, config_path: Optional[Path] = None) -> bool:
    """Validate if baseline is reliable."""
    migrate_old_baseline(config_path)  # Auto-migrate if needed
    
    index = get_baselines_index(config_path)
    
    if not baseline_name:
        baseline_name = index.get("active", "default")
    
    if baseline_name not in index.get("baselines", {}):
        print(f"No baseline '{baseline_name}' found - not reliable")
        return False
    
    baseline_file = get_baseline_file(baseline_name, config_path)
    history = load_baseline_history(baseline_file)
    
    if not history:
        print(f"No baseline data found for '{baseline_name}' - not reliable")
        return False
    
    latest = history[-1]
    
    # Check if baseline has reasonable statistics
    if "rms_std" in latest and isinstance(latest["rms_std"], (float, int)) and np.isfinite(latest["rms_std"]):
        if latest["rms_std"] > 5.0:
            print(f"Baseline has high variance (std={latest['rms_std']:.2f}) - may not be reliable")
            return False
    
    # Check if baseline is too quiet or too loud (might indicate measurement error)
    rms_db = latest.get("rms_db")
    if rms_db is None or not np.isfinite(rms_db):
        print("Baseline RMS missing or invalid")
        return False
    
    if rms_db < -80:
        print(f"Baseline very quiet ({rms_db:.1f} dBFS) - may indicate measurement issue")
        return False
    
    if rms_db > -20:
        print(f"Baseline very loud ({rms_db:.1f} dBFS) - may indicate measurement issue")
        return False
    
    print(f"Baseline '{baseline_name}' appears valid")
    return True


# Backward compatibility
def set_baseline(duration_sec=10, config_path: Optional[Path] = None):
    """Set baseline (backward compatibility - creates/updates 'default' baseline)."""
    create_baseline("default", duration_sec, "Default baseline", config_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Baseline management")
    parser.add_argument("command", choices=["list", "create", "delete", "switch", "show", "analyze", "validate", "set"], 
                       help="Command to run")
    parser.add_argument("name", nargs="?", help="Baseline name (for create/delete/switch/show/analyze/validate)")
    parser.add_argument("--config", type=Path, help="Path to config.json")
    parser.add_argument("--duration", type=int, default=10, help="Duration for create/set command (seconds)")
    parser.add_argument("--description", type=str, default="", help="Description for create command")
    
    args = parser.parse_args()
    
    if args.command == "list":
        list_baselines(args.config)
    elif args.command == "create":
        if not args.name:
            print("[ERROR] Baseline name is required for 'create' command")
            sys.exit(1)
        create_baseline(args.name, args.duration, args.description, args.config)
    elif args.command == "delete":
        if not args.name:
            print("[ERROR] Baseline name is required for 'delete' command")
            sys.exit(1)
        delete_baseline(args.name, args.config)
    elif args.command == "switch":
        if not args.name:
            print("[ERROR] Baseline name is required for 'switch' command")
            sys.exit(1)
        switch_baseline(args.name, args.config)
    elif args.command == "show":
        show_baseline(args.name, args.config)
    elif args.command == "analyze":
        analyze_baseline(args.name, args.config)
    elif args.command == "validate":
        validate_baseline(args.name, args.config)
    elif args.command == "set":
        set_baseline(args.duration, args.config)
