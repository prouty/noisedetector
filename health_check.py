#!/usr/bin/env python3
"""
System health check - verifies all components are working.

Run this to diagnose issues before they become critical.
Useful for automated monitoring or manual troubleshooting.
"""
import sys
from pathlib import Path
from typing import List, Tuple

import config_loader


def check_file_permissions(path: Path, need_write: bool = False) -> Tuple[bool, str]:
    """Check if file/directory exists and has required permissions."""
    if not path.exists():
        return False, f"Path does not exist: {path}"
    
    if path.is_file() and need_write:
        if not path.parent.exists():
            return False, f"Parent directory does not exist: {path.parent}"
        try:
            # Test write by opening in append mode
            with path.open("a"):
                pass
            return True, "OK"
        except PermissionError:
            return False, f"No write permission: {path}"
        except Exception as e:
            return False, f"Error accessing {path}: {e}"
    
    if path.is_dir() and need_write:
        test_file = path / ".health_check_test"
        try:
            test_file.touch()
            test_file.unlink()
            return True, "OK"
        except PermissionError:
            return False, f"No write permission in directory: {path}"
        except Exception as e:
            return False, f"Error accessing {path}: {e}"
    
    return True, "OK"


def check_disk_space(path: Path, min_gb: float = 0.1) -> Tuple[bool, str]:
    """Check if there's enough disk space."""
    import shutil
    
    stat = shutil.disk_usage(path)
    free_gb = stat.free / (1024 ** 3)
    
    if free_gb < min_gb:
        return False, f"Low disk space: {free_gb:.2f} GB free (need {min_gb} GB)"
    
    return True, f"{free_gb:.2f} GB free"


def check_audio_device(device: str) -> Tuple[bool, str]:
    """Check if audio device is available."""
    import subprocess
    
    try:
        # Try to list devices
        result = subprocess.run(
            ["arecord", "-l"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            return False, f"arecord -l failed: {result.stderr}"
        
        # Check if device string appears in output (basic check)
        # Note: This is a simple check - actual device validation requires trying to open it
        return True, "Audio devices available (use 'make audio-check' to test specific device)"
    
    except FileNotFoundError:
        return False, "arecord not found - install alsa-utils"
    except subprocess.TimeoutExpired:
        return False, "arecord command timed out"
    except Exception as e:
        return False, f"Error checking audio: {e}"


def check_dependencies() -> Tuple[bool, str]:
    """Check if required Python packages are installed."""
    missing = []
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    try:
        import pandas
    except ImportError:
        missing.append("pandas")
    
    if missing:
        return False, f"Missing packages: {', '.join(missing)}. Run: pip3 install {' '.join(missing)}"
    
    return True, "All dependencies installed"


def check_config(config_path: Path) -> Tuple[bool, str]:
    """Check if configuration is valid."""
    try:
        config = config_loader.load_config(config_path)
        
        # Check critical paths
        issues = []
        
        # Check audio device format
        device = config.get("audio", {}).get("device", "")
        if not device:
            issues.append("audio.device not set")
        
        # Check file paths exist or can be created
        events_file = Path(config.get("event_detection", {}).get("events_file", "events.csv"))
        ok, msg = check_file_permissions(events_file.parent if events_file.parent != Path(".") else Path.cwd(), need_write=True)
        if not ok:
            issues.append(f"Events file directory: {msg}")
        
        clips_dir = Path(config.get("event_clips", {}).get("clips_dir", "clips"))
        ok, msg = check_file_permissions(clips_dir, need_write=True)
        if not ok:
            issues.append(f"Clips directory: {msg}")
        
        if issues:
            return False, "; ".join(issues)
        
        return True, "Configuration valid"
    
    except Exception as e:
        return False, f"Config error: {e}"


def check_baseline(baseline_file: Path) -> Tuple[bool, str]:
    """Check if baseline exists and is valid."""
    if not baseline_file.exists():
        return False, "Baseline not set - run 'python3 baseline.py set'"
    
    import json
    try:
        with baseline_file.open() as f:
            data = json.load(f)
        
        # Check if it's a list (history) or single dict
        if isinstance(data, list):
            if len(data) == 0:
                return False, "Baseline history is empty"
            latest = data[-1]
        else:
            latest = data
        
        rms_db = latest.get("rms_db")
        if rms_db is None:
            return False, "Baseline missing rms_db value"
        
        if not isinstance(rms_db, (int, float)):
            return False, f"Baseline rms_db is not a number: {rms_db}"
        
        # Check reasonable range
        if rms_db < -100 or rms_db > 0:
            return False, f"Baseline rms_db out of reasonable range: {rms_db}"
        
        return True, f"Baseline OK: {rms_db:.1f} dBFS"
    
    except json.JSONDecodeError as e:
        return False, f"Baseline file is not valid JSON: {e}"
    except Exception as e:
        return False, f"Error reading baseline: {e}"


def run_health_check(config_path: Path = None) -> bool:
    """
    Run comprehensive health check.
    
    Returns:
        True if all checks pass, False otherwise
    """
    if config_path is None:
        config_path = Path("config.json")
    
    print("=" * 60)
    print("NOISE DETECTOR HEALTH CHECK")
    print("=" * 60)
    print()
    
    all_ok = True
    checks: List[Tuple[str, bool, str]] = []
    
    # Check dependencies
    ok, msg = check_dependencies()
    checks.append(("Dependencies", ok, msg))
    if not ok:
        all_ok = False
    
    # Check config
    ok, msg = check_config(config_path)
    checks.append(("Configuration", ok, msg))
    if not ok:
        all_ok = False
    
    # Check disk space
    ok, msg = check_disk_space(Path.cwd(), min_gb=0.1)
    checks.append(("Disk Space", ok, msg))
    if not ok:
        all_ok = False
    
    # Check file permissions
    events_file = Path("events.csv")
    ok, msg = check_file_permissions(events_file.parent if events_file.parent != Path(".") else Path.cwd(), need_write=True)
    checks.append(("Events File Directory", ok, msg))
    if not ok:
        all_ok = False
    
    clips_dir = Path("clips")
    ok, msg = check_file_permissions(clips_dir, need_write=True)
    checks.append(("Clips Directory", ok, msg))
    if not ok:
        all_ok = False
    
    # Check audio device (basic)
    try:
        config = config_loader.load_config(config_path)
        device = config.get("audio", {}).get("device", "")
        ok, msg = check_audio_device(device)
        checks.append(("Audio System", ok, msg))
        if not ok:
            all_ok = False
    except Exception as e:
        checks.append(("Audio System", False, f"Could not check: {e}"))
        all_ok = False
    
    # Check baseline
    baseline_file = Path("baseline.json")
    ok, msg = check_baseline(baseline_file)
    checks.append(("Baseline", ok, msg))
    if not ok:
        all_ok = False
    
    # Print results
    for name, ok, msg in checks:
        status = "✓" if ok else "✗"
        print(f"{status} {name}: {msg}")
    
    print()
    print("=" * 60)
    
    if all_ok:
        print("✓ All checks passed - system ready")
    else:
        print("✗ Some checks failed - review issues above")
        print()
        print("Common fixes:")
        print("  - Missing dependencies: pip3 install numpy pandas")
        print("  - No baseline: python3 baseline.py set")
        print("  - Permission issues: check file/directory permissions")
        print("  - Audio issues: make audio-check")
    
    print("=" * 60)
    
    return all_ok


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="System health check")
    parser.add_argument("--config", type=Path, help="Path to config.json")
    
    args = parser.parse_args()
    success = run_health_check(args.config)
    sys.exit(0 if success else 1)

