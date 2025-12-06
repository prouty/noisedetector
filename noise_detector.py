#!/usr/bin/env python3

import sys
import argparse
from pathlib import Path
import baseline
import sampler
import monitor
import monitor_refactored

MENU = """
Noise Detector – Main Menu
1) Set baseline noise level
2) Take a sample (live RMS/peak only)
3) Run full noise monitor
4) Show last baseline
5) Exit
"""

def main():
    parser = argparse.ArgumentParser(
        description="Noise Detector - Audio monitoring and chirp classification",
        epilog="""
Examples:
  python3 noise_detector.py monitor          # Start monitoring (legacy)
  python3 noise_detector.py monitor-refactored  # Start monitoring (refactored)
  python3 noise_detector.py monitor --debug   # Start with debug logging
  python3 noise_detector.py baseline          # Set baseline
  python3 noise_detector.py                   # Interactive menu
        """
    )
    parser.add_argument("mode", nargs="?", help="Mode: monitor, monitor-refactored, baseline, sample, show-baseline")
    parser.add_argument("--config", type=Path, help="Path to config.json file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (verbose logging)")
    
    args = parser.parse_args()
    
    # If an argument is given, skip the menu and run directly
    if args.mode:
        mode = args.mode.lower()

        if mode == "monitor":
            monitor.run_monitor(args.config, debug=args.debug)
            return
        elif mode == "monitor-refactored":
            monitor_refactored.run_monitor(args.config, debug=args.debug)
            return
        elif mode == "baseline":
            baseline.set_baseline(config_path=args.config)
            return
        elif mode == "sample":
            sampler.live_sample()
            return
        elif mode == "show-baseline":
            baseline.show_baseline(args.config)
            return
        elif mode == "baseline-analyze":
            baseline.analyze_baseline(args.config)
            return
        elif mode == "baseline-validate":
            baseline.validate_baseline(args.config)
            return
        else:
            print(f"Unknown mode: {mode}")
            return

    # Interactive menu mode
    while True:
        print(MENU)
        choice = input("Choose an option: ").strip()

        if choice == "1":
            baseline.set_baseline()
        elif choice == "2":
            sampler.live_sample()
        elif choice == "3":
            monitor.run_monitor()
        elif choice == "4":
            baseline.show_baseline()
        elif choice == "5":
            print("Goodbye.")
            break
        else:
            print("Invalid option.\n")

if __name__ == "__main__":
    main()
