#!/usr/bin/env python3

import sys
import baseline
import sampler
import monitor

MENU = """
Noise Detector – Main Menu
1) Set baseline noise level
2) Take a sample (live RMS/peak only)
3) Run full noise monitor
4) Show last baseline
5) Exit
"""

def main():
    # If an argument is given, skip the menu and run directly
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

        if mode == "monitor":
            monitor.run_monitor()
            return
        elif mode == "baseline":
            baseline.set_baseline()
            return
        elif mode == "sample":
            sampler.live_sample()
            return
        elif mode == "show-baseline":
            baseline.show_baseline()
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
