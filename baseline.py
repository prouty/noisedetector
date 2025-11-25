import json
import time
import datetime
from utils import read_stream_chunks, dbfs, BASELINE_FILE

def set_baseline(duration_sec=10):
    print(f"\nCollecting {duration_sec} seconds for baseline...")
    rms_vals = []
    peak_vals = []

    start = time.time()

    for samples in read_stream_chunks():
        peak = max(abs(s) for s in samples)
        rms = (sum(s*s for s in samples) / len(samples)) ** 0.5

        peak_vals.append(dbfs(peak))
        rms_vals.append(dbfs(rms))

        if time.time() - start >= duration_sec:
            break

    baseline = {
        "timestamp": datetime.datetime.now().isoformat(),
        "rms_db": sum(rms_vals) / len(rms_vals),
        "peak_db": sum(peak_vals) / len(peak_vals)
    }

    with open(BASELINE_FILE, "w") as f:
        json.dump(baseline, f, indent=2)

    print("\nBaseline saved:")
    print(baseline)
    print()

def show_baseline():
    try:
        with open(BASELINE_FILE) as f:
            data = json.load(f)
        print("\nLast Baseline:")
        for k, v in data.items():
            print(f"{k}: {v}")
        print()
    except FileNotFoundError:
        print("No baseline set yet.\n")
