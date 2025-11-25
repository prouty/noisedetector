from utils import read_stream_chunks, dbfs

def live_sample():
    print("\nLive sampling (Ctrl+C to stop)...")
    try:
        for samples in read_stream_chunks():
            peak = max(abs(s) for s in samples)
            rms = (sum(s*s for s in samples) / len(samples)) ** 0.5
            print(f"peak: {dbfs(peak):6.1f} dBFS | rms: {dbfs(rms):6.1f} dBFS")
    except KeyboardInterrupt:
        print("\nStopped.\n")
