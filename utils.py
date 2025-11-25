import subprocess
import numpy as np

DEVICE = "plughw:CARD=Device,DEV=0"
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
BYTES_PER_SAMPLE = 2
CHUNK_BYTES = CHUNK_SAMPLES * BYTES_PER_SAMPLE
INT16_FULL_SCALE = 32768.0

BASELINE_FILE = "baseline.json"

def dbfs(value, eps=1e-12):
    return 20 * np.log10(value + eps)

def start_arecord():
    cmd = [
        "arecord",
        "-D", DEVICE,
        "-f", "S16_LE",
        "-r", str(SAMPLE_RATE),
        "-c", "1",
        "-q",
        "-t", "raw"
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE)

def read_stream_chunks():
    proc = start_arecord()

    try:
        while True:
            data = proc.stdout.read(CHUNK_BYTES)
            if not data:
                break

            samples = np.frombuffer(data, dtype="<i2").astype(np.float32)
            samples /= INT16_FULL_SCALE
            yield samples

    finally:
        proc.terminate()
