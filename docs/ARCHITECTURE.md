# System Architecture

This document explains how the noise detector works at a high level. Read this first when debugging.

## Overview

The system continuously monitors audio, detects events above a baseline threshold, and classifies them as chirps or noise using spectral fingerprinting.

## Data Flow

```
Audio Hardware (USB Mic/I2S)
    ↓
arecord (ALSA) → Raw PCM Stream
    ↓
monitor.py → Chunk Processing (0.5s chunks)
    ↓
RMS Calculation → Event Detection (threshold check)
    ↓
Event Capture → Clip Saving (WAV files)
    ↓
Classification → Spectral Analysis → Chirp/Noise Decision
    ↓
Logging → events.csv
```

## Key Components

### 1. Audio Capture (`monitor.py::start_arecord`)

- Uses `arecord` subprocess to capture raw PCM
- Outputs to stdout as binary stream
- Device specified in `config.json` → `audio.device`
- **Common failure point:** Device busy, wrong device name, permissions

### 2. Chunk Processing (`monitor.py::run_monitor`)

- Reads fixed-size chunks (0.5s default)
- Converts int16 PCM → float32 normalized samples
- Removes DC offset (hardware artifact)
- Calculates RMS and peak for each chunk

### 3. Baseline Tracking

- **Initial baseline:** Loaded from `baseline.json` (if exists)
- **Rolling baseline:** 20th percentile of last 120 chunks (~60s)
- **Why 20th percentile?** Ignores occasional spikes, captures true quiet level
- **State:** Stored in `baseline_window` list, updated only when not in event

### 4. Event Detection State Machine

```
IDLE → (RMS > baseline + threshold) → IN_EVENT
IN_EVENT → (RMS <= baseline + threshold) → IDLE (save event)
```

- **Pre-roll buffer:** Stores 2s of audio before event (for context)
- **Event chunks:** Accumulates audio during event
- **Actual start index:** Tracks where real event begins (after pre-roll)

### 5. Chirp Classification (`monitor.py::classify_event_is_chirp`)

**Fingerprint System:**
- **Chirp fingerprint:** Spectral template created from positive training examples (`training/chirp/chirp_*.wav`)
- **Non-chirp fingerprint (optional):** Spectral template from negative examples (`training/not_chirp/not_chirp_*.wav`)
- If both are available, classification uses: high similarity to chirp AND low similarity to non-chirp

**Multi-stage filtering:**

1. **Sliding Window:** Finds best matching segment (handles noise at start, chirp at end)
   - Tries windows: 14%, 25%, 33%, 50%, 75%, 100% of event
   - Scores by similarity × frequency quality × window size
   - Uses best scoring segment

2. **Duration Check:** Rejects events > 2.0s (sustained sounds like doors)

3. **Frequency Filtering:**
   - Low-freq rejection: < 30% energy below 500 Hz (fan noise)
   - High-freq requirement: > 10% energy above 1000 Hz (chirp range)
   - **Relaxed threshold:** If similarity > 0.8, allows up to 32% low-freq (handles residual noise)

4. **Temporal Analysis:**
   - Energy concentration: > 50% in first half (staccato chirps)
   - **Relaxed for short segments:** 2-3 chunks use 30% threshold (metric less meaningful)

5. **Spectral Similarity:**
   - Cosine similarity to trained chirp fingerprint
   - Threshold: 0.8 (configurable)
   - **If non-chirp fingerprint available:**
     - Reject if similarity to non-chirp > (chirp_threshold - 0.1)
     - Require chirp similarity is at least 0.2 higher than non-chirp similarity
     - This ensures events are clearly chirps, not ambiguous

6. **Confidence Score:**
   - Weighted: 60% similarity, 20% frequency, 20% temporal

### 6. File I/O

- **Segments:** 5-minute WAV files in `clips/` directory
- **Event clips:** Short clips of detected events (with pre-roll)
- **Events log:** CSV file with all events and classification results

## Critical State Variables

When debugging, check these in `run_monitor`:

- `baseline_rms_db`: Current baseline level (None if not initialized)
- `baseline_window`: Rolling window of RMS values
- `in_event`: Boolean - are we currently in an event?
- `event_chunks`: List of raw audio chunks for current event
- `event_actual_start_idx`: Where real event starts (after pre-roll)
- `pre_roll_buffer`: Audio before event (for context in clips)

## Common Failure Modes

1. **No events detected:**
   - Baseline too high (recalibrate)
   - Threshold too high (lower in config)
   - Audio device not working (check with `audio-check`)

2. **Too many false positives:**
   - Threshold too low
   - Baseline too low
   - Classification thresholds need tuning

3. **Missing chirps:**
   - Similarity threshold too high
   - Sliding window not finding best segment
   - Fingerprint doesn't match actual chirps

4. **System crashes:**
   - Disk full (check `df -h`)
   - Audio device disconnects (USB issues)
   - Memory issues (check with `top`)

## Debugging Strategy

1. **Check logs:** `make logs` or `journalctl -u noise-monitor`
2. **Verify audio:** `make audio-check`
3. **Check baseline:** `python3 baseline.py show`
4. **Review recent events:** `make chirps`
5. **Diagnose specific event:** `python3 diagnose_event.py clips/clip_*.wav`
6. **Enable debug mode:** Run with `--debug` flag for verbose logging

## Configuration Dependencies

- `audio.device`: Must match actual hardware (check with `arecord -l`)
- `audio.sample_rate`: Must match fingerprint sample rate
- `event_detection.threshold_above_baseline_db`: Key sensitivity parameter
- `chirp_classification.similarity_threshold`: Classification sensitivity

## Performance Considerations

- **Chunk size:** 0.5s chunks = 2 Hz update rate (good balance)
- **Baseline window:** 120 chunks = ~60s history (adjusts to environment)
- **FFT size:** 2048 at 16kHz = ~7.8 Hz/bin resolution
- **CPU usage:** Primarily FFT operations during classification

## Edge Cases Handled

- **Very short events:** Minimum 0.5s duration
- **Very long events:** Maximum 2.0s for chirp classification
- **Silent chunks:** Skipped in baseline calculation
- **DC offset:** Removed with exponential moving average
- **Device disconnects:** arecord process monitoring
- **Disk full:** WAV file write errors caught
- **Invalid baseline:** Falls back to rolling baseline only

