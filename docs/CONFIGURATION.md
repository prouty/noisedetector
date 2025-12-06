# Configuration Reference

Complete guide to configuring the Noise Detector system.

## Environment Variables

Create a `.env` file in the project root to customize deployment settings:

```bash
PI_USER=prouty                    # SSH username
PI_HOSTNAME=raspberrypi.local     # Pi hostname or IP
PI_DIR=/home/prouty/projects/noisedetector  # Remote project directory
LOCAL_DIR=$HOME/projects/noisedetector      # Local project directory
SERVICE_USER=prouty               # Service user (defaults to PI_USER)
SERVICE_WORKING_DIR=/home/prouty/projects/noisedetector  # Service working dir
```

## Application Configuration (config.json)

Create `config.json` from the template:

```bash
cp config.example.json config.json
```

### Audio Settings

```json
{
  "audio": {
    "device": "plughw:CARD=Device,DEV=0",  // ALSA device (use 'arecord -l' to find)
    "sample_rate": 16000,                    // Sample rate in Hz
    "channels": 1,                          // Number of channels
    "sample_format": "S16_LE",              // Sample format
    "chunk_duration": 0.5,                  // Processing chunk duration (seconds)
    "dc_offset_removal": true,              // Remove DC offset
    "high_pass_filter_hz": 20               // High-pass filter cutoff
  }
}
```

### Event Detection

```json
{
  "event_detection": {
    "baseline_file": "data/baseline.json",
    "baseline_name": null,                  // Named baseline (null = active)
    "events_file": "data/events.csv",
    "threshold_above_baseline_db": 10,      // dB above baseline to trigger event
    "min_event_duration_sec": 0.5,         // Minimum event duration
    "baseline_window_chunks": 120           // Rolling baseline window size
  }
}
```

### Chirp Classification

```json
{
  "chirp_classification": {
    "fingerprint_file": "data/chirp_fingerprint.json",
    "use_ml_classifier": false,             // Use ML model instead of fingerprint
    "similarity_threshold": 0.8,            // Minimum cosine similarity (0-1)
    
    "frequency_filtering": {
      "fan_noise_max_freq_hz": 500,        // Max freq for "low frequency" (fan noise)
      "chirp_min_freq_hz": 1000,           // Min freq for "high frequency" (chirp)
      "low_freq_energy_threshold": 0.30,    // Max ratio of low-freq energy
      "high_freq_energy_min_ratio": 0.10    // Min ratio of high-freq energy
    },
    
    "temporal_filtering": {
      "max_duration_sec": 2.0,              // Max event duration
      "energy_concentration_threshold": 0.50 // Min energy in first half
    },
    
    "confidence": {
      "enabled": true,
      "similarity_weight": 0.6,            // Weight for similarity score
      "frequency_weight": 0.2,               // Weight for frequency score
      "temporal_weight": 0.2                 // Weight for temporal score
    }
  }
}
```

### Recording Settings

```json
{
  "recording": {
    "output_dir": "clips",                  // Directory for audio segments
    "segment_duration_sec": 300             // Segment file duration (5 minutes)
  },
  
  "event_clips": {
    "clips_dir": "clips",                   // Directory for event clips
    "pre_roll_sec": 0.5                     // Audio context before event
  }
}
```

## Tuning Thresholds

See [TUNING.md](TUNING.md) for detailed guidance on adjusting thresholds for your environment.

### Quick Adjustments

**Too many false positives (non-chirps classified as chirps):**
- Increase `similarity_threshold` (e.g., 0.85)
- Decrease `low_freq_energy_threshold` (e.g., 0.25)
- Decrease `max_duration_sec` (e.g., 1.5)

**Too many false negatives (missing actual chirps):**
- Decrease `similarity_threshold` (e.g., 0.75)
- Increase `high_freq_energy_min_ratio` (e.g., 0.08)
- Increase `max_duration_sec` (e.g., 2.5)

## Validation

After changing configuration, validate with:

```bash
python3 scripts/validate_classification.py
```

This shows accuracy metrics and helps identify needed adjustments.

