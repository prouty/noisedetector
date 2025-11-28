# Noise Detector

A Raspberry Pi-based audio monitoring system that detects noise events, records clips, and classifies chirp sounds using spectral fingerprinting.

## Features

- **Continuous Audio Monitoring**: Records 5-minute audio segments while monitoring for noise events
- **Event Detection**: Detects noise events above a configurable baseline threshold with adaptive baseline tracking
- **Advanced Chirp Classification**: Uses spectral fingerprinting with frequency filtering, temporal analysis, and confidence scoring
- **Event Clips**: Automatically saves short audio clips of detected events with pre-roll context
- **Event Logging**: Records all events to CSV with timestamps, duration, classification results, confidence scores, and rejection reasons
- **Configuration System**: All thresholds and settings configurable via `config.json`
- **Validation & Tuning Tools**: Tools to analyze accuracy, tune thresholds, and diagnose classification decisions

## Configuration

### Environment Variables

Create a `.env` file from `.env.example` to customize Pi connection settings:

```bash
cp .env.example .env
# Edit .env with your settings
```

**Environment Variables:**
- `PI_USER`: SSH username (default: `prouty`)
- `PI_HOSTNAME`: Pi hostname or IP (default: `raspberrypi.local`)
- `PI_DIR`: Remote project directory (default: `/home/prouty/projects/noisedetector`)
- `LOCAL_DIR`: Local project directory (default: `$HOME/projects/noisedetector`)
- `SERVICE_USER`: Service user (defaults to `PI_USER`)
- `SERVICE_WORKING_DIR`: Service working directory (defaults to `PI_DIR`)

### Application Configuration (config.json)

All detection and classification thresholds are configurable via `config.json`. Create from example:

```bash
cp config.example.json config.json
# Edit config.json with your settings
```

**Key Configuration Sections:**

- **`audio`**: Audio device, sample rate, chunk duration
- **`event_detection`**: Baseline threshold, minimum event duration, baseline window size
- **`chirp_classification`**: Similarity threshold, frequency filtering (fan noise rejection), temporal filtering (sustained sound rejection), confidence scoring weights

See `config.example.json` for full documentation of all parameters.

## Setup

### Raspberry Pi

1. **Install dependencies:**
   ```bash
   sudo apt-get update
   sudo apt-get install -y python3 python3-pip alsa-utils
   
   # Option 1: Use system packages (recommended, faster)
   sudo apt-get install -y python3-numpy python3-pandas
   pip3 install python-dateutil pytz six tzdata
   
   # Option 2: Use pip with proper installation script
   # (Use this if system packages are outdated)
   ./install_pi_requirements.sh
   
   # Option 3: Manual pip install (if above fail)
   pip3 install --upgrade pip setuptools wheel
   pip3 install numpy  # Install numpy first
   pip3 install pandas  # Then pandas
   pip3 install -r requirements.txt
   ```

2. **Configure audio device:**
   ```bash
   arecord -l    # List audio devices
   ```
   Update `audio.device` in `config.json` with your audio device.

3. **Create configuration:**
   ```bash
   cp config.example.json config.json
   # Edit config.json as needed
   ```

4. **Set baseline:**
   ```bash
   python3 noise_detector.py baseline
   # Or: python3 baseline.py set
   ```

4. **Install service:**
   ```bash
   # On development machine
   ./install_service.sh
   scp /tmp/noise-monitor.service ${PI_USER}@${PI_HOSTNAME}:/tmp/
   ssh ${PI_USER}@${PI_HOSTNAME} "sudo cp /tmp/noise-monitor.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable noise-monitor && sudo systemctl start noise-monitor"
   ```

### Development Machine

1. **Setup:**
   ```bash
   make init && make shell
   pip install -r requirements.txt
   ```

2. **Train chirp fingerprint:**
   - **Positive examples (required):** Place chirp training files in `training/chirp/` as `chirp_*.wav`
   - **Negative examples (optional but recommended):** Place non-chirp examples in `training/not_chirp/` as `not_chirp_*.wav`
     - Examples: door sounds, fan noise, other bird calls, mechanical clicks
     - Helps the system learn what NOT to classify as chirps
   - Run: `make train`
   
   **Training tips:**
   - Use 5-10+ chirp examples for best results
   - Include diverse non-chirp examples (fan, doors, other sounds you want to reject)
   - All files must have the same sample rate (matches config.json)
   - The system will create both a chirp fingerprint and (if provided) a non-chirp fingerprint

## Usage

### Direct Commands

```bash
python3 noise_detector.py              # Interactive menu
python3 noise_detector.py monitor      # Start monitoring
python3 noise_detector.py baseline     # Set baseline
python3 noise_detector.py sample       # Live sample
```

### Makefile Commands

All commands use `.env` configuration automatically.

**Data & Reports:**
- `make pull` - Pull events.csv and all clips from Pi
- `make pull-chirps` - Pull only clips identified as chirps from Pi
- `make report` - Generate chirp report
- `make workflow` - Pull + report

**Chirp Classification:**
- `make train` - Train fingerprint from training/chirp/*.wav
- `make deploy` - Deploy fingerprint to Pi

**Service Management:**
- `make start` - Start service on Pi
- `make stop` - Stop service on Pi
- `make restart` - Restart service on Pi
- `make status` - Check service status
- `make logs` - View live logs

**Audio Calibration:**
- `make audio-check` - Validate audio capture levels on Pi (checks for clipping, DC offset, noise floor)

**Quick Status:**
- `make chirps` - Quick check for detected chirps in events.csv
- `make chirps-recent` - Show chirps from last 24 hours
- `make health` - Run system health check (dependencies, config, disk space, etc.)

**Development:**
- `make init` - Create virtual environment
- `make shell` - Activate venv and start shell

**Deployment:**
- `./deploy_to_pi.sh` - Deploy code to Pi (excludes clips, training, venv, .git)

## Chirp Classification

The classification system uses multiple techniques to accurately identify chirps:

1. **Spectral Fingerprinting**: Cosine similarity to trained chirp spectrum
2. **Frequency Filtering**: Rejects fan noise (low-frequency energy) and requires high-frequency content
3. **Temporal Filtering**: Rejects sustained sounds (door movements) by checking duration and energy concentration
4. **Confidence Scoring**: Weighted combination of similarity, frequency, and temporal features

### Training

1. Collect chirp samples → `training/chirp/chirp_*.wav`
2. Train: `make train`
3. Deploy: `make deploy`
4. Restart: `make restart`

### Classification Algorithm

- **Similarity Threshold**: Minimum cosine similarity to fingerprint (default: 0.8)
- **Frequency Filtering**: 
  - Rejects if low-freq energy (>500 Hz) > 30% of total
  - Requires high-freq energy (>1000 Hz) > 10% of total
- **Temporal Filtering**:
  - Rejects events longer than 2.0 seconds
  - Requires >50% of energy in first half of event
- **Confidence Score**: Weighted combination (60% similarity, 20% frequency, 20% temporal)

## Tuning Guide

### Step 1: Collect and Review Events

1. Run the monitor and collect events
2. Review clips and mark in `events.csv` `reviewed` column:
   - Mark chirps as: `chirp`, `true`, `yes`, or `positive`
   - Mark non-chirps as: `not_chirp`, `false`, `no`, or `negative`

### Step 2: Validate Current Performance

```bash
python3 validate_classification.py
```

This will show:
- Accuracy, precision, recall, F1 score
- Confusion matrix
- List of false positives and false negatives
- Tuning recommendations

### Step 3: Analyze Clip Characteristics

```bash
python3 analyze_clips.py
```

This generates `clip_analysis.csv` with features for all clips, helping identify patterns in chirps vs non-chirps.

### Step 4: Diagnose Specific Events

```bash
python3 diagnose_event.py clips/clip_2025-11-24_20-02-05.wav
```

Shows detailed analysis of why an event was/wasn't classified as chirp, including:
- All filter results (frequency, temporal, similarity)
- Which filters passed/failed
- Confidence score breakdown

### Step 5: Tune Thresholds

```bash
# Grid search for optimal thresholds
python3 tune_thresholds.py --events events.csv

# Or specify custom ranges
python3 tune_thresholds.py \
  --similarity-range 0.7 0.75 0.8 0.85 0.9 \
  --low-freq-range 0.2 0.25 0.3 0.35 \
  --max-duration-range 1.5 2.0 2.5 \
  --energy-conc-range 0.4 0.5 0.6
```

This will:
- Test all combinations of threshold values
- Find the configuration with best F1 score
- Save results to `tuning_results.csv`
- Generate `config.recommended.json` with optimal settings

Review `config.recommended.json` and update your `config.json` if desired.

### Step 6: Iterate

Repeat steps 2-5 until satisfied with accuracy.

### Common Tuning Scenarios

**Too many false positives (non-chirps classified as chirps):**
- Increase `similarity_threshold`
- Decrease `low_freq_energy_threshold` (stricter fan noise rejection)
- Decrease `max_duration_sec` (stricter duration limit)
- Increase `energy_concentration_threshold` (require more energy in first half)

**Too many false negatives (missing actual chirps):**
- Decrease `similarity_threshold`
- Increase `high_freq_energy_min_ratio` (allow less high-freq content)
- Increase `max_duration_sec` (allow longer events)
- Decrease `energy_concentration_threshold` (allow more spread-out energy)

## Baseline Management

The system supports baseline history tracking and validation:

```bash
# Set baseline
python3 baseline.py set --duration 10

# Show current baseline
python3 baseline.py show

# Analyze baseline history
python3 baseline.py analyze

# Validate baseline reliability
python3 baseline.py validate
```

Baseline history is stored in `baseline.json` as an array, allowing you to track changes over time.

## Output Files

- **`events.csv`**: Event log with timestamps, duration, peak/RMS levels, chirp classification, confidence scores, rejection reasons, and `reviewed` column
- **`clips/clip_*.wav`**: Audio clips of detected events
- **`baseline.json`**: Baseline noise level (with history)
- **`chirp_fingerprint.json`**: Trained classifier
- **`chirp_report_*.md`**: Generated reports
- **`config.json`**: Application configuration (create from `config.example.json`)
- **`validation_results.csv`**: Results from validation runs
- **`clip_analysis.csv`**: Audio feature analysis of all clips
- **`tuning_results.csv`**: Results from threshold tuning

## Tools Reference

### Validation & Analysis

- **`validate_classification.py`**: Validate classification accuracy against reviewed events
  ```bash
  python3 validate_classification.py [--config config.json] [--events events.csv] [--date-start YYYY-MM-DD] [--date-end YYYY-MM-DD]
  ```

- **`analyze_clips.py`**: Analyze audio characteristics of all clips
  ```bash
  python3 analyze_clips.py [--config config.json] [--events events.csv] [--output clip_analysis.csv]
  ```

- **`diagnose_event.py`**: Diagnose why a specific event was/wasn't classified as chirp
  ```bash
  python3 diagnose_event.py clips/clip_*.wav [--config config.json]
  ```

- **`tune_thresholds.py`**: Grid search for optimal threshold values
  ```bash
  python3 tune_thresholds.py [--config config.json] [--events events.csv] [--similarity-range ...] [--low-freq-range ...] [--max-duration-range ...] [--energy-conc-range ...]
  ```

### Baseline Management

- **`baseline.py`**: Baseline management commands
  ```bash
  python3 baseline.py set [--duration 10] [--config config.json]
  python3 baseline.py show [--config config.json]
  python3 baseline.py analyze [--config config.json]
  python3 baseline.py validate [--config config.json]
  ```

- **`audio_analysis.py`**: Audio quality analysis and calibration
  ```bash
  python3 audio_analysis.py validate-levels [--config config.json] [--duration 5.0]
  python3 audio_analysis.py analyze --clip clips/clip_*.wav [--config config.json]
  python3 audio_analysis.py check-dc --clip clips/clip_*.wav
  ```

## Troubleshooting

- **Service not starting**: `make logs` or `journalctl -u noise-monitor -n 50`
- **No audio**: Verify with `arecord -l` and update `audio.device` in `config.json`
- **False positives**: Use tuning tools to adjust thresholds, or increase `similarity_threshold` in `config.json`
- **Chirp misclassification**: 
  - Use `diagnose_event.py` to understand why events are misclassified
  - Use `validate_classification.py` to see overall accuracy
  - Use `tune_thresholds.py` to find optimal thresholds
  - Retrain fingerprint with better samples if needed
- **Config errors**: Ensure `config.json` is valid JSON and matches structure in `config.example.json`

## Quick Reference

```bash
# SSH to Pi
ssh ${PI_USER}@${PI_HOSTNAME}

# View logs on Pi
journalctl -u noise-monitor -f

# Reboot Pi
ssh ${PI_USER}@${PI_HOSTNAME} "sudo reboot"
```
