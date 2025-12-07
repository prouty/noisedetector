# Usage Reference

Complete reference for all commands and tools.

## Makefile Commands

All commands use `.env` configuration automatically.

### Service Management

```bash
make start      # Start monitoring service on Pi
make stop       # Stop service on Pi
make restart    # Deploy service file and restart (updates systemd)
make status     # Check service status
make logs       # View live logs (shows recent + follows)
make reload     # Reload systemd daemon (after service file changes)
```

### Data & Reports

```bash
make pull           # Pull events.csv and clips (<=10s) from Pi
make pull-chirps    # Pull only clips identified as chirps
make report         # Generate chirp report from events.csv
make workflow       # Pull + generate report
```

### Chirp Classification

```bash
make train      # Train fingerprint from training/chirp/*.wav
make deploy     # Deploy fingerprint to Pi
```

### Email Reports

```bash
make email-report         # Send email report manually
make email-report-test    # Test email report (no email sent)
make install-email-timer  # Install automated email timer (every 2 hours)
make email-timer-status   # Check email timer status
make email-timer-logs     # View email report logs
```

### Quick Status

```bash
make chirps         # Quick check for detected chirps in events.csv
make chirps-recent  # Show chirps from last 24 hours
make health         # System health check (dependencies, config, disk space)
make audio-check    # Validate audio capture levels on Pi
```

### Testing

```bash
make test               # Run all tests
make test-features      # Run feature extraction tests
make test-email         # Run email functionality tests
make test-reporting    # Run reporting tests
make test-core          # Run all core module tests (features, email, reporting)
make test-capture-ml    # Run ML capture tests
```

**Note:** All tests run locally on your development machine, not on the Pi. Tests use mocking for external dependencies.

See [tests/README.md](../tests/README.md) for detailed testing documentation.

### Development

```bash
make init       # Create virtual environment
make shell       # Activate venv and start shell
make help        # Show all available make commands
```

## Direct Python Commands

### Main Entry Point

```bash
python3 noise_detector.py              # Interactive menu
python3 noise_detector.py monitor       # Start monitoring
python3 noise_detector.py baseline     # Set baseline
python3 noise_detector.py sample       # Live audio sample
python3 noise_detector.py show-baseline # Show current baseline
```

### Baseline Management

```bash
python3 baseline.py set [--duration 10] [--config config.json]
python3 baseline.py show [--config config.json]
python3 baseline.py analyze [--config config.json]
python3 baseline.py validate [--config config.json]
```

## Analysis & Validation Scripts

### Validation

```bash
python3 scripts/validate_classification.py \
  [--config config.json] \
  [--events events.csv] \
  [--date-start YYYY-MM-DD] \
  [--date-end YYYY-MM-DD]
```

Shows accuracy metrics, confusion matrix, and tuning recommendations.

### Clip Analysis

```bash
python3 scripts/analyze_clips.py \
  [--config config.json] \
  [--events events.csv] \
  [--output clip_analysis.csv]
```

Generates CSV with audio features for all clips.

### Event Diagnosis

```bash
python3 scripts/diagnose_event.py clips/clip_*.wav \
  [--config config.json]
```

Shows detailed analysis of why an event was/wasn't classified as chirp.

### Threshold Tuning

```bash
python3 scripts/tune_thresholds.py \
  [--config config.json] \
  [--events events.csv] \
  [--similarity-range 0.7 0.75 0.8 0.85 0.9] \
  [--low-freq-range 0.2 0.25 0.3 0.35] \
  [--max-duration-range 1.5 2.0 2.5] \
  [--energy-conc-range 0.4 0.5 0.6]
```

Grid search for optimal threshold values. Generates `config.recommended.json`.

### Audio Analysis

```bash
python3 audio_analysis.py validate-levels \
  [--config config.json] \
  [--duration 5.0]

python3 audio_analysis.py analyze \
  --clip clips/clip_*.wav \
  [--config config.json]

python3 audio_analysis.py check-dc \
  --clip clips/clip_*.wav
```

### Compare Classifiers

```bash
python3 scripts/compare_classifiers.py \
  [--config config.json] \
  [--events events.csv]
```

Compares ML model vs fingerprint classification on reviewed events.

### Rediagnose Training Files

```bash
python3 scripts/rediagnose_events.py \
  [--config config.json] \
  [--training-dir training]
```

Re-classifies files in training directories to find mislabeled samples.

## Direct SSH Commands

```bash
# SSH to Pi
ssh ${PI_USER}@${PI_HOSTNAME}

# View logs on Pi
journalctl -u noise-monitor -f

# Check service status
sudo systemctl status noise-monitor

# Reboot Pi
ssh ${PI_USER}@${PI_HOSTNAME} "sudo reboot"
```

## File Locations

- **Events**: `data/events.csv`
- **Clips**: `clips/clip_*.wav`
- **Segments**: `clips/YYYY-MM-DD_HH-MM-SS.wav` (5-minute segments)
- **Baseline**: `data/baseline.json`
- **Fingerprint**: `data/chirp_fingerprint.json`
- **Config**: `config.json`
- **Reports**: `reports/chirp_report_*.md`

