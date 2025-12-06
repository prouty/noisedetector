# Noise Detector

A Raspberry Pi-based audio monitoring system that detects noise events, records clips, and classifies chirp sounds using spectral fingerprinting.

## Quick Start

### On Raspberry Pi

1. **Install dependencies:**
   ```bash
   sudo apt-get update
   sudo apt-get install -y python3 python3-pip alsa-utils python3-numpy python3-pandas
   pip3 install python-dateutil pytz six tzdata
   ```

2. **Configure audio device:**
   ```bash
   arecord -l    # List audio devices
   ```
   Update `audio.device` in `config.json`.

3. **Create configuration:**
   ```bash
   cp config.example.json config.json
   # Edit config.json as needed
   ```

4. **Set baseline:**
   ```bash
   python3 noise_detector.py baseline
   ```

5. **Install and start service:**
   ```bash
   make restart  # Deploys service file and starts monitoring
   ```

### On Development Machine

1. **Setup:**
   ```bash
   make init && make shell
   pip install -r requirements.txt
   ```

2. **Train chirp classifier:**
   ```bash
   # Place chirp samples in training/chirp/chirp_*.wav
   # Place non-chirp examples in training/not_chirp/not_chirp_*.wav (optional)
   make train
   make deploy  # Deploy to Pi
   ```

## Basic Usage

### Service Management

```bash
make start      # Start monitoring service
make stop       # Stop service
make restart    # Restart service (updates service file)
make status     # Check service status
make logs       # View live logs
```

### Data & Reports

```bash
make pull           # Pull events.csv and clips from Pi
make pull-chirps    # Pull only chirp clips
make report         # Generate chirp report
make workflow       # Pull + generate report
```

### Quick Status

```bash
make chirps         # Show detected chirps
make chirps-recent  # Show chirps from last 24 hours
make health         # System health check
```

### Email Reports (Optional)

```bash
make install-email-timer  # Install automated reports (every 2 hours)
make email-report         # Send report manually
```

See [docs/EMAIL_SETUP.md](docs/EMAIL_SETUP.md) for email configuration.

## Configuration

### Environment Variables

Create `.env` file to customize Pi connection:

```bash
PI_USER=prouty
PI_HOSTNAME=raspberrypi.local
PI_DIR=/home/prouty/projects/noisedetector
```

### Application Settings

All detection and classification settings are in `config.json`:

- **`audio`**: Device, sample rate, chunk duration
- **`event_detection`**: Baseline threshold, minimum event duration
- **`chirp_classification`**: Similarity threshold, frequency/temporal filters

See `config.example.json` for all options.

## Output Files

- **`events.csv`**: Event log with timestamps, classification, confidence scores
- **`clips/clip_*.wav`**: Audio clips of detected events
- **`chirp_fingerprint.json`**: Trained classifier (after `make train`)

## Troubleshooting

- **Service not starting**: Check `make logs` for errors
- **No audio detected**: Verify device with `arecord -l` and update `config.json`
- **Too many false positives**: See [docs/TUNING.md](docs/TUNING.md)
- **Import errors**: Run `make health` to check dependencies

For detailed troubleshooting, see [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

## Documentation

- **[Setup Guide](docs/SETUP_PI.md)**: Detailed installation instructions
- **[Configuration](docs/CONFIGURATION.md)**: Complete configuration reference
- **[Tuning Guide](docs/TUNING.md)**: How to tune classification thresholds
- **[Usage Reference](docs/USAGE.md)**: All commands and tools
- **[Architecture](docs/ARCHITECTURE.md)**: System design and components
- **[Troubleshooting](docs/TROUBLESHOOTING.md)**: Common issues and solutions

## Project Structure

```
noisedetector/
├── monitor.py              # Main monitoring loop
├── noise_detector.py        # Entry point
├── core/                    # Core domain model (SOLID architecture)
│   ├── audio.py            # AudioCapture
│   ├── baseline.py         # BaselineTracker
│   ├── classifier.py       # Classification logic
│   ├── detector.py         # EventDetector
│   └── repository.py       # EventRepository, SegmentRepository
├── scripts/                 # Analysis and utility scripts
├── docs/                    # Documentation
└── config.json             # Configuration (create from config.example.json)
```
