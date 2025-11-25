# Noise Detector

A Raspberry Pi-based audio monitoring system that detects noise events, records clips, and classifies chirp sounds using spectral fingerprinting.

## Features

- **Continuous Audio Monitoring**: Records 5-minute audio segments while monitoring for noise events
- **Event Detection**: Detects noise events above a configurable baseline threshold
- **Chirp Classification**: Uses spectral fingerprinting to identify specific chirp sounds with frequency filtering to reject fan noise
- **Event Clips**: Automatically saves short audio clips of detected events with pre-roll context
- **Event Logging**: Records all events to CSV with timestamps, duration, and classification results

## Configuration

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

**Audio Settings** (in `monitor.py`):
- `DEVICE`: ALSA audio device (default: `"plughw:CARD=Device,DEV=0"`)
- `SAMPLE_RATE`: Audio sample rate in Hz (default: `16000`)
- `THRESHOLD_ABOVE_BASELINE_DB`: Event detection threshold (default: `10.0` dB)
- `CHIRP_SIMILARITY_THRESHOLD`: Chirp classification threshold (default: `0.8`)

## Setup

### Raspberry Pi

1. **Install dependencies:**
   ```bash
   sudo apt-get update
   sudo apt-get install -y python3 python3-pip python3-numpy alsa-utils
   pip3 install -r requirements.txt
   ```

2. **Configure audio device:**
   ```bash
   arecord -l    # List audio devices
   ```
   Update `DEVICE` in `monitor.py` with your audio device.

3. **Set baseline:**
   ```bash
   python3 noise_detector.py baseline
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
   - Place training files in `training/chirp/` as `chirp_*.wav`
   - Run: `make train`

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
- `make pull` - Pull events.csv and clips from Pi
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

**Development:**
- `make init` - Create virtual environment
- `make shell` - Activate venv and start shell

**Deployment:**
- `./deploy_to_pi.sh` - Deploy code to Pi (excludes clips, training, venv, .git)

## Chirp Classification

1. Collect chirp samples → `training/chirp/chirp_*.wav`
2. Train: `make train`
3. Deploy: `make deploy`
4. Restart: `make restart`

The algorithm uses frequency filtering to reject fan noise (low-pitched sounds below 500 Hz) and requires significant energy in chirp range (1000+ Hz).

## Output Files

- **`events.csv`**: Event log with timestamps, duration, peak/RMS levels, chirp classification, and `reviewed` column
- **`clips/clip_*.wav`**: Audio clips of detected events
- **`baseline.json`**: Baseline noise level
- **`chirp_fingerprint.json`**: Trained classifier
- **`chirp_report_*.md`**: Generated reports

## Troubleshooting

- **Service not starting**: `make logs` or `journalctl -u noise-monitor -n 50`
- **No audio**: Verify with `arecord -l` and update `DEVICE` in `monitor.py`
- **False positives**: Adjust `THRESHOLD_ABOVE_BASELINE_DB` or recalibrate baseline
- **Chirp misclassification**: Retrain with better samples or adjust `CHIRP_SIMILARITY_THRESHOLD`

## Quick Reference

```bash
# SSH to Pi
ssh ${PI_USER}@${PI_HOSTNAME}

# View logs on Pi
journalctl -u noise-monitor -f

# Reboot Pi
ssh ${PI_USER}@${PI_HOSTNAME} "sudo reboot"
```
