# Noise Detector

A Raspberry Pi-based audio monitoring system that detects noise events, records clips, and classifies chirp sounds using spectral fingerprinting.

## Features

- **Continuous Audio Monitoring**: Records 5-minute audio segments while monitoring for noise events
- **Event Detection**: Detects noise events above a configurable baseline threshold
- **Chirp Classification**: Uses spectral fingerprinting to identify specific chirp sounds
- **Event Clips**: Automatically saves short audio clips of detected events with pre-roll context
- **Event Logging**: Records all events to CSV with timestamps, duration, and classification results
- **Baseline Tracking**: Maintains a rolling baseline for adaptive noise detection

## Project Structure

```
noisedetector/
├── monitor.py              # Main monitoring script with event detection
├── noise_detector.py       # CLI interface and menu system
├── baseline.py             # Baseline noise level measurement
├── sampler.py              # Live audio sampling utility
├── train_chirp_fingerprint.py  # Train chirp classifier from WAV files
├── generate_chirp_report.py    # Generate reports from events.csv
├── deploy_to_pi.sh        # Deploy code to Raspberry Pi
├── noise-monitor.service   # Systemd service file
├── Makefile                # Common workflow commands
├── requirements.txt        # Python dependencies
├── clips/                  # Event audio clips (clip_YYYY-MM-DD_HH-MM-SS.wav)
├── training/chirp/         # Training data for chirp fingerprint
├── events.csv              # Event log (timestamp, duration, classification)
└── baseline.json           # Baseline noise level configuration
```

## Setup

### On Raspberry Pi

1. **Install dependencies:**
   ```bash
   sudo apt-get update
   sudo apt-get install -y python3 python3-pip python3-numpy alsa-utils
   pip3 install -r requirements.txt
   ```

2. **Configure audio device:**
   ```bash
   arecord -l                    # List audio devices
   arecord -L                   # List PCM devices
   ```
   Update `DEVICE` in `monitor.py` with your audio device (e.g., `plughw:CARD=Device,DEV=0`)

3. **Set baseline noise level:**
   ```bash
   python3 noise_detector.py baseline
   ```

4. **Install systemd service:**
   ```bash
   sudo cp noise-monitor.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable noise-monitor
   sudo systemctl start noise-monitor
   ```

### On Development Machine (Mac/Linux)

1. **Clone repository and set up virtual environment:**
   ```bash
   make init                    # Create virtual environment
   make shell                   # Activate venv and start shell
   pip install -r requirements.txt
   ```

2. **Train chirp fingerprint:**
   - Place chirp training files in `training/chirp/` (named `chirp_*.wav`)
   - Run: `make train` or `python3 train_chirp_fingerprint.py`

## Usage

### Interactive Menu

```bash
python3 noise_detector.py
```

Options:
1. Set baseline noise level
2. Take a sample (live RMS/peak only)
3. Run full noise monitor
4. Show last baseline
5. Exit

### Direct Commands

```bash
python3 noise_detector.py monitor        # Start monitoring
python3 noise_detector.py baseline       # Set baseline
python3 noise_detector.py sample         # Live sample
python3 noise_detector.py show-baseline  # Show baseline
```

### Makefile Commands

```bash
make pull      # Pull events.csv and clips from Pi
make train     # Train chirp fingerprint from training/chirp/*.wav
make deploy    # Deploy chirp_fingerprint.json to Pi
make restart   # Restart noise-monitor service on Pi
make report    # Generate chirp report from events.csv
make workflow  # Run pull + report
```

## Configuration

Key settings in `monitor.py`:

- `DEVICE`: ALSA audio device (default: `"plughw:CARD=Device,DEV=0"`)
- `SAMPLE_RATE`: Audio sample rate in Hz (default: `16000`)
- `THRESHOLD_ABOVE_BASELINE_DB`: Event detection threshold (default: `10.0` dB)
- `MIN_EVENT_DURATION_SEC`: Minimum event duration (default: `0.5` seconds)
- `PRE_ROLL_SEC`: Audio before event in clips (default: `2.0` seconds)
- `CHIRP_SIMILARITY_THRESHOLD`: Chirp classification threshold (default: `0.8`)

## Deployment

### Deploy Code to Pi

```bash
./deploy_to_pi.sh
```

This syncs all files except:
- `.git/`, `venv/`, `__pycache__/`
- `clips/`, `training/`, `recordings/`
- `events.csv`, `baseline.json`
- `.DS_Store`

### Manual Deployment

```bash
rsync -avz --exclude='.git' --exclude='venv' --exclude='clips' \
  ./ prouty@raspberrypi.local:/home/prouty/projects/noisedetector/
```

## Service Management

### Start/Stop Service

```bash
sudo systemctl start noise-monitor
sudo systemctl stop noise-monitor
sudo systemctl restart noise-monitor
sudo systemctl status noise-monitor
```

### View Logs

```bash
# Follow logs live
journalctl -u noise-monitor -f

# View recent logs
journalctl -u noise-monitor -n 100
```

## File Transfer

### Pull Data from Pi

```bash
# Pull events.csv and clips
make pull

# Or manually:
rsync -avz prouty@raspberrypi.local:/home/prouty/projects/noisedetector/events.csv ./
rsync -avz prouty@raspberrypi.local:/home/prouty/projects/noisedetector/clips/ ./clips/
```

### Push Code to Pi

```bash
./deploy_to_pi.sh
```

## Audio Debugging

```bash
arecord -l                    # List audio capture devices
arecord -L                   # List PCM devices
aplay test.wav               # Test playback
```

## Chirp Classification

### Training

1. Collect chirp audio samples and place in `training/chirp/` as `chirp_*.wav`
2. Train fingerprint: `make train` or `python3 train_chirp_fingerprint.py`
3. Deploy to Pi: `make deploy`
4. Restart service: `make restart`

### Reports

Generate reports from events:
```bash
make report
# or
python3 generate_chirp_report.py
```

Reports are saved as `chirp_report_YYYY-MM-DD.md`.

## SSH Access

```bash
ssh prouty@raspberrypi.local
hostname -I                  # Get Pi IP address
```

## System Commands

```bash
sudo reboot                  # Reboot Pi
sudo shutdown now            # Shutdown Pi
```

## Output Files

- **`events.csv`**: Event log with timestamps, duration, peak/RMS levels, and chirp classification
- **`clips/clip_*.wav`**: Audio clips of detected events
- **`baseline.json`**: Baseline noise level configuration
- **`chirp_fingerprint.json`**: Trained chirp classifier fingerprint
- **`chirp_report_*.md`**: Generated reports

## Troubleshooting

1. **Service not starting**: Check logs with `journalctl -u noise-monitor -n 50`
2. **No audio detected**: Verify device with `arecord -l` and update `DEVICE` in `monitor.py`
3. **Too many false positives**: Adjust `THRESHOLD_ABOVE_BASELINE_DB` or recalibrate baseline
4. **Chirp classification inaccurate**: Retrain with more/better training samples and adjust `CHIRP_SIMILARITY_THRESHOLD`

