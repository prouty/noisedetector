# Raspberry Pi Setup Guide

Complete step-by-step instructions for setting up the noise detector on a fresh Raspberry Pi.

## Prerequisites

- Raspberry Pi with Raspberry Pi OS (Debian-based)
- SSH access to the Pi
- Audio device connected (USB microphone or built-in audio)

## Step 1: Install System Dependencies

```bash
# Update package list
sudo apt-get update

# Install Python and audio tools
sudo apt-get install -y python3 python3-pip alsa-utils

# Install Python packages via system packages (recommended, faster)
sudo apt-get install -y python3-numpy python3-pandas python3-dateutil

# Install remaining Python dependencies
pip3 install pytz six tzdata
```

**Alternative:** If system packages fail or are outdated, use the installation script:
```bash
# After cloning the repo (see Step 2), run:
./deploy/install_pi_requirements.sh
```

**Note:** You may see a warning: `WARNING: Error parsing dependencies of pyzmq: Invalid version: 'cpython'`
This is harmless and can be safely ignored. It's a known issue with pip's dependency resolver and doesn't affect installation.

## Step 2: Deploy the Project

### From Your Development Machine

Deploy all files to the Pi:
```bash
cd /path/to/noisedetector
./deploy/deploy_to_pi.sh
```

This will sync all files to the Pi (excluding clips, training data, venv, data, reports, etc.).

**Note:** The deploy script will create the directory structure automatically. If deploying manually, ensure these directories exist:
- `data/` - For events.csv, chirp_fingerprint.json, baseline.json
- `clips/` - For recorded event clips
- `scripts/` - Python analysis scripts
- `systemd/` - Service files
- `deploy/` - Deployment scripts
- `docs/` - Documentation

## Step 3: Verify Directory Structure

After deployment, verify the structure on the Pi:
```bash
cd ~/projects/noisedetector
ls -la
```

You should see:
- Core Python files (noise_detector.py, monitor.py, etc.)
- `config.example.json` and `config.json`
- Directories: `scripts/`, `deploy/`, `systemd/`, `data/`, `docs/`

## Step 4: Install Python Dependencies (if not using system packages)

If you didn't use system packages in Step 1:
```bash
cd ~/projects/noisedetector
./deploy/install_pi_requirements.sh
```

**If you encounter NumPy/OpenBLAS errors:**
```bash
./deploy/fix_numpy_deps.sh
```

## Step 5: Create Data Directory

Ensure the data directory exists (for events.csv, baseline.json, etc.):
```bash
cd ~/projects/noisedetector
mkdir -p data
```

## Step 6: Configure Audio Device

```bash
# List available audio devices
arecord -l

# Test audio capture (5 seconds)
arecord -d 5 -f cd test.wav
aplay test.wav
rm test.wav
```

Note the device name (e.g., `plughw:CARD=Device,DEV=0`)

## Step 7: Create Configuration File

```bash
cd ~/projects/noisedetector
cp config.example.json config.json
```

Edit `config.json` and update:
- `audio.device` - Use the device from Step 6
- `email.*` - If you want email reports (optional)

## Step 8: Set Baseline

```bash
cd ~/projects/noisedetector
python3 noise_detector.py baseline
```

This will record 10 seconds of ambient noise to establish the baseline threshold.

## Step 9: Train Chirp Fingerprint (Optional but Recommended)

If you have training samples:
```bash
# Place chirp training files in training/chirp/ as chirp_*.wav
# Place non-chirp examples in training/not_chirp/ as not_chirp_*.wav

# Train the fingerprint
python3 scripts/train_chirp_fingerprint.py
```

This creates `data/chirp_fingerprint.json`.

## Step 10: Install Systemd Service

### Generate Service File
From your development machine:
```bash
cd /path/to/noisedetector
./deploy/install_service.sh
```

This creates `/tmp/noise-monitor.service` with your settings.

### Install on Pi
```bash
# Copy service file to Pi
scp /tmp/noise-monitor.service prouty@raspberrypi:/tmp/

# On the Pi, install the service
ssh prouty@raspberrypi
sudo cp /tmp/noise-monitor.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable noise-monitor
sudo systemctl start noise-monitor
```

### Verify Service is Running
```bash
sudo systemctl status noise-monitor
```

You should see `Active: active (running)`

## Step 11: Setup Email Reports (Optional)

### Configure Email Settings

Edit `config.json` on the Pi and add email section:
```json
{
  "email": {
    "smtp_server": "smtp.fastmail.com",
    "smtp_port": 587,
    "smtp_username": "yourname@fastmail.com",
    "smtp_password": "your-app-password",
    "from_address": "yourname@fastmail.com",
    "to_address": "recipient@example.com",
    "use_tls": true,
    "report_hours": 2
  }
}
```

### Install Email Timer

From your development machine:
```bash
cd /path/to/noisedetector
make install-email-timer
```

Or manually:
```bash
scp systemd/email-report.service systemd/email-report.timer prouty@raspberrypi:/tmp/
ssh prouty@raspberrypi "sudo cp /tmp/email-report.* /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable --now email-report.timer"
```

## Step 12: Verify Everything Works

### Check Service Status
```bash
sudo systemctl status noise-monitor
```

### View Live Logs
```bash
sudo journalctl -u noise-monitor -f
```

### Check for Events
```bash
cd ~/projects/noisedetector
python3 scripts/check_chirps.py
```

### Test Email Report (if configured)
```bash
python3 scripts/email_report.py --no-email
```

## Troubleshooting

### Service Won't Start

1. **Check logs:**
   ```bash
   sudo journalctl -u noise-monitor -n 50
   ```

2. **Common issues:**
   - Missing dependencies: Run `./deploy/fix_numpy_deps.sh`
   - Audio device busy: `pkill arecord`
   - Missing config: Ensure `config.json` exists
   - Permissions: Add user to audio group: `sudo usermod -a -G audio $USER`

### NumPy/Pandas Import Errors

```bash
./deploy/fix_numpy_deps.sh
```

### Audio Device Not Found

```bash
# List devices
arecord -l

# Update config.json with correct device
nano config.json
```

### No Events Detected

1. Check baseline is set: `cat data/baseline.json`
2. Check threshold in `config.json` - may need to lower `threshold_above_baseline_db`
3. Verify audio is working: `arecord -d 5 test.wav && aplay test.wav`

## Quick Reference

**Service Management:**
```bash
sudo systemctl start noise-monitor      # Start
sudo systemctl stop noise-monitor       # Stop
sudo systemctl restart noise-monitor    # Restart
sudo systemctl status noise-monitor     # Status
sudo journalctl -u noise-monitor -f     # Live logs
```

**Manual Commands:**
```bash
python3 noise_detector.py monitor       # Run monitor manually
python3 noise_detector.py baseline      # Set baseline
python3 scripts/check_chirps.py         # Check for chirps
python3 scripts/email_report.py         # Send email report
```

## Next Steps

- Review detected events: `python3 scripts/check_chirps.py`
- Generate reports: `python3 scripts/generate_chirp_report.py`
- Tune thresholds: `python3 scripts/tune_thresholds.py`
- See `docs/TROUBLESHOOTING.md` for more help

