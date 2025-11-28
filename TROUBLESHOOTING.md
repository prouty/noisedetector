# Troubleshooting Guide

This guide helps you debug issues when the system isn't working as expected.

## Quick Diagnostic Commands

```bash
# Check if service is running
make status

# View recent logs
make logs

# Check for detected chirps
make chirps

# Validate audio capture
make audio-check

# Check baseline
python3 baseline.py show
python3 baseline.py validate
```

## Common Issues

### 0. Dependency Installation Failures

**Symptoms:** `pip install -r requirements.txt` fails with pandas build errors

**Diagnosis:**
```bash
# Check if numpy is installed
python3 -c "import numpy; print(numpy.__version__)"

# Check Python version
python3 --version
```

**Solutions:**
- **Use system packages (recommended):**
  ```bash
  sudo apt-get install python3-numpy python3-pandas
  pip3 install python-dateutil pytz six tzdata
  ```
  
- **Install in correct order:**
  ```bash
  pip3 install numpy  # Must install first
  pip3 install pandas  # Then pandas
  pip3 install -r requirements.txt
  ```
  
- **Use installation script:**
  ```bash
  ./install_pi_requirements.sh
  ```
  
- **If pandas build fails:** Use older version with better ARM support:
  ```bash
  pip3 install "pandas>=1.5.0,<2.0.0"
  ```

### 1. Service Won't Start

**Symptoms:** `systemctl status noise-monitor` shows failed state

**Diagnosis:**
```bash
# Check service logs
sudo journalctl -u noise-monitor -n 50

# Check for common errors:
# - "Device or resource busy" → Audio device in use
# - "No such file or directory" → Missing config.json or dependencies
# - "Permission denied" → User doesn't have audio device access
```

**Solutions:**
- **Device busy:** Stop other processes using audio: `pkill arecord`, `pkill pulseaudio`
- **Missing config:** Create `config.json` from `config.example.json`
- **Permissions:** Add user to `audio` group: `sudo usermod -a -G audio $USER`
- **Missing dependencies:** `pip3 install numpy`

### 2. No Events Detected

**Symptoms:** `events.csv` is empty or not updating

**Diagnosis:**
```bash
# Check if audio is being captured
python3 audio_analysis.py validate-levels

# Check baseline level
python3 baseline.py show

# Check if threshold is too high
# Look at config.json: threshold_above_baseline_db
```

**Solutions:**
- **No audio:** Check microphone connection, verify device in `config.json`
- **Baseline too high:** Recalibrate: `python3 baseline.py set`
- **Threshold too high:** Lower `threshold_above_baseline_db` in config
- **Service not running:** `make start`

### 3. Too Many False Positives

**Symptoms:** Many events logged but few are actual chirps

**Diagnosis:**
```bash
# Review recent events
python3 check_chirps.py

# Diagnose specific event
python3 diagnose_event.py clips/clip_YYYY-MM-DD_HH-MM-SS.wav

# Validate classification accuracy
python3 validate_classification.py
```

**Solutions:**
- **Adjust thresholds:** Use `tune_thresholds.py` to find optimal values
- **Retrain fingerprint:** Collect better training samples
- **Increase similarity threshold:** Raise `similarity_threshold` in config
- **Tighten frequency filters:** Lower `low_freq_energy_threshold`

### 4. Missing Chirps (False Negatives)

**Symptoms:** Known chirps not being detected

**Diagnosis:**
```bash
# Check if events are being detected at all
tail -20 events.csv

# Diagnose a known chirp clip
python3 diagnose_event.py clips/clip_KNOWN_CHIRP.wav

# Check similarity scores
python3 analyze_clips.py
```

**Solutions:**
- **Lower similarity threshold:** Decrease `similarity_threshold` in config
- **Relax frequency filters:** Increase `high_freq_energy_min_ratio`
- **Retrain fingerprint:** Add more diverse training samples
- **Check sliding window:** Verify algorithm is finding best segment

### 5. Audio Device Errors

**Symptoms:** "Device or resource busy", "No such file or directory"

**Diagnosis:**
```bash
# List available audio devices
arecord -l

# Check what's using the device
lsof /dev/snd/*

# Test device directly
arecord -D plughw:CARD=Device,DEV=0 -f S16_LE -r 16000 -c 1 -d 2 test.wav
```

**Solutions:**
- **Wrong device:** Update `audio.device` in `config.json`
- **Device in use:** Stop other audio processes
- **Device not found:** Check USB connection, verify device name
- **Permissions:** Add user to `audio` group

### 6. High CPU Usage

**Symptoms:** Pi running hot, system sluggish

**Diagnosis:**
```bash
# Check CPU usage
top -p $(pgrep -f monitor.py)

# Check if too many events being processed
wc -l events.csv
```

**Solutions:**
- **Too many events:** Increase `threshold_above_baseline_db`
- **Long events:** Decrease `max_duration_sec` in config
- **Optimize chunk size:** Increase `chunk_duration` (trades latency for CPU)

### 7. Disk Space Issues

**Symptoms:** "No space left on device", clips not saving

**Diagnosis:**
```bash
# Check disk usage
df -h

# Check clip directory size
du -sh clips/

# Count clips
ls clips/*.wav | wc -l
```

**Solutions:**
- **Clean old clips:** Delete clips older than N days
- **Reduce segment duration:** Lower `segment_duration_sec` in config
- **Archive clips:** Move old clips to external storage
- **Increase disk space:** Use larger SD card or external drive

### 8. Classification Inconsistency

**Symptoms:** Same clip classified differently on different runs

**Diagnosis:**
```bash
# Re-classify events
python3 rediagnose_events.py

# Compare results
diff events.csv events.csv.backup
```

**Solutions:**
- **Check config consistency:** Ensure same `config.json` used
- **Verify fingerprint:** Check `chirp_fingerprint.json` hasn't changed
- **Check for updates:** Algorithm may have changed, review code changes

## Debug Mode

Enable verbose logging for detailed diagnostics:

```bash
# Run monitor with debug logging
python3 monitor.py --debug --log-file monitor.log

# Or set in config.json
{
  "logging": {
    "level": "DEBUG",
    "file": "monitor.log"
  }
}
```

## System Health Check

Run this to verify everything is working:

```bash
#!/bin/bash
echo "=== Noise Detector Health Check ==="
echo "1. Service status:"
make status
echo ""
echo "2. Audio device:"
arecord -l | grep -i device
echo ""
echo "3. Recent events:"
tail -5 events.csv
echo ""
echo "4. Disk space:"
df -h | grep -E "Filesystem|/dev/root"
echo ""
echo "5. Recent chirps:"
make chirps
```

## Getting Help

If you're stuck:

1. **Check logs:** `make logs` or `journalctl -u noise-monitor -n 100`
2. **Enable debug mode:** Run with `--debug` flag
3. **Review config:** Verify `config.json` matches `config.example.json`
4. **Check dependencies:** `pip3 list | grep -E "numpy|pandas"`
5. **System info:** `uname -a`, `python3 --version`

## Emergency Recovery

If the system is completely broken:

```bash
# 1. Stop service
make stop

# 2. Backup current state
cp events.csv events.csv.backup
cp config.json config.json.backup

# 3. Reset to defaults
cp config.example.json config.json
# Edit config.json with your settings

# 4. Recalibrate
python3 baseline.py set

# 5. Test
python3 noise_detector.py sample

# 6. Restart
make start
```

