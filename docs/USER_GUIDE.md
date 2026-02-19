# User Guide

Task-oriented guide for operating the Noise Detector day to day.

This guide is for the person who runs the system and reviews chirps. It focuses on
what to do, in what order, and which commands are safe to run.

For deep technical reference, see:
- [Setup Guide](SETUP_PI.md)
- [Configuration Reference](CONFIGURATION.md)
- [Usage Reference](USAGE.md)
- [Troubleshooting](TROUBLESHOOTING.md)

## 1) What this application does

Noise Detector runs on a Raspberry Pi and:
- listens to audio continuously
- detects noise events above baseline
- saves short event clips
- classifies events as chirp or not_chirp
- stores results in `data/events.csv`

## 2) Before you start

### A. Set your local `.env` (development machine)

Most `make` commands run from your development machine and connect to the Pi over SSH.

Create `.env` in the project root:

```bash
PI_USER=your-pi-user
PI_HOSTNAME=raspberrypi.local
PI_DIR=/home/your-pi-user/projects/noisedetector
LOCAL_DIR=$HOME/projects/noisedetector
```

### B. Ensure `config.json` exists

On the Pi project directory:

```bash
cp config.example.json config.json
```

At minimum, verify:
- `audio.device` matches your microphone (`arecord -l`)
- detection/classification defaults are acceptable for your environment

## 3) First-time startup workflow

If your Pi is already deployed and configured, skip to section 4.

1. Set baseline:
   ```bash
   make baseline-set
   ```
2. Start or restart monitor service:
   ```bash
   make restart
   ```
3. Verify service is healthy:
   ```bash
   make status
   make logs
   ```
4. Quick chirp check:
   ```bash
   make chirps-recent
   ```

## 4) Daily operator workflow

### Step 1: Check system status

```bash
make status
make events-recent
make chirps-recent
```

Optional deeper health check:

```bash
make health
```

### Step 2: Pull new events and clips locally

```bash
make pull
```

Important behavior:
- `make pull`, `make pull-chirps`, and `make pull-not-chirps` **delete transferred clips from the Pi by default**.
- To keep clips on the Pi, use `KEEP=1`:

```bash
make pull KEEP=1
make pull-chirps KEEP=1
```

### Step 3: Review and label clips

Mark a specific clip:

```bash
make mark-chirp CLIP=clips/clip_2025-01-01_12-00-00.wav
make mark-not-chirp CLIP=clips/clip_2025-01-01_12-00-00.wav
```

Mark latest unreviewed event:

```bash
make mark-chirp-latest
make mark-not-chirp-latest
```

Interactive review (play multiple clips and classify):

```bash
make evaluate FILES="clips/clip1.wav clips/clip2.wav"
```

### Step 4: Generate report

```bash
make report
```

Or run pull + report in one command:

```bash
make workflow
```

## 5) Baseline management

Use named baselines if your environment changes (day/night, fan on/off, etc.).

```bash
make baseline-list
make baseline-create NAME=daytime DURATION=10 DESC="Daytime ambient"
make baseline-switch NAME=daytime
make baseline-show
make baseline-validate
```

Notes:
- baseline creation temporarily stops the monitor service, then restarts it.
- switching baselines restarts the service.

## 6) Classification maintenance (optional)

If chirp accuracy drifts:

1. collect better training samples in:
   - `training/chirp/`
   - `training/not_chirp/`
2. retrain and deploy fingerprint:
   ```bash
   make train
   make deploy-restart
   ```
3. validate performance:
   ```bash
   python3 scripts/validate_classification.py
   ```

If using ML classifier workflow:

```bash
make train-ml
make deploy-ml-restart
```

## 7) Optional email reporting

1. configure `email` settings in `config.json` (or environment variables)
2. install timer:
   ```bash
   make install-email-timer
   ```
3. verify and test:
   ```bash
   make email-timer-status
   make email-report-test
   make email-report
   ```

See [EMAIL_SETUP.md](EMAIL_SETUP.md) for provider-specific details.

## 8) Useful recovery commands

```bash
make logs           # Follow service logs
make audio-check    # Validate audio levels (stops/restarts service)
make debug-state    # Dump runtime debug info from Pi
make restart        # Regenerate + redeploy service file, then restart
```

If you need to capture a chirp around a known time:

```bash
make capture-chirp TIME="2025-01-15 14:30"
```

`TIME` is interpreted in US East Coast timezone (EST/EDT).

## 9) Where data lives

- Event log: `data/events.csv`
- Baseline: `data/baseline.json`
- Fingerprint model: `data/chirp_fingerprint.json`
- ML files: `data/chirp_model.pkl`, `data/chirp_scaler.pkl`, metadata JSON
- Event clips: `clips/clip_*.wav`
- Generated reports: `reports/chirp_report_*.md`

## 10) Quick command cheat sheet

```bash
make status            # Service status
make logs              # Live logs
make pull KEEP=1       # Pull clips without deleting from Pi
make chirps-recent     # Chirps in the last 24h
make report            # Generate chirp report
make baseline-set      # Recalibrate baseline
make help              # Full command list
```
