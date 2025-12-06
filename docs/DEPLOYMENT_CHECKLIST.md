# Deployment Test Checklist

## Pre-Deployment
- [ ] Code committed and pushed
- [ ] Deploy to Pi: `make deploy`

## Basic Service Test
- [ ] Service starts: `make start` (or `ssh pi 'sudo systemctl start noise-monitor'`)
- [ ] Service is running: `make status` (should show "active (running)")
- [ ] No immediate crashes: Check logs for errors in first 30 seconds

## Import/Module Test
- [ ] No import errors in logs: `make logs | grep -i "import\|error\|traceback"` (should be clean)
- [ ] Core modules load: `ssh pi 'cd ~/projects/noisedetector && python3 -c "from core import create_classifier; print(\"OK\")"'`
- [ ] Classification functions available: `ssh pi 'cd ~/projects/noisedetector && python3 -c "from core.classifier import classify_event_is_chirp; print(\"OK\")"'`

## Functionality Test
- [ ] Audio capture working: Logs show "peak" and "rms" values updating
- [ ] Baseline tracking: Logs show baseline values (may take ~60 seconds to stabilize)
- [ ] Event detection: Make a noise (clap, speak) - should see "EVENT" status in logs
- [ ] Event classification: After event ends, check logs for classification result (CHIRP or noise)
- [ ] Event logging: Check `events.csv` has new entry after test event

## Quick Manual Test
```bash
# On Pi, test imports
cd ~/projects/noisedetector
python3 -c "from core import create_classifier; from core.classifier import classify_event_is_chirp; print('All imports OK')"

# Check service logs (should see continuous monitoring output)
make logs

# Trigger a test event (make noise), then check:
# 1. Logs show "EVENT" status
# 2. After event, logs show classification result
# 3. events.csv has new entry
```

## If Issues Found
- [ ] Check full error: `make logs | tail -50`
- [ ] Verify Python version: `ssh pi 'python3 --version'` (should be 3.9+)
- [ ] Check dependencies: `ssh pi 'cd ~/projects/noisedetector && pip3 list | grep -E "numpy|pandas"'`
- [ ] Verify config: `ssh pi 'cd ~/projects/noisedetector && python3 -c "import config_loader; print(config_loader.load_config())"'`

## Success Criteria
✅ Service runs without errors  
✅ Logs show continuous monitoring output  
✅ Events are detected and classified  
✅ No import errors or tracebacks

