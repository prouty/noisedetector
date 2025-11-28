# Load environment variables from .env file if it exists
-include .env
export

# Default values if not set in .env
PI_USER ?= piuser
PI_HOSTNAME ?= raspberrypi.local
PI_HOST ?= $(PI_USER)@$(PI_HOSTNAME)
PI_DIR ?= /home/$(PI_USER)/projects/noisedetector
LOCAL_DIR ?= $(HOME)/projects/noisedetector

.PHONY: pull train deploy deploy-restart restart reload stop start report workflow

pull:
	@echo "==> Pulling events.csv and clips from Pi..."
	rsync -avz $(PI_HOST):$(PI_DIR)/data/events.csv $(LOCAL_DIR)/data/
	rsync -avzu --exclude='.DS_Store' --include="*/" --include="clip_*.wav" --exclude="*" \
		$(PI_HOST):$(PI_DIR)/clips/ $(LOCAL_DIR)/clips/

pull-chirps:
	@echo "==> Pulling events.csv to identify chirps..."
	@rsync -avz $(PI_HOST):$(PI_DIR)/data/events.csv $(LOCAL_DIR)/data/events.csv.tmp > /dev/null 2>&1
	@echo "==> Extracting chirp clip filenames..."
	@cd $(LOCAL_DIR) && source venv/bin/activate && python3 scripts/pull_chirps.py data/events.csv.tmp > /tmp/chirp_clips.txt
	@if [ -s /tmp/chirp_clips.txt ]; then \
		echo "==> Transferring $$(wc -l < /tmp/chirp_clips.txt) chirp clips..."; \
		rsync -avz --files-from=/tmp/chirp_clips.txt \
			$(PI_HOST):$(PI_DIR)/clips/ $(LOCAL_DIR)/clips/; \
		echo "==> Done! Chirp clips saved to $(LOCAL_DIR)/clips/"; \
	else \
		echo "==> No chirps found in events.csv"; \
	fi
	@rm -f $(LOCAL_DIR)/data/events.csv.tmp /tmp/chirp_clips.txt

train:
	@echo "==> Training chirp fingerprint from training/chirp/*.wav..."
	cd $(LOCAL_DIR) && source venv/bin/activate && python3 scripts/train_chirp_fingerprint.py

deploy:
	@echo "==> Deploying chirp_fingerprint.json to Pi..."
	rsync -avz $(LOCAL_DIR)/data/chirp_fingerprint.json $(PI_HOST):$(PI_DIR)/data/
	@echo "==> Note: Service restart required for new fingerprint to take effect"
	@echo "==> Run 'make restart' to restart the service"

deploy-restart: deploy restart
	@echo "==> Deployed fingerprint and restarted service"

reload:
	@echo "==> Reloading noise-monitor service on Pi..."
	ssh $(PI_HOST) 'cd $(PI_DIR) && sudo systemctl daemon-reload'

restart:
	@echo "==> Restarting noise-monitor service on Pi..."
	ssh $(PI_HOST) 'cd $(PI_DIR) && sudo systemctl restart noise-monitor'

stop:
	@echo "==> Stopping noise-monitor service on Pi..."
	ssh $(PI_HOST) 'cd $(PI_DIR) && sudo systemctl stop noise-monitor'

start:
	@echo "==> Starting noise-monitor service on Pi..."
	ssh $(PI_HOST) 'cd $(PI_DIR) && sudo systemctl start noise-monitor'

status:
	@echo "==> Checking status of noise-monitor service on Pi..."
	ssh $(PI_HOST) 'cd $(PI_DIR) && sudo systemctl status noise-monitor'

logs:
	@echo "==> Viewing logs of noise-monitor service on Pi..."
	ssh $(PI_HOST) 'cd $(PI_DIR) && sudo journalctl -u noise-monitor -f'

fix-deps:
	@echo "==> Fixing NumPy dependencies on Pi..."
	ssh $(PI_HOST) 'mkdir -p $(PI_DIR)/deploy'
	scp deploy/fix_numpy_deps.sh $(PI_HOST):$(PI_DIR)/deploy/
	ssh $(PI_HOST) 'cd $(PI_DIR) && chmod +x deploy/fix_numpy_deps.sh && ./deploy/fix_numpy_deps.sh'

report:
	@echo "==> Generating chirp report from events.csv..."
	cd $(LOCAL_DIR) && source venv/bin/activate && python3 scripts/generate_chirp_report.py

rediagnose:
	@echo "==> Re-classifying all events in events.csv..."
	cd $(LOCAL_DIR) && source venv/bin/activate && python3 scripts/rediagnose_events.py

rediagnose-report: rediagnose report
	@echo "==> Re-classified events and generated updated report"

audio-check:
	@echo "==> Stopping noise-monitor service (audio device needed)..."
	@ssh $(PI_HOST) 'cd $(PI_DIR) && sudo systemctl stop noise-monitor' || true
	@echo "==> Validating audio capture levels on Pi..."
	@ssh $(PI_HOST) 'cd $(PI_DIR) && python3 audio_analysis.py validate-levels'
	@echo "==> Restarting noise-monitor service..."
	@ssh $(PI_HOST) 'cd $(PI_DIR) && sudo systemctl start noise-monitor'

chirps:
	@echo "==> Checking for detected chirps..."
	cd $(LOCAL_DIR) && source venv/bin/activate && python3 scripts/check_chirps.py

chirps-recent:
	@echo "==> Checking for recent chirps (last 24 hours)..."
	cd $(LOCAL_DIR) && source venv/bin/activate && python3 scripts/check_chirps.py --recent 24

health:
	@echo "==> Running system health check..."
	cd $(LOCAL_DIR) && source venv/bin/activate && python3 scripts/health_check.py

debug-state:
	@echo "==> Dumping system state for debugging..."
	cd $(LOCAL_DIR) && source venv/bin/activate && python3 scripts/debug_state.py

init:
	@echo "==> Initializing Python virtual environment locally..."
	python3 -m venv venv

shell:
	@echo "==> Activating virtual environment and starting zsh locally..."
	@source venv/bin/activate && exec zsh

email-report:
	@echo "==> Running email report on Pi..."
	ssh $(PI_HOST) 'cd $(PI_DIR) && python3 scripts/email_report.py'

email-report-test:
	@echo "==> Testing email report on Pi (no email sent)..."
	ssh $(PI_HOST) 'cd $(PI_DIR) && python3 scripts/email_report.py --no-email'

install-email-timer:
	@echo "==> Installing email report timer on Pi..."
	scp systemd/email-report.service systemd/email-report.timer $(PI_HOST):$(PI_DIR)/systemd/
	ssh $(PI_HOST) "sudo cp $(PI_DIR)/systemd/email-report.service $(PI_DIR)/systemd/email-report.timer /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable --now email-report.timer"

email-timer-status:
	@echo "==> Checking email report timer status on Pi..."
	ssh $(PI_HOST) 'systemctl status email-report.timer'

email-timer-logs:
	@echo "==> Viewing email report logs on Pi..."
	ssh $(PI_HOST) 'journalctl -u email-report.service -n 50'

workflow: pull report
