# Load environment variables from .env file if it exists
-include .env
export

# Default values if not set in .env
PI_USER ?= piuser
PI_HOSTNAME ?= raspberrypi.local
PI_HOST ?= $(PI_USER)@$(PI_HOSTNAME)
PI_DIR ?= /home/$(PI_USER)/projects/noisedetector
LOCAL_DIR ?= $(HOME)/projects/noisedetector

.PHONY: pull train deploy restart report workflow

pull:
	@echo "==> Pulling events.csv and clips from Pi..."
	rsync -avz $(PI_HOST):$(PI_DIR)/events.csv $(LOCAL_DIR)/
	rsync -avzu --exclude='.DS_Store' --include="*/" --include="clip_*.wav" --exclude="*" \
		$(PI_HOST):$(PI_DIR)/clips/ $(LOCAL_DIR)/clips/

train:
	@echo "==> Training chirp fingerprint from training/chirp/*.wav..."
	cd $(LOCAL_DIR) && python3 train_chirp_fingerprint.py

deploy:
	@echo "==> Deploying chirp_fingerprint.json to Pi..."
	rsync -avz $(LOCAL_DIR)/chirp_fingerprint.json $(PI_HOST):$(PI_DIR)/

reload:
	@echo "==> Reloading noise-monitor service on Pi..."
	ssh $(PI_HOST) 'cd $(PI_DIR) && sudo systemctl daemon-reload'

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

report:
	@echo "==> Generating chirp report from events.csv..."
	cd $(LOCAL_DIR) && python3 generate_chirp_report.py

init:
	@echo "==> Initializing Python virtual environment locally..."
	python3 -m venv venv

shell:
	@echo "==> Activating virtual environment and starting zsh locally..."
	@source venv/bin/activate && exec zsh

workflow: pull report
