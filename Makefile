# Load environment variables from .env file if it exists
-include .env
export

# Default values if not set in .env
PI_USER ?= piuser
PI_HOSTNAME ?= raspberrypi.local
PI_HOST ?= $(PI_USER)@$(PI_HOSTNAME)
PI_DIR ?= /home/$(PI_USER)/projects/noisedetector
LOCAL_DIR ?= $(HOME)/projects/noisedetector

.PHONY: pull pull-chirps train train-ml train-ml-svm train-capture-ml deploy deploy-ml deploy-restart deploy-ml-restart restart reload stop start status logs fix-deps report rediagnose rediagnose-report compare-classifiers mark-chirp mark-chirp-latest mark-not-chirp mark-not-chirp-latest audio-check chirps chirps-recent health baseline-list baseline-create baseline-delete baseline-switch baseline-show baseline-analyze baseline-validate baseline-set baseline-set-duration debug-state init shell email-report email-report-test install-email-timer email-timer-status email-timer-logs workflow test test-capture-ml

pull:
	@echo "==> Pulling events.csv and clips (<=10s) from Pi..."
	@if [ -f $(LOCAL_DIR)/data/events.csv ]; then \
		echo "==> Backing up local events.csv..."; \
		cp $(LOCAL_DIR)/data/events.csv $(LOCAL_DIR)/data/events.csv.backup; \
	fi
	@echo "==> Pulling remote events.csv..."
	@rsync -avz $(PI_HOST):$(PI_DIR)/data/events.csv $(LOCAL_DIR)/data/events.csv.remote > /dev/null 2>&1
	@if [ -f $(LOCAL_DIR)/data/events.csv.remote ]; then \
		if [ -f $(LOCAL_DIR)/data/events.csv ]; then \
			echo "==> Merging local and remote events.csv (preserving reviewed status)..."; \
			cd $(LOCAL_DIR) && . venv/bin/activate && python3 scripts/merge_events.py \
				data/events.csv.backup data/events.csv.remote data/events.csv; \
		else \
			echo "==> No local events.csv found, using remote as-is..."; \
			mv $(LOCAL_DIR)/data/events.csv.remote $(LOCAL_DIR)/data/events.csv; \
		fi; \
	else \
		echo "==> Warning: Could not pull remote events.csv"; \
	fi
	@echo "==> Filtering clips by duration (<=10s)..."
	@cd $(LOCAL_DIR) && . venv/bin/activate && python3 scripts/pull_short_clips.py data/events.csv > /tmp/short_clips.txt
	@if [ -s /tmp/short_clips.txt ]; then \
		echo "==> Checking which clips exist on Pi (excluding moved/reviewed clips)..."; \
		total_clips=$$(wc -l < /tmp/short_clips.txt | tr -d ' '); \
		echo "  Found $$total_clips clips in events.csv (<=10s)"; \
		if ! ssh $(PI_HOST) "test -d $(PI_DIR)/clips" 2>/dev/null; then \
			echo "  [ERROR] Clips directory not found on Pi: $(PI_DIR)/clips"; \
			echo "  Check PI_HOST and PI_DIR settings"; \
		else \
			> /tmp/existing_clips.txt; \
			found_count=0; \
			checked_count=0; \
			while IFS= read -r clip || [ -n "$$clip" ]; do \
				clip=$$(echo "$$clip" | tr -d '\r\n' | xargs); \
				echo "Checking clip: $(PI_DIR)/clips/$$clip"; \
				if [ -z "$$clip" ]; then continue; fi; \
				checked_count=$$((checked_count + 1)); \
				ssh_result=$$(ssh $(PI_HOST) "test -f $(PI_DIR)/clips/$$clip && echo 'EXISTS' || echo 'NOT_FOUND'" 2>/dev/null); \
				if [ "$$ssh_result" = "EXISTS" ]; then \
					echo "$$clip" >> /tmp/existing_clips.txt; \
					found_count=$$((found_count + 1)); \
				fi; \
			done < /tmp/short_clips.txt; \
			echo "  Checked $$checked_count clips, found $$found_count on Pi"; \
			if [ $$found_count -gt 0 ]; then \
				echo "  Verifying /tmp/existing_clips.txt has $$found_count lines..."; \
				file_lines=$$(wc -l < /tmp/existing_clips.txt | tr -d ' '); \
				echo "  File actually contains $$file_lines lines"; \
			fi; \
			if [ -s /tmp/existing_clips.txt ]; then \
				echo "  Found $$found_count clips on Pi"; \
				echo "==> Transferring $$found_count clips (<=10s)..."; \
				rsync -avz --files-from=/tmp/existing_clips.txt \
					$(PI_HOST):$(PI_DIR)/clips/ $(LOCAL_DIR)/clips/; \
				echo "==> Done! Clips saved to $(LOCAL_DIR)/clips/"; \
			else \
				echo "==> No clips found on Pi (all may have been moved/reviewed)"; \
				echo "  Checked $$total_clips clips from events.csv"; \
			fi; \
			rm -f /tmp/existing_clips.txt; \
		fi; \
	else \
		echo "==> No clips <=10s found in events.csv"; \
	fi
	@rm -f $(LOCAL_DIR)/data/events.csv.remote /tmp/short_clips.txt
	@echo "==> Backup saved to $(LOCAL_DIR)/data/events.csv.backup (delete manually if not needed)"

pull-chirps:
	@echo "==> Pulling events.csv to identify chirps..."
	@rsync -avz $(PI_HOST):$(PI_DIR)/data/events.csv $(LOCAL_DIR)/data/events.csv.tmp > /dev/null 2>&1
	@echo "==> Extracting chirp clip filenames..."
	@cd $(LOCAL_DIR) && . venv/bin/activate && python3 scripts/pull_chirps.py data/events.csv.tmp > /tmp/chirp_clips.txt
	@if [ -s /tmp/chirp_clips.txt ]; then \
		echo "==> Transferring $$(wc -l < /tmp/chirp_clips.txt) chirp clips..."; \
		rsync -avz --files-from=/tmp/chirp_clips.txt \
			$(PI_HOST):$(PI_DIR)/clips/ $(LOCAL_DIR)/clips/; \
		echo "==> Done! Chirp clips saved to $(LOCAL_DIR)/clips/"; \
	else \
		echo "==> No chirps found in events.csv"; \
	fi
	@rm -f $(LOCAL_DIR)/data/events.csv.tmp /tmp/chirp_clips.txt

test-ml:
	@echo "==> Testing ML model on training data..."
	cd $(LOCAL_DIR) && . venv/bin/activate && python3 scripts/test_chirp_ml.py --training

train:
	@echo "==> Training chirp fingerprint from training/chirp/*.wav..."
	cd $(LOCAL_DIR) && . venv/bin/activate && python3 scripts/train_chirp_fingerprint.py

train-ml:
	@echo "==> Training ML model for chirp classification..."
	cd $(LOCAL_DIR) && . venv/bin/activate && python3 scripts/train_chirp_ml.py

train-capture-ml:
	@echo "==> Training ML model for capture decision..."
	cd $(LOCAL_DIR) && . venv/bin/activate && python3 scripts/train_capture_ml.py

test:
	@echo "==> Running all tests..."
	cd $(LOCAL_DIR) && . venv/bin/activate && pytest tests/ -v

test-capture-ml:
	@echo "==> Running ML capture validation tests..."
	cd $(LOCAL_DIR) && . venv/bin/activate && pytest tests/test_capture_ml.py -v

train-ml-svm:
	@echo "==> Training SVM model for chirp classification..."
	cd $(LOCAL_DIR) && . venv/bin/activate && python3 scripts/train_chirp_ml.py --model-type svm

deploy:
	@echo "==> Deploying chirp_fingerprint.json to Pi..."
	rsync -avz $(LOCAL_DIR)/data/chirp_fingerprint.json $(PI_HOST):$(PI_DIR)/data/
	@echo "==> Note: Service restart required for new fingerprint to take effect"
	@echo "==> Run 'make restart' to restart the service"

deploy-ml:
	@echo "==> Deploying ML model files to Pi..."
	rsync -avz $(LOCAL_DIR)/data/chirp_model.pkl $(LOCAL_DIR)/data/chirp_scaler.pkl $(LOCAL_DIR)/data/chirp_model_metadata.json $(PI_HOST):$(PI_DIR)/data/
	@echo "==> Note: Service restart required for new model to take effect"
	@echo "==> Run 'make restart' to restart the service"

deploy-restart: deploy restart
	@echo "==> Deployed fingerprint and restarted service"

deploy-ml-restart: deploy-ml restart
	@echo "==> Deployed ML model and restarted service"

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
	cd $(LOCAL_DIR) && . venv/bin/activate && python3 scripts/generate_chirp_report.py

rediagnose:
	@echo "==> Re-classifying all events in events.csv..."
	cd $(LOCAL_DIR) && . venv/bin/activate && python3 scripts/rediagnose_events.py

rediagnose-report: rediagnose report
	@echo "==> Re-classified events and generated updated report"

compare-classifiers:
	@echo "==> Comparing ML vs fingerprint classifiers on reviewed events..."
	cd $(LOCAL_DIR) && . venv/bin/activate && python3 scripts/compare_classifiers.py

mark-chirp:
	@echo "==> Marking clip as valid chirp..."
	@echo "Usage: make mark-chirp CLIP=clips/clip_2025-01-01_12-00-00.wav"
	@echo "   Or: make mark-chirp-latest (marks latest unreviewed event)"
	@if [ -z "$(CLIP)" ]; then \
		echo "Error: CLIP not set. Example: make mark-chirp CLIP=clips/clip_2025-01-01_12-00-00.wav"; \
		exit 1; \
	fi
	cd $(LOCAL_DIR) && . venv/bin/activate && python3 scripts/mark_clip.py --chirp --clip "$(CLIP)"

mark-chirp-latest:
	@echo "==> Marking latest unreviewed event as valid chirp..."
	cd $(LOCAL_DIR) && . venv/bin/activate && python3 scripts/mark_clip.py --chirp --from-events

mark-not-chirp:
	@echo "==> Marking clip as not chirp..."
	@echo "Usage: make mark-not-chirp CLIP=clips/clip_2025-01-01_12-00-00.wav"
	@echo "   Or: make mark-not-chirp-latest (marks latest unreviewed event)"
	@if [ -z "$(CLIP)" ]; then \
		echo "Error: CLIP not set. Example: make mark-not-chirp CLIP=clips/clip_2025-01-01_12-00-00.wav"; \
		exit 1; \
	fi
	cd $(LOCAL_DIR) && . venv/bin/activate && python3 scripts/mark_clip.py --not-chirp --clip "$(CLIP)"

mark-not-chirp-latest:
	@echo "==> Marking latest unreviewed event as not chirp..."
	cd $(LOCAL_DIR) && . venv/bin/activate && python3 scripts/mark_clip.py --not-chirp --from-events

audio-check:
	@echo "==> Stopping noise-monitor service (audio device needed)..."
	@ssh $(PI_HOST) 'cd $(PI_DIR) && sudo systemctl stop noise-monitor' || true
	@echo "==> Validating audio capture levels on Pi..."
	@ssh $(PI_HOST) 'cd $(PI_DIR) && python3 audio_analysis.py validate-levels'
	@echo "==> Restarting noise-monitor service..."
	@ssh $(PI_HOST) 'cd $(PI_DIR) && sudo systemctl start noise-monitor'

chirps:
	@echo "==> Checking for detected chirps..."
	cd $(LOCAL_DIR) && . venv/bin/activate && python3 scripts/check_chirps.py

chirps-recent:
	@echo "==> Checking for recent chirps (last 24 hours)..."
	cd $(LOCAL_DIR) && . venv/bin/activate && python3 scripts/check_chirps.py --recent 24

health:
	@echo "==> Running system health check on Pi..."
	ssh $(PI_HOST) 'cd $(PI_DIR) && python3 scripts/health_check.py'

baseline-list:
	@echo "==> Listing baselines on Pi..."
	ssh $(PI_HOST) 'cd $(PI_DIR) && python3 baseline.py list'

baseline-create:
	@echo "==> Creating baseline on Pi (this will stop the service temporarily)..."
	@echo "Usage: make baseline-create NAME=daytime [DURATION=10] [DESC=\"Daytime baseline\"]"
	@if [ -z "$(NAME)" ]; then \
		echo "Error: NAME not set. Example: make baseline-create NAME=daytime"; \
		exit 1; \
	fi
	@ssh $(PI_HOST) 'cd $(PI_DIR) && sudo systemctl stop noise-monitor' || true
	@echo "==> Collecting baseline data for '$(NAME)'..."
	@if [ -n "$(DESC)" ]; then \
		if [ -n "$(DURATION)" ]; then \
			ssh $(PI_HOST) 'cd $(PI_DIR) && python3 baseline.py create "$(NAME)" --duration $(DURATION) --description "$(DESC)"'; \
		else \
			ssh $(PI_HOST) 'cd $(PI_DIR) && python3 baseline.py create "$(NAME)" --duration 10 --description "$(DESC)"'; \
		fi; \
	else \
		if [ -n "$(DURATION)" ]; then \
			ssh $(PI_HOST) 'cd $(PI_DIR) && python3 baseline.py create "$(NAME)" --duration $(DURATION)'; \
		else \
			ssh $(PI_HOST) 'cd $(PI_DIR) && python3 baseline.py create "$(NAME)" --duration 10'; \
		fi; \
	fi
	@echo "==> Restarting noise-monitor service..."
	@ssh $(PI_HOST) 'cd $(PI_DIR) && sudo systemctl start noise-monitor'

baseline-delete:
	@echo "==> Deleting baseline on Pi..."
	@echo "Usage: make baseline-delete NAME=daytime"
	@if [ -z "$(NAME)" ]; then \
		echo "Error: NAME not set. Example: make baseline-delete NAME=daytime"; \
		exit 1; \
	fi
	ssh $(PI_HOST) 'cd $(PI_DIR) && python3 baseline.py delete "$(NAME)"'

baseline-switch:
	@echo "==> Switching baseline on Pi..."
	@echo "Usage: make baseline-switch NAME=daytime"
	@if [ -z "$(NAME)" ]; then \
		echo "Error: NAME not set. Example: make baseline-switch NAME=daytime"; \
		exit 1; \
	fi
	@ssh $(PI_HOST) 'cd $(PI_DIR) && python3 baseline.py switch "$(NAME)"'
	@echo "==> Restarting noise-monitor service for changes to take effect..."
	@ssh $(PI_HOST) 'cd $(PI_DIR) && sudo systemctl restart noise-monitor'

baseline-show:
	@echo "==> Showing baseline on Pi..."
	@if [ -n "$(NAME)" ]; then \
		ssh $(PI_HOST) 'cd $(PI_DIR) && python3 baseline.py show "$(NAME)"'; \
	else \
		ssh $(PI_HOST) 'cd $(PI_DIR) && python3 baseline.py show'; \
	fi

baseline-analyze:
	@echo "==> Analyzing baseline history on Pi..."
	@if [ -n "$(NAME)" ]; then \
		ssh $(PI_HOST) 'cd $(PI_DIR) && python3 baseline.py analyze "$(NAME)"'; \
	else \
		ssh $(PI_HOST) 'cd $(PI_DIR) && python3 baseline.py analyze'; \
	fi

baseline-validate:
	@echo "==> Validating baseline on Pi..."
	@if [ -n "$(NAME)" ]; then \
		ssh $(PI_HOST) 'cd $(PI_DIR) && python3 baseline.py validate "$(NAME)"'; \
	else \
		ssh $(PI_HOST) 'cd $(PI_DIR) && python3 baseline.py validate'; \
	fi

# Backward compatibility - creates/updates 'default' baseline
baseline-set:
	@echo "==> Setting default baseline on Pi (this will stop the service temporarily)..."
	@ssh $(PI_HOST) 'cd $(PI_DIR) && sudo systemctl stop noise-monitor' || true
	@echo "==> Collecting baseline data..."
	@ssh $(PI_HOST) 'cd $(PI_DIR) && python3 baseline.py set'
	@echo "==> Restarting noise-monitor service..."
	@ssh $(PI_HOST) 'cd $(PI_DIR) && sudo systemctl start noise-monitor'

baseline-set-duration:
	@echo "==> Setting default baseline on Pi with custom duration (this will stop the service temporarily)..."
	@echo "Usage: make baseline-set-duration DURATION=20"
	@if [ -z "$(DURATION)" ]; then \
		echo "Error: DURATION not set. Example: make baseline-set-duration DURATION=20"; \
		exit 1; \
	fi
	@ssh $(PI_HOST) 'cd $(PI_DIR) && sudo systemctl stop noise-monitor' || true
	@echo "==> Collecting baseline data for $(DURATION) seconds..."
	@ssh $(PI_HOST) 'cd $(PI_DIR) && python3 baseline.py set --duration $(DURATION)'
	@echo "==> Restarting noise-monitor service..."
	@ssh $(PI_HOST) 'cd $(PI_DIR) && sudo systemctl start noise-monitor'

debug-state:
	@echo "==> Dumping system state for debugging on Pi..."
	ssh $(PI_HOST) 'cd $(PI_DIR) && python3 scripts/debug_state.py'

init:
	@echo "==> Initializing Python virtual environment locally..."
	python3 -m venv venv

shell:
	@echo "==> Activating virtual environment and starting zsh locally..."
	@. venv/bin/activate && exec zsh

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

config-merge:
	@echo "==> Merging config.example.json with config.json..."
	@jq '. * input' config.example.json config.json > config.json.tmp && mv config.json.tmp config.json
	@echo "✓ Config merged (your values preserved, missing keys added)"

workflow: pull report
