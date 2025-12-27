# Load environment variables from .env file if it exists
-include .env
export

# Default values if not set in .env
PI_USER ?= piuser
PI_HOSTNAME ?= raspberrypi.local
PI_HOST ?= $(PI_USER)@$(PI_HOSTNAME)
PI_DIR ?= /home/$(PI_USER)/projects/noisedetector
LOCAL_DIR ?= $(HOME)/projects/noisedetector

.PHONY: help pull pull-chirps pull-not-chirps train train-ml train-ml-svm train-capture-ml deploy deploy-ml deploy-restart deploy-ml-restart restart reload stop start status logs logs-refactored fix-deps report rediagnose rediagnose-report compare-classifiers mark-chirp mark-chirp-latest mark-not-chirp mark-not-chirp-latest evaluate audio-check chirps chirps-recent health baseline-list baseline-create baseline-delete baseline-switch baseline-show baseline-analyze baseline-validate baseline-set baseline-set-duration debug-state init shell email-report email-report-test install-email-timer email-timer-status email-timer-logs workflow test test-capture-ml test-features test-email test-reporting test-core capture-chirp extract-segment

help:
	@echo "Noise Detector - Makefile Commands"
	@echo "===================================="
	@echo ""
	@echo "Service Management (on Raspberry Pi):"
	@echo "  make start              Start monitoring service"
	@echo "  make stop               Stop monitoring service"
	@echo "  make restart            Deploy service file and restart (updates systemd)"
	@echo "  make status             Check service status"
	@echo "  make logs               View live service logs"
	@echo "  make reload             Reload systemd daemon (after service file changes)"
	@echo ""
	@echo "Data & Reports:"
	@echo "  make pull               Pull events.csv and clips (<=10s) from Pi"
	@echo "  make pull-chirps        Pull only clips identified as chirps"
	@echo "  make pull-not-chirps    Pull only non-chirp clips"
	@echo "  make report             Generate chirp report from events.csv"
	@echo "  make workflow           Pull data + generate report"
	@echo ""
	@echo "Training & Classification:"
	@echo "  make train              Train fingerprint from training/chirp/*.wav"
	@echo "  make train-ml           Train ML model (Random Forest)"
	@echo "  make train-ml-svm       Train ML model (SVM)"
	@echo "  make train-capture-ml   Train ML model for capture decision"
	@echo "  make test-ml            Test ML model on training data"
	@echo ""
	@echo "Deployment:"
	@echo "  make deploy             Deploy code to Pi (rsync)"
	@echo "  make deploy-ml          Deploy ML model files to Pi"
	@echo "  make deploy-restart     Deploy code + restart service"
	@echo "  make deploy-ml-restart Deploy ML model + restart service"
	@echo ""
	@echo "Quick Status & Analysis:"
	@echo "  make chirps             Show detected chirps in events.csv"
	@echo "  make chirps-recent      Show chirps from last 24 hours"
	@echo "  make health             System health check (dependencies, config, disk)"
	@echo "  make audio-check        Validate audio capture levels on Pi"
	@echo "  make capture-chirp       Retroactively capture chirp from specific timestamp (on Pi)"
	@echo "  make extract-segment     Extract specific segment from a clip file"
	@echo "  make debug-state        Show current system state"
	@echo ""
	@echo "Baseline Management:"
	@echo "  make baseline-list      List all saved baselines"
	@echo "  make baseline-create    Create new baseline (interactive)"
	@echo "  make baseline-set       Set baseline (10s duration)"
	@echo "  make baseline-set-duration  Set baseline (custom duration)"
	@echo "  make baseline-show      Show current baseline"
	@echo "  make baseline-switch    Switch to different baseline"
	@echo "  make baseline-delete    Delete a baseline"
	@echo "  make baseline-analyze   Analyze baseline stability"
	@echo "  make baseline-validate  Validate baseline configuration"
	@echo ""
	@echo "Event Management:"
	@echo "  make mark-chirp         Mark events as chirps (interactive)"
	@echo "  make mark-chirp-latest  Mark latest unreviewed event as chirp"
	@echo "  make mark-not-chirp     Mark events as not chirps (interactive)"
	@echo "  make mark-not-chirp-latest  Mark latest unreviewed event as not chirp"
	@echo "  make rediagnose         Re-classify training files"
	@echo "  make rediagnose-report  Re-classify + generate report"
	@echo "  make compare-classifiers  Compare ML vs fingerprint classifiers"
	@echo "  make evaluate           Evaluate classification accuracy"
	@echo ""
	@echo "Email Reports:"
	@echo "  make email-report       Send email report manually"
	@echo "  make email-report-test  Test email report (no email sent)"
	@echo "  make install-email-timer Install automated email timer (every 2 hours)"
	@echo "  make email-timer-status Check email timer status"
	@echo "  make email-timer-logs   View email report logs"
	@echo "  make email-timer-stop   Stop email timer"
	@echo "  make email-timer-disable  Disable email timer"
	@echo "  make email-timer-off    Stop and disable email timer"
	@echo ""
	@echo "Testing:"
	@echo "  make test               Run all tests"
	@echo "  make test-capture-ml    Run ML capture tests"
	@echo "  make test-features      Run feature extraction tests"
	@echo "  make test-email         Run email functionality tests"
	@echo "  make test-reporting    Run reporting tests"
	@echo "  make test-core          Run all core module tests"
	@echo ""
	@echo "Development:"
	@echo "  make init               Create virtual environment"
	@echo "  make shell               Activate venv and start shell"
	@echo "  make fix-deps           Fix NumPy/Pandas dependencies on Pi"
	@echo ""
	@echo "Configuration:"
	@echo "  Create .env file to customize Pi connection:"
	@echo "    PI_USER=prouty"
	@echo "    PI_HOSTNAME=raspberrypi.local"
	@echo "    PI_DIR=/home/prouty/projects/noisedetector"
	@echo ""
	@echo "For detailed documentation, see:"
	@echo "  docs/README.md          - Project overview"
	@echo "  docs/USAGE.md           - Complete command reference"
	@echo "  docs/CONFIGURATION.md   - Configuration guide"
	@echo "  docs/TROUBLESHOOTING.md - Common issues and solutions"

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
			echo "==> Filtering out clips that already exist locally..."; \
			cd $(LOCAL_DIR) && . venv/bin/activate && python3 scripts/filter_existing_clips.py < /tmp/short_clips.txt > /tmp/new_clips.txt 2> /tmp/filter_summary.txt; \
			if [ -s /tmp/filter_summary.txt ]; then cat /tmp/filter_summary.txt; fi; \
			if [ ! -s /tmp/new_clips.txt ]; then \
				echo "==> All clips already exist locally (in clips/, training/review/, training/chirp/, or training/not_chirp/)"; \
				rm -f /tmp/new_clips.txt; \
			else \
				> /tmp/existing_clips.txt; \
				found_count=0; \
				checked_count=0; \
				while IFS= read -r clip || [ -n "$$clip" ]; do \
					clip=$$(echo "$$clip" | tr -d '\r\n' | xargs); \
					if [ -z "$$clip" ]; then continue; fi; \
					checked_count=$$((checked_count + 1)); \
					ssh_result=$$(ssh $(PI_HOST) "test -f $(PI_DIR)/clips/$$clip && echo 'EXISTS' || echo 'NOT_FOUND'" 2>/dev/null); \
					if [ "$$ssh_result" = "EXISTS" ]; then \
						echo "$$clip" >> /tmp/existing_clips.txt; \
						found_count=$$((found_count + 1)); \
					fi; \
				done < /tmp/new_clips.txt; \
				echo "  Checked $$checked_count new clips, found $$found_count on Pi"; \
				if [ -s /tmp/existing_clips.txt ]; then \
					echo "==> Transferring $$found_count clips (<=10s)..."; \
					rsync -avz --files-from=/tmp/existing_clips.txt \
						$(PI_HOST):$(PI_DIR)/clips/ $(LOCAL_DIR)/clips/; \
					echo "==> Done! Clips saved to $(LOCAL_DIR)/clips/"; \
				else \
					echo "==> No clips found on Pi (all may have been moved/reviewed)"; \
				fi; \
				rm -f /tmp/existing_clips.txt /tmp/new_clips.txt /tmp/filter_summary.txt; \
			fi; \
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
		echo "==> Filtering out clips that already exist locally..."; \
		cd $(LOCAL_DIR) && . venv/bin/activate && python3 scripts/filter_existing_clips.py < /tmp/chirp_clips.txt > /tmp/new_chirp_clips.txt 2> /tmp/filter_summary.txt; \
		if [ -s /tmp/filter_summary.txt ]; then cat /tmp/filter_summary.txt; fi; \
		if [ ! -s /tmp/new_chirp_clips.txt ]; then \
			echo "==> All chirp clips already exist locally (in clips/, training/review/, training/chirp/, or training/not_chirp/)"; \
		else \
			new_count=$$(wc -l < /tmp/new_chirp_clips.txt | tr -d ' '); \
			echo "==> Transferring $$new_count new chirp clips..."; \
			rsync -avz --files-from=/tmp/new_chirp_clips.txt \
				$(PI_HOST):$(PI_DIR)/clips/ $(LOCAL_DIR)/clips/; \
			echo "==> Done! Chirp clips saved to $(LOCAL_DIR)/clips/"; \
		fi; \
		rm -f /tmp/new_chirp_clips.txt /tmp/filter_summary.txt; \
	else \
		echo "==> No chirps found in events.csv"; \
	fi
	@rm -f $(LOCAL_DIR)/data/events.csv.tmp /tmp/chirp_clips.txt

pull-not-chirps:
	@echo "==> Pulling events.csv to identify non-chirps..."
	@rsync -avz $(PI_HOST):$(PI_DIR)/data/events.csv $(LOCAL_DIR)/data/events.csv.tmp > /dev/null 2>&1
	@echo "==> Extracting non-chirp clip filenames (<=10s)..."
	@cd $(LOCAL_DIR) && . venv/bin/activate && python3 scripts/pull_not_chirps.py data/events.csv.tmp --max-duration 10.0 > /tmp/not_chirp_clips.txt
	@mkdir -p $(LOCAL_DIR)/training/not_chirp
	@if [ -s /tmp/not_chirp_clips.txt ]; then \
		echo "==> Filtering out clips that already exist locally..."; \
		cd $(LOCAL_DIR) && . venv/bin/activate && python3 scripts/filter_existing_clips.py < /tmp/not_chirp_clips.txt > /tmp/new_not_chirp_clips.txt 2> /tmp/filter_summary.txt; \
		if [ -s /tmp/filter_summary.txt ]; then cat /tmp/filter_summary.txt; fi; \
		if [ ! -s /tmp/new_not_chirp_clips.txt ]; then \
			echo "==> All non-chirp clips already exist locally (in clips/, training/review/, training/chirp/, or training/not_chirp/)"; \
		else \
			new_count=$$(wc -l < /tmp/new_not_chirp_clips.txt | tr -d ' '); \
			echo "==> Transferring $$new_count new non-chirp clips to training/not_chirp/..."; \
			rsync -avz --files-from=/tmp/new_not_chirp_clips.txt \
				$(PI_HOST):$(PI_DIR)/clips/ $(LOCAL_DIR)/training/not_chirp/; \
			echo "==> Done! Non-chirp clips saved to $(LOCAL_DIR)/training/not_chirp/"; \
		fi; \
		rm -f /tmp/new_not_chirp_clips.txt /tmp/filter_summary.txt; \
	else \
		echo "==> No non-chirps found in events.csv"; \
	fi
	@rm -f $(LOCAL_DIR)/data/events.csv.tmp /tmp/not_chirp_clips.txt

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

test-features:
	@echo "==> Running feature extraction tests..."
	cd $(LOCAL_DIR) && . venv/bin/activate && pytest tests/test_audio_io.py tests/test_audio_features.py -v

test-email:
	@echo "==> Running email functionality tests..."
	cd $(LOCAL_DIR) && . venv/bin/activate && pytest tests/test_email.py -v

test-reporting:
	@echo "==> Running reporting tests..."
	cd $(LOCAL_DIR) && . venv/bin/activate && pytest tests/test_reporting.py -v

test-core:
	@echo "==> Running all core module tests..."
	cd $(LOCAL_DIR) && . venv/bin/activate && pytest tests/test_audio_io.py tests/test_audio_features.py tests/test_email.py tests/test_reporting.py -v

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
	@echo "==> Generating service file from template..."
	@./deploy/generate_service.sh noise-monitor
	@echo "==> Deploying service file and restarting noise-monitor on Pi..."
	@scp /tmp/noise-monitor.service $(PI_HOST):/tmp/
	@ssh $(PI_HOST) "sudo cp /tmp/noise-monitor.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl restart noise-monitor"

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
	@echo "==> Showing last 50 lines, then following new logs (Ctrl+C to exit)..."
	@ssh $(PI_HOST) 'echo "=== Service Status ===" && sudo systemctl status noise-monitor --no-pager -l | head -10 && echo "" && echo "=== Recent Logs ===" && sudo journalctl -u noise-monitor -n 50 --no-pager && echo "" && echo "=== Following new logs (Ctrl+C to exit) ===" && sudo journalctl -u noise-monitor -f'


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

evaluate:
	@echo "==> Evaluating audio clips..."
	@echo "Usage: make evaluate FILES=\"clips/clip1.wav clips/clip2.wav clips/clip3.wav\""
	@if [ -z "$(FILES)" ]; then \
		echo "Error: FILES not set. Example: make evaluate FILES=\"clips/clip1.wav clips/clip2.wav\""; \
		exit 1; \
	fi
	@chirp_count=0; \
	not_chirp_count=0; \
	error_count=0; \
	total=0; \
	for clip in $(FILES); do \
		total=$$((total + 1)); \
		echo ""; \
		echo "==========================================="; \
		echo "Clip $$total: $$clip"; \
		echo "==========================================="; \
		if [ ! -f "$$clip" ]; then \
			echo "ERROR: File not found: $$clip"; \
			error_count=$$((error_count + 1)); \
			continue; \
		fi; \
		echo "Playing audio..."; \
		if ! play "$$clip" > /dev/null 2>&1; then \
			echo "ERROR: Failed to play $$clip (is 'play' command available?)"; \
			error_count=$$((error_count + 1)); \
			continue; \
		fi; \
		while true; do \
			echo ""; \
			printf "[C]hirp or [N]o Chirp? "; \
			read -r response < /dev/tty || read -r response; \
			case "$$response" in \
				[Cc]) \
					echo "Marking as chirp..."; \
					if $(MAKE) -s mark-chirp CLIP="$$clip" > /dev/null 2>&1; then \
						chirp_count=$$((chirp_count + 1)); \
						echo "✓ Marked as chirp"; \
					else \
						error_count=$$((error_count + 1)); \
						echo "✗ Failed to mark as chirp"; \
					fi; \
					break; \
					;; \
				[Nn]) \
					echo "Marking as not chirp..."; \
					if $(MAKE) -s mark-not-chirp CLIP="$$clip" > /dev/null 2>&1; then \
						not_chirp_count=$$((not_chirp_count + 1)); \
						echo "✓ Marked as not chirp"; \
					else \
						error_count=$$((error_count + 1)); \
						echo "✗ Failed to mark as not chirp"; \
					fi; \
					break; \
					;; \
				*) \
					echo "Invalid input. Please type 'C' or 'c' for Chirp, 'N' or 'n' for No Chirp."; \
					;; \
			esac; \
		done; \
	done; \
	echo ""; \
	echo "==========================================="; \
	echo "Evaluation Complete"; \
	echo "==========================================="; \
	echo "Total clips: $$total"; \
	echo "  Chirps: $$chirp_count"; \
	echo "  Not Chirps: $$not_chirp_count"; \
	if [ $$error_count -gt 0 ]; then \
		echo "  Errors: $$error_count"; \
	fi

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
	@echo "==> Generating email service file from template..."
	@./deploy/generate_service.sh email-report
	@echo "==> Installing email report timer on Pi..."
	@scp /tmp/email-report.service systemd/email-report.timer $(PI_HOST):/tmp/
	@ssh $(PI_HOST) "sudo cp /tmp/email-report.service /tmp/email-report.timer /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable --now email-report.timer"

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

email-timer-stop:
	@echo "==> Stopping email report timer on Pi..."
	ssh $(PI_HOST) 'sudo systemctl stop email-report.timer'

email-timer-disable:
	@echo "==> Disabling email report timer on Pi..."
	ssh $(PI_HOST) 'sudo systemctl disable email-report.timer'

email-timer-off: email-timer-stop email-timer-disable
	@echo "==> Email timer stopped and disabled"

workflow: pull report

capture-chirp:
	@echo "==> Capturing chirp from specified timestamp on Pi..."
	@if [ -z "$(TIME)" ]; then \
		echo "Error: TIME not set. Example: make capture-chirp TIME='2025-01-15 14:30'"; \
		echo "  Time should be in USA East Coast timezone (EST/EDT)"; \
		exit 1; \
	fi
	@ssh $(PI_HOST) 'cd $(PI_DIR) && python3 scripts/capture_chirp_at_time.py "$(TIME)"'

extract-segment:
	@echo "==> Extracting segment from clip..."
	@if [ -z "$(CLIP)" ] || [ -z "$(START)" ] || [ -z "$(END)" ]; then \
		echo "Error: CLIP, START, and END must be set."; \
		echo "  Example: make extract-segment CLIP=clips/clip_2025-12-27_14-30-00.wav START=45 END=55"; \
		echo "  Optional: PADDING=5 (default: 2 seconds)"; \
		echo "  Optional: UPDATE_EVENTS=1 (update events.csv)"; \
		exit 1; \
	fi
	@PADDING=$${PADDING:-2}; \
	UPDATE_FLAG=$$([ -n "$$UPDATE_EVENTS" ] && echo "--update-events" || echo ""); \
	python3 scripts/extract_chirp_segment.py "$(CLIP)" $(START) $(END) --padding $$PADDING $$UPDATE_FLAG
