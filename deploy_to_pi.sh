#!/usr/bin/env bash

PI_USER="prouty"
PI_HOST="raspberrypi.local"
REMOTE_DIR="/home/prouty/projects/noisedetector"

# This rsync command syncs all files and folders in the current directory (./)
# to the remote target, except for the files and directories explicitly excluded below.
rsync -avz \
	--exclude '.DS_Store' \
	--exclude 'clips' \
	--exclude 'training' \
	--exclude 'old_events' \
	--exclude 'recordings' \
	--exclude 'baseline.json' \
	--exclude 'events.csv' \
	--exclude '__pycache__' \
	--exclude 'venv' \
	--exclude '.git' \
	./ \
	"$PI_USER"@"$PI_HOST":"$REMOTE_DIR"



