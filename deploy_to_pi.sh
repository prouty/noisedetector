#!/usr/bin/env bash

PI_USER="prouty"
PI_HOST="raspberrypi.local"
REMOTE_DIR="/home/prouty/projects/noisedetector"

rsync -avz \
	--exclude '.DS_Store' \
	--exclude 'clips' \
	--exclude 'training' \
	--exclude 'old_events' \
	--exclude 'recordings' \
	--exclude 'events.csv' \
	--exclude '__pycache__' \
	./\
	"$PI_USER"@"$PI_HOST":"$REMOTE_DIR"

