#!/usr/bin/env bash

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    set -a  # automatically export all variables
    # Filter out comments and empty lines, then source
    tmpfile=$(mktemp)
    # Remove lines that start with # (with optional leading whitespace) or are empty
    sed -e 's/^[[:space:]]*#.*$//' -e '/^[[:space:]]*$/d' .env > "$tmpfile"
    source "$tmpfile"
    rm "$tmpfile"
    set +a
fi

# Use environment variables with defaults
PI_USER="${PI_USER:-prouty}"
PI_HOSTNAME="${PI_HOSTNAME:-raspberrypi.local}"
PI_HOST="${PI_USER}@${PI_HOSTNAME}"
REMOTE_DIR="${PI_DIR:-/home/prouty/projects/noisedetector}"

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
	--exclude 'chirp_report*.md' \
	--exclude 'validation_results.csv' \
	--exclude 'clip_analysis.csv' \
	--exclude 'tuning_results.csv' \
	--exclude '__pycache__' \
	--exclude 'venv' \
	--exclude '.git' \
	./ \
	"$PI_HOST":"$REMOTE_DIR"



