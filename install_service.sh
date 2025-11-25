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
PI_DIR="${PI_DIR:-/home/prouty/projects/noisedetector}"
SERVICE_USER="${SERVICE_USER:-${PI_USER}}"
SERVICE_WORKING_DIR="${SERVICE_WORKING_DIR:-${PI_DIR}}"

# Generate service file from template
sed -e "s|\${SERVICE_WORKING_DIR}|${SERVICE_WORKING_DIR}|g" \
    -e "s|\${SERVICE_USER}|${SERVICE_USER}|g" \
    noise-monitor.service.example > /tmp/noise-monitor.service

echo "Generated service file with:"
echo "  User: ${SERVICE_USER}"
echo "  Working Directory: ${SERVICE_WORKING_DIR}"
echo ""
echo "To install, run on the Pi:"
echo "  sudo cp /tmp/noise-monitor.service /etc/systemd/system/"
echo "  sudo systemctl daemon-reload"
echo "  sudo systemctl enable noise-monitor"
echo "  sudo systemctl start noise-monitor"
echo ""
echo "To verify the service is running, run:"
echo "  sudo systemctl status noise-monitor"
echo "  sudo journalctl -u noise-monitor -f"

