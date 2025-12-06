#!/usr/bin/env bash

# Generate systemd service files from templates
# Usage: ./deploy/generate_service.sh [service-name]
# If no service name provided, generates all services

# Get the script's directory and find the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load environment variables from .env file if it exists
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a  # automatically export all variables
    tmpfile=$(mktemp)
    sed -e 's/^[[:space:]]*#.*$//' -e '/^[[:space:]]*$/d' "$PROJECT_ROOT/.env" > "$tmpfile"
    source "$tmpfile"
    rm "$tmpfile"
    set +a
fi

# Use environment variables with defaults
PI_USER="${PI_USER:-prouty}"
PI_DIR="${PI_DIR:-/home/${PI_USER}/projects/noisedetector}"
SERVICE_USER="${SERVICE_USER:-${PI_USER}}"
SERVICE_WORKING_DIR="${SERVICE_WORKING_DIR:-${PI_DIR}}"

generate_service() {
    local service_name=$1
    local template_file="$PROJECT_ROOT/systemd/${service_name}.service.example"
    local output_file="/tmp/${service_name}.service"
    
    if [ ! -f "$template_file" ]; then
        echo "Error: Template file not found: $template_file" >&2
        return 1
    fi
    
    sed -e "s|\${SERVICE_WORKING_DIR}|${SERVICE_WORKING_DIR}|g" \
        -e "s|\${SERVICE_USER}|${SERVICE_USER}|g" \
        "$template_file" > "$output_file"
    
    echo "Generated: $output_file"
    echo "  User: ${SERVICE_USER}"
    echo "  Working Directory: ${SERVICE_WORKING_DIR}"
}

# Generate requested service or all services
if [ $# -eq 0 ]; then
    # Generate all services
    generate_service "noise-monitor"
    generate_service "email-report"
else
    generate_service "$1"
fi

