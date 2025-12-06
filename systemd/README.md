# Systemd Service Files

Service files in this directory are **generated from templates** using your `.env` configuration.

## Templates

- `*.service.example` - Template files with variables (${SERVICE_WORKING_DIR}, ${SERVICE_USER})
- `*.service` - Generated files (do not edit - they're generated from templates)

## Generating Service Files

Service files are automatically generated when you run:
```bash
make restart          # Generates and deploys noise-monitor.service
make install-email-timer  # Generates and deploys email-report.service
```

Or manually:
```bash
./deploy/generate_service.sh noise-monitor
./deploy/generate_service.sh email-report
```

## Configuration

Service files use these environment variables (from `.env`):
- `SERVICE_USER` - User to run service as (defaults to `PI_USER`)
- `SERVICE_WORKING_DIR` - Working directory (defaults to `PI_DIR`)

## Important

**Do not edit the `.service` files directly** - they are generated from templates.
Edit the `.example` files if you need to change the service configuration.

