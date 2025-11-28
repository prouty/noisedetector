# Email Report Setup

This guide explains how to set up automated email reports that run every 2 hours with summaries of detected clips and chirps.

## Quick Setup

1. **Configure email settings** (choose one method):

   **Option A: Environment variables (recommended for security)**
   ```bash
   # On the Raspberry Pi, add to ~/.bashrc or create /etc/systemd/system/email-report.service.d/email.conf
   export EMAIL_SMTP_SERVER="smtp.gmail.com"
   export EMAIL_SMTP_PORT="587"
   export EMAIL_SMTP_USERNAME="your-email@gmail.com"
   export EMAIL_SMTP_PASSWORD="your-app-password"  # Use app password for Gmail
   export EMAIL_FROM="your-email@gmail.com"
   export EMAIL_TO="recipient@example.com"
   export EMAIL_USE_TLS="true"
   ```

   **Option B: config.json**
   ```json
   {
     "email": {
       "smtp_server": "smtp.gmail.com",
       "smtp_port": 587,
       "smtp_username": "your-email@gmail.com",
       "smtp_password": "your-app-password",
       "from_address": "your-email@gmail.com",
       "to_address": "recipient@example.com",
       "use_tls": true,
       "report_hours": 2
     }
   }
   ```

2. **Install the timer service:**
   ```bash
   make install-email-timer
   ```

3. **Verify it's running:**
   ```bash
   make email-timer-status
   ```

## Manual Testing

Test the email report without sending email:
```bash
make email-report-test
```

Test with actual email:
```bash
make email-report
```

## Email Provider Setup

### Gmail
1. Enable 2-factor authentication
2. Generate an [App Password](https://myaccount.google.com/apppasswords)
3. Use the app password (not your regular password) in `EMAIL_SMTP_PASSWORD`
4. Settings:
   - SMTP Server: `smtp.gmail.com`
   - Port: `587`
   - TLS: `true`

### Other Providers
- **Outlook/Hotmail**: `smtp-mail.outlook.com`, port `587`
- **Yahoo**: `smtp.mail.yahoo.com`, port `587`
- **Custom SMTP**: Check your provider's documentation

## Timer Schedule

The timer runs:
- Every 2 hours at :00 (12:00, 2:00, 4:00, etc.)
- 2 hours after system boot
- If a run is missed, it runs 2 hours after the last successful run

## Report Contents

Each email report includes:
- Total number of clips created in the last 2 hours
- Number of events identified as chirps
- Detailed list of each chirp with:
  - Timestamp
  - Duration
  - Confidence score
  - Similarity score
  - Clip filename

## Troubleshooting

**Check timer status:**
```bash
make email-timer-status
```

**View recent email report logs:**
```bash
make email-timer-logs
```

**Manually trigger a report:**
```bash
make email-report
```

**Common issues:**
- **Email not sending**: Check SMTP credentials and firewall
- **Timer not running**: Check `systemctl status email-report.timer`
- **Permission errors**: Ensure user has read access to `events.csv`

## Disabling Email Reports

To stop the timer:
```bash
ssh prouty@raspberrypi "sudo systemctl stop email-report.timer && sudo systemctl disable email-report.timer"
```

