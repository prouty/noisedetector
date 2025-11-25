# Useful Raspberry Pi Noise Monitor Commands

## SSH
ssh prouty@raspberrypi.local
hostname -I

## File Transfer (Mac ↔ Pi)
# Copy project from Pi to Mac
rsync -avz prouty@raspberrypi.local:/home/prouty/projects/noisedetector ~/projects/

# Push updates from Mac to Pi
rsync -avz ~/Projects/noisedetector/ prouty@raspberrypi.local:/home/prouty/projects/noisedetector/

## Systemd Service
sudo systemctl start noise-monitor
sudo systemctl stop noise-monitor
sudo systemctl restart noise-monitor
sudo systemctl status noise-monitor

# Follow logs live
journalctl -u noise-monitor -f

## Audio Debug
arecord -l
arecord -L
aplay test.wav

## Run Script Manually
cd ~/projects/noisedetector
python3 noise_detector.py monitor

## Pi Power / System
sudo reboot
sudo shutdown now
