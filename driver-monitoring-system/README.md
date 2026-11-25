# Driver Monitoring System - BASIC v1

Pi 4 camera watches the driver. Laptop shows Green / Yellow / Red.

## BASIC v1 - In this version
- Drowsiness: eyes closed detection, with calibration + smoothing fix for false drowsy bug
- 3 levels: GREEN Safe, YELLOW Warning, RED Danger
- Voice warning in the vehicle (espeak on Pi)
- Laptop dashboard for 1 vehicle, with alert + photo

## LATER - Not in v1, upgrade later if needed
- Fleet Monitoring Dashboard (multiple vehicles at once)
- Authorized driver check - face recognition, admin alert if another driver drives
- Yawning / MAR, PERCLOS, head pose
- Better night / glasses handling

## Why the old camera showed drowsy when driver was active
It decided on 1 frame. v1 fixes it: 5 sec calibration, eyes must stay closed 1.5 sec continuously, and low-confidence frames are ignored.

## Run - laptop first
```
pip install Flask requests
python laptop/app.py
python laptop/test_basic.py
# Open http://127.0.0.1:5000 - you should see GREEN, YELLOW, RED
```

## Run - Pi
```
# Edit pi/config.json: fleet_server_url = http://YOUR_LAPTOP_IP:5000
sudo apt install -y espeak
python3 -m pip install opencv-python-headless numpy requests
python3 pi/dms_basic.py
```

## Files
- pi/dms_basic.py - basic detector, Green/Yellow/Red, voice
- pi/config.json - laptop URL, vehicle_id, driver
- laptop/app.py - dashboard server
- laptop/templates/dashboard.html
- pi/later-upgrade/ - face enroll + voice module, for later only, do not use in v1
