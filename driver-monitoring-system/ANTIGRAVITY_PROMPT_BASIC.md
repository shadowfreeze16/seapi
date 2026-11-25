# Antigravity Prompt - BASIC Driver Monitoring System v1 ONLY

Copy everything in the code block and paste into Antigravity. Do NOT build the full version yet.

```
Build a BASIC Driver Monitoring System v1 in Python. Keep it simple. Do not build fleet, face recognition, or advanced features yet. Those are later upgrades.

Hardware: Raspberry Pi 4 Model B, USB webcam, Windows laptop, same Wi-Fi, no internet needed.

## BASIC v1 - What to build
1. Pi detector: pi/dms_basic.py
   - Use OpenCV Haar face + eye cascades only. No MediaPipe, no dlib, no YOLO.
   - Camera 640x480. First calibrate for 5 seconds while driver is active.
   - Fix for false drowsy bug: do NOT decide on 1 frame. Eyes must stay closed continuously to count.
     Eyes closed 1.5 sec = YELLOW, over 2 sec = RED. No face 3-5 sec = YELLOW, over 5 sec = RED. Otherwise GREEN.
   - Levels: GREEN = Safe, YELLOW = Warning, RED = Danger.
   - Voice warning on Pi with espeak, 20 sec cooldown: YELLOW says "Warning, you look drowsy. Please stay alert.", RED says "Danger. Pull over safely now."
   - On level change, save snapshot and POST to laptop /api/alert. Every 2 sec POST heartbeat to laptop /api/status. Read laptop URL, vehicle_id, driver name from pi/config.json.
2. Laptop app: laptop/app.py with Flask, port 5000, host 0.0.0.0
   - POST /api/alert, POST /api/status, GET /, POST /api/test, GET /alerts/<filename>
   - Save alerts to laptop/alerts.json, keep last 200, save snapshots to laptop/alerts/
3. Dashboard: laptop/templates/dashboard.html, dark theme
   - One big card showing current level GREEN / YELLOW / RED with colour, vehicle, driver, message, time.
   - Alert history table below, auto-refresh every 5 seconds, Send test alerts button.
4. Test: laptop/test_basic.py sends GREEN, YELLOW, RED alerts to http://127.0.0.1:5000
5. README with laptop first, then Pi, how to find laptop IP with ipconfig, and how to test without Pi.

## File structure
driver-monitoring-system/
  pi/dms_basic.py
  pi/config.json
  pi/requirements-pi.txt
  laptop/app.py
  laptop/test_basic.py
  laptop/templates/dashboard.html
  README.md

## Constraints
- Python only, Flask only, no Docker, no cloud.
- No hardcoded laptop IP except example in config.json.
- Do not use em dashes. Use plain hyphens only.
- Verify: python -m py_compile all files, start app.py, run test_basic.py, confirm 3 alerts return 201 and dashboard returns 200.

## LATER UPGRADES - Do NOT build now, only list them in README
- Fleet Monitoring Dashboard for multiple vehicles
- Authorized driver face recognition, admin notification if another driver drives
- Yawning detection with MAR, PERCLOS, head pose
- Better camera / night / glasses handling
```
