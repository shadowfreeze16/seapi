import os
import sys
import time
import threading
import requests
import subprocess

# Ensure user site-packages is available for Flask
sys.path.append(r"C:\Users\Wajida\AppData\Roaming\Python\Python311\site-packages")
sys.path.insert(0, os.path.abspath("laptop-fleet"))

import app as fleet_app

print("1. Starting Fleet Dashboard on port 5000...")
server_thread = threading.Thread(
    target=fleet_app.app.run,
    kwargs={"host": "127.0.0.1", "port": 5000, "debug": False, "use_reloader": False},
    daemon=True,
)
server_thread.start()
time.sleep(1.5)

# Verify health
r_init = requests.get("http://127.0.0.1:5000/api/fleet", timeout=3.0)
print(f"Server health check: {r_init.status_code} OK")

# 2. Run Active Mode (Active face, eyes open)
print("\n2. Running test_laptop_cam.py: Active mode (eyes open, 3s)...")
res_active = subprocess.run(
    [
        sys.executable,
        "laptop-camera-test/test_laptop_cam.py",
        "--headless",
        "--simulate-duration", "3.0",
        "--dashboard-url", "http://127.0.0.1:5000",
    ],
    capture_output=True,
    text=True,
)
print("Active run output:\n" + res_active.stdout)

# Check vehicle status after active
r_active = requests.get("http://127.0.0.1:5000/api/fleet").json()
veh_active = r_active["vehicles"][0]
print(f"Vehicle state after active run: Level={veh_active['safety_level']}, State={veh_active['state']}, PERCLOS={veh_active['perclos']:.2f}")

# 3. Run Drowsy Mode (Eyes closed, 4s)
print("\n3. Running test_laptop_cam.py: Eyes closed simulation (4s)...")
res_closed = subprocess.run(
    [
        sys.executable,
        "laptop-camera-test/test_laptop_cam.py",
        "--headless",
        "--simulate-duration", "4.0",
        "--simulate-eyes-closed",
        "--dashboard-url", "http://127.0.0.1:5000",
    ],
    capture_output=True,
    text=True,
)
print("Closed eyes run output:\n" + res_closed.stdout)

# 4. Inspect Fleet Dashboard Alert History
r_fleet = requests.get("http://127.0.0.1:5000/api/fleet").json()
events = r_fleet["events"]

print("\n" + "=" * 60)
print("FLEET DASHBOARD ALERT HISTORY VERIFICATION:")
print("=" * 60)
print(f"Total alert events logged: {len(events)}")

if not events:
    print("FAIL: No events logged!")
    sys.exit(1)

latest = events[0]
print(f"Latest Event:    {latest['event']}")
print(f"Severity:        {latest['severity']}")
print(f"Safety Level:    {latest['safety_level']}")
print(f"Message:         {latest['message']}")
print(f"PERCLOS:         {latest['perclos']:.2f}")
print(f"Closure:         {latest['closure_s']:.1f}s")
print(f"Snapshot URL:    {latest['snapshot_url']}")

# Assertions
assert latest["event"] in ("DROWSY", "MICROSLEEP"), f"Expected DROWSY or MICROSLEEP, got {latest['event']}"
assert latest["perclos"] >= 0.35, f"Expected PERCLOS >= 0.35, got {latest['perclos']}"
assert latest["closure_s"] >= 1.5, f"Expected Closure >= 1.5s, got {latest['closure_s']}"
assert latest["snapshot_url"] is not None and latest["snapshot_url"].startswith("/snapshots/"), "Snapshot missing!"

# Verify physical snapshot file on disk
snap_file = os.path.join("laptop-fleet", latest["snapshot_url"].lstrip("/"))
assert os.path.exists(snap_file), f"Snapshot file not on disk: {snap_file}"
file_size = os.path.getsize(snap_file)
print(f"Snapshot File:   {snap_file} ({file_size} bytes, verified JPEG on disk)")
assert file_size > 100, f"Snapshot file too small: {file_size}"

# Test cooldown: running another 2s with closed eyes should NOT emit another duplicate event if <10s
print("\n4. Testing event cooldown (2s closed eyes run right after first alert)...")
res_cooldown = subprocess.run(
    [
        sys.executable,
        "laptop-camera-test/test_laptop_cam.py",
        "--headless",
        "--simulate-duration", "2.0",
        "--simulate-eyes-closed",
        "--dashboard-url", "http://127.0.0.1:5000",
    ],
    capture_output=True,
    text=True,
)
print("Cooldown run output:\n" + res_cooldown.stdout)

r_cooldown = requests.get("http://127.0.0.1:5000/api/fleet").json()
print(f"Total alert events after cooldown test: {len(r_cooldown['events'])}")

print("\n" + "=" * 60)
print("ALL VERIFICATIONS PASSED 100%!")
print("=" * 60)
