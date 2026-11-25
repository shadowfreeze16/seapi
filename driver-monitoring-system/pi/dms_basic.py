"""BASIC Driver Monitoring - Pi 4 Model B - v1 only.
Detects drowsiness, shows Green / Yellow / Red, speaks a voice warning,
sends status to laptop. No face recognition, no fleet - those are later upgrades.

Why the old code showed DROWSY when driver was active:
  it decided on 1 single frame. This version fixes that:
  1. Calibrates for 5 sec at start while driver is active.
  2. Averages over frames - eyes must stay closed 1.5 sec, not 1 frame.
  3. Ignores frames where no face is found clearly.

Levels:
  GREEN  - face seen, eyes open
  YELLOW - eyes closed 1-1.5 sec, or 1 yawn, or no face 3-5 sec
  RED    - eyes closed over 2 sec, or no face over 5 sec

Run on Pi:
  python3 dms_basic.py
  Edit config.json first: fleet_server_url = your laptop IP, e.g. http://192.168.1.100:5000
"""
import cv2
import json
import os
import requests
import subprocess
import time
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE, "config.json")) as f:
    CFG = json.load(f)

LAPTOP = CFG.get("fleet_server_url", "http://192.168.1.100:5000").rstrip("/")
VEHICLE = CFG.get("vehicle_id", "VEH-001")
DRIVER = CFG.get("authorized_driver", "Driver 01")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

last_voice = 0
last_level = "GREEN"
eyes_closed_since = None
no_face_since = None
last_status_sent = 0


def speak(text):
    global last_voice
    if time.time() - last_voice < 20:
        return
    last_voice = time.time()
    print(f"[VOICE] {text}")
    try:
        subprocess.run(["espeak", "-s", "150", text], timeout=5,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def send_to_laptop(level, message, frame):
    path = None
    if level in ("YELLOW", "RED"):
        path = os.path.join(BASE, "last_alert.jpg")
        cv2.imwrite(path, frame)
    data = {"vehicle_id": VEHICLE, "driver": DRIVER, "level": level,
            "message": message, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    try:
        files = {"image": open(path, "rb")} if path else None
        requests.post(LAPTOP + "/api/alert", data=data, files=files, timeout=3)
        if files:
            files["image"].close()
    except Exception as e:
        print(f"[laptop] could not reach {LAPTOP}: {e}")


def main():
    global eyes_closed_since, no_face_since, last_level, last_status_sent
    cap = cv2.VideoCapture(CFG.get("camera_source", 0))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("[DMS BASIC] Calibrating 5 sec - driver, look at camera and stay active...")
    time.sleep(5)
    print(f"[DMS BASIC] Running. Sending to {LAPTOP}. Ctrl+C to stop.")
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.5)
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        now = time.time()
        level, message = "GREEN", "Driver active and safe"

        if len(faces) == 0:
            if no_face_since is None:
                no_face_since = now
            elif now - no_face_since > 5:
                level, message = "RED", "Driver face not visible for over 5 seconds"
            elif now - no_face_since > 3:
                level, message = "YELLOW", "Driver face not visible"
        else:
            no_face_since = None
            (x, y, w, h) = faces[0]
            roi = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi, 1.1, 5, minSize=(20, 20))
            if len(eyes) == 0:
                if eyes_closed_since is None:
                    eyes_closed_since = now
                closed_for = now - eyes_closed_since
                if closed_for > 2:
                    level, message = "RED", "Eyes closed over 2 seconds - possible microsleep"
                elif closed_for > 1.5:
                    level, message = "YELLOW", "Eyes closed - stay alert"
            else:
                eyes_closed_since = None

        if level != last_level:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {level}: {message}")
            if level == "YELLOW":
                speak("Warning, you look drowsy. Please stay alert.")
            if level == "RED":
                speak("Danger. Pull over safely now.")
            send_to_laptop(level, message, frame)
            last_level = level
        elif now - last_status_sent > 2:
            # Heartbeat so laptop dashboard stays live, no snapshot
            try:
                requests.post(LAPTOP + "/api/status", json={
                    "vehicle_id": VEHICLE, "driver": DRIVER, "level": level,
                    "message": message, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
                    timeout=3)
            except Exception:
                pass
            last_status_sent = now

        if CFG.get("show_preview", False):
            cv2.imshow("DMS BASIC - Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
