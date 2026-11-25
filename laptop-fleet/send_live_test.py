#!/usr/bin/env python3
"""Send live status stream for VEH-202 to the Fleet Dashboard.

Usage:
    python send_live_test.py [URL]

Behavior:
  - Connects to http://127.0.0.1:5000 (or custom URL)
  - Sends /api/session/start for VEH-202
  - Seconds 1-25: Streams GREEN attentive status every 1.0 sec (perclos=0.04, closure=0.0s)
  - Second 26:    Triggers a DROWSY warning event with an annotated snapshot image
  - Seconds 27-30: Streams YELLOW drowsy status so you can see the card turn amber with photo
  - Watch your browser at http://127.0.0.1:5000 update in real time!
"""

from __future__ import annotations

import base64
import io
import os
import sys
import time
from datetime import datetime

import requests

BASE_URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5000").rstrip("/")
SESSION_ID = "VEH-202"
DRIVER_NAME = "Bob - Driver 02"


def make_snapshot_base64() -> str:
    """Generate a 320x240 alert snapshot image."""
    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (320, 240), color=(30, 20, 20))
        draw = ImageDraw.Draw(img)
        # Red warning border
        draw.rectangle([5, 5, 315, 235], outline=(239, 68, 68), width=3)
        # Status header
        draw.rectangle([10, 10, 310, 45], fill=(239, 68, 68))
        draw.text((20, 18), "DMS ALERT: VEH-202 DROWSY", fill=(255, 255, 255))
        # Metrics text
        draw.text((20, 70), f"Driver: {DRIVER_NAME}", fill=(241, 245, 249))
        draw.text((20, 100), "Eye Closure: 1.62 seconds", fill=(251, 191, 36))
        draw.text((20, 130), "PERCLOS: 38.5% (Threshold: 35%)", fill=(251, 191, 36))
        draw.text((20, 160), "Head Pose: Yaw +2.1, Pitch -3.5", fill=(148, 163, 184))
        draw.text((20, 190), f"Timestamp: {datetime.now().strftime('%H:%M:%S')}", fill=(148, 163, 184))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return (
            "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////"
            "////////////////////////////////////////////////////wgALCAABAAEBAREA"
            "/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA="
        )


def main():
    print("=" * 60)
    print(f"  LIVE VEHICLE STREAM SIMULATOR: {SESSION_ID}")
    print(f"  Target: {BASE_URL}")
    print(f"  Open dashboard at {BASE_URL} to watch live updates!")
    print("=" * 60)

    # Check connection
    try:
        r = requests.get(f"{BASE_URL}/", timeout=3.0)
        if r.status_code != 200:
            print(f"[ERROR] Server returned HTTP {r.status_code}")
            return 1
    except Exception as exc:
        print(f"[ERROR] Could not reach {BASE_URL}: {exc}")
        print("Make sure Terminal 1 is running: python app.py")
        return 1

    # 1. Start Session
    now_iso = datetime.now().isoformat()
    try:
        requests.post(
            f"{BASE_URL}/api/session/start",
            json={"session_id": SESSION_ID, "started_at": now_iso, "timestamp": now_iso},
            timeout=3.0,
        )
        print(f"[00s] Session started for {SESSION_ID} ({DRIVER_NAME})")
    except Exception as exc:
        print(f"[ERROR] /api/session/start failed: {exc}")
        return 1

    total_seconds = 30
    for sec in range(1, total_seconds + 1):
        now_time = datetime.now().strftime("%H:%M:%S")

        if sec <= 25:
            # Normal driving: GREEN
            level = "GREEN"
            label = "ATTENTIVE"
            reason = "NORMAL"
            state = "NORMAL"
            msg = "Driver attentive and alert"
            perclos = 0.04
            closure_s = 0.0
            fps = 28.4
        else:
            # Drowsiness: YELLOW
            level = "YELLOW"
            label = "CAUTION"
            reason = "DROWSINESS"
            state = "DROWSY"
            msg = "DROWSINESS WARNING - Eyes closed > 1.5s"
            perclos = 0.38
            closure_s = 1.6
            fps = 26.8

        # At second 26, trigger the Event with snapshot
        if sec == 26:
            print("\n" + "!" * 50)
            print(f"[{sec:02d}s] TRIGGERING DROWSY ALERT WITH SNAPSHOT PHOTO...")
            print("!" * 50)
            event_payload = {
                "event_id": f"evt_drowsy_{int(time.time()*1000)}",
                "session_id": SESSION_ID,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "event": "DROWSY",
                "kind": "DROWSY",
                "status": "DROWSY",
                "severity": "warning",
                "direction": "FORWARD",
                "message": msg,
                "safety_level": "YELLOW",
                "safety_reason": "DROWSINESS",
                "face_count": 1,
                "yaw": 2.1,
                "pitch": -3.5,
                "roll": 0.4,
                "eye_left": 0.16,
                "eye_right": 0.14,
                "perclos": perclos,
                "blink_rate": 8.0,
                "closure_s": closure_s,
                "fps": fps,
                "driver_name": DRIVER_NAME,
                "identity_state": "MATCH",
                "snapshot_jpeg_base64": make_snapshot_base64(),
            }
            try:
                requests.post(f"{BASE_URL}/api/event", json=event_payload, timeout=3.0)
                print(f"[{sec:02d}s] Alert event logged! Check alert table in browser.")
            except Exception as e:
                print(f"[ERROR] Failed to post event: {e}")

        # Send regular 1.0s heartbeat status
        status_payload = {
            "session_id": SESSION_ID,
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "safety_level": level,
            "safety_label": label,
            "safety_reason": reason,
            "state": state,
            "reason": reason,
            "message": msg,
            "zone": "FORWARD",
            "face_present": True,
            "yaw": 1.2,
            "pitch": -1.0,
            "roll": 0.2,
            "eye_left": 0.92 if level == "GREEN" else 0.16,
            "eye_right": 0.90 if level == "GREEN" else 0.14,
            "perclos": perclos,
            "blink_rate": 16.0 if level == "GREEN" else 8.0,
            "closure_s": closure_s,
            "offroad_seconds": 0.0,
            "no_face_seconds": 0.0,
            "status_seconds": float(sec),
            "fps": fps,
            "identity_state": "MATCH",
            "driver_name": DRIVER_NAME,
            "identity_score": 0.62,
        }
        try:
            requests.post(f"{BASE_URL}/api/status", json=status_payload, timeout=3.0)
            tag = f"\033[92m{level}\033[0m" if level == "GREEN" else f"\033[93m{level}\033[0m"
            print(f"[{sec:02d}/30s] {now_time} -> {SESSION_ID}: {level} ({msg}) | PERCLOS={perclos:.2f} | FPS={fps:.1f}")
        except Exception as e:
            print(f"[{sec:02d}/30s] Failed to send status: {e}")

        time.sleep(1.0)

    print("\n" + "=" * 60)
    print("  30-second live test complete!")
    print(f"  VEH-202 card is now visible on {BASE_URL}.")
    print("  After 10 seconds of no status, watch it transition to OFFLINE.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[STOPPED] Live test cancelled.")
