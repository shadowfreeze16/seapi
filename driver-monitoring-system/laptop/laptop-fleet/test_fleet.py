"""Integration test suite for Fleet Monitoring Dashboard (real HTTP).

Sends real HTTP POST and GET requests to http://127.0.0.1:5000 matching
the exact payload format of dms.dashboard_client:
  - POST /api/session/start
  - POST /api/status
  - POST /api/event (with JPEG snapshot)
  - GET /
  - GET /api/fleet
  - POST /api/session/end
"""

from __future__ import annotations

import base64
import io
import os
import sys
import time
import unittest

import requests

BASE_URL = os.environ.get("FLEET_URL", "http://127.0.0.1:5000").rstrip("/")


def _ensure_server_running():
    """Verify live HTTP server on BASE_URL; start a background thread if not running."""
    try:
        r = requests.get(f"{BASE_URL}/", timeout=0.8)
        if r.status_code == 200:
            return None
    except Exception:
        pass

    import threading
    from app import app
    t = threading.Thread(
        target=lambda: app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False),
        daemon=True,
    )
    t.start()
    time.sleep(1.2)
    return t


def _make_sample_jpeg_base64() -> str:
    """Generate a valid base64-encoded JPEG image for snapshot tests."""
    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (320, 240), color=(26, 32, 44))
        draw = ImageDraw.Draw(img)
        draw.rectangle([10, 10, 310, 230], outline=(239, 68, 68), width=3)
        draw.text((30, 100), "DMS ALERT: VEH-202 DROWSY", fill=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        # Fallback 1x1 valid JPEG
        return (
            "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////"
            "////////////////////////////////////////////////////wgALCAABAAEBAREA"
            "/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA="
        )


class RealFleetDashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._bg_thread = _ensure_server_running()

    def test_live_fleet_http_flow(self):
        # 1. Start Session for VEH-101
        start_payload = {
            "session_id": "VEH-101",
            "started_at": "2026-11-24T22:30:00",
            "timestamp": "2026-11-24T22:30:00.000",
        }
        r = requests.post(f"{BASE_URL}/api/session/start", json=start_payload, timeout=5)
        self.assertEqual(r.status_code, 200, f"Failed /api/session/start: {r.text}")
        self.assertTrue(r.json().get("ok"))

        # 2. Status update for VEH-101 (GREEN, Attentive)
        status_101 = {
            "session_id": "VEH-101",
            "timestamp": "2026-11-24T22:30:01.000",
            "safety_level": "GREEN",
            "safety_label": "ATTENTIVE",
            "safety_reason": "NORMAL",
            "state": "NORMAL",
            "reason": "FORWARD",
            "message": "Driver attentive",
            "zone": "FORWARD",
            "face_present": True,
            "yaw": 0.8,
            "pitch": -1.2,
            "roll": 0.1,
            "eye_left": 0.92,
            "eye_right": 0.90,
            "perclos": 0.05,
            "blink_rate": 17.5,
            "closure_s": 0.0,
            "fps": 26.4,
            "identity_state": "MATCH",
            "driver_name": "Alice - Driver 01",
            "identity_score": 0.58,
        }
        r = requests.post(f"{BASE_URL}/api/status", json=status_101, timeout=5)
        self.assertEqual(r.status_code, 200, f"Failed /api/status for VEH-101: {r.text}")

        # 3. Status update for VEH-202 (YELLOW, Caution)
        status_202 = {
            "session_id": "VEH-202",
            "timestamp": "2026-11-24T22:30:02.000",
            "safety_level": "YELLOW",
            "safety_label": "CAUTION",
            "safety_reason": "DROWSINESS",
            "state": "DROWSY",
            "reason": "EYE_CLOSURE",
            "message": "DROWSINESS WARNING",
            "zone": "FORWARD",
            "face_present": True,
            "yaw": 1.5,
            "pitch": -2.0,
            "roll": 0.4,
            "eye_left": 0.18,
            "eye_right": 0.15,
            "perclos": 0.36,
            "blink_rate": 9.2,
            "closure_s": 1.6,
            "fps": 22.1,
            "identity_state": "MATCH",
            "driver_name": "Bob - Driver 02",
            "identity_score": 0.61,
        }
        r = requests.post(f"{BASE_URL}/api/status", json=status_202, timeout=5)
        self.assertEqual(r.status_code, 200, f"Failed /api/status for VEH-202: {r.text}")

        # 4. Event with real base64 snapshot for VEH-202
        event_202 = {
            "event_id": f"evt_{int(time.time()*1000)}",
            "session_id": "VEH-202",
            "timestamp": "2026-11-24T22:30:05",
            "event": "DROWSY",
            "kind": "DROWSY",
            "status": "DROWSY",
            "severity": "warning",
            "direction": "FORWARD",
            "message": "DROWSINESS WARNING - Eyes closed 1.6s",
            "safety_level": "YELLOW",
            "safety_reason": "DROWSINESS",
            "face_count": 1,
            "yaw": 1.5,
            "pitch": -2.0,
            "roll": 0.4,
            "eye_left": 0.18,
            "eye_right": 0.15,
            "perclos": 0.36,
            "blink_rate": 9.2,
            "closure_s": 1.6,
            "fps": 22.1,
            "identity_state": "MATCH",
            "driver_name": "Bob - Driver 02",
            "snapshot_jpeg_base64": _make_sample_jpeg_base64(),
        }
        r = requests.post(f"{BASE_URL}/api/event", json=event_202, timeout=5)
        self.assertEqual(r.status_code, 200, f"Failed /api/event: {r.text}")

        # 5. GET /api/fleet JSON check
        r = requests.get(f"{BASE_URL}/api/fleet", timeout=5)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertGreaterEqual(data["total_vehicles"], 2)
        veh_ids = [v["session_id"] for v in data["vehicles"]]
        self.assertIn("VEH-101", veh_ids)
        self.assertIn("VEH-202", veh_ids)

        v202 = next(v for v in data["vehicles"] if v["session_id"] == "VEH-202")
        self.assertEqual(v202["safety_level"], "YELLOW")
        self.assertEqual(v202["driver_name"], "Bob - Driver 02")
        self.assertIsNotNone(v202.get("snapshot_url"))

        # 6. GET / Dashboard HTML check
        r = requests.get(f"{BASE_URL}/", timeout=5)
        self.assertEqual(r.status_code, 200)
        html = r.text
        self.assertIn("VEH-101", html)
        self.assertIn("VEH-202", html)
        self.assertIn("Bob - Driver 02", html)
        self.assertIn("DROWSINESS WARNING", html)

        print(f"\n[OK] Real HTTP test passed on {BASE_URL}!")
        print("  - VEH-101 (Alice) posted GREEN")
        print("  - VEH-202 (Bob) posted YELLOW with snapshot")
        print("  - Dashboard at http://127.0.0.1:5000 is showing both vehicles live.")


if __name__ == "__main__":
    unittest.main()
