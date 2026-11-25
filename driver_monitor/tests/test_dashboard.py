"""Dashboard HTTP client, payload builders and offline outbox."""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from dms.config import DMSConfig
from dms.dashboard_client import (
    DashboardClient,
    encode_snapshot,
    make_event_payload,
    make_status_payload,
)
from dms.safety import GREEN, SafetyState
from dms.state import NORMAL, DriverState, EventRequest


class CaptureHandler(BaseHTTPRequestHandler):
    requests: list[dict] = []
    lock = threading.Lock()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body)
        except ValueError:
            payload = {}
        with self.lock:
            self.requests.append({"path": self.path, "payload": payload})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def log_message(self, *args) -> None:
        pass


class PayloadTests(unittest.TestCase):
    def test_event_payload_fields(self) -> None:
        request = EventRequest("SIDE_ATTENTION", "DISTRACTED", "warning", "LEFT", "msg")
        row = {
            "event": request.kind,
            "yaw": 31.234,
            "perclos": 0.12,
            "photo": "/tmp/events/20260101_000000_SIDE_ATTENTION_LEFT.jpg",
        }
        safety = SafetyState()
        payload = make_event_payload(
            "session-1",
            request,
            row,
            safety,
            None,
            {"text": "GPS: waiting"},
            "base64data",
        )
        self.assertEqual(payload["session_id"], "session-1")
        self.assertEqual(payload["event"], "SIDE_ATTENTION")
        self.assertEqual(payload["severity"], "warning")
        self.assertEqual(payload["safety_level"], GREEN)
        self.assertEqual(payload["yaw"], 31.23)
        self.assertEqual(payload["perclos"], 0.12)
        self.assertEqual(payload["gps"]["text"], "GPS: waiting")
        self.assertEqual(payload["snapshot_jpeg_base64"], "base64data")
        self.assertEqual(
            payload["photo_name"], "20260101_000000_SIDE_ATTENTION_LEFT.jpg"
        )
        self.assertTrue(payload["event_id"])

    def test_status_payload_fields(self) -> None:
        driver = DriverState(status=NORMAL, zone="FORWARD", reason="FORWARD")
        payload = make_status_payload(
            "session-1",
            SafetyState(),
            driver,
            None,
            None,
            None,
            fps=18.4,
            gps={"text": "GPS: off"},
            voice_enabled=True,
            camera_setup={"calibrated": False},
            session_seconds=12.3,
        )
        self.assertEqual(payload["safety_level"], GREEN)
        self.assertEqual(payload["state"], NORMAL)
        self.assertEqual(payload["fps"], 18.4)
        self.assertTrue(payload["voice_enabled"])
        self.assertEqual(payload["gps"]["text"], "GPS: off")
        self.assertEqual(payload["session_seconds"], 12.3)

    def test_encode_snapshot(self) -> None:
        frame = np.full((480, 640, 3), 120, dtype=np.uint8)
        encoded = encode_snapshot(frame, max_width=320, quality=70)
        self.assertIsInstance(encoded, str)
        self.assertGreater(len(encoded), 100)
        self.assertIsNone(encode_snapshot(None))


class DashboardClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        CaptureHandler.requests = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), CaptureHandler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(
            target=self.server.serve_forever, daemon=True
        )
        self.thread.start()

        self.config = DMSConfig()
        self.config.dashboard_url = f"http://127.0.0.1:{self.port}"
        self.config.dashboard_interval = 0.05
        self.config.dashboard_timeout = 2.0
        self.config.dashboard_outbox = self.root / "dashboard_outbox.jsonl"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)
        self.temp.cleanup()

    def make_event(self, event_id: str = "event-1") -> dict:
        request = EventRequest("SIDE_ATTENTION", "DISTRACTED", "warning", "LEFT", "msg")
        payload = make_event_payload("session-test", request)
        payload["event_id"] = event_id
        return payload

    def test_posts_status_event_and_session(self) -> None:
        client = DashboardClient(self.config, "session-test")
        client.start({"session_id": "session-test"})
        client.set_status({"session_id": "session-test", "safety_level": GREEN})
        client.send_event(self.make_event())
        self.assertTrue(client.drain(3.0))
        client.end_session({"session_id": "session-test"})
        paths = [item["path"] for item in CaptureHandler.requests]
        self.assertIn("/api/session/start", paths)
        self.assertIn("/api/status", paths)
        self.assertIn("/api/event", paths)
        self.assertIn("/api/session/end", paths)
        event = [
            item["payload"]
            for item in CaptureHandler.requests
            if item["path"] == "/api/event"
        ][0]
        self.assertEqual(event["event_id"], "event-1")

    def test_disabled_client_is_noop(self) -> None:
        config = DMSConfig()
        client = DashboardClient(config)
        self.assertFalse(client.enabled)
        client.start()
        client.set_status({})
        client.send_event({})
        client.end_session()
        self.assertIsNone(client._thread)

    def test_offline_events_are_queued_and_replayed(self) -> None:
        offline = DMSConfig()
        offline.dashboard_url = "http://127.0.0.1:1"
        offline.dashboard_interval = 0.05
        offline.dashboard_timeout = 0.5
        offline.dashboard_outbox = self.config.dashboard_outbox
        client = DashboardClient(offline, "session-offline")
        client.start({"session_id": "session-offline"})
        client.send_event(self.make_event("queued-event"))
        self.assertTrue(client.drain(3.0))
        client.stop()
        self.assertTrue(offline.dashboard_outbox.exists())
        self.assertIn(
            "queued-event", offline.dashboard_outbox.read_text(encoding="utf-8")
        )

        replay = DashboardClient(self.config, "session-replay")
        replay.start({"session_id": "session-replay"})
        deadline = time.monotonic() + 3.0
        received = False
        while time.monotonic() < deadline:
            with CaptureHandler.lock:
                received = any(
                    item["path"] == "/api/event"
                    and item["payload"].get("event_id") == "queued-event"
                    for item in CaptureHandler.requests
                )
            if received:
                break
            time.sleep(0.05)
        replay.stop()
        self.assertTrue(received)
        self.assertFalse(self.config.dashboard_outbox.exists())


if __name__ == "__main__":
    unittest.main()
