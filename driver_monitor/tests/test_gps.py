"""gpsd JSON parsing and stale-fix reporting."""

from __future__ import annotations

import json
import time
import unittest

from dms.config import DMSConfig
from dms.gps import GPSClient, GPSFix, disabled_gps


class FixTests(unittest.TestCase):
    def test_waiting_without_fix(self) -> None:
        fix = GPSFix()
        self.assertFalse(fix.has_fix)
        self.assertEqual(fix.text(), "GPS: waiting")
        payload = fix.as_dict()
        self.assertFalse(payload["fix"])
        self.assertTrue(payload["stale"])

    def test_fix_text_and_staleness(self) -> None:
        now = time.monotonic()
        fix = GPSFix(
            lat=-34.6037,
            lon=-58.3816,
            mode=3,
            satellites=9,
            updated=now,
        )
        self.assertTrue(fix.has_fix)
        self.assertEqual(fix.text(now, 5.0), "GPS: -34.60370, -58.38160")
        self.assertTrue(fix.stale(now + 6.0, 5.0))
        self.assertEqual(fix.text(now + 6.0, 5.0), "GPS: waiting")
        payload = fix.as_dict(now, 5.0)
        self.assertTrue(payload["fix"])
        self.assertEqual(payload["mode_label"], "3D")
        self.assertEqual(payload["satellites"], 9)

    def test_disabled_payload(self) -> None:
        payload = disabled_gps()
        self.assertFalse(payload["enabled"])
        self.assertEqual(payload["text"], "GPS: off")
        self.assertIsNone(payload["lat"])


class GPSClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DMSConfig()
        self.config.gps_source = "gpsd"
        self.client = GPSClient(self.config)

    def tpv(self, **values) -> None:
        payload = {"class": "TPV", "time": "2026-01-01T00:00:00Z"}
        payload.update(values)
        self.client._handle_line(json.dumps(payload))

    def test_tpv_and_sky(self) -> None:
        self.client._handle_line("{not json")
        self.tpv(mode=1, lat=0.0, lon=0.0)
        self.assertFalse(self.client.fix().has_fix)
        self.tpv(mode=3, lat=12.3456789, lon=-99.1234567, alt=42.5, speed=11.2)
        self.client._handle_line(
            json.dumps(
                {
                    "class": "SKY",
                    "satellites": [
                        {"used": True},
                        {"used": False},
                        {"used": True},
                    ],
                }
            )
        )
        fix = self.client.fix()
        self.assertAlmostEqual(fix.lat, 12.3456789, places=6)
        self.assertAlmostEqual(fix.lon, -99.1234567, places=6)
        self.assertEqual(fix.mode, 3)
        self.assertEqual(fix.satellites, 2)
        self.assertAlmostEqual(fix.speed_mps, 11.2, places=2)
        snapshot = self.client.snapshot()
        self.assertTrue(snapshot["fix"])
        self.assertAlmostEqual(snapshot["lat"], 12.3456789, places=5)

    def test_off_mode_returns_disabled_payload(self) -> None:
        config = DMSConfig()
        client = GPSClient(config)
        self.assertFalse(client.enabled)
        self.assertEqual(client.snapshot()["text"], "GPS: off")
        client.start()
        self.assertIsNone(client._thread)
        client.stop()


if __name__ == "__main__":
    unittest.main()
