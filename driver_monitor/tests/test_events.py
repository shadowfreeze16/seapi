"""CSV logging, photos, pruning and session summaries."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from dms.config import DMSConfig
from dms.events import LEGACY_FIELDS, LOG_FIELDS, V2_FIELDS, EventLogger
from dms.state import CRITICAL, DISTRACTED, EventRequest


def warning_request() -> EventRequest:
    return EventRequest(
        "SIDE_ATTENTION",
        DISTRACTED,
        "warning",
        "LEFT",
        "DISTRACTION WARNING",
    )


def critical_request() -> EventRequest:
    return EventRequest(
        "MICROSLEEP", CRITICAL, "critical", "FORWARD", "MICROSLEEP"
    )


class EventLoggerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = DMSConfig()
        self.config.event_dir = self.root / "events"
        self.config.log_file = self.config.event_dir / "dms_log.csv"
        self.frame = np.full((48, 64, 3), 100, dtype=np.uint8)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_creates_header(self) -> None:
        EventLogger(self.config)
        with self.config.log_file.open("r", encoding="utf-8") as handle:
            header = next(csv.reader(handle))
        self.assertEqual(header, LOG_FIELDS)
        self.assertEqual(header[:4], LEGACY_FIELDS)

    def test_legacy_log_is_rotated(self) -> None:
        self.config.event_dir.mkdir(parents=True)
        self.config.log_file.write_text(
            "timestamp,event,direction,face_count\n"
            "2026-10-02 13:09:00,SIDE_ATTENTION,LEFT,1\n",
            encoding="utf-8",
        )
        logger = EventLogger(self.config)
        rotated = list(self.config.event_dir.glob("dms_log_v1_*.csv"))
        self.assertEqual(len(rotated), 1)
        self.assertIn("SIDE_ATTENTION", rotated[0].read_text(encoding="utf-8"))
        self.assertIsNotNone(logger._rotated)
        with self.config.log_file.open("r", encoding="utf-8") as handle:
            self.assertEqual(next(csv.reader(handle)), LOG_FIELDS)

    def test_v2_log_is_rotated(self) -> None:
        self.config.event_dir.mkdir(parents=True, exist_ok=True)
        header = ",".join(V2_FIELDS)
        self.config.log_file.write_text(
            header + "\n"
            + ",".join("" for _ in V2_FIELDS) + "\n",
            encoding="utf-8",
        )
        logger = EventLogger(self.config)
        rotated = list(self.config.event_dir.glob("dms_log_v2_*.csv"))
        self.assertEqual(len(rotated), 1)
        with self.config.log_file.open("r", encoding="utf-8") as handle:
            self.assertEqual(next(csv.reader(handle)), LOG_FIELDS)

    def test_record_writes_safety_identity_gps_fields(self) -> None:
        logger = EventLogger(self.config)
        row = logger.record(
            warning_request(),
            frame=None,
            context={
                "safety_level": "YELLOW",
                "safety_reason": "DISTRACTION",
                "identity_state": "MATCH",
                "identity_name": "Alice",
                "identity_score": 0.7123,
                "gps": {
                    "lat": -34.60374,
                    "lon": -58.38162,
                    "text": "GPS: -34.60374, -58.38162",
                },
            },
            now=100.0,
        )
        assert row is not None
        self.assertEqual(row["safety_level"], "YELLOW")
        self.assertEqual(row["identity_name"], "Alice")
        self.assertEqual(row["identity_score"], 0.712)
        self.assertEqual(row["gps_lat"], -34.60374)
        self.assertEqual(row["gps_lon"], -58.38162)
        self.assertEqual(row["gps_text"], "GPS: -34.60374, -58.38162")
        with self.config.log_file.open("r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["safety_level"], "YELLOW")

    def test_record_writes_row_and_photo(self) -> None:
        logger = EventLogger(self.config)
        row = logger.record(
            warning_request(),
            frame=self.frame,
            context={
                "face_count": 1,
                "yaw": 31.234,
                "pitch": -4.0,
                "eye_left": 0.9,
                "perclos": 0.12,
            },
            now=100.0,
        )
        self.assertIsNotNone(row)
        self.assertEqual(row["event"], "SIDE_ATTENTION")
        self.assertEqual(row["direction"], "LEFT")
        self.assertEqual(row["yaw"], 31.23)
        self.assertEqual(row["face_count"], 1)
        self.assertTrue(Path(row["photo"]).exists())
        self.assertEqual(logger.counts["SIDE_ATTENTION"], 1)

        with self.config.log_file.open("r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], DISTRACTED)
        self.assertEqual(rows[0]["severity"], "warning")

    def test_warning_cooldown(self) -> None:
        logger = EventLogger(self.config)
        self.assertIsNotNone(logger.record(warning_request(), now=100.0))
        self.assertIsNone(logger.record(warning_request(), now=104.9))
        self.assertIsNotNone(logger.record(warning_request(), now=105.0))

    def test_critical_cooldown_is_shorter(self) -> None:
        logger = EventLogger(self.config)
        self.assertIsNotNone(logger.record(critical_request(), now=200.0))
        self.assertIsNone(logger.record(critical_request(), now=203.9))
        self.assertIsNotNone(logger.record(critical_request(), now=204.0))

    def test_record_without_frame(self) -> None:
        logger = EventLogger(self.config)
        row = logger.record(warning_request(), frame=None, now=300.0)
        self.assertIsNotNone(row)
        self.assertEqual(row["photo"], "")
        self.assertEqual(logger.photos, [])

    def test_prune_by_count(self) -> None:
        logger = EventLogger(self.config)
        self.config.max_events = 2
        self.config.max_event_mb = 1000.0
        paths = []
        for index in range(5):
            path = self.config.event_dir / f"2026010{index + 1}_000000_EVENT_{index}.jpg"
            path.write_bytes(b"x" * 32)
            os.utime(path, (1000 + index, 1000 + index))
            paths.append(path)
        removed = logger.prune()
        self.assertEqual(removed, 3)
        remaining = sorted(p.name for p in self.config.event_dir.glob("2026*.jpg"))
        self.assertEqual(remaining, [paths[3].name, paths[4].name])

    def test_prune_by_size(self) -> None:
        logger = EventLogger(self.config)
        self.config.max_events = 100
        self.config.max_event_mb = 0.0001  # 104 bytes
        for index in range(3):
            path = self.config.event_dir / f"2026010{index + 1}_000000_EVENT_{index}.jpg"
            path.write_bytes(b"x" * 60)
            os.utime(path, (1000 + index, 1000 + index))
        logger.prune()
        remaining = list(self.config.event_dir.glob("2026*.jpg"))
        self.assertLessEqual(len(remaining), 1)

    def test_close_writes_session_summary(self) -> None:
        logger = EventLogger(self.config)
        logger.record(warning_request(), self.frame, now=400.0)
        logger.record(critical_request(), self.frame, now=500.0)
        summary = logger.close({"frames": 42, "events": {"SIDE_ATTENTION": 1}})
        self.assertIsNotNone(summary)
        payload = json.loads(summary.read_text(encoding="utf-8"))
        self.assertEqual(payload["summary"]["frames"], 42)
        self.assertEqual(payload["event_counts"]["SIDE_ATTENTION"], 1)
        self.assertEqual(payload["event_counts"]["MICROSLEEP"], 1)
        self.assertEqual(payload["photos"], 2)
        self.assertIn("duration_seconds", payload)

    def test_save_session_disabled(self) -> None:
        self.config.save_session_summary = False
        logger = EventLogger(self.config)
        self.assertIsNone(logger.close())


if __name__ == "__main__":
    unittest.main()