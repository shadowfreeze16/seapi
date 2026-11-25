"""State machine, zone tracking, calibration and event requests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dms.config import DMSConfig
from dms.eyes import EyeMetrics
from dms.state import (
    CRITICAL,
    DISTRACTED,
    DROWSY,
    FORWARD,
    NORMAL,
    NO_FACE,
    SHORT_GLANCE,
    EventRequest,
    StateMachine,
    ZoneTracker,
    attention_score,
)


def eyes(
    *,
    drowsy: bool = False,
    critical: bool = False,
    closure: float = 0.0,
) -> EyeMetrics:
    closed = drowsy or critical
    return EyeMetrics(
        valid=True,
        left_open=0.0 if closed else 1.0,
        right_open=0.0 if closed else 1.0,
        left_closed=closed,
        right_closed=closed,
        both_closed=closed,
        closure_seconds=closure,
        drowsy=drowsy,
        critical=critical,
    )


def drive(
    machine: StateMachine,
    start: float,
    end: float,
    *,
    step: float = 0.1,
    yaw: float = 0.0,
    pitch: float = 0.0,
    face: bool = True,
    eye_metrics: EyeMetrics | None = None,
    pose_valid: bool = True,
    events: list[EventRequest] | None = None,
):
    state = None
    now = start
    observations = eye_metrics or eyes()
    while now <= end + 1e-9:
        state = machine.update(
            now, yaw, pitch, 0.0, observations, face, pose_valid=pose_valid
        )
        if events is not None:
            events.extend(state.events)
        now += step
    return state


class ZoneTrackerTests(unittest.TestCase):
    def test_raw_zone(self) -> None:
        tracker = ZoneTracker(12.0, 12.0)
        self.assertEqual(tracker.raw_zone(0.0, 0.0), FORWARD)
        self.assertEqual(tracker.raw_zone(20.0, 0.0), "LEFT")
        self.assertEqual(tracker.raw_zone(-20.0, 0.0), "RIGHT")
        self.assertEqual(tracker.raw_zone(0.0, 20.0), "UP")
        self.assertEqual(tracker.raw_zone(0.0, -20.0), "DOWN")
        self.assertEqual(tracker.raw_zone(20.0, 20.0), "LEFT_UP")
        self.assertEqual(tracker.raw_zone(-20.0, -20.0), "RIGHT_DOWN")

    def test_debounce(self) -> None:
        tracker = ZoneTracker(12.0, 12.0, debounce=0.15)
        self.assertEqual(tracker.update(20.0, 0.0, 0.0), FORWARD)
        self.assertEqual(tracker.update(20.0, 0.0, 0.1), FORWARD)
        self.assertEqual(tracker.update(20.0, 0.0, 0.2), "LEFT")
        self.assertEqual(tracker.update(20.0, 0.0, 0.3), "LEFT")

    def test_candidate_reset_when_returning_forward(self) -> None:
        tracker = ZoneTracker(12.0, 12.0, debounce=0.15)
        tracker.update(20.0, 0.0, 0.0)
        self.assertEqual(tracker.update(0.0, 0.0, 0.1), FORWARD)
        self.assertEqual(tracker.update(20.0, 0.0, 0.2), FORWARD)
        self.assertEqual(tracker.update(20.0, 0.0, 0.4), "LEFT")

    def test_reset(self) -> None:
        tracker = ZoneTracker(12.0, 12.0)
        tracker.update(20.0, 0.0, 0.0)
        tracker.reset()
        self.assertEqual(tracker.zone, FORWARD)


class StateMachineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DMSConfig()
        self.machine = StateMachine(self.config)

    def test_forward_is_normal(self) -> None:
        state = drive(self.machine, 0.0, 1.0, yaw=5.0, pitch=-3.0)
        self.assertEqual(state.status, NORMAL)
        self.assertFalse(state.alert)
        self.assertEqual(state.reason, "FORWARD")

    def test_short_glance(self) -> None:
        state = drive(self.machine, 0.0, 1.0, yaw=20.0)
        self.assertEqual(state.status, SHORT_GLANCE)
        self.assertTrue(state.alert)
        self.assertFalse(state.alarm)

    def test_distraction_warning(self) -> None:
        collected: list[EventRequest] = []
        state = drive(
            self.machine, 0.0, 3.0, yaw=20.0, events=collected
        )
        self.assertEqual(state.status, DISTRACTED)
        self.assertTrue(state.alarm)
        self.assertIn("SIDE_ATTENTION", [event.kind for event in collected])

    def test_critical_distraction(self) -> None:
        collected: list[EventRequest] = []
        state = drive(
            self.machine, 0.0, 5.5, yaw=20.0, events=collected
        )
        self.assertEqual(state.status, CRITICAL)
        self.assertEqual(state.reason, "EXTREME_ATTENTION")
        self.assertIn(
            "EXTREME_ATTENTION", [event.kind for event in collected]
        )

    def test_extreme_angle_is_critical_sooner(self) -> None:
        state = drive(self.machine, 0.0, 2.4, yaw=60.0)
        self.assertEqual(state.status, CRITICAL)
        self.assertEqual(state.reason, "EXTREME_ATTENTION")

    def test_pitch_is_used(self) -> None:
        state = drive(self.machine, 0.0, 3.0, pitch=30.0)
        self.assertEqual(state.status, DISTRACTED)
        self.assertIn("UP", state.zone)

    def test_drowsy(self) -> None:
        collected: list[EventRequest] = []
        state = drive(
            self.machine,
            0.0,
            0.1,
            eye_metrics=eyes(drowsy=True, closure=1.4),
            events=collected,
        )
        self.assertEqual(state.status, DROWSY)
        self.assertTrue(state.alarm)
        self.assertIn("DROWSY", [event.kind for event in collected])

    def test_microsleep(self) -> None:
        collected: list[EventRequest] = []
        state = drive(
            self.machine,
            0.0,
            0.1,
            eye_metrics=eyes(critical=True, drowsy=True, closure=3.5),
            events=collected,
        )
        self.assertEqual(state.status, CRITICAL)
        self.assertEqual(state.reason, "MICROSLEEP")
        self.assertIn("MICROSLEEP", [event.kind for event in collected])

    def test_no_face_warning_and_critical(self) -> None:
        collected: list[EventRequest] = []
        state = drive(self.machine, 0.0, 2.6, face=False, events=collected)
        self.assertEqual(state.status, NO_FACE)
        self.assertIn("NO_FACE", [event.kind for event in collected])
        state = drive(self.machine, 2.7, 6.2, face=False, events=collected)
        self.assertEqual(state.status, CRITICAL)
        self.assertEqual(state.reason, "NO_FACE_CRITICAL")
        self.assertIn("NO_FACE_CRITICAL", [event.kind for event in collected])

    def test_pose_unavailable_is_not_an_alert(self) -> None:
        state = drive(self.machine, 0.0, 1.0, yaw=40.0, pose_valid=False)
        self.assertEqual(state.status, NORMAL)
        self.assertEqual(state.reason, "POSE_UNAVAILABLE")
        self.assertEqual(state.zone, "UNKNOWN")
        self.assertFalse(state.alert)

    def test_microsleep_is_detected_without_pose(self) -> None:
        state = drive(
            self.machine,
            0.0,
            0.1,
            pose_valid=False,
            eye_metrics=eyes(critical=True, drowsy=True, closure=3.5),
        )
        self.assertEqual(state.status, CRITICAL)
        self.assertEqual(state.reason, "MICROSLEEP")

    def test_critical_repeats(self) -> None:
        collected: list[EventRequest] = []
        drive(
            self.machine,
            0.0,
            10.0,
            eye_metrics=eyes(critical=True, drowsy=True, closure=4.0),
            events=collected,
        )
        kinds = [event.kind for event in collected]
        self.assertGreaterEqual(kinds.count("CRITICAL"), 2)
        self.assertIn("MICROSLEEP", kinds)

    def test_calibration(self) -> None:
        self.machine.calibrate(10.0, 5.0, -3.0)
        self.assertTrue(self.machine.calibrated)
        state = drive(self.machine, 0.0, 0.5, yaw=10.0, pitch=5.0)
        self.assertAlmostEqual(state.yaw, 0.0, places=5)
        self.assertAlmostEqual(state.pitch, 0.0, places=5)
        self.assertEqual(state.status, NORMAL)

    def test_calibration_roundtrip(self) -> None:
        self.machine.calibrate(-8.0, 4.0, 2.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.json"
            self.machine.save_calibration(path)
            payload = json.loads(path.read_text())
            self.assertEqual(payload["version"], 2)

            other = StateMachine(self.config)
            self.assertTrue(other.load_calibration(path))
            self.assertAlmostEqual(other.yaw_offset, -8.0)
            self.assertAlmostEqual(other.pitch_offset, 4.0)
            self.assertAlmostEqual(other.roll_offset, 2.0)
            self.assertTrue(other.calibrated)

            path.write_text("{broken", encoding="utf-8")
            self.assertFalse(other.load_calibration(path))

    def test_clear_calibration(self) -> None:
        self.machine.calibrate(5.0, 5.0, 5.0)
        self.machine.clear_calibration()
        self.assertFalse(self.machine.calibrated)
        self.assertEqual(self.machine.yaw_offset, 0.0)

    def test_reset(self) -> None:
        drive(self.machine, 0.0, 3.0, yaw=20.0)
        self.machine.reset()
        state = drive(self.machine, 10.0, 11.0)
        self.assertEqual(state.status, NORMAL)

    def test_attention_score(self) -> None:
        self.assertAlmostEqual(attention_score(0.0, 0.0, self.config), 1.0)
        self.assertAlmostEqual(
            attention_score(self.config.yaw_extreme, 0.0, self.config), 0.0
        )
        self.assertAlmostEqual(
            attention_score(0.0, self.config.pitch_extreme, self.config), 0.0
        )
        score = attention_score(14.0, 0.0, self.config)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)


if __name__ == "__main__":
    unittest.main()