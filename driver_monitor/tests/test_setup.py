"""Camera framing checks and automatic neutral-pose calibration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dms.config import DMSConfig
from dms.models import Face, HeadPose
from dms.setup import (
    FrameSetup,
    assess_frame,
    draw_setup_overlay,
    save_setup_calibration,
)


def textured_frame(value: int = 120, size: tuple[int, int] = (480, 640)) -> np.ndarray:
    rng = np.random.default_rng(7)
    low = max(0, value - 50)
    high = min(255, value + 50)
    return rng.integers(low, high, size=(*size, 3), dtype=np.uint8)


def make_face(box=(220, 150, 200, 200)) -> Face:
    x, y, w, h = box
    return Face(
        box=np.array(box, dtype=np.float32),
        landmarks=np.array(
            [
                [x + w * 0.3, y + h * 0.35],
                [x + w * 0.7, y + h * 0.35],
                [x + w * 0.5, y + h * 0.55],
                [x + w * 0.35, y + h * 0.8],
                [x + w * 0.65, y + h * 0.8],
            ],
            dtype=np.float32,
        ),
        score=0.98,
    )


def frontal_pose() -> HeadPose:
    return HeadPose(
        yaw=2.0,
        pitch=-1.0,
        roll=1.0,
        rotation=np.eye(3, dtype=np.float32),
    )


class FrameQualityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DMSConfig()

    def test_good_frame(self) -> None:
        quality = assess_frame(
            self.config, textured_frame(), make_face(), frontal_pose()
        )
        self.assertTrue(quality.face_present)
        self.assertTrue(quality.ok, quality.issues)
        self.assertAlmostEqual(quality.face_ratio, 200 / 640, places=3)
        self.assertEqual(quality.instruction, "HOLD STILL")

    def test_no_face(self) -> None:
        quality = assess_frame(
            self.config, textured_frame(), None, None
        )
        self.assertFalse(quality.face_present)
        self.assertFalse(quality.ok)
        self.assertIn("NO FACE", quality.issues[0])

    def test_too_far_off_centre_and_turned(self) -> None:
        quality = assess_frame(
            self.config,
            textured_frame(),
            make_face((20, 20, 90, 90)),
            HeadPose(
                yaw=40.0,
                pitch=30.0,
                roll=20.0,
                rotation=np.eye(3, dtype=np.float32),
            ),
        )
        self.assertFalse(quality.ok)
        self.assertIn("MOVE CLOSER", quality.issues)
        self.assertIn("CENTER YOUR FACE", quality.issues)
        self.assertIn("LOOK STRAIGHT", quality.issues)

    def test_dark_and_blurry(self) -> None:
        frame = np.full((480, 640, 3), 10, dtype=np.uint8)
        quality = assess_frame(self.config, frame, make_face(), frontal_pose())
        self.assertIn("TOO DARK - ADD LIGHT", quality.issues)
        self.assertIn("HOLD STILL / CLEAN THE LENS", quality.issues)

    def test_overlay_draws_on_frame(self) -> None:
        frame = textured_frame()
        before = frame.copy()
        quality = assess_frame(self.config, frame, make_face(), frontal_pose())
        result = draw_setup_overlay(frame, quality, "Alice")
        self.assertIs(result, frame)
        self.assertGreater(float(np.abs(frame.astype(int) - before.astype(int)).sum()), 0)


class FrameSetupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DMSConfig()
        self.config.setup_settle_seconds = 2.0
        self.frame = textured_frame()
        self.face = make_face()
        self.pose = frontal_pose()

    def test_ready_after_settle(self) -> None:
        setup = FrameSetup(self.config)
        first = setup.update(self.frame, self.face, self.pose, 0.0)
        self.assertTrue(first.ok)
        self.assertFalse(first.ready)
        second = setup.update(self.frame, self.face, self.pose, 1.0)
        self.assertFalse(second.ready)
        third = setup.update(self.frame, self.face, self.pose, 2.1)
        self.assertTrue(third.ready)
        self.assertIn("READY", third.instruction)

    def test_bad_frame_resets_timer(self) -> None:
        setup = FrameSetup(self.config)
        setup.update(self.frame, self.face, self.pose, 0.0)
        setup.update(self.frame, None, None, 1.0)
        again = setup.update(self.frame, self.face, self.pose, 2.2)
        self.assertFalse(again.ready)
        self.assertAlmostEqual(again.stable_seconds, 0.0, places=3)


class CalibrationFileTests(unittest.TestCase):
    def test_save_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "calibration.json"
            quality = assess_frame(
                DMSConfig(), textured_frame(), make_face(), frontal_pose()
            )
            save_setup_calibration(
                path,
                3.0,
                -2.0,
                1.5,
                quality=quality,
                resolution=(640, 480),
                source="test",
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["version"], 3)
            self.assertAlmostEqual(payload["yaw"], 3.0)
            self.assertAlmostEqual(payload["pitch"], -2.0)
            self.assertAlmostEqual(payload["roll"], 1.5)
            self.assertTrue(payload["calibrated"])
            self.assertEqual(payload["resolution"], [640, 480])
            self.assertEqual(payload["source"], "test")


if __name__ == "__main__":
    unittest.main()
