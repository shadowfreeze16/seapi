"""HUD rendering with faces, poses, alerts and overlays."""

from __future__ import annotations

import unittest

import numpy as np

from dms.config import DMSConfig
from dms.head3d import HeadRenderer
from dms.identity import IdentityResult, MATCH, MISMATCH
from dms.models import Face, HeadPose
from dms.render import HUDRenderer
from dms.safety import GREEN, RED, YELLOW, SafetyState
from dms.state import CRITICAL, DISTRACTED, NORMAL, DriverState


def make_face() -> Face:
    return Face(
        box=np.array([200, 100, 160, 180], dtype=np.float32),
        landmarks=np.array(
            [
                [250, 150],
                [310, 150],
                [280, 180],
                [255, 225],
                [305, 225],
            ],
            dtype=np.float32,
        ),
        score=0.97,
    )


def make_pose() -> HeadPose:
    import math

    return HeadPose(
        yaw=18.0,
        pitch=-4.0,
        roll=3.0,
        rotation=np.eye(3, dtype=np.float32),
        latency_ms=21.0,
        crop_box=(180, 80, 200, 200),
    )


class HUDRendererTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DMSConfig()
        self.hud = HUDRenderer(self.config)
        self.frame = np.full((480, 640, 3), 70, dtype=np.uint8)
        self.face = make_face()
        self.pose = make_pose()
        self.head = HeadRenderer(168).render(18.0, -4.0, 3.0)

    def draw(self, state: DriverState, **kwargs) -> np.ndarray:
        return self.hud.draw(
            self.frame,
            state,
            pose=self.pose,
            face=self.face,
            head_img=self.head,
            fps=29.5,
            face_count=1,
            latencies={"face": 9.0, "pose": 21.0, "eye": 1.5, "head": 11.0},
            now=10.0,
            **kwargs,
        )

    def test_draw_returns_frame(self) -> None:
        state = DriverState(status=NORMAL, zone="FORWARD", message="Attention OK")
        result = self.draw(state)
        self.assertIs(result, self.frame)
        self.assertEqual(result.shape, (480, 640, 3))
        self.assertGreater(float(result.std()), 1.0)

    def test_draw_alert_states(self) -> None:
        for status, message in (
            (DISTRACTED, "DISTRACTION WARNING"),
            (CRITICAL, "CRITICAL: MICROSLEEP"),
        ):
            frame = np.full((480, 640, 3), 70, dtype=np.uint8)
            state = DriverState(
                status=status,
                zone="LEFT",
                message=message,
                alert=True,
                alarm=status == CRITICAL,
                offroad_seconds=3.0,
                status_seconds=1.0,
            )
            result = self.hud.draw(
                frame,
                state,
                pose=self.pose,
                face=self.face,
                head_img=self.head,
                fps=20.0,
                face_count=1,
                now=10.0,
            )
            self.assertEqual(result.shape, (480, 640, 3))
            self.assertGreater(float(result.std()), 1.0)

    def test_safety_badge_and_identity_chip(self) -> None:
        state = DriverState(status=NORMAL, zone="FORWARD", message="Attention OK")
        for level in (GREEN, YELLOW, RED):
            frame = np.full((480, 640, 3), 70, dtype=np.uint8)
            result = self.hud.draw(
                frame,
                state,
                pose=self.pose,
                face=self.face,
                head_img=self.head,
                fps=20.0,
                now=1.0,
                safety=SafetyState(level=level, reason="TEST", message="TEST"),
                identity=IdentityResult(
                    state=MATCH if level == GREEN else MISMATCH,
                    name="Alice" if level == GREEN else None,
                    score=0.81 if level == GREEN else None,
                ),
            )
            self.assertEqual(result.shape, (480, 640, 3))
            self.assertGreater(float(result.std()), 1.0)

    def test_safety_badge_in_compact_mode(self) -> None:
        state = DriverState(status=NORMAL, zone="FORWARD")
        frame = np.full((480, 640, 3), 70, dtype=np.uint8)
        result = self.hud.draw(
            frame,
            state,
            minimal=True,
            now=1.0,
            safety=SafetyState(level=YELLOW, reason="TEST", message="TEST"),
        )
        self.assertEqual(result.shape, (480, 640, 3))

    def test_compact_help_and_debug(self) -> None:
        state = DriverState(status=NORMAL, zone="FORWARD")
        for kwargs in (
            {"minimal": True},
            {"help_visible": True},
            {"debug": True},
            {"mirror": True, "calibrated": True, "recording": True},
        ):
            frame = np.full((480, 640, 3), 70, dtype=np.uint8)
            result = self.hud.draw(
                frame,
                state,
                pose=self.pose,
                face=self.face,
                head_img=self.head,
                fps=20.0,
                face_count=1,
                now=10.0,
                **kwargs,
            )
            self.assertEqual(result.shape, (480, 640, 3))

    def test_without_face_or_pose(self) -> None:
        state = DriverState(status="NO_FACE", zone="NO_FACE", alert=True)
        result = self.hud.draw(
            self.frame,
            state,
            pose=None,
            face=None,
            head_img=self.head,
            fps=0.0,
            now=1.0,
        )
        self.assertEqual(result.shape, (480, 640, 3))

    def test_eye_boxes_and_held_face(self) -> None:
        self.face.detected = False
        self.face.age = 2
        state = DriverState(status=NORMAL, zone="FORWARD")
        result = self.hud.draw(
            self.frame,
            state,
            pose=self.pose,
            face=self.face,
            eye_boxes=[(245, 145, 30, 18), (300, 145, 30, 18)],
            head_img=self.head,
            fps=20.0,
            now=1.0,
        )
        self.assertEqual(result.shape, (480, 640, 3))


if __name__ == "__main__":
    unittest.main()