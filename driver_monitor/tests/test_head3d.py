"""The dependency-free 3D head renderer."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from dms.head3d import HeadRenderer


class HeadRendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.renderer = HeadRenderer(240)

    def test_output_shape_and_dtype(self) -> None:
        image = self.renderer.render(0.0, 0.0, 0.0)
        self.assertEqual(image.shape, (240, 240, 3))
        self.assertEqual(image.dtype, np.uint8)
        self.assertGreater(float(image.std()), 5.0)

    def test_rotation_changes_pixels(self) -> None:
        front = self.renderer.render(0.0, 0.0, 0.0)
        left = self.renderer.render(30.0, 0.0, 0.0)
        up = self.renderer.render(0.0, 20.0, 0.0)
        tilt = self.renderer.render(0.0, 0.0, 20.0)
        for rotated in (left, up, tilt):
            self.assertGreater(
                float(cv2.absdiff(front, rotated).mean()), 0.5
            )

    def test_cache_returns_same_object(self) -> None:
        first = self.renderer.render(12.0, 4.0, 2.0)
        second = self.renderer.render(12.0, 4.0, 2.0)
        self.assertIs(first, second)
        third = self.renderer.render(12.1, 4.0, 2.0)
        self.assertIsNot(first, third)

    def test_cache_key_includes_eye_and_status(self) -> None:
        open_eyes = self.renderer.render(5.0, 0.0, 0.0, (1.0, 1.0), "NORMAL")
        closed_eyes = self.renderer.render(5.0, 0.0, 0.0, (0.0, 0.0), "DROWSY")
        self.assertIsNot(open_eyes, closed_eyes)
        self.assertGreater(
            float(cv2.absdiff(open_eyes, closed_eyes).mean()), 0.1
        )

    def test_no_face_render(self) -> None:
        normal = self.renderer.render(0.0, 0.0, 0.0, status="NORMAL")
        missing = self.renderer.render(
            0.0, 0.0, 0.0, status="NO_FACE", pose_valid=False
        )
        self.assertEqual(normal.shape, missing.shape)
        self.assertGreater(float(cv2.absdiff(normal, missing).mean()), 1.0)

    def test_extreme_angles_are_finite(self) -> None:
        for yaw in (-70.0, -30.0, 0.0, 30.0, 70.0):
            for pitch in (-35.0, 0.0, 35.0):
                for roll in (-60.0, 0.0, 60.0):
                    image = self.renderer.render(yaw, pitch, roll)
                    self.assertEqual(image.shape, (240, 240, 3))
                    self.assertTrue(np.isfinite(image).all())

    def test_custom_size(self) -> None:
        small = HeadRenderer(160)
        image = small.render(10.0, 5.0, 3.0)
        self.assertEqual(image.shape, (160, 160, 3))


if __name__ == "__main__":
    unittest.main()