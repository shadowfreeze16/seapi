"""Geometry helpers."""

from __future__ import annotations

import math
import unittest

import numpy as np

from dms.geometry import (
    clamp,
    euler_to_matrix,
    normalize,
    project_points,
    render_rotation,
    rotation_matrix_to_euler,
    rotation_x,
    rotation_y,
    rotation_z,
    wrap_angle_deg,
)


class WrapAngleTests(unittest.TestCase):
    def test_common_values(self) -> None:
        self.assertAlmostEqual(wrap_angle_deg(0.0), 0.0)
        self.assertAlmostEqual(wrap_angle_deg(370.0), 10.0)
        self.assertAlmostEqual(wrap_angle_deg(-190.0), 170.0)
        self.assertAlmostEqual(wrap_angle_deg(180.0), 180.0)
        self.assertAlmostEqual(wrap_angle_deg(-180.0), 180.0)
        self.assertAlmostEqual(wrap_angle_deg(720.0), 0.0)


class RotationTests(unittest.TestCase):
    def test_axis_rotations(self) -> None:
        x = rotation_x(90.0) @ np.array([0.0, 1.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(x, [0.0, 0.0, 1.0], atol=1e-6)

        y = rotation_y(90.0) @ np.array([1.0, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(y, [0.0, 0.0, -1.0], atol=1e-6)

        z = rotation_z(90.0) @ np.array([1.0, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(z, [0.0, 1.0, 0.0], atol=1e-6)

    def test_euler_decomposition_roundtrip(self) -> None:
        yaw, pitch, roll = 27.0, -11.0, 6.0
        matrix = euler_to_matrix(yaw, pitch, roll)
        out_pitch, out_yaw, out_roll = rotation_matrix_to_euler(matrix)
        self.assertAlmostEqual(out_yaw, yaw, places=4)
        self.assertAlmostEqual(out_pitch, pitch, places=4)
        self.assertAlmostEqual(out_roll, roll, places=4)

    def test_gimbal_lock_is_finite(self) -> None:
        yaw, pitch, roll = rotation_matrix_to_euler(rotation_y(90.0))
        for value in (yaw, pitch, roll):
            self.assertTrue(math.isfinite(value))

    def test_render_rotation_is_orthonormal(self) -> None:
        rotation = render_rotation(20.0, 10.0, 5.0)
        np.testing.assert_allclose(
            rotation @ rotation.T, np.eye(3), atol=1e-5
        )
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=5)


class ProjectionTests(unittest.TestCase):
    def test_principal_point(self) -> None:
        points = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        pixels, depth = project_points(points, focal=500.0, cx=320.0, cy=240.0, camera_distance=4.0)
        np.testing.assert_allclose(pixels[0], [320.0, 240.0], atol=1e-6)
        np.testing.assert_allclose(depth, [4.0], atol=1e-6)

    def test_perspective_offset(self) -> None:
        points = np.array(
            [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
        )
        pixels, depth = project_points(
            points, focal=100.0, cx=50.0, cy=40.0, camera_distance=5.0
        )
        np.testing.assert_allclose(pixels[0], [70.0, 20.0], atol=1e-6)
        np.testing.assert_allclose(depth, [5.0, 6.0], atol=1e-6)

    def test_depth_is_clamped(self) -> None:
        points = np.array([[0.0, 0.0, -10.0]], dtype=np.float32)
        _, depth = project_points(
            points, focal=100.0, cx=0.0, cy=0.0, camera_distance=5.0
        )
        self.assertGreaterEqual(float(depth[0]), 0.05)


class HelperTests(unittest.TestCase):
    def test_normalize(self) -> None:
        np.testing.assert_allclose(
            normalize(np.array([3.0, 4.0, 0.0])), [0.6, 0.8, 0.0]
        )
        np.testing.assert_allclose(normalize(np.zeros(3)), np.zeros(3))

    def test_clamp(self) -> None:
        self.assertEqual(clamp(5.0, 0.0, 1.0), 1.0)
        self.assertEqual(clamp(-5.0, 0.0, 1.0), 0.0)
        self.assertEqual(clamp(0.5, 0.0, 1.0), 0.5)


if __name__ == "__main__":
    unittest.main()