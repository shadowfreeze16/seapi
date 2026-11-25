"""Small geometry helpers shared by models, the state machine and the 3D head."""

from __future__ import annotations

import math

import numpy as np


def wrap_angle_deg(angle: float) -> float:
    """Wrap an angle in degrees into (-180, 180]."""
    a = (float(angle) + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


def rotation_x(angle_deg: float) -> np.ndarray:
    r = math.radians(angle_deg)
    c, s = math.cos(r), math.sin(r)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32
    )


def rotation_y(angle_deg: float) -> np.ndarray:
    r = math.radians(angle_deg)
    c, s = math.cos(r), math.sin(r)
    return np.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32
    )


def rotation_z(angle_deg: float) -> np.ndarray:
    r = math.radians(angle_deg)
    c, s = math.cos(r), math.sin(r)
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )


def euler_to_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Standard Z-Y-X rotation matrix (used for tests / math)."""
    return (rotation_z(roll) @ rotation_y(yaw) @ rotation_x(pitch)).astype(
        np.float32
    )


def render_rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Rotation used by the 3D head renderer.

    Sign conventions (matching 6DRepNet output on this project's camera):

    * positive yaw   -> face turns to image-left
    * positive pitch -> face looks up
    * positive roll  -> head tilts clockwise on screen

    The renderer uses a right-handed 3D space with +x right, +y up, +z toward
    the viewer.  The signs below make the rendered mannequin follow the driver.
    """
    return (
        rotation_z(-roll) @ rotation_x(-pitch) @ rotation_y(-yaw)
    ).astype(np.float32)


def rotation_matrix_to_euler(rotation: np.ndarray) -> tuple[float, float, float]:
    """Convert a 3x3 rotation matrix to (pitch, yaw, roll) in degrees.

    This is the same decomposition used by the 6DRepNet reference code.
    Singular (gimbal-lock) cases are handled by falling back to zero.
    """
    r = np.asarray(rotation, dtype=np.float64)
    sy = math.sqrt(float(r[0, 0] ** 2 + r[1, 0] ** 2))
    singular = sy < 1e-6
    if singular:
        pitch = math.atan2(-r[1, 2], r[1, 1])
        yaw = math.atan2(-r[2, 0], sy)
        roll = 0.0
    else:
        pitch = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(-r[2, 0], sy)
        roll = math.atan2(r[1, 0], r[0, 0])
    return (
        float(math.degrees(pitch)),
        float(math.degrees(yaw)),
        float(math.degrees(roll)),
    )


def project_points(
    points: np.ndarray,
    focal: float,
    cx: float,
    cy: float,
    camera_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Perspective-project 3D points into image coordinates.

    Args:
        points: (N, 3) array in world/head space (+x right, +y up, +z forward).
        focal: focal length in pixels.
        cx, cy: principal point.
        camera_distance: camera z-offset.

    Returns:
        (N, 2) pixel coordinates and (N,) camera-space depth.
    """
    pts = np.asarray(points, dtype=np.float32)
    depth = pts[:, 2] + float(camera_distance)
    depth = np.maximum(depth, 0.05)
    u = cx + focal * pts[:, 0] / depth
    v = cy - focal * pts[:, 1] / depth
    return np.stack([u, v], axis=1), depth


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return v / n


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))