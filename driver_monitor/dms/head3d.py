"""A dependency-free 3D head renderer driven by the estimated head pose.

The renderer builds a low-poly parametric head (cranium, jaw, hair, ears,
nose, eyes, lips, neck and shoulders) and draws it with per-face diffuse and
specular shading.  It only uses NumPy and OpenCV, so it runs comfortably on a
Raspberry Pi 4 alongside the neural networks.

Rendering strategy
------------------
A naive painter's algorithm (one ``fillConvexPoly`` call per triangle) is far
too slow on a Pi.  The mesh is therefore pre-processed once:

* face normals, centroids and a per-face specular term are cached,
* all vertices / normals / centroids live in one contiguous array so that a
  frame needs only a handful of vectorised NumPy operations,
* visible triangles are quantised into a few shade bins and drawn with one
  batched ``fillPoly`` call per bin and part.

The eyes follow the measured eye-open probabilities, so a blink or a
microsleep is visible on the mannequin as well as in the camera image.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .geometry import clamp, normalize, render_rotation


# ---------------------------------------------------------------------------
# Mesh containers
# ---------------------------------------------------------------------------


@dataclass
class _Part:
    name: str
    vertices: np.ndarray  # (N, 3)
    faces: np.ndarray  # (M, 3)
    color: tuple[int, int, int]  # BGR
    specular: float = 0.05
    hull: bool = True  # fill silhouette first
    two_sided: bool = False
    order: int = 0
    levels: int = 8
    # Filled in by _MeshBuilder.build()
    normals: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    centroids: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    v_slice: slice = field(default_factory=lambda: slice(0, 0))
    f_slice: slice = field(default_factory=lambda: slice(0, 0))


@dataclass
class _Mesh:
    parts: list[_Part]
    vertices: np.ndarray
    normals: np.ndarray
    centroids: np.ndarray
    specular: np.ndarray
    colors: np.ndarray

    @property
    def triangle_count(self) -> int:
        return sum(len(p.faces) for p in self.parts)


class _MeshBuilder:
    def __init__(self) -> None:
        self.parts: list[_Part] = []

    def add(
        self,
        name: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        color: tuple[int, int, int],
        center: tuple[float, float, float],
        specular: float = 0.05,
        hull: bool = True,
        two_sided: bool = False,
        order: int = 0,
        levels: int = 8,
    ) -> None:
        vertices = np.asarray(vertices, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
        if len(faces) == 0:
            return

        # Orient every face outward from the part centre.
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        normals = np.cross(v1 - v0, v2 - v0)
        centroids = (v0 + v1 + v2) / 3.0
        outward = centroids - np.asarray(center, dtype=np.float32)[None, :]
        flip = np.einsum("ij,ij->i", normals, outward) < 0.0
        faces = faces.copy()
        faces[flip, 1], faces[flip, 2] = (
            faces[flip, 2].copy(),
            faces[flip, 1].copy(),
        )

        # Recompute after flipping and drop degenerate triangles (poles).
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        raw = np.cross(v1 - v0, v2 - v0)
        area = np.linalg.norm(raw, axis=1) * 0.5
        keep = area > 1e-10
        faces = faces[keep]
        raw = raw[keep]
        centroids = ((v0 + v1 + v2) / 3.0)[keep]
        normals = raw / np.maximum(
            np.linalg.norm(raw, axis=1, keepdims=True), 1e-8
        )

        self.parts.append(
            _Part(
                name=name,
                vertices=vertices,
                faces=faces,
                color=color,
                specular=float(specular),
                hull=bool(hull),
                two_sided=bool(two_sided),
                order=int(order),
                levels=max(4, int(levels)),
                normals=normals.astype(np.float32),
                centroids=centroids.astype(np.float32),
            )
        )

    def build(self) -> _Mesh:
        self.parts.sort(key=lambda p: p.order)
        v_offset = f_offset = 0
        for part in self.parts:
            part.v_slice = slice(v_offset, v_offset + len(part.vertices))
            part.f_slice = slice(f_offset, f_offset + len(part.faces))
            v_offset += len(part.vertices)
            f_offset += len(part.faces)

        return _Mesh(
            parts=self.parts,
            vertices=np.concatenate([p.vertices for p in self.parts], axis=0),
            normals=np.concatenate([p.normals for p in self.parts], axis=0),
            centroids=np.concatenate([p.centroids for p in self.parts], axis=0),
            specular=np.concatenate(
                [
                    np.full(len(p.faces), p.specular, dtype=np.float32)
                    for p in self.parts
                ]
            ),
            colors=np.concatenate(
                [
                    np.repeat(
                        np.asarray(p.color, dtype=np.float32)[None, :],
                        len(p.faces),
                        axis=0,
                    )
                    for p in self.parts
                ],
                axis=0,
            ),
        )


# ---------------------------------------------------------------------------
# Primitive geometry
# ---------------------------------------------------------------------------


def _ellipsoid(
    center: tuple[float, float, float],
    radii: tuple[float, float, float],
    lat: int = 16,
    lon: int = 22,
    taper_bottom: float = 0.0,
    lower_compress: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    cx, cy, cz = center
    rx, ry, rz = radii
    vertices: list[tuple[float, float, float]] = []
    for i in range(lat + 1):
        theta = math.pi * i / lat
        ring = math.sin(theta)
        yy = ry * math.cos(theta)
        for j in range(lon):
            phi = 2.0 * math.pi * j / lon
            x = rx * ring * math.cos(phi)
            z = rz * ring * math.sin(phi)
            if yy < 0.0:
                yy *= lower_compress
            if yy < 0.0 and taper_bottom > 0.0:
                t = min(1.0, -yy / max(ry, 1e-6))
                scale = 1.0 - taper_bottom * (t ** 1.6)
                x *= scale
                z *= scale
            vertices.append((cx + x, cy + yy, cz + z))

    faces: list[tuple[int, int, int]] = []
    for i in range(lat):
        for j in range(lon):
            a = i * lon + j
            b = i * lon + (j + 1) % lon
            c = (i + 1) * lon + j
            d = (i + 1) * lon + (j + 1) % lon
            faces.append((a, b, c))
            faces.append((b, d, c))
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def _hair_cap(
    center: tuple[float, float, float],
    radii: tuple[float, float, float],
    lat: int,
    lon: int,
    theta_front: float,
    theta_side: float,
    theta_back: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Hair shell whose lower edge follows a natural hairline.

    The polar angle limit varies with longitude: it stops high on the
    forehead, a little lower at the temples and lowest at the nape.  Building
    the boundary directly avoids the jagged triangles a planar cut produces.
    """
    cx, cy, cz = center
    rx, ry, rz = radii
    tf = math.radians(theta_front)
    ts = math.radians(theta_side)
    tb = math.radians(theta_back)

    vertices: list[tuple[float, float, float]] = []
    for j in range(lon):
        phi = 2.0 * math.pi * j / lon
        s = math.sin(phi)
        front = max(0.0, s) ** 0.8
        back = max(0.0, -s) ** 0.8
        tmax = ts + (tf - ts) * front + (tb - ts) * back
        for i in range(lat + 1):
            theta = tmax * i / lat
            ring = math.sin(theta)
            vertices.append(
                (
                    cx + rx * ring * math.cos(phi),
                    cy + ry * math.cos(theta),
                    cz + rz * ring * math.sin(phi),
                )
            )

    stride = lat + 1
    faces: list[tuple[int, int, int]] = []
    for j in range(lon):
        jn = (j + 1) % lon
        for i in range(lat):
            a = j * stride + i
            b = jn * stride + i
            c = j * stride + i + 1
            d = jn * stride + i + 1
            faces.append((a, b, c))
            faces.append((b, d, c))
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def _frustum(
    y0: float,
    y1: float,
    rx0: float,
    rz0: float,
    rx1: float,
    rz1: float,
    segments: int = 24,
    center_z: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[tuple[float, float, float]] = []
    for y, rx, rz in ((y0, rx0, rz0), (y1, rx1, rz1)):
        for j in range(segments):
            phi = 2.0 * math.pi * j / segments
            vertices.append((rx * math.cos(phi), y, center_z + rz * math.sin(phi)))
    faces: list[tuple[int, int, int]] = []
    for j in range(segments):
        a = j
        b = (j + 1) % segments
        c = segments + j
        d = segments + (j + 1) % segments
        faces.append((a, b, c))
        faces.append((b, d, c))
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def _ellipse_disc(
    center: tuple[float, float, float],
    rx: float,
    ry: float,
    segments: int = 18,
) -> tuple[np.ndarray, np.ndarray]:
    cx, cy, cz = center
    vertices: list[tuple[float, float, float]] = [(cx, cy, cz)]
    for j in range(segments):
        phi = 2.0 * math.pi * j / segments
        vertices.append((cx + rx * math.cos(phi), cy + ry * math.sin(phi), cz))
    faces = [(0, 1 + j, 1 + (j + 1) % segments) for j in range(segments)]
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def _quad(
    points: list[tuple[float, float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(points, dtype=np.float32), np.asarray(
        [(0, 1, 2), (0, 2, 3)], dtype=np.int32
    )


# ---------------------------------------------------------------------------
# Head mesh
# ---------------------------------------------------------------------------


def _build_head_mesh() -> _Mesh:
    b = _MeshBuilder()

    skin = (188, 205, 232)
    skin_shadow = (158, 176, 202)
    skin_ear = (174, 192, 218)
    hair = (38, 46, 58)
    brow = (44, 54, 68)
    lip = (112, 116, 204)
    lip_dark = (78, 82, 130)
    lip_hi = (128, 134, 214)
    shirt = (128, 94, 72)
    collar = (98, 70, 54)

    # Back to front ------------------------------------------------------
    v, f = _frustum(-2.30, -1.48, 1.06, 0.50, 0.56, 0.36, segments=28)
    b.add("shoulders", v, f, shirt, center=(0.0, -1.9, 0.0), specular=0.03, order=0)

    v, f = _frustum(-1.52, -1.40, 0.40, 0.36, 0.38, 0.34, segments=24)
    b.add("collar", v, f, collar, center=(0.0, -1.46, 0.0), specular=0.02, order=1)

    v, f = _frustum(-1.55, -0.44, 0.34, 0.32, 0.31, 0.30, segments=24)
    b.add("neck", v, f, skin_shadow, center=(0.0, -1.1, 0.0), specular=0.02, order=2)

    # Head ---------------------------------------------------------------
    v, f = _ellipsoid(
        center=(0.0, 0.08, 0.0),
        radii=(0.78, 0.90, 0.84),
        lat=14,
        lon=20,
        taper_bottom=0.30,
        lower_compress=1.0,
    )
    b.add(
        "head",
        v,
        f,
        skin,
        center=(0.0, 0.0, 0.0),
        specular=0.10,
        order=3,
        levels=12,
    )

    # Hair shell: a slightly larger cap with a natural hairline. ----------
    hv, hf = _hair_cap(
        center=(0.0, 0.085, -0.005),
        radii=(0.815, 0.940, 0.875),
        lat=10,
        lon=22,
        theta_front=66.0,
        theta_side=73.0,
        theta_back=100.0,
    )
    b.add(
        "hair",
        hv,
        hf,
        hair,
        center=(0.0, 0.02, -0.05),
        specular=0.18,
        hull=False,
        order=4,
        levels=9,
    )

    # Ears ---------------------------------------------------------------
    for side in (-1.0, 1.0):
        v, f = _ellipsoid(
            center=(side * 0.775, 0.09, -0.02),
            radii=(0.070, 0.155, 0.105),
            lat=6,
            lon=8,
        )
        b.add(
            "ear",
            v,
            f,
            skin_ear,
            center=(side * 0.70, 0.02, -0.02),
            specular=0.04,
            order=5,
        )

    return b.build()


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


class HeadRenderer:
    """Renders a rotating stylized 3D head into a BGR image."""

    _SHADE_LEVELS = 14

    def __init__(self, size: int = 420) -> None:
        self.size = int(size)
        self.mesh = _build_head_mesh()
        self.focal = 0.86 * self.size
        self.cx = self.size / 2.0
        self.cy = self.size * 0.44
        self.camera_distance = 4.2
        self.camera = np.array(
            [0.0, 0.0, self.camera_distance], dtype=np.float32
        )
        self.background = self._build_background()

        self.light = normalize(np.array([-0.45, 0.70, 0.85], dtype=np.float32))
        self.view = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.half = normalize(self.light + self.view)

        # Cache the last frame.  A stationary driver reuses it unchanged,
        # which keeps the UI at full frame-rate even on a busy Pi.
        self._cache_key: tuple | None = None
        self._cache: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Background
    # ------------------------------------------------------------------
    def _build_background(self) -> np.ndarray:
        s = self.size
        y = np.linspace(0.0, 1.0, s, dtype=np.float32)[:, None]
        top = np.array([34, 30, 27], dtype=np.float32)  # BGR
        bottom = np.array([11, 11, 14], dtype=np.float32)
        grad = top[None, :] * (1.0 - y) + bottom[None, :] * y
        bg = np.repeat(grad[:, None, :], s, axis=1).astype(np.uint8)

        yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
        d = np.sqrt((xx - s * 0.5) ** 2 + (yy - s * 0.44) ** 2) / (s * 0.55)
        glow = np.clip(1.0 - d, 0.0, 1.0) ** 2
        glow = (glow * 34.0).astype(np.float32)
        add = np.stack([glow * 0.75, glow * 0.95, glow * 1.15], axis=-1)
        bg = np.clip(bg.astype(np.float32) + add, 0, 255).astype(np.uint8)

        cv2.ellipse(
            bg,
            (int(s * 0.5), int(s * 0.875)),
            (int(s * 0.30), int(s * 0.052)),
            0,
            0,
            360,
            (54, 50, 45),
            1,
            cv2.LINE_AA,
        )
        return bg

    # ------------------------------------------------------------------
    # Projection / shading
    # ------------------------------------------------------------------
    def _project_all(
        self, vertices: np.ndarray, rotation: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        rotated = vertices @ rotation.T
        depth = np.maximum(self.camera_distance - rotated[:, 2], 0.15)
        u = self.cx + self.focal * rotated[:, 0] / depth
        v = self.cy - self.focal * rotated[:, 1] / depth
        return np.stack([u, v], axis=1), rotated

    def _project_points(
        self, points: np.ndarray, rotation: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        rotated = points @ rotation.T
        depth = np.maximum(self.camera_distance - rotated[:, 2], 0.15)
        u = self.cx + self.focal * rotated[:, 0] / depth
        v = self.cy - self.focal * rotated[:, 1] / depth
        return np.stack([u, v], axis=1), depth

    def _shade_all(
        self, rotated_normals: np.ndarray, rotation: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        diff = np.clip(rotated_normals @ self.light, 0.0, None)
        spec = np.clip(rotated_normals @ self.half, 0.0, None) ** 26.0
        rim = (1.0 - np.abs(rotated_normals @ self.view)) ** 3.0
        shade = (
            0.36
            + 0.64 * diff
            + 0.30 * spec * self.mesh.specular
            + 0.10 * rim
        )

        shaded = np.clip(self.mesh.colors * shade[:, None], 0, 255)
        # Subtle cool rim light.
        shaded[:, 0] += rim * 9.0
        shaded[:, 2] += rim * 3.0
        shaded = np.clip(shaded, 0, 255)
        return shade, shaded

    # ------------------------------------------------------------------
    # Part drawing
    # ------------------------------------------------------------------
    def _draw_part(
        self,
        canvas: np.ndarray,
        part: _Part,
        pts_i: np.ndarray,
        front: np.ndarray,
        shade: np.ndarray,
        colors: np.ndarray,
    ) -> None:
        part_pts = pts_i[part.v_slice]

        if part.hull:
            hull = cv2.convexHull(part_pts)
            base = tuple(int(c * 0.52) for c in part.color)
            cv2.fillConvexPoly(canvas, hull, base, cv2.LINE_AA)

        if not part.hull and not np.any(front):
            return

        tris = part_pts[part.faces]
        part_bins = np.clip(
            np.rint(shade[part.f_slice] * part.levels) / part.levels, 0.12, 1.35
        )
        for value in np.unique(part_bins[front]):
            idx = front & (part_bins == value)
            if not np.any(idx):
                continue
            color = tuple(
                int(min(255.0, c * value)) for c in part.color
            )
            cv2.fillPoly(canvas, tris[idx], color, cv2.LINE_8)

        if part.hull:
            hull = cv2.convexHull(part_pts)
            dark = (
                max(0, part.color[0] // 3),
                max(0, part.color[1] // 3),
                max(0, part.color[2] // 3),
            )
            cv2.polylines(canvas, [hull], True, dark, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Brows and lips (flat features, drawn after the shaded mesh)
    # ------------------------------------------------------------------
    def _draw_brows(self, canvas: np.ndarray, rotation: np.ndarray) -> None:
        for side in (-1.0, 1.0):
            ex = side * 0.305
            y0 = 0.190
            points = np.asarray(
                [
                    (ex - 0.112, y0, 0.768),
                    (ex + 0.112, y0 + 0.006, 0.768),
                    (ex + 0.098, y0 + 0.040, 0.768),
                    (ex - 0.098, y0 + 0.032, 0.768),
                ],
                dtype=np.float32,
            )
            p, _ = self._project_points(points, rotation)
            cv2.fillPoly(
                canvas,
                [np.round(p).astype(np.int32)],
                (44, 54, 68),
                cv2.LINE_AA,
            )

    def _draw_lips(self, canvas: np.ndarray, rotation: np.ndarray) -> None:
        n = 11
        upper: list[tuple[float, float, float]] = []
        mouth: list[tuple[float, float, float]] = []
        lower: list[tuple[float, float, float]] = []
        for k in range(n):
            t = k / (n - 1)
            x = -0.138 + 0.276 * t
            arc = math.sin(math.pi * t)
            # Cupid's bow: a gentle dip at the centre of the upper lip.
            bow = 0.010 * math.cos(2.0 * math.pi * t) * arc
            upper.append((x, -0.460 + 0.040 * arc + bow, 0.682))
            mouth.append((x, -0.460 + 0.004 * math.sin(2.0 * math.pi * t), 0.683))
            lower.append((x, -0.460 - 0.035 * arc, 0.682))

        upper3 = np.asarray(upper + list(reversed(mouth)), dtype=np.float32)
        lower3 = np.asarray(mouth + list(reversed(lower)), dtype=np.float32)
        pu, _ = self._project_points(upper3, rotation)
        pl, _ = self._project_points(lower3, rotation)
        pm, _ = self._project_points(np.asarray(mouth, dtype=np.float32), rotation)
        pu_i = np.round(pu).astype(np.int32)
        cv2.fillPoly(canvas, [pu_i], (104, 108, 196), cv2.LINE_AA)
        cv2.fillPoly(
            canvas, [np.round(pl).astype(np.int32)], (128, 134, 214), cv2.LINE_AA
        )
        cv2.polylines(
            canvas, [np.round(pm).astype(np.int32)], False, (78, 82, 130), 1, cv2.LINE_AA
        )

    # ------------------------------------------------------------------
    # Nose (drawn as flat shading, keeps the low-poly look without a spike)
    # ------------------------------------------------------------------
    def _draw_nose(self, canvas: np.ndarray, rotation: np.ndarray) -> None:
        points = np.asarray(
            [
                (0.0, 0.160, 0.837),  # bridge top
                (0.0, -0.140, 0.814),  # tip
                (-0.072, -0.200, 0.798),  # left base
                (0.072, -0.200, 0.798),  # right base
                (0.0, -0.230, 0.787),  # under-tip
            ],
            dtype=np.float32,
        )
        p, _ = self._project_points(points, rotation)
        p = np.round(p).astype(np.int32)
        bridge, tip, left, right, under = p

        # Left side in shadow, right side catching the key light.
        cv2.fillPoly(
            canvas,
            [np.array([bridge, left, tip], dtype=np.int32)],
            (166, 182, 205),
            cv2.LINE_AA,
        )
        cv2.fillPoly(
            canvas,
            [np.array([bridge, tip, right], dtype=np.int32)],
            (203, 219, 240),
            cv2.LINE_AA,
        )
        # Small shadow under the tip and two nostrils.
        cv2.fillPoly(
            canvas,
            [np.array([left, right, under], dtype=np.int32)],
            (150, 166, 190),
            cv2.LINE_AA,
        )
        angle = math.degrees(
            math.atan2(float(right[1] - left[1]), float(right[0] - left[0]))
        )
        for base in (left, right):
            cx = int(0.70 * base[0] + 0.30 * tip[0])
            cy = int(0.70 * base[1] + 0.30 * tip[1])
            cv2.ellipse(
                canvas,
                (cx, cy),
                (3, 2),
                angle,
                0,
                360,
                (92, 104, 126),
                -1,
                cv2.LINE_AA,
            )

    # ------------------------------------------------------------------
    # Eyes
    # ------------------------------------------------------------------
    def _draw_eyes(
        self,
        canvas: np.ndarray,
        rotation: np.ndarray,
        yaw: float,
        eye_open: tuple[float, float],
        skin_color: tuple[int, int, int],
        valid: bool,
    ) -> None:
        """Draw almond-shaped eyes with a per-eye opening (0 closed .. 1 open)."""
        for idx, side in enumerate((-1.0, 1.0)):
            # At strong yaw the far eye disappears behind the nose bridge.
            # Positive yaw turns the head to image-left, so the image-left
            # (side=-1) eye is the far one.
            if yaw > 32.0 and side < 0.0:
                continue
            if yaw < -32.0 and side > 0.0:
                continue

            openness = clamp(float(eye_open[idx]), 0.0, 1.0) if valid else 1.0
            ex = side * 0.305
            ey = 0.020
            ez = 0.777

            n = 9
            upper: list[tuple[float, float, float]] = []
            lower: list[tuple[float, float, float]] = []
            half_w = 0.105
            up_h = 0.046 * openness
            lo_h = 0.028 * openness
            for k in range(n):
                t = k / (n - 1)
                x = ex - half_w + 2.0 * half_w * t
                arc = math.sin(math.pi * t)
                upper.append((x, ey + up_h * arc, ez))
                lower.append((x, ey - lo_h * arc, ez))

            # Soft socket shadow.
            socket = self._project_points(
                np.asarray([(ex, ey, ez - 0.012)], dtype=np.float32), rotation
            )[0][0]
            cv2.ellipse(
                canvas,
                (int(socket[0]), int(socket[1])),
                (
                    max(2, int(0.112 * self.focal / 3.4)),
                    max(1, int(0.050 * self.focal / 3.4)),
                ),
                0,
                0,
                360,
                (158, 174, 198),
                -1,
                cv2.LINE_AA,
            )

            poly3 = np.asarray(upper + list(reversed(lower)), dtype=np.float32)
            poly2, _ = self._project_points(poly3, rotation)
            poly2 = np.round(poly2).astype(np.int32)

            if openness < 0.06:
                # Closed eye: the socket is already skin-toned; draw the lash
                # line along the lid.
                lash3 = np.asarray(
                    [
                        (ex - half_w * 0.85, ey, ez + 0.004),
                        (ex, ey - 0.007, ez + 0.004),
                        (ex + half_w * 0.85, ey, ez + 0.004),
                    ],
                    dtype=np.float32,
                )
                lash, _ = self._project_points(lash3, rotation)
                cv2.polylines(
                    canvas,
                    [np.round(lash).astype(np.int32)],
                    False,
                    (62, 66, 88),
                    1,
                    cv2.LINE_AA,
                )
                continue

            # Eye white.
            cv2.fillPoly(canvas, [poly2], (245, 247, 247), cv2.LINE_AA)
            iris_r = min(0.034, 0.044 * openness + 0.005)
            iris_y = ey - 0.008 * (1.0 - openness)
            center, depth = self._project_points(
                np.asarray([(ex, iris_y, ez + 0.004)], dtype=np.float32), rotation
            )
            cxy = (int(center[0][0]), int(center[0][1]))
            r_px = max(2, int(iris_r * self.focal / depth[0]))
            cv2.circle(canvas, cxy, r_px, (122, 94, 54), -1, cv2.LINE_AA)
            cv2.circle(
                canvas, cxy, max(1, int(r_px * 0.46)), (24, 24, 28), -1, cv2.LINE_8
            )
            hi, _ = self._project_points(
                np.asarray([(ex + 0.014, iris_y + 0.014, ez + 0.007)], dtype=np.float32),
                rotation,
            )
            cv2.circle(
                canvas,
                (int(hi[0][0]), int(hi[0][1])),
                max(1, int(r_px * 0.20)),
                (250, 250, 250),
                -1,
                cv2.LINE_8,
            )

            # Upper lash line.
            lash, _ = self._project_points(np.asarray(upper, dtype=np.float32), rotation)
            cv2.polylines(
                canvas,
                [np.round(lash).astype(np.int32)],
                False,
                (56, 60, 80),
                1,
                cv2.LINE_AA,
            )

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    @staticmethod
    def _rounded_rect(
        img: np.ndarray,
        rect: tuple[int, int, int, int],
        color: tuple[int, int, int],
        thickness: int = -1,
        radius: int = 8,
    ) -> None:
        x, y, w, h = rect
        r = min(radius, w // 2, h // 2)
        if thickness < 0:
            cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, -1)
            cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, -1)
            for cx, cy in (
                (x + r, y + r),
                (x + w - r, y + r),
                (x + r, y + h - r),
                (x + w - r, y + h - r),
            ):
                cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
        else:
            cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, thickness)
            cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, thickness)
            for cx, cy in (
                (x + r, y + r),
                (x + w - r, y + r),
                (x + r, y + h - r),
                (x + w - r, y + h - r),
            ):
                cv2.circle(img, (cx, cy), r, color, thickness, cv2.LINE_AA)

    def _draw_top_view(
        self, canvas: np.ndarray, yaw: float, pitch: float, color: tuple[int, int, int]
    ) -> None:
        s = self.size
        cx, cy = s - 52, 52
        r = 28
        cv2.circle(canvas, (cx, cy), r, (72, 68, 62), 1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), r - 5, (56, 53, 49), 1, cv2.LINE_AA)
        for angle in range(0, 360, 45):
            a = math.radians(angle)
            x0 = int(cx + (r - 6) * math.sin(a))
            y0 = int(cy - (r - 6) * math.cos(a))
            x1 = int(cx + (r - 1) * math.sin(a))
            y1 = int(cy - (r - 1) * math.cos(a))
            cv2.line(canvas, (x0, y0), (x1, y1), (85, 80, 74), 1, cv2.LINE_AA)

        yaw_r = math.radians(yaw)
        dx, dy = -math.sin(yaw_r), -math.cos(yaw_r)
        px, py = -dy, dx
        nose_len = r * 0.72
        tip = (int(cx + nose_len * dx), int(cy + nose_len * dy))
        left = (int(cx + 8 * px + 6 * dx), int(cy + 8 * py + 6 * dy))
        right = (int(cx - 8 * px + 6 * dx), int(cy - 8 * py + 6 * dy))
        cv2.circle(canvas, (cx, cy), 10, (82, 88, 100), -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 10, color, 1, cv2.LINE_AA)
        cv2.fillPoly(
            canvas, [np.array([tip, left, right], dtype=np.int32)], color, cv2.LINE_AA
        )
        pitch_off = clamp(pitch / 45.0, -1.0, 1.0) * 8.0
        cv2.circle(
            canvas,
            (int(cx + pitch_off * dx), int(cy + pitch_off * dy)),
            2,
            (235, 235, 235),
            -1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "TOP",
            (cx - 12, cy + r + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (150, 145, 138),
            1,
            cv2.LINE_AA,
        )

    def _draw_angle_row(
        self,
        canvas: np.ndarray,
        y: int,
        label: str,
        value: float,
        limit: float,
        marker_color: tuple[int, int, int],
    ) -> None:
        x0, x1 = 92, self.size - 58
        cv2.putText(
            canvas,
            label,
            (22, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (175, 170, 162),
            1,
            cv2.LINE_AA,
        )
        cv2.line(canvas, (x0, y), (x1, y), (74, 70, 64), 1, cv2.LINE_AA)
        for t in (-limit, -limit / 2, 0, limit / 2, limit):
            xx = int(x0 + (t + limit) / (2 * limit) * (x1 - x0))
            cv2.line(canvas, (xx, y - 4), (xx, y + 4), (95, 90, 82), 1, cv2.LINE_AA)
        v = clamp(value, -limit, limit)
        xx = int(x0 + (v + limit) / (2 * limit) * (x1 - x0))
        cv2.circle(canvas, (xx, y), 4, marker_color, -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{value:+5.1f}\u00b0",
            (x1 + 6, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            marker_color,
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def _status_color(status: str) -> tuple[int, int, int]:
        s = status.upper()
        if s in ("CRITICAL", "DROWSY", "DISTRACTED", "NO_FACE"):
            return (38, 38, 240)
        if s in ("SHORT_GLANCE", "ATTENTION"):
            return (0, 185, 255)
        return (80, 215, 120)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def render(
        self,
        yaw: float,
        pitch: float,
        roll: float,
        eye_open: tuple[float, float] = (1.0, 1.0),
        status: str = "NORMAL",
        zone: str = "FORWARD",
        eyes_valid: bool = True,
        pose_valid: bool = True,
    ) -> np.ndarray:
        key = (
            round(float(yaw), 1),
            round(float(pitch), 1),
            round(float(roll), 1),
            round(float(eye_open[0]), 2),
            round(float(eye_open[1]), 2),
            status,
            zone,
            bool(eyes_valid),
            bool(pose_valid),
        )
        if key == self._cache_key and self._cache is not None:
            return self._cache

        canvas = self.background.copy()
        color = self._status_color(status if pose_valid else "NO_FACE")
        rotation = (
            render_rotation(yaw, pitch, roll)
            if pose_valid
            else np.eye(3, dtype=np.float32)
        )

        rotated_normals = self.mesh.normals @ rotation.T
        rotated_centroids = self.mesh.centroids @ rotation.T
        pts, _ = self._project_all(self.mesh.vertices, rotation)
        pts_i = np.round(pts).astype(np.int32)

        to_camera = self.camera[None, :] - rotated_centroids
        visibility = np.einsum("ij,ij->i", rotated_normals, to_camera)
        shade, colors = self._shade_all(rotated_normals, rotation)
        # Flat features (two-sided) stay visible until they turn away sharply.
        two_sided_mask = np.zeros(len(self.mesh.normals), dtype=bool)
        for part in self.mesh.parts:
            if part.two_sided:
                two_sided_mask[part.f_slice] = True
        front = (visibility > 0.0) | (two_sided_mask & (visibility > -0.22))

        for part in self.mesh.parts:
            self._draw_part(
                canvas,
                part,
                pts_i,
                front[part.f_slice],
                shade,
                colors,
            )

        self._draw_brows(canvas, rotation)
        self._draw_nose(canvas, rotation)
        self._draw_lips(canvas, rotation)
        self._draw_eyes(
            canvas,
            rotation,
            yaw if pose_valid else 0.0,
            eye_open,
            skin_color=(176, 193, 219),
            valid=eyes_valid and pose_valid,
        )

        if not pose_valid:
            # Dim the mannequin when there is no driver, but keep the UI crisp.
            dim = cv2.convertScaleAbs(canvas, alpha=0.55, beta=0)
            canvas = cv2.addWeighted(canvas, 0.45, dim, 0.55, 0)

        # --------------------------------------------------------------
        # Overlay
        # --------------------------------------------------------------
        cv2.putText(
            canvas,
            "3D HEAD POSE",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (238, 236, 232),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "live driver orientation",
            (20, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (145, 141, 134),
            1,
            cv2.LINE_AA,
        )

        label = status.upper().replace("_", " ") if pose_valid else "NO FACE"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        pill_w = tw + 24
        self._rounded_rect(canvas, (20, 60, pill_w, 26), (38, 36, 33), -1, 8)
        cv2.rectangle(canvas, (20, 60), (24, 86), color, -1)
        cv2.putText(
            canvas,
            label,
            (32, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            color,
            1,
            cv2.LINE_AA,
        )

        if pose_valid and eyes_valid:
            lo = clamp(eye_open[0], 0.0, 1.0)
            ro = clamp(eye_open[1], 0.0, 1.0)
            eye_color = (80, 215, 120) if min(lo, ro) > 0.45 else (38, 38, 240)
            eye_label = "EYES OPEN" if min(lo, ro) > 0.45 else "EYES CLOSED"
            cv2.putText(
                canvas,
                f"{eye_label}  {lo:.2f} / {ro:.2f}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                eye_color,
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                canvas,
                "EYES UNKNOWN",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                (150, 145, 138),
                1,
                cv2.LINE_AA,
            )

        if pose_valid:
            cv2.putText(
                canvas,
                f"ZONE  {zone}",
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (185, 181, 174),
                1,
                cv2.LINE_AA,
            )

        self._draw_top_view(
            canvas,
            yaw if pose_valid else 0.0,
            pitch if pose_valid else 0.0,
            color,
        )

        panel_y = self.size - 92
        overlay = canvas[panel_y : self.size, :].copy()
        cv2.rectangle(
            canvas, (0, panel_y), (self.size, self.size), (18, 18, 20), -1
        )
        canvas[panel_y : self.size, :] = cv2.addWeighted(
            canvas[panel_y : self.size, :], 0.65, overlay, 0.35, 0
        )
        cv2.line(
            canvas,
            (18, panel_y + 10),
            (self.size - 18, panel_y + 10),
            (68, 64, 58),
            1,
            cv2.LINE_AA,
        )
        self._draw_angle_row(
            canvas, panel_y + 30, "YAW", yaw if pose_valid else 0.0, 90.0, color
        )
        self._draw_angle_row(
            canvas, panel_y + 50, "PITCH", pitch if pose_valid else 0.0, 45.0, color
        )
        self._draw_angle_row(
            canvas, panel_y + 70, "ROLL", roll if pose_valid else 0.0, 90.0, color
        )

        self._cache_key = key
        self._cache = canvas
        return canvas


__all__ = ["HeadRenderer"]