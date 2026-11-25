"""Camera-frame HUD, face overlay, 3D axes and the 3D-head inset.

The HUD is intentionally lightweight (plain OpenCV drawing) so it does not
compete with the neural networks on a Raspberry Pi.  It can also be switched to
a compact mode with the ``H`` key.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from .config import DMSConfig
from .eyes import EyeMetrics
from .identity import MATCH, MISMATCH, IdentityResult
from .models import Face, HeadPose
from .safety import GREEN as SAFETY_GREEN
from .safety import RED as SAFETY_RED
from .safety import YELLOW as SAFETY_YELLOW
from .safety import SafetyState
from .state import CRITICAL, DISTRACTED, DROWSY, DriverState, NO_FACE

# BGR palette
INK = (238, 238, 234)
MUTED = (158, 160, 158)
DIM = (105, 108, 110)
PANEL = (24, 25, 28)
PANEL_EDGE = (68, 70, 74)
GREEN = (105, 215, 125)
AMBER = (0, 185, 255)
RED = (55, 55, 240)
CYAN = (220, 190, 80)
BLUE = (235, 175, 85)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float = 0.4,
    color: tuple[int, int, int] = INK,
    thickness: int = 1,
    shadow: bool = True,
) -> None:
    if shadow:
        cv2.putText(
            img,
            text,
            (org[0] + 1, org[1] + 1),
            FONT,
            scale,
            (0, 0, 0),
            thickness + 1,
            cv2.LINE_AA,
        )
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def _panel(
    img: np.ndarray,
    rect: tuple[int, int, int, int],
    color: tuple[int, int, int] = PANEL,
    alpha: float = 0.78,
    edge: tuple[int, int, int] | None = PANEL_EDGE,
    radius: int = 10,
) -> None:
    x, y, w, h = rect
    x = max(0, x)
    y = max(0, y)
    w = min(w, img.shape[1] - x)
    h = min(h, img.shape[0] - y)
    if w <= 0 or h <= 0:
        return
    roi = img[y : y + h, x : x + w]
    overlay = np.empty_like(roi)
    overlay[:] = color
    cv2.addWeighted(roi, 1.0 - alpha, overlay, alpha, 0.0, dst=roi)
    if edge is not None:
        r = min(radius, w // 2, h // 2)
        cv2.rectangle(img, (x + r, y), (x + w - r, y + h), edge, 1)
        cv2.rectangle(img, (x, y + r), (x + w, y + h - r), edge, 1)
        for cx, cy in (
            (x + r, y + r),
            (x + w - r, y + r),
            (x + r, y + h - r),
            (x + w - r, y + h - r),
        ):
            cv2.circle(img, (cx, cy), r, edge, 1, cv2.LINE_AA)


def _safety_color(level: str | None) -> tuple[int, int, int]:
    if level == SAFETY_RED:
        return RED
    if level == SAFETY_YELLOW:
        return AMBER
    return GREEN


def _identity_chip(
    identity: IdentityResult | None,
) -> tuple[str, tuple[int, int, int]] | None:
    if identity is None or identity.state in ("OFF", ""):
        return None
    if identity.state == MATCH:
        name = identity.name or "DRIVER"
        score = f" {identity.score:.2f}" if identity.score is not None else ""
        return f"ID {name}{score}", GREEN
    if identity.state == MISMATCH:
        return "ID MISMATCH", RED
    if identity.state == "NO_FACE":
        return "ID NO FACE", MUTED
    return f"ID {identity.state}", AMBER


def _badge(
    frame: np.ndarray,
    rect: tuple[int, int, int, int],
    text: str,
    color: tuple[int, int, int],
    scale: float = 0.40,
) -> None:
    x, y, w, h = rect
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 1)
    _text(
        frame,
        text,
        (x + max(4, (w - tw) // 2), y + (h + th) // 2),
        scale,
        (12, 14, 16),
        1,
        False,
    )


def _status_color(status: str) -> tuple[int, int, int]:
    if status == CRITICAL:
        return RED
    if status in (DISTRACTED, DROWSY, NO_FACE):
        return AMBER if status == NO_FACE else RED
    if status == "SHORT_GLANCE":
        return AMBER
    return GREEN


class HUDRenderer:
    def __init__(self, config: DMSConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Face overlay
    # ------------------------------------------------------------------
    def _draw_face(
        self,
        frame: np.ndarray,
        face: Face,
        state: DriverState,
        pose: HeadPose | None,
        eye_boxes: list[tuple[int, int, int, int]] | None,
        debug: bool,
    ) -> None:
        h, w = frame.shape[:2]
        x = max(0, int(face.x))
        y = max(0, int(face.y))
        x2 = min(w - 1, int(face.x + face.width))
        y2 = min(h - 1, int(face.y + face.height))
        color = _status_color(state.status)
        length = max(10, int(0.18 * min(x2 - x, y2 - y)))

        for (px, py, dx, dy) in (
            (x, y, 1, 1),
            (x2, y, -1, 1),
            (x, y2, 1, -1),
            (x2, y2, -1, -1),
        ):
            cv2.line(frame, (px, py), (px + dx * length, py), color, 2, cv2.LINE_AA)
            cv2.line(frame, (px, py), (px, py + dy * length), color, 2, cv2.LINE_AA)

        label = f"DRIVER {face.score:.2f}" if face.detected else "DRIVER (HELD)"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.38, 1)
        ly = max(14, y - 6)
        _panel(frame, (x, ly - th - 5, tw + 10, th + 8), PANEL, 0.72, None)
        _text(frame, label, (x + 5, ly), 0.38, color, 1, False)

        # Five YuNet landmarks.
        for idx, point in enumerate(face.landmarks):
            px, py = int(point[0]), int(point[1])
            if idx < 2:
                cv2.circle(frame, (px, py), 2, CYAN, -1, cv2.LINE_AA)
            elif idx == 2:
                cv2.circle(frame, (px, py), 2, BLUE, -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (px, py), 2, MUTED, -1, cv2.LINE_AA)

        if eye_boxes:
            for box in eye_boxes:
                ex, ey, ew, eh = [int(v) for v in box]
                cv2.rectangle(
                    frame,
                    (ex, ey),
                    (ex + ew, ey + eh),
                    CYAN,
                    1,
                    cv2.LINE_AA,
                )

        if pose is not None:
            self._draw_axes(frame, face, pose)

        if debug and pose is not None and pose.crop_box is not None:
            cx, cy, cw, ch = pose.crop_box
            cv2.rectangle(
                frame,
                (int(cx), int(cy)),
                (int(cx + cw), int(cy + ch)),
                (110, 110, 60),
                1,
                cv2.LINE_AA,
            )
            _text(
                frame,
                f"pose crop {cw}x{ch}  age={face.age}",
                (max(2, int(cx)), max(12, int(cy) - 4)),
                0.32,
                (170, 170, 110),
                1,
            )

    def _draw_axes(
        self, frame: np.ndarray, face: Face, pose: HeadPose
    ) -> None:
        """Project the head's local X/Y/Z axes onto the image."""
        origin = face.landmarks[2]
        ox, oy = float(origin[0]), float(origin[1])
        focal = max(40.0, face.width * 1.6)
        distance = 4.2
        axis_len = max(12.0, face.width * 0.32)
        rotation = np.asarray(pose.rotation, dtype=np.float32)
        axes = (
            (np.array([1.0, 0.0, 0.0]), (0, 0, 255), "X"),
            (np.array([0.0, 1.0, 0.0]), (0, 255, 0), "Y"),
            (np.array([0.0, 0.0, 1.0]), (255, 0, 0), "Z"),
        )
        for axis, color, label in axes:
            v = rotation @ axis
            denom = max(0.6, distance - float(v[2]))
            du = focal * float(v[0]) / denom
            dv = -focal * float(v[1]) / denom
            norm = math.hypot(du, dv)
            if norm < 1e-3:
                cv2.circle(
                    frame,
                    (int(ox), int(oy)),
                    max(2, int(axis_len * 0.16)),
                    color,
                    -1,
                    cv2.LINE_AA,
                )
                _text(
                    frame,
                    label,
                    (int(ox) + 3, int(oy) + 8),
                    0.30,
                    color,
                    1,
                )
                continue
            ex = int(ox + axis_len * du / norm)
            ey = int(oy + axis_len * dv / norm)
            cv2.arrowedLine(
                frame,
                (int(ox), int(oy)),
                (ex, ey),
                color,
                1,
                cv2.LINE_AA,
                tipLength=0.25,
            )
            _text(
                frame,
                label,
                (ex + 2, ey - 2),
                0.30,
                color,
                1,
            )

    # ------------------------------------------------------------------
    # Main HUD
    # ------------------------------------------------------------------
    def draw(
        self,
        frame: np.ndarray,
        state: DriverState,
        pose: HeadPose | None = None,
        face: Face | None = None,
        eyes: EyeMetrics | None = None,
        eye_boxes: list[tuple[int, int, int, int]] | None = None,
        head_img: np.ndarray | None = None,
        fps: float = 0.0,
        face_count: int = 0,
        latencies: dict[str, float] | None = None,
        calibrated: bool = False,
        recording: bool = False,
        debug: bool = False,
        help_visible: bool = False,
        minimal: bool = False,
        mirror: bool = False,
        now: float = 0.0,
        safety: SafetyState | None = None,
        identity: IdentityResult | None = None,
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        eyes = eyes or EyeMetrics()
        latencies = latencies or {}

        if face is not None:
            self._draw_face(frame, face, state, pose, eye_boxes, debug)

        color = (
            _safety_color(safety.level) if safety is not None
            else _status_color(state.status)
        )

        if not minimal:
            self._draw_main_panel(
                frame,
                state,
                pose,
                eyes,
                fps,
                face_count,
                latencies,
                calibrated,
                mirror,
                now,
                safety,
                identity,
            )
        else:
            self._draw_compact_panel(frame, state, color, safety, identity)

        if head_img is not None and self.config.show_head_inset:
            self._draw_head_inset(frame, head_img, state, color)

        if recording:
            self._draw_recording(frame)

        if (safety is not None and safety.alert) or state.alert:
            self._draw_banner(frame, state, now, safety)

        if not minimal:
            if debug:
                self._draw_debug(
                    frame, state, pose, latencies, face_count, fps
                )
            hint_y = (h - 58) if state.alert else (h - 8)
            _text(
                frame,
                "Q quit  C calibrate  F setup  H hud  M mirror  B buzzer  "
                "D debug  S snapshot  ? help",
                (12, hint_y),
                0.30,
                DIM,
                1,
            )

        if help_visible:
            self._draw_help(frame)

        if safety is not None:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 3)
        elif state.status == CRITICAL:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), RED, 3)

        return frame

    def _draw_main_panel(
        self,
        frame: np.ndarray,
        state: DriverState,
        pose: HeadPose | None,
        eyes: EyeMetrics,
        fps: float,
        face_count: int,
        latencies: dict[str, float],
        calibrated: bool,
        mirror: bool,
        now: float,
        safety: SafetyState | None = None,
        identity: IdentityResult | None = None,
    ) -> None:
        w = 350
        h = 196
        x, y = 12, 10
        _panel(frame, (x, y, w, h), PANEL, 0.80, PANEL_EDGE)
        color = (
            _safety_color(safety.level) if safety is not None
            else _status_color(state.status)
        )

        # Traffic-light badge.
        level = safety.level if safety is not None else state.status
        badge_text = (
            "GREEN" if level == "GREEN"
            else "YELLOW" if level == "YELLOW"
            else "RED" if level == "RED"
            else state.status.replace("_", " ")[:10]
        )
        _badge(frame, (x + 12, y + 12, 92, 27), badge_text, color, 0.46)

        chip = _identity_chip(identity)
        if chip is not None:
            label, chip_color = chip
            (cw, _), _ = cv2.getTextSize(label, FONT, 0.32, 1)
            _text(
                frame,
                label,
                (x + w - cw - 16, y + 30),
                0.32,
                chip_color,
                1,
                False,
            )
        elif safety is not None:
            _text(
                frame,
                f"SAFETY {safety.level}",
                (x + w - 104, y + 30),
                0.30,
                color,
                1,
                False,
            )

        # Driver state title.
        _text(
            frame,
            state.status.replace("_", " "),
            (x + 14, y + 60),
            0.55,
            color,
            2,
        )

        # Zone chip.
        zone = state.zone if state.zone not in ("UNKNOWN",) else "POSE ?"
        zlabel = f"ZONE {zone}"
        (zw, _), _ = cv2.getTextSize(zlabel, FONT, 0.32, 1)
        _panel(
            frame,
            (x + w - zw - 26, y + 42, zw + 14, 21),
            (38, 40, 44),
            0.9,
            None,
            6,
        )
        _text(
            frame,
            zlabel,
            (x + w - zw - 19, y + 57),
            0.32,
            MUTED,
            1,
            False,
        )

        message = safety.message if safety is not None else state.message
        _text(
            frame,
            message,
            (x + 14, y + 78),
            0.35,
            INK if (state.alert or (safety and safety.alert)) else MUTED,
            1,
        )

        if calibrated:
            _text(frame, "CAL", (x + w - 96, y + 78), 0.30, GREEN, 1)
        if mirror:
            _text(frame, "MIRROR", (x + w - 52, y + 78), 0.28, BLUE, 1)

        cv2.line(
            frame,
            (x + 14, y + 88),
            (x + w - 14, y + 88),
            (62, 64, 68),
            1,
            cv2.LINE_AA,
        )

        if pose is not None:
            angles = (
                f"YAW {pose.yaw:+6.1f}   PITCH {pose.pitch:+6.1f}   "
                f"ROLL {pose.roll:+6.1f}"
            )
            _text(frame, angles, (x + 16, y + 106), 0.36, INK, 1)
        else:
            _text(
                frame,
                "HEAD POSE  unavailable",
                (x + 16, y + 106),
                0.36,
                MUTED,
                1,
            )

        if eyes.valid:
            eye_color = GREEN if not eyes.both_closed else RED
            _text(
                frame,
                f"EYES {eyes.left_open:.2f}/{eyes.right_open:.2f}"
                f"   PERCLOS {eyes.perclos * 100:4.0f}%"
                f"{'*' if eyes.perclos_ready else ''}",
                (x + 16, y + 126),
                0.34,
                eye_color,
                1,
            )
            _text(
                frame,
                f"BLINK {eyes.blink_rate:4.1f}/min"
                f"   CLOSURE {eyes.closure_seconds:4.2f}s"
                f"   N {eyes.blink_count}",
                (x + 16, y + 144),
                0.33,
                MUTED,
                1,
            )
        else:
            _text(
                frame,
                "EYES  not measurable (pose / crop)",
                (x + 16, y + 126),
                0.34,
                MUTED,
                1,
            )
            _text(
                frame,
                f"PERCLOS {eyes.perclos * 100:4.0f}%   "
                f"BLINK {eyes.blink_rate:4.1f}/min",
                (x + 16, y + 144),
                0.33,
                DIM,
                1,
            )

        timer = ""
        if state.status == NO_FACE:
            timer = f"NO FACE {state.no_face_seconds:4.1f}s"
        elif state.offroad_seconds > 0.05:
            timer = f"OFF-ROAD {state.offroad_seconds:4.1f}s"
        elif state.status in (DROWSY, CRITICAL):
            timer = f"ALERT {state.status_seconds:4.1f}s"
        elif safety is not None and safety.yellow_seconds > 0.05:
            timer = f"CAUTION {safety.yellow_seconds:4.1f}s"
        else:
            timer = (
                f"FWD {state.forward_seconds:4.1f}s"
                if state.forward_seconds
                else ""
            )
        _text(frame, timer, (x + 16, y + 164), 0.33, color, 1)

        lat = "  ".join(
            f"{name[:2].upper()}{value:4.0f}"
            for name, value in latencies.items()
            if value is not None
        )
        _text(
            frame,
            f"FPS {fps:4.1f}   {lat}",
            (x + 16, y + 182),
            0.31,
            DIM,
            1,
        )

    def _draw_compact_panel(
        self,
        frame: np.ndarray,
        state: DriverState,
        color: tuple[int, int, int],
        safety: SafetyState | None = None,
        identity: IdentityResult | None = None,
    ) -> None:
        level = safety.level if safety is not None else ""
        label = state.status.replace("_", " ")
        text = f"{level}  {label}" if level else label
        (tw, _), _ = cv2.getTextSize(text, FONT, 0.50, 2)
        _panel(frame, (12, 10, tw + 34, 38), PANEL, 0.80, PANEL_EDGE)
        cv2.rectangle(frame, (12, 10), (17, 48), color, -1)
        _text(frame, text, (26, 37), 0.50, color, 2)
        chip = _identity_chip(identity)
        if chip is not None:
            chip_label, chip_color = chip
            (cw, _), _ = cv2.getTextSize(chip_label, FONT, 0.30, 1)
            _panel(
                frame,
                (12, 52, cw + 16, 22),
                PANEL,
                0.78,
                None,
                6,
            )
            _text(frame, chip_label, (20, 67), 0.30, chip_color, 1, False)

    # ------------------------------------------------------------------
    # 3D head inset
    # ------------------------------------------------------------------
    def _draw_head_inset(
        self,
        frame: np.ndarray,
        head_img: np.ndarray,
        state: DriverState,
        color: tuple[int, int, int],
    ) -> None:
        size = int(self.config.inset_size)
        h, w = frame.shape[:2]
        x = w - size - 14
        y = 12
        if head_img.shape[0] != size or head_img.shape[1] != size:
            inset = cv2.resize(
                head_img, (size, size), interpolation=cv2.INTER_AREA
            )
        else:
            inset = head_img
        frame[y : y + size, x : x + size] = inset
        cv2.rectangle(
            frame,
            (x - 1, y - 1),
            (x + size, y + size),
            (70, 68, 62),
            2,
            cv2.LINE_AA,
        )
        cv2.rectangle(frame, (x - 1, y - 1), (x + size, y - 1), color, 2)
        _panel(
            frame,
            (x, y + size + 4, size, 20),
            PANEL,
            0.85,
            None,
            4,
        )
        _text(
            frame,
            "3D HEAD  yaw / pitch / roll",
            (x + 4, y + size + 18),
            0.30,
            MUTED,
            1,
            False,
        )

    def _draw_recording(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        x = w - 92
        y = 28
        cv2.circle(frame, (x, y), 6, RED, -1, cv2.LINE_AA)
        _text(frame, "REC", (x + 11, y + 5), 0.42, INK, 1)

    # ------------------------------------------------------------------
    # Alert banner / help / debug
    # ------------------------------------------------------------------
    def _draw_banner(
        self,
        frame: np.ndarray,
        state: DriverState,
        now: float,
        safety: SafetyState | None = None,
    ) -> None:
        h, w = frame.shape[:2]
        color = (
            _safety_color(safety.level) if safety is not None
            else _status_color(state.status)
        )
        if safety is None and state.status == CRITICAL:
            color = RED
        height = 46
        y = h - height - 4
        _panel(frame, (0, y, w, height), (18, 16, 16), 0.82, None, 0)
        cv2.rectangle(frame, (0, y), (w, y + height), color, 2)

        # A subtle blink for critical/drowsy banners.
        pulse = 1.0
        if state.alarm or (safety is not None and safety.alarm):
            pulse = 0.65 + 0.35 * (0.5 + 0.5 * math.sin(now * 7.0))
        text = safety.message if safety is not None else state.message
        if state.offroad_seconds > 0.2 and state.status in (
            DISTRACTED,
            CRITICAL,
        ):
            text += f"   {state.offroad_seconds:0.1f}s"
        if state.no_face_seconds > 0.2 and state.status == NO_FACE:
            text += f"   {state.no_face_seconds:0.1f}s"
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.63, 2)
        tx = (w - tw) // 2
        ty = y + height // 2 + th // 2
        _text(
            frame,
            text,
            (tx, ty),
            0.63,
            tuple(int(c * pulse) for c in color),
            2,
        )

    def _draw_help(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        pw, ph = 470, 300
        x, y = (w - pw) // 2, (h - ph) // 2
        _panel(frame, (x, y, pw, ph), (18, 19, 22), 0.92, PANEL_EDGE, 12)
        _text(frame, "DRIVER MONITORING SYSTEM", (x + 20, y + 30), 0.56, INK, 2)
        _text(frame, "keyboard shortcuts", (x + 20, y + 50), 0.34, MUTED, 1)
        cv2.line(frame, (x + 20, y + 60), (x + pw - 20, y + 60), PANEL_EDGE, 1)
        lines = [
            ("Q / Esc", "quit"),
            ("C", "calibrate neutral head pose"),
            ("F", "framing / auto-calibrate overlay"),
            ("R", "reset session state"),
            ("H", "toggle compact HUD"),
            ("M", "mirror camera"),
            ("B", "toggle buzzer"),
            ("D", "debug overlay"),
            ("S", "save snapshot"),
            ("?", "show / hide this help"),
        ]
        for i, (key, desc) in enumerate(lines):
            yy = y + 86 + i * 22
            _text(frame, key, (x + 24, yy), 0.38, CYAN, 1)
            _text(frame, desc, (x + 105, yy), 0.38, INK, 1)
        _text(
            frame,
            f"models: {self.config.headpose_model} + {self.config.eye_model}",
            (x + 20, y + ph - 18),
            0.30,
            DIM,
            1,
        )

    def _draw_debug(
        self,
        frame: np.ndarray,
        state: DriverState,
        pose: HeadPose | None,
        latencies: dict[str, float],
        face_count: int,
        fps: float,
    ) -> None:
        h, w = frame.shape[:2]
        pw, ph = 280, 86
        x, y = 12, h - 118
        _panel(frame, (x, y, pw, ph), (16, 17, 20), 0.82, PANEL_EDGE, 8)
        _text(frame, "DEBUG", (x + 10, y + 18), 0.34, CYAN, 1)
        lines = [
            f"faces {face_count}  status {state.status}",
            f"reason {state.reason}",
            "  ".join(f"{k}:{v:.1f}ms" for k, v in latencies.items()),
            f"fps {fps:.1f}  pose_age {pose.latency_ms:.1f}ms"
            if pose
            else "pose: none",
        ]
        for i, line in enumerate(lines):
            _text(
                frame,
                line,
                (x + 10, y + 36 + i * 14),
                0.30,
                MUTED if i else INK,
                1,
            )


__all__ = ["HUDRenderer"]