"""Camera framing and neutral-pose calibration.

The head-pose model measures angles relative to the camera.  If the camera is
moved, the old neutral pose is wrong.  This module provides a short, guided
setup screen that checks framing, distance, tilt, lighting and sharpness, then
captures the neutral pose automatically once the driver has held still.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .camera import create_source
from .config import DMSConfig
from .models import Face, FaceDetector, FaceTracker, HeadPose, HeadPoseEstimator

GREEN = (105, 215, 125)
AMBER = (0, 185, 255)
RED = (55, 55, 240)
INK = (238, 238, 234)
MUTED = (158, 160, 158)
PANEL = (20, 22, 26)
FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class FrameQuality:
    face_present: bool = False
    face_box: tuple[int, int, int, int] | None = None
    face_ratio: float = 0.0
    center_x: float = 0.0
    center_y: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    brightness: float = 0.0
    sharpness: float = 0.0
    stable_seconds: float = 0.0
    ok: bool = False
    ready: bool = False
    issues: list[str] = field(default_factory=list)
    instruction: str = "NO FACE"

    @property
    def color(self) -> tuple[int, int, int]:
        if self.ready:
            return GREEN
        if self.ok:
            return AMBER
        return RED

    def as_dict(self) -> dict[str, object]:
        return {
            "face_present": self.face_present,
            "face_ratio": round(self.face_ratio, 3),
            "center_x": round(self.center_x, 3),
            "center_y": round(self.center_y, 3),
            "yaw": round(self.yaw, 1),
            "pitch": round(self.pitch, 1),
            "roll": round(self.roll, 1),
            "brightness": round(self.brightness, 1),
            "sharpness": round(self.sharpness, 1),
            "stable_seconds": round(self.stable_seconds, 2),
            "ok": self.ok,
            "ready": self.ready,
            "issues": list(self.issues),
            "instruction": self.instruction,
        }


def assess_frame(
    config: DMSConfig,
    frame: np.ndarray,
    face: Face | None,
    pose: HeadPose | None,
) -> FrameQuality:
    """Measure framing and image quality for one camera frame."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    quality = FrameQuality(
        face_present=face is not None,
        brightness=brightness,
        sharpness=sharpness,
    )

    issues: list[str] = []
    if face is None:
        issues.append("NO FACE - SIT IN FRONT OF THE CAMERA")
        quality.issues = issues
        quality.instruction = issues[0]
        return quality

    x, y, fw, fh = [int(v) for v in face.box]
    quality.face_box = (x, y, fw, fh)
    quality.face_ratio = float(face.width / max(1, w))
    quality.center_x = float(
        abs(face.center[0] - w / 2.0) / max(1.0, w / 2.0)
    )
    quality.center_y = float(
        abs(face.center[1] - h / 2.0) / max(1.0, h / 2.0)
    )

    if pose is not None:
        quality.yaw = float(pose.yaw)
        quality.pitch = float(pose.pitch)
        quality.roll = float(pose.roll)
    else:
        issues.append("FACE THE CAMERA")

    if quality.face_ratio < config.setup_face_ratio_min:
        issues.append("MOVE CLOSER")
    elif quality.face_ratio > config.setup_face_ratio_max:
        issues.append("MOVE BACK")

    if (
        quality.center_x > config.setup_center_tolerance
        or quality.center_y > config.setup_center_tolerance
    ):
        issues.append("CENTER YOUR FACE")

    if pose is not None:
        if abs(pose.yaw) > config.setup_max_yaw:
            issues.append("LOOK STRAIGHT")
        if abs(pose.pitch) > config.setup_max_pitch:
            issues.append("LOOK STRAIGHT")
        if abs(pose.roll) > config.setup_max_roll:
            issues.append("LEVEL YOUR HEAD")

    if brightness < config.setup_min_brightness:
        issues.append("TOO DARK - ADD LIGHT")
    elif brightness > config.setup_max_brightness:
        issues.append("TOO BRIGHT - REDUCE LIGHT")

    if sharpness < config.setup_min_sharpness:
        issues.append("HOLD STILL / CLEAN THE LENS")

    # Keep only unique messages while preserving priority order.
    seen: set[str] = set()
    unique = []
    for issue in issues:
        if issue not in seen:
            seen.add(issue)
            unique.append(issue)

    quality.issues = unique
    quality.ok = not unique
    quality.instruction = "HOLD STILL" if quality.ok else unique[0]
    return quality


class FrameSetup:
    """Tracks how long the framing has been good enough to calibrate."""

    def __init__(self, config: DMSConfig) -> None:
        self.config = config
        self._stable_since: float | None = None
        self.last: FrameQuality | None = None

    def update(
        self,
        frame: np.ndarray,
        face: Face | None,
        pose: HeadPose | None,
        now: float,
    ) -> FrameQuality:
        quality = assess_frame(self.config, frame, face, pose)
        if quality.ok:
            if self._stable_since is None:
                self._stable_since = float(now)
            quality.stable_seconds = max(0.0, float(now) - self._stable_since)
            quality.ready = (
                quality.stable_seconds >= self.config.setup_settle_seconds
            )
            if quality.ready:
                quality.instruction = "READY - CALIBRATING"
        else:
            self._stable_since = None
        self.last = quality
        return quality

    def reset(self) -> None:
        self._stable_since = None
        self.last = None


def save_setup_calibration(
    path: Path | str,
    yaw: float,
    pitch: float,
    roll: float,
    quality: FrameQuality | None = None,
    resolution: tuple[int, int] | None = None,
    source: str = "",
) -> Path:
    """Write the neutral pose in the format used by StateMachine."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "version": 3,
        "yaw": float(yaw),
        "pitch": float(pitch),
        "roll": float(roll),
        "calibrated": True,
        "created": datetime.now().isoformat(timespec="seconds"),
        "source": source,
    }
    if quality is not None:
        payload["face_ratio"] = round(quality.face_ratio, 3)
        payload["brightness"] = round(quality.brightness, 1)
        payload["sharpness"] = round(quality.sharpness, 1)
        if quality.face_box is not None:
            payload["face_box"] = list(quality.face_box)
    if resolution is not None:
        payload["resolution"] = [int(resolution[0]), int(resolution[1])]
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def _panel(
    frame: np.ndarray,
    rect: tuple[int, int, int, int],
    color: tuple[int, int, int] = PANEL,
    alpha: float = 0.82,
) -> None:
    x, y, w, h = rect
    h_img, w_img = frame.shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(w, w_img - x)
    h = min(h, h_img - y)
    if w <= 0 or h <= 0:
        return
    roi = frame[y : y + h, x : x + w]
    overlay = np.empty_like(roi)
    overlay[:] = color
    cv2.addWeighted(roi, 1.0 - alpha, overlay, alpha, 0.0, dst=roi)


def _text(
    frame: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float = 0.45,
    color: tuple[int, int, int] = INK,
    thickness: int = 1,
) -> None:
    cv2.putText(
        frame,
        text,
        (org[0] + 1, org[1] + 1),
        FONT,
        scale,
        (0, 0, 0),
        thickness + 1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame, text, org, FONT, scale, color, thickness, cv2.LINE_AA
    )


def draw_setup_overlay(
    frame: np.ndarray,
    quality: FrameQuality,
    driver_name: str = "",
) -> np.ndarray:
    """Draw the framing guide, face box and setup instructions."""
    h, w = frame.shape[:2]
    color = quality.color

    # Target frame: a comfortable face position.
    tw = int(w * 0.32)
    th = int(h * 0.52)
    tx = (w - tw) // 2
    ty = (h - th) // 2
    cv2.rectangle(frame, (tx, ty), (tx + tw, ty + th), color, 2, cv2.LINE_AA)
    cv2.line(frame, (w // 2, ty - 14), (w // 2, ty + 14), color, 2, cv2.LINE_AA)
    cv2.line(frame, (tx - 14, h // 2), (tx + 14, h // 2), color, 2, cv2.LINE_AA)

    if quality.face_box is not None:
        x, y, fw, fh = quality.face_box
        cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2, cv2.LINE_AA)

    _panel(frame, (10, 10, 310, 52), PANEL, 0.86)
    _text(frame, "CAMERA SETUP", (22, 32), 0.55, INK, 2)
    _text(
        frame,
        driver_name or "position the camera at eye level",
        (22, 50),
        0.36,
        MUTED,
        1,
    )

    _panel(frame, (10, h - 112, w - 20, 102), PANEL, 0.88)
    _text(frame, quality.instruction, (24, h - 82), 0.62, color, 2)
    _text(
        frame,
        f"FACE {quality.face_ratio * 100:4.0f}%   "
        f"CENTER {quality.center_x * 100:3.0f}/{quality.center_y * 100:3.0f}%   "
        f"YAW {quality.yaw:+5.1f}  PITCH {quality.pitch:+5.1f}  "
        f"ROLL {quality.roll:+5.1f}",
        (24, h - 58),
        0.36,
        INK,
        1,
    )
    _text(
        frame,
        f"LIGHT {quality.brightness:5.0f}   SHARP {quality.sharpness:5.0f}   "
        f"STABLE {quality.stable_seconds:4.1f}s",
        (24, h - 38),
        0.35,
        MUTED,
        1,
    )
    _text(
        frame,
        "Hold still and look straight. Q/Esc cancels.",
        (24, h - 18),
        0.32,
        MUTED,
        1,
    )
    return frame


def run_camera_setup(
    config: DMSConfig,
    headless: bool = False,
    timeout: float | None = None,
) -> bool:
    """Open the source, guide framing, and save the neutral pose."""
    timeout = config.setup_timeout if timeout is None else float(timeout)
    source = create_source(config)
    detector = FaceDetector(
        config.face_model_path,
        config.width,
        config.height,
        confidence=config.face_confidence,
        nms=config.face_nms,
        top_k=config.face_top_k,
        input_scale=config.face_input_scale,
    )
    pose_estimator = HeadPoseEstimator(
        config.headpose_model_path,
        threads=config.headpose_threads,
        expand=config.headpose_expand,
    )
    tracker = FaceTracker(
        smoothing=config.track_smoothing,
        max_missing=config.track_max_missing,
    )
    setup = FrameSetup(config)
    started = time.monotonic()
    last_instruction = ""
    window = "DMS Camera Setup"

    if not headless:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, raw, _ = source.read()
            if not ok or raw is None:
                if source.info.is_file:
                    break
                time.sleep(0.05)
                continue
            frame = cv2.resize(raw, (config.width, config.height))
            if config.mirror:
                frame = cv2.flip(frame, 1)
            faces = detector.detect(frame)
            face = tracker.update(faces, config.width, config.height)
            pose = (
                pose_estimator.estimate(frame, face)
                if face is not None
                else None
            )
            now = time.monotonic()
            quality = setup.update(frame, face, pose, now)

            if quality.instruction != last_instruction:
                print(f"[SETUP] {quality.instruction}")
                last_instruction = quality.instruction

            if not headless:
                draw_setup_overlay(frame, quality)
                cv2.imshow(window, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("[SETUP] cancelled")
                    return False

            if quality.ready and pose is not None:
                target = save_setup_calibration(
                    config.calibration_file,
                    pose.yaw,
                    pose.pitch,
                    pose.roll,
                    quality=quality,
                    resolution=(config.width, config.height),
                    source=source.info.description,
                )
                print(
                    "[SETUP] saved neutral pose "
                    f"yaw={pose.yaw:+.1f} pitch={pose.pitch:+.1f} "
                    f"roll={pose.roll:+.1f} -> {target}"
                )
                return True

            if timeout > 0 and (now - started) >= timeout:
                print(
                    "[SETUP] timed out. Use --setup-camera again in better "
                    "light and with your whole face visible."
                )
                return False
    finally:
        source.release()
        if not headless:
            cv2.destroyWindow(window)


__all__ = [
    "FrameQuality",
    "FrameSetup",
    "assess_frame",
    "draw_setup_overlay",
    "run_camera_setup",
    "save_setup_calibration",
]
