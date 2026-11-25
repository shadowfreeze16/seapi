"""Guided driver enrolment for the SFace identity profile.

    python driver_monitor.py --enroll-driver Alice

The camera shows a live preview with a progress bar.  Frames are accepted only
when the face is large enough and the head is frontal.  About 30 accepted
frames are averaged into one embedding, then written to
``driver_identity.json`` with mode ``0600``.  Only the embedding is stored,
never photographs.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from .camera import create_source
from .config import DMSConfig
from .eyes import EyeAnalyzer
from .identity import (
    EnrollmentSession,
    FaceEmbedder,
    IdentityStore,
    identity_quality_ok,
)
from .models import (
    EyeStateEstimator,
    FaceDetector,
    FaceTracker,
    HeadPoseEstimator,
)

GREEN = (105, 215, 125)
AMBER = (0, 185, 255)
RED = (55, 55, 240)
INK = (238, 238, 234)
MUTED = (158, 160, 158)
PANEL = (20, 22, 26)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _panel(frame: np.ndarray, rect: tuple[int, int, int, int], alpha: float = 0.84) -> None:
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
    overlay[:] = PANEL
    cv2.addWeighted(roi, 1.0 - alpha, overlay, alpha, 0.0, dst=roi)


def _text(
    frame: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float = 0.46,
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
    cv2.putText(frame, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_enrollment_overlay(
    frame: np.ndarray,
    name: str,
    count: int,
    target: int,
    instruction: str,
    ok: bool,
) -> np.ndarray:
    h, w = frame.shape[:2]
    color = GREEN if ok else AMBER
    _panel(frame, (10, 10, 340, 52))
    _text(frame, "DRIVER ENROLMENT", (22, 32), 0.55, INK, 2)
    _text(frame, name, (22, 50), 0.38, MUTED, 1)

    bar_w = w - 40
    bar_x = 20
    bar_y = h - 82
    _panel(frame, (10, h - 116, w - 20, 106))
    _text(frame, instruction, (24, h - 88), 0.55, color, 2)
    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + bar_w, bar_y + 14),
        (48, 50, 54),
        -1,
    )
    ratio = 0.0 if target <= 0 else min(1.0, count / float(target))
    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + int(bar_w * ratio), bar_y + 14),
        color,
        -1,
    )
    _text(
        frame,
        f"{count} / {target} good frames",
        (bar_x, bar_y - 8),
        0.38,
        INK,
        1,
    )
    _text(
        frame,
        "Look straight at the camera. Q / Esc cancels.",
        (bar_x, h - 20),
        0.34,
        MUTED,
        1,
    )
    return frame


def run_enrollment(
    config: DMSConfig,
    headless: bool = False,
    timeout: float | None = None,
) -> bool:
    """Capture frontal frames and save an identity profile."""
    name = (config.enroll_driver or "").strip()
    if not name:
        print("[ENROLL] no driver name supplied")
        return False
    limit = config.enroll_timeout if timeout is None else float(timeout)

    store = IdentityStore(config.identity_store, model=config.identity_model)
    if store.get(name) is not None and not config.enroll_force:
        print(
            f"[ENROLL] driver '{name}' already exists in {store.path}\n"
            "         use --enroll-force to replace the profile"
        )
        return False
    try:
        embedder = FaceEmbedder(
            config.identity_model_path, threads=config.identity_threads
        )
    except FileNotFoundError as exc:
        print(f"[ENROLL] {exc}")
        return False

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
    eye_estimator = EyeStateEstimator(
        config.eye_model_path,
        threads=config.eye_threads,
        crop_scale=config.eye_crop_scale,
        crop_aspect=config.eye_crop_aspect,
        min_crop_px=config.min_eye_crop_px,
    )
    tracker = FaceTracker(
        smoothing=config.track_smoothing,
        max_missing=config.track_max_missing,
    )
    eye_analyzer = EyeAnalyzer(config)
    session = EnrollmentSession(
        name=name,
        embedder=embedder,
        config=config,
        target_samples=config.enroll_samples,
    )
    window = "DMS Driver Enrolment"
    if not headless:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    print(
        f"[ENROLL] capturing {config.enroll_samples} frontal frames for "
        f"'{name}'"
    )
    print("[ENROLL] look straight at the camera in good light")

    started = time.monotonic()
    last_count = 0
    last_instruction = ""
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
            observation = None
            if face is not None:
                observation = eye_estimator.estimate(
                    frame,
                    face,
                    yaw=pose.yaw if pose is not None else None,
                    max_yaw=config.max_eye_yaw,
                )
            eyes = eye_analyzer.update(observation, time.monotonic())

            quality_ok, instruction = identity_quality_ok(
                config, face, pose, eyes
            )
            session.add(frame, face, pose, eyes, time.monotonic())

            if session.count != last_count:
                last_count = session.count
                if last_count % 5 == 0 or session.complete:
                    print(
                        f"[ENROLL] {last_count}/{config.enroll_samples} frames"
                    )
            if instruction != last_instruction:
                print(f"[ENROLL] {instruction}")
                last_instruction = instruction

            if not headless:
                draw_enrollment_overlay(
                    frame,
                    name,
                    session.count,
                    config.enroll_samples,
                    instruction,
                    quality_ok,
                )
                cv2.imshow(window, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("[ENROLL] cancelled")
                    return False

            if session.complete:
                profile = session.finish(
                    store, replace=config.enroll_force
                )
                print(
                    f"[ENROLL] saved '{profile.name}' "
                    f"({profile.samples} frames) -> {store.path}"
                )
                print("[ENROLL] the file contains embeddings only (mode 0600)")
                return True

            if limit > 0 and (time.monotonic() - started) >= limit:
                print(
                    f"[ENROLL] timed out after {limit:.0f}s with "
                    f"{session.count}/{config.enroll_samples} frames"
                )
                return False
    finally:
        source.release()
        if not headless:
            cv2.destroyWindow(window)


__all__ = ["draw_enrollment_overlay", "run_enrollment"]
