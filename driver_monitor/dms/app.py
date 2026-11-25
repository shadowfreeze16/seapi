"""Application loop: camera -> models -> state machine -> HUD/events."""

from __future__ import annotations

import os
import sys
import time
from collections import Counter
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .camera import FrameSource, create_source
from .config import DMSConfig
from .dashboard_client import (
    DashboardClient,
    encode_snapshot,
    make_event_payload,
    make_status_payload,
)
from .events import EventLogger
from .eyes import EyeAnalyzer, EyeMetrics
from .filters import AngleSmoother, RateMeter, TimingStats
from .gps import GPSClient
from .head3d import HeadRenderer
from .identity import (
    OFF,
    FaceEmbedder,
    IdentityResult,
    IdentityStore,
    IdentityVerifier,
)
from .models import (
    EyeObservation,
    EyeStateEstimator,
    Face,
    FaceDetector,
    FaceTracker,
    HeadPose,
    HeadPoseEstimator,
)
from .render import HUDRenderer
from .safety import SafetyAssessor, SafetyState
from .setup import FrameSetup, draw_setup_overlay, save_setup_calibration
from .state import DriverState, StateMachine
from .voice import VoiceWarner


def display_available() -> bool:
    if sys.platform == "darwin" or os.name == "nt":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class BuzzerController:
    """Non-blocking buzzer with a simple beep pattern."""

    def __init__(self, enabled: bool, gpio: int = 18) -> None:
        self.requested = bool(enabled)
        self.enabled = False
        self.device = None
        self._on_until = 0.0
        self._next_beep = 0.0
        if not enabled:
            return
        try:
            from gpiozero import Buzzer  # type: ignore

            self.device = Buzzer(int(gpio))
            self.enabled = True
        except Exception as exc:
            print(f"[BUZZER] disabled ({exc})")

    def update(self, alarm: bool, now: float) -> None:
        if not self.enabled or self.device is None:
            return
        if self._on_until and now >= self._on_until:
            self.device.off()
            self._on_until = 0.0
        if alarm and now >= self._next_beep:
            self.device.on()
            self._on_until = now + 0.16
            self._next_beep = now + 0.55
        if not alarm:
            self._next_beep = now

    def off(self) -> None:
        if self.device is not None:
            try:
                self.device.off()
            except Exception:
                pass
        self._on_until = 0.0

    def toggle(self) -> bool:
        if self.device is None:
            self.enabled = False
            return False
        self.enabled = not self.enabled
        if not self.enabled:
            self.off()
        return self.enabled


class DriverMonitorApp:
    def __init__(
        self,
        config: DMSConfig,
        headless: bool = False,
        show_windows: bool | None = None,
    ) -> None:
        self.config = config
        config.validate()
        self.started_wall = datetime.now()
        self.started = time.monotonic()
        self.session_id = self.started_wall.strftime("%Y%m%d_%H%M%S")
        self._last_fps_log = 0.0

        if config.fast_mode:
            config.show_head_window = False
            config.show_head_inset = False
            if not display_available():
                headless = True
                config.show_windows = False

        if show_windows is None:
            show_windows = config.show_windows
        self.headless = bool(headless or not show_windows or not display_available())
        if self.headless:
            config.show_windows = False
            config.show_head_window = False

        self.source: FrameSource = create_source(config)
        self.mirror = bool(config.mirror)
        self.recording_path = Path(config.record_path) if config.record_path else None

        self.detector = FaceDetector(
            config.face_model_path,
            config.width,
            config.height,
            confidence=config.face_confidence,
            nms=config.face_nms,
            top_k=config.face_top_k,
            input_scale=config.face_input_scale,
        )
        self.pose_estimator = HeadPoseEstimator(
            config.headpose_model_path,
            threads=config.headpose_threads,
            expand=config.headpose_expand,
        )
        self.eye_estimator = EyeStateEstimator(
            config.eye_model_path,
            threads=config.eye_threads,
            crop_scale=config.eye_crop_scale,
            crop_aspect=config.eye_crop_aspect,
            min_crop_px=config.min_eye_crop_px,
        )

        self.tracker = FaceTracker(
            smoothing=config.track_smoothing,
            max_missing=config.track_max_missing,
        )
        self.smoother = AngleSmoother(
            min_cutoff=config.smooth_min_cutoff,
            beta=config.smooth_beta,
            d_cutoff=config.smooth_d_cutoff,
        )
        self.eye_analyzer = EyeAnalyzer(config)
        self.state_machine = StateMachine(config)
        self.safety = SafetyAssessor(config)
        self.events = EventLogger(config)
        self.hud = HUDRenderer(config)

        self.identity_store: IdentityStore | None = None
        self.identity_verifier: IdentityVerifier | None = None
        self._initialize_identity()
        self.last_identity = IdentityResult(
            state=OFF, message="IDENTITY OFF"
        )
        self.last_safety = SafetyState()

        self.gps = GPSClient(config)
        self.voice = VoiceWarner(config)
        self.dashboard = DashboardClient(config, self.session_id)
        self.camera_setup: dict[str, Any] = {
            "calibrated": bool(self.state_machine.calibrated),
            "mode": "auto" if config.auto_calibrate else "manual"
            if self.state_machine.calibrated else "none",
            "driver": config.authorized_drivers[0]
            if config.authorized_drivers else None,
        }
        self.frame_setup: FrameSetup | None = (
            FrameSetup(config) if config.auto_calibrate else None
        )
        self._setup_instruction = ""
        self.last_setup_quality: dict[str, Any] = {}
        self.safety_counts: Counter[str] = Counter()

        self.head_renderer: HeadRenderer | None = None
        want_head = (
            config.show_head_inset
            and (not self.headless or self.recording_path is not None)
        ) or (config.show_head_window and not self.headless)
        if want_head:
            self.head_renderer = HeadRenderer(config.head_canvas_size)

        self.buzzer = BuzzerController(
            enabled=config.buzzer_enabled,
            gpio=config.buzzer_gpio,
        )

        self.rate = RateMeter()
        self.timings = {
            "face": TimingStats("face"),
            "pose": TimingStats("pose"),
            "eyes": TimingStats("eyes"),
            "head": TimingStats("head"),
        }
        self.last_pose: HeadPose | None = None
        self.last_pose_now = 0.0
        self.last_face_now = 0.0
        self.last_eyes = EyeMetrics()
        self.last_observation: EyeObservation | None = None
        self.last_state = DriverState()
        self.frame_index = 0
        self.frames_written = 0
        self.state_counts: Counter[str] = Counter()
        self.max_offroad = 0.0
        self.max_closure = 0.0
        self.max_no_face = 0.0
        self.min_perclos: float | None = None
        self.max_perclos: float | None = None
        self.max_blink_rate = 0.0
        self.buzzer_on = self.buzzer.enabled

        self.hud_minimal = False
        self.debug = bool(config.debug)
        self.help_visible = False
        self.toast = ""
        self.toast_until = 0.0

        self.recorder: cv2.VideoWriter | None = None
        self.record_fps = float(config.fps)
        self.last_annotated: np.ndarray | None = None
        self._initialize_recorder()

        if config.use_calibration:
            if self.state_machine.load_calibration(config.calibration_file):
                print(
                    "[CAL] loaded neutral pose: "
                    f"yaw={self.state_machine.yaw_offset:+.1f} "
                    f"pitch={self.state_machine.pitch_offset:+.1f} "
                    f"roll={self.state_machine.roll_offset:+.1f}"
                )
            else:
                if not config.auto_calibrate:
                    print("NO CALIBRATION - run --setup-camera first")

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    def _initialize_identity(self) -> None:
        if not self.config.identity_enabled:
            return
        try:
            self.identity_store = IdentityStore(
                self.config.identity_store,
                model=self.config.identity_model,
            )
        except (OSError, ValueError) as exc:
            print(f"[ID] profile store unavailable: {exc}")
            return
        if not self.identity_store.profiles:
            print("IDENTITY OFF - no enrolled driver")
            print(
                "[ID] no enrolled drivers.  Enrol one with:\n"
                "     python driver_monitor.py --enroll-driver NAME"
            )
            return
        try:
            embedder = FaceEmbedder(
                self.config.identity_model_path,
                threads=self.config.identity_threads,
            )
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"[ID] disabled: {exc}")
            return
        self.identity_verifier = IdentityVerifier(
            self.config,
            self.identity_store.profiles,
            embedder,
            self.config.authorized_drivers,
        )
        enrolled = ", ".join(self.identity_store.names())
        if self.config.authorized_drivers:
            missing = [
                name
                for name in self.config.authorized_drivers
                if self.identity_store.get(name) is None
            ]
            if missing:
                print(
                    f"[ID] WARNING: not enrolled: {', '.join(missing)}. "
                    "Every face will be treated as unauthorized."
                )
            print(
                f"[ID] enforcing driver: {', '.join(self.config.authorized_drivers)}"
            )
        else:
            print(f"[ID] identify-only mode. Enrolled: {enrolled}")

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def _initialize_recorder(self) -> None:
        if self.recording_path is None:
            return
        self.recording_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = self.recording_path.suffix.lower()
        if suffix == ".mp4":
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        else:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        # Match the source clock when playing a file so the recording has the
        # same duration as the input.
        source_fps = float(self.source.info.fps)
        self.record_fps = (
            source_fps if 1.0 <= source_fps <= 240.0 else float(self.config.fps)
        )
        writer = cv2.VideoWriter(
            str(self.recording_path),
            fourcc,
            self.record_fps,
            (int(self.config.width), int(self.config.height)),
        )
        if not writer.isOpened():
            raise RuntimeError(
                f"Cannot open video writer for {self.recording_path}"
            )
        self.recorder = writer
        print(f"[REC] {self.recording_path}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> dict[str, Any]:
        self._print_banner()
        self.gps.start()
        self.dashboard.start(
            {
                "session_id": self.session_id,
                "timestamp": self.started_wall.isoformat(timespec="seconds"),
                "started_at": self.started_wall.isoformat(timespec="seconds"),
                "source": asdict(self.source.info),
                "driver": self.config.authorized_drivers[0]
                if self.config.authorized_drivers else None,
                "identity_enforced": bool(self.config.authorized_drivers),
                "voice_enabled": self.voice.active,
                "gps_enabled": self.gps.enabled,
            }
        )
        if self.headless:
            print("[UI] headless mode (no cv2 windows)")
        if not self.headless:
            cv2.namedWindow(self.config.window_name, cv2.WINDOW_NORMAL)
            if self.config.show_head_window and self.head_renderer is not None:
                cv2.namedWindow(
                    self.config.head_window_name, cv2.WINDOW_NORMAL
                )

        camera_failures = 0
        failure_limit = 600 if self.source.info.is_network else 40
        last_wall = time.monotonic()

        try:
            while True:
                ok, raw, media_now = self.source.read()
                if not ok or raw is None:
                    if self.source.info.is_file:
                        break
                    camera_failures += 1
                    if camera_failures > failure_limit:
                        raise RuntimeError(
                            "Camera stopped delivering frames."
                        )
                    time.sleep(0.05)
                    continue
                camera_failures = 0

                self.frame_index += 1
                wall_now = time.monotonic()
                now = media_now if self.source.info.is_file else wall_now
                last_wall = wall_now

                frame = cv2.resize(
                    raw, (self.config.width, self.config.height)
                )
                if self.mirror:
                    frame = cv2.flip(frame, 1)

                # ------------------------------------------------------
                # Detection + tracking
                # ------------------------------------------------------
                t0 = time.perf_counter()
                faces = self.detector.detect(frame)
                face = self.tracker.update(
                    faces, self.config.width, self.config.height
                )
                face_present = face is not None
                self.timings["face"].add(
                    (time.perf_counter() - t0) * 1000.0
                )
                if face is not None:
                    self.last_face_now = now

                # ------------------------------------------------------
                # Head pose (scheduled every N frames)
                # ------------------------------------------------------
                pose: HeadPose | None = None
                if face is not None:
                    run_pose = (
                        self.last_pose is None
                        or self.frame_index % self.config.headpose_every == 0
                        or not face.detected
                    )
                    if run_pose:
                        t0 = time.perf_counter()
                        raw_pose = self.pose_estimator.estimate(frame, face)
                        self.timings["pose"].add(
                            (time.perf_counter() - t0) * 1000.0
                        )
                        if raw_pose is not None:
                            yaw, pitch, roll = self.smoother.update(
                                raw_pose.yaw,
                                raw_pose.pitch,
                                raw_pose.roll,
                                now,
                            )
                            raw_pose.yaw = yaw
                            raw_pose.pitch = pitch
                            raw_pose.roll = roll
                            self.last_pose = raw_pose
                            self.last_pose_now = now
                    if (
                        self.last_pose is not None
                        and (now - self.last_pose_now) <= 0.75
                    ):
                        pose = self.last_pose
                else:
                    if now - self.last_face_now > 0.6:
                        self.last_pose = None
                        self.smoother.reset()

                # ------------------------------------------------------
                # Eyes
                # ------------------------------------------------------
                observation: EyeObservation | None = None
                if face is not None:
                    t0 = time.perf_counter()
                    observation = self.eye_estimator.estimate(
                        frame,
                        face,
                        yaw=pose.yaw if pose is not None else None,
                        max_yaw=self.config.max_eye_yaw,
                    )
                    self.timings["eyes"].add(
                        (time.perf_counter() - t0) * 1000.0
                    )
                self.last_observation = observation
                eyes = self.eye_analyzer.update(observation, now)
                self.last_eyes = eyes

                # ------------------------------------------------------
                # Identity (scheduled every N frames)
                # ------------------------------------------------------
                identity = self.last_identity
                if self.identity_verifier is not None:
                    if face is None or not face.detected:
                        identity = self.identity_verifier.no_face()
                    elif self.frame_index % self.config.identity_every == 0:
                        identity = self.identity_verifier.update(
                            frame,
                            face,
                            pose,
                            wall_now,
                            self.frame_index,
                        )
                    self.last_identity = identity

                # ------------------------------------------------------
                # Camera framing / automatic calibration
                # ------------------------------------------------------
                if self.frame_setup is not None:
                    quality = self.frame_setup.update(
                        frame, face, pose, wall_now
                    )
                    self.last_setup_quality = quality.as_dict()
                    if quality.instruction != self._setup_instruction:
                        print(f"[SETUP] {quality.instruction}")
                        self._setup_instruction = quality.instruction
                    if quality.ready and pose is not None:
                        self.state_machine.calibrate(
                            pose.yaw, pose.pitch, pose.roll
                        )
                        try:
                            save_setup_calibration(
                                self.config.calibration_file,
                                pose.yaw,
                                pose.pitch,
                                pose.roll,
                                quality=quality,
                                resolution=(
                                    self.config.width,
                                    self.config.height,
                                ),
                                source=self.source.info.description,
                            )
                            self.camera_setup = {
                                "calibrated": True,
                                "mode": "auto",
                                "driver": self.config.authorized_drivers[0]
                                if self.config.authorized_drivers else None,
                                "quality": quality.as_dict(),
                            }
                            self.toast = "AUTO CALIBRATED"
                            self.toast_until = wall_now + 3.0
                            print(
                                "[SETUP] neutral pose saved: "
                                f"yaw={pose.yaw:+.1f} "
                                f"pitch={pose.pitch:+.1f} "
                                f"roll={pose.roll:+.1f}"
                            )
                            self.voice.announce_setup_complete(wall_now)
                        except OSError as exc:
                            self.toast = f"CALIBRATION FAILED: {exc}"
                            self.toast_until = wall_now + 3.0
                            print(f"[SETUP] save failed: {exc}")
                        self.frame_setup = None

                # ------------------------------------------------------
                # State
                # ------------------------------------------------------
                state = self.state_machine.update(
                    now,
                    pose.yaw if pose is not None else 0.0,
                    pose.pitch if pose is not None else 0.0,
                    pose.roll if pose is not None else 0.0,
                    eyes,
                    face_present,
                    pose_valid=pose is not None,
                )
                self.last_state = state
                self.state_counts[state.status] += 1

                # ------------------------------------------------------
                # Three-level safety status
                # ------------------------------------------------------
                safety = self.safety.update(
                    wall_now, state, identity, face_present
                )
                self.last_safety = safety
                self.safety_counts[safety.level] += 1
                self.voice.update(
                    wall_now, safety, state.reason, state.status
                )
                self.max_offroad = max(self.max_offroad, state.offroad_seconds)
                self.max_closure = max(self.max_closure, eyes.closure_seconds)
                self.max_no_face = max(self.max_no_face, state.no_face_seconds)
                self.max_blink_rate = max(self.max_blink_rate, eyes.blink_rate)
                if eyes.valid:
                    if self.min_perclos is None:
                        self.min_perclos = eyes.perclos
                    self.min_perclos = min(self.min_perclos, eyes.perclos)
                    self.max_perclos = max(
                        self.max_perclos or 0.0, eyes.perclos
                    )

                # ------------------------------------------------------
                # 3D head
                # ------------------------------------------------------
                head_img: np.ndarray | None = None
                if self.head_renderer is not None:
                    t0 = time.perf_counter()
                    if face_present:
                        head_img = self.head_renderer.render(
                            pose.yaw if pose is not None else 0.0,
                            pose.pitch if pose is not None else 0.0,
                            pose.roll if pose is not None else 0.0,
                            eye_open=(
                                eyes.left_open if eyes.valid else 1.0,
                                eyes.right_open if eyes.valid else 1.0,
                            ),
                            status=state.status,
                            zone=state.zone,
                            eyes_valid=eyes.valid,
                            pose_valid=True,
                        )
                    else:
                        head_img = self.head_renderer.render(
                            0.0,
                            0.0,
                            0.0,
                            status="NO_FACE",
                            zone="NO_FACE",
                            eyes_valid=False,
                            pose_valid=False,
                        )
                    self.timings["head"].add(
                        (time.perf_counter() - t0) * 1000.0
                    )

                # ------------------------------------------------------
                # HUD
                # ------------------------------------------------------
                fps = self.rate.tick()
                latencies = {
                    "face": self.timings["face"].last,
                    "pose": self.timings["pose"].last,
                    "eye": self.timings["eyes"].last,
                    "head": self.timings["head"].last,
                }
                eye_boxes = (
                    self.last_observation.boxes
                    if self.last_observation is not None
                    else None
                )
                self.hud.draw(
                    frame,
                    state,
                    pose=pose,
                    face=face,
                    eyes=eyes,
                    eye_boxes=eye_boxes,
                    head_img=head_img,
                    fps=fps,
                    face_count=len(faces),
                    latencies=latencies,
                    calibrated=self.state_machine.calibrated,
                    recording=self.recorder is not None,
                    debug=self.debug,
                    help_visible=self.help_visible,
                    minimal=self.hud_minimal,
                    mirror=self.mirror,
                    now=wall_now,
                    safety=safety,
                    identity=identity,
                )

                # 5-second periodic debug and FPS monitoring
                if wall_now - self._last_fps_log >= 5.0:
                    self._last_fps_log = wall_now
                    lo = round(eyes.left_open, 2) if eyes.valid else 0.0
                    ro = round(eyes.right_open, 2) if eyes.valid else 0.0
                    perc = round(eyes.perclos, 2)
                    cs = round(eyes.closure_seconds, 2)
                    f_val = round(fps, 1)
                    print(
                        f"[STATUS 5s] FPS={f_val} | Left={lo} Right={ro} | "
                        f"PERCLOS={perc} Closure={cs}s | State={state.status} Safety={safety.level}"
                    )
                    if fps < 8.0:
                        print("LOW FPS - check camera / models")

                if (
                    self.frame_setup is not None
                    and self.frame_setup.last is not None
                ):
                    draw_setup_overlay(
                        frame,
                        self.frame_setup.last,
                        self.config.authorized_drivers[0]
                        if self.config.authorized_drivers else "",
                    )

                if self.toast and wall_now < self.toast_until:
                    self._draw_toast(frame, self.toast)
                self.last_annotated = frame

                # ------------------------------------------------------
                # Events
                # ------------------------------------------------------
                gps_snapshot = self.gps.snapshot(wall_now)
                context = {
                    "face_count": len(faces),
                    "yaw": pose.yaw if pose is not None else None,
                    "pitch": pose.pitch if pose is not None else None,
                    "roll": pose.roll if pose is not None else None,
                    "eye_left": eyes.left_open if eyes.valid else None,
                    "eye_right": eyes.right_open if eyes.valid else None,
                    "perclos": eyes.perclos,
                    "blink_rate": eyes.blink_rate,
                    "closure_s": eyes.closure_seconds,
                    "fps": fps,
                    "latency_ms": self.timings["face"].last
                    + self.timings["pose"].last
                    + self.timings["eyes"].last
                    + self.timings["head"].last,
                    "status": state.status,
                    "safety_level": safety.level,
                    "safety_reason": safety.reason,
                    "identity_state": identity.state,
                    "identity_name": identity.name or identity.best_name,
                    "identity_score": identity.score or identity.best_score,
                    "gps": gps_snapshot,
                }
                for request in [*state.events, *safety.events]:
                    row = self.events.record(
                        request,
                        frame=frame,
                        context=context,
                        now=wall_now,
                    )
                    if row is not None and self.dashboard.enabled:
                        snapshot_b64 = encode_snapshot(
                            frame,
                            max_width=self.config.dashboard_snapshot_max_width,
                            quality=self.config.dashboard_snapshot_quality,
                        )
                        self.dashboard.send_event(
                            make_event_payload(
                                self.session_id,
                                request,
                                row,
                                safety,
                                identity,
                                gps_snapshot,
                                snapshot_b64,
                            )
                        )

                # Dashboard status snapshot (the client rate-limits this).
                self.dashboard.set_status(
                    make_status_payload(
                        self.session_id,
                        safety,
                        state,
                        identity,
                        eyes,
                        pose,
                        fps,
                        gps_snapshot,
                        self.voice.active,
                        self.camera_setup,
                        wall_now - self.started,
                    )
                )

                # ------------------------------------------------------
                # Buzzer / recording / display
                # ------------------------------------------------------
                self.buzzer.update(state.alarm or safety.alarm, wall_now)
                if self.recorder is not None:
                    self.recorder.write(frame)
                    self.frames_written += 1

                if not self.headless:
                    cv2.imshow(self.config.window_name, frame)
                    if (
                        self.config.show_head_window
                        and head_img is not None
                        and self.head_renderer is not None
                    ):
                        shown = head_img
                        target = int(self.config.head_window_size)
                        if shown.shape[0] != target:
                            shown = cv2.resize(
                                shown,
                                (target, target),
                                interpolation=cv2.INTER_AREA,
                            )
                        cv2.imshow(self.config.head_window_name, shown)
                    key = cv2.waitKey(1) & 0xFF
                    if key != 255 and not self._handle_key(key):
                        break

                if (
                    self.config.max_frames > 0
                    and self.frame_index >= self.config.max_frames
                ):
                    break
        finally:
            self._shutdown()

        summary = self._build_summary()
        return summary

    # ------------------------------------------------------------------
    # Keyboard / feedback
    # ------------------------------------------------------------------
    def _handle_key(self, key: int) -> bool:
        if key in (ord("q"), 27):
            return False
        if key == ord("c"):
            if self.last_pose is not None:
                self.state_machine.calibrate(
                    self.last_pose.yaw,
                    self.last_pose.pitch,
                    self.last_pose.roll,
                )
                try:
                    self.state_machine.save_calibration(
                        self.config.calibration_file
                    )
                    self.toast = "CALIBRATED"
                except OSError as exc:
                    self.toast = f"CALIBRATION FAILED: {exc}"
                self.toast_until = time.monotonic() + 2.0
                print(
                    "[CAL] neutral pose saved: "
                    f"yaw={self.state_machine.yaw_offset:+.1f} "
                    f"pitch={self.state_machine.pitch_offset:+.1f} "
                    f"roll={self.state_machine.roll_offset:+.1f}"
                )
            else:
                self.toast = "CALIBRATION NEEDS A FACE"
                self.toast_until = time.monotonic() + 2.0
        elif key == ord("f"):
            if self.frame_setup is None:
                self.frame_setup = FrameSetup(self.config)
                self._setup_instruction = ""
                self.toast = "FRAMING SETUP - HOLD STILL"
            else:
                self.frame_setup = None
                self._setup_instruction = ""
                self.toast = "FRAMING SETUP CLOSED"
            self.toast_until = time.monotonic() + 2.0
        elif key == ord("v"):
            if not self.voice.available:
                self.toast = "VOICE UNAVAILABLE"
            else:
                self.voice.enabled = not self.voice.enabled
                self.toast = f"VOICE {'ON' if self.voice.enabled else 'OFF'}"
            self.toast_until = time.monotonic() + 1.5
        elif key == ord("r"):
            self.state_machine.reset()
            self.eye_analyzer.reset()
            self.smoother.reset()
            self.tracker.reset()
            self.last_pose = None
            self.toast = "SESSION STATE RESET"
            self.toast_until = time.monotonic() + 2.0
            print("[STATE] reset")
        elif key == ord("h"):
            self.hud_minimal = not self.hud_minimal
        elif key == ord("m"):
            self.mirror = not self.mirror
        elif key == ord("b"):
            enabled = self.buzzer.toggle()
            self.buzzer_on = enabled
            self.toast = f"BUZZER {'ON' if enabled else 'OFF'}"
            self.toast_until = time.monotonic() + 1.5
        elif key == ord("d"):
            self.debug = not self.debug
        elif key == ord("s"):
            self._save_snapshot()
        elif key in (ord("?"), ord("/")):
            self.help_visible = not self.help_visible
        return True

    def _save_snapshot(self) -> None:
        if self.last_annotated is None:
            self.toast = "SNAPSHOT UNAVAILABLE"
            self.toast_until = time.monotonic() + 2.0
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        target = self.config.event_dir / f"snapshot_{stamp}.jpg"
        self.config.event_dir.mkdir(parents=True, exist_ok=True)
        if cv2.imwrite(
            str(target),
            self.last_annotated,
            [int(cv2.IMWRITE_JPEG_QUALITY), 92],
        ):
            print(f"[SNAP] {target}")
            self.toast = f"SNAPSHOT {target.name}"
        else:
            self.toast = "SNAPSHOT FAILED"
        self.toast_until = time.monotonic() + 2.0

    @staticmethod
    def _draw_toast(frame: np.ndarray, text: str) -> None:
        h, w = frame.shape[:2]
        (tw, th), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        )
        x = (w - tw) // 2
        y = h - 76
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x - 12, y - th - 8),
            (x + tw + 12, y + 8),
            (20, 20, 22),
            -1,
        )
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.putText(
            frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )

    # ------------------------------------------------------------------
    # Summary / shutdown
    # ------------------------------------------------------------------
    def _build_summary(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "started": self.started_wall.isoformat(timespec="seconds"),
            "frames": self.frame_index,
            "frames_recorded": self.frames_written,
            "processing_fps": round(self.rate.fps, 2),
            "source": asdict(self.source.info),
            "record": str(self.recording_path) if self.recording_path else None,
            "state_frames": dict(self.state_counts),
            "safety_frames": dict(self.safety_counts),
            "identity": {
                "state": self.last_identity.state,
                "driver": self.last_identity.name,
                "score": None
                if self.last_identity.score is None
                else round(self.last_identity.score, 3),
                "enforced": self.last_identity.enforced,
            },
            "gps": self.gps.snapshot(time.monotonic()),
            "dashboard": self.dashboard.stats(),
            "max_offroad_seconds": round(self.max_offroad, 2),
            "max_closure_seconds": round(self.max_closure, 2),
            "max_no_face_seconds": round(self.max_no_face, 2),
            "min_perclos": None
            if self.min_perclos is None
            else round(self.min_perclos, 3),
            "max_perclos": None
            if self.max_perclos is None
            else round(self.max_perclos, 3),
            "max_blink_rate": round(self.max_blink_rate, 1),
            "blinks": self.eye_analyzer.blink_count,
            "long_blinks": self.eye_analyzer.long_blink_count,
            "microsleeps": self.eye_analyzer.microsleep_count,
            "events": dict(self.events.counts),
            "calibration": self.state_machine.calibration_dict(),
            "latency_ms": {
                name: {
                    "mean": round(stats.mean, 2),
                    "p95": round(stats.p95, 2),
                    "last": round(stats.last, 2),
                }
                for name, stats in self.timings.items()
            },
            "config": _config_snapshot(self.config),
        }

    def _shutdown(self) -> None:
        self.buzzer.off()
        self.frame_setup = None
        if self.recorder is not None:
            self.recorder.release()
            self.recorder = None
        summary: dict[str, Any] = {}
        try:
            summary = self._build_summary()
            self.events.close(summary)
        except Exception as exc:
            print(f"[SESSION] summary failed: {exc}")
        self.dashboard.end_session(
            {
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "ended_at": datetime.now().isoformat(timespec="seconds"),
                "summary": summary,
            }
        )
        self.gps.stop()
        self.voice.shutdown()
        self.source.release()
        if not self.headless:
            cv2.destroyAllWindows()

    def _print_banner(self) -> None:
        c = self.config
        print()
        print("==========================================================")
        print("        DRIVER MONITORING SYSTEM  v2")
        print("==========================================================")
        info = self.source.info
        print(f"Source       : {info.backend}  {info.description}")
        print(
            f"Resolution   : {info.width} x {info.height} "
            f"@ {info.fps:.0f} fps"
        )
        print(f"Face model   : {c.face_model}")
        print(f"Head pose    : {c.headpose_model}")
        print(f"Eye model    : {c.eye_model}")
        print(f"Identity     : {c.identity_model}")
        if self.identity_verifier is not None:
            print(
                "Driver       : "
                + (
                    ", ".join(c.authorized_drivers)
                    if c.authorized_drivers
                    else "identify-only"
                )
            )
        print(f"Event log    : {c.log_file}")
        print(f"Voice        : {'on' if self.voice.active else 'off'}")
        print(f"GPS          : {c.gps_source}")
        print(f"Dashboard    : {c.dashboard_url or 'off'}")
        print(f"Mode         : {'headless' if self.headless else 'windowed'}")
        if self.recording_path:
            print(f"Recording    : {self.recording_path}")
        print()
        print("Safety levels: GREEN attentive | YELLOW caution | RED serious")
        print("Rules:")
        print(f"  glance zone      : |yaw| >= {c.yaw_glance:.0f} or |pitch| >= {c.pitch_glance:.0f} deg")
        print(f"  distraction      : {c.distraction_seconds:.1f} s off-road")
        print(f"  critical         : {c.distraction_critical_seconds:.1f} s off-road")
        print(f"  drowsy closure   : {c.eye_closed_seconds:.1f} s")
        print(f"  microsleep       : {c.eye_closed_critical_seconds:.1f} s")
        print(f"  no face warning  : {c.no_face_seconds:.1f} s")
        print()
        print("Press Q to quit; ? for the full key list.")
        print()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config_snapshot(config: DMSConfig) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for key, value in asdict(config).items():
        if isinstance(value, Path):
            snapshot[key] = str(value)
        elif is_dataclass(value):
            snapshot[key] = str(value)
        else:
            snapshot[key] = value
    return snapshot


__all__ = ["BuzzerController", "DriverMonitorApp", "display_available"]