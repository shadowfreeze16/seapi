"""Configuration for the driver monitoring system.

All values are intentionally plain data so they can be overridden from the
command line (see ``driver_monitor.py``) and from unit tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class DMSConfig:
    # ------------------------------------------------------------------
    # Camera / video source
    # ------------------------------------------------------------------
    width: int = 640
    height: int = 480
    fps: int = 30
    camera: str = "auto"  # auto | picamera2 | v4l2
    camera_index: int = _env_int("DROIDCAM_DEV", 0)
    video: str | None = None
    loop_video: bool = False
    mirror: bool = False

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    model_dir: Path = PROJECT_ROOT / "models"
    face_model: str = "face_detection_yunet_2026may.onnx"
    headpose_model: str = "headpose_mobilenetv3_small.onnx"
    eye_model: str = "eyes_ocec_s.onnx"
    identity_model: str = "face_recognition_sface_2021dec.onnx"

    # YuNet.  Detecting at half resolution is ~4x faster on a Pi 4; the
    # returned boxes and landmarks are scaled back to the camera frame.
    face_confidence: float = 0.60
    face_nms: float = 0.30
    face_top_k: float = 5000
    face_input_scale: float = 0.50

    # Head pose inference scheduling.  The model takes ~20-25 ms on a Pi 4
    # (MobileNetV3-small), so running it every second frame is a good tradeoff.
    headpose_every: int = 2
    headpose_expand: float = 0.30
    headpose_threads: int = 3
    eye_threads: int = 3
    identity_threads: int = 2

    # ------------------------------------------------------------------
    # Face tracking
    # ------------------------------------------------------------------
    track_smoothing: float = 0.55
    track_max_missing: int = 3
    min_face_px: int = 48

    # ------------------------------------------------------------------
    # Attention thresholds (degrees)
    # ------------------------------------------------------------------
    yaw_glance: float = 12.0
    yaw_extreme: float = 55.0
    pitch_glance: float = 12.0
    pitch_extreme: float = 35.0

    # Time thresholds (seconds)
    offroad_debounce: float = 0.20
    forward_debounce: float = 0.40
    distraction_seconds: float = 2.50
    distraction_critical_seconds: float = 5.00
    extreme_seconds: float = 2.00
    no_face_seconds: float = 2.50
    no_face_critical_seconds: float = 6.00

    # ------------------------------------------------------------------
    # Eye / drowsiness thresholds
    # ------------------------------------------------------------------
    eye_closed_enter: float = 0.38
    eye_closed_exit: float = 0.60
    eye_crop_scale: float = 0.34
    eye_crop_aspect: float = 0.62
    max_eye_yaw: float = 60.0
    min_eye_crop_px: int = 10
    eye_closed_seconds: float = 1.20
    eye_closed_critical_seconds: float = 3.00
    perclos_window: float = 60.0
    perclos_threshold: float = 0.35
    perclos_min_time: float = 20.0
    blink_max_seconds: float = 0.50
    blink_rate_window: float = 60.0

    # Head-pose smoothing (One Euro filter)
    smooth_min_cutoff: float = 1.4
    smooth_beta: float = 0.015
    smooth_d_cutoff: float = 1.0

    # ------------------------------------------------------------------
    # Events / logging
    # ------------------------------------------------------------------
    event_dir: Path = PROJECT_ROOT / "events"
    log_file: Path = PROJECT_ROOT / "events" / "dms_log.csv"
    calibration_file: Path = PROJECT_ROOT / "calibration.json"
    use_calibration: bool = True
    event_cooldown: float = 5.0
    critical_repeat: float = 4.0
    max_events: int = 1000
    max_event_mb: float = 250.0
    save_session_summary: bool = True

    # ------------------------------------------------------------------
    # Buzzer
    # ------------------------------------------------------------------
    buzzer_gpio: int = 18
    buzzer_enabled: bool = True

    # ------------------------------------------------------------------
    # Three-level safety status
    # ------------------------------------------------------------------
    safety_red_after: float = 8.0
    safety_red_hold: float = 1.5
    safety_green_hold: float = 2.0

    # ------------------------------------------------------------------
    # Driver identity (SFace embedding)
    # ------------------------------------------------------------------
    identity_store: Path = PROJECT_ROOT / "driver_identity.json"
    identity_threshold: float = 0.363
    identity_every: int = 5
    identity_min_face_px: int = 64
    identity_confirm_checks: int = 3
    identity_enabled: bool = True
    # Names supplied with --driver.  An empty list means identify-only mode.
    authorized_drivers: list[str] = field(default_factory=list)
    enroll_driver: str | None = None
    enroll_force: bool = False
    enroll_samples: int = 30
    enroll_timeout: float = 60.0

    # ------------------------------------------------------------------
    # Camera setup / automatic neutral-pose calibration
    # ------------------------------------------------------------------
    setup_camera: bool = False
    auto_calibrate: bool = False
    setup_timeout: float = 30.0
    setup_settle_seconds: float = 2.0
    setup_min_face_px: int = 80
    setup_face_ratio_min: float = 0.16
    setup_face_ratio_max: float = 0.62
    setup_center_tolerance: float = 0.20
    setup_max_yaw: float = 15.0
    setup_max_pitch: float = 15.0
    setup_max_roll: float = 12.0
    setup_min_brightness: float = 45.0
    setup_max_brightness: float = 215.0
    setup_min_sharpness: float = 25.0

    # ------------------------------------------------------------------
    # Voice warnings (espeak-ng)
    # ------------------------------------------------------------------
    voice_enabled: bool = False
    voice_command: str = "espeak-ng"
    voice_lang: str = "en"
    voice_rate: int = 150
    voice_volume: int = 180
    voice_cooldown: float = 10.0

    # ------------------------------------------------------------------
    # GPS (gpsd)
    # ------------------------------------------------------------------
    gps_source: str = "off"  # off | gpsd
    gps_host: str = "127.0.0.1"
    gps_port: int = 2947
    gps_stale_seconds: float = 5.0
    gps_reconnect_seconds: float = 2.0

    # ------------------------------------------------------------------
    # Laptop dashboard client
    # ------------------------------------------------------------------
    dashboard_url: str | None = None
    dashboard_interval: float = 1.0
    dashboard_timeout: float = 5.0
    dashboard_queue_size: int = 200
    dashboard_snapshot_quality: int = 80
    dashboard_snapshot_max_width: int = 960
    dashboard_outbox: Path = PROJECT_ROOT / "events" / "dashboard_outbox.jsonl"

    # ------------------------------------------------------------------
    # UI / runtime
    # ------------------------------------------------------------------
    window_name: str = "Driver Monitoring System"
    head_window_name: str = "DMS 3D Head Pose"
    show_windows: bool = True
    show_head_window: bool = True
    show_head_inset: bool = True
    head_canvas_size: int = 420
    head_window_size: int = 420
    inset_size: int = 168
    debug: bool = False
    record_path: str | None = None
    max_frames: int = 0

    # Derived / runtime fields
    start_time: float = field(default_factory=lambda: 0.0, init=False)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    @property
    def face_model_path(self) -> Path:
        return self.model_dir / self.face_model

    @property
    def headpose_model_path(self) -> Path:
        p = Path(self.headpose_model)
        return p if p.is_absolute() else self.model_dir / p

    @property
    def eye_model_path(self) -> Path:
        p = Path(self.eye_model)
        return p if p.is_absolute() else self.model_dir / p

    @property
    def identity_model_path(self) -> Path:
        p = Path(self.identity_model)
        return p if p.is_absolute() else self.model_dir / p

    def validate(self) -> None:
        if self.width < 160 or self.height < 120:
            raise ValueError("Camera resolution is too small.")
        if self.distraction_seconds >= self.distraction_critical_seconds:
            raise ValueError("distraction_seconds must be < distraction_critical_seconds")
        if self.eye_closed_seconds >= self.eye_closed_critical_seconds:
            raise ValueError("eye_closed_seconds must be < eye_closed_critical_seconds")
        if not 0.0 < self.eye_crop_scale < 1.0:
            raise ValueError("eye_crop_scale must be in (0, 1)")
        if not 0.1 <= self.face_input_scale <= 1.0:
            raise ValueError("face_input_scale must be in [0.1, 1.0]")
        if self.headpose_every < 1:
            raise ValueError("headpose_every must be >= 1")
        if self.identity_every < 1:
            raise ValueError("identity_every must be >= 1")
        if self.identity_confirm_checks < 1:
            raise ValueError("identity_confirm_checks must be >= 1")
        if not -1.0 <= self.identity_threshold <= 1.0:
            raise ValueError("identity_threshold must be in [-1, 1]")
        if self.safety_red_after <= 0.0:
            raise ValueError("safety_red_after must be > 0")
        if self.safety_red_hold < 0.0 or self.safety_green_hold < 0.0:
            raise ValueError("safety hold times must be >= 0")
        if self.gps_source not in ("off", "gpsd"):
            raise ValueError("gps_source must be 'off' or 'gpsd'")
        if not 0 <= self.gps_port <= 65535:
            raise ValueError("gps_port must be in [0, 65535]")
        if self.dashboard_url:
            if not self.dashboard_url.startswith(("http://", "https://")):
                raise ValueError("dashboard_url must start with http:// or https://")
        if self.dashboard_interval <= 0.0:
            raise ValueError("dashboard_interval must be > 0")