#!/usr/bin/env python3
"""Driver Monitoring System entry point (v2).

Examples
--------
    python driver_monitor.py
    python driver_monitor.py --phone
    python driver_monitor.py --video events/demo.mp4 --headless
    python driver_monitor.py --video http://192.168.1.50:4747/video
    python driver_monitor.py --headless --record events/session.mp4
    python driver_monitor.py --list-cameras
    python driver_monitor.py --no-buzzer --no-head-window

The old v1 application is preserved at ``docs/driver_monitor_v1.py``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# OpenCV's bundled Qt build ships only the XCB platform plugin.  On a Wayland
# desktop Qt otherwise tries to load a missing "wayland" plugin and prints
# noisy warnings (or fails to open the camera window).  Xwayland provides
# DISPLAY=:0 in the normal desktop session, so XCB is the reliable choice.
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault(
    "QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu"
)

from dms import __version__
from dms.app import DriverMonitorApp, display_available
from dms.camera import list_video_devices
from dms.config import DMSConfig
from dms.enroll import run_enrollment
from dms.identity import IdentityStore
from dms.phone import phone_stream_url, setup_phone_camera
from dms.setup import run_camera_setup
from dms.voice import voice_test


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="driver_monitor.py",
        description=(
            "Driver Monitoring System with YuNet face detection, "
            "6DRepNet head pose, OCEC eye state and a live 3D head."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python driver_monitor.py\n"
            "  python driver_monitor.py --phone\n"
            "  python driver_monitor.py --video clip.mp4 --headless\n"
            "  python driver_monitor.py --video http://PHONE_IP:4747/video\n"
            "  python driver_monitor.py --headless --record session.mp4\n"
            "  python driver_monitor.py --no-buzzer --debug\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"DMS {__version__}")

    source = parser.add_argument_group("camera / video source")
    source.add_argument(
        "--video",
        type=str,
        metavar="FILE|URL",
        help="video file or HTTP/RTSP stream (e.g. DroidCam)",
    )
    source.add_argument(
        "--phone",
        action="store_true",
        help=(
            "use an Android phone via DroidCam; show setup steps and "
            "prompt for IP/port"
        ),
    )
    source.add_argument(
        "--phone-ip",
        type=str,
        metavar="IP",
        help="DroidCam phone IP or hostname (with --phone)",
    )
    source.add_argument(
        "--phone-port",
        type=int,
        metavar="PORT",
        help="DroidCam port (with --phone; default 4747)",
    )
    source.add_argument(
        "--camera",
        choices=("auto", "picamera2", "v4l2"),
        default="auto",
        help="camera backend (default: auto)",
    )
    source.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="V4L2 device index (default: 0)",
    )
    source.add_argument("--width", type=int, default=640)
    source.add_argument("--height", type=int, default=480)
    source.add_argument("--fps", type=int, default=30)
    source.add_argument(
        "--loop-video", action="store_true", help="loop the video file"
    )
    source.add_argument(
        "--mirror", action="store_true", help="mirror the camera image"
    )

    models = parser.add_argument_group("models")
    models.add_argument(
        "--model-dir",
        type=Path,
        default=DMSConfig.model_dir,
        help="directory containing the ONNX models",
    )
    models.add_argument(
        "--headpose-model",
        type=str,
        default=DMSConfig.headpose_model,
        help="head-pose ONNX filename or absolute path",
    )
    models.add_argument(
        "--eye-model",
        type=str,
        default=DMSConfig.eye_model,
        help="eye-state ONNX filename or absolute path",
    )
    models.add_argument(
        "--headpose-every",
        type=int,
        default=DMSConfig.headpose_every,
        help="run head pose every N frames (default: 2)",
    )

    runtime = parser.add_argument_group("runtime / output")
    runtime.add_argument(
        "--headless",
        action="store_true",
        help="do not open OpenCV windows",
    )
    runtime.add_argument(
        "--record",
        type=str,
        metavar="FILE",
        help="write the annotated stream to FILE (.mp4 or .avi)",
    )
    runtime.add_argument(
        "--event-dir",
        type=Path,
        metavar="DIR",
        help="directory for event photos and the CSV log (default: events/)",
    )
    runtime.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="stop after N frames (0 = unlimited)",
    )
    runtime.add_argument(
        "--fast",
        action="store_true",
        help="fast mode for Pi 4: disable 3D head window and head inset",
    )
    runtime.add_argument(
        "--no-head-window", action="store_true", help="hide the separate 3D head window"
    )
    runtime.add_argument(
        "--no-inset", action="store_true", help="hide the 3D head inset in the HUD"
    )
    runtime.add_argument(
        "--no-buzzer", action="store_true", help="disable the GPIO buzzer"
    )
    runtime.add_argument(
        "--no-calibration",
        action="store_true",
        help="ignore the saved neutral-pose calibration",
    )
    runtime.add_argument("--debug", action="store_true", help="show debug overlay")
    runtime.add_argument(
        "--list-cameras",
        action="store_true",
        help="list /dev/video* devices and Picamera2 availability, then exit",
    )

    identity = parser.add_argument_group("driver identity")
    identity.add_argument(
        "--driver",
        action="append",
        metavar="NAME",
        help="enforce an enrolled driver (repeat for several authorized drivers)",
    )
    identity.add_argument(
        "--enroll-driver",
        metavar="NAME",
        help="capture a new face profile and exit",
    )
    identity.add_argument(
        "--enroll-force",
        action="store_true",
        help="replace an existing profile during enrolment",
    )
    identity.add_argument(
        "--enroll-samples",
        type=int,
        default=DMSConfig.enroll_samples,
        help="good frames to capture during enrolment (default: 30)",
    )
    identity.add_argument(
        "--enroll-timeout",
        type=float,
        default=DMSConfig.enroll_timeout,
        help="enrolment timeout in seconds (default: 60)",
    )
    identity.add_argument(
        "--identity-store",
        type=Path,
        default=DMSConfig.identity_store,
        help="face profile JSON path",
    )
    identity.add_argument(
        "--identity-threshold",
        type=float,
        default=DMSConfig.identity_threshold,
        help="SFace cosine threshold (default: 0.363)",
    )
    identity.add_argument(
        "--identity-every",
        type=int,
        default=DMSConfig.identity_every,
        help="run identity checks every N frames (default: 5)",
    )
    identity.add_argument(
        "--no-identity",
        action="store_true",
        help="disable face identity verification",
    )
    identity.add_argument(
        "--list-drivers",
        action="store_true",
        help="list enrolled drivers and exit",
    )

    setup = parser.add_argument_group("camera setup / calibration")
    setup.add_argument(
        "--setup-camera",
        action="store_true",
        help="run the guided framing setup, then start monitoring",
    )
    setup.add_argument(
        "--setup-only",
        action="store_true",
        help="run the guided framing setup and exit",
    )
    setup.add_argument(
        "--auto-calibrate",
        action="store_true",
        help="run the framing guide on start and calibrate automatically",
    )
    setup.add_argument(
        "--setup-timeout",
        type=float,
        default=DMSConfig.setup_timeout,
        help="setup timeout in seconds (default: 30)",
    )

    voice = parser.add_argument_group("voice warnings")
    voice.add_argument(
        "--voice", action="store_true", help="enable spoken warnings (espeak-ng)"
    )
    voice.add_argument(
        "--voice-test",
        action="store_true",
        help="speak a few sample warnings and exit",
    )
    voice.add_argument(
        "--voice-command",
        default=DMSConfig.voice_command,
        help="speech command (default: espeak-ng)",
    )
    voice.add_argument(
        "--voice-rate",
        type=int,
        default=DMSConfig.voice_rate,
        help="speech rate in words per minute (default: 150)",
    )

    gps = parser.add_argument_group("GPS")
    gps.add_argument(
        "--gps",
        choices=("off", "gpsd"),
        default=DMSConfig.gps_source,
        help="read position from a local gpsd (default: off)",
    )
    gps.add_argument("--gps-host", default=DMSConfig.gps_host)
    gps.add_argument("--gps-port", type=int, default=DMSConfig.gps_port)

    dashboard = parser.add_argument_group("laptop dashboard")
    dashboard.add_argument(
        "--dashboard-url",
        metavar="URL",
        help="dashboard base URL, e.g. http://192.168.1.20:8080",
    )
    dashboard.add_argument(
        "--dashboard-interval",
        type=float,
        default=DMSConfig.dashboard_interval,
        help="status push interval in seconds (default: 1.0)",
    )
    dashboard.add_argument(
        "--dashboard-timeout",
        type=float,
        default=DMSConfig.dashboard_timeout,
        help="HTTP timeout in seconds (default: 5)",
    )
    return parser


def list_cameras() -> int:
    devices = list_video_devices()
    print("V4L2 devices:")
    if devices:
        for device in devices:
            print(f"  {device}")
    else:
        print("  none")
    try:
        from picamera2 import Picamera2  # noqa: F401

        print("Picamera2: available")
    except Exception as exc:
        print(f"Picamera2: not available ({exc})")
    print(f"Display: {'available' if display_available() else 'not detected'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_cameras:
        return list_cameras()
    if args.phone and args.video:
        parser.error("--phone and --video cannot be used together")
    if not args.phone and (args.phone_ip or args.phone_port is not None):
        parser.error("--phone-ip and --phone-port require --phone")

    config = DMSConfig()
    if args.phone:
        try:
            settings = setup_phone_camera(args.phone_ip, args.phone_port)
        except (RuntimeError, ValueError) as exc:
            print(f"\n[ERROR] {exc}", file=sys.stderr)
            return 4
        config.video = phone_stream_url(settings.ip, settings.port)
    else:
        config.video = args.video
    config.camera = args.camera
    config.camera_index = args.camera_index
    config.width = args.width
    config.height = args.height
    config.fps = args.fps
    config.loop_video = args.loop_video
    config.mirror = args.mirror
    config.model_dir = args.model_dir
    config.headpose_model = args.headpose_model
    config.eye_model = args.eye_model
    config.headpose_every = args.headpose_every
    config.record_path = args.record
    if args.event_dir is not None:
        config.event_dir = args.event_dir
        config.log_file = args.event_dir / "dms_log.csv"
    config.max_frames = max(0, args.max_frames)
    config.show_windows = not args.headless
    config.show_head_window = not args.no_head_window
    config.show_head_inset = not args.no_inset
    config.fast_mode = args.fast
    if args.fast:
        config.show_head_window = False
        config.show_head_inset = False
        if not display_available():
            config.show_windows = False
    config.buzzer_enabled = not args.no_buzzer
    config.use_calibration = not args.no_calibration
    config.debug = args.debug

    # Calibration check
    if config.use_calibration and not config.calibration_file.exists():
        if not config.auto_calibrate:
            print("NO CALIBRATION - run --setup-camera first")
        print("[SETUP] Forcing --setup-camera because calibration.json is missing.")
        config.setup_camera = True

    # Identity.
    config.identity_store = args.identity_store
    config.identity_threshold = args.identity_threshold
    config.identity_every = args.identity_every
    config.identity_enabled = not args.no_identity
    config.enroll_driver = args.enroll_driver
    config.enroll_force = args.enroll_force
    config.enroll_samples = max(1, args.enroll_samples)
    config.enroll_timeout = max(1.0, args.enroll_timeout)
    if args.driver:
        config.authorized_drivers = list(dict.fromkeys(args.driver))

    # Camera setup.
    config.setup_camera = bool(args.setup_camera or args.setup_only)
    config.auto_calibrate = args.auto_calibrate
    config.setup_timeout = max(1.0, args.setup_timeout)

    # Voice / GPS / dashboard.
    config.voice_enabled = bool(args.voice or args.voice_test)
    config.voice_command = args.voice_command
    config.voice_rate = args.voice_rate
    config.gps_source = args.gps
    config.gps_host = args.gps_host
    config.gps_port = args.gps_port
    config.dashboard_url = args.dashboard_url
    config.dashboard_interval = max(0.1, args.dashboard_interval)
    config.dashboard_timeout = max(0.5, args.dashboard_timeout)

    try:
        config.validate()
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    headless = bool(args.headless or not display_available())

    if args.list_drivers:
        store = IdentityStore(config.identity_store, model=config.identity_model)
        print(f"Identity store: {store.path}")
        if not store.profiles:
            print("  no enrolled drivers")
        for profile in store.profiles:
            print(
                f"  {profile.name:24s} samples={profile.samples:3d} "
                f"created={profile.created or 'unknown'}"
            )
        return 0

    if args.voice_test:
        return voice_test(config)

    if args.enroll_driver:
        ok = run_enrollment(
            config, headless=headless, timeout=args.enroll_timeout
        )
        return 0 if ok else 1

    if args.setup_camera or args.setup_only:
        ok = run_camera_setup(
            config, headless=headless, timeout=args.setup_timeout
        )
        if not ok:
            return 5
        if args.setup_only:
            return 0

    if config.authorized_drivers:
        store = IdentityStore(config.identity_store, model=config.identity_model)
        missing = [
            name
            for name in config.authorized_drivers
            if store.get(name) is None
        ]
        if missing:
            print("IDENTITY OFF - no enrolled driver", file=sys.stderr)
            print(
                "[ERROR] not enrolled: " + ", ".join(missing),
                file=sys.stderr,
            )
            print(
                "Enrol first with:\n"
                f"    python driver_monitor.py --enroll-driver {missing[0]}",
                file=sys.stderr,
            )
            return 7

    try:
        app = DriverMonitorApp(
            config,
            headless=args.headless,
            show_windows=not args.headless,
        )
        summary = app.run()
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        print(
            "Download the models with:\n"
            "    python tools/download_models.py",
            file=sys.stderr,
        )
        return 2
    except RuntimeError as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("\n[STOP] interrupted")
        return 130

    print()
    print("Session summary")
    print("---------------")
    print(f"  frames processed : {summary.get('frames', 0)}")
    print(f"  processing fps   : {summary.get('processing_fps', 0):.1f}")
    print(f"  events           : {summary.get('events', {})}")
    print(f"  state frames     : {summary.get('state_frames', {})}")
    print(f"  safety frames    : {summary.get('safety_frames', {})}")
    identity = summary.get("identity") or {}
    if identity.get("state") and identity.get("state") != "OFF":
        print(
            f"  identity         : {identity.get('state')} "
            f"{(identity.get('driver') or '')} "
            f"{identity.get('score') if identity.get('score') is not None else ''}"
        )
    gps = summary.get("gps") or {}
    print(f"  gps              : {gps.get('text', 'GPS: off')}")
    dashboard = summary.get("dashboard") or {}
    if dashboard.get("enabled"):
        print(
            f"  dashboard        : sent={dashboard.get('sent', 0)} "
            f"failed={dashboard.get('failed', 0)}"
        )
    if summary.get("max_offroad_seconds"):
        print(f"  longest off-road : {summary['max_offroad_seconds']:.1f} s")
    if summary.get("max_closure_seconds"):
        print(f"  longest closure  : {summary['max_closure_seconds']:.2f} s")
    print("Stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())