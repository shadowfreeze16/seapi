#!/usr/bin/env python3
"""End-to-end self test for the driver monitoring system.

    python tools/self_test.py
    python tools/self_test.py --image events/driver.jpg
    python tools/self_test.py --skip-inference

The test verifies:

* the Python environment and every package import,
* the pinned SHA-256 of the required ONNX models,
* one real inference of YuNet, 6DRepNet and OCEC on an event photo,
* the 3D head renderer (shape, content, rotation),
* the eye analyser, state machine and event logger,
* the command-line entry point.

It never opens a camera and never touches the real event log: the logger test
runs in a temporary directory.  The exit code is 0 only if every required check
passes.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, TOOLS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from download_models import EXTRA, MODELS, verify  # noqa: E402

from dms.config import DMSConfig  # noqa: E402
from dms.events import EventLogger, LOG_FIELDS  # noqa: E402
from dms.eyes import EyeAnalyzer, EyeMetrics  # noqa: E402
from dms.head3d import HeadRenderer  # noqa: E402
from dms.models import (  # noqa: E402
    EyeObservation,
    EyeStateEstimator,
    FaceDetector,
    HeadPoseEstimator,
)
from dms.state import (  # noqa: E402
    CRITICAL,
    DISTRACTED,
    DROWSY,
    NORMAL,
    NO_FACE,
    EventRequest,
    StateMachine,
)

_PASS = "PASS"
_FAIL = "FAIL"
_SKIP = "SKIP"


class Report:
    def __init__(self) -> None:
        self.checks: list[tuple[str, str, str]] = []

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.checks.append((name, status, detail))
        marker = {_PASS: "[PASS]", _FAIL: "[FAIL]", _SKIP: "[SKIP]"}[status]
        print(f"  {marker} {name}" + (f": {detail}" if detail else ""))

    def ok(self, name: str, detail: str = "") -> None:
        self.add(name, _PASS, detail)

    def fail(self, name: str, detail: str = "") -> None:
        self.add(name, _FAIL, detail)

    def skip(self, name: str, detail: str = "") -> None:
        self.add(name, _SKIP, detail)

    @property
    def failures(self) -> int:
        return sum(1 for _, status, _ in self.checks if status == _FAIL)


def section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def check_environment(report: Report) -> None:
    section("Environment")
    import onnxruntime as ort

    report.ok(
        "Python",
        f"{sys.version.split()[0]} ({sys.implementation.name})",
    )
    report.ok("OpenCV", cv2.__version__)
    report.ok("NumPy", np.__version__)
    report.ok("ONNX Runtime", ort.__version__)
    try:
        import dms.app  # noqa: F401
        import dms.camera  # noqa: F401
        import dms.events  # noqa: F401
        import dms.eyes  # noqa: F401
        import dms.filters  # noqa: F401
        import dms.geometry  # noqa: F401
        import dms.head3d  # noqa: F401
        import dms.models  # noqa: F401
        import dms.render  # noqa: F401
        import dms.state  # noqa: F401

        report.ok("dms package imports")
    except Exception as exc:  # pragma: no cover - only on a broken install
        report.fail("dms package imports", str(exc))


def check_models(report: Report, model_dir: Path) -> bool:
    section("Models")
    all_ok = True
    for name, spec in MODELS.items():
        ok, message = verify(model_dir / name, spec)
        if ok:
            size = (model_dir / name).stat().st_size / 1e6
            report.ok(name, f"{size:.2f} MB, sha256 verified")
        else:
            report.fail(name, message)
            all_ok = False

    for name, spec in EXTRA.items():
        path = model_dir / name
        if not path.exists():
            continue
        ok, message = verify(path, spec)
        if ok:
            report.ok(f"{name} (optional)", "sha256 verified")
        else:
            report.skip(
                f"{name} (optional)", f"present but {message}"
            )
    return all_ok


def find_face_image(
    detector: FaceDetector, explicit: Path | None
) -> tuple[np.ndarray, object, Path] | None:
    candidates: list[Path]
    if explicit is not None:
        candidates = [explicit]
    else:
        candidates = sorted((PROJECT_ROOT / "events").glob("*.jpg"))
    for path in candidates:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        faces = detector.detect(frame)
        if faces:
            return frame, max(faces, key=lambda f: f.area), path
    return None


def check_inference(
    report: Report, config: DMSConfig, image: Path | None
) -> None:
    section("Model inference")
    try:
        detector = FaceDetector(
            config.face_model_path,
            config.width,
            config.height,
            confidence=config.face_confidence,
            nms=config.face_nms,
            top_k=config.face_top_k,
            input_scale=config.face_input_scale,
        )
        found = find_face_image(detector, image)
        if found is None:
            detail = (
                f"no face in {image}"
                if image is not None
                else "no face found in events/*.jpg"
            )
            report.skip("face detection", detail)
            return
        frame, face, used = found
        report.ok(
            "face detection",
            f"{face.width:.0f}x{face.height:.0f} px in {used.name}",
        )
    except Exception as exc:
        report.fail("face detection", str(exc))
        return

    try:
        pose_estimator = HeadPoseEstimator(
            config.headpose_model_path,
            threads=config.headpose_threads,
            expand=config.headpose_expand,
        )
        pose = pose_estimator.estimate(frame, face)
        if pose is None:
            raise RuntimeError("model returned no pose")
        values = (pose.yaw, pose.pitch, pose.roll)
        if not all(np.isfinite(v) for v in values):
            raise RuntimeError(f"non-finite angles: {values}")
        report.ok(
            "head pose",
            f"yaw={pose.yaw:+.1f} pitch={pose.pitch:+.1f} "
            f"roll={pose.roll:+.1f} ({pose.latency_ms:.1f} ms)",
        )
    except Exception as exc:
        report.fail("head pose", str(exc))
        pose = None

    try:
        eye_estimator = EyeStateEstimator(
            config.eye_model_path,
            threads=config.eye_threads,
            crop_scale=config.eye_crop_scale,
            crop_aspect=config.eye_crop_aspect,
            min_crop_px=config.min_eye_crop_px,
        )
        observation = eye_estimator.estimate(
            frame,
            face,
            yaw=pose.yaw if pose is not None else None,
            max_yaw=config.max_eye_yaw,
        )
        if not observation.valid:
            raise RuntimeError("classifier did not return two eye probabilities")
        if not (
            0.0 <= observation.left_open <= 1.0
            and 0.0 <= observation.right_open <= 1.0
        ):
            raise RuntimeError("probabilities outside [0, 1]")
        report.ok(
            "eye state",
            f"open={observation.left_open:.2f}/{observation.right_open:.2f} "
            f"({observation.latency_ms:.1f} ms)",
        )
    except Exception as exc:
        report.fail("eye state", str(exc))


def check_head(report: Report) -> None:
    section("3D head renderer")
    try:
        renderer = HeadRenderer(320)
        front = renderer.render(0.0, 0.0, 0.0)
        turned = renderer.render(25.0, 10.0, 8.0)
        if front.shape != (320, 320, 3) or front.dtype != np.uint8:
            raise RuntimeError(f"unexpected output {front.shape} {front.dtype}")
        if float(front.std()) < 5.0:
            raise RuntimeError("render looks blank")
        difference = float(
            cv2.absdiff(front, turned).mean()
        )
        if difference < 1.0:
            raise RuntimeError("rotation did not change the render")
        report.ok(
            "render + rotation",
            f"{front.shape[1]}x{front.shape[0]}, diff={difference:.1f}",
        )
    except Exception as exc:
        report.fail("render + rotation", str(exc))


def _feed_state(
    machine: StateMachine,
    start: float,
    end: float,
    step: float,
    *,
    yaw: float = 0.0,
    pitch: float = 0.0,
    face: bool = True,
    eyes: EyeMetrics | None = None,
    pose_valid: bool = True,
    events: list[EventRequest] | None = None,
) -> object:
    state = None
    now = start
    observations = eyes or EyeMetrics(valid=True)
    while now <= end + 1e-9:
        state = machine.update(
            now,
            yaw,
            pitch,
            0.0,
            observations,
            face,
            pose_valid=pose_valid,
        )
        if events is not None:
            events.extend(state.events)
        now += step
    return state


def check_state(report: Report, config: DMSConfig) -> None:
    section("State machine")
    try:
        machine = StateMachine(config)
        state = _feed_state(machine, 0.0, 1.0, 0.1)
        if state is None or state.status != NORMAL:
            raise RuntimeError(f"expected NORMAL, got {state.status if state else None}")
        report.ok("forward attention", state.status)

        machine.reset()
        collected: list[EventRequest] = []
        state = _feed_state(
            machine, 0.0, 3.5, 0.1, yaw=40.0, events=collected
        )
        kinds = {event.kind for event in collected}
        if state.status != DISTRACTED:
            raise RuntimeError(
                f"expected DISTRACTED, got {state.status} ({state.reason})"
            )
        if "SIDE_ATTENTION" not in kinds:
            raise RuntimeError("no SIDE_ATTENTION event request")
        report.ok(
            "sustained off-road",
            f"{state.status} after {state.offroad_seconds:.1f} s",
        )

        machine.reset()
        drowsy = EyeMetrics(
            valid=True,
            left_closed=True,
            right_closed=True,
            both_closed=True,
            closure_seconds=1.5,
            drowsy=True,
        )
        state = _feed_state(machine, 0.0, 0.2, 0.1, eyes=drowsy)
        if state is None or state.status != DROWSY or not state.alarm:
            raise RuntimeError(
                f"expected DROWSY alarm, got {state.status if state else None}"
            )
        report.ok("drowsiness", state.status)

        machine.reset()
        critical = EyeMetrics(
            valid=True,
            left_closed=True,
            right_closed=True,
            both_closed=True,
            closure_seconds=3.5,
            drowsy=True,
            critical=True,
        )
        state = _feed_state(machine, 0.0, 0.2, 0.1, eyes=critical)
        if state is None or state.status != CRITICAL:
            raise RuntimeError(
                f"expected CRITICAL, got {state.status if state else None}"
            )
        report.ok("microsleep", state.status)

        machine.reset()
        collected = []
        state = _feed_state(
            machine, 0.0, 2.6, 0.1, face=False, events=collected
        )
        if state is None or state.status != NO_FACE:
            raise RuntimeError(
                f"expected NO_FACE, got {state.status if state else None}"
            )
        if "NO_FACE" not in {event.kind for event in collected}:
            raise RuntimeError("no NO_FACE event request")
        state = _feed_state(machine, 2.7, 6.2, 0.1, face=False)
        if state is None or state.status != CRITICAL:
            raise RuntimeError(
                f"expected CRITICAL after face loss, got {state.status if state else None}"
            )
        report.ok("face loss", f"{state.status} at {state.no_face_seconds:.1f} s")
    except Exception as exc:
        report.fail("state machine", str(exc))


def check_eyes(report: Report, config: DMSConfig) -> None:
    section("Eye analyser")
    try:
        analyzer = EyeAnalyzer(config)
        closed = EyeObservation(0.0, 0.0, True, [])
        opened = EyeObservation(1.0, 1.0, True, [])
        analyzer.update(closed, 0.0)
        analyzer.update(closed, 0.1)
        analyzer.update(closed, 0.2)
        analyzer.update(closed, 0.3)
        metrics = analyzer.update(opened, 0.4)
        metrics = analyzer.update(opened, 0.5)
        if metrics.blink_count != 1:
            raise RuntimeError(f"expected 1 blink, got {metrics.blink_count}")
        report.ok("blink detection", f"blink_count={metrics.blink_count}")

        analyzer.reset()
        analyzer.update(closed, 0.0)
        analyzer.update(closed, 0.1)
        analyzer.update(closed, 1.5)
        metrics = analyzer.update(opened, 1.6)
        metrics = analyzer.update(opened, 1.7)
        if metrics.long_blink_count != 1:
            raise RuntimeError(
                f"expected 1 long blink, got {metrics.long_blink_count}"
            )
        report.ok("long blink", f"long_blink_count={metrics.long_blink_count}")

        analyzer.reset()
        metrics = analyzer.update(None, 0.0)
        if metrics.valid:
            raise RuntimeError("invalid observation reported as valid")
        report.ok("dropout handling", "invalid observation ignored")
    except Exception as exc:
        report.fail("eye analyser", str(exc))


def check_events(report: Report, config: DMSConfig) -> None:
    section("Event logger")
    try:
        with tempfile.TemporaryDirectory(prefix="dms-selftest-") as tmp:
            config.event_dir = Path(tmp) / "events"
            config.log_file = config.event_dir / "dms_log.csv"
            logger = EventLogger(config)
            frame = np.full((48, 64, 3), 90, dtype=np.uint8)
            request = EventRequest(
                "SIDE_ATTENTION",
                DISTRACTED,
                "warning",
                "LEFT",
                "DISTRACTION WARNING",
            )
            row = logger.record(request, frame, context={"yaw": 30.0}, now=100.0)
            if row is None:
                raise RuntimeError("first event was not recorded")
            duplicate = logger.record(
                request, frame, context={"yaw": 30.0}, now=101.0
            )
            if duplicate is not None:
                raise RuntimeError("event cooldown did not suppress duplicate")

            second = EventRequest(
                "MICROSLEEP", CRITICAL, "critical", "FORWARD", "MICROSLEEP"
            )
            if logger.record(second, frame, now=103.0) is None:
                raise RuntimeError("critical event was not recorded")

            with config.log_file.open("r", encoding="utf-8") as handle:
                lines = handle.read().splitlines()
            if lines[0].split(",") != LOG_FIELDS:
                raise RuntimeError("unexpected CSV header")
            if len(lines) != 3:
                raise RuntimeError(f"expected 2 event rows, got {len(lines) - 1}")
            summary = logger.close({"frames": 10})
            if summary is None or not summary.exists():
                raise RuntimeError("session summary was not written")
            report.ok(
                "CSV + photo + session summary",
                f"{len(logger.photos)} photo(s), 2 rows",
            )
    except Exception as exc:
        report.fail("event logger", str(exc))


def check_cli(report: Report) -> None:
    section("Command line")
    entry = PROJECT_ROOT / "driver_monitor.py"
    try:
        result = subprocess.run(
            [sys.executable, str(entry), "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "non-zero exit code")
        report.ok("driver_monitor.py --version", result.stdout.strip())
    except Exception as exc:
        report.fail("driver_monitor.py --version", str(exc))


def check_camera_hint(report: Report) -> None:
    section("Camera (informational)")
    devices = sorted(Path("/dev").glob("video*"))
    report.add(
        "V4L2 devices",
        _PASS,
        ", ".join(f"/dev/{p.name}" for p in devices) if devices else "none",
    )
    try:
        from picamera2 import Picamera2  # noqa: F401

        report.ok("Picamera2 import", "available")
    except Exception as exc:
        report.skip("Picamera2 import", str(exc))
    display_available = bool(
        sys.platform in ("darwin", "win32")
        or os.environ.get("DISPLAY")
        or os.environ.get("WAYLAND_DISPLAY")
    )
    report.add(
        "Display",
        _PASS if display_available else _SKIP,
        "available" if display_available else "not detected (use --headless)",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", type=Path, default=DMSConfig.model_dir
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="face photo for the inference test (default: scan events/)",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="skip loading and running the neural networks",
    )
    args = parser.parse_args(argv)

    print("=" * 58)
    print("  DRIVER MONITORING SYSTEM - SELF TEST")
    print("=" * 58)
    print(f"Project: {PROJECT_ROOT}")
    print(f"Models : {args.model_dir}")

    report = Report()
    check_environment(report)
    models_ok = check_models(report, args.model_dir)

    config = DMSConfig()
    config.model_dir = args.model_dir

    if models_ok and not args.skip_inference:
        check_inference(report, config, args.image)
    elif not args.skip_inference:
        report.skip("model inference", "required model files are missing")

    check_head(report)
    check_state(report, config)
    check_eyes(report, config)
    check_events(report, config)
    check_cli(report)
    check_camera_hint(report)

    print()
    print("=" * 58)
    total = len(report.checks)
    failed = report.failures
    skipped = sum(1 for _, status, _ in report.checks if status == _SKIP)
    if failed:
        print(f"  RESULT: FAIL - {failed} of {total} checks failed")
    else:
        print(
            f"  RESULT: PASS - {total - skipped} checks"
            + (f", {skipped} skipped" if skipped else "")
        )
    print("=" * 58)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())