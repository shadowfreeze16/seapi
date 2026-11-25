#!/usr/bin/env python3
"""Laptop Camera Test Harness for Driver Monitoring System.

Runs the Driver Monitoring System core pipeline directly on a laptop using
its built-in webcam (cv2.VideoCapture(0)) and the pretrained ONNX models from
driver_monitor/models/.

Connects to the Fleet Dashboard (POST /api/status, POST /api/event) with
Vehicle ID VEH-LAPTOP-TEST.

Controls:
  Q or ESC : Quit
  C        : Force neutral pose calibration check
"""

from __future__ import annotations

import argparse
import base64
import datetime
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

# Ensure project root is available for driver_monitor imports
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
DRIVER_MONITOR_DIR = PROJECT_ROOT / "driver_monitor"

if str(DRIVER_MONITOR_DIR) not in sys.path:
    sys.path.insert(0, str(DRIVER_MONITOR_DIR))

# Verification of Pi-only hardware exclusion
PI_HARDWARE_MODULES = ["camera", "head3d", "gps", "voice", "picamera2", "gpiozero"]
for mod in PI_HARDWARE_MODULES:
    if f"dms.{mod}" in sys.modules or mod in sys.modules:
        print(f"[WARN] Pi-only hardware module '{mod}' detected in sys.modules; ignoring.")

# Import strictly the required core algorithmic modules from DMS
try:
    from dms.eyes import EyeAnalyzer, EyeMetrics
    from dms.state import StateMachine as AttentionStateMachine, DriverState
    from dms.safety import SafetyAssessor, SafetyState
    from dms.config import DMSConfig as Config
    from dms.models import EyeObservation, EyeStateEstimator, FaceDetector, HeadPoseEstimator
except ImportError as err:
    print(f"[ERROR] Failed to import core DMS module: {err}")
    print("[ERROR] Ensure opencv-python, onnxruntime, and numpy are installed in this environment.")
    raise

try:
    import cv2
    import numpy as np
except ImportError:
    print("[ERROR] OpenCV (cv2) or NumPy is not installed.")
    raise

try:
    import requests
except ImportError:
    print("[WARN] 'requests' not installed; falling back to urllib for dashboard client.")
    requests = None


def send_dashboard_request(url: str, endpoint: str, payload: dict[str, Any], timeout: float = 1.0) -> bool:
    """Send JSON payload to Fleet Dashboard."""
    if not url:
        return False
    target = f"{url.rstrip('/')}{endpoint}"
    try:
        if requests is not None:
            resp = requests.post(target, json=payload, timeout=timeout)
            return resp.status_code == 200
        else:
            import json
            import urllib.request
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(target, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status == 200
    except Exception:
        return False


def encode_snapshot(frame: np.ndarray, max_width: int = 320, quality: int = 70) -> str:
    """Encode current frame to JPEG base64 string for dashboard card."""
    try:
        h, w = frame.shape[:2]
        if w > max_width:
            scale = max_width / float(w)
            resized = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            resized = frame
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        ok, buffer = cv2.imencode(".jpg", resized, encode_param)
        if ok:
            return base64.b64encode(buffer.tobytes()).decode("ascii")
    except Exception:
        pass
    return ""


def main():
    parser = argparse.ArgumentParser(description="Laptop Webcam DMS Test Harness")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Camera capture width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Camera capture height (default: 480)")
    parser.add_argument("--dashboard-url", type=str, default="http://127.0.0.1:5000", help="Fleet dashboard URL")
    parser.add_argument("--vehicle-id", type=str, default="VEH-LAPTOP-TEST", help="Vehicle identifier")
    parser.add_argument("--driver-name", type=str, default="Laptop Test", help="Driver name")
    parser.add_argument("--headless", action="store_true", help="Run without live GUI window")
    parser.add_argument("--simulate-duration", type=float, default=0.0, help="Run simulation test for N seconds")
    parser.add_argument("--simulate-eyes-closed", action="store_true", help="Simulate closed eyes for testing")
    args = parser.parse_args()

    models_dir = DRIVER_MONITOR_DIR / "models"
    face_model_path = models_dir / "face_detection_yunet_2026may.onnx"
    headpose_model_path = models_dir / "headpose_mobilenetv3_small.onnx"
    eye_model_path = models_dir / "eyes_ocec_s.onnx"

    for m_path in [face_model_path, headpose_model_path, eye_model_path]:
        if not m_path.exists():
            print(f"[ERROR] Required model not found: {m_path}")
            sys.exit(1)

    # Initialize DMS configuration with the exact fixed parameters
    cfg = Config()
    cfg.model_dir = models_dir
    cfg.width = args.width
    cfg.height = args.height
    cfg.eye_closed_enter = 0.30
    cfg.eye_closed_exit = 0.50
    cfg.eye_closed_seconds = 1.50
    cfg.perclos_threshold = 0.35

    print("==================================================")
    print("      LAPTOP CAMERA DMS TEST HARNESS")
    print("==================================================")
    print(f"Models Dir:         {models_dir}")
    print(f"Face Detector:      {face_model_path.name}")
    print(f"Head Pose:          {headpose_model_path.name}")
    print(f"Eye Classifier:     {eye_model_path.name}")
    print(f"Config Parameters:  enter={cfg.eye_closed_enter:.2f}, exit={cfg.eye_closed_exit:.2f}, "
          f"closed_sec={cfg.eye_closed_seconds:.2f}s, perclos_thr={cfg.perclos_threshold:.2f}")
    print(f"Dashboard URL:      {args.dashboard_url}")
    print(f"Vehicle ID:         {args.vehicle_id}")
    print("==================================================")

    # Instantiate estimators and analyzers
    detector = FaceDetector(
        face_model_path,
        cfg.width,
        cfg.height,
        confidence=cfg.face_confidence,
        nms=cfg.face_nms,
        top_k=cfg.face_top_k,
        input_scale=cfg.face_input_scale,
    )
    pose_estimator = HeadPoseEstimator(headpose_model_path, threads=2)
    eye_estimator = EyeStateEstimator(eye_model_path, threads=2)

    eye_analyzer = EyeAnalyzer(cfg)
    state_machine = AttentionStateMachine(cfg)
    safety_assessor = SafetyAssessor(cfg)

    # Open webcam
    cap = None
    if not args.simulate_duration or not args.headless:
        cap = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        if not cap.isOpened():
            print(f"[WARN] Could not open webcam at index {args.camera_index}.")
            if not args.simulate_duration:
                print("[ERROR] No webcam available. Use --simulate-duration <sec> to run simulation.")
                sys.exit(1)

    # Register session on dashboard
    session_start_payload = {
        "session_id": args.vehicle_id,
        "driver_name": args.driver_name,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "camera_backend": "laptop_webcam",
        "resolution": [cfg.width, cfg.height],
        "fps_target": 30.0,
    }
    if args.dashboard_url:
        ok = send_dashboard_request(args.dashboard_url, "/api/session/start", session_start_payload)
        print(f"[DASHBOARD] Session start registered: {'OK' if ok else 'OFFLINE/FAILED'}")

    last_status_push = 0.0
    last_log_print = time.perf_counter()
    last_event_times: dict[str, float] = {}
    EVENT_COOLDOWN_SECONDS = 10.0
    fps_sample_count = 0
    fps_timer = time.perf_counter()
    fps = 30.0
    start_time = time.perf_counter()
    prev_state_status = "NORMAL"

    print("\nStarting video processing. Press 'Q' to quit, 'C' to calibrate.\n")

    try:
        while True:
            loop_time = time.perf_counter()
            if args.simulate_duration and (loop_time - start_time) > args.simulate_duration:
                print(f"[TEST] Simulation complete ({args.simulate_duration:.1f}s).")
                break

            # Capture or simulate frame
            if cap is not None and cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue
            else:
                # Generate synthetic test frame
                frame = np.full((cfg.height, cfg.width, 3), 40, dtype=np.uint8)
                cv2.putText(frame, "SIMULATION MODE", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                time.sleep(0.033)

            fps_sample_count += 1
            now_sec = time.perf_counter()
            elapsed_fps = now_sec - fps_timer
            if elapsed_fps >= 1.0:
                fps = fps_sample_count / elapsed_fps
                fps_sample_count = 0
                fps_timer = now_sec

            now_ts = time.time()
            face = None
            if not args.simulate_eyes_closed and cap is not None and cap.isOpened():
                faces = detector.detect(frame)
                face = max(faces, key=lambda f: f.area) if faces else None

            left_open = 0.85
            right_open = 0.85
            yaw, pitch, roll = 0.0, 0.0, 0.0

            if face is not None:
                pose = pose_estimator.estimate(frame, face)
                if pose is not None:
                    yaw, pitch, roll = pose.yaw, pose.pitch, pose.roll

                eye_res = eye_estimator.estimate(frame, face, yaw=yaw)
                if eye_res is not None:
                    left_open = eye_res.left_open
                    right_open = eye_res.right_open
                obs = eye_res
            elif args.simulate_eyes_closed:
                left_open = 0.05
                right_open = 0.05
                obs = EyeObservation(left_open=0.05, right_open=0.05, valid=True, boxes=[])
            else:
                # Active simulation or no face seen
                obs = EyeObservation(left_open=0.85, right_open=0.85, valid=True, boxes=[])

            # Update eye metrics
            eye_metrics = eye_analyzer.update(obs, now=now_ts)

            # Update driver attention state machine
            driver_state = state_machine.update(
                now=now_ts,
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                eyes=eye_metrics,
                face_present=True,
                pose_valid=True,
            )

            # Update safety assessor
            safety = safety_assessor.update(now_ts, driver=driver_state, face_present=True)

            perclos = eye_metrics.perclos
            closure_s = eye_metrics.closure_seconds

            # 5-Second Log output matching Pi format
            if (now_sec - last_log_print) >= 5.0:
                print(
                    f"[STATUS 5s] FPS={fps:4.1f} | Left={left_open:.2f} Right={right_open:.2f} | "
                    f"PERCLOS={perclos:.2f} Closure={closure_s:4.1f}s | "
                    f"State={driver_state.status:10s} Safety={safety.level}"
                )
                last_log_print = now_sec

            # Send event to dashboard when safety degrades (with 10-second cooldown)
            is_alert_state = driver_state.status in ("DROWSY", "MICROSLEEP", "DISTRACTED")
            last_sent = last_event_times.get(driver_state.status, 0.0)
            cooldown_expired = (now_sec - last_sent) >= EVENT_COOLDOWN_SECONDS
            state_changed = driver_state.status != prev_state_status

            if is_alert_state and (state_changed or cooldown_expired):
                snapshot_b64 = encode_snapshot(frame, max_width=320, quality=70)
                event_payload = {
                    "event_id": uuid.uuid4().hex,
                    "session_id": args.vehicle_id,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "event": driver_state.status,
                    "kind": driver_state.status,
                    "severity": "CRITICAL" if safety.level == "RED" else "WARNING",
                    "direction": driver_state.zone,
                    "message": f"{driver_state.status}: {driver_state.reason}",
                    "safety_level": safety.level,
                    "safety_reason": safety.reason,
                    "driver_name": args.driver_name,
                    "perclos": round(float(perclos), 3),
                    "closure_s": round(float(closure_s), 2),
                    "closure_seconds": round(float(closure_s), 2),
                    "fps": round(float(fps), 1),
                    "snapshot_jpeg_base64": snapshot_b64,
                    "photo": "",
                }
                send_dashboard_request(args.dashboard_url, "/api/event", event_payload)
                print(
                    f"[EVENT SENT] {driver_state.status} | "
                    f"PERCLOS={perclos:.2f} | "
                    f"Closure={closure_s:.1f}s | "
                    f"Left={left_open:.2f} Right={right_open:.2f}"
                )
                last_event_times[driver_state.status] = now_sec
                prev_state_status = driver_state.status
            elif driver_state.status == "NORMAL":
                prev_state_status = "NORMAL"

            # 1-Second Status Push to Fleet Dashboard
            if (now_sec - last_status_push) >= 1.0:
                snapshot_b64 = encode_snapshot(frame, max_width=320, quality=65)
                status_payload = {
                    "session_id": args.vehicle_id,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "safety_level": safety.level,
                    "safety_label": safety.label,
                    "safety_reason": safety.reason,
                    "state": driver_state.status,
                    "reason": driver_state.reason,
                    "message": safety.message,
                    "zone": driver_state.zone,
                    "driver_name": args.driver_name,
                    "identity_state": "VERIFIED",
                    "perclos": round(float(perclos), 3),
                    "closure_seconds": round(float(closure_s), 2),
                    "fps": round(float(fps), 1),
                    "snapshot_jpeg_base64": snapshot_b64,
                }
                send_dashboard_request(args.dashboard_url, "/api/status", status_payload)
                last_status_push = now_sec

            # Render HUD on live frame
            if not args.headless:
                if face is not None:
                    bx, by, bw, bh = [int(v) for v in face.box]
                    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)

                if safety.level == "GREEN":
                    color = (0, 255, 0)
                elif safety.level == "YELLOW":
                    color = (0, 255, 255)
                else:
                    color = (0, 0, 255)

                cv2.rectangle(frame, (0, 0), (cfg.width, 95), (20, 20, 20), -1)

                cv2.putText(
                    frame,
                    f"SAFETY: {safety.level} ({safety.label}) | STATE: {driver_state.status}",
                    (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                )
                cv2.putText(
                    frame,
                    f"FPS: {fps:4.1f} | PERCLOS: {perclos:.2f} | Closure: {closure_s:3.1f}s",
                    (15, 52),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (220, 220, 220),
                    1,
                )
                cv2.putText(
                    frame,
                    f"Left Eye: {left_open:.2f} | Right Eye: {right_open:.2f} | Q: Quit  C: Calibrate",
                    (15, 78),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (180, 180, 180),
                    1,
                )

                cv2.imshow("Laptop Driver Monitoring Test", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("[USER] Quit requested.")
                    break
                elif key in (ord("c"), ord("C")):
                    state_machine.calibrate(yaw, pitch, roll)
                    print(f"[CALIBRATE] Calibrated neutral pose to yaw={yaw:.1f}, pitch={pitch:.1f}, roll={roll:.1f}")

    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

        if args.dashboard_url:
            send_dashboard_request(
                args.dashboard_url,
                "/api/session/end",
                {"session_id": args.vehicle_id, "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()},
            )
            print("[DASHBOARD] Session end sent.")

    print("Test harness finished cleanly.")


if __name__ == "__main__":
    main()
