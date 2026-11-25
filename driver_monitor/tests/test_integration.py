"""Headless end-to-end pipeline test on a generated video.

The test converts a real event photo into a short video, then runs the full
application (models, state machine, HUD, 3D head, recording and logging) with
temporary output paths.  It skips cleanly when the models or a face photo are
not available.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2

from dms.app import DriverMonitorApp
from dms.config import DMSConfig
from dms.events import LOG_FIELDS
from dms.models import FaceDetector


def find_driver_image(config: DMSConfig) -> Path | None:
    detector = FaceDetector(
        config.face_model_path,
        config.width,
        config.height,
        confidence=config.face_confidence,
        nms=config.face_nms,
        top_k=config.face_top_k,
        input_scale=config.face_input_scale,
    )
    root = Path(__file__).resolve().parent.parent
    for path in sorted((root / "events").glob("*.jpg")):
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        if detector.detect(frame):
            return path
    return None


def make_video(image_path: Path, target: Path, frames: int = 8) -> None:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Cannot read {image_path}")
    frame = cv2.resize(frame, (640, 480))
    writer = cv2.VideoWriter(
        str(target),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (640, 480),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV cannot create a test video")
    try:
        for _ in range(frames):
            writer.write(frame)
    finally:
        writer.release()


class PipelineIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = DMSConfig()
        if not all(
            path.exists()
            for path in (
                cls.config.face_model_path,
                cls.config.headpose_model_path,
                cls.config.eye_model_path,
            )
        ):
            raise unittest.SkipTest("ONNX models are not downloaded")
        try:
            cls.image = find_driver_image(cls.config)
        except Exception as exc:  # pragma: no cover - environment dependent
            raise unittest.SkipTest(f"cannot load models: {exc}") from exc
        if cls.image is None:
            raise unittest.SkipTest("no event photo with a face")

    def test_headless_pipeline_records_and_logs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dms-integration-") as tmp:
            root = Path(tmp)
            video = root / "input.avi"
            make_video(self.image, video)

            config = DMSConfig()
            config.width = 640
            config.height = 480
            config.fps = 10
            config.video = str(video)
            config.show_windows = False
            config.show_head_window = False
            config.show_head_inset = True
            config.buzzer_enabled = False
            config.use_calibration = False
            config.max_frames = 8
            config.debug = True
            config.event_dir = root / "events"
            config.log_file = config.event_dir / "dms_log.csv"
            config.calibration_file = root / "calibration.json"
            config.record_path = str(root / "session.avi")

            app = DriverMonitorApp(config, headless=True, show_windows=False)
            summary = app.run()

            self.assertEqual(summary["frames"], 8)
            self.assertEqual(summary["state_frames"].get("NO_FACE", 0), 0)
            self.assertGreater(summary["latency_ms"]["face"]["mean"], 0.0)
            self.assertGreater(summary["latency_ms"]["pose"]["mean"], 0.0)
            self.assertGreater(summary["latency_ms"]["head"]["mean"], 0.0)

            recording = root / "session.avi"
            self.assertTrue(recording.exists())
            self.assertGreater(recording.stat().st_size, 0)

            self.assertTrue(config.log_file.exists())
            header = config.log_file.read_text(encoding="utf-8").splitlines()[0]
            self.assertEqual(header.split(","), LOG_FIELDS)

            sessions = list((root / "events").glob("session_*.json"))
            self.assertEqual(len(sessions), 1)
            self.assertIn('"frames": 8', sessions[0].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()