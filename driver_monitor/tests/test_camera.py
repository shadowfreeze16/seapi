"""Video-file sources and source selection."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from dms.camera import (
    NetworkSource,
    SourceInfo,
    VideoFileSource,
    create_source,
    is_stream_url,
)
from dms.config import DMSConfig


def write_video(path: Path, frames: int = 8) -> bool:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    if not writer.isOpened():
        return False
    for index in range(frames):
        frame = np.full((48, 64, 3), index * 20 % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return True


class VideoFileSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "clip.avi"
        if not write_video(self.path):
            self.skipTest("OpenCV cannot write MJPG video on this system")
        self.source = VideoFileSource(self.path)

    def tearDown(self) -> None:
        self.source.release()
        self.temp.cleanup()

    def test_info(self) -> None:
        info = self.source.info
        self.assertTrue(info.is_file)
        self.assertEqual(info.backend, "file")
        self.assertEqual((info.width, info.height), (64, 48))
        self.assertAlmostEqual(info.fps, 10.0)
        self.assertEqual(info.frame_count, 8)

    def test_media_timestamps(self) -> None:
        ok, frame, timestamp = self.source.read()
        self.assertTrue(ok)
        self.assertEqual(frame.shape, (48, 64, 3))
        self.assertAlmostEqual(timestamp, 0.0)
        ok, _, timestamp = self.source.read()
        self.assertTrue(ok)
        self.assertAlmostEqual(timestamp, 0.1)
        self.assertAlmostEqual(self.source.now(), 0.2)

    def test_end_of_file(self) -> None:
        for _ in range(8):
            self.assertTrue(self.source.read()[0])
        self.assertFalse(self.source.read()[0])

    def test_missing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            VideoFileSource(Path(self.temp.name) / "missing.avi")

    def test_create_source_uses_file(self) -> None:
        config = DMSConfig()
        config.video = str(self.path)
        source = create_source(config)
        self.assertIsInstance(source, VideoFileSource)
        source.release()


class LoopingSourceTests(unittest.TestCase):
    def test_loop_wraps_to_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.avi"
            if not write_video(path, frames=3):
                self.skipTest("OpenCV cannot write MJPG video")
            source = VideoFileSource(path, loop=True)
            try:
                for _ in range(6):
                    ok, _, _ = source.read()
                    self.assertTrue(ok)
            finally:
                source.release()


class FakeCapture:
    """Minimal cv2.VideoCapture replacement for network-source tests."""

    instances: list["FakeCapture"] = []
    frame_batches: list[list[np.ndarray]] = []

    def __init__(self, url, api=None, params=None) -> None:
        self.url = url
        self.api = api
        self.params = params
        self.frame_batch = (
            FakeCapture.frame_batches.pop(0)
            if FakeCapture.frame_batches
            else [np.zeros((48, 64, 3), dtype=np.uint8)]
        )
        self.opened = True
        self.released = False
        FakeCapture.instances.append(self)

    def isOpened(self) -> bool:
        return self.opened

    def set(self, prop, value) -> bool:
        return True

    def read(self):
        if self.released or not self.frame_batch:
            return False, None
        return True, self.frame_batch.pop(0)

    def release(self) -> None:
        self.released = True


class NetworkSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeCapture.instances = []
        FakeCapture.frame_batches = []

    def test_stream_url_detection(self) -> None:
        for url in (
            "http://192.168.1.50:4747/video",
            "https://camera.example/stream.mjpg",
            "rtsp://user:pass@192.168.1.51:554/live",
            "rtmp://host/live",
        ):
            self.assertTrue(is_stream_url(url), url)
        for value in (None, "", "clip.mp4", "/tmp/http_clip.avi"):
            self.assertFalse(is_stream_url(value), value)

    def test_reads_frames_and_updates_dimensions(self) -> None:
        FakeCapture.frame_batches = [
            [np.zeros((480, 640, 3), dtype=np.uint8)]
        ]
        with patch("dms.camera.cv2.VideoCapture", FakeCapture):
            source = NetworkSource(
                "http://user:secret@192.168.1.50:4747/video?token=hidden",
                width=320,
                height=240,
            )
            try:
                self.assertTrue(source.info.is_network)
                self.assertEqual(source.info.backend, "network")
                self.assertEqual(
                    source.info.description,
                    "http://192.168.1.50:4747/video",
                )
                ok, frame, timestamp = source.read()
                self.assertTrue(ok)
                self.assertEqual(frame.shape, (480, 640, 3))
                self.assertEqual(
                    (source.info.width, source.info.height), (640, 480)
                )
                self.assertGreaterEqual(timestamp, 0.0)
            finally:
                source.release()
        self.assertEqual(len(FakeCapture.instances), 1)
        self.assertTrue(FakeCapture.instances[0].released)

    def test_reconnects_after_dropout(self) -> None:
        FakeCapture.frame_batches = [
            [np.zeros((48, 64, 3), dtype=np.uint8)],
            [np.full((48, 64, 3), 7, dtype=np.uint8)],
        ]
        with patch("dms.camera.cv2.VideoCapture", FakeCapture):
            source = NetworkSource(
                "http://192.168.1.50:4747/video", reconnect_delay=0.05
            )
            try:
                self.assertTrue(source.read()[0])
                self.assertFalse(source.read()[0])
                self.assertTrue(FakeCapture.instances[0].released)
                source._next_retry = 0.0
                ok, frame, _ = source.read()
                self.assertTrue(ok)
                self.assertEqual(int(frame[0, 0, 0]), 7)
                self.assertEqual(len(FakeCapture.instances), 2)
            finally:
                source.release()

    def test_unreachable_stream_raises(self) -> None:
        class ClosedCapture(FakeCapture):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.opened = False

        with patch("dms.camera.cv2.VideoCapture", ClosedCapture):
            with self.assertRaises(RuntimeError) as raised:
                NetworkSource("http://192.168.1.50:4747/video")
        self.assertIn("Cannot open network stream", str(raised.exception))

    def test_create_source_uses_network(self) -> None:
        config = DMSConfig()
        config.video = "http://192.168.1.50:4747/video"
        config.width = 800
        config.height = 600
        config.fps = 15
        with patch("dms.camera.NetworkSource") as network:
            source = create_source(config)
        network.assert_called_once_with(
            config.video, width=800, height=600, fps=15
        )
        self.assertIs(source, network.return_value)


class SourceInfoTests(unittest.TestCase):
    def test_dataclass_defaults(self) -> None:
        info = SourceInfo("file", "clip.avi", 64, 48, 10.0)
        self.assertFalse(info.is_file)
        self.assertFalse(info.is_network)
        self.assertEqual(info.frame_count, 0)


if __name__ == "__main__":
    unittest.main()