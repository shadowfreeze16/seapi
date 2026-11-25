"""Camera and video-file sources with graceful fallbacks.

Supported backends:

* ``picamera2`` -- the CSI camera on Raspberry Pi OS (preferred on a Pi).
* ``v4l2``     -- USB webcams, DroidCam and other /dev/video* devices.
* video files  -- useful for demos and repeatable integration tests.

The application asks for :class:`FrameSource`, which tries the configured
backend and falls back to the others.  ``read()`` returns ``(ok, frame,
timestamp)``; for video files the timestamp is the media clock so that state
machines and PERCLOS behave deterministically even if the CPU is slower than
real time.
"""

from __future__ import annotations

import glob
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np

from .config import DMSConfig


_STREAM_SCHEMES = ("http://", "https://", "rtsp://", "rtmp://", "tcp://", "udp://")


def is_stream_url(value: str | None) -> bool:
    """Return True when *value* is a network video URL rather than a file."""
    return bool(value) and str(value).lower().startswith(_STREAM_SCHEMES)


def _safe_url(url: str) -> str:
    """Hide credentials and query tokens before a URL is printed or logged."""
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        if parts.port:
            host = f"{host}:{parts.port}"
        return urlunsplit((parts.scheme, host, parts.path, "", ""))
    except (ValueError, TypeError):
        return url


@dataclass
class SourceInfo:
    backend: str
    description: str
    width: int
    height: int
    fps: float
    is_file: bool = False
    frame_count: int = 0
    is_network: bool = False


class FrameSource:
    """Base class for a frame source."""

    info: SourceInfo

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        raise NotImplementedError

    def release(self) -> None:
        pass

    def now(self) -> float:
        return time.monotonic()


class VideoFileSource(FrameSource):
    def __init__(self, path: str | Path, loop: bool = False) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video file not found: {self.path}")
        self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Cannot open video file: {self.path}")
        fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        if not (1.0 <= fps <= 240.0):
            fps = 30.0
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = fps
        self.loop = bool(loop)
        self.frame_index = 0
        self.info = SourceInfo(
            backend="file",
            description=str(self.path),
            width=width,
            height=height,
            fps=fps,
            is_file=True,
            frame_count=count,
        )

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        ok, frame = self.capture.read()
        if not ok and self.loop:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_index = 0
            ok, frame = self.capture.read()
        if not ok:
            return False, None, self.now()
        timestamp = self.frame_index / self.fps
        self.frame_index += 1
        return True, frame, timestamp

    def now(self) -> float:
        return self.frame_index / self.fps

    def release(self) -> None:
        self.capture.release()


class NetworkSource(FrameSource):
    """Live video from an HTTP(S) MJPEG, RTSP, RTMP or TCP/UDP stream.

    DroidCam's Android app exposes its camera at ``http://PHONE_IP:4747/video``;
    that feed can be opened directly without the x86-only desktop client or a
    v4l2loopback kernel driver.  The source reconnects automatically after a
    short interruption and uses the monotonic clock for all timing decisions.
    """

    def __init__(
        self,
        url: str,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        open_timeout_ms: int = 5000,
        read_timeout_ms: int = 5000,
        reconnect_delay: float = 0.5,
    ) -> None:
        self.url = str(url)
        self.open_timeout_ms = int(open_timeout_ms)
        self.read_timeout_ms = int(read_timeout_ms)
        self.reconnect_delay = max(0.05, float(reconnect_delay))
        self._capture: cv2.VideoCapture | None = None
        self._next_retry = 0.0
        self._last = time.monotonic()
        self.info = SourceInfo(
            backend="network",
            description=_safe_url(self.url),
            width=int(width),
            height=int(height),
            fps=float(fps),
            is_network=True,
        )
        self._open()

    def _open(self) -> None:
        params = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
            self.open_timeout_ms,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC,
            self.read_timeout_ms,
        ]
        try:
            capture = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG, params)
        except TypeError:
            # Older OpenCV builds do not accept the params list.
            capture = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                f"Cannot open network stream: {self.info.description}. "
                "Check the phone IP/port, Wi-Fi, and that the app is streaming."
            )
        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self._capture = capture

    def _close_capture(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _update_info(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        if (width, height) != (self.info.width, self.info.height):
            self.info.width = int(width)
            self.info.height = int(height)

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        now = time.monotonic()
        if self._capture is None:
            if now < self._next_retry:
                return False, None, now
            try:
                self._open()
            except RuntimeError:
                self._next_retry = now + self.reconnect_delay
                return False, None, now

        ok, frame = self._capture.read()
        now = time.monotonic()
        if not ok or frame is None:
            self._close_capture()
            self._next_retry = now + self.reconnect_delay
            return False, None, now
        self._update_info(frame)
        self._last = now
        return True, frame, now

    def now(self) -> float:
        return self._last

    def release(self) -> None:
        self._close_capture()


class V4L2Source(FrameSource):
    def __init__(
        self,
        index: int,
        width: int,
        height: int,
        fps: int,
    ) -> None:
        self.index = int(index)
        self.capture = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f"Cannot open V4L2 camera {self.index}")
        # MJPG gives higher frame-rates on many USB webcams.
        self.capture.set(
            cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")
        )
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        self.capture.set(cv2.CAP_PROP_FPS, int(fps))
        try:
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        actual_w = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
        actual_h = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
        actual_fps = float(self.capture.get(cv2.CAP_PROP_FPS)) or float(fps)
        self.info = SourceInfo(
            backend="v4l2",
            description=f"/dev/video{self.index}",
            width=actual_w,
            height=actual_h,
            fps=actual_fps,
        )
        self._last = time.monotonic()

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        ok, frame = self.capture.read()
        now = time.monotonic()
        self._last = now
        return ok, frame, now

    def now(self) -> float:
        return self._last

    def release(self) -> None:
        self.capture.release()


class Picamera2Source(FrameSource):
    def __init__(self, width: int, height: int, fps: int) -> None:
        try:
            from picamera2 import Picamera2  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Picamera2 is not installed (sudo apt install python3-picamera2)"
            ) from exc

        try:
            self.camera = Picamera2()
        except Exception as exc:
            raise RuntimeError(f"Picamera2 could not open a camera: {exc}") from exc

        config = self.camera.create_preview_configuration(
            main={"size": (int(width), int(height)), "format": "RGB888"},
            buffer_count=2,
        )
        self.camera.configure(config)
        self.camera.start()
        # Let exposure settle.
        time.sleep(0.4)
        self.info = SourceInfo(
            backend="picamera2",
            description="CSI camera",
            width=int(width),
            height=int(height),
            fps=float(fps),
        )
        self._last = time.monotonic()

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        try:
            frame = self.camera.capture_array("main")
        except Exception:
            return False, None, time.monotonic()
        # Picamera2 RGB888 is RGB; the rest of the pipeline expects BGR.
        if frame.ndim == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self._last = time.monotonic()
        return True, frame, self._last

    def now(self) -> float:
        return self._last

    def release(self) -> None:
        try:
            self.camera.stop()
            self.camera.close()
        except Exception:
            pass


def list_video_devices() -> list[str]:
    return sorted(glob.glob("/dev/video*"))


def _try_v4l2(
    config: DMSConfig, indices: list[int]
) -> tuple[FrameSource | None, str | None]:
    errors: list[str] = []
    for index in indices:
        try:
            source = V4L2Source(index, config.width, config.height, config.fps)
            return source, None
        except RuntimeError as exc:
            errors.append(str(exc))
    return None, "; ".join(errors) if errors else "no video devices"


def create_source(config: DMSConfig) -> FrameSource:
    """Create the best available source for this configuration."""
    if config.video:
        if is_stream_url(config.video):
            return NetworkSource(
                config.video,
                width=config.width,
                height=config.height,
                fps=config.fps,
            )
        return VideoFileSource(config.video, loop=config.loop_video)

    backend = config.camera.lower()
    errors: list[str] = []

    if backend in ("auto", "picamera2"):
        try:
            return Picamera2Source(config.width, config.height, config.fps)
        except RuntimeError as exc:
            errors.append(f"picamera2: {exc}")
            if backend == "picamera2":
                raise

    if backend in ("auto", "v4l2"):
        indices = [config.camera_index]
        if backend == "auto":
            indices += [i for i in (0, 1, 2) if i != config.camera_index]
        source, error = _try_v4l2(config, indices)
        if source is not None:
            return source
        if error:
            errors.append(f"v4l2: {error}")
        if backend == "v4l2":
            raise RuntimeError("; ".join(errors) or "no V4L2 camera")

    devices = ", ".join(list_video_devices()) or "none"
    raise RuntimeError(
        "No camera source available. "
        f"Tried: {'; '.join(errors)}. "
        f"Video devices: {devices}. "
        "Start DroidCam / connect the CSI camera, or use --video FILE."
    )


__all__ = [
    "FrameSource",
    "NetworkSource",
    "Picamera2Source",
    "SourceInfo",
    "V4L2Source",
    "VideoFileSource",
    "create_source",
    "is_stream_url",
    "list_video_devices",
]