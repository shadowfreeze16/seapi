"""Push status and events to the laptop dashboard.

The dashboard runs on the driver's laptop and is reachable over the local
network.  This client is deliberately small and standard-library only:

* status snapshots are sent at most once per ``dashboard_interval`` seconds,
* events are queued and sent in the background,
* failed events are appended to ``events/dashboard_outbox.jsonl`` and replayed
  when the dashboard comes back,
* the video loop never waits for the network.

The JSON contract matches the laptop dashboard described in the README.
"""

from __future__ import annotations

import base64
import json
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import DMSConfig
from .identity import IdentityResult
from .safety import SafetyState
from .state import DriverState, EventRequest


def _iso_now() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def _round(value: Any, digits: int = 2) -> Any:
    if value is None or value == "":
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def encode_snapshot(
    frame: np.ndarray | None,
    max_width: int = 960,
    quality: int = 80,
) -> str | None:
    """Resize and JPEG-encode a frame for the dashboard event payload."""
    if frame is None or not getattr(frame, "size", 0):
        return None
    image = frame
    height, width = image.shape[:2]
    if max_width > 0 and width > max_width:
        scale = max_width / float(width)
        image = cv2.resize(
            image,
            (max_width, max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        return None
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _identity_fields(identity: IdentityResult | None) -> dict[str, Any]:
    if identity is None:
        return {
            "identity_state": "OFF",
            "driver_name": None,
            "identity_score": None,
            "identity_message": None,
        }
    return {
        "identity_state": identity.state,
        "driver_name": identity.name or identity.best_name,
        "identity_score": _round(identity.score or identity.best_score, 3),
        "identity_message": identity.message,
    }


def make_event_payload(
    session_id: str,
    request: EventRequest,
    row: dict[str, Any] | None = None,
    safety: SafetyState | None = None,
    identity: IdentityResult | None = None,
    gps: dict[str, Any] | None = None,
    snapshot_jpeg_base64: str | None = None,
    photo_name: str | None = None,
    event_id: str | None = None,
) -> dict[str, Any]:
    """Build the dashboard event object for one logged event."""
    row = row or {}
    identity_fields = _identity_fields(identity)
    photo = photo_name or row.get("photo") or ""
    return {
        "event_id": event_id or uuid.uuid4().hex,
        "session_id": session_id,
        "timestamp": _iso_now(),
        "event": request.kind,
        "kind": request.kind,
        "status": request.status,
        "severity": request.severity,
        "direction": request.direction,
        "message": request.message,
        "safety_level": safety.level if safety else row.get("safety_level"),
        "safety_reason": safety.reason if safety else row.get("safety_reason"),
        "face_count": row.get("face_count"),
        "yaw": _round(row.get("yaw")),
        "pitch": _round(row.get("pitch")),
        "roll": _round(row.get("roll")),
        "eye_left": _round(row.get("eye_left")),
        "eye_right": _round(row.get("eye_right")),
        "perclos": _round(row.get("perclos")),
        "blink_rate": _round(row.get("blink_rate")),
        "closure_s": _round(row.get("closure_s")),
        "fps": _round(row.get("fps")),
        "latency_ms": _round(row.get("latency_ms")),
        **identity_fields,
        "gps": gps or {},
        "snapshot_jpeg_base64": snapshot_jpeg_base64,
        "photo_name": Path(str(photo)).name if photo else None,
    }


def make_status_payload(
    session_id: str,
    safety: SafetyState,
    driver: DriverState,
    identity: IdentityResult | None = None,
    eyes: Any | None = None,
    pose: Any | None = None,
    fps: float = 0.0,
    gps: dict[str, Any] | None = None,
    voice_enabled: bool = False,
    camera_setup: dict[str, Any] | None = None,
    session_seconds: float = 0.0,
) -> dict[str, Any]:
    """Build the one-second status object consumed by ``GET /api/status``."""
    identity_fields = _identity_fields(identity)
    return {
        "session_id": session_id,
        "timestamp": _iso_now(),
        "safety_level": safety.level,
        "safety_label": safety.label,
        "safety_reason": safety.reason,
        "state": driver.status,
        "reason": driver.reason,
        "message": safety.message,
        "zone": driver.zone,
        "face_present": driver.status != "NO_FACE",
        "yaw": _round(pose.yaw if pose is not None else driver.yaw),
        "pitch": _round(pose.pitch if pose is not None else driver.pitch),
        "roll": _round(pose.roll if pose is not None else driver.roll),
        "eye_left": _round(getattr(eyes, "left_open", None)),
        "eye_right": _round(getattr(eyes, "right_open", None)),
        "perclos": _round(getattr(eyes, "perclos", None), 3),
        "blink_rate": _round(getattr(eyes, "blink_rate", None), 1),
        "closure_s": _round(getattr(eyes, "closure_seconds", None)),
        "offroad_seconds": _round(driver.offroad_seconds),
        "no_face_seconds": _round(driver.no_face_seconds),
        "status_seconds": _round(driver.status_seconds),
        "fps": _round(fps, 1),
        **identity_fields,
        "gps": gps or {},
        "voice_enabled": bool(voice_enabled),
        "camera_setup": camera_setup or {},
        "session_seconds": _round(session_seconds, 1),
    }


class DashboardClient:
    """Background HTTP sender with a disk-backed offline queue."""

    def __init__(
        self,
        config: DMSConfig,
        session_id: str | None = None,
    ) -> None:
        self.config = config
        self.base_url = (
            config.dashboard_url.rstrip("/") if config.dashboard_url else None
        )
        self.enabled = bool(self.base_url)
        self.session_id = session_id or datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        self.outbox = Path(config.dashboard_outbox)
        self._queue: deque[tuple[str, dict[str, Any]]] = deque(
            maxlen=max(1, int(config.dashboard_queue_size))
        )
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest_status: dict[str, Any] | None = None
        self._status_pending = False
        self._status_sent_at = 0.0
        self._last_outbox_attempt = 0.0
        self._connected: bool | None = None
        self.sent = 0
        self.failed = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self, payload: dict[str, Any] | None = None) -> None:
        if not self.enabled or self._thread is not None:
            return
        start_payload = payload or {
            "session_id": self.session_id,
            "timestamp": _iso_now(),
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }
        start_payload.setdefault("session_id", self.session_id)
        self._enqueue("/api/session/start", start_payload)
        self._thread = threading.Thread(
            target=self._run,
            name="dms-dashboard",
            daemon=True,
        )
        self._thread.start()
        print(f"[DASH] sending to {self.base_url}")

    def stop(self, timeout: float = 4.0) -> None:
        if not self.enabled:
            return
        self._stop.set()
        with self._lock:
            # Keep only queued events (including the session-end marker).
            self._latest_status = None
            self._status_pending = False
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def end_session(self, payload: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        end_payload = payload or {
            "session_id": self.session_id,
            "timestamp": _iso_now(),
            "ended_at": datetime.now().isoformat(timespec="seconds"),
        }
        end_payload.setdefault("session_id", self.session_id)
        self._enqueue("/api/session/end", end_payload)
        # Give the worker a moment to send the queued events and the end
        # marker before the process exits.
        self.stop(timeout=5.0)

    # ------------------------------------------------------------------
    # Producer API (called from the video loop)
    # ------------------------------------------------------------------
    def set_status(self, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._latest_status = payload
            self._status_pending = True
        self._wake.set()

    def send_event(self, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._enqueue("/api/event", payload)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            queued = len(self._queue)
            pending_status = self._status_pending
        return {
            "enabled": self.enabled,
            "url": self.base_url,
            "session_id": self.session_id,
            "sent": self.sent,
            "failed": self.failed,
            "queued": queued,
            "pending_status": pending_status,
            "connected": self._connected,
            "outbox": str(self.outbox),
        }

    def drain(self, timeout: float = 5.0) -> bool:
        """Wait until the queue and latest status have been sent."""
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() < deadline:
            with self._lock:
                idle = not self._queue and not self._status_pending
            if idle:
                return True
            time.sleep(0.02)
        return False

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------
    def _enqueue(self, endpoint: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._queue.append((endpoint, payload))
        self._wake.set()

    def _run(self) -> None:
        while True:
            self._wake.clear()
            now = time.monotonic()

            status = None
            with self._lock:
                if (
                    self._latest_status is not None
                    and self._status_pending
                    and not self._stop.is_set()
                    and (now - self._status_sent_at)
                    >= self.config.dashboard_interval
                ):
                    status = self._latest_status
                    self._status_pending = False
                    self._status_sent_at = now
                item = self._queue.popleft() if self._queue else None

            if status is not None:
                self._deliver("/api/status", status)
            if item is not None:
                self._deliver(item[0], item[1])

            if (
                now - self._last_outbox_attempt
                >= max(5.0, self.config.dashboard_interval * 5.0)
            ):
                self._last_outbox_attempt = now
                self._flush_outbox()

            with self._lock:
                idle = not self._queue and not self._status_pending
            if self._stop.is_set() and idle:
                break
            self._wake.wait(0.1)
        self._flush_outbox()

    def _deliver(self, endpoint: str, payload: dict[str, Any]) -> bool:
        if self._post(endpoint, payload):
            self.sent += 1
            return True
        self.failed += 1
        self._append_outbox(endpoint, payload)
        return False

    # ------------------------------------------------------------------
    # HTTP / outbox
    # ------------------------------------------------------------------
    def _post(self, endpoint: str, payload: dict[str, Any]) -> bool:
        if not self.base_url:
            return False
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.dashboard_timeout
            ) as response:
                ok = 200 <= int(response.status) < 300
            if ok != self._connected:
                print(
                    "[DASH] connected"
                    if ok
                    else f"[DASH] server returned HTTP {response.status}"
                )
            self._connected = ok
            return ok
        except (urllib.error.URLError, OSError, ValueError) as exc:
            if self._connected is not False:
                print(f"[DASH] offline ({exc}); events will be queued")
            self._connected = False
            return False

    def _append_outbox(self, endpoint: str, payload: dict[str, Any]) -> None:
        try:
            self.outbox.parent.mkdir(parents=True, exist_ok=True)
            with self.outbox.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "endpoint": endpoint,
                            "payload": payload,
                            "queued_at": _iso_now(),
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        except OSError as exc:
            print(f"[DASH] cannot write outbox: {exc}")

    def _flush_outbox(self) -> None:
        if not self.outbox.exists():
            return
        try:
            lines = self.outbox.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        if not lines:
            try:
                self.outbox.unlink()
            except OSError:
                pass
            return
        remaining: list[str] = []
        limit = max(1, int(self.config.dashboard_queue_size))
        for index, line in enumerate(lines):
            if index >= limit:
                remaining.append(line)
                continue
            try:
                item = json.loads(line)
                endpoint = str(item["endpoint"])
                payload = item["payload"]
            except (ValueError, KeyError, TypeError):
                continue
            if isinstance(payload, dict) and self._post(endpoint, payload):
                self.sent += 1
            else:
                remaining.append(line)
                self.failed += 1
        try:
            if remaining:
                self.outbox.write_text(
                    "\n".join(remaining) + "\n", encoding="utf-8"
                )
            else:
                self.outbox.unlink()
        except OSError:
            pass


__all__ = [
    "DashboardClient",
    "encode_snapshot",
    "make_event_payload",
    "make_status_payload",
]
