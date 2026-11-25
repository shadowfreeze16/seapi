"""GPS position from ``gpsd``.

The monitor does not talk to a serial GPS device directly.  ``gpsd`` owns the
receiver and exposes a line-delimited JSON protocol on TCP port 2947.  This
client runs in a background thread, keeps the most recent fix, and reports a
stale fix as "waiting" instead of showing an old position as if it were live.

If gpsd is missing the client stays quiet; the dashboard simply shows
``GPS: waiting`` or ``GPS: off``.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass
from datetime import datetime

from .config import DMSConfig

MODE_LABELS = {
    0: "unknown",
    1: "no fix",
    2: "2D",
    3: "3D",
}


@dataclass
class GPSFix:
    """Latest position and quality information."""

    lat: float | None = None
    lon: float | None = None
    alt: float | None = None
    speed_mps: float | None = None
    track: float | None = None
    mode: int = 0
    satellites: int = 0
    hdop: float | None = None
    updated: float | None = None
    received: str | None = None
    error: str | None = None

    @property
    def has_fix(self) -> bool:
        return (
            self.lat is not None
            and self.lon is not None
            and self.mode >= 2
        )

    def age(self, now: float | None = None) -> float | None:
        if self.updated is None:
            return None
        current = time.monotonic() if now is None else float(now)
        return max(0.0, current - self.updated)

    def stale(self, now: float | None = None, limit: float = 5.0) -> bool:
        age = self.age(now)
        return age is None or age > limit

    def text(self, now: float | None = None, limit: float = 5.0) -> str:
        if not self.has_fix:
            return "GPS: waiting"
        if self.stale(now, limit):
            return "GPS: waiting"
        return f"GPS: {self.lat:.5f}, {self.lon:.5f}"

    def as_dict(
        self, now: float | None = None, limit: float = 5.0
    ) -> dict[str, object]:
        age = self.age(now)
        return {
            "enabled": True,
            "fix": bool(self.has_fix and not self.stale(now, limit)),
            "lat": None if self.lat is None else round(self.lat, 6),
            "lon": None if self.lon is None else round(self.lon, 6),
            "alt": None if self.alt is None else round(self.alt, 2),
            "speed_mps": None
            if self.speed_mps is None
            else round(self.speed_mps, 2),
            "track": None if self.track is None else round(self.track, 1),
            "mode": int(self.mode),
            "mode_label": MODE_LABELS.get(int(self.mode), "unknown"),
            "satellites": int(self.satellites),
            "hdop": None if self.hdop is None else round(self.hdop, 1),
            "age_s": None if age is None else round(age, 1),
            "stale": bool(self.stale(now, limit)),
            "text": self.text(now, limit),
            "received": self.received,
            "error": self.error,
        }


def disabled_gps() -> dict[str, object]:
    """Payload used when GPS support is switched off."""
    return {
        "enabled": False,
        "fix": False,
        "lat": None,
        "lon": None,
        "alt": None,
        "speed_mps": None,
        "track": None,
        "mode": 0,
        "mode_label": "off",
        "satellites": 0,
        "hdop": None,
        "age_s": None,
        "stale": True,
        "text": "GPS: off",
        "received": None,
        "error": None,
    }


class GPSClient:
    """Threaded gpsd JSON client."""

    def __init__(self, config: DMSConfig) -> None:
        self.config = config
        self.enabled = config.gps_source == "gpsd"
        self._lock = threading.Lock()
        self._fix = GPSFix()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._socket: socket.socket | None = None
        self._reported_error = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="dms-gps",
            daemon=True,
        )
        self._thread.start()
        print(
            f"[GPS] gpsd client started ({self.config.gps_host}:"
            f"{self.config.gps_port})"
        )

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        sock = self._socket
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def snapshot(self, now: float | None = None) -> dict[str, object]:
        if not self.enabled:
            return disabled_gps()
        with self._lock:
            fix = GPSFix(
                lat=self._fix.lat,
                lon=self._fix.lon,
                alt=self._fix.alt,
                speed_mps=self._fix.speed_mps,
                track=self._fix.track,
                mode=self._fix.mode,
                satellites=self._fix.satellites,
                hdop=self._fix.hdop,
                updated=self._fix.updated,
                received=self._fix.received,
                error=self._fix.error,
            )
        return fix.as_dict(now, self.config.gps_stale_seconds)

    def fix(self) -> GPSFix:
        with self._lock:
            return GPSFix(
                lat=self._fix.lat,
                lon=self._fix.lon,
                alt=self._fix.alt,
                speed_mps=self._fix.speed_mps,
                track=self._fix.track,
                mode=self._fix.mode,
                satellites=self._fix.satellites,
                hdop=self._fix.hdop,
                updated=self._fix.updated,
                received=self._fix.received,
                error=self._fix.error,
            )

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._connect_and_read()
            except OSError as exc:
                if self._stop.is_set():
                    break
                if not self._reported_error:
                    print(f"[GPS] gpsd unavailable: {exc}")
                    self._reported_error = True
                with self._lock:
                    self._fix.error = str(exc)
            finally:
                sock = self._socket
                self._socket = None
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass
            if self._stop.wait(self.config.gps_reconnect_seconds):
                break
        print("[GPS] client stopped")

    def _connect_and_read(self) -> None:
        sock = socket.create_connection(
            (self.config.gps_host, self.config.gps_port), timeout=3.0
        )
        self._socket = sock
        sock.settimeout(1.0)
        sock.sendall(b'?WATCH={"enable":true,"json":true}\n')
        print("[GPS] connected to gpsd")
        self._reported_error = False
        with self._lock:
            self._fix.error = None
        buffer = b""
        while not self._stop.is_set():
            try:
                chunk = sock.recv(4096)
            except socket.timeout:
                continue
            if not chunk:
                raise OSError("gpsd closed the connection")
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                text = line.decode("utf-8", "replace").strip()
                if text:
                    self._handle_line(text)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    def _handle_line(self, line: str) -> None:
        try:
            payload = json.loads(line)
        except ValueError:
            return
        if not isinstance(payload, dict):
            return
        cls = payload.get("class")
        if cls == "TPV":
            self._update_tpv(payload)
        elif cls == "SKY":
            satellites = payload.get("satellites")
            if isinstance(satellites, list):
                used = sum(
                    1
                    for sat in satellites
                    if isinstance(sat, dict) and sat.get("used")
                )
                if used:
                    with self._lock:
                        self._fix.satellites = used

    def _update_tpv(self, payload: dict[str, object]) -> None:
        def number(key: str) -> float | None:
            value = payload.get(key)
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        mode = int(number("mode") or 0)
        lat = number("lat")
        lon = number("lon")
        hdop = number("hdop")
        with self._lock:
            self._fix.lat = lat
            self._fix.lon = lon
            self._fix.alt = number("alt")
            self._fix.speed_mps = number("speed")
            self._fix.track = number("track")
            self._fix.mode = mode
            self._fix.hdop = hdop
            self._fix.updated = time.monotonic()
            self._fix.received = datetime.now().isoformat(timespec="seconds")
            self._fix.error = None


__all__ = [
    "GPSClient",
    "GPSFix",
    "MODE_LABELS",
    "disabled_gps",
]
