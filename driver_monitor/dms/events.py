"""Event photos, CSV logging, pruning and session summaries.

The version-2 log keeps the four legacy columns first so existing scripts keep
working, then adds pose, eye and timing columns.  A legacy four-column log is
rotated out of the way on first start instead of being appended to with a
mismatched header.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import DMSConfig
from .state import EventRequest

LOG_FIELDS = [
    "timestamp",
    "event",
    "direction",
    "face_count",
    "status",
    "severity",
    "yaw",
    "pitch",
    "roll",
    "eye_left",
    "eye_right",
    "perclos",
    "blink_rate",
    "closure_s",
    "fps",
    "latency_ms",
    "message",
    "photo",
    # v3: three-level safety, identity and GPS.
    "safety_level",
    "safety_reason",
    "identity_state",
    "identity_name",
    "identity_score",
    "gps_lat",
    "gps_lon",
    "gps_text",
]

LEGACY_FIELDS = ["timestamp", "event", "direction", "face_count"]

# The 18-column v2 header.  Old v2 logs are rotated aside on first start so
# they keep their own schema and are not overwritten.
V2_FIELDS = LOG_FIELDS[:18]

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe(value: str) -> str:
    return _SAFE.sub("_", value.strip())[:48] or "event"


def _photo_pattern() -> re.Pattern[str]:
    return re.compile(r"^\d{8}_\d{6}(?:_\d{6})?_.+\.jpg$")


class EventLogger:
    """Writes event rows, optionally saves photos, and closes the session."""

    def __init__(self, config: DMSConfig) -> None:
        self.config = config
        self.event_dir = Path(config.event_dir)
        self.log_file = Path(config.log_file)
        self.event_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.session_start = datetime.now()
        self.counts: Counter[str] = Counter()
        self.photos: list[Path] = []
        self.last_event: dict[str, float] = {}
        self._rotated: Path | None = None
        self._prepare_log()

    # ------------------------------------------------------------------
    # Log lifecycle
    # ------------------------------------------------------------------
    def _prepare_log(self) -> None:
        if not self.log_file.exists():
            self._write_header()
            return

        try:
            with self.log_file.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                header = next(reader, [])
        except OSError:
            header = []

        if header == LOG_FIELDS:
            return
        if header == LEGACY_FIELDS:
            prefix = "dms_log_v1"
        elif header == V2_FIELDS:
            prefix = "dms_log_v2"
        elif header:
            prefix = "dms_log_previous"
        else:
            self._write_header()
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rotated = self.log_file.with_name(f"{prefix}_{stamp}.csv")
        try:
            shutil.move(str(self.log_file), str(rotated))
            self._rotated = rotated
        except OSError:
            pass
        self._write_header()

    def _write_header(self) -> None:
        with self.log_file.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(LOG_FIELDS)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def record(
        self,
        request: EventRequest,
        frame: np.ndarray | None = None,
        context: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> dict[str, Any] | None:
        """Record one event unless its cooldown is still active.

        ``context`` may contain any of: face_count, yaw, pitch, roll,
        eye_left, eye_right, perclos, blink_rate, closure_s, fps,
        latency_ms, status, safety_level, safety_reason, identity_state,
        identity_name, identity_score, gps.
        """
        import time

        now = time.monotonic() if now is None else float(now)
        context = {} if context is None else dict(context)
        cooldown = (
            self.config.critical_repeat
            if request.severity == "critical"
            else self.config.event_cooldown
        )
        previous = self.last_event.get(request.kind)
        if previous is not None and (now - previous) < cooldown:
            return None
        self.last_event[request.kind] = now

        stamp = datetime.now()
        photo_path = ""
        if frame is not None and frame.size:
            name = (
                f"{stamp.strftime('%Y%m%d_%H%M%S_%f')}_"
                f"{_safe(request.kind)}_{_safe(request.direction)}.jpg"
            )
            target = self.event_dir / name
            ok = cv2.imwrite(
                str(target),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 90],
            )
            if ok:
                photo_path = str(target)
                self.photos.append(target)

        row = {
            "timestamp": stamp.strftime("%Y-%m-%d %H:%M:%S"),
            "event": request.kind,
            "direction": request.direction,
            "face_count": int(context.get("face_count", 0)),
            "status": request.status,
            "severity": request.severity,
            "yaw": _round(context.get("yaw")),
            "pitch": _round(context.get("pitch")),
            "roll": _round(context.get("roll")),
            "eye_left": _round(context.get("eye_left")),
            "eye_right": _round(context.get("eye_right")),
            "perclos": _round(context.get("perclos")),
            "blink_rate": _round(context.get("blink_rate")),
            "closure_s": _round(context.get("closure_s")),
            "fps": _round(context.get("fps")),
            "latency_ms": _round(context.get("latency_ms")),
            "message": request.message,
            "photo": photo_path,
            "safety_level": context.get("safety_level", ""),
            "safety_reason": context.get("safety_reason", ""),
            "identity_state": context.get("identity_state", ""),
            "identity_name": context.get("identity_name") or "",
            "identity_score": _round(context.get("identity_score"), 3),
            "gps_lat": _round(_gps_value(context, "lat"), 6),
            "gps_lon": _round(_gps_value(context, "lon"), 6),
            "gps_text": _gps_text(context),
        }
        with self.log_file.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=LOG_FIELDS).writerow(row)

        self.counts[request.kind] += 1
        print(
            f"[EVENT] {request.kind:20s} {request.direction:12s} "
            f"{request.message}"
            + (f" -> {photo_path}" if photo_path else "")
        )
        if len(self.photos) % 20 == 0:
            self.prune()
        return row

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------
    def prune(self) -> int:
        """Delete the oldest event photos until the limits are met."""
        files = sorted(
            (
                p
                for p in self.event_dir.glob("*.jpg")
                if _photo_pattern().match(p.name)
            ),
            key=lambda p: p.stat().st_mtime,
        )
        removed = 0
        total = sum(p.stat().st_size for p in files)
        limit = int(self.config.max_event_mb * 1024 * 1024)
        while files and (
            len(files) > self.config.max_events or total > limit
        ):
            victim = files.pop(0)
            try:
                total -= victim.stat().st_size
                victim.unlink()
                removed += 1
            except OSError:
                pass
        return removed

    def close(self, summary: dict[str, Any] | None = None) -> Path | None:
        """Write the session JSON and prune old photos."""
        self.prune()
        if not self.config.save_session_summary:
            return None
        now = datetime.now()
        payload = {
            "session_start": self.session_start.isoformat(timespec="seconds"),
            "session_end": now.isoformat(timespec="seconds"),
            "duration_seconds": round(
                (now - self.session_start).total_seconds(), 2
            ),
            "event_counts": dict(self.counts),
            "photos": len(self.photos),
            "log_file": str(self.log_file),
            "rotated_legacy_log": str(self._rotated) if self._rotated else None,
            "summary": summary or {},
        }
        target = self.event_dir / (
            f"session_{self.session_start.strftime('%Y%m%d_%H%M%S')}.json"
        )
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[SESSION] {target}")
        return target


def _gps_value(context: dict[str, Any], key: str) -> Any:
    gps = context.get("gps")
    if isinstance(gps, dict):
        return gps.get(key)
    return None


def _gps_text(context: dict[str, Any]) -> str:
    gps = context.get("gps")
    if isinstance(gps, dict):
        return str(gps.get("text", ""))
    return ""


def _round(value: Any, digits: int = 2) -> Any:
    if value is None:
        return ""
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return ""


__all__ = ["EventLogger", "LEGACY_FIELDS", "LOG_FIELDS", "V2_FIELDS"]