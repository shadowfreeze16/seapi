"""Driver attention and drowsiness state machine.

The state machine is deliberately kept independent of OpenCV and the models so
that it can be unit-tested with synthetic poses and eye metrics.

States
------
``NORMAL``       forward attention, eyes open
``SHORT_GLANCE`` brief look away (allowed)
``DISTRACTED``   sustained off-road attention
``DROWSY``       long closure / high PERCLOS
``NO_FACE``      driver not visible
``CRITICAL``     microsleep, extreme distraction or prolonged face loss

Priority (highest first): face loss, microsleep, extreme distraction,
drowsiness, distraction, short glance, normal.

Event requests are emitted on state transitions and periodically while a
critical state persists.  The event logger applies its own cooldown and writes
the CSV/photo, so the state machine never touches the filesystem.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .config import DMSConfig
from .eyes import EyeMetrics
from .geometry import clamp, wrap_angle_deg


# State constants (plain strings keep logging and UI code simple).
NORMAL = "NORMAL"
SHORT_GLANCE = "SHORT_GLANCE"
DISTRACTED = "DISTRACTED"
DROWSY = "DROWSY"
NO_FACE = "NO_FACE"
CRITICAL = "CRITICAL"

FORWARD = "FORWARD"

ALERT_STATES = {SHORT_GLANCE, DISTRACTED, DROWSY, NO_FACE, CRITICAL}
# Short glances stay silent; sustained distraction, face loss, drowsiness and
# critical states sound the buzzer.  This matches the v1 behaviour where a
# warning produced a short beep.
ALARM_STATES = {DISTRACTED, DROWSY, NO_FACE, CRITICAL}


@dataclass(frozen=True)
class EventRequest:
    """A request to log an event.  The logger decides whether to keep it."""

    kind: str
    status: str
    severity: str
    direction: str
    message: str


@dataclass
class DriverState:
    status: str = NORMAL
    zone: str = FORWARD
    reason: str = "FORWARD"
    message: str = "Attention OK"
    alert: bool = False
    alarm: bool = False
    offroad_seconds: float = 0.0
    forward_seconds: float = 0.0
    no_face_seconds: float = 0.0
    status_seconds: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    pose_valid: bool = False
    events: list[EventRequest] = field(default_factory=list)

    @property
    def severity(self) -> str:
        if self.status == CRITICAL:
            return "critical"
        if self.status in (DISTRACTED, DROWSY, NO_FACE):
            return "warning"
        if self.status == SHORT_GLANCE:
            return "info"
        return "ok"


class ZoneTracker:
    """Debounces the coarse gaze zone so the label does not flicker."""

    def __init__(
        self,
        yaw_glance: float,
        pitch_glance: float,
        debounce: float = 0.15,
    ) -> None:
        self.yaw_glance = float(yaw_glance)
        self.pitch_glance = float(pitch_glance)
        self.debounce = max(0.0, float(debounce))
        self.zone = FORWARD
        self._candidate: str | None = None
        self._candidate_since = 0.0

    def raw_zone(self, yaw: float, pitch: float) -> str:
        horizontal = ""
        vertical = ""
        if yaw >= self.yaw_glance:
            horizontal = "LEFT"
        elif yaw <= -self.yaw_glance:
            horizontal = "RIGHT"
        if pitch >= self.pitch_glance:
            vertical = "UP"
        elif pitch <= -self.pitch_glance:
            vertical = "DOWN"
        if horizontal and vertical:
            return f"{horizontal}_{vertical}"
        if horizontal:
            return horizontal
        if vertical:
            return vertical
        return FORWARD

    def update(self, yaw: float, pitch: float, now: float) -> str:
        raw = self.raw_zone(yaw, pitch)
        if raw == self.zone:
            self._candidate = None
            return self.zone

        if self._candidate != raw:
            self._candidate = raw
            self._candidate_since = now
        if (now - self._candidate_since) >= self.debounce:
            self.zone = raw
            self._candidate = None
        return self.zone

    def reset(self) -> None:
        self.zone = FORWARD
        self._candidate = None
        self._candidate_since = 0.0


class StateMachine:
    """Attention + drowsiness state machine with calibration support."""

    def __init__(self, config: DMSConfig) -> None:
        self.config = config
        self.zone_tracker = ZoneTracker(
            config.yaw_glance,
            config.pitch_glance,
            debounce=config.offroad_debounce,
        )
        self.yaw_offset = 0.0
        self.pitch_offset = 0.0
        self.roll_offset = 0.0
        self.calibrated = False
        self._offroad_start: float | None = None
        self._forward_start: float | None = None
        self._forward_since: float | None = None
        self._no_face_start: float | None = None
        self._status_since = 0.0
        self._last_emit = 0.0
        self._last_emit_kind = ""
        self._no_face_event_sent = False
        self._status = NORMAL
        self._reason = "FORWARD"
        self._started = False

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def calibrate(self, yaw: float, pitch: float, roll: float) -> None:
        self.yaw_offset = wrap_angle_deg(yaw)
        self.pitch_offset = wrap_angle_deg(pitch)
        self.roll_offset = wrap_angle_deg(roll)
        self.calibrated = True

    def clear_calibration(self) -> None:
        self.yaw_offset = 0.0
        self.pitch_offset = 0.0
        self.roll_offset = 0.0
        self.calibrated = False

    def calibration_dict(self) -> dict[str, float | bool]:
        return {
            "yaw": self.yaw_offset,
            "pitch": self.pitch_offset,
            "roll": self.roll_offset,
            "calibrated": self.calibrated,
        }

    def save_calibration(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.calibration_dict()
        payload["version"] = 2
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def load_calibration(self, path: Path | str) -> bool:
        path = Path(path)
        if not path.exists():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.yaw_offset = float(payload.get("yaw", 0.0))
            self.pitch_offset = float(payload.get("pitch", 0.0))
            self.roll_offset = float(payload.get("roll", 0.0))
            self.calibrated = bool(payload.get("calibrated", True))
            return True
        except (OSError, ValueError, TypeError):
            return False

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------
    def update(
        self,
        now: float,
        yaw: float,
        pitch: float,
        roll: float,
        eyes: EyeMetrics,
        face_present: bool,
        pose_valid: bool = True,
    ) -> DriverState:
        now = float(now)
        events: list[EventRequest] = []

        if not self._started:
            self._started = True
            self._status_since = now

        yaw_adj = wrap_angle_deg(yaw - self.yaw_offset) if pose_valid else 0.0
        pitch_adj = (
            wrap_angle_deg(pitch - self.pitch_offset) if pose_valid else 0.0
        )
        roll_adj = wrap_angle_deg(roll - self.roll_offset) if pose_valid else 0.0

        # ------------------------------------------------------------------
        # Face presence timers
        # ------------------------------------------------------------------
        if face_present:
            self._no_face_start = None
            self._no_face_event_sent = False
            no_face_seconds = 0.0
        else:
            if self._no_face_start is None:
                self._no_face_start = now
            no_face_seconds = max(0.0, now - self._no_face_start)
            self._offroad_start = None
            self._forward_start = None
            self._forward_since = None

        # ------------------------------------------------------------------
        # Gaze zone and off-road timer
        # ------------------------------------------------------------------
        zone = self.zone_tracker.update(yaw_adj, pitch_adj, now) if (
            face_present and pose_valid
        ) else (FORWARD if not face_present else "UNKNOWN")
        if not face_present:
            zone = "NO_FACE"
        elif not pose_valid:
            zone = "UNKNOWN"

        offroad_seconds = 0.0
        forward_seconds = 0.0
        if face_present and pose_valid:
            if zone != FORWARD:
                self._forward_start = None
                self._forward_since = None
                if self._offroad_start is None:
                    self._offroad_start = now
                offroad_seconds = max(0.0, now - self._offroad_start)
            else:
                if self._forward_since is None:
                    self._forward_since = now
                if self._offroad_start is not None:
                    if self._forward_start is None:
                        self._forward_start = now
                    if (
                        now - self._forward_start
                        >= self.config.forward_debounce
                    ):
                        self._offroad_start = None
                        self._forward_start = None
                    else:
                        offroad_seconds = max(
                            0.0, now - self._offroad_start
                        )
                forward_seconds = max(0.0, now - self._forward_since)
        else:
            self._offroad_start = None
            self._forward_start = None
            self._forward_since = None

        # ------------------------------------------------------------------
        # Decide status
        # ------------------------------------------------------------------
        status = NORMAL
        reason = "FORWARD"
        message = "Attention OK"

        # Eye closure is meaningful even when the head-pose model fails, so it
        # must not be gated on pose_valid.
        eye_critical = bool(
            face_present and eyes.valid and eyes.critical
        )
        eye_drowsy = bool(
            face_present and eyes.valid and eyes.drowsy
        )
        extreme = bool(
            pose_valid
            and (
                abs(yaw_adj) >= self.config.yaw_extreme
                or abs(pitch_adj) >= self.config.pitch_extreme
            )
        )
        attention_critical = bool(
            offroad_seconds >= self.config.distraction_critical_seconds
            or (
                extreme
                and offroad_seconds >= self.config.extreme_seconds
            )
        )

        if not face_present:
            if no_face_seconds >= self.config.no_face_critical_seconds:
                status = CRITICAL
                reason = "NO_FACE_CRITICAL"
                message = "CRITICAL: DRIVER NOT VISIBLE"
            else:
                status = NO_FACE
                reason = "NO_FACE"
                message = "NO DRIVER FACE"
        elif eye_critical:
            status = CRITICAL
            reason = "MICROSLEEP"
            message = "CRITICAL: MICROSLEEP"
        elif attention_critical:
            status = CRITICAL
            reason = "EXTREME_ATTENTION"
            message = "CRITICAL: EXTREME DISTRACTION"
        elif eye_drowsy:
            status = DROWSY
            reason = (
                "PERCLOS"
                if (eyes.perclos_ready and not eyes.both_closed)
                else "EYE_CLOSURE"
            )
            message = "DROWSINESS WARNING"
        elif offroad_seconds >= self.config.distraction_seconds:
            status = DISTRACTED
            reason = "DISTRACTION"
            message = "DISTRACTION WARNING"
        elif zone not in (FORWARD, "UNKNOWN"):
            status = SHORT_GLANCE
            reason = "GLANCE"
            message = "SHORT GLANCE"
        elif face_present and pose_valid:
            status = NORMAL
            reason = "FORWARD"
            message = "Attention OK"
        elif face_present and not pose_valid:
            status = NORMAL
            reason = "POSE_UNAVAILABLE"
            message = "HEAD POSE UNAVAILABLE"

        if status != self._status or reason != self._reason:
            self._status_since = now

        status_seconds = max(0.0, now - self._status_since)

        # ------------------------------------------------------------------
        # Event requests
        # ------------------------------------------------------------------
        transition = status != self._status or reason != self._reason

        def emit(kind: str, severity: str, text: str) -> None:
            events.append(
                EventRequest(
                    kind=kind,
                    status=status,
                    severity=severity,
                    direction=zone,
                    message=text,
                )
            )

        if transition and status == DISTRACTED:
            emit("SIDE_ATTENTION", "warning", message)
        elif transition and status == DROWSY:
            emit("DROWSY", "warning", message)
        elif transition and status == CRITICAL:
            if reason == "MICROSLEEP":
                emit("MICROSLEEP", "critical", message)
            elif reason == "NO_FACE_CRITICAL":
                emit("NO_FACE_CRITICAL", "critical", message)
            else:
                emit("EXTREME_ATTENTION", "critical", message)

        if (
            status == NO_FACE
            and no_face_seconds >= self.config.no_face_seconds
            and not self._no_face_event_sent
        ):
            emit("NO_FACE", "warning", message)
            self._no_face_event_sent = True

        # Repeat alarms while a critical/drowsy state persists.
        if status in (CRITICAL, DROWSY):
            if transition or (
                now - self._last_emit >= self.config.critical_repeat
            ):
                if events:
                    self._last_emit = now
                    self._last_emit_kind = events[-1].kind
                else:
                    kind = (
                        "CRITICAL"
                        if status == CRITICAL
                        else "DROWSY"
                    )
                    emit(kind, status.lower(), message)
                    self._last_emit = now
                    self._last_emit_kind = kind
        elif transition:
            self._last_emit = now
            self._last_emit_kind = events[-1].kind if events else ""

        self._status = status
        self._reason = reason

        return DriverState(
            status=status,
            zone=zone,
            reason=reason,
            message=message,
            alert=status in ALERT_STATES,
            alarm=status in ALARM_STATES,
            offroad_seconds=offroad_seconds,
            forward_seconds=forward_seconds,
            no_face_seconds=no_face_seconds,
            status_seconds=status_seconds,
            yaw=yaw_adj,
            pitch=pitch_adj,
            roll=roll_adj,
            pose_valid=pose_valid,
            events=events,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.zone_tracker.reset()
        self._offroad_start = None
        self._forward_start = None
        self._forward_since = None
        self._no_face_start = None
        self._status_since = 0.0
        self._last_emit = 0.0
        self._last_emit_kind = ""
        self._no_face_event_sent = False
        self._status = NORMAL
        self._reason = "FORWARD"
        self._started = False


def attention_score(yaw: float, pitch: float, config: DMSConfig) -> float:
    """A simple 0..1 attention score for dashboards/tests.

    1.0 means centred, 0.0 means at or beyond the critical limits.  The score
    is the smaller of the horizontal and vertical normalised margins.
    """
    horizontal = 1.0 - clamp(
        abs(yaw) / max(config.yaw_extreme, 1e-6), 0.0, 1.0
    )
    vertical = 1.0 - clamp(
        abs(pitch) / max(config.pitch_extreme, 1e-6), 0.0, 1.0
    )
    return float(min(horizontal, vertical))


__all__ = [
    "ALERT_STATES",
    "ALARM_STATES",
    "CRITICAL",
    "DISTRACTED",
    "DROWSY",
    "DriverState",
    "EventRequest",
    "FORWARD",
    "NORMAL",
    "NO_FACE",
    "SHORT_GLANCE",
    "StateMachine",
    "ZoneTracker",
    "attention_score",
]