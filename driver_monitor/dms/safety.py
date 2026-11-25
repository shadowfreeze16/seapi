"""Three-level driver safety status.

The existing :class:`dms.state.StateMachine` describes *why* the driver is
being warned (distraction, drowsiness, face loss, ...).  This module turns that
diagnosis plus the identity result into the operator-facing traffic-light
status:

``GREEN``   attentive
``YELLOW``  caution condition
``RED``     serious or persistent warning

The layer is deliberately independent of OpenCV so it can be tested with plain
objects.  It also owns the small amount of hysteresis needed to stop the badge
from flickering between levels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import DMSConfig
from .state import (
    CRITICAL,
    DISTRACTED,
    DROWSY,
    NORMAL,
    NO_FACE,
    SHORT_GLANCE,
    DriverState,
    EventRequest,
)

GREEN = "GREEN"
YELLOW = "YELLOW"
RED = "RED"

LEVEL_ORDER = {GREEN: 0, YELLOW: 1, RED: 2}
LEVEL_LABELS = {
    GREEN: "ATTENTIVE",
    YELLOW: "CAUTION",
    RED: "SERIOUS",
}

YELLOW_STATES = {DISTRACTED, DROWSY, NO_FACE}


@dataclass
class SafetyState:
    """Traffic-light status for one frame."""

    level: str = GREEN
    label: str = LEVEL_LABELS[GREEN]
    reason: str = "NORMAL"
    message: str = "Driver attentive"
    level_seconds: float = 0.0
    yellow_seconds: float = 0.0
    identity_state: str = "OFF"
    driver_name: str | None = None
    identity_score: float | None = None
    changed: bool = False
    events: list[EventRequest] = field(default_factory=list)

    @property
    def alert(self) -> bool:
        return self.level != GREEN

    @property
    def alarm(self) -> bool:
        return self.level == RED

    @property
    def color_key(self) -> str:
        return self.level.lower()


def level_for_state(status: str) -> str:
    """Base mapping before identity and persistence escalation."""
    if status == CRITICAL:
        return RED
    if status in YELLOW_STATES:
        return YELLOW
    return GREEN


def _base_reason(status: str) -> str:
    if status == CRITICAL:
        return "CRITICAL"
    if status == DISTRACTED:
        return "DISTRACTION"
    if status == DROWSY:
        return "DROWSINESS"
    if status == NO_FACE:
        return "NO_FACE"
    if status == SHORT_GLANCE:
        return "SHORT_GLANCE"
    return "NORMAL"


class SafetyAssessor:
    """Combines driver state, identity and persistence into GREEN/YELLOW/RED."""

    def __init__(self, config: DMSConfig) -> None:
        self.config = config
        self._level = GREEN
        self._reason = "NORMAL"
        self._message = "Driver attentive"
        self._level_since: float | None = None
        self._yellow_since: float | None = None
        self._downgrade_since: float | None = None
        self._downgrade_target: str | None = None

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------
    def update(
        self,
        now: float,
        driver: DriverState,
        identity: object | None = None,
        face_present: bool = True,
    ) -> SafetyState:
        now = float(now)
        if self._level_since is None:
            self._level_since = now

        target = level_for_state(driver.status)
        reason = _base_reason(driver.status)
        message = driver.message

        identity_state = getattr(identity, "state", "OFF") or "OFF"
        identity_enforced = bool(getattr(identity, "enforced", False))
        driver_name = getattr(identity, "name", None)
        identity_score = getattr(identity, "score", None)

        if identity_enforced and identity_state == "MISMATCH":
            target = RED
            reason = "IDENTITY_MISMATCH"
            message = "UNAUTHORIZED DRIVER"
        elif target == GREEN and driver.status == NORMAL:
            message = "Driver attentive"

        events: list[EventRequest] = []
        previous = self._level

        # A YELLOW condition that never clears becomes RED.  The timer only
        # runs while the traffic light is actually yellow, so a CRITICAL state
        # does not start a second independent countdown.
        if (
            target == YELLOW
            and self._level == YELLOW
            and self._yellow_since is not None
            and (now - self._yellow_since) >= self.config.safety_red_after
        ):
            target = RED
            reason = "PERSISTENT_WARNING"
            message = "SAFETY RED: PERSISTENT WARNING"

        # A persistent-warning RED stays latched while the underlying YELLOW
        # condition continues.  It is released only when the condition clears.
        if (
            self._level == RED
            and self._reason == "PERSISTENT_WARNING"
            and target == YELLOW
        ):
            target = RED
            reason = "PERSISTENT_WARNING"
            message = "SAFETY RED: PERSISTENT WARNING"

        if LEVEL_ORDER[target] > LEVEL_ORDER[self._level]:
            self._set_level(target, reason, message, now)
            self._downgrade_since = None
            self._downgrade_target = None
            if target == YELLOW:
                self._yellow_since = now
            elif target == RED:
                self._yellow_since = None
        elif target == self._level:
            self._reason = reason
            self._message = message
            self._downgrade_since = None
            self._downgrade_target = None
            if target == YELLOW and self._yellow_since is None:
                self._yellow_since = now
        else:
            # Downgrades are held briefly to avoid flicker.
            hold = (
                self.config.safety_red_hold
                if self._level == RED
                else self.config.safety_green_hold
            )
            if self._downgrade_target != target:
                self._downgrade_target = target
                self._downgrade_since = now
            if (
                self._downgrade_since is not None
                and (now - self._downgrade_since) >= hold
            ):
                self._set_level(target, reason, message, now)
                self._downgrade_since = None
                self._downgrade_target = None
                if target == YELLOW:
                    self._yellow_since = now
                elif target == GREEN:
                    self._yellow_since = None

        if self._level == GREEN:
            self._yellow_since = None

        changed = self._level != previous

        if changed and self._level == GREEN and previous in (YELLOW, RED):
            events.append(
                EventRequest(
                    kind="SAFETY_RECOVERED",
                    status=driver.status,
                    severity="info",
                    direction=driver.zone,
                    message="SAFETY GREEN: DRIVER ATTENTIVE",
                )
            )
        if changed and self._level == RED and self._reason == "IDENTITY_MISMATCH":
            events.append(
                EventRequest(
                    kind="IDENTITY_MISMATCH",
                    status=driver.status,
                    severity="critical",
                    direction=driver.zone,
                    message="UNAUTHORIZED DRIVER",
                )
            )
        if changed and self._level == RED and self._reason == "PERSISTENT_WARNING":
            events.append(
                EventRequest(
                    kind="SAFETY_RED",
                    status=driver.status,
                    severity="critical",
                    direction=driver.zone,
                    message=self._message,
                )
            )

        level_seconds = max(0.0, now - self._level_since)
        yellow_seconds = (
            max(0.0, now - self._yellow_since)
            if self._yellow_since is not None
            else 0.0
        )

        return SafetyState(
            level=self._level,
            label=LEVEL_LABELS[self._level],
            reason=self._reason,
            message=self._message,
            level_seconds=level_seconds,
            yellow_seconds=yellow_seconds,
            identity_state=identity_state,
            driver_name=driver_name,
            identity_score=identity_score,
            changed=changed,
            events=events,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _set_level(
        self, level: str, reason: str, message: str, now: float
    ) -> None:
        self._level = level
        self._reason = reason
        self._message = message
        self._level_since = now

    def reset(self) -> None:
        self._level = GREEN
        self._reason = "NORMAL"
        self._message = "Driver attentive"
        self._level_since = None
        self._yellow_since = None
        self._downgrade_since = None
        self._downgrade_target = None


__all__ = [
    "GREEN",
    "LEVEL_LABELS",
    "LEVEL_ORDER",
    "RED",
    "SafetyAssessor",
    "SafetyState",
    "YELLOW",
    "level_for_state",
]
