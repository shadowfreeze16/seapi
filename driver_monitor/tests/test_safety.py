"""Three-level safety status: mapping, escalation and recovery."""

from __future__ import annotations

import unittest

from dms.config import DMSConfig
from dms.identity import IdentityResult, MISMATCH, MATCH
from dms.safety import (
    GREEN,
    RED,
    YELLOW,
    SafetyAssessor,
    level_for_state,
)
from dms.state import (
    CRITICAL,
    DISTRACTED,
    DROWSY,
    NORMAL,
    NO_FACE,
    SHORT_GLANCE,
    DriverState,
)


def driver(status: str, reason: str = "", zone: str = "FORWARD") -> DriverState:
    return DriverState(
        status=status,
        reason=reason or status,
        zone=zone,
        message=status,
        alert=status != NORMAL,
        alarm=status in (DISTRACTED, DROWSY, NO_FACE, CRITICAL),
    )


def identity(state: str, enforced: bool = True) -> IdentityResult:
    return IdentityResult(
        state=state,
        name="Alice" if state == MATCH else None,
        score=0.72 if state == MATCH else 0.12,
        enforced=enforced,
    )


class LevelMappingTests(unittest.TestCase):
    def test_mapping(self) -> None:
        self.assertEqual(level_for_state(NORMAL), GREEN)
        self.assertEqual(level_for_state(SHORT_GLANCE), GREEN)
        self.assertEqual(level_for_state(DISTRACTED), YELLOW)
        self.assertEqual(level_for_state(DROWSY), YELLOW)
        self.assertEqual(level_for_state(NO_FACE), YELLOW)
        self.assertEqual(level_for_state(CRITICAL), RED)


class SafetyAssessorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DMSConfig()
        self.config.safety_red_after = 8.0
        self.config.safety_red_hold = 1.5
        self.config.safety_green_hold = 2.0
        self.assessor = SafetyAssessor(self.config)

    def test_green_to_yellow(self) -> None:
        state = self.assessor.update(0.0, driver(NORMAL))
        self.assertEqual(state.level, GREEN)
        state = self.assessor.update(1.0, driver(DISTRACTED, "DISTRACTION"))
        self.assertEqual(state.level, YELLOW)
        self.assertEqual(state.reason, "DISTRACTION")
        self.assertTrue(state.alert)
        self.assertFalse(state.alarm)

    def test_critical_is_red(self) -> None:
        state = self.assessor.update(0.0, driver(CRITICAL, "MICROSLEEP"))
        self.assertEqual(state.level, RED)
        self.assertTrue(state.alarm)

    def test_persistent_yellow_becomes_red_and_latches(self) -> None:
        self.assessor.update(0.0, driver(DISTRACTED, "DISTRACTION"))
        state = self.assessor.update(4.0, driver(DISTRACTED, "DISTRACTION"))
        self.assertEqual(state.level, YELLOW)
        state = self.assessor.update(8.1, driver(DISTRACTED, "DISTRACTION"))
        self.assertEqual(state.level, RED)
        self.assertEqual(state.reason, "PERSISTENT_WARNING")
        # The underlying condition is still yellow, but the warning remains red.
        state = self.assessor.update(10.0, driver(DISTRACTED, "DISTRACTION"))
        self.assertEqual(state.level, RED)
        # Clearing the condition recovers through the hold time.
        state = self.assessor.update(11.0, driver(NORMAL))
        self.assertEqual(state.level, RED)
        state = self.assessor.update(12.6, driver(NORMAL))
        self.assertEqual(state.level, GREEN)

    def test_identity_mismatch_is_red(self) -> None:
        state = self.assessor.update(
            0.0, driver(NORMAL), identity(MISMATCH), face_present=True
        )
        self.assertEqual(state.level, RED)
        self.assertEqual(state.reason, "IDENTITY_MISMATCH")
        self.assertEqual(state.identity_state, MISMATCH)
        self.assertEqual(
            [event.kind for event in state.events], ["IDENTITY_MISMATCH"]
        )

    def test_identity_match_does_not_change_level(self) -> None:
        state = self.assessor.update(
            0.0, driver(NORMAL), identity(MATCH), face_present=True
        )
        self.assertEqual(state.level, GREEN)
        self.assertEqual(state.driver_name, "Alice")

    def test_recovery_event_emitted_once(self) -> None:
        self.assessor.update(0.0, driver(NORMAL))
        yellow = self.assessor.update(1.0, driver(DROWSY, "EYE_CLOSURE"))
        self.assertEqual(yellow.level, YELLOW)
        green = self.assessor.update(4.0, driver(NORMAL))
        self.assertEqual(green.level, YELLOW)
        green = self.assessor.update(6.1, driver(NORMAL))
        self.assertEqual(green.level, GREEN)
        self.assertTrue(green.changed)
        self.assertEqual([event.kind for event in green.events], ["SAFETY_RECOVERED"])
        again = self.assessor.update(7.0, driver(NORMAL))
        self.assertEqual(again.events, [])

    def test_zero_hold_recovers_immediately(self) -> None:
        self.config.safety_red_hold = 0.0
        self.assessor.update(0.0, driver(CRITICAL))
        state = self.assessor.update(0.1, driver(NORMAL))
        self.assertEqual(state.level, GREEN)


if __name__ == "__main__":
    unittest.main()
