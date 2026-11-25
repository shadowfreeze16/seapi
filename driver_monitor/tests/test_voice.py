"""Voice phrase selection, rate limiting and graceful degradation."""

from __future__ import annotations

import unittest

from dms.config import DMSConfig
from dms.safety import GREEN, RED, YELLOW, SafetyState
from dms.state import CRITICAL, DISTRACTED, DROWSY, NO_FACE
from dms.voice import (
    RED_IDENTITY,
    RED_MICROSLEEP,
    RED_NO_FACE,
    TEST_PHRASES,
    VoiceWarner,
    phrase_for,
    voice_test,
)


def safety(level: str, reason: str) -> SafetyState:
    return SafetyState(level=level, reason=reason, message=reason)


class PhraseTests(unittest.TestCase):
    def test_yellow_phrases(self) -> None:
        self.assertIn(
            "eyes on the road",
            phrase_for(safety(YELLOW, "DISTRACTION"), "DISTRACTION", DISTRACTED).text,
        )
        self.assertIn(
            "drowsy",
            phrase_for(safety(YELLOW, "DROWSINESS"), "EYE_CLOSURE", DROWSY).text,
        )
        self.assertIn(
            "not visible",
            phrase_for(safety(YELLOW, "NO_FACE"), "NO_FACE", NO_FACE).text,
        )

    def test_red_phrases(self) -> None:
        self.assertEqual(
            phrase_for(safety(RED, "CRITICAL"), "MICROSLEEP", CRITICAL).text,
            RED_MICROSLEEP,
        )
        self.assertEqual(
            phrase_for(safety(RED, "CRITICAL"), "NO_FACE_CRITICAL", CRITICAL).text,
            RED_NO_FACE,
        )
        self.assertEqual(
            phrase_for(safety(RED, "IDENTITY_MISMATCH"), "FORWARD", "NORMAL").text,
            RED_IDENTITY,
        )
        self.assertIn(
            "persists",
            phrase_for(safety(RED, "PERSISTENT_WARNING"), "DISTRACTION", DISTRACTED).text,
        )

    def test_green_is_silent(self) -> None:
        self.assertIsNone(phrase_for(safety(GREEN, "NORMAL"), "FORWARD", "NORMAL"))


class VoiceWarnerTests(unittest.TestCase):
    def test_disabled_warner(self) -> None:
        config = DMSConfig()
        config.voice_enabled = False
        warner = VoiceWarner(config)
        self.assertFalse(warner.active)
        self.assertIsNone(warner.update(1.0, safety(YELLOW, "DISTRACTION")))
        warner.shutdown()

    def test_missing_command_disables_cleanly(self) -> None:
        config = DMSConfig()
        config.voice_enabled = True
        config.voice_command = "dms-command-that-does-not-exist"
        warner = VoiceWarner(config)
        self.assertFalse(warner.active)
        self.assertFalse(warner.enabled)
        warner.shutdown()

    def test_cooldown_and_recovery(self) -> None:
        config = DMSConfig()
        config.voice_enabled = True
        config.voice_command = "/bin/true"
        config.voice_cooldown = 10.0
        warner = VoiceWarner(config)
        self.assertTrue(warner.active)
        yellow = safety(YELLOW, "DISTRACTION")
        first = warner.update(100.0, yellow, "DISTRACTION", DISTRACTED)
        self.assertIsNotNone(first)
        self.assertTrue(
            warner._announce(first, 101.0) is False,
            "the same phrase must respect the cooldown",
        )
        recovered = warner.update(103.0, safety(GREEN, "NORMAL"))
        self.assertEqual(recovered.key, "recovered")
        warner.shutdown()

    def test_voice_test_without_command(self) -> None:
        config = DMSConfig()
        config.voice_command = "dms-command-that-does-not-exist"
        self.assertEqual(voice_test(config, ("hello",)), 1)

    def test_phrases_are_non_empty(self) -> None:
        for phrase in TEST_PHRASES:
            self.assertTrue(phrase.strip().endswith("."))


if __name__ == "__main__":
    unittest.main()
