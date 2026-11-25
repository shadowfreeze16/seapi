"""Spoken warnings through ``espeak-ng``.

Speech must never stall the video loop, so the warner owns a small background
worker that plays one phrase at a time.  Phrases are selected from the
traffic-light status and the underlying driver state, and repeated phrases are
rate-limited.

The module degrades quietly: if the command is missing or the feature is off,
every call is a no-op.  ``voice_test`` is available for the command-line
``--voice-test`` switch.
"""

from __future__ import annotations

import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass

from .config import DMSConfig
from .safety import GREEN, RED, YELLOW, SafetyState
from .state import (
    CRITICAL,
    DISTRACTED,
    DROWSY,
    NO_FACE,
)

# Reason strings produced by StateMachine and SafetyAssessor.
MICROSLEEP = "MICROSLEEP"
NO_FACE_CRITICAL = "NO_FACE_CRITICAL"
EXTREME_ATTENTION = "EXTREME_ATTENTION"
IDENTITY_MISMATCH = "IDENTITY_MISMATCH"
PERSISTENT_WARNING = "PERSISTENT_WARNING"
DISTRACTION = "DISTRACTION"
DROWSINESS = "DROWSINESS"

YELLOW_DISTRACTION = "Please keep your eyes on the road."
YELLOW_DROWSY = "You seem drowsy. Please take a break."
YELLOW_NO_FACE = "Driver not visible. Please sit in front of the camera."
RED_MICROSLEEP = "Wake up. You are falling asleep."
RED_NO_FACE = "Driver not visible. Pull over safely."
RED_DISTRACTION = "Eyes on the road. Serious distraction."
RED_IDENTITY = "Unauthorized driver detected."
RED_PERSISTENT = "Warning persists. Please refocus on driving."
RED_GENERIC = "Critical driver warning."
RECOVERED = "Thank you. Attention restored."
SETUP_COMPLETE = "Camera setup complete. Driver monitoring is active."
ENROLLED = "Driver profile saved."

TEST_PHRASES = (
    "Driver monitoring voice test.",
    "Please keep your eyes on the road.",
    "Wake up. You are falling asleep.",
    "Camera setup complete.",
)


@dataclass(frozen=True)
class Phrase:
    key: str
    text: str


def phrase_for(
    safety: SafetyState,
    driver_reason: str | None = None,
    driver_status: str | None = None,
) -> Phrase | None:
    """Pick the phrase for the current status, or ``None`` for silence."""
    if safety.level == RED:
        reason = safety.reason
        if reason == IDENTITY_MISMATCH:
            return Phrase("red:identity", RED_IDENTITY)
        if reason == PERSISTENT_WARNING:
            return Phrase("red:persistent", RED_PERSISTENT)
        if driver_reason == MICROSLEEP:
            return Phrase("red:microsleep", RED_MICROSLEEP)
        if driver_reason == NO_FACE_CRITICAL:
            return Phrase("red:no_face", RED_NO_FACE)
        if driver_reason == EXTREME_ATTENTION:
            return Phrase("red:distraction", RED_DISTRACTION)
        if driver_status == CRITICAL:
            return Phrase("red:critical", RED_GENERIC)
        return Phrase("red:generic", RED_GENERIC)

    if safety.level == YELLOW:
        if driver_status == NO_FACE or driver_reason == NO_FACE:
            return Phrase("yellow:no_face", YELLOW_NO_FACE)
        if driver_status == DROWSY or driver_reason == DROWSINESS:
            return Phrase("yellow:drowsy", YELLOW_DROWSY)
        if driver_status == DISTRACTED or driver_reason == DISTRACTION:
            return Phrase("yellow:distraction", YELLOW_DISTRACTION)
        return Phrase("yellow:generic", "Caution. Please stay focused.")

    return None


class VoiceWarner:
    """Non-blocking, rate-limited espeak-ng announcer."""

    def __init__(self, config: DMSConfig) -> None:
        self.config = config
        self.enabled = bool(config.voice_enabled)
        self.command = shutil.which(config.voice_command)
        self.available = self.command is not None
        self.last_error: str | None = None
        self._queue: queue.Queue[str] = queue.Queue(maxsize=4)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_spoken: dict[str, float] = {}
        self._last_any = 0.0
        self._previous_level: str | None = None

        if self.enabled and not self.available:
            print(
                f"[VOICE] disabled: '{config.voice_command}' not found. "
                "Install it with: sudo apt install espeak-ng"
            )
            self.enabled = False
        if self.enabled:
            self._thread = threading.Thread(
                target=self._worker,
                name="dms-voice",
                daemon=True,
            )
            self._thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def active(self) -> bool:
        return bool(self.enabled and self.available)

    def update(
        self,
        now: float,
        safety: SafetyState,
        driver_reason: str | None = None,
        driver_status: str | None = None,
    ) -> Phrase | None:
        """Announce a new warning or a recovery.  Returns the chosen phrase."""
        if not self.active:
            return None
        previous = self._previous_level
        self._previous_level = safety.level

        if safety.level == GREEN:
            if previous in (YELLOW, RED):
                phrase = Phrase("recovered", RECOVERED)
                self._announce(phrase, now)
                return phrase
            return None

        phrase = phrase_for(safety, driver_reason, driver_status)
        if phrase is None:
            return None
        self._announce(phrase, now)
        return phrase

    def announce_setup_complete(self, now: float | None = None) -> None:
        self._announce(Phrase("setup", SETUP_COMPLETE), now)

    def announce_enrolled(self, now: float | None = None) -> None:
        self._announce(Phrase("enrolled", ENROLLED), now)

    def announce(self, text: str, key: str = "manual", now: float | None = None) -> None:
        self._announce(Phrase(key, text), now)

    def shutdown(self, timeout: float = 1.0) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait("")
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _announce(self, phrase: Phrase, now: float | None = None) -> bool:
        if not self.active:
            return False
        current = time.monotonic() if now is None else float(now)
        last = self._last_spoken.get(phrase.key)
        if last is not None and (current - last) < self.config.voice_cooldown:
            return False
        if (current - self._last_any) < 0.5 and last is None:
            return False
        self._last_spoken[phrase.key] = current
        self._last_any = current
        try:
            self._queue.put_nowait(phrase.text)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(phrase.text)
            except queue.Full:
                return False
        return True

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                text = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if not text:
                break
            self._speak(text)

    def _speak(self, text: str) -> None:
        if self.command is None:
            return
        command = [
            self.command,
            "-v",
            self.config.voice_lang,
            "-s",
            str(int(self.config.voice_rate)),
            "-a",
            str(int(self.config.voice_volume)),
            text,
        ]
        try:
            subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15.0,
                check=False,
            )
            self.last_error = None
        except (OSError, subprocess.SubprocessError) as exc:
            self.last_error = str(exc)
            print(f"[VOICE] {exc}")


def voice_test(config: DMSConfig, phrases: tuple[str, ...] = TEST_PHRASES) -> int:
    """Speak a few sample phrases synchronously (used by ``--voice-test``)."""
    command = shutil.which(config.voice_command)
    if command is None:
        print(
            f"[VOICE] '{config.voice_command}' not found.\n"
            "Install it with:\n"
            "    sudo apt install espeak-ng",
        )
        return 1
    print(f"[VOICE] using {command}")
    for text in phrases:
        print(f"  speaking: {text}")
        try:
            subprocess.run(
                [
                    command,
                    "-v",
                    config.voice_lang,
                    "-s",
                    str(int(config.voice_rate)),
                    "-a",
                    str(int(config.voice_volume)),
                    text,
                ],
                timeout=20.0,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"[VOICE] failed: {exc}")
            return 1
    print("[VOICE] test complete")
    return 0


__all__ = [
    "Phrase",
    "TEST_PHRASES",
    "VoiceWarner",
    "phrase_for",
    "voice_test",
]
