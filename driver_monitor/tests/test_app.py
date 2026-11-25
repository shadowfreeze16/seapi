"""Application helpers that do not need a camera."""

from __future__ import annotations

import unittest

import numpy as np

from dms.app import BuzzerController, DriverMonitorApp, _config_snapshot, display_available
from dms.config import DMSConfig


class BuzzerTests(unittest.TestCase):
    def test_disabled_buzzer(self) -> None:
        buzzer = BuzzerController(enabled=False)
        self.assertFalse(buzzer.enabled)
        buzzer.update(alarm=True, now=0.0)
        buzzer.off()
        self.assertFalse(buzzer.toggle())

    def test_toggle_without_gpio(self) -> None:
        buzzer = BuzzerController(enabled=True)
        # On a desktop there is no gpiozero device; the controller disables
        # itself instead of raising.  On a Pi with GPIO it starts enabled.
        if buzzer.device is None:
            self.assertFalse(buzzer.toggle())
            self.assertFalse(buzzer.enabled)
        else:
            before = buzzer.enabled
            self.assertEqual(buzzer.toggle(), not before)
        buzzer.off()


class SnapshotTests(unittest.TestCase):
    def test_paths_are_strings(self) -> None:
        config = DMSConfig()
        snapshot = _config_snapshot(config)
        self.assertIsInstance(snapshot["model_dir"], str)
        self.assertIsInstance(snapshot["event_dir"], str)
        self.assertEqual(snapshot["width"], config.width)

    def test_display_available_is_bool(self) -> None:
        self.assertIsInstance(display_available(), bool)

    def test_toast(self) -> None:
        frame = np.zeros((120, 320, 3), dtype=np.uint8)
        DriverMonitorApp._draw_toast(frame, "TEST TOAST")
        self.assertGreater(int(frame.sum()), 0)

    def test_app_class_exists(self) -> None:
        self.assertTrue(callable(DriverMonitorApp))


if __name__ == "__main__":
    unittest.main()