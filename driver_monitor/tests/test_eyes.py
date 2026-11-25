"""Eye-state analysis: blink, closure, PERCLOS and blink rate."""

from __future__ import annotations

import unittest

from dms.config import DMSConfig
from dms.eyes import EyeAnalyzer
from dms.models import EyeObservation


def observation(left: float, right: float | None = None, valid: bool = True) -> EyeObservation:
    return EyeObservation(
        left_open=float(left),
        right_open=float(left if right is None else right),
        valid=valid,
        boxes=[],
    )


class EyeAnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DMSConfig()
        self.analyzer = EyeAnalyzer(self.config)

    def test_invalid_observation(self) -> None:
        metrics = self.analyzer.update(None, 0.0)
        self.assertFalse(metrics.valid)
        self.assertEqual(metrics.blink_count, 0)
        self.assertEqual(self.analyzer.update(observation(0.0, valid=False), 0.1).valid, False)

    def test_blink_counts_once(self) -> None:
        for t in (0.0, 0.1, 0.2):
            self.analyzer.update(observation(0.0), t)
        self.analyzer.update(observation(1.0), 0.3)
        metrics = self.analyzer.update(observation(1.0), 0.4)
        self.assertEqual(metrics.blink_count, 1)
        self.assertAlmostEqual(metrics.last_closure_seconds, 0.3, places=3)
        self.assertFalse(metrics.both_closed)

    def test_long_blink(self) -> None:
        self.analyzer.update(observation(0.0), 0.0)
        self.analyzer.update(observation(0.0), 1.0)
        self.analyzer.update(observation(1.0), 1.1)
        metrics = self.analyzer.update(observation(1.0), 1.2)
        self.assertEqual(metrics.long_blink_count, 1)
        self.assertEqual(metrics.blink_count, 0)

    def test_microsleep(self) -> None:
        self.analyzer.update(observation(0.0), 0.0)
        self.analyzer.update(observation(0.0), 3.1)
        self.analyzer.update(observation(1.0), 3.2)
        metrics = self.analyzer.update(observation(1.0), 3.3)
        self.assertEqual(metrics.microsleep_count, 1)
        self.assertGreaterEqual(metrics.last_closure_seconds, 3.0)

    def test_dropout_does_not_count_as_closure(self) -> None:
        self.analyzer.update(observation(0.0), 0.0)
        self.analyzer.update(observation(0.0), 0.5)
        invalid = self.analyzer.update(None, 2.0)
        self.assertFalse(invalid.valid)
        self.assertEqual(invalid.closure_seconds, 0.0)
        self.assertEqual(self.analyzer.blink_count, 0)
        self.analyzer.update(observation(1.0), 2.1)
        self.assertEqual(self.analyzer.blink_count, 0)

    def test_hysteresis_keeps_eye_closed(self) -> None:
        self.analyzer.update(observation(0.0), 0.0)
        self.analyzer.update(observation(0.55), 0.1)
        metrics = self.analyzer.update(observation(0.55), 0.2)
        self.assertTrue(metrics.both_closed)
        self.analyzer.update(observation(1.0), 0.3)
        metrics = self.analyzer.update(observation(1.0), 0.4)
        self.assertFalse(metrics.both_closed)

    def test_one_eye_closed_is_not_both_closed(self) -> None:
        self.analyzer.update(observation(0.0, 1.0), 0.0)
        metrics = self.analyzer.update(observation(0.0, 1.0), 0.1)
        self.assertTrue(metrics.left_closed)
        self.assertFalse(metrics.right_closed)
        self.assertFalse(metrics.both_closed)

    def test_perclos_and_drowsiness(self) -> None:
        for i in range(101):
            self.analyzer.update(observation(1.0), i * 0.2)
        metrics = self.analyzer.update(observation(1.0), 20.2)
        self.assertTrue(metrics.perclos_ready)
        self.assertAlmostEqual(metrics.perclos, 0.0, places=3)

        for i in range(80):
            metrics = self.analyzer.update(observation(0.0), 20.3 + i * 0.2)
        self.assertGreater(metrics.perclos, self.config.perclos_threshold)
        self.assertTrue(metrics.drowsy)
        self.assertTrue(metrics.both_closed)

    def test_blink_rate(self) -> None:
        self.analyzer.update(observation(0.0), 0.0)
        self.analyzer.update(observation(0.0), 0.1)
        self.analyzer.update(observation(1.0), 0.2)
        self.analyzer.update(observation(1.0), 0.3)
        metrics = self.analyzer.update(observation(1.0), 11.0)
        self.assertGreater(metrics.blink_rate, 0.0)
        self.assertLessEqual(metrics.blink_rate, 60.0)

    def test_perclos_does_not_latch_when_open(self) -> None:
        # Build up high PERCLOS with 30s of closed eyes
        for i in range(150):
            self.analyzer.update(observation(0.0), i * 0.2)
        # Now driver opens eyes
        self.analyzer.update(observation(1.0), 30.2)
        # Within 2s of closure ending, recent_closure may still be true
        # But after 2.5s of open eyes, drowsy MUST be False even if perclos >= 0.35
        metrics = self.analyzer.update(observation(1.0), 32.8)
        self.assertFalse(metrics.both_closed)
        self.assertFalse(metrics.drowsy)

    def test_perclos_decays_faster_when_open_3s(self) -> None:
        # Build up high PERCLOS
        for i in range(150):
            self.analyzer.update(observation(0.0), i * 0.2)
        # Eyes open for 4+ seconds continuously
        for i in range(25):
            metrics = self.analyzer.update(observation(1.0), 30.2 + i * 0.2)
        # PERCLOS buffer should have pruned older closed samples
        self.assertLess(metrics.perclos, 0.35)

    def test_reset(self) -> None:
        self.analyzer.update(observation(0.0), 0.0)
        self.analyzer.update(observation(1.0), 0.4)
        self.analyzer.reset()
        self.assertEqual(self.analyzer.blink_count, 0)
        metrics = self.analyzer.update(observation(1.0), 0.5)
        self.assertEqual(metrics.perclos, 0.0)
        self.assertFalse(metrics.perclos_ready)


if __name__ == "__main__":
    unittest.main()