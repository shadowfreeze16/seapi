"""Filters, smoothers and timing helpers."""

from __future__ import annotations

import unittest

from dms.filters import (
    AngleSmoother,
    EMA,
    LowPassFilter,
    OneEuroFilter,
    RateMeter,
    TimingStats,
)


class LowPassTests(unittest.TestCase):
    def test_initialization_and_smoothing(self) -> None:
        low = LowPassFilter(alpha=0.5)
        self.assertEqual(low(10.0), 10.0)
        self.assertEqual(low(20.0), 15.0)
        self.assertEqual(low(0.0, alpha=0.25), 11.25)

    def test_ema(self) -> None:
        ema = EMA(alpha=0.5)
        self.assertEqual(ema.update(10.0), 10.0)
        self.assertEqual(ema.update(20.0), 15.0)
        self.assertEqual(ema.update(0.0), 7.5)
        ema.reset()
        self.assertIsNone(ema.value)
        self.assertEqual(ema.update(3.0), 3.0)


class OneEuroTests(unittest.TestCase):
    def test_constant_signal_is_preserved(self) -> None:
        one = OneEuroFilter()
        for i in range(20):
            out = one(12.0, i / 30.0)
        self.assertAlmostEqual(out, 12.0, places=5)

    def test_step_is_smoothed(self) -> None:
        one = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        one(0.0, 0.0)
        out = one(30.0, 1.0 / 30.0)
        self.assertGreater(out, 0.0)
        self.assertLess(out, 30.0)

    def test_reset(self) -> None:
        one = OneEuroFilter()
        one(5.0, 0.0)
        one.reset()
        self.assertEqual(one(99.0, 1.0), 99.0)

    def test_angle_smoother(self) -> None:
        smoother = AngleSmoother()
        yaw, pitch, roll = smoother.update(10.0, -5.0, 2.0, 0.0)
        self.assertAlmostEqual(yaw, 10.0)
        self.assertAlmostEqual(pitch, -5.0)
        self.assertAlmostEqual(roll, 2.0)
        yaw, _, _ = smoother.update(370.0, 0.0, 0.0, 0.1)
        self.assertLess(abs(yaw), 180.0)
        smoother.reset()
        self.assertFalse(smoother.initialized)


class TimingTests(unittest.TestCase):
    def test_percentile(self) -> None:
        values = list(range(101))
        self.assertEqual(RateMeter.percentile(values, 0.0), 0)
        self.assertEqual(RateMeter.percentile(values, 0.5), 50)
        self.assertEqual(RateMeter.percentile(values, 1.0), 100)
        self.assertEqual(RateMeter.percentile([], 0.5), 0.0)

    def test_rate_meter(self) -> None:
        meter = RateMeter()
        self.assertEqual(meter.fps, 0.0)
        meter.tick()
        self.assertGreaterEqual(meter.frame_ms, 0.0)

    def test_timing_stats(self) -> None:
        stats = TimingStats("pose")
        self.assertEqual(stats.mean, 0.0)
        self.assertEqual(stats.p95, 0.0)
        for value in (1.0, 2.0, 3.0, 4.0):
            stats.add(value)
        stats.add(-1.0)  # ignored
        self.assertAlmostEqual(stats.mean, 2.5)
        self.assertEqual(stats.last, 4.0)
        self.assertGreaterEqual(stats.p95, 3.0)


if __name__ == "__main__":
    unittest.main()