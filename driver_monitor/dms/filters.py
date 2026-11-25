"""Signal filters and timing helpers.

The One Euro filter is a lightweight adaptive low-pass filter.  It removes
jitter when the head is still while keeping latency low during fast movements,
which is exactly what a head-pose overlay needs.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field

from .geometry import wrap_angle_deg


class LowPassFilter:
    def __init__(self, alpha: float) -> None:
        self.alpha = float(alpha)
        self.initialized = False
        self.value = 0.0

    def __call__(self, value: float, alpha: float | None = None) -> float:
        a = self.alpha if alpha is None else float(alpha)
        if not self.initialized:
            self.value = float(value)
            self.initialized = True
        else:
            self.value = a * float(value) + (1.0 - a) * self.value
        return self.value


class OneEuroFilter:
    """One Euro filter for a scalar signal.

    Reference: Casiez, Roussel & Vogel (CHI 2012).
    """

    def __init__(
        self,
        freq: float = 30.0,
        min_cutoff: float = 1.4,
        beta: float = 0.015,
        d_cutoff: float = 1.0,
    ) -> None:
        self.freq = max(1e-6, float(freq))
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_filter = LowPassFilter(1.0)
        self.dx_filter = LowPassFilter(1.0)
        self._last_value: float | None = None
        self._last_time: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def __call__(self, value: float, timestamp: float | None = None) -> float:
        now = time.monotonic() if timestamp is None else float(timestamp)
        if self._last_time is None:
            dt = 1.0 / self.freq
        else:
            dt = max(now - self._last_time, 1e-4)
        self._last_time = now
        self.freq = 1.0 / dt

        if self._last_value is None:
            dx = 0.0
        else:
            dx = (float(value) - self._last_value) / dt
        edx = self.dx_filter(dx, self._alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        filtered = self.x_filter(float(value), self._alpha(cutoff, dt))
        self._last_value = float(value)
        return float(filtered)

    def reset(self) -> None:
        self.x_filter = LowPassFilter(1.0)
        self.dx_filter = LowPassFilter(1.0)
        self._last_value = None
        self._last_time = None


class AngleSmoother:
    """One Euro filters for yaw, pitch and roll, with angle wrapping."""

    def __init__(
        self,
        min_cutoff: float = 1.4,
        beta: float = 0.015,
        d_cutoff: float = 1.0,
    ) -> None:
        self.filters = [
            OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
            for _ in range(3)
        ]
        self.initialized = False

    def update(
        self, yaw: float, pitch: float, roll: float, timestamp: float
    ) -> tuple[float, float, float]:
        yaw = wrap_angle_deg(yaw)
        pitch = wrap_angle_deg(pitch)
        roll = wrap_angle_deg(roll)
        out = (
            self.filters[0](yaw, timestamp),
            self.filters[1](pitch, timestamp),
            self.filters[2](roll, timestamp),
        )
        self.initialized = True
        return out

    def reset(self) -> None:
        for f in self.filters:
            f.reset()
        self.initialized = False


class EMA:
    """Exponential moving average."""

    def __init__(self, alpha: float = 0.3) -> None:
        self.alpha = float(alpha)
        self.value: float | None = None

    def update(self, value: float) -> float:
        if self.value is None:
            self.value = float(value)
        else:
            self.value = (
                self.alpha * float(value) + (1.0 - self.alpha) * self.value
            )
        return self.value

    def reset(self) -> None:
        self.value = None


@dataclass
class RateMeter:
    """Measures frames per second and per-stage timings."""

    fps: float = 0.0
    _frames: int = 0
    _started: float = field(default_factory=time.monotonic)
    _last: float = field(default_factory=time.monotonic)
    _samples: deque[float] = field(default_factory=lambda: deque(maxlen=120))

    def tick(self) -> float:
        now = time.monotonic()
        self._frames += 1
        self._samples.append(now - self._last)
        self._last = now
        elapsed = now - self._started
        if elapsed >= 0.5:
            self.fps = self._frames / elapsed
            self._frames = 0
            self._started = now
        return self.fps

    @property
    def frame_ms(self) -> float:
        if not self._samples:
            return 0.0
        return 1000.0 * sum(self._samples) / len(self._samples)

    @staticmethod
    def percentile(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        values = sorted(values)
        idx = int(round((len(values) - 1) * p))
        return values[max(0, min(len(values) - 1, idx))]


@dataclass
class TimingStats:
    """Rolling latency statistics for one pipeline stage."""

    name: str
    samples: deque[float] = field(default_factory=lambda: deque(maxlen=90))

    def add(self, milliseconds: float) -> None:
        if milliseconds >= 0:
            self.samples.append(float(milliseconds))

    @property
    def mean(self) -> float:
        return sum(self.samples) / len(self.samples) if self.samples else 0.0

    @property
    def last(self) -> float:
        return self.samples[-1] if self.samples else 0.0

    @property
    def p95(self) -> float:
        if not self.samples:
            return 0.0
        values = sorted(self.samples)
        idx = int(round((len(values) - 1) * 0.95))
        return values[idx]