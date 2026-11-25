"""Eye-state analysis: blinks, closures, PERCLOS and blink rate.

OCEC gives a per-eye probability of being open.  Raw probabilities flicker a
little, so this module adds:

* a small EMA on the probabilities,
* hysteresis (separate enter/exit thresholds),
* blink and long-blink detection,
* continuous-closure timing,
* PERCLOS over a rolling window (the fraction of time the eyes are closed),
* blink rate per minute.

All timestamps are seconds from any monotonic clock; unit tests pass explicit
values, the application uses the camera/video clock.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .config import DMSConfig
from .filters import EMA
from .models import EyeObservation


@dataclass
class EyeMetrics:
    """Derived eye / drowsiness metrics for one frame."""

    valid: bool = False
    left_open: float = 0.0
    right_open: float = 0.0
    left_closed: bool = False
    right_closed: bool = False
    both_closed: bool = False
    closure_seconds: float = 0.0
    last_closure_seconds: float = 0.0
    blink_count: int = 0
    long_blink_count: int = 0
    microsleep_count: int = 0
    blink_rate: float = 0.0
    perclos: float = 0.0
    perclos_ready: bool = False
    drowsy: bool = False
    critical: bool = False

    @property
    def closed_fraction(self) -> float:
        return 1.0 if self.both_closed else 0.0


@dataclass
class EyeAnalyzer:
    """Turns per-frame eye probabilities into drowsiness metrics."""

    config: DMSConfig
    _left_ema: EMA = field(init=False)
    _right_ema: EMA = field(init=False)
    _left_closed: bool = field(default=False, init=False)
    _right_closed: bool = field(default=False, init=False)
    _closure_start: float | None = field(default=None, init=False)
    _last_closure: float = field(default=0.0, init=False)
    _blink_times: deque[float] = field(default_factory=deque, init=False)
    _perclos_samples: deque[tuple[float, float]] = field(
        default_factory=deque, init=False
    )
    _perclos_sum: float = field(default=0.0, init=False)
    _perclos_since: float | None = field(default=None, init=False)
    _first_sample: float | None = field(default=None, init=False)
    _last_closure_end: float | None = field(default=None, init=False)
    _open_since: float | None = field(default=None, init=False)
    blink_count: int = field(default=0, init=False)
    long_blink_count: int = field(default=0, init=False)
    microsleep_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._left_ema = EMA(alpha=0.55)
        self._right_ema = EMA(alpha=0.55)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(
        self, observation: EyeObservation | None, now: float
    ) -> EyeMetrics:
        now = float(now)
        self._prune(now)

        if observation is None or not observation.valid:
            # A detection dropout must not count as a closure.  Drop the
            # in-progress closure (it cannot be measured reliably).
            self._closure_start = None
            return EyeMetrics(
                valid=False,
                blink_rate=self._blink_rate(now),
                perclos=self._perclos(),
                perclos_ready=self._perclos_ready(now),
                blink_count=self.blink_count,
                long_blink_count=self.long_blink_count,
                microsleep_count=self.microsleep_count,
                last_closure_seconds=self._last_closure,
            )

        if self._first_sample is None:
            self._first_sample = now
        if self._perclos_since is None:
            self._perclos_since = now

        left = self._left_ema.update(float(observation.left_open))
        right = self._right_ema.update(float(observation.right_open))
        self._left_closed = self._hysteresis(
            left, self._left_closed
        )
        self._right_closed = self._hysteresis(
            right, self._right_closed
        )
        both_closed = self._left_closed and self._right_closed

        if both_closed:
            self._open_since = None
            if self._closure_start is None:
                self._closure_start = now
        else:
            if self._open_since is None:
                self._open_since = now
            elif (now - self._open_since) >= 3.0:
                # Eyes open for 3+ seconds continuously: decay PERCLOS buffer faster
                cutoff = now - 5.0
                while self._perclos_samples and self._perclos_samples[0][0] < cutoff:
                    _, value = self._perclos_samples.popleft()
                    self._perclos_sum -= value
                if (now - self._open_since) >= 5.0 and self._perclos_sum > 0.0:
                    self._perclos_samples = deque(
                        (t, 0.0) for t, _ in self._perclos_samples
                    )
                    self._perclos_sum = 0.0

            if self._closure_start is not None:
                duration = max(0.0, now - self._closure_start)
                self._last_closure = duration
                self._last_closure_end = now
                if duration <= self.config.blink_max_seconds:
                    self.blink_count += 1
                    self._blink_times.append(now)
                else:
                    self.long_blink_count += 1
                if duration >= self.config.eye_closed_critical_seconds:
                    self.microsleep_count += 1
                self._closure_start = None

        closure_seconds = (
            max(0.0, now - self._closure_start)
            if self._closure_start is not None
            else 0.0
        )

        # PERCLOS only counts frames where the eyes were measurable.
        self._perclos_samples.append((now, 1.0 if both_closed else 0.0))
        self._perclos_sum += 1.0 if both_closed else 0.0

        perclos_ready = self._perclos_ready(now)
        perclos = self._perclos()
        recent_closure = both_closed or (
            self._last_closure_end is not None
            and (now - self._last_closure_end) <= 2.0
        )
        drowsy = bool(
            (both_closed and closure_seconds >= self.config.eye_closed_seconds)
            or (
                perclos_ready
                and perclos >= self.config.perclos_threshold
                and recent_closure
            )
        )
        critical = bool(
            both_closed
            and closure_seconds >= self.config.eye_closed_critical_seconds
        )

        return EyeMetrics(
            valid=True,
            left_open=float(max(0.0, min(1.0, left))),
            right_open=float(max(0.0, min(1.0, right))),
            left_closed=self._left_closed,
            right_closed=self._right_closed,
            both_closed=both_closed,
            closure_seconds=closure_seconds,
            last_closure_seconds=self._last_closure,
            blink_count=self.blink_count,
            long_blink_count=self.long_blink_count,
            microsleep_count=self.microsleep_count,
            blink_rate=self._blink_rate(now),
            perclos=perclos,
            perclos_ready=perclos_ready,
            drowsy=drowsy,
            critical=critical,
        )

    def reset(self) -> None:
        self._left_ema.reset()
        self._right_ema.reset()
        self._left_closed = False
        self._right_closed = False
        self._closure_start = None
        self._last_closure = 0.0
        self._last_closure_end = None
        self._open_since = None
        self._blink_times.clear()
        self._perclos_samples.clear()
        self._perclos_sum = 0.0
        self._perclos_since = None
        self._first_sample = None
        self.blink_count = 0
        self.long_blink_count = 0
        self.microsleep_count = 0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _hysteresis(self, probability: float, was_closed: bool) -> bool:
        if probability <= self.config.eye_closed_enter:
            return True
        if probability >= self.config.eye_closed_exit:
            return False
        return was_closed

    def _prune(self, now: float) -> None:
        limit = now - self.config.blink_rate_window
        while self._blink_times and self._blink_times[0] < limit:
            self._blink_times.popleft()

        limit = now - self.config.perclos_window
        while self._perclos_samples and self._perclos_samples[0][0] < limit:
            _, value = self._perclos_samples.popleft()
            self._perclos_sum -= value

    def _perclos(self) -> float:
        if not self._perclos_samples:
            return 0.0
        return max(
            0.0, min(1.0, self._perclos_sum / len(self._perclos_samples))
        )

    def _perclos_ready(self, now: float) -> bool:
        if self._perclos_since is None:
            return False
        return (now - self._perclos_since) >= self.config.perclos_min_time

    def _blink_rate(self, now: float) -> float:
        """Blinks per minute, using the observed part of the window."""
        if self._first_sample is None:
            return 0.0
        observed = min(
            self.config.blink_rate_window, max(0.0, now - self._first_sample)
        )
        if observed < 10.0:
            return 0.0
        return float(len(self._blink_times)) * 60.0 / observed


__all__ = ["EyeAnalyzer", "EyeMetrics"]