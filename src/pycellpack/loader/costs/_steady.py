"""Steady-state windows of a run, from periodic samples of its counters.

Two signals are watched: throughput (accesses per second) and page reads per
access (the cache's state). Tolerances follow the run's own noise: the
spread of its one-second rates around their linear trend (as a share of the
mean), which shared and network storage put near twenty percent. Warm-up
ends once, over the last `window` seconds, neither signal drifts (the means
of the window's two halves differ by at most `drift`, or two standard errors
of a half's mean if more) and the noise is at most `noise` (no oscillation),
at least `max(min_warmup, window)` seconds in. The steady time then
accumulates; if the last five seconds' throughput leaves the measurement's
mean by more than `guard` (or three standard errors of a five-second mean),
or the full window's linear trend changes the rate by more than twice `drift`
(or two standard errors), the measurement restarts from there. A window of
`seconds` steady seconds ends it. Past `max_warmup + seconds` (or after more
than `restarts` restarts) the quietest `seconds`-long stretch after
`min_warmup` is taken instead, marked unsteady.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Absolute drift allowed in page reads per access (for caches that read little).
READS_FLOOR = 0.01
#: Seconds of recent throughput the measurement guard compares.
GUARD_SECONDS = 5.0


@dataclass(frozen=True)
class Window:
    """Samples `first..last` (indices) bound the window."""

    first: int
    last: int
    seconds: float
    steady: bool
    cv: float
    restarts: int


class Steady:
    """Steady-state detector over periodic samples of a run's counters.

    Samples are time, cumulative accesses and cumulative page reads, as
    `Run.snapshot` gives them. `push` takes one and reports when a window of
    `seconds` steady seconds is ready (`result`); warm-up, restarts and the
    unsteady fallback follow the module's description. Two signals are watched,
    throughput and page reads per access, so a window is steady only while the
    cache's state is too.
    """

    def __init__(self, seconds=30.0, *, window=10.0, drift=0.03, noise=0.5, guard=0.10,
                 min_warmup=2.0, max_warmup=None, restarts=3):
        """`seconds` of steady time end a measurement; `window` seconds are
        judged for warm-up; `drift`, `noise` and `guard` are the tolerance
        floors (relative) that the run's own noise raises; `min_warmup` is the
        least time before any window; `max_warmup` defaults to
        `max(60, 2 * seconds)`; `restarts` bounds the restarts of one
        measurement."""
        if seconds <= 0 or window <= 0:
            raise ValueError("steady seconds and the detection window must be positive")
        self.seconds, self.window, self.drift, self.noise, self.guard = seconds, window, drift, noise, guard
        self.min_warmup = min_warmup
        self.max_warmup = max(60.0, 2 * seconds) if max_warmup is None else max_warmup
        self.max_restarts = restarts
        self.t: list[float] = []
        self.accesses: list[float] = []
        self.reads: list[float] = []
        self.start: int | None = None
        #: The noise measured when the current measurement started.
        self.level = 0.0
        self.restarts = 0
        self.result: Window | None = None

    # Indices and rates over the samples.
    def _at(self, t):
        """The last sample at or before time `t` (the first if none)."""
        lo, hi = 0, len(self.t) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.t[mid] <= t:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _after(self, t):
        """The first sample at or after time `t` (the last if none)."""
        lo, hi = 0, len(self.t) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self.t[mid] >= t:
                hi = mid
            else:
                lo = mid + 1
        return lo

    def _rate(self, i, j):
        """Accesses per second between samples i and j."""
        dt = self.t[j] - self.t[i]
        return (self.accesses[j] - self.accesses[i]) / dt if dt > 0 else 0.0

    def _per_access(self, i, j):
        """Page reads per access between samples i and j."""
        da = self.accesses[j] - self.accesses[i]
        return (self.reads[j] - self.reads[i]) / da if da > 0 else 0.0

    def _seconds(self, i, j):
        """One-second rates over samples i..j, with their mid times."""
        rates, times, a = [], [], i
        for b in range(i + 1, j + 1):
            if self.t[b] - self.t[a] >= 1.0 or b == j:
                if self.t[b] > self.t[a]:
                    rates.append(self._rate(a, b))
                    times.append((self.t[a] + self.t[b]) / 2)
                a = b
        return rates, times

    def _trend(self, i, j):
        """The one-second rates' linear fit over samples i..j: the noise
        around it and the change it makes over the span (both as shares of
        the mean), and that change's standard error."""
        rates, times = self._seconds(i, j)
        n = len(rates)
        if n < 4:
            return math.inf, 0.0, math.inf
        mean_r, mean_t = sum(rates) / n, sum(times) / n
        stt = sum((x - mean_t) ** 2 for x in times)
        if mean_r <= 0 or stt <= 0:
            return math.inf, 0.0, math.inf
        slope = sum((x - mean_t) * (r - mean_r) for x, r in zip(times, rates)) / stt
        residual = sum((r - mean_r - slope * (x - mean_t)) ** 2 for x, r in zip(times, rates))
        sigma = math.sqrt(residual / (n - 2))
        span = times[-1] - times[0]
        return sigma / mean_r, slope * span / mean_r, sigma / math.sqrt(stt) * span / mean_r

    def _noise(self, i, j):
        """Spread of the one-second rates around their linear trend, as a
        share of their mean."""
        return self._trend(i, j)[0]

    def _cv(self, i, j):
        """Coefficient of variation of one-second rates over samples i..j."""
        rates, _ = self._seconds(i, j)
        if len(rates) < 2:
            return math.inf
        mean = sum(rates) / len(rates)
        if mean <= 0:
            return math.inf
        var = sum((r - mean) ** 2 for r in rates) / (len(rates) - 1)
        return math.sqrt(var) / mean

    def _quiet(self, k):
        """Whether the `window` seconds before sample k are steady."""
        t = self.t[k]
        if t - self.t[0] < max(self.min_warmup, self.window):
            return False
        i = self._at(t - self.window)
        m = self._at(t - self.window / 2)
        if not i < m < k:
            return False
        noise = self._noise(i, k)
        if noise > self.noise:
            return False
        drift = max(self.drift, 2 * noise / math.sqrt(self.window / 2))
        r1, r2 = self._rate(i, m), self._rate(m, k)
        if r1 <= 0 or r2 <= 0 or abs(r2 - r1) / ((r1 + r2) / 2) > drift:
            return False
        q1, q2 = self._per_access(i, m), self._per_access(m, k)
        if abs(q2 - q1) > max(READS_FLOOR, drift * (q1 + q2) / 2):
            return False
        self.level = noise
        return True

    def push(self, t: float, accesses: float, reads: float) -> bool:
        """Adds a sample (time in seconds, cumulative accesses and page
        reads); returns whether a window is ready (`result`)."""
        if self.result is not None:
            return True
        if self.t and t <= self.t[-1]:
            return False
        self.t.append(t)
        self.accesses.append(accesses)
        self.reads.append(reads)
        k = len(self.t) - 1
        if self.start is None:
            if self._quiet(k):
                self.start = k
        else:
            span = t - self.t[self.start]
            if span >= GUARD_SECONDS:
                recent = self._rate(self._at(t - GUARD_SECONDS), k)
                mean = self._rate(self.start, k)
                guard = max(self.guard, 3 * self.level / math.sqrt(GUARD_SECONDS))
                if mean <= 0 or abs(recent - mean) / mean > guard:
                    self.restarts += 1
                    self.start = None
            if self.start is not None and span >= self.seconds:
                # The whole window's trend, which halves of a short window
                # cannot resolve in noisy runs.
                _, change, error = self._trend(self.start, k)
                if abs(change) > max(2 * self.drift, 2 * error):
                    self.restarts += 1
                    self.start = None
                else:
                    self.result = Window(self.start, k, span, True, self._cv(self.start, k), self.restarts)
                    return True
        if t - self.t[0] >= self.max_warmup + self.seconds or self.restarts > self.max_restarts:
            self.result = self.best()
            return self.result is not None
        return False

    def best(self) -> Window | None:
        """The quietest `seconds`-long stretch after `min_warmup` (unsteady),
        or None when the samples are shorter."""
        best = None
        for i in range(len(self.t)):
            if self.t[i] - self.t[0] < self.min_warmup:
                continue
            j = self._after(self.t[i] + self.seconds)
            if self.t[j] - self.t[i] < self.seconds:
                break
            cv = self._cv(i, j)
            if best is None or cv < best.cv:
                best = Window(i, j, self.t[j] - self.t[i], False, cv, self.restarts)
        return best
