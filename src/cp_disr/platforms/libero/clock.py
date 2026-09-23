"""Simulation-second clock bound to MuJoCo time. Binding unit is seconds."""
from __future__ import annotations

import math

from cp_disr.common import ClockIntegrityError

# Frozen control period of the registered OSC interface.
CONTROL_DT = 1.0 / 20.0


class DurationProvider:
    """Frozen sim-clock endpoints only. Never invent a positive duration."""

    def __init__(self, env):
        self.env = env
        self._origin = float(env.sim.data.time)

    def now_seconds(self) -> float:
        now = float(self.env.sim.data.time) - self._origin
        if not math.isfinite(now):
            raise ClockIntegrityError("nonfinite clock.now_seconds: %r" % (now,))
        return now

    def duration_seconds(self, start: float, end: float) -> float:
        start_f = float(start)
        end_f = float(end)
        if not math.isfinite(start_f) or not math.isfinite(end_f):
            raise ClockIntegrityError("nonfinite clock endpoints start=%r end=%r" % (start, end))
        d = end_f - start_f
        if not math.isfinite(d):
            raise ClockIntegrityError("nonfinite duration start=%r end=%r" % (start, end))
        if d < 0:
            raise ClockIntegrityError("negative duration start=%r end=%r d=%r" % (start, end, d))
        return d