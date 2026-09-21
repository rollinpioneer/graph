"""Simulation-second clock bound to MuJoCo time. Binding unit token is seconds."""
from __future__ import annotations

class DurationProvider:
    def __init__(self, env):
        self.env = env
        self._origin = float(env.sim.data.time)

    def now_seconds(self) -> float:
        return float(self.env.sim.data.time) - self._origin

    def duration_seconds(self, start: float, end: float) -> float:
        d = float(end) - float(start)
        if d <= 0:
            d = 1.0 / 20.0
        return d
