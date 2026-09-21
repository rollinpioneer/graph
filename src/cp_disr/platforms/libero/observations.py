"""Observation wrapper that never includes hidden object poses."""
from __future__ import annotations

from cp_disr.adapters import Observation


class ObservationProvider:
    def __init__(self, env, clock):
        self.env = env
        self.clock = clock
        self._n = 0

    def observe(self) -> Observation:
        obs = self.env.public_observation()
        self._n += 1
        t = self.clock.now_seconds()
        return Observation(
            frame_id=f"frame-{self._n}",
            rgb_ref="agentview_rgb",
            depth_ref="agentview_depth",
            proprioception=tuple(float(x) for x in obs["proprio"]),
            capture_seconds=t,
            available_seconds=t,
        )
