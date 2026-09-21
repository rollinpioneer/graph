"""Workspace, action, and episode safety manager."""
from __future__ import annotations

import numpy as np


class SafetyManager:
    def __init__(self, env, workspace=((-0.45, 0.45), (-0.45, 0.45), (0.75, 1.35))):
        self.env = env
        self.workspace = workspace

    def _eef(self, observation):
        if isinstance(observation, dict) and "eef_pos" in observation:
            return np.asarray(observation["eef_pos"], dtype=float)
        return np.asarray(self.env.public_observation()["eef_pos"], dtype=float)

    def can_execute(self, candidate_id, observation) -> bool:
        parts = str(candidate_id).split(":")
        if len(parts) < 2 or parts[1] == "MOVE":
            return False
        eef = self._eef(observation)
        if not np.all(np.isfinite(eef)):
            return False
        for i, (lo, hi) in enumerate(self.workspace):
            if not (lo - 0.05 <= eef[i] <= hi + 0.05):
                return False
        return True

    def end_no_candidates(self, env_id, episode_id) -> str:
        return f"NO_CANDIDATE_SAFE_TERMINATION:{env_id}:{episode_id}"
