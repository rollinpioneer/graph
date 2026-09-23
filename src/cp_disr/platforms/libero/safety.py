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

    def _workspace_ok(self, eef) -> bool:
        if eef is None:
            return False
        eef = np.asarray(eef, dtype=float)
        if eef.size < 3 or not np.all(np.isfinite(eef.reshape(-1)[:3])):
            return False
        for i, (lo, hi) in enumerate(self.workspace):
            if not (lo - 0.05 <= float(eef.reshape(-1)[i]) <= hi + 0.05):
                return False
        return True

    def can_execute(self, candidate_id, observation) -> bool:
        parts = str(candidate_id).split(":")
        if len(parts) < 2 or parts[1] == "MOVE":
            return False
        try:
            eef = self._eef(observation)
        except Exception:
            return False
        return self._workspace_ok(eef)

    def can_hold(self, observation) -> bool:
        """Independent hold/confirm permission. Fail closed if unknown."""
        try:
            if observation is None:
                observation = self.env.public_observation()
            eef = self._eef(observation)
        except Exception:
            return False
        if not self._workspace_ok(eef):
            return False
        if isinstance(observation, dict) and "gripper_qpos" in observation:
            grip = np.asarray(observation["gripper_qpos"], dtype=float)
            if grip.size == 0 or not np.all(np.isfinite(grip)):
                return False
        return True

    def end_no_candidates(self, env_id, episode_id) -> str:
        return f"NO_CANDIDATE_SAFE_TERMINATION:{env_id}:{episode_id}"