"""Independent TaskEvaluator using hidden MuJoCo truth. Never reads VLM/controller success."""
from __future__ import annotations

from cp_disr.adapters import TaskResult
from .d0_env import CONTAINER_INNER, OBJECT_HALF


EVALUATOR_VERSION = "cp-disr-d0-task-evaluator-v1"


class TaskEvaluator:
    def __init__(self, env, deadline_seconds: float):
        self.env = env
        self.deadline = float(deadline_seconds)
        self._rewarded = False
        self._success_time = None

    def reset_episode(self):
        self._rewarded = False
        self._success_time = None

    def goal_true(self) -> bool:
        h = self.env.hidden_truth()
        t = h["target"]
        c = h["container"]
        lid = h["lid"]
        inside_xy = abs(t[0]-c[0]) <= CONTAINER_INNER[0] and abs(t[1]-c[1]) <= CONTAINER_INNER[1]
        inside_z = t[2] <= float(h["table_top_z"]) + 0.12
        lid_away = ((lid[0]-c[0])**2 + (lid[1]-c[1])**2) ** 0.5 > 0.10
        return bool(inside_xy and inside_z and lid_away)

    def evaluate(self, value):
        elapsed = float(value.elapsed_seconds)
        success_now = self.goal_true()
        reward_events = []
        if success_now and not self._rewarded:
            self._rewarded = True
            self._success_time = elapsed
            reward_events.append((elapsed, 1.0))
        terminated = bool(self._rewarded)
        truncated = (not terminated) and elapsed >= self.deadline
        reason = "TASK_SUCCESS" if terminated else ("DEADLINE" if truncated else "CONTINUE")
        return TaskResult(
            success=bool(self._rewarded),
            terminated=terminated,
            truncated=truncated,
            reason=reason,
            reward_events=tuple(reward_events),
        )
