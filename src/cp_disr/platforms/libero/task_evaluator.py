"""Independent TaskEvaluator using hidden MuJoCo truth. Never reads VLM/controller success."""
from __future__ import annotations

from cp_disr.adapters import TaskResult
from .d0_env import CONTAINER_INNER


EVALUATOR_VERSION = "cp-disr-d0-task-evaluator-v1"
TA_EVALUATOR_VERSION = "cp-disr-ta-task-evaluator-v1"
TC_EVALUATOR_VERSION = "cp-disr-tc-task-evaluator-v1"


class TaskEvaluator:
    def __init__(self, env, deadline_seconds: float, task_id: str = "D0"):
        self.env = env
        self.deadline = float(deadline_seconds)
        self.task_id = str(task_id)
        self._rewarded = False
        self._success_time = None

    def reset_episode(self):
        self._rewarded = False
        self._success_time = None

    def _lid_away(self, h) -> bool:
        lid = h["lid"]
        c = h["container"]
        return ((lid[0] - c[0]) ** 2 + (lid[1] - c[1]) ** 2) ** 0.5 > 0.10

    def _inside(self, h, name: str) -> bool:
        obj = h[name]
        c = h["container"]
        inside_xy = abs(obj[0] - c[0]) <= CONTAINER_INNER[0] and abs(obj[1] - c[1]) <= CONTAINER_INNER[1]
        inside_z = obj[2] <= float(h["table_top_z"]) + 0.12
        return bool(inside_xy and inside_z and self._lid_away(h))

    def _second_role(self) -> str:
        return getattr(self.env, "second_role", getattr(getattr(self.env, "case", None), "second_role", "second_object"))

    def goal_true(self) -> bool:
        h = self.env.hidden_truth()
        if self.task_id == "T_A":
            return self._inside(h, "target") and self._inside(h, self._second_role())
        return self._inside(h, "target")

    def evaluate(self, value):
        elapsed = float(value.elapsed_seconds)
        interval_start = float(value.interval_start_seconds)
        interval_end = float(value.interval_end_seconds)
        success_now = self.goal_true()
        reward_events = []
        if success_now and not self._rewarded:
            self._rewarded = True
            self._success_time = elapsed
            offset = interval_end - interval_start
            if offset < 0:
                offset = 0.0
            reward_events.append((float(offset), 1.0))
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
