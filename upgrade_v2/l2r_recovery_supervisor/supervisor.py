from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable


RETRYABLE_EXECUTION_FAILURES = {"RELOCATION_ERROR", "REGRASP_ERROR"}


@dataclass
class SupervisorResult:
    success: bool
    failure_stage: str | None
    outer_cycles: int
    execution_results: list[dict[str, Any]] = field(default_factory=list)
    goal_verifications: list[bool] = field(default_factory=list)
    rearmed_loss_episodes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RecoverySupervisorV2:
    """Retry execution, verify the goal, and re-arm for a later loss episode."""

    def __init__(self, maximum_outer_cycles: int = 3) -> None:
        if maximum_outer_cycles != 3:
            raise ValueError("R25_SUPERVISOR_OUTER_CYCLES_FROZEN_AT_3")
        self.maximum_outer_cycles = maximum_outer_cycles

    def run(self, *, initial_action: str,
            execute_recovery: Callable[[str, int], dict[str, Any]],
            verify_goal: Callable[[int], bool],
            next_loss_action: Callable[[int], str | None],
            rearm_after_verified_hold: Callable[[int], None]) -> dict[str, Any]:
        action = initial_action
        executions: list[dict[str, Any]] = []
        goals: list[bool] = []
        rearmed = 0
        for cycle in range(1, self.maximum_outer_cycles + 1):
            result = dict(execute_recovery(action, cycle))
            executions.append(result)
            if not result.get("success"):
                stage = result.get("failure_stage") or "RECOVERY_EXECUTION_ERROR"
                if stage in RETRYABLE_EXECUTION_FAILURES and cycle < self.maximum_outer_cycles:
                    continue
                return SupervisorResult(False, stage, cycle, executions, goals, rearmed).to_dict()

            rearm_after_verified_hold(cycle)
            rearmed += 1
            follow_up = next_loss_action(cycle)
            goal_ok = bool(verify_goal(cycle))
            goals.append(goal_ok)
            if follow_up is not None:
                action = follow_up
                if cycle < self.maximum_outer_cycles:
                    continue
                return SupervisorResult(False, "SECONDARY_LOSS_LIMIT", cycle, executions, goals, rearmed).to_dict()
            if goal_ok:
                return SupervisorResult(True, None, cycle, executions, goals, rearmed).to_dict()
            if cycle == self.maximum_outer_cycles:
                return SupervisorResult(False, "TASK_RECOVERY_ERROR", cycle, executions, goals, rearmed).to_dict()
            action = "recover_object"
        return SupervisorResult(False, "RECOVERY_LOOP_LIMIT", self.maximum_outer_cycles,
                                executions, goals, rearmed).to_dict()
