"""Controller-owned lifecycle markers for one grasp attempt.

The lifecycle is emitted by the skill/controller schedule.  It is deliberately
independent of contact, hold evidence, success, and retry decisions so the
event interface can use it as causal context rather than reconstructing an
attempt boundary from its own outcome predicates.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


ATTEMPT_PHASES = (
    "inactive",
    "acquiring",
    "settling",
    "post_contact_motion",
    "ended",
)
END_REASONS = ("segment_complete", "cancelled", "release")


@dataclass
class AttemptLifecycleState:
    attempt_id: int = 0
    attempt_phase: str = "inactive"
    attempt_active: bool = False
    attempt_end: bool = False
    attempt_end_reason: str | None = None


class AttemptLifecycle:
    """Small controller-side state machine with a one-cycle end edge."""

    def __init__(self) -> None:
        self.state = AttemptLifecycleState()

    def begin(self, phase: str = "acquiring") -> int:
        if phase not in ATTEMPT_PHASES or phase in {"inactive", "ended"}:
            raise ValueError(f"invalid active attempt phase: {phase}")
        if self.state.attempt_active:
            raise RuntimeError("an attempt is already active")
        self.state.attempt_id += 1
        self.state.attempt_phase = phase
        self.state.attempt_active = True
        self.state.attempt_end = False
        self.state.attempt_end_reason = None
        return self.state.attempt_id

    def set_phase(self, phase: str) -> None:
        if phase not in ATTEMPT_PHASES:
            raise ValueError(f"unknown attempt phase: {phase}")
        if phase == "inactive":
            self.state.attempt_active = False
        elif phase == "ended":
            self.state.attempt_active = False
        else:
            if self.state.attempt_id == 0:
                self.begin(phase)
                return
            self.state.attempt_active = True
        self.state.attempt_phase = phase

    def end(self, reason: str = "segment_complete") -> None:
        if reason not in END_REASONS:
            raise ValueError(f"unknown attempt end reason: {reason}")
        if self.state.attempt_id == 0:
            raise RuntimeError("cannot end an attempt before it starts")
        if self.state.attempt_end:
            raise RuntimeError("attempt_end is a one-cycle edge and is already set")
        self.state.attempt_phase = "ended"
        self.state.attempt_active = False
        self.state.attempt_end = True
        self.state.attempt_end_reason = reason

    def begin_next_cycle(self) -> None:
        """Clear the end edge before the next controller update."""
        if self.state.attempt_end:
            self.state.attempt_end = False
            self.state.attempt_end_reason = None
            self.state.attempt_phase = "inactive"
            self.state.attempt_active = False

    def snapshot(self) -> dict[str, Any]:
        return asdict(self.state)
