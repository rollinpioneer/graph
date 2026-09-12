from __future__ import annotations

from dataclasses import dataclass
from typing import Any

INTERCEPT_REASON = "historical_hold_established_then_observed_non_release_contact_loss"
PENDING_REASON = "CONTACT_LOSS_PENDING_LOGICAL_CLOCK"
CONFIRMED_REASON = "NON_RELEASE_CONTACT_LOSS_PERSISTENCE_CONFIRMED_LOGICAL_CLOCK"


@dataclass
class Pending:
    active: bool = False
    attempt_id: int | None = None
    physical_time: float | None = None
    capture_order: int | None = None


class LogicalObservationClockGuard:
    """Two observations ordered by capture_order, bounded by physical time.

    A later capture may share the same post-step physical timestamp.  It is a
    distinct causal observation only when capture_order strictly increases.
    """

    required_consecutive_valid_observations = 2
    maximum_physical_gap_s = 0.10
    minimum_physical_gap_s = 0.0

    def __init__(self) -> None:
        self.pending = Pending()
        self.previous_capture_order: int | None = None
        self.previous_physical_time: float | None = None

    def clear(self) -> None:
        self.pending = Pending()

    def step(self, observation: dict[str, Any], proposed: dict[str, Any]) -> dict[str, Any]:
        now = observation.get("time")
        order = observation.get("capture_order")
        attempt = observation.get("attempt_id")
        contact = observation.get("contact_present")
        command = observation.get("gripper_command")
        release = (command == "open" or observation.get("requested_effect") == "RELEASE_OBJECT"
                   or observation.get("attempt_end_reason") == "release")
        fields_valid = (now is not None and order is not None and attempt is not None
                        and isinstance(contact, bool) and command in {"open", "closed"}
                        and observation.get("context_valid") is True)
        logical_forward = self.previous_capture_order is None or int(order) > self.previous_capture_order
        physical_nondecreasing = self.previous_physical_time is None or float(now) >= self.previous_physical_time
        valid = fields_valid and logical_forward and physical_nondecreasing
        selected = proposed.get("selected_action", "none")
        reason = proposed.get("reason_code")
        state = "PASS_THROUGH"

        if not valid or release or observation.get("attempt_end") is True:
            self.clear()
            state = "CLEAR"
        elif self.pending.active:
            physical_dt = float(now) - float(self.pending.physical_time)
            confirms = (int(attempt) == self.pending.attempt_id and contact is False
                        and int(order) > int(self.pending.capture_order)
                        and 0.0 <= physical_dt <= self.maximum_physical_gap_s)
            if confirms:
                selected, reason, state = "recover_object", CONFIRMED_REASON, "CONFIRMED"
                self.clear()
            else:
                selected, reason, state = "none", "CONTACT_LOSS_PENDING_LOGICAL_CLOCK_CLEARED", "CLEAR"
                self.clear()
        elif (selected == "recover_object" and reason == INTERCEPT_REASON and contact is False):
            self.pending = Pending(True, int(attempt), float(now), int(order))
            selected, reason, state = "none", PENDING_REASON, "PENDING"

        if fields_valid:
            self.previous_capture_order = int(order)
            self.previous_physical_time = float(now)
        return {**proposed, "time": now, "capture_order": order,
                "selected_action": selected, "reason_code": reason,
                "logical_clock": order, "physical_time": now,
                "guard_state": state, "contact_loss_pending": self.pending.active,
                "pending_attempt_id": self.pending.attempt_id,
                "pending_physical_time": self.pending.physical_time,
                "pending_capture_order": self.pending.capture_order}


def apply_guard(observations: list[dict[str, Any]], proposals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    guard = LogicalObservationClockGuard()
    return [guard.step(observation, proposal) for observation, proposal in zip(observations, proposals)]

