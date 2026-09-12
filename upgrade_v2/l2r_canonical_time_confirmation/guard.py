from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from upgrade_v2.l2r_logical_observation_clock.guard import INTERCEPT_REASON

PENDING_REASON = "CONTACT_LOSS_PENDING_CANONICAL_TIME"
SAME_TIME_PENDING_REASON = "CONTACT_LOSS_PENDING_SAME_PHYSICAL_TIME"
CONFIRMED_REASON = "NON_RELEASE_CONTACT_LOSS_PERSISTENCE_CONFIRMED_CANONICAL_TIME"
MAXIMUM_PHYSICAL_GAP_NS = 100_000_000


@dataclass
class Pending:
    active: bool = False
    attempt_id: int | None = None
    physical_time_ns: int | None = None
    capture_order: int | None = None


class CanonicalTimeGuard:
    required_consecutive_valid_observations = 2
    minimum_physical_gap_ns = 1
    maximum_physical_gap_ns = MAXIMUM_PHYSICAL_GAP_NS

    def __init__(self) -> None:
        self.pending = Pending()
        self.previous_order: int | None = None
        self.previous_ns: int | None = None

    def clear(self) -> None: self.pending = Pending()

    def step(self, observation: dict[str, Any], proposed: dict[str, Any]) -> dict[str, Any]:
        ns, order = observation.get("physical_time_ns"), observation.get("capture_order")
        attempt, contact, command = observation.get("attempt_id"), observation.get("contact_present"), observation.get("gripper_command")
        integer_ns = isinstance(ns, int) and not isinstance(ns, bool)
        fields_valid = (integer_ns and isinstance(order, int) and isinstance(attempt, int)
                        and isinstance(contact, bool) and command in {"open", "closed"}
                        and observation.get("context_valid") is True)
        forward = self.previous_order is None or (isinstance(order, int) and order > self.previous_order)
        nondecreasing = self.previous_ns is None or (integer_ns and ns >= self.previous_ns)
        release = command == "open" or observation.get("requested_effect") == "RELEASE_OBJECT" or observation.get("attempt_end_reason") == "release"
        valid = fields_valid and forward and nondecreasing
        selected, reason, state = proposed.get("selected_action", "none"), proposed.get("reason_code"), "PASS_THROUGH"
        if not valid or release or observation.get("attempt_end") is True:
            self.clear(); state = "CLEAR"
        elif self.pending.active:
            dt = ns - int(self.pending.physical_time_ns)
            same_attempt = attempt == self.pending.attempt_id
            if same_attempt and contact is False and dt == 0:
                selected, reason, state = "none", SAME_TIME_PENDING_REASON, "PENDING"
            elif (same_attempt and contact is False and order > int(self.pending.capture_order)
                  and self.minimum_physical_gap_ns <= dt <= self.maximum_physical_gap_ns):
                selected, reason, state = "recover_object", CONFIRMED_REASON, "CONFIRMED"; self.clear()
            else:
                selected, reason, state = "none", "CONTACT_LOSS_PENDING_CANONICAL_TIME_CLEARED", "CLEAR"; self.clear()
        elif selected == "recover_object" and reason == INTERCEPT_REASON and contact is False:
            self.pending = Pending(True, attempt, ns, order)
            selected, reason, state = "none", PENDING_REASON, "PENDING"
        if fields_valid:
            self.previous_order, self.previous_ns = order, ns
        return {**proposed, "time": observation.get("time"), "physical_time_ns": ns, "capture_order": order,
                "selected_action": selected, "reason_code": reason, "guard_state": state,
                "contact_loss_pending": self.pending.active, "pending_physical_time_ns": self.pending.physical_time_ns,
                "pending_capture_order": self.pending.capture_order, "pending_attempt_id": self.pending.attempt_id}


def apply_guard(rows: list[dict[str, Any]], proposals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    guard = CanonicalTimeGuard()
    return [guard.step(row, proposal) for row, proposal in zip(rows, proposals)]
