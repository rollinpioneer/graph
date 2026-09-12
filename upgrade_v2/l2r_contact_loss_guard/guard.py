from __future__ import annotations

from dataclasses import dataclass
from typing import Any

INTERCEPT_REASON = "historical_hold_established_then_observed_non_release_contact_loss"
PENDING = "CONTACT_LOSS_PENDING"
CONFIRMED_REASON = "NON_RELEASE_CONTACT_LOSS_PERSISTENCE_CONFIRMED"


@dataclass
class ContactLossPending:
    active: bool = False
    attempt_id: int | None = None
    start_time: float | None = None
    start_capture_order: int | None = None


class ContactLossPersistenceGuard:
    """Fixed two-valid-sample, 0.10 s guard for one non-release loss path."""

    required_consecutive_valid_samples = 2
    maximum_pending_gap_s = 0.10

    def __init__(self) -> None:
        self.pending = ContactLossPending()
        self.previous_time: float | None = None
        self.previous_order: int | None = None

    def clear(self) -> None:
        self.pending = ContactLossPending()

    def step(self, observation: dict[str, Any], proposed: dict[str, Any]) -> dict[str, Any]:
        now = observation.get("time")
        order = observation.get("capture_order")
        attempt = observation.get("attempt_id")
        contact = observation.get("contact_present")
        command = observation.get("gripper_command")
        release = command == "open" or observation.get("requested_effect") == "RELEASE_OBJECT" \
            or observation.get("attempt_end_reason") == "release"
        valid = (now is not None and order is not None and attempt is not None
                 and isinstance(contact, bool) and command in {"open", "closed"}
                 and observation.get("context_valid") is True)
        causal_order = (self.previous_order is None or int(order) > self.previous_order)
        causal_time = (self.previous_time is None or float(now) > self.previous_time)
        if not valid or not causal_order or not causal_time or release or observation.get("attempt_end") is True:
            self.clear()
            selected, reason, state = proposed.get("selected_action", "none"), proposed.get("reason_code"), "CLEAR"
        elif self.pending.active:
            dt = float(now) - float(self.pending.start_time)
            confirms = (int(attempt) == self.pending.attempt_id and contact is False
                        and 0.0 < dt <= 0.10)
            if confirms:
                selected, reason, state = "recover_object", CONFIRMED_REASON, "CONFIRMED"
                self.clear()
            else:
                self.clear()
                selected, reason, state = "none", "CONTACT_LOSS_PENDING_CLEARED", "CLEAR"
        elif (proposed.get("selected_action") == "recover_object"
              and proposed.get("reason_code") == INTERCEPT_REASON
              and contact is False):
            self.pending = ContactLossPending(True, int(attempt), float(now), int(order))
            selected, reason, state = "none", PENDING, PENDING
        else:
            selected, reason, state = proposed.get("selected_action", "none"), proposed.get("reason_code"), "PASS_THROUGH"
        self.previous_time = float(now) if now is not None else self.previous_time
        self.previous_order = int(order) if order is not None else self.previous_order
        # A guarded action is emitted at the current causal observation.  Never
        # inherit/backdate the timestamp or capture order from the intercepted
        # first-sample proposal.
        return {**proposed, "time": now, "capture_order": order,
                "selected_action": selected, "reason_code": reason,
                "guard_state": state, "contact_loss_pending": self.pending.active,
                "pending_attempt_id": self.pending.attempt_id,
                "pending_start_time": self.pending.start_time,
                "pending_start_capture_order": self.pending.start_capture_order}


def apply_guard(observations: list[dict[str, Any]], proposed_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    guard = ContactLossPersistenceGuard()
    return [guard.step(observation, proposed) for observation, proposed in zip(observations, proposed_records)]
