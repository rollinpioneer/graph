"""Causal event memory for the L2RA candidates.

Only already-arrived observable predicates are accepted.  The implementation
uses an explicit three-valued logic so missing observations cannot silently
become false.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable

TRUE, FALSE, UNKNOWN = "true", "false", "unknown"


def tri(value: Any) -> str:
    if value in (TRUE, FALSE, UNKNOWN):
        return value
    if value is True:
        return TRUE
    if value is False:
        return FALSE
    return UNKNOWN


def tri_not(value: Any) -> str:
    value = tri(value)
    return {TRUE: FALSE, FALSE: TRUE, UNKNOWN: UNKNOWN}[value]


def tri_and(*values: Any) -> str:
    values = [tri(value) for value in values]
    if FALSE in values:
        return FALSE
    if UNKNOWN in values:
        return UNKNOWN
    return TRUE


def tri_or(*values: Any) -> str:
    values = [tri(value) for value in values]
    if TRUE in values:
        return TRUE
    if UNKNOWN in values:
        return UNKNOWN
    return FALSE


@dataclass
class MemoryState:
    attempt_id: int = 1
    hold_evidence_in_current_attempt: str = UNKNOWN
    pending_event: str = "none"
    last_contact_state: str = UNKNOWN
    release_command_observed: str = UNKNOWN
    hold_confirm_run_length: int = 0
    loss_confirm_run_length: int = 0
    last_event_onset_time: float | None = None
    resolved_event_id: str | None = None
    event_counter: int = 0
    previous_closed: str = UNKNOWN
    contact_seen_without_hold: str = FALSE


class AttemptScopedMemory:
    def __init__(self, hold_confirm_observations: int = 1, loss_confirm_observations: int = 1, history_complete: bool = True) -> None:
        if hold_confirm_observations not in (1, 2) or loss_confirm_observations not in (1, 2):
            raise ValueError("confirmation observations must be 1 or 2")
        self.hold_confirm = hold_confirm_observations
        self.loss_confirm = loss_confirm_observations
        self.history_complete = history_complete
        self.state = MemoryState(hold_evidence_in_current_attempt=FALSE if history_complete else UNKNOWN)

    def _new_event(self, kind: str, time: float | None) -> str:
        self.state.event_counter += 1
        self.state.last_event_onset_time = time
        self.state.resolved_event_id = None
        return f"event_{self.state.event_counter:03d}_{kind}"

    def observe(self, predicates: dict[str, Any], time: float | None = None, frame_index: int | None = None) -> dict[str, Any]:
        p = {key: tri(value) for key, value in predicates.items()}
        contact = p.get("contact_present", UNKNOWN)
        closed = p.get("gripper_command_closed", UNKNOWN)
        opened = p.get("gripper_command_open", UNKNOWN)
        stable = p.get("stable_hold_observed", UNKNOWN)
        prior_contact = self.state.last_contact_state
        prior_closed = self.state.previous_closed
        release = tri_and(opened, prior_closed)
        if release == TRUE and contact == FALSE:
            self.state.release_command_observed = TRUE
        elif opened == UNKNOWN or prior_closed == UNKNOWN:
            self.state.release_command_observed = UNKNOWN
        else:
            self.state.release_command_observed = FALSE
        # A new closed interval after an observed open starts a new attempt;
        # this depends only on the command transition, never on action names.
        if closed == TRUE and prior_closed == FALSE:
            self.state.attempt_id += 1
            self.state.pending_event = "none"
            self.state.hold_evidence_in_current_attempt = FALSE if self.history_complete else UNKNOWN
            self.state.release_command_observed = FALSE
            self.state.hold_confirm_run_length = 0
            self.state.loss_confirm_run_length = 0
            self.state.resolved_event_id = None
            self.state.contact_seen_without_hold = FALSE

        if stable == TRUE:
            self.state.hold_confirm_run_length += 1
            if self.state.hold_confirm_run_length >= self.hold_confirm:
                self.state.hold_evidence_in_current_attempt = TRUE
        elif stable == FALSE:
            self.state.hold_confirm_run_length = 0
        else:
            self.state.hold_confirm_run_length = 0

        recovery = FALSE
        # A retry is resolved by the first confirmed stable hold in the same
        # attempt; it must not keep emitting retry forever after recovery.
        if self.state.pending_event == "missed_grasp" and stable == TRUE and self.state.hold_evidence_in_current_attempt == TRUE:
            recovery = TRUE
            self.state.resolved_event_id = self.state.resolved_event_id or f"resolved_{self.state.event_counter:03d}"
            self.state.pending_event = "none"

        if contact == TRUE and self.state.hold_evidence_in_current_attempt != TRUE:
            self.state.contact_seen_without_hold = TRUE

        loss_observed = tri_and(closed, tri_not(contact), tri_or(p.get("contact_recently_lost", UNKNOWN),
                                                                 tri_and(prior_contact, tri_not(contact))))
        failed_observed = tri_and(closed, tri_not(contact))
        if loss_observed == TRUE and self.state.hold_evidence_in_current_attempt == TRUE and self.state.release_command_observed != TRUE:
            self.state.loss_confirm_run_length += 1
            if self.state.loss_confirm_run_length >= self.loss_confirm:
                if self.state.pending_event != "held_object_loss":
                    self._new_event("held_object_loss", time)
                self.state.pending_event = "held_object_loss"
        elif loss_observed == FALSE:
            self.state.loss_confirm_run_length = 0

        if failed_observed == TRUE and self.state.hold_evidence_in_current_attempt == FALSE:
            if self.state.contact_seen_without_hold == TRUE:
                self.state.pending_event = "unknown"
            else:
                if self.state.pending_event not in {"missed_grasp", "held_object_loss"}:
                    self._new_event("missed_grasp", time)
                self.state.pending_event = "missed_grasp"

        if self.state.pending_event == "held_object_loss" and self.state.hold_evidence_in_current_attempt == TRUE and stable == TRUE:
            recovery = TRUE
            self.state.resolved_event_id = self.state.resolved_event_id or f"resolved_{self.state.event_counter:03d}"
            self.state.pending_event = "none"
        elif self.state.pending_event == "held_object_loss" and stable == UNKNOWN:
            recovery = UNKNOWN

        needs = UNKNOWN
        if any(value == UNKNOWN for value in (contact, closed, stable)):
            needs = TRUE
        elif self.state.pending_event == "unknown":
            needs = TRUE
        elif self.state.pending_event in {"missed_grasp", "held_object_loss"}:
            needs = FALSE
        else:
            needs = FALSE

        retry = FALSE
        recover = FALSE
        if self.state.pending_event == "missed_grasp":
            retry = TRUE
        elif self.state.pending_event == "held_object_loss":
            recover = TRUE
        elif self.state.pending_event == "unknown" or needs == TRUE:
            retry = recover = UNKNOWN

        self.state.last_contact_state = contact
        self.state.previous_closed = closed
        return {
            "frame_index": frame_index, "time": time, "attempt_id": self.state.attempt_id,
            "raw_predicates": {"grasp_failed_observed": failed_observed, "slip_observed": loss_observed,
                                "stable_hold_observed": stable, "contact_present": contact,
                                "gripper_command_closed": closed, "gripper_command_open": opened},
            "semantic_events": {
                "missed_grasp_retry_required": retry,
                "held_object_loss_recovery_required": recover,
                "release_expected": self.state.release_command_observed,
                "recovery_in_progress": TRUE if self.state.pending_event in {"missed_grasp", "held_object_loss"} else FALSE,
                "recovery_achieved_observed": recovery,
                "needs_observation": needs,
            },
            "effective_guards": {"retry_grasp": retry, "recover_object": recover},
            "pending_event": self.state.pending_event,
            "hold_evidence_in_current_attempt": self.state.hold_evidence_in_current_attempt,
            "state": asdict(self.state),
            "selected_action": "retry_grasp" if retry == TRUE else "recover_object" if recover == TRUE else "needs_observation" if needs == TRUE else "none",
        }


def run_memory(predictions: Iterable[dict[str, Any]], hold_confirm_observations: int = 1, loss_confirm_observations: int = 1, history_complete: bool = True) -> list[dict[str, Any]]:
    memory = AttemptScopedMemory(hold_confirm_observations, loss_confirm_observations, history_complete)
    return [memory.observe(row.get("predicates", row), row.get("time"), row.get("frame_index")) for row in predictions]


def c1_effective_guard(predicates: dict[str, Any]) -> dict[str, str]:
    failed = tri(predicates.get("grasp_failed_observed"))
    slip = tri(predicates.get("slip_observed"))
    return {"retry_grasp": tri_and(failed, tri_not(slip)), "recover_object": slip}
