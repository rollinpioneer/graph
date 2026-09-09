from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from upgrade_v2.l2r_ambiguity.event_memory import FALSE, TRUE, UNKNOWN, tri

from .contract import ControllerRequest, RequestedEffect, validate_request_provenance


@dataclass
class TaskState:
    attempt_id: int = 0
    attempt_started_observed: bool = False
    hold_confirmed: bool = False
    pending_event: str = "none"
    pending_event_id: str | None = None
    event_counter: int = 0
    last_contact: str = UNKNOWN
    last_closed: str = UNKNOWN
    consumed_end_keys: tuple[str, ...] = ()


class TaskConditionedEventInterface:
    """Causal M1 interface using only arrived observations and dispatch context."""

    def __init__(self, *, history_complete: bool = True) -> None:
        self.history_complete = bool(history_complete)
        self.state = TaskState()

    def _reset_attempt(self, attempt_id: int, started_observed: bool) -> None:
        self.state.attempt_id = attempt_id
        self.state.attempt_started_observed = started_observed
        self.state.hold_confirmed = False
        self.state.pending_event = "none"
        self.state.pending_event_id = None
        self.state.last_contact = UNKNOWN
        self.state.last_closed = UNKNOWN

    def _start_event(self, kind: str) -> None:
        if self.state.pending_event != kind:
            self.state.event_counter += 1
            self.state.pending_event_id = f"m1_event_{self.state.event_counter:03d}_{kind}"
        self.state.pending_event = kind

    def observe(
        self,
        observation: dict[str, Any],
        evidence: dict[str, Any],
        request: ControllerRequest | None,
    ) -> dict[str, Any]:
        now = float(observation.get("time", 0.0))
        capture_order = int(observation.get("capture_order", observation.get("frame_index", 0) or 0))
        attempt_id = int(observation.get("attempt_id") or 0)
        active = tri(observation.get("attempt_active"))
        end = tri(observation.get("attempt_end"))
        end_reason = observation.get("attempt_end_reason")
        phase = str(observation.get("attempt_phase", "unknown"))
        predicates = observation.get("predicates", observation)
        contact = tri(predicates.get("contact_present"))
        closed = tri(predicates.get("gripper_command_closed"))
        opened = tri(predicates.get("gripper_command_open"))
        hold_evidence = tri(evidence.get("hold_evidence"))
        sensor_valid = UNKNOWN not in {contact, closed, opened, hold_evidence}

        if attempt_id > 0 and attempt_id != self.state.attempt_id:
            started_observed = active == TRUE and phase not in {"ended", "inactive", "unknown"}
            self._reset_attempt(attempt_id, started_observed)
        elif active == TRUE and phase not in {"ended", "inactive", "unknown"}:
            self.state.attempt_started_observed = True

        context_valid, context_reason = (False, "request_missing")
        if request is not None and attempt_id > 0:
            context_valid, context_reason = validate_request_provenance(
                request, now=now, capture_order=capture_order, attempt_id=attempt_id
            )
        effect = request.requested_effect if context_valid and request is not None else RequestedEffect.UNKNOWN

        if hold_evidence == TRUE and contact == TRUE and closed == TRUE:
            self.state.hold_confirmed = True
            if self.state.pending_event in {"missed_grasp", "held_object_loss"}:
                self.state.pending_event = "none"
                self.state.pending_event_id = None

        release_observed = opened == TRUE or end_reason == "release" or effect == RequestedEffect.RELEASE_OBJECT
        contact_loss = self.state.last_contact == TRUE and contact == FALSE
        if self.state.hold_confirmed and contact_loss and not release_observed:
            self._start_event("held_object_loss")

        end_sequence = observation.get("attempt_end_sequence", 0)
        end_key = f"{request.request_id if request else 'missing'}:{attempt_id}:{end_sequence}"
        end_fresh = end == TRUE and end_key not in self.state.consumed_end_keys
        if end_fresh:
            self.state.consumed_end_keys = (*self.state.consumed_end_keys, end_key)

        retry_guard = FALSE
        recover_guard = TRUE if self.state.pending_event == "held_object_loss" else FALSE
        needs = FALSE
        reason = "no_emergency_condition"

        if recover_guard == TRUE:
            reason = "confirmed_hold_then_uncommanded_contact_loss"
        elif end_fresh and end_reason == "segment_complete" and closed == TRUE and contact == FALSE and not self.state.hold_confirmed:
            if not self.history_complete or not self.state.attempt_started_observed:
                needs, reason = TRUE, "history_incomplete_for_attempt_failure"
            elif not sensor_valid:
                needs, reason = TRUE, "sensor_state_invalid"
            elif not context_valid:
                needs, reason = TRUE, context_reason
            elif effect == RequestedEffect.HOLD_OBJECT:
                retry_guard, reason = TRUE, "hold_request_normally_ended_without_hold"
                self._start_event("missed_grasp")
            elif effect in {RequestedEffect.TOUCH_OBJECT, RequestedEffect.OBSERVE}:
                reason = "non_hold_request_normally_completed"
            else:
                needs, reason = TRUE, "requested_effect_does_not_support_retry"
        elif end_fresh and end_reason in {"cancelled", "release"}:
            reason = f"{end_reason}_end_does_not_imply_failure"
        elif active == TRUE and contact == FALSE and closed == TRUE and not self.state.hold_confirmed:
            reason = "acquisition_active_wait_for_request_end"
        elif self.state.pending_event == "missed_grasp":
            retry_guard, reason = TRUE, "unresolved_hold_request_failure"
        elif not sensor_valid:
            needs, reason = TRUE, "sensor_state_invalid"

        if recover_guard == TRUE:
            retry_guard = FALSE
            needs = FALSE
        selected = "recover_object" if recover_guard == TRUE else "retry_grasp" if retry_guard == TRUE else "needs_observation" if needs == TRUE else "none"
        row = {
            "frame_index": observation.get("frame_index"),
            "time": observation.get("time"),
            "capture_order": capture_order,
            "attempt_id": attempt_id,
            "attempt_phase": phase,
            "attempt_active": active,
            "attempt_end": end,
            "attempt_end_reason": end_reason,
            "request_id": request.request_id if request else None,
            "requested_effect": effect.value,
            "context_valid": context_valid,
            "context_provenance": context_reason,
            "history_complete": self.history_complete,
            "sensor_valid": sensor_valid,
            "hold_evidence": hold_evidence,
            "hold_evidence_in_current_attempt": TRUE if self.state.hold_confirmed else FALSE,
            "end_edge_fresh": end_fresh,
            "effective_guards": {"retry_grasp": retry_guard, "recover_object": recover_guard},
            "needs_observation": needs,
            "selected_action": selected,
            "reason_code": reason,
            "pending_event": self.state.pending_event,
            "pending_event_id": self.state.pending_event_id,
            "state": asdict(self.state),
        }
        self.state.last_contact = contact
        self.state.last_closed = closed
        return row


def run_m1(
    observations: list[dict[str, Any]],
    evidence_rows: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    *,
    history_complete: bool = True,
) -> list[dict[str, Any]]:
    parsed = [ControllerRequest.from_mapping(row) for row in requests]
    interface = TaskConditionedEventInterface(history_complete=history_complete)
    output = []
    for observation, evidence in zip(observations, evidence_rows):
        now = float(observation.get("time", 0.0))
        order = int(observation.get("capture_order", observation.get("frame_index", 0) or 0))
        arrived = [request for request in parsed if request.received_time < now - 1e-9 or (abs(request.received_time - now) <= 1e-9 and request.issued_capture_order <= order)]
        request = arrived[-1] if arrived else None
        output.append(interface.observe(observation, evidence, request))
    return output
