"""R15-B shared hold/loss state machine for the three S-tier ablations.

    S_G_H   pure height ablation: height entry, height-return exit only.
    S_G_R   pure relative-motion ablation: co-motion entry, relative exit only.
    S_G_HR  height + relative motion: two entry routes, relative/contact exit.

The machine is causal: the height baseline is taken online, invalid intervals
never accumulate evidence, and earlier records never change when later samples
arrive.  It consumes plain S-tier samples and never opens files.
"""

from __future__ import annotations

from typing import Any

from .adapters import DATA_VALID
from .features import (
    co_motion_qualifies,
    height_above_baseline,
    motion_features,
    relative_drift,
    relative_vector,
)

UNESTABLISHED = "UNESTABLISHED"
HELD = "HELD"
LOSS_PENDING = "LOSS_PENDING"
RELEASED = "RELEASED"

ROUTE_HEIGHT = "HEIGHT_THRESHOLD"
ROUTE_RELATIVE = "RELATIVE_CO_MOTION"
ROUTE_LIFT = "LIFT_COUPLED"
ROUTE_LOW = "LOW_HEIGHT_CO_MOTION"

METHOD_G_H = "S_G_H"
METHOD_G_R = "S_G_R"
METHOD_G_HR = "S_G_HR"
S_METHODS = (METHOD_G_H, METHOD_G_R, METHOD_G_HR)


class GeometryEventStateMachine:
    def __init__(
        self,
        method_id: str,
        protocol: dict[str, Any],
        rollout_id: str | None = None,
        root_family_id: str | None = None,
    ) -> None:
        if method_id not in S_METHODS:
            raise ValueError("unknown method_id: " + str(method_id))
        parameters: dict[str, float] = {}
        for key, value in protocol["parameters"].items():
            try:
                parameters[key] = float(value)
            except (TypeError, ValueError):
                continue
        self.method_id = method_id
        self.rollout_id = rollout_id
        self.root_family_id = root_family_id
        self.parameters = parameters
        self.reset_state()

    def reset_state(self) -> None:
        self.hold_state = UNESTABLISHED
        self.current_quality = "MISSING_OR_INVALID"
        self.lift_confirmed: Any = "unknown"
        self.pending_event_id: str | None = None
        self.hold_entry_route: str | None = None
        self.ever_held = False
        self.released = False
        self.gap_seen = False
        self.z0: float | None = None
        self.z0_time: float | None = None
        self.z0_capture_order: int | None = None
        self.r_hold_anchor: list[float] | None = None
        self.hold_anchor_time: float | None = None
        self.hold_anchor_capture_order: int | None = None
        self.candidate_anchor: list[float] | None = None
        self.candidate_max_drift = 0.0
        self.motion_coverage = 0.0
        self.height_coverage = 0.0
        self.loss_coverage = 0.0
        self.height_off_coverage = 0.0
        self.event_counter = 0
        self.retry_issued = False
        self.segment_started = False
        self.segment_start_gripper_z: float | None = None
        self.current_attempt: Any = None
        self.attempt_seen: dict[Any, dict[str, Any]] = {}

    def _start_event(self) -> str:
        self.event_counter += 1
        label = "%s:%s:loss:%d" % (self.rollout_id or "rollout", self.method_id, self.event_counter)
        self.pending_event_id = label
        self.hold_state = LOSS_PENDING
        return label

    def _break_candidate_segment(self) -> None:
        self.candidate_anchor = None
        self.candidate_max_drift = 0.0
        self.motion_coverage = 0.0
        self.height_coverage = 0.0
        self.segment_started = False
        self.segment_start_gripper_z = None

    def _reset_attempt(self, attempt_id: Any) -> None:
        self.current_attempt = attempt_id
        self.hold_state = UNESTABLISHED
        self.pending_event_id = None
        self.hold_entry_route = None
        self.lift_confirmed = "unknown"
        self.retry_issued = False
        self.loss_coverage = 0.0
        self.height_off_coverage = 0.0
        self.released = False
        self._break_candidate_segment()

    def _confirm_hold(
        self,
        route: str,
        anchor: list[float] | None,
        lift: Any,
        time_value: float | None,
        capture_order: Any,
    ) -> bool:
        if anchor is None:
            return False
        self.hold_state = HELD
        self.ever_held = True
        self.r_hold_anchor = list(anchor)
        self.hold_anchor_time = time_value
        self.hold_anchor_capture_order = capture_order
        self.hold_entry_route = route
        self.lift_confirmed = lift
        return True

    def run(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        records = []
        previous = None
        for index, sample in enumerate(samples):
            records.append(self.step(sample, previous, index))
            previous = sample
        return records

    def step(self, sample: dict[str, Any], previous: dict[str, Any] | None, index: int) -> dict[str, Any]:
        valid = sample.get("data_quality") == DATA_VALID
        self.current_quality = sample.get("data_quality", "MISSING_OR_INVALID")
        time_value = sample.get("time")
        time_value = float(time_value) if time_value is not None else None
        capture_order = sample.get("capture_order")
        contact = sample.get("contact_present")
        closed = sample.get("gripper_command") == "closed"
        open_arrived = sample.get("gripper_command") == "open"

        attempt_id = sample.get("attempt_id")
        if attempt_id is not None and attempt_id != self.current_attempt:
            if self.current_attempt is not None:
                self._reset_attempt(attempt_id)
            else:
                self.current_attempt = attempt_id

        # Causal height baseline: only the current or an already-seen sample.
        if self.z0 is None and valid and sample.get("gripper_command") == "open" and contact is False:
            object_xyz = sample.get("object_xyz")
            if object_xyz is not None:
                self.z0 = float(object_xyz[2])
                self.z0_time = time_value
                self.z0_capture_order = capture_order

        h = height_above_baseline(sample, self.z0)
        motion = motion_features(previous, sample, max_gap_s=self.parameters["max_gap_s"])
        motion_valid = (
            bool(motion.get("motion_interval_valid"))
            and previous is not None
            and previous.get("data_quality") == DATA_VALID
        )
        dt_value = motion.get("dt_seconds")
        dt = float(dt_value) if (motion_valid and dt_value is not None and float(dt_value) > 0) else 0.0
        r = relative_vector(sample)
        e = relative_drift(self.r_hold_anchor, r) if self.r_hold_anchor is not None else None

        # Gripper rise is measured from the start of the current contact segment.
        gripper_rise = None
        if self.method_id == METHOD_G_HR and valid:
            gripper_xyz = sample.get("gripper_xyz")
            if self.segment_start_gripper_z is not None and gripper_xyz:
                gripper_rise = float(gripper_xyz[2]) - self.segment_start_gripper_z

        loss_source = None
        reason_code = "OK"
        hold_transition = False
        height_return_signal = False
        release_observed = False

        if not valid:
            self.gap_seen = True
            reason_code = str(sample.get("data_quality_reason") or "SAMPLE_NOT_VALID")
            self.loss_coverage = 0.0
            self._break_candidate_segment()
        else:
            if not self.segment_started:
                if closed and contact is True:
                    self.segment_started = True
                    gripper_xyz = sample.get("gripper_xyz")
                    self.segment_start_gripper_z = float(gripper_xyz[2]) if gripper_xyz else None
                    self.candidate_anchor = r
                    self.candidate_max_drift = 0.0
                    self.motion_coverage = 0.0
                    self.height_coverage = 0.0
            elif not (closed and contact is True):
                self._break_candidate_segment()

            # A gap, missing sample, oracle mismatch, duplicate capture_order or
            # non-positive dt contributes no evidence and clears the unconfirmed
            # candidate segment (an established history is kept).
            if not motion_valid and previous is not None:
                self._break_candidate_segment()

            if open_arrived:
                release_observed = True
                if self.hold_state in (HELD, LOSS_PENDING):
                    self.hold_state = RELEASED
                self.released = True
                self.pending_event_id = None
                self.loss_coverage = 0.0
                self.height_off_coverage = 0.0
                reason_code = "ACTIVE_RELEASE_ARRIVED"
                self._break_candidate_segment()

            if self.method_id in (METHOD_G_H, METHOD_G_HR) and not open_arrived:
                height_entry = (
                    h is not None
                    and float(h) >= self.parameters["height_on_m"] - 1e-12
                    and closed
                    and contact is True
                )
                self.height_coverage = self.height_coverage + dt if (motion_valid and height_entry) else 0.0

            # S_G_HR prefers the height-coupled route whenever its conditions
            # hold; the low-height route is the fallback for holds that never
            # reach the height threshold.
            lift_route_ready = (
                self.method_id == METHOD_G_HR
                and h is not None
                and gripper_rise is not None
                and float(h) >= self.parameters["height_on_m"] - 1e-12
                and float(gripper_rise) >= self.parameters["height_on_m"] - 1e-12
                and closed
                and contact is True
                and self.height_coverage >= self.parameters["evidence_duration_s"] - 1e-12
            )

            if not open_arrived and self.method_id in (METHOD_G_R, METHOD_G_HR):
                if self.segment_started and self.candidate_anchor is not None and previous is not None:
                    if co_motion_qualifies(previous, sample, motion, self.parameters):
                        self.motion_coverage += dt
                    drift = relative_drift(self.candidate_anchor, r)
                    if drift is not None:
                        self.candidate_max_drift = max(self.candidate_max_drift, float(drift))
                    if (
                        self.hold_state == UNESTABLISHED
                        and self.candidate_max_drift <= self.parameters["relative_hold_drift_m"] + 1e-12
                        and self.motion_coverage >= self.parameters["evidence_duration_s"] - 1e-12
                        and not lift_route_ready
                    ):
                        route = ROUTE_RELATIVE if self.method_id == METHOD_G_R else ROUTE_LOW
                        if self._confirm_hold(route, self.candidate_anchor, False, time_value, capture_order):
                            hold_transition = True
                            reason_code = "RELATIVE_MOTION_HOLD_CONFIRMED"

            if self.method_id == METHOD_G_H and self.hold_state == UNESTABLISHED and not open_arrived:
                if (
                    h is not None
                    and float(h) >= self.parameters["height_on_m"] - 1e-12
                    and closed
                    and contact is True
                    and self.height_coverage >= self.parameters["evidence_duration_s"] - 1e-12
                ):
                    if self._confirm_hold(ROUTE_HEIGHT, r, True, time_value, capture_order):
                        hold_transition = True
                        reason_code = "HEIGHT_HOLD_CONFIRMED"

            if self.method_id == METHOD_G_HR and self.hold_state == UNESTABLISHED and not open_arrived:
                lift_ok = (
                    lift_route_ready
                    and self.candidate_anchor is not None
                    and self.candidate_max_drift <= self.parameters["relative_hold_drift_m"] + 1e-12
                )
                if lift_ok and self._confirm_hold(ROUTE_LIFT, self.candidate_anchor, True, time_value, capture_order):
                    hold_transition = True
                    reason_code = "LIFT_COUPLED_HOLD_CONFIRMED"

            if self.hold_state in (HELD, LOSS_PENDING) and not open_arrived:
                contact_drop = previous is not None and previous.get("contact_present") is True and contact is False
                relative_loss = (
                    e is not None
                    and float(e) >= self.parameters["relative_loss_drift_m"] - 1e-12
                    and motion_valid
                )
                self.loss_coverage = self.loss_coverage + dt if relative_loss else 0.0
                sustained_relative = self.loss_coverage >= self.parameters["evidence_duration_s"] - 1e-12
                if self.method_id in (METHOD_G_H, METHOD_G_HR) and h is not None:
                    height_off = float(h) <= self.parameters["height_off_m"] + 1e-12
                    self.height_off_coverage = (
                        self.height_off_coverage + dt if (height_off and motion_valid) else 0.0
                    )
                    height_return_signal = self.height_off_coverage >= self.parameters["evidence_duration_s"] - 1e-12
                trigger = None
                if self.method_id == METHOD_G_H:
                    trigger = "height_return_loss_signal" if height_return_signal else None
                elif contact_drop or sustained_relative:
                    trigger = "contact_true_to_false" if contact_drop else "relative_detach_sustained"
                if trigger:
                    loss_source = trigger
                    if self.pending_event_id is None:
                        self._start_event()
                        reason_code = "LOSS_EVENT_RAISED"
                    else:
                        self.hold_state = LOSS_PENDING

        if attempt_id is not None:
            record = self.attempt_seen.setdefault(
                attempt_id,
                {"started": False, "ended": False, "request": None, "end_reason": None},
            )
            if sample.get("attempt_active") is True:
                record["started"] = True
            if sample.get("requested_effect"):
                record["request"] = sample.get("requested_effect")
            if sample.get("attempt_end") is True and not record["ended"]:
                record["ended"] = True
                record["end_reason"] = sample.get("attempt_end_reason")
                if (
                    not self.ever_held
                    and not self.retry_issued
                    and record["started"]
                    and record["request"] == "HOLD_OBJECT"
                    and record["end_reason"] == "segment_complete"
                    and sample.get("gripper_command") == "closed"
                    and sample.get("contact_present") is False
                ):
                    self.retry_issued = True
                    reason_code = "RETRY_ISSUED_AFTER_UNHELD_HOLD_ATTEMPT"

        if self.retry_issued and not self.ever_held:
            selected_action = "retry_grasp"
        elif self.hold_state == LOSS_PENDING and self.pending_event_id is not None:
            selected_action = "recover_object"
        else:
            selected_action = "none"

        return {
            "method": self.method_id,
            "input_tier": "S",
            "rollout_id": self.rollout_id,
            "root_family_id": self.root_family_id,
            "index": index,
            "time": time_value,
            "capture_order": capture_order,
            "anchor_source": "first_valid_open_no_contact" if self.z0 is not None else None,
            "h": h,
            "r": r,
            "e": e,
            "cos": motion.get("direction_cosine"),
            "rho": motion.get("relative_vector_error"),
            "evidence_valid": motion_valid,
            "evidence_duration": self.motion_coverage,
            "hold_entry_route": self.hold_entry_route,
            "hold_transition_event": hold_transition,
            "ever_held": self.ever_held,
            "current_quality": self.current_quality,
            "history_quality": "COMPLETE" if (self.z0 is not None and not self.gap_seen) else "INCOMPLETE",
            "hold_state": self.hold_state,
            "lift_confirmed": self.lift_confirmed,
            "loss_source": loss_source,
            "height_return_signal": height_return_signal,
            "selected_action": selected_action,
            "release_observed": release_observed,
            "context_valid": bool(sample.get("context_valid")),
            "requested_effect": sample.get("requested_effect"),
            "reason_code": reason_code,
            "event_id": self.pending_event_id,
            "grasp_supported": bool(self.ever_held),
        }
