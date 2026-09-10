"""Shared hold/loss state machine for the three S-tier geometry methods.

    S_G_H   height baseline (relative initial height + valid closure/contact)
    S_G_R   relative-motion baseline (3D co-motion + relative relation)
    S_G_HR  height + relative-motion hypothesis with two entry routes

The machine consumes plain S-tier samples and never opens files.
"""

from __future__ import annotations

from typing import Any

from .adapters import DATA_VALID
from .features import (
    co_motion_qualifies,
    height_above_baseline,
    height_baseline,
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
                # Non-numeric annotations (for example a status string) are not
                # thresholds and must never be coerced into one.
                continue
        self.method_id = method_id
        self.rollout_id = rollout_id
        self.root_family_id = root_family_id
        self.parameters = parameters
        self.protocol = {
            **parameters,
            "min_motion_3d_m": float(parameters["min_motion_3d_m"]),
            "direction_cosine_min": float(parameters["direction_cosine_min"]),
            "relative_rho_max": float(parameters["relative_rho_max"]),
            "max_gap_s": float(parameters["max_gap_s"]),
        }
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
        self.r_hold_anchor: list[float] | None = None
        self.candidate_anchor: list[float] | None = None
        self.candidate_max_drift = 0.0
        self.candidate_coverage = 0.0
        self.segment_coverage = 0.0
        self.height_progress = 0.0
        self.loss_progress = 0.0
        self.height_off_progress = 0.0
        self.event_counter = 0
        self.retry_issued = False
        self.segment_started = False
        self.segment_start_gripper_z: float | None = None
        self.attempt_seen: dict[Any, dict[str, Any]] = {}

    def _start_event(self, time_value: float | None) -> str:
        self.event_counter += 1
        label = "%s:%s:loss:%d" % (self.rollout_id or "rollout", self.method_id, self.event_counter)
        self.pending_event_id = label
        self.hold_state = LOSS_PENDING
        return label

    def _break_segment(self) -> None:
        self.candidate_anchor = None
        self.candidate_max_drift = 0.0
        self.candidate_coverage = 0.0
        self.segment_coverage = 0.0
        self.height_progress = 0.0
        self.segment_started = False
        self.segment_start_gripper_z = None

    def _confirm_hold(self, route: str, anchor: list[float] | None, lift: Any) -> None:
        if anchor is None:
            return
        self.hold_state = HELD
        self.ever_held = True
        self.r_hold_anchor = list(anchor)
        self.hold_entry_route = route
        self.lift_confirmed = lift

    def run(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        baseline = height_baseline(samples)
        z0 = baseline["z0"]
        records = []
        previous = None
        for index, sample in enumerate(samples):
            records.append(self.step(sample, previous, index, z0, baseline))
            previous = sample
        return records

    def step(
        self,
        sample: dict[str, Any],
        previous: dict[str, Any] | None,
        index: int,
        z0: float | None,
        baseline: dict[str, Any],
    ) -> dict[str, Any]:
        valid = sample.get("data_quality") == DATA_VALID
        self.current_quality = sample.get("data_quality", "MISSING_OR_INVALID")
        if not valid:
            self.gap_seen = True
        time_value = sample.get("time")
        time_value = float(time_value) if time_value is not None else None
        contact = sample.get("contact_present")
        closed = sample.get("gripper_command") == "closed"
        open_arrived = sample.get("gripper_command") == "open"

        h = height_above_baseline(sample, z0)
        motion = motion_features(previous, sample, max_gap_s=self.protocol["max_gap_s"])
        r = relative_vector(sample)
        e = relative_drift(self.r_hold_anchor, r) if self.r_hold_anchor is not None else None
        dt_value = motion.get("dt_seconds")
        dt = float(dt_value) if dt_value is not None and float(dt_value) > 0 else 0.0

        loss_source = None
        reason_code = "OK"

        if not valid:
            reason_code = str(sample.get("data_quality_reason") or "SAMPLE_NOT_VALID")
            self.loss_progress = 0.0
            self.height_progress = 0.0
        else:
            if not self.segment_started:
                if closed and contact is True:
                    self.segment_started = True
                    gripper_xyz = sample.get("gripper_xyz")
                    self.segment_start_gripper_z = float(gripper_xyz[2]) if gripper_xyz else None
                    self.candidate_anchor = r
                    self.candidate_max_drift = 0.0
                    self.candidate_coverage = 0.0
                    self.segment_coverage = 0.0

            if (
                self.segment_started
                and previous is not None
                and previous.get("data_quality") == DATA_VALID
                and dt > 0
                and closed
                and contact is True
            ):
                # Contact-continuity coverage: real elapsed time on a valid,
                # unbroken closed/contact interval (no bridging over a gap).
                self.segment_coverage += dt

            if open_arrived:
                if self.hold_state in (HELD, LOSS_PENDING):
                    self.hold_state = RELEASED
                self.released = True
                self.pending_event_id = None
                self.lift_confirmed = "unknown" if self.lift_confirmed == "unknown" else self.lift_confirmed
                reason_code = "ACTIVE_RELEASE_ARRIVED"
                self._break_segment()

            if not open_arrived and not (closed and contact is True) and self.segment_started:
                self._break_segment()

            relative_route = ROUTE_RELATIVE if self.method_id == METHOD_G_R else ROUTE_LOW
            if (
                self.method_id in (METHOD_G_R, METHOD_G_HR)
                and self.segment_started
                and self.candidate_anchor is not None
                and previous is not None
                and not open_arrived
            ):
                if co_motion_qualifies(previous, sample, motion, self.protocol):
                    self.candidate_coverage += dt
                drift = relative_drift(self.candidate_anchor, r)
                if drift is not None:
                    self.candidate_max_drift = max(self.candidate_max_drift, float(drift))
                if (
                    self.hold_state == UNESTABLISHED
                    and self.candidate_max_drift <= self.protocol["relative_hold_drift_m"] + 1e-12
                    and self.candidate_coverage >= self.parameters["evidence_duration_s"] - 1e-12
                ):
                    self._confirm_hold(relative_route, self.candidate_anchor, False)
                    reason_code = "RELATIVE_MOTION_HOLD_CONFIRMED"

            if self.method_id == METHOD_G_H and not loss_source:
                height_entry = (
                    h is not None
                    and float(h) >= self.parameters["height_on_m"] - 1e-12
                    and closed
                    and contact is True
                    and not open_arrived
                )
                self.height_progress = self.height_progress + dt if height_entry else 0.0
                if (
                    self.hold_state == UNESTABLISHED
                    and height_entry
                    and self.height_progress >= self.parameters["evidence_duration_s"] - 1e-12
                ):
                    anchor = r
                    self._confirm_hold(ROUTE_HEIGHT, anchor, True)
                    reason_code = "HEIGHT_HOLD_CONFIRMED"

            if self.method_id == METHOD_G_HR and self.hold_state == UNESTABLISHED and not open_arrived:
                gripper_xyz = sample.get("gripper_xyz")
                gripper_rise = None
                if self.segment_start_gripper_z is not None and gripper_xyz:
                    gripper_rise = float(gripper_xyz[2]) - self.segment_start_gripper_z
                lift_route_ok = (
                    h is not None
                    and gripper_rise is not None
                    and float(h) >= self.parameters["height_on_m"] - 1e-12
                    and float(gripper_rise) >= self.parameters["height_on_m"] - 1e-12
                    and closed
                    and contact is True
                    and self.candidate_max_drift <= self.parameters["relative_hold_drift_m"] + 1e-12
                    and self.segment_coverage >= self.parameters["evidence_duration_s"] - 1e-12
                )
                if lift_route_ok:
                    self._confirm_hold(ROUTE_LIFT, self.candidate_anchor or r, True)
                    reason_code = "LIFT_COUPLED_HOLD_CONFIRMED"

            if self.hold_state in (HELD, LOSS_PENDING) and not open_arrived:
                contact_drop = previous is not None and previous.get("contact_present") is True and contact is False
                if e is not None and float(e) >= self.parameters["relative_loss_drift_m"] - 1e-12:
                    self.loss_progress += dt
                else:
                    self.loss_progress = 0.0
                relative_loss = self.loss_progress >= self.parameters["evidence_duration_s"] - 1e-12
                if self.method_id == METHOD_G_H and h is not None:
                    height_off_ok = float(h) <= self.parameters["height_off_m"] + 1e-12
                    self.height_off_progress = self.height_off_progress + dt if height_off_ok else 0.0
                    if (
                        self.height_off_progress >= self.parameters["evidence_duration_s"] - 1e-12
                        and self.pending_event_id is None
                    ):
                        loss_source = "height_return_loss_signal"
                        self._start_event(time_value)
                        reason_code = "LOSS_EVENT_RAISED"
                if loss_source is None and (contact_drop or relative_loss):
                    loss_source = "contact_true_to_false" if contact_drop else "relative_detach_sustained"
                    if self.pending_event_id is None:
                        self._start_event(time_value)
                        reason_code = "LOSS_EVENT_RAISED"
                    else:
                        self.hold_state = LOSS_PENDING

        attempt_id = sample.get("attempt_id")
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

        history_quality = (
            "COMPLETE"
            if (baseline.get("status") == "AVAILABLE" and not self.gap_seen)
            else "INCOMPLETE"
        )
        return {
            "method": self.method_id,
            "input_tier": "S",
            "rollout_id": self.rollout_id,
            "root_family_id": self.root_family_id,
            "index": index,
            "time": time_value,
            "capture_order": sample.get("capture_order"),
            "anchor_source": baseline.get("anchor_source"),
            "h": h,
            "r": r,
            "e": e,
            "cos": motion.get("direction_cosine"),
            "rho": motion.get("relative_vector_error"),
            "evidence_valid": bool(motion.get("motion_interval_valid")),
            "evidence_duration": self.candidate_coverage,
            "hold_entry_route": self.hold_entry_route,
            "ever_held": self.ever_held,
            "current_quality": self.current_quality,
            "hold_state": self.hold_state,
            "history_quality": history_quality,
            "lift_confirmed": self.lift_confirmed,
            "loss_source": loss_source,
            "selected_action": selected_action,
            "reason_code": reason_code,
            "event_id": self.pending_event_id,
            "grasp_supported": bool(self.ever_held),
        }
