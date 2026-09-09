from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from pathlib import Path

from .io import read_json, sha256, write_csv, write_json


class RequestedEffect(str, Enum):
    HOLD_OBJECT = "HOLD_OBJECT"
    TOUCH_OBJECT = "TOUCH_OBJECT"
    OBSERVE = "OBSERVE"
    RELEASE_OBJECT = "RELEASE_OBJECT"
    UNKNOWN = "UNKNOWN"


END_REASONS = {"segment_complete", "cancelled", "release"}


@dataclass(frozen=True)
class ControllerRequest:
    request_id: str
    attempt_id: int
    target_track_id: str
    requested_effect: RequestedEffect
    issued_time: float
    issued_capture_order: int
    received_time: float
    source: str = "controller_dispatch"

    def __post_init__(self) -> None:
        if not self.request_id or self.attempt_id <= 0:
            raise ValueError("request_id and positive attempt_id are required")
        if self.source != "controller_dispatch":
            raise ValueError("controller request must originate at dispatch")
        if self.issued_time > self.received_time:
            raise ValueError("request cannot be received before issue")

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "ControllerRequest":
        return cls(
            request_id=str(value["request_id"]),
            attempt_id=int(value["attempt_id"]),
            target_track_id=str(value["target_track_id"]),
            requested_effect=RequestedEffect(value["requested_effect"]),
            issued_time=float(value["issued_time"]),
            issued_capture_order=int(value["issued_capture_order"]),
            received_time=float(value["received_time"]),
            source=str(value.get("source", "controller_dispatch")),
        )

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["requested_effect"] = self.requested_effect.value
        value["schema"] = "l2rar2_controller_request_v1"
        return value


def validate_request_provenance(request: ControllerRequest, *, now: float, capture_order: int, attempt_id: int) -> tuple[bool, str]:
    if request.attempt_id != attempt_id:
        return False, "cross_attempt_request"
    if request.received_time > now + 1e-9:
        return False, "request_not_yet_received"
    if abs(request.received_time - now) <= 1e-9 and request.issued_capture_order > capture_order:
        return False, "request_later_in_capture_order"
    return True, "matched_pre_action_controller_dispatch"


def controller_contract() -> dict[str, Any]:
    return {
        "schema": "l2rar2_controller_context_contract_v1",
        "status": "LOCKED_FOR_NEW_ONLINE_COLLECTION",
        "requested_effect_values": [item.value for item in RequestedEffect],
        "end_reason_values": sorted(END_REASONS),
        "request_required_before_first_physical_action": True,
        "request_source": "controller_dispatch",
        "outcome_fields_forbidden": [
            "scenario", "case_id", "root_family_id", "oracle", "weld_state",
            "future_outcome", "retry_grasp", "recover_object", "missed_grasp",
        ],
        "matching": "request.attempt_id == lifecycle.attempt_id and request already received",
        "late_or_cross_attempt_behavior": "block unsupported retry; preserve evidence-backed held loss recovery",
        "lifecycle_binding": "planned request boundary emitted by controller before execution; independent of observed result",
        "historical_cache_context_status": "MISSING_NOT_RECONSTRUCTED",
    }


def reference_contract() -> dict[str, Any]:
    return {
        "schema": "l2rar2_reference_contract_v1",
        "status": "LOCKED_FOR_NEW_ONLINE_COLLECTION",
        "version": "l2rar2_task_physical_reference_v1",
        "decision_window_seconds": 0.5,
        "hold_proxy": {
            "source": "frozen R1 proxy",
            "minimum_object_displacement_m": 0.01,
            "maximum_relative_position_drift_m": 0.02,
        },
        "rules": {
            "hold_normal_end_without_hold": "retry_grasp",
            "touch_normal_end_without_hold": "none",
            "held_loss_not_release": "recover_object",
            "cancel_or_release_end_only": "none",
            "missing_required_history_or_context": "needs_observation",
            "active_acquisition_after_transient_contact": "none",
        },
        "reference_only_inputs": ["case_id", "events.jsonl", "oracle_timeline.csv", "weld_state"],
        "online_forbidden": ["case_id", "scenario", "oracle", "future_outcome", "expected_action"],
    }


def freeze_contract(inputs_path: Path, trace_root: Path, protocol_path: Path, output_root: Path) -> dict[str, Any]:
    inputs = read_json(inputs_path)
    protocol = read_json(protocol_path)
    trace_summary = read_json(trace_root / "attribution_summary.json")
    if trace_summary.get("status") != "TASK_CONTEXT_HYPOTHESIS_SUPPORTED_FOR_NEW_TEST":
        raise ValueError("T1 attribution does not support opening the proposed contract")
    if inputs.get("historical_cache", {}).get("controller_requested_effect_available") is not False:
        raise ValueError("historical requested-effect status changed; explicit review required")
    if protocol.get("new_selectable_candidate_count_max") != 1:
        raise ValueError("contract requires exactly one selectable new candidate")
    controller = controller_contract()
    reference = reference_contract()
    write_json(output_root / "controller_context_contract.json", controller)
    write_json(output_root / "reference_contract.json", reference)
    write_csv(output_root / "label_contract_diff.csv", [
        {
            "context": "HOLD_OBJECT normal end, closed/no-contact, no prior hold",
            "historical_generic_action": "retry_grasp",
            "r2_action": "retry_grasp",
            "change": "none",
        },
        {
            "context": "TOUCH_OBJECT normal end, closed/no-contact, no prior hold",
            "historical_generic_action": "retry_grasp",
            "r2_action": "none",
            "change": "task-conditioned end interpretation",
        },
        {
            "context": "confirmed hold then non-release contact loss",
            "historical_generic_action": "recover_object",
            "r2_action": "recover_object",
            "change": "none; recovery cannot be masked by purpose",
        },
        {
            "context": "missing/stale/cross-attempt purpose at otherwise retryable end",
            "historical_generic_action": "retry_grasp",
            "r2_action": "needs_observation",
            "change": "unsupported retry blocked",
        },
    ])
    result = {
        "schema": "pathgraph_l2rar2_contract_freeze_v1",
        "status": "CONTRACT_LOCKED_FOR_NEW_COLLECTION",
        "historical_context": "MISSING_NOT_RECONSTRUCTED",
        "controller_contract_sha256": sha256(output_root / "controller_context_contract.json"),
        "reference_contract_sha256": sha256(output_root / "reference_contract.json"),
        "selectable_candidate": "M1_requested_effect_gate",
        "baseline": "D0_generic_attempt_end",
        "new_physical_rollouts": 0,
        "api_calls": 0,
        "training_jobs": 0,
    }
    write_json(output_root / "contract_status.json", result)
    return result
