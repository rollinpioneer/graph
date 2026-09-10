"""Contracts and small pure helpers for the L2RAR2 R11 loss audit.

The R11 audit is diagnostic only.  These constants deliberately keep the
R2 reference and candidate set immutable while giving each new audit field a
separate meaning.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


CASES = (
    "K1_hold_request_ends_without_hold",
    "K2_touch_request_completes_without_hold",
    "K3_normal_hold_pause_resume",
    "K4_regular_hold_loss",
    "K5_brief_hold_loss",
    "K6_long_gap_after_loss",
    "K7_commanded_release",
    "K8_acquisition_touch_then_continue",
)
LOSS_CASES = ("K4_regular_hold_loss", "K5_brief_hold_loss", "K6_long_gap_after_loss")
CONTRAST_CASES = (
    "K3_normal_hold_pause_resume",
    "K7_commanded_release",
    "K8_acquisition_touch_then_continue",
)
FIXED_CANDIDATES = ("B_count2", "C3_vector_rho035")
REFERENCE_DRIFT_LIMIT_M = 0.02
REFERENCE_MIN_DISPLACEMENT_M = 0.01
PROXY_OFFSET_Z_M = -0.13
PROXY_MARGIN_M = 0.045
MAX_PHYSICAL_EXECUTIONS = 8
R11_VERSION = "l2rar2_loss_observability_r11_v1"
PHYSICS_PROBE_VERSION = "l2rar2_r11_physics_probe_v1"
PHYSICS_PROBE_CASES = (
    "K3_normal_hold_pause_resume",
    "K4_regular_hold_loss",
    "K5_brief_hold_loss",
    "K6_long_gap_after_loss",
)


@dataclass(frozen=True)
class Rollout:
    rollout_id: str
    root_family_id: str
    case_id: str
    path: Path
    family_seed: int
    rollout_seed: int
    requested_effect: str
    control_variant: str
    repair_version: str
    collection_version: str


def bool_text(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def vector3(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        import json

        value = json.loads(value)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    return tuple(float(item) for item in value)


def euclidean(left: tuple[float, ...] | None, right: tuple[float, ...] | None) -> float | None:
    if left is None or right is None or len(left) != len(right):
        return None
    return sum((a - b) ** 2 for a, b in zip(left, right)) ** 0.5


def relative_vector(object_xyz: Any, gripper_xyz: Any) -> tuple[float, float, float] | None:
    obj = vector3(object_xyz)
    grip = vector3(gripper_xyz)
    if obj is None or grip is None:
        return None
    return tuple(obj[index] - grip[index] for index in range(3))


def control_variant_text(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("variant_id", "unknown"))
    return str(value)


def required_artifacts() -> tuple[str, ...]:
    return (
        "entry_audit.json",
        "cache_audit_summary.json",
        "event_chain_trace.csv",
        "loss_event_audit.csv",
        "regression_audit.csv",
        "regression_metrics.json",
        "sensor_proxy_audit.csv",
        "visual_loss_evidence_audit.csv",
        "physics_probe_manifest.json",
        "physical_execution_accounting.json",
        "probe_cached_equivalence.json",
        "probe_execution_ledger.csv",
        "time_audit_v2.csv",
        "source_contract_audit.json",
        "repair_route_decision.json",
        "next_stage_handoff.json",
        "run_manifest.json",
        "actual_commands.txt",
        "external_artifacts.tsv",
        "secret_scan.json",
        "legacy_zip_verification.json",
        "validation_results.json",
        "final_report.md",
    )
