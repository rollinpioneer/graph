from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from . import PROTOCOL_ID

ROOT_FAMILY_ID = "L2RAR2_REPRO_BASE_00_870000"
CASE_ID = "K3_normal_hold_pause_resume"
SCENARIO = "normal_pick_place"
CONTROL_VARIANT = "v0_short"
FAMILY_INDEX = 0
FAMILY_SEED = 870000
ROLLOUT_SEED = 87100002
PROGRAM = [
    "observe_scene", "approach_object", "close_gripper", "lift",
    "verify", "verify", "transport_to_target", "verify",
]
EXPECTED_COUNTS = {
    "actions": 8,
    "control_callbacks": 29,
    "action_end_callbacks": 8,
    "render_callbacks": 37,
    "physics_steps": 145,
    "ordinary_checkpoint_rows": 54,
    "instrumented_before_after_rows": 290,
}
REQUIRED_ENVIRONMENT = {
    "python": "3.10.19",
    "numpy": "2.2.6",
    "mujoco": "3.4.0",
    "opencv": "4.13.0",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "MUJOCO_GL": "egl",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def physical_spec(repo: Path | None = None) -> dict[str, Any]:
    del repo
    from upgrade_v2.visual_refine_l2.dynamic_simulator import family_spec

    spec = family_spec(ROOT_FAMILY_ID, SCENARIO, FAMILY_SEED, ROLLOUT_SEED, probe_variant="default")
    return {
        "object_radius": spec.object_radius,
        "friction": spec.friction,
        "camera_jitter": spec.camera_jitter,
        "object_x": spec.object_x,
        "object_y": spec.object_y,
        "target_x": spec.target_x,
        "target_y": spec.target_y,
    }


def make_protocol() -> dict[str, Any]:
    physical = physical_spec()
    return {
        "schema": "l2rar2_r14b_protocol_lock_v2",
        "static_contract_version": "l2rar2_r14b_pre_execution_contract_v2",
        "protocol_id": PROTOCOL_ID,
        "status": "DRAFT_NOT_AUTHORIZED",
        "route": "B_NEW_REPRODUCIBLE_BASELINE_REQUIRED",
        "base_commit": "ab024c71d315100f0e375a844856807114bedaab",
        "formal_main_commit": "234cb6dc0e2767fa62cd2dbec4a868b8d0711bb2",
        "repository": "rollinpioneer/graph",
        "root_family_id": ROOT_FAMILY_ID,
        "family_index": FAMILY_INDEX,
        "family_seed": FAMILY_SEED,
        "rollout_seed": ROLLOUT_SEED,
        "case_id": CASE_ID,
        "scenario": SCENARIO,
        "control_variant": CONTROL_VARIANT,
        "program": PROGRAM,
        "case": {
            "case_id": CASE_ID,
            "root_family_id": ROOT_FAMILY_ID,
            "family_index": FAMILY_INDEX,
            "family_seed": FAMILY_SEED,
            "rollout_seed": ROLLOUT_SEED,
            "scenario": SCENARIO,
            "control_variant": CONTROL_VARIANT,
            "program": PROGRAM,
        },
        "program_sha256": canonical_hash(PROGRAM),
        "physical_spec": physical,
        "physical_spec_sha256": canonical_hash(physical),
        "expected_counts": EXPECTED_COUNTS,
        "comparison": {
            "numeric_atol": 0.0,
            "numeric_rtol": 0.0,
            "array_raw_bytes_sha256_exact": True,
            "raw_rgb_sha256_exact": True,
            "jpeg_sha256_exact": True,
            "detection_canonical_sha256_exact": True,
            "tolerance_adjustment_allowed": False,
        },
        "required_environment": REQUIRED_ENVIRONMENT,
        "old_cache_role": "HISTORICAL_NON_GATING_ONLY",
        "scientific_status": "L2RAR2_PARTIAL_KEEP_G1",
        "retained_graph": "G1_predicate_bound",
        "selected_candidate_id": None,
        "confirmation_run": False,
        "l3_entry_allowed": False,
        "physical_authorization_total": 0,
        "stages": {
            "A": {"stage": "R14B_ORDINARY_BASELINE_A", "requested_instances": 1, "authorized_instances": 0},
            "B": {"stage": "R14B_ORDINARY_REPEAT_B", "requested_instances": 1, "authorized_instances": 0, "requires": "A_HUMAN_REVIEW_PASS"},
            "C": {"stage": "R14B_INSTRUMENTED_C", "requested_instances": 1, "authorized_instances": 0, "requires": "A_VS_B_ALL_MAIN_GATES_PASS"},
        },
    }


def load_protocol(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("invalid R14-B protocol")
    if value.get("schema") != "l2rar2_r14b_protocol_lock_v2" or value.get("static_contract_version") != "l2rar2_r14b_pre_execution_contract_v2":
        raise ValueError("protocol is not bound to the v2 pre-execution contract")
    if value.get("comparison", {}).get("numeric_atol") != 0.0 or value.get("comparison", {}).get("numeric_rtol") != 0.0:
        raise ValueError("numeric tolerances must be exact zero")
    if value.get("status") != "DRAFT_NOT_AUTHORIZED":
        raise ValueError("protocol is not the zero-authorization draft")
    return value
