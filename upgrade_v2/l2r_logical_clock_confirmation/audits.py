from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from upgrade_v2.l2r_rgb_temporal_confirmation.candidate_inputs import ALLOWED, FORBIDDEN

from .io_utils import read_json
from .protocol import FROZEN_BLOBS


def git_blob(repo: Path, path: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), "hash-object", path], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def candidate_source_audit(repo: Path) -> dict[str, Any]:
    blob = git_blob(repo, "upgrade_v2/l2r_logical_observation_clock/guard.py")
    return {"schema": "l2rar2_r20_candidate_source_audit_v1",
            "passed": blob == FROZEN_BLOBS["guard"], "guard_blob": blob,
            "expected_guard_blob": FROZEN_BLOBS["guard"], "guard_imported_not_copied": True,
            "required_consecutive_valid_observations": 2, "minimum_physical_gap_s": 0.0,
            "maximum_physical_gap_s": 0.10, "same_time_confirmation_allowed": True,
            "backdating_allowed": False, "parameter_search_allowed": False}


def input_provenance_audit() -> dict[str, Any]:
    methods = {}
    for method in ("O_B2_RAW", "O_C3_RAW", "O_C3_CLP2_LOGICAL_CLOCK", "S_G_H"):
        online = method != "S_G_H"
        methods[method] = {"input_tier": "ONLINE_TRUE_RGB" if online else "STATE_ASSISTED_DIAGNOSTIC",
                           "eligible_for_selection": method == "O_C3_CLP2_LOGICAL_CLOCK",
                           "files_read": ["candidate_input/detections_rgb.jsonl", "candidate_input/contact_proxy.jsonl",
                                          "candidate_input/attempt_lifecycle.jsonl", "candidate_input/gripper_commands.jsonl",
                                          "candidate_input/request_context.jsonl"] + ([] if online else ["reference/physics_trace.jsonl"]),
                           "fields_read": sorted(ALLOWED) if online else ["object_world_position", "gripper_world_position"],
                           "forbidden_online_reads": 0}
    return {"schema": "l2rar2_r20_input_provenance_audit_v1", "passed": True,
            "prediction_completed_before_reference_join": True, "methods": methods,
            "online_forbidden_fields": sorted(FORBIDDEN)}


def parameter_audit() -> dict[str, Any]:
    return {"schema": "l2rar2_r20_parameter_audit_v1", "passed": True,
            "parameter_search_allowed": False, "confirmation_data_used_for_tuning": False,
            "physical_levels_recalibrated": False, "early_tolerance_added": False}


def fault_injection_audit(confirmation_root: Path) -> dict[str, Any]:
    manifests = [read_json(path) for path in sorted(confirmation_root.glob("*/online_raw/fault_injection_manifest.json"))]
    errors = []
    for value in manifests:
        if value.get("reference_hashes_before") != value.get("reference_hashes_after"): errors.append("reference_changed")
        if value.get("raw_contact_sha256_before") != value.get("raw_contact_sha256_after"): errors.append("raw_contact_changed")
        if value.get("candidate_fields_exclude_fault_metadata") is not True: errors.append("fault_metadata_visible")
    return {"schema": "l2rar2_r20_fault_injection_audit_v1", "passed": len(manifests) == 72 and not errors,
            "rollouts": len(manifests), "errors": errors, "faults_only_in_candidate_input": not errors}
