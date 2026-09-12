#!/usr/bin/env python3
"""Strict zero-physics validator for the frozen R14B V2 A2 template."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HEX64 = re.compile(r"^[0-9a-f]{64}$")
STATIC_FIELDS = (
    "schema", "status", "protocol_id", "static_contract_version", "stage", "requested_instances",
    "authorized_instances", "automatic_retry", "on_any_main_gate_mismatch", "agent_self_authorization_prohibited",
    "runner_commit", "generation_runner_file_hashes", "protocol_sha256", "source_lock_sha256",
    "environment_contract_sha256", "generated_model_xml_sha256", "root_family_id", "case_id", "family_seed",
    "rollout_seed", "program_sha256", "physical_spec_sha256", "instrumented_replay_authorized_instances",
    "r16_calibration_authorized_instances", "r16_development_authorized_instances",
)
HUMAN_NULL_FIELDS = ("authorization_id", "reviewer_id", "approved_at_utc", "expires_at_utc", "single_use_nonce", "output_root")


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("template must be a JSON object")
    return value


def validate_template(template_path: Path, protocol_path: Path, source_lock_path: Path) -> list[str]:
    template, protocol, lock = load(template_path), load(protocol_path), load(source_lock_path)
    errors: list[str] = []
    expected = {
        "schema": "l2rar2_r14b_execution_authorization_v3", "status": "NOT_AUTHORIZED",
        "protocol_id": "L2RAR2_R14B_NEW_REPRODUCIBLE_BASELINE_V2", "static_contract_version": "l2rar2_r14b_pre_execution_contract_v3",
        "stage": "R14B_V2_ORDINARY_BASELINE_A", "requested_instances": 1, "authorized_instances": 0,
        "automatic_retry": False, "on_any_main_gate_mismatch": "STOP_AFTER_EXECUTION_1", "agent_self_authorization_prohibited": True,
        "runner_commit": "15d1b12999adad5172c2dacf8e8a5488d79df6bc", "root_family_id": "L2RAR2_REPRO_BASE_00_870000",
        "case_id": "K3_normal_hold_pause_resume", "family_seed": 870000, "rollout_seed": 87100002,
        "instrumented_replay_authorized_instances": 0, "r16_calibration_authorized_instances": 0, "r16_development_authorized_instances": 0,
    }
    for field, wanted in expected.items():
        if field not in template:
            errors.append(f"missing:{field}")
        elif template[field] != wanted:
            errors.append(f"mismatch:{field}")
    for field in HUMAN_NULL_FIELDS:
        if field not in template:
            errors.append(f"missing:{field}")
        elif template[field] is not None:
            errors.append(f"must_be_null:{field}")
    if protocol.get("schema") != "l2rar2_r14b_protocol_lock_v3": errors.append("protocol_schema")
    if lock.get("schema") != "l2rar2_r14b_pre_execution_source_lock_v3": errors.append("source_lock_schema")
    for field in ("protocol_sha256", "source_lock_sha256", "environment_contract_sha256", "generated_model_xml_sha256", "program_sha256", "physical_spec_sha256"):
        if field not in template:
            errors.append(f"missing:{field}")
            continue
        value = template.get(field)
        if not isinstance(value, str) or not HEX64.fullmatch(value): errors.append(f"invalid_hash:{field}")
    for field in ("protocol_sha256", "source_lock_sha256", "environment_contract_sha256", "generated_model_xml_sha256"):
        if template.get(field) != lock.get(field): errors.append(f"lock_mismatch:{field}")
    for field in ("protocol_id", "static_contract_version", "program_sha256", "physical_spec_sha256"):
        if template.get(field) != protocol.get(field): errors.append(f"protocol_mismatch:{field}")
    supplied = template.get("generation_runner_file_hashes")
    actual = lock.get("generation_runner_file_hashes")
    if not isinstance(supplied, dict) or not supplied: errors.append("runner_hashes_empty")
    elif supplied != actual: errors.append("runner_hashes_mismatch")
    if isinstance(supplied, dict) and any(not isinstance(v, str) or not HEX64.fullmatch(v) for v in supplied.values()): errors.append("runner_hash_invalid")
    if template.get("schema") == "l2rar2_r14b_execution_authorization_v1": errors.append("old_v1_template")
    return sorted(set(errors))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    args = parser.parse_args()
    errors = validate_template(args.template, args.protocol, args.source_lock)
    result = {"schema": "l2rar2_r14b_A2_template_validation_v1", "status": "PASS" if not errors else "FAIL", "errors": errors, "physical_executions": 0, "mujoco_imported": False}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
