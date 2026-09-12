#!/usr/bin/env python3
"""Build the zero-authorization A2 template from frozen V2 contracts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def build(protocol_path: Path, source_lock_path: Path) -> dict:
    protocol = load(protocol_path)
    lock = load(source_lock_path)
    required = {
        "protocol_id": "L2RAR2_R14B_NEW_REPRODUCIBLE_BASELINE_V2",
        "protocol_schema": "l2rar2_r14b_protocol_lock_v3",
        "source_lock_schema": "l2rar2_r14b_pre_execution_source_lock_v3",
        "static_contract_version": "l2rar2_r14b_pre_execution_contract_v3",
        "runner_commit": "15d1b12999adad5172c2dacf8e8a5488d79df6bc",
    }
    if protocol.get("protocol_id") != required["protocol_id"] or protocol.get("schema") != required["protocol_schema"]:
        raise ValueError("protocol is not the frozen R14B V2 protocol")
    if lock.get("schema") != required["source_lock_schema"] or lock.get("static_contract_version") != required["static_contract_version"]:
        raise ValueError("source lock is not the frozen R14B V2 source lock")
    if lock.get("runner_commit") != required["runner_commit"]:
        raise ValueError("source lock runner commit mismatch")
    hashes = lock.get("generation_runner_file_hashes")
    if not isinstance(hashes, dict) or not hashes or any(not isinstance(v, str) or len(v) != 64 for v in hashes.values()):
        raise ValueError("source lock runner hashes are incomplete")
    return {
        "schema": "l2rar2_r14b_execution_authorization_v3",
        "status": "NOT_AUTHORIZED",
        "protocol_id": protocol["protocol_id"],
        "static_contract_version": protocol["static_contract_version"],
        "stage": protocol["stages"]["A"]["stage"],
        "requested_instances": 1,
        "authorized_instances": 0,
        "automatic_retry": False,
        "on_any_main_gate_mismatch": "STOP_AFTER_EXECUTION_1",
        "agent_self_authorization_prohibited": True,
        "runner_commit": lock["runner_commit"],
        "generation_runner_file_hashes": hashes,
        "protocol_sha256": lock["protocol_sha256"],
        "source_lock_sha256": lock["source_lock_sha256"],
        "environment_contract_sha256": lock["environment_contract_sha256"],
        "generated_model_xml_sha256": lock["generated_model_xml_sha256"],
        "root_family_id": protocol["root_family_id"],
        "case_id": protocol["case_id"],
        "family_seed": protocol["family_seed"],
        "rollout_seed": protocol["rollout_seed"],
        "program_sha256": protocol["program_sha256"],
        "physical_spec_sha256": protocol["physical_spec_sha256"],
        "instrumented_replay_authorized_instances": 0,
        "r16_calibration_authorized_instances": 0,
        "r16_development_authorized_instances": 0,
        "authorization_id": None,
        "reviewer_id": None,
        "approved_at_utc": None,
        "expires_at_utc": None,
        "single_use_nonce": None,
        "output_root": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build(args.protocol, args.source_lock)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": value["status"], "schema": value["schema"], "runner_hash_count": len(value["generation_runner_file_hashes"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
