from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


STAGE = "R14_ORDINARY_REPLAY"
SCHEMA = "l2rar2_r14_ordinary_replay_protocol_v1"
CASE_ID = "K3_normal_hold_pause_resume"
ROOT_FAMILY_ID = "L2RAR2_REPAIR_00_840000"
ROLLOUT_SEED = 84100002


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_protocol(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if not isinstance(value, dict):
        raise ValueError("protocol must be a JSON object")
    required = {
        "schema": SCHEMA,
        "stage": STAGE,
        "authorized_instances_per_authorization": 1,
        "case_id": CASE_ID,
        "root_family_id": ROOT_FAMILY_ID,
        "rollout_seed": ROLLOUT_SEED,
        "automatic_retry": False,
        "ordinary_failure_action": "STOP_AFTER_EXECUTION_1",
    }
    errors = [key for key, expected in required.items() if value.get(key) != expected]
    if type(value.get("authorized_instances_per_authorization")) is not int:
        errors.append("authorized_instances_per_authorization.type")
    disabled = value.get("other_stage_authorized_instances", {})
    for key in ("R14_INSTRUMENTED_REPLAY", "R16_CALIBRATION", "R16_DEVELOPMENT"):
        if type(disabled.get(key)) is not int or disabled.get(key) != 0:
            errors.append(f"other_stage_authorized_instances.{key}")
    tolerances = value.get("tolerances", {})
    expected_tolerances = {"rtol": 0, "time_atol": 1e-12, "geometry_atol": 1e-12, "mocap_atol": 2e-9}
    if tolerances != expected_tolerances:
        errors.append("tolerances")
    required_keys = (
        "family_index",
        "family_seed",
        "control_variant",
        "program_sha256",
        "physical_spec_sha256",
        "repair_version",
        "collection_version",
        "expected_action_count",
        "expected_control_callback_count",
        "expected_action_end_callback_count",
        "expected_renderer_callback_count",
        "cache_file_hashes",
        "source_file_hashes",
        "required_environment",
    )
    for key in required_keys:
        if key not in value:
            errors.append(key)
    for key in (
        "family_index",
        "family_seed",
        "rollout_seed",
        "expected_action_count",
        "expected_control_callback_count",
        "expected_action_end_callback_count",
        "expected_renderer_callback_count",
    ):
        if key in value and type(value[key]) is not int:
            errors.append(f"{key}.type")
    if isinstance(value.get("cache_file_hashes"), dict) and not value["cache_file_hashes"]:
        errors.append("cache_file_hashes.empty")
    if isinstance(value.get("source_file_hashes"), dict) and not value["source_file_hashes"]:
        errors.append("source_file_hashes.empty")
    if errors:
        raise ValueError("invalid R14 ordinary protocol fields: " + ", ".join(errors))
    return value


def verify_file_hashes(root: Path, expected: dict[str, str], label: str) -> None:
    errors: list[str] = []
    for relative, digest in expected.items():
        path = root / relative
        if not path.is_file():
            errors.append(f"missing:{relative}")
        elif sha256(path) != digest:
            errors.append(f"hash:{relative}")
    if errors:
        raise ValueError(f"{label} verification failed: " + ", ".join(errors))
