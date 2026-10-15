from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .inputs import sha256_file, write_json


def _lock_mismatches(lock: dict[str, Any]) -> list[dict[str, Any]]:
    mismatches = []
    for item in lock.get("locked_files", []):
        path = Path(item.get("path", ""))
        expected = item.get("sha256")
        actual = sha256_file(path) if path.is_file() else None
        if not expected or actual != expected:
            mismatches.append({"path": str(path), "expected_sha256": expected, "actual_sha256": actual})
    for key in ("generator_hash", "reference_contract_hash", "interface_hash"):
        item = lock.get(key)
        if not isinstance(item, dict):
            mismatches.append({"dependency": key, "reason": "missing lock entry"})
            continue
        path = Path(item.get("path", ""))
        expected = item.get("sha256")
        actual = sha256_file(path) if path.is_file() else None
        if not expected or actual != expected:
            mismatches.append({"dependency": key, "path": str(path), "expected_sha256": expected, "actual_sha256": actual})
    return mismatches


def confirm(selection_lock: Path, protocol: dict[str, Any], inputs: dict[str, Any], output_root: Path) -> dict[str, Any]:
    lock = json.loads(selection_lock.read_text(encoding="utf-8"))
    output_root.mkdir(parents=True, exist_ok=True)
    if not lock.get("ready_for_confirmation") or not lock.get("selected_candidate_id"):
        result = {"schema": "pathgraph_l2rar1_confirmation_consumption_v1", "status": "NOT_RUN_DEVELOPMENT_FAILED", "reason": "selection_lock.ready_for_confirmation is false or selected_candidate_id is null", "standard_confirmation_status": "NOT_RUN", "challenge_confirmation_status": "NOT_RUN", "new_data_consumed": False, "api_calls": 0, "training_jobs": 0, "api_key_read": False}
    elif mismatches := _lock_mismatches(lock):
        result = {"schema": "pathgraph_l2rar1_confirmation_consumption_v1", "status": "BLOCKED_LOCK_HASH_MISMATCH", "reason": "confirmation dependencies do not match the final selection lock", "lock_mismatches": mismatches, "standard_confirmation_status": "NOT_RUN", "challenge_confirmation_status": "NOT_RUN", "new_data_consumed": False, "api_calls": 0, "training_jobs": 0, "api_key_read": False}
    else:
        result = {"schema": "pathgraph_l2rar1_confirmation_consumption_v1", "status": "NOT_RUN_CONFIRMATION_GENERATOR_REQUIRED", "reason": "Confirmation data generator is intentionally not reused from the historical L2RA cache; no independent R4 family was consumed in this run.", "standard_confirmation_status": "NOT_RUN", "challenge_confirmation_status": "NOT_RUN", "new_data_consumed": False, "api_calls": 0, "training_jobs": 0, "api_key_read": False}
    write_json(output_root / "confirmation_consumption.json", result)
    write_json(output_root / "confirmation_status.json", result)
    return result
