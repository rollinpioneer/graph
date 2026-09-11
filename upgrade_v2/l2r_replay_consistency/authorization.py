from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .protocol import CASE_ID, ROOT_FAMILY_ID, ROLLOUT_SEED, STAGE, sha256


AUTH_SCHEMA = "l2rar2_r14_execution_authorization_v1"
RUNNER_FILES = (
    "upgrade_v2/l2r_replay_consistency/__init__.py",
    "upgrade_v2/l2r_replay_consistency/authorization.py",
    "upgrade_v2/l2r_replay_consistency/cli.py",
    "upgrade_v2/l2r_replay_consistency/comparison.py",
    "upgrade_v2/l2r_replay_consistency/ordinary.py",
    "upgrade_v2/l2r_replay_consistency/protocol.py",
    "upgrade_v2/l2r_task_context/collector.py",
    "upgrade_v2/l2r_task_context/io.py",
    "upgrade_v2/l2r_task_context/repair_collection.py",
    "upgrade_v2/l2r_hold_evidence/probe_adapter.py",
    "upgrade_v2/visual_refine_l2/dynamic_simulator.py",
    "upgrade_v2/visual_refine_l2/repaired_simulator.py",
    "upgrade_v2/visual_refine_l2/renderer.py",
    "upgrade_v2/visual_refine_l2/vision.py",
)


class AuthorizationDenied(RuntimeError):
    pass


@dataclass(frozen=True)
class Authorization:
    authorization_id: str
    runner_commit: str
    runner_file_hashes: dict[str, str]
    protocol_sha256: str
    output_root: str
    single_use_nonce: str
    expires_at_utc: str


def current_commit(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def runner_file_hashes(repo: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in RUNNER_FILES:
        path = repo / relative
        if not path.is_file():
            raise AuthorizationDenied(f"runner file missing: {relative}")
        hashes[relative] = sha256(path)
    return hashes


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AuthorizationDenied(f"authorization unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise AuthorizationDenied("authorization must be a JSON object")
    return value


def validate_authorization(
    path: Path,
    *,
    repo: Path,
    protocol_sha256: str,
    requested_output_root: Path,
) -> Authorization:
    data = _load(path)
    errors: list[str] = []
    exact = {
        "schema": AUTH_SCHEMA,
        "status": "AUTHORIZED",
        "stage": STAGE,
        "authorized_instances": 1,
        "case": CASE_ID,
        "root_family_id": ROOT_FAMILY_ID,
        "rollout_seed": ROLLOUT_SEED,
        "automatic_retry": False,
        "on_any_main_gate_mismatch": "STOP_AFTER_EXECUTION_1",
        "instrumented_replay_authorized_instances": 0,
        "r16_calibration_authorized_instances": 0,
        "r16_development_authorized_instances": 0,
        "agent_self_authorization_prohibited": True,
    }
    for key, expected in exact.items():
        if data.get(key) != expected:
            errors.append(key)
    for key in (
        "authorized_instances",
        "instrumented_replay_authorized_instances",
        "r16_calibration_authorized_instances",
        "r16_development_authorized_instances",
    ):
        if type(data.get(key)) is not int:
            errors.append(f"{key}.type")
    commit = current_commit(repo)
    if data.get("runner_commit") != commit:
        errors.append("runner_commit")
    if data.get("protocol_sha256") != protocol_sha256:
        errors.append("protocol_sha256")
    actual_hashes = runner_file_hashes(repo)
    if data.get("runner_file_hashes") != actual_hashes:
        errors.append("runner_file_hashes")
    requested = str(requested_output_root.resolve(strict=False))
    if data.get("output_root") != requested:
        errors.append("output_root")
    nonce = data.get("single_use_nonce")
    if not isinstance(nonce, str) or len(nonce) < 32:
        errors.append("single_use_nonce")
    try:
        expires = datetime.fromisoformat(str(data.get("expires_at_utc", "")).replace("Z", "+00:00"))
        if expires.tzinfo is None or expires <= datetime.now(timezone.utc):
            errors.append("expires_at_utc")
    except ValueError:
        errors.append("expires_at_utc")
    if not data.get("authorization_id") or not data.get("approved_at_utc") or not data.get("reviewer_id"):
        errors.append("human_authorization_identity")
    else:
        try:
            approved = datetime.fromisoformat(str(data["approved_at_utc"]).replace("Z", "+00:00"))
            if approved.tzinfo is None or approved > datetime.now(timezone.utc):
                errors.append("approved_at_utc")
        except ValueError:
            errors.append("approved_at_utc")
    if errors:
        raise AuthorizationDenied("authorization denied: " + ", ".join(errors))
    return Authorization(
        authorization_id=str(data["authorization_id"]),
        runner_commit=commit,
        runner_file_hashes=actual_hashes,
        protocol_sha256=protocol_sha256,
        output_root=requested,
        single_use_nonce=nonce,
        expires_at_utc=str(data["expires_at_utc"]),
    )


def consume_authorization(output_root: Path, authorization: Authorization, authorization_path: Path) -> Path:
    output_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.mkdir(output_root)
    except FileExistsError as exc:
        raise AuthorizationDenied("single-use authorization already consumed: output_root exists") from exc
    record = {
        "schema": "l2rar2_r14_authorization_consumption_v1",
        "status": "CONSUMED_BEFORE_MODEL_CONSTRUCTION",
        "consumed_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_id": authorization.authorization_id,
        "authorization_path": str(authorization_path.resolve()),
        "runner_commit": authorization.runner_commit,
        "protocol_sha256": authorization.protocol_sha256,
        "runner_file_hashes": authorization.runner_file_hashes,
        "single_use_nonce": authorization.single_use_nonce,
        "authorized_instances": 1,
        "instance_index": 1,
        "automatic_retry": False,
        "other_stage_authorized_instances": {
            "R14_INSTRUMENTED_REPLAY": 0,
            "R16_CALIBRATION": 0,
            "R16_DEVELOPMENT": 0,
        },
    }
    target = output_root / "authorization_consumption.json"
    temporary = output_root / ".authorization_consumption.tmp"
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return target
