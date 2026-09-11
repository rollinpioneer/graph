from __future__ import annotations

import fcntl
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .protocol import PROTOCOL_ID, canonical_hash, sha256
from .source_lock import runner_file_hashes

AUTH_SCHEMA = "l2rar2_r14b_execution_authorization_v1"


class AuthorizationDenied(RuntimeError):
    pass


@dataclass(frozen=True)
class Authorization:
    authorization_id: str
    stage: str
    runner_commit: str
    runner_file_hashes: dict[str, str]
    protocol_sha256: str
    output_root: str
    single_use_nonce: str
    expires_at_utc: str
    data: dict[str, Any]


def current_commit(repo: Path) -> str:
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, text=True, stdout=subprocess.PIPE).stdout.strip()


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AuthorizationDenied(f"authorization unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise AuthorizationDenied("authorization must be an object")
    return value


def validate_authorization(path: Path, *, repo: Path, protocol: dict[str, Any], protocol_sha256: str, requested_output_root: Path, stage: str, baseline: dict[str, Any] | None = None) -> Authorization:
    data = _load(path)
    expected_stage = protocol["stages"].get({"R14B_ORDINARY_BASELINE_A": "A", "R14B_ORDINARY_REPEAT_B": "B", "R14B_INSTRUMENTED_C": "C"}.get(stage, ""), {}).get("stage")
    errors: list[str] = []
    exact = {
        "schema": AUTH_SCHEMA,
        "status": "AUTHORIZED",
        "protocol_id": PROTOCOL_ID,
        "stage": expected_stage,
        "authorized_instances": 1,
        "requested_instances": 1,
        "automatic_retry": False,
        "agent_self_authorization_prohibited": True,
        "r16_calibration_authorized_instances": 0,
        "r16_development_authorized_instances": 0,
    }
    for key, wanted in exact.items():
        if data.get(key) != wanted:
            errors.append(key)
    if stage == "R14B_ORDINARY_BASELINE_A" and data.get("instrumented_replay_authorized_instances") != 0:
        errors.append("instrumented_replay_authorized_instances")
    if data.get("runner_commit") != current_commit(repo):
        errors.append("runner_commit")
    actual_hashes = runner_file_hashes(repo)
    if data.get("runner_file_hashes") != actual_hashes:
        errors.append("runner_file_hashes")
    if data.get("protocol_sha256") != protocol_sha256:
        errors.append("protocol_sha256")
    if str(requested_output_root.resolve(strict=False)) != data.get("output_root"):
        errors.append("output_root")
    if not isinstance(data.get("single_use_nonce"), str) or len(data["single_use_nonce"]) < 32:
        errors.append("single_use_nonce")
    try:
        expires = datetime.fromisoformat(str(data.get("expires_at_utc", "")).replace("Z", "+00:00"))
        if expires.tzinfo is None or expires <= datetime.now(timezone.utc):
            errors.append("expires_at_utc")
    except ValueError:
        errors.append("expires_at_utc")
    try:
        approved = datetime.fromisoformat(str(data.get("approved_at_utc", "")).replace("Z", "+00:00"))
        if approved.tzinfo is None or approved > datetime.now(timezone.utc):
            errors.append("approved_at_utc")
    except ValueError:
        errors.append("approved_at_utc")
    if not data.get("authorization_id") or not data.get("reviewer_id"):
        errors.append("human_authorization_identity")
    if baseline is not None:
        for key, wanted in baseline.items():
            if data.get(key) != wanted:
                errors.append(key)
    if errors:
        raise AuthorizationDenied("authorization denied: " + ", ".join(errors))
    return Authorization(
        authorization_id=str(data["authorization_id"]), stage=stage,
        runner_commit=str(data["runner_commit"]), runner_file_hashes=actual_hashes,
        protocol_sha256=protocol_sha256, output_root=str(requested_output_root.resolve(strict=False)),
        single_use_nonce=str(data["single_use_nonce"]), expires_at_utc=str(data["expires_at_utc"]), data=data,
    )


def _append_journal(repo: Path, record: dict[str, Any]) -> None:
    common_dir = Path(subprocess.run(["git", "-C", str(repo), "rev-parse", "--git-common-dir"], check=True, text=True, stdout=subprocess.PIPE).stdout.strip())
    if not common_dir.is_absolute():
        common_dir = (repo / common_dir).resolve()
    control = common_dir / "research_controls" / PROTOCOL_ID
    control.mkdir(parents=True, exist_ok=True)
    journal = control / "consumption_journal.jsonl"
    lock_path = control / "registry.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        previous = "GENESIS"
        if journal.is_file():
            lines = journal.read_text(encoding="utf-8").splitlines()
            if lines:
                previous = json.loads(lines[-1])["record_hash"]
        record = dict(record)
        if journal.is_file():
            for line in journal.read_text(encoding="utf-8").splitlines():
                if line and json.loads(line).get("single_use_nonce") == record.get("single_use_nonce"):
                    raise AuthorizationDenied("single_use_nonce has already been consumed")
        record["previous_record_hash"] = previous
        record["record_hash"] = canonical_hash(record)
        with journal.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def consume_authorization(output_root: Path, authorization: Authorization, authorization_path: Path, *, repo: Path) -> Path:
    output_root = output_root.resolve(strict=False)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.mkdir(output_root, 0o700)
    except FileExistsError as exc:
        raise AuthorizationDenied("output_root already exists; single-use authorization denied") from exc
    record = {
        "schema": "l2rar2_r14b_authorization_consumption_v1",
        "status": "CONSUMED_BEFORE_MODEL_CONSTRUCTION",
        "consumed_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_id": authorization.authorization_id,
        "authorization_path": str(authorization_path.resolve()),
        "stage": authorization.stage,
        "runner_commit": authorization.runner_commit,
        "runner_file_hashes": authorization.runner_file_hashes,
        "protocol_sha256": authorization.protocol_sha256,
        "single_use_nonce": authorization.single_use_nonce,
        "authorized_instances": 1,
        "instance_index": 1,
        "automatic_retry": False,
    }
    target = output_root / "authorization_consumption.json"
    target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with target.open("rb") as handle:
        os.fsync(handle.fileno())
    _append_journal(repo, record)
    return target
